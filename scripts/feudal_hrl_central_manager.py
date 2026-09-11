#Centralized Manager, Distributed Cluster Controllers

from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import wandb

os.chdir(os.path.dirname(os.path.abspath(__file__)))
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from baseline_models import BaseLearningModel
from scripts.controller import FeudalController
from scripts.manager import FeudalManager
from routerl import TrafficEnvironment
from utils import (
    clear_SUMO_files,
    print_agent_counts,
    run_metrics_analysis,
    save_loss_records,
    script_path_for_config,
)

def load_cluster_lookup(cluster_csv_path, key_columns):
    df = pd.read_csv(cluster_csv_path)
    if "cluster" not in df.columns:
        raise ValueError(f"No 'cluster' column in {cluster_csv_path}")
    unique_clusters = sorted(df["cluster"].unique())
    cluster_to_idx = {c: i + 1 for i, c in enumerate(unique_clusters)}
    lookup = {}
    for _, row in df.iterrows():
        key = tuple(row[col] for col in key_columns)
        lookup[key] = cluster_to_idx[row["cluster"]]
    num_clusters = len(unique_clusters) + 1
    return lookup, num_clusters

def build_agent_cluster_map(agents_csv_path, cluster_lookup, key_columns):
    agents_df = pd.read_csv(agents_csv_path)
    cluster_map = {}
    missing = []
    for idx, row in agents_df.iterrows():
        key = tuple(row[col] for col in key_columns)
        if key in cluster_lookup:
            cluster_map[idx] = int(cluster_lookup[key])
        else:
            cluster_map[idx] = 0
            missing.append(idx)
    return cluster_map, missing

def build_mlp_optimizer(module: nn.Module, lr: float) -> optim.Optimizer:
    return optim.Adam(module.parameters(), lr=lr)


@dataclass
class ManagerTransition:
    state: np.ndarray
    cluster_id: int
    subgoal: int
    log_prob: float
    reward: float

@dataclass
class ControllerTransition:
    state: np.ndarray
    subgoal: int
    action: int
    log_prob: float
    reward: float


class CentralManagerCore(BaseLearningModel):
    #ONE GLOBAL Manager - collects experiences from all clusters and distributes subgoals
    def __init__(self, state_size: int, config: Dict, device: torch.device):
        super().__init__()
        self.device = device
        self.use_cluster_embedding = bool(config.get("use_cluster_embedding", False))
        self.num_clusters = int(config.get("num_clusters", 0))
        self.num_subgoals = int(config["num_subgoals"])

        self.batch_size = int(config["batch_size"])
        self.epochs = int(config["manager_epochs"])
        self.clip_eps = float(config["manager_clip_eps"])
        self.entropy_coef = float(config["manager_entropy_coef"])
        self.normalize_advantage = bool(config["normalize_advantage"])
        self.reward_weight = float(config["manager_reward_weight"])

        self.deterministic = False
        self.memory: List[ManagerTransition] = []
        self.loss: List[Dict[str, float]] = []

        self.manager = FeudalManager(
            obs_dim=state_size,
            num_subgoals=self.num_subgoals,
            hidden_dims=config["manager_hidden_dims"],
            use_cluster_embedding=self.use_cluster_embedding,
            num_clusters=self.num_clusters,
            cluster_embed_dim=int(config.get("cluster_embed_dim", 8)),
        ).to(self.device)

        self.optimizer = build_mlp_optimizer(self.manager, float(config["manager_lr"]))

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if not self.normalize_advantage or x.numel() <= 1:
            return x
        return (x - x.mean()) / (x.std() + 1e-8)

    def learn(self) -> Optional[Dict[str, float]]:
        if len(self.memory) < self.batch_size:
            return None

        batch = self.memory[:]
        self.memory.clear()

        states = torch.FloatTensor(np.stack([b.state for b in batch])).to(self.device)
        subgoals = torch.LongTensor([b.subgoal for b in batch]).to(self.device)
        old_log_probs = torch.FloatTensor([b.log_prob for b in batch]).to(self.device)
        rewards = torch.FloatTensor([self.reward_weight * b.reward for b in batch]).to(self.device)
        advantages = self._normalize(rewards)

        cluster_ids = None
        if self.use_cluster_embedding:
            cluster_ids = torch.LongTensor([b.cluster_id for b in batch]).to(self.device)

        losses = []
        for _ in range(self.epochs):
            if self.use_cluster_embedding:
                dist = self.manager.dist(states, cluster_ids=cluster_ids)
            else:
                dist = self.manager.dist(states)

            new_log_probs = dist.log_prob(subgoals)
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            entropy = dist.entropy().mean()

            loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.manager.parameters(), max_norm=1.0)
            self.optimizer.step()
            losses.append(loss.item())

        loss_dict = {"manager_loss": float(np.mean(losses))}
        self.loss.append(loss_dict)
        return loss_dict
    def act(self, state):
        pass
    def push(self, reward):
        pass

class ClusterControllerCore(BaseLearningModel):
    # Controller for each cluster - maps global subgoals to specific actions of vechicles in the cluster
    def __init__(
        self, state_size: int, action_space_size: int, config: Dict, device: torch.device, cluster_id: int
    ):
        super().__init__()
        self.device = device
        self.cluster_id = cluster_id
        self.action_space_size = action_space_size
        self.num_subgoals = int(config["num_subgoals"])
        self.action_mask_strategy = str(config.get("action_mask_strategy", "uniform_bins"))

        self.batch_size = int(config["batch_size"])
        self.epochs = int(config["controller_epochs"])
        self.clip_eps = float(config["controller_clip_eps"])
        self.entropy_coef = float(config["controller_entropy_coef"])
        self.normalize_advantage = bool(config["normalize_advantage"])
        self.intrinsic_reward_weight = float(config["intrinsic_reward_weight"])

        self.deterministic = False
        self.memory: List[ControllerTransition] = []
        self.loss: List[Dict[str, float]] = []

        self.controller = FeudalController(
            obs_dim=state_size,
            action_dim=self.action_space_size,
            num_subgoals=self.num_subgoals,
            hidden_dims=config["controller_hidden_dims"],
            subgoal_embed_dim=int(config["subgoal_embed_dim"]),
        ).to(self.device)

        self.optimizer = build_mlp_optimizer(self.controller, float(config["controller_lr"]))

    def _build_uniform_subgoal_mask(self, subgoal: int) -> torch.Tensor:
        if self.action_mask_strategy != "uniform_bins":
            return torch.ones((1, self.action_space_size), dtype=torch.float32, device=self.device)

        bins = np.array_split(np.arange(self.action_space_size), self.num_subgoals)
        chosen = bins[subgoal % self.num_subgoals]
        mask = torch.zeros((1, self.action_space_size), dtype=torch.float32, device=self.device)
        mask[0, chosen] = 1.0
        return mask

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if not self.normalize_advantage or x.numel() <= 1:
            return x
        return (x - x.mean()) / (x.std() + 1e-8)

    def learn(self) -> Optional[Dict[str, float]]:
        if len(self.memory) < self.batch_size:
            return None

        batch = self.memory[:]
        self.memory.clear()

        states = torch.FloatTensor(np.stack([b.state for b in batch])).to(self.device)
        subgoals = torch.LongTensor([b.subgoal for b in batch]).to(self.device)
        actions = torch.LongTensor([b.action for b in batch]).to(self.device)
        old_log_probs = torch.FloatTensor([b.log_prob for b in batch]).to(self.device)
        rewards = torch.FloatTensor([b.reward for b in batch]).to(self.device)
        advantages = self._normalize(rewards)

        losses = []
        for _ in range(self.epochs):
            action_masks = torch.cat([self._build_uniform_subgoal_mask(int(sg)) for sg in subgoals.tolist()], dim=0)
            dist = self.controller.dist(states, subgoals, action_mask=action_masks)
            new_log_probs = dist.log_prob(actions)

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            entropy = dist.entropy().mean()

            loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.controller.parameters(), max_norm=1.0)
            self.optimizer.step()
            losses.append(loss.item())

        loss_dict = {"controller_loss": float(np.mean(losses))}
        self.loss.append(loss_dict)
        return loss_dict
    def act(self, state):
        pass

    def push(self, reward):
        pass

class FeudalAgent:
    def __init__(self, manager_core: CentralManagerCore, controller_core: ClusterControllerCore, config: Dict):
        self.manager_core = manager_core
        self.controller_core = controller_core
        self.device = manager_core.device

        self.cluster_id = controller_core.cluster_id
        self.manager_period = int(config["manager_period"])
        self.num_subgoals = int(config["num_subgoals"])
        self.goal_switch_penalty = float(config.get("goal_switch_penalty", 0.0))
        self.intrinsic_reward_weight = float(config["intrinsic_reward_weight"])

        self.decision_count = 0
        self.current_subgoal: Optional[int] = None
        self.previous_subgoal: Optional[int] = None
        self.last_transition_stub = None

    def _to_tensor(self, state: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

    def _select_subgoal(self, state: np.ndarray) -> tuple[int, float]:
        state_tensor = self._to_tensor(state)
        if self.manager_core.use_cluster_embedding:
            cluster_tensor = torch.tensor([self.cluster_id], dtype=torch.long, device=self.device)
            out = self.manager_core.manager.act(state_tensor, cluster_ids=cluster_tensor, deterministic=self.manager_core.deterministic)
        else:
            out = self.manager_core.manager.act(state_tensor, deterministic=self.manager_core.deterministic)
        return out.subgoal, out.log_prob

    def act(self, state):
        state = np.asarray(state, dtype=np.float32)
        manager_step = self.current_subgoal is None or (self.decision_count % self.manager_period == 0)

        if manager_step:
            new_subgoal, manager_log_prob = self._select_subgoal(state)
            if self.current_subgoal is not None and new_subgoal != self.current_subgoal:
                self.previous_subgoal = self.current_subgoal
            self.current_subgoal = new_subgoal
        else:
            manager_log_prob = 0.0

        state_tensor = self._to_tensor(state)
        subgoal_tensor = torch.tensor([self.current_subgoal], dtype=torch.long, device=self.device)
        action_mask = self.controller_core._build_uniform_subgoal_mask(self.current_subgoal)

        controller_output = self.controller_core.controller.act(
            state_tensor, subgoal_tensor, action_mask=action_mask, deterministic=self.controller_core.deterministic
        )

        self.last_transition_stub = {
            "state": state.copy(),
            "subgoal": int(self.current_subgoal),
            "action": int(controller_output.action),
            "manager_log_prob": float(manager_log_prob),
            "controller_log_prob": float(controller_output.log_prob),
            "manager_step": manager_step,
        }
        self.decision_count += 1
        return int(controller_output.action)

    def push(self, reward):
        if self.last_transition_stub is None:
            return

        reward = float(reward)

        # Intrinsic reward for the controller based on subgoal switching and exploration
        intrinsic_reward = 0.0
        if self.previous_subgoal is not None and self.current_subgoal != self.previous_subgoal:
            intrinsic_reward -= self.goal_switch_penalty
        intrinsic_reward += 1.0 / max(self.num_subgoals, 1)
        controller_total_reward = reward + self.intrinsic_reward_weight * intrinsic_reward

        if self.last_transition_stub["manager_step"]:
            self.manager_core.memory.append(ManagerTransition(
                state=self.last_transition_stub["state"],
                cluster_id=self.cluster_id,
                subgoal=self.last_transition_stub["subgoal"],
                log_prob=self.last_transition_stub["manager_log_prob"],
                reward=reward
            ))

        self.controller_core.memory.append(ControllerTransition(
            state=self.last_transition_stub["state"],
            subgoal=self.last_transition_stub["subgoal"],
            action=self.last_transition_stub["action"],
            log_prob=self.last_transition_stub["controller_log_prob"],
            reward=controller_total_reward
        ))

        self.last_transition_stub = None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=str, required=True)
    parser.add_argument("--env-conf", type=str, default="config1")
    parser.add_argument("--task-conf", type=str, required=True)
    parser.add_argument("--alg-conf", type=str, required=True)
    parser.add_argument("--net", type=str, required=True)
    parser.add_argument("--env-seed", type=int, default=42)
    parser.add_argument("--torch-seed", type=int, default=42)
    args = parser.parse_args()

    ALGORITHM = "feudal_hrl_central_manager"
    exp_id = args.id
    alg_config = args.alg_conf
    env_config = args.env_conf
    task_config = args.task_conf
    network = args.net
    env_seed = args.env_seed
    torch_seed = args.torch_seed

    print("### STARTING EXPERIMENT ###")
    print(f"Algorithm: Central Manager + Distributed Controllers")
    print(f"Experiment ID: {exp_id}")

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    logging.getLogger("matplotlib").setLevel(logging.ERROR)

    torch.manual_seed(torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(torch_seed)
        torch.cuda.manual_seed_all(torch_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(env_seed)
    np.random.seed(env_seed)

    device = torch.device(0) if torch.cuda.is_available() else torch.device("cpu")
    print("Device is:", device)

    params = {}
    alg_params = json.load(open(f"../config/algo_config/{ALGORITHM}/{alg_config}.json"))
    env_params = json.load(open(f"../config/env_config/{env_config}.json"))
    task_params = json.load(open(f"../config/task_config/{task_config}.json"))
    params.update(alg_params)
    params.update(env_params)
    params.update(task_params)
    del params["desc"]

    for key, value in params.items():
        globals()[key] = value

    custom_network_folder = f"../networks/{network}"
    phases = [1, human_learning_episodes, int(training_eps) + human_learning_episodes]
    phase_names = ["Human stabilization", "Mutation and AV learning", "Testing phase"]
    records_folder = f"../results/{exp_id}"
    plots_folder = f"../results/{exp_id}/plots"

    od_file_path = os.path.join(custom_network_folder, f"od_{network}.txt")
    with open(od_file_path, "r", encoding="utf-8") as f:
        data = ast.literal_eval(f.read())
    origins = data["origins"]
    destinations = data["destinations"]

    agents_csv_path = os.path.join(custom_network_folder, "agents.csv")
    num_agents = len(pd.read_csv(agents_csv_path))
    if os.path.exists(agents_csv_path):
        os.makedirs(records_folder, exist_ok=True)
        new_agents_csv_path = os.path.join(records_folder, "agents.csv")
        with open(agents_csv_path, "r", encoding="utf-8") as src, open(new_agents_csv_path, "w", encoding="utf-8") as dst:
            dst.write(src.read())
        max_start_time = pd.read_csv(new_agents_csv_path)["start_time"].max()

    num_machines = int(num_agents * ratio_machines)
    total_episodes = human_learning_episodes + training_eps + test_eps

    cluster_csv_path = None
    if alg_params.get("use_cluster_embedding", False) and alg_params.get("cluster_csv_path"):
        cluster_csv_path = os.path.join(repo_root, alg_params["cluster_csv_path"])

    key_columns = alg_params.get("cluster_key_columns", ["start_time", "origin", "destination"])
    agent_cluster_map = {}

    if alg_params.get("use_cluster_embedding", False) and cluster_csv_path and os.path.exists(cluster_csv_path):
        cluster_lookup, num_clusters = load_cluster_lookup(cluster_csv_path, key_columns)
        agent_cluster_map, missing_indices = build_agent_cluster_map(agents_csv_path, cluster_lookup, key_columns)
        params["num_clusters"] = num_clusters
    else:
        params["num_clusters"] = 1

    exp_config_path = os.path.join(records_folder, "exp_config.json")
    dump_config = params.copy()
    dump_config.update({
        "network": network, "env_seed": env_seed, "torch_seed": torch_seed,
        "algorithm": ALGORITHM, "num_agents": num_agents, "num_machines": num_machines,
    })
    with open(exp_config_path, "w", encoding="utf-8") as f:
        json.dump(dump_config, f, indent=4)

    wandb.init(entity="mk-hrl", project="sandbox", name=exp_id, config=dump_config)

    env = TrafficEnvironment(
        seed=env_seed, create_agents=False, create_paths=True, save_detectors_info=False,
        agent_parameters={
            "new_machines_after_mutation": num_machines,
            "human_parameters": {"model": human_model, "alpha": human_alpha, "beta": human_beta, "beta_randomness": human_beta_randomness, "deterministic": human_deterministic},
            "machine_parameters": {"behavior": av_behavior, "observation_type": "previous_agents_plus_start_time"},
        },
        environment_parameters={"save_every": save_every},
        simulator_parameters={"network_name": network, "custom_network_folder": custom_network_folder, "sumo_type": "sumo", "simulation_timesteps": max_start_time},
        plotter_parameters={"phases": phases, "phase_names": phase_names, "smooth_by": smooth_by, "plot_choices": plot_choices, "records_folder": records_folder, "plots_folder": plots_folder},
        path_generation_parameters={"origins": origins, "destinations": destinations, "number_of_paths": number_of_paths, "beta": path_gen_beta, "num_samples": num_samples, "path_gen_workers": path_gen_workers, "visualize_paths": False},
    )

    env.start()
    env.reset()

    pbar = tqdm(total=total_episodes, desc="Human learning")
    for _ in range(human_learning_episodes):
        env.step()
        pbar.update()

    env.mutation(disable_human_learning=not should_humans_adapt, mutation_start_percentile=-1)

    obs_size = env.observation_space(env.possible_agents[0]).shape[0]

    central_manager = CentralManagerCore(state_size=obs_size, config=params, device=device)
    cluster_controllers = {}

    for idx, agent_obj in enumerate(env.machine_agents):
        try:
            agent_int_id = int(str(agent_obj.id).split('_')[-1])
        except:
            agent_int_id = idx

        c_id = agent_cluster_map.get(agent_int_id, 0)

        if c_id not in cluster_controllers:
            cluster_controllers[c_id] = ClusterControllerCore(
                state_size=obs_size,
                action_space_size=agent_obj.action_space_size,
                config=params,
                device=device,
                cluster_id=c_id
            )

        agent_obj.model = FeudalAgent(
            manager_core=central_manager,
            controller_core=cluster_controllers[c_id],
            config=params,
        )

    agent_lookup = {str(agent.id): agent for agent in env.machine_agents}

    os.makedirs(plots_folder, exist_ok=True)
    pbar.set_description("AV learning")
    for episode in range(training_eps):
        env.reset()
        episode_rewards, episode_travel_times = [], []
        manager_losses, controller_losses = [], []

        for agent_id in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()

            agent_lookup[agent_id].model.push(reward)

            if termination or truncation:
                reward = float(reward)
                episode_rewards.append(reward)
                if isinstance(info, dict) and "travel_time" in info:
                    episode_travel_times.append(float(info["travel_time"]))
                else:
                    episode_travel_times.append(-reward)

                if episode % update_every == 0:
                    m_loss = central_manager.learn()
                    if m_loss: manager_losses.append(m_loss["manager_loss"])

                    c_loss = agent_lookup[agent_id].model.controller_core.learn()
                    if c_loss: controller_losses.append(c_loss["controller_loss"])

                action = None
            else:
                action = agent_lookup[agent_id].model.act(observation)

            env.step(action)

        log_data = {
            "episode": human_learning_episodes + episode,
            "training/reward_sum": float(np.sum(episode_rewards)),
            "training/reward_mean": float(np.mean(episode_rewards)),
            "training/travel_time_mean": float(np.mean(episode_travel_times)),
        }
        if manager_losses: log_data["training/manager_loss"] = float(np.mean(manager_losses))
        if controller_losses: log_data["training/controller_loss"] = float(np.mean(controller_losses))

        wandb.log(log_data, step=human_learning_episodes + episode)

        if episode % plot_every == 0:
            env.plot_results()
        pbar.update()

    central_manager.deterministic = True
    central_manager.manager.eval()
    for core in cluster_controllers.values():
        core.deterministic = True
        core.controller.eval()

    pbar.set_description("Testing")
    for episode in range(test_eps):
        env.reset()
        episode_rewards, episode_travel_times = [], []

        for agent_id in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()

            if termination or truncation:
                episode_rewards.append(float(reward))
                if isinstance(info, dict) and "travel_time" in info:
                    episode_travel_times.append(float(info["travel_time"]))
                action = None
            else:
                action = agent_lookup[agent_id].model.act(observation)
            env.step(action)

        wandb.log({
            "episode": human_learning_episodes + training_eps + episode,
            "testing/reward_sum": float(np.sum(episode_rewards)),
            "testing/travel_time_mean": float(np.mean(episode_travel_times)),
        }, step=human_learning_episodes + training_eps + episode)
        pbar.update()

    pbar.close()
    env.plot_results()

    loss_records = []
    for i, m_loss in enumerate(central_manager.loss, start=1):
        loss_records.append({"iteration": i, "cluster_id": "GLOBAL", "manager_loss": m_loss["manager_loss"], "controller_loss": 0, "loss": m_loss["manager_loss"]})

    for c_id, core in cluster_controllers.items():
        for i, c_loss in enumerate(core.loss, start=1):
            loss_records.append({"iteration": i, "cluster_id": c_id, "manager_loss": 0, "controller_loss": c_loss["controller_loss"], "loss": c_loss["controller_loss"]})

    save_loss_records(records_folder, loss_records, columns=["iteration", "cluster_id", "manager_loss", "controller_loss", "loss"])

    env.stop_simulation()
    clear_SUMO_files(os.path.join(records_folder, "SUMO_output"), os.path.join(records_folder, "episodes"), remove_additional_files=True)
    run_metrics_analysis(exp_id, results_folder="../results")

    plots_to_log = {}
    if os.path.exists(os.path.join(plots_folder, "rewards.png")): plots_to_log["Plots/Rewards"] = wandb.Image(os.path.join(plots_folder, "rewards.png"))
    if os.path.exists(os.path.join(plots_folder, "travel_times.png")): plots_to_log["Plots/Travel_Times"] = wandb.Image(os.path.join(plots_folder, "travel_times.png"))
    if plots_to_log: wandb.log(plots_to_log)

    wandb.finish()