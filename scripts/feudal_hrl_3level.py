# Global Director +  Cluster Manager + Vehicle Controller (Agent)

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
    for idx, row in agents_df.iterrows():
        key = tuple(row[col] for col in key_columns)
        cluster_map[idx] = int(cluster_lookup[key]) if key in cluster_lookup else 0
    return cluster_map

def build_mlp_optimizer(module: nn.Module, lr: float) -> optim.Optimizer:
    return optim.Adam(module.parameters(), lr=lr)


@dataclass
class GlobalTransition:
    state: np.ndarray
    action: int # Global Goal
    log_prob: float
    reward: float

@dataclass
class ManagerTransition:
    state: np.ndarray
    condition: int # Global Goal (input)
    action: int # Cluster Subgoal (output)
    log_prob: float
    reward: float

@dataclass
class ControllerTransition:
    state: np.ndarray
    condition: int # Cluster Subgoal (input)
    action: int # Route (output)
    log_prob: float
    reward: float


class GlobalDirectorCore(BaseLearningModel):
    def __init__(self, state_size: int, config: Dict, device: torch.device):
        super().__init__()
        self.device = device
        self.num_global_goals = int(config.get("num_global_goals", 3))
        self.batch_size = int(config.get("batch_size", 64))
        self.epochs = int(config.get("global_epochs", config.get("manager_epochs", 3)))
        self.clip_eps = float(config.get("manager_clip_eps", 0.2))
        self.entropy_coef = float(config.get("global_entropy_coef", 0.01))

        self.memory: List[GlobalTransition] = []
        self.loss: List[Dict[str, float]] = []

        self.director = FeudalManager(
            obs_dim=state_size,
            num_subgoals=self.num_global_goals,
            hidden_dims=config.get("global_hidden_dims", [128, 128]),
            use_cluster_embedding=False,
            num_clusters=1,
            cluster_embed_dim=8,
        ).to(self.device)

        self.optimizer = build_mlp_optimizer(self.director, float(config.get("global_lr", 0.0003)))
        self.deterministic = False

    def act(self, state): pass
    def push(self, reward): pass

    def learn(self) -> Optional[Dict[str, float]]:
        if len(self.memory) < self.batch_size: return None

        batch = self.memory[:]
        self.memory.clear()

        states = torch.FloatTensor(np.stack([b.state for b in batch])).to(self.device)
        actions = torch.LongTensor([b.action for b in batch]).to(self.device)
        old_log_probs = torch.FloatTensor([b.log_prob for b in batch]).to(self.device)
        rewards = torch.FloatTensor([b.reward for b in batch]).to(self.device)
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8) if len(rewards) > 1 else rewards

        losses = []
        for _ in range(self.epochs):
            dist = self.director.dist(states)
            new_log_probs = dist.log_prob(actions)
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            entropy = dist.entropy().mean()
            loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.director.parameters(), 1.0)
            self.optimizer.step()
            losses.append(loss.item())

        loss_dict = {"global_loss": float(np.mean(losses))}
        self.loss.append(loss_dict)
        return loss_dict


class ClusterCore(BaseLearningModel):
    def __init__(self, state_size: int, action_space_size: int, config: Dict, device: torch.device, cluster_id: int):
        super().__init__()
        self.device = device
        self.cluster_id = cluster_id

        self.num_global_goals = int(config.get("num_global_goals", 3))
        self.num_cluster_subgoals = int(config["num_subgoals"])
        self.action_space_size = action_space_size

        self.batch_size = int(config["batch_size"])
        self.manager_epochs = int(config["manager_epochs"])
        self.controller_epochs = int(config["controller_epochs"])
        self.clip_eps = float(config["controller_clip_eps"])
        self.m_entropy_coef = float(config["manager_entropy_coef"])
        self.c_entropy_coef = float(config["controller_entropy_coef"])
        self.action_mask_strategy = str(config.get("action_mask_strategy", "uniform_bins"))

        self.manager_memory: List[ManagerTransition] = []
        self.controller_memory: List[ControllerTransition] = []
        self.loss: List[Dict[str, float]] = []

        #State + Global Goal -> Cluster Subgoal
        self.manager = FeudalController(
            obs_dim=state_size,
            action_dim=self.num_cluster_subgoals,
            num_subgoals=self.num_global_goals,
            hidden_dims=config["manager_hidden_dims"],
            subgoal_embed_dim=int(config.get("global_embed_dim", 8)),
        ).to(self.device)

        # State + Cluster Subgoal -> Route
        self.controller = FeudalController(
            obs_dim=state_size,
            action_dim=self.action_space_size,
            num_subgoals=self.num_cluster_subgoals,
            hidden_dims=config["controller_hidden_dims"],
            subgoal_embed_dim=int(config["subgoal_embed_dim"]),
        ).to(self.device)

        self.manager_optimizer = build_mlp_optimizer(self.manager, float(config["manager_lr"]))
        self.controller_optimizer = build_mlp_optimizer(self.controller, float(config["controller_lr"]))
        self.deterministic = False

    def act(self, state): pass
    def push(self, reward): pass

    def _build_route_mask(self, cluster_subgoal: int) -> torch.Tensor:
        if self.action_mask_strategy != "uniform_bins":
            return torch.ones((1, self.action_space_size), dtype=torch.float32, device=self.device)
        bins = np.array_split(np.arange(self.action_space_size), self.num_cluster_subgoals)
        chosen = bins[cluster_subgoal % self.num_cluster_subgoals]
        mask = torch.zeros((1, self.action_space_size), dtype=torch.float32, device=self.device)
        mask[0, chosen] = 1.0
        return mask

    def learn(self) -> Optional[Dict[str, float]]:
        loss_dict = {}

        #manager learning
        if len(self.manager_memory) >= self.batch_size:
            b_m = self.manager_memory[:]
            self.manager_memory.clear()
            s = torch.FloatTensor(np.stack([b.state for b in b_m])).to(self.device)
            cond = torch.LongTensor([b.condition for b in b_m]).to(self.device)
            act = torch.LongTensor([b.action for b in b_m]).to(self.device)
            old_lp = torch.FloatTensor([b.log_prob for b in b_m]).to(self.device)
            rew = torch.FloatTensor([b.reward for b in b_m]).to(self.device)
            adv = (rew - rew.mean()) / (rew.std() + 1e-8) if len(rew) > 1 else rew

            # No masking for manager
            mask_ones = torch.ones((len(b_m), self.num_cluster_subgoals), device=self.device)

            m_losses = []
            for _ in range(self.manager_epochs):
                dist = self.manager.dist(s, cond, action_mask=mask_ones)
                new_lp = dist.log_prob(act)
                ratio = torch.exp(new_lp - old_lp)
                loss = -torch.min(ratio * adv, torch.clamp(ratio, 1-self.clip_eps, 1+self.clip_eps) * adv).mean() - self.m_entropy_coef * dist.entropy().mean()
                self.manager_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.manager.parameters(), 1.0)
                self.manager_optimizer.step()
                m_losses.append(loss.item())
            loss_dict["manager_loss"] = float(np.mean(m_losses))

        #controller learning
        if len(self.controller_memory) >= self.batch_size:
            b_c = self.controller_memory[:]
            self.controller_memory.clear()
            s = torch.FloatTensor(np.stack([b.state for b in b_c])).to(self.device)
            cond = torch.LongTensor([b.condition for b in b_c]).to(self.device)
            act = torch.LongTensor([b.action for b in b_c]).to(self.device)
            old_lp = torch.FloatTensor([b.log_prob for b in b_c]).to(self.device)
            rew = torch.FloatTensor([b.reward for b in b_c]).to(self.device)
            adv = (rew - rew.mean()) / (rew.std() + 1e-8) if len(rew) > 1 else rew

            c_losses = []
            for _ in range(self.controller_epochs):
                masks = torch.cat([self._build_route_mask(int(sg)) for sg in cond.tolist()], dim=0)
                dist = self.controller.dist(s, cond, action_mask=masks)
                new_lp = dist.log_prob(act)
                ratio = torch.exp(new_lp - old_lp)
                loss = -torch.min(ratio * adv, torch.clamp(ratio, 1-self.clip_eps, 1+self.clip_eps) * adv).mean() - self.c_entropy_coef * dist.entropy().mean()
                self.controller_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.controller.parameters(), 1.0)
                self.controller_optimizer.step()
                c_losses.append(loss.item())
            loss_dict["controller_loss"] = float(np.mean(c_losses))

        if loss_dict:
            self.loss.append(loss_dict)
            return loss_dict
        return None


class FeudalAgent3L:
    def __init__(self, global_core: GlobalDirectorCore, cluster_core: ClusterCore, config: Dict):
        self.global_core = global_core
        self.cluster_core = cluster_core
        self.device = global_core.device

        self.global_period = int(config.get("global_period", 15))
        self.manager_period = int(config.get("manager_period", 5))

        self.decision_count = 0
        self.current_global_goal: Optional[int] = None
        self.current_cluster_subgoal: Optional[int] = None

        self.last_global_stub = None
        self.last_manager_stub = None
        self.last_controller_stub = None

    def act(self, state: np.ndarray) -> int:
        state_np = np.asarray(state, dtype=np.float32)
        state_t = torch.FloatTensor(state_np).unsqueeze(0).to(self.device)

        #Global level decision
        is_global_step = (self.current_global_goal is None) or (self.decision_count % self.global_period == 0)
        if is_global_step:
            g_dist = self.global_core.director.act(state_t, deterministic=self.global_core.deterministic)
            self.current_global_goal = int(g_dist.subgoal)
            self.last_global_stub = {"state": state_np.copy(), "action": self.current_global_goal, "log_prob": float(g_dist.log_prob)}

        #Cluster manager level decision
        is_manager_step = (self.current_cluster_subgoal is None) or (self.decision_count % self.manager_period == 0)
        if is_manager_step:
            g_goal_t = torch.tensor([self.current_global_goal], dtype=torch.long, device=self.device)
            m_mask = torch.ones((1, self.cluster_core.num_cluster_subgoals), dtype=torch.float32, device=self.device)
            m_dist = self.cluster_core.manager.act(state_t, g_goal_t, action_mask=m_mask, deterministic=self.cluster_core.deterministic)
            self.current_cluster_subgoal = int(m_dist.action)
            self.last_manager_stub = {"state": state_np.copy(), "condition": self.current_global_goal, "action": self.current_cluster_subgoal, "log_prob": float(m_dist.log_prob)}

        #Controller level decision
        c_subgoal_t = torch.tensor([self.current_cluster_subgoal], dtype=torch.long, device=self.device)
        c_mask = self.cluster_core._build_route_mask(self.current_cluster_subgoal)
        c_dist = self.cluster_core.controller.act(state_t, c_subgoal_t, action_mask=c_mask, deterministic=self.cluster_core.deterministic)

        self.last_controller_stub = {"state": state_np.copy(), "condition": self.current_cluster_subgoal, "action": int(c_dist.action), "log_prob": float(c_dist.log_prob)}
        self.decision_count += 1

        return int(c_dist.action)

    def push(self, reward):
        reward = float(reward)
        if self.last_global_stub:
            self.global_core.memory.append(GlobalTransition(**self.last_global_stub, reward=reward))
            self.last_global_stub = None

        if self.last_manager_stub:
            self.cluster_core.manager_memory.append(ManagerTransition(**self.last_manager_stub, reward=reward))
            self.last_manager_stub = None

        if self.last_controller_stub:
            self.cluster_core.controller_memory.append(ControllerTransition(**self.last_controller_stub, reward=reward))
            self.last_controller_stub = None


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

    ALGORITHM = "feudal_hrl_3level"
    exp_id, alg_config, env_config, task_config = args.id, args.alg_conf, args.env_conf, args.task_conf
    network, env_seed, torch_seed = args.net, args.env_seed, args.torch_seed

    print("### STARTING EXPERIMENT ###")
    print(f"Algorithm: 3-LEVEL HRL (Global -> Cluster -> Vehicle)")

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    logging.getLogger("matplotlib").setLevel(logging.ERROR)

    torch.manual_seed(torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(torch_seed)
    torch.backends.cudnn.deterministic = True
    random.seed(env_seed)
    np.random.seed(env_seed)

    device = torch.device(0) if torch.cuda.is_available() else torch.device("cpu")

    params = {}
    params.update(json.load(open(f"../config/algo_config/feudal_hrl_3level/{alg_config}.json")))
    params.update(json.load(open(f"../config/env_config/{env_config}.json")))
    params.update(json.load(open(f"../config/task_config/{task_config}.json")))
    if "desc" in params: del params["desc"]
    for key, value in params.items(): globals()[key] = value

    custom_network_folder = f"../networks/{network}"
    phases = [1, human_learning_episodes, int(training_eps) + human_learning_episodes]
    phase_names = ["Human stabilization", "Mutation and AV learning", "Testing phase"]
    records_folder, plots_folder = f"../results/{exp_id}", f"../results/{exp_id}/plots"

    with open(os.path.join(custom_network_folder, f"od_{network}.txt"), "r") as f:
        data = ast.literal_eval(f.read())
    origins, destinations = data["origins"], data["destinations"]

    agents_csv_path = os.path.join(custom_network_folder, "agents.csv")
    num_agents = len(pd.read_csv(agents_csv_path))
    os.makedirs(records_folder, exist_ok=True)
    new_agents_csv_path = os.path.join(records_folder, "agents.csv")
    with open(agents_csv_path, "r") as src, open(new_agents_csv_path, "w") as dst: dst.write(src.read())
    max_start_time = pd.read_csv(new_agents_csv_path)["start_time"].max()

    num_machines = int(num_agents * ratio_machines)
    total_episodes = human_learning_episodes + training_eps + test_eps

    cluster_csv_path = os.path.join(repo_root, params["cluster_csv_path"]) if params.get("use_cluster_embedding", False) and params.get("cluster_csv_path") else None
    key_columns = params.get("cluster_key_columns", ["start_time", "origin", "destination"])
    agent_cluster_map = {}

    if cluster_csv_path and os.path.exists(cluster_csv_path):
        cluster_lookup, num_clusters = load_cluster_lookup(cluster_csv_path, key_columns)
        agent_cluster_map = build_agent_cluster_map(agents_csv_path, cluster_lookup, key_columns)
        params["num_clusters"] = num_clusters
    else:
        params["num_clusters"] = 1

    dump_config = params.copy()
    dump_config.update({"network": network, "env_seed": env_seed, "torch_seed": torch_seed, "algorithm": ALGORITHM})
    with open(os.path.join(records_folder, "exp_config.json"), "w") as f: json.dump(dump_config, f, indent=4)
    wandb.init(entity="mk-hrl", project="sandbox", name=exp_id, config=dump_config)

    env = TrafficEnvironment(
        seed=env_seed, create_agents=False, create_paths=True, save_detectors_info=False,
        agent_parameters={"new_machines_after_mutation": num_machines, "human_parameters": {"model": human_model, "alpha": human_alpha, "beta": human_beta, "beta_randomness": human_beta_randomness, "deterministic": human_deterministic}, "machine_parameters": {"behavior": av_behavior, "observation_type": "previous_agents_plus_start_time"}},
        environment_parameters={"save_every": save_every}, simulator_parameters={"network_name": network, "custom_network_folder": custom_network_folder, "sumo_type": "sumo", "simulation_timesteps": max_start_time},
        plotter_parameters={"phases": phases, "phase_names": phase_names, "smooth_by": smooth_by, "plot_choices": plot_choices, "records_folder": records_folder, "plots_folder": plots_folder},
        path_generation_parameters={"origins": origins, "destinations": destinations, "number_of_paths": number_of_paths, "beta": path_gen_beta, "num_samples": num_samples, "path_gen_workers": 4, "visualize_paths": False}
    )

    env.start()
    env.reset()

    pbar = tqdm(total=total_episodes, desc="Human learning")
    for _ in range(human_learning_episodes): env.step(); pbar.update()

    env.mutation(disable_human_learning=not should_humans_adapt, mutation_start_percentile=-1)
    obs_size = env.observation_space(env.possible_agents[0]).shape[0]

    global_core = GlobalDirectorCore(state_size=obs_size, config=params, device=device)
    cluster_cores = {}

    for idx, agent_obj in enumerate(env.machine_agents):
        agent_int_id = int(str(agent_obj.id).split('_')[-1]) if '_' in str(agent_obj.id) else idx
        c_id = agent_cluster_map.get(agent_int_id, 0)

        if c_id not in cluster_cores:
            cluster_cores[c_id] = ClusterCore(state_size=obs_size, action_space_size=agent_obj.action_space_size, config=params, device=device, cluster_id=c_id)
        agent_obj.model = FeudalAgent3L(global_core=global_core, cluster_core=cluster_cores[c_id], config=params)

    agent_lookup = {str(a.id): a for a in env.machine_agents}
    os.makedirs(plots_folder, exist_ok=True)

    pbar.set_description("AV learning")
    for episode in range(training_eps):
        env.reset()
        ep_rews, ep_times, g_loss_list, m_loss_list, c_loss_list = [], [], [], [], []

        for agent_id in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            agent_lookup[agent_id].model.push(reward)

            if termination or truncation:
                ep_rews.append(float(reward))
                ep_times.append(float(info["travel_time"]) if isinstance(info, dict) and "travel_time" in info else -float(reward))

                if episode % update_every == 0:
                    g_res = global_core.learn()
                    if g_res: g_loss_list.append(g_res["global_loss"])

                    c_res = agent_lookup[agent_id].model.cluster_core.learn()
                    if c_res:
                        if "manager_loss" in c_res: m_loss_list.append(c_res["manager_loss"])
                        if "controller_loss" in c_res: c_loss_list.append(c_res["controller_loss"])
                action = None
            else:
                action = agent_lookup[agent_id].model.act(observation)
            env.step(action)

        log_data = {"episode": human_learning_episodes + episode, "train/reward_sum": float(np.sum(ep_rews)), "train/travel_time_mean": float(np.mean(ep_times))}
        if g_loss_list: log_data["train/global_loss"] = float(np.mean(g_loss_list))
        if m_loss_list: log_data["train/manager_loss"] = float(np.mean(m_loss_list))
        if c_loss_list: log_data["train/controller_loss"] = float(np.mean(c_loss_list))
        wandb.log(log_data, step=human_learning_episodes + episode)

        if episode % plot_every == 0: env.plot_results()
        pbar.update()

    global_core.deterministic = True
    global_core.director.eval()
    for core in cluster_cores.values():
        core.deterministic = True
        core.manager.eval()
        core.controller.eval()

    pbar.set_description("Testing")
    for episode in range(test_eps):
        env.reset()
        ep_rews, ep_times = [], []
        for agent_id in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            if termination or truncation:
                ep_rews.append(float(reward))
                if isinstance(info, dict) and "travel_time" in info: ep_times.append(float(info["travel_time"]))
                action = None
            else:
                action = agent_lookup[agent_id].model.act(observation)
            env.step(action)
        wandb.log({"episode": human_learning_episodes + training_eps + episode, "test/reward_sum": float(np.sum(ep_rews)), "test/travel_time_mean": float(np.mean(ep_times))}, step=human_learning_episodes + training_eps + episode)
        pbar.update()

    pbar.close()
    env.plot_results()

    loss_records = []
    for i, g_loss in enumerate(global_core.loss, start=1):
        loss_records.append({"iteration": i, "cluster_id": "GLOBAL", "manager_loss": g_loss["global_loss"], "controller_loss": 0})
    for c_id, core in cluster_cores.items():
        for i, c_loss in enumerate(core.loss, start=1):
            loss_records.append({"iteration": i, "cluster_id": c_id, "manager_loss": c_loss.get("manager_loss", 0), "controller_loss": c_loss.get("controller_loss", 0)})
    save_loss_records(records_folder, loss_records, columns=["iteration", "cluster_id", "manager_loss", "controller_loss"])

    env.stop_simulation()
    clear_SUMO_files(os.path.join(records_folder, "SUMO_output"), os.path.join(records_folder, "episodes"), remove_additional_files=True)
    run_metrics_analysis(exp_id, results_folder="../results")
    
    if wandb is not None and wandb.run is not None:
            # One W&B chart with exactly one line per active AV cluster.
            # x-axis spans AV training + deterministic testing episodes.
            if cluster_tt_steps and active_clusters:
                cluster_tt_plot = wandb.plot.line_series(
                    xs=cluster_tt_steps,
                    ys=[cluster_tt_history[c_id] for c_id in active_clusters],
                    keys=[f"Cluster {c_id}" for c_id in active_clusters],
                    title="Cluster-wise mean travel time",
                    xname="Episode",
                )
                wandb.log({"Plots/Cluster-wise Travel Time": cluster_tt_plot})
    
            rewards_path = os.path.join(plots_folder, "rewards.png")
            travel_times_path = os.path.join(plots_folder, "travel_times.png")
            plots_to_log = {}
            if os.path.exists(rewards_path):
                plots_to_log["Plots/Rewards"] = wandb.Image(rewards_path)
            if os.path.exists(travel_times_path):
                plots_to_log["Plots/Travel_Times"] = wandb.Image(travel_times_path)
            if plots_to_log:
                wandb.log(plots_to_log)
            
    wandb.finish()