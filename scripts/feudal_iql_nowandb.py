"""
Feudal-IQL experiment for the HierarchicalRL / URB repository.

This script is intentionally designed as a controlled ablation between flat IQL
and cluster-level Feudal IQL. Low-level workers remain independent value-based
learners; one high-level DQN manager is shared by AVs in each cluster.

The script merges the existing IQL and Feudal-HRL config files with the same
config name, so it can be dropped into scripts/ without requiring a new config
folder immediately.
"""
from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import random
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

os.chdir(os.path.dirname(os.path.abspath(__file__)))
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from baseline_models import BaseLearningModel
from routerl import TrafficEnvironment
from utils import (
    clear_SUMO_files,
    print_agent_counts,
    run_metrics_analysis,
    save_loss_records,
    script_path_for_config,
)

try:
    import wandb
except ImportError:
    wandb = None


def load_cluster_lookup(cluster_csv_path: str, key_columns: List[str]) -> Tuple[Dict[Tuple, int], int]:
    """Load clustering output and map raw labels to contiguous ids.

    Cluster id 0 is reserved for unmatched agents; real clusters become 1..K.
    """
    df = pd.read_csv(cluster_csv_path)
    if "cluster" not in df.columns:
        raise ValueError(f"No 'cluster' column in {cluster_csv_path}")
    missing_columns = [c for c in key_columns if c not in df.columns]
    if missing_columns:
        raise ValueError(f"Cluster CSV missing key columns {missing_columns}: {cluster_csv_path}")

    unique_clusters = sorted(df["cluster"].dropna().unique().tolist())
    cluster_to_idx = {raw: i + 1 for i, raw in enumerate(unique_clusters)}
    lookup = {}
    for _, row in df.iterrows():
        key = tuple(row[col] for col in key_columns)
        lookup[key] = cluster_to_idx[row["cluster"]]
    return lookup, len(unique_clusters) + 1


def build_agent_cluster_map(agents_csv_path: str, cluster_lookup: Dict[Tuple, int], key_columns: List[str]) -> Tuple[Dict[int, int], List[int]]:
    """Map rows of agents.csv to cluster ids using the configured join key."""
    agents_df = pd.read_csv(agents_csv_path)
    missing_columns = [c for c in key_columns if c not in agents_df.columns]
    if missing_columns:
        raise ValueError(f"agents.csv missing key columns {missing_columns}: {agents_csv_path}")

    cluster_map, missing = {}, []
    for idx, row in agents_df.iterrows():
        key = tuple(row[col] for col in key_columns)
        if key in cluster_lookup:
            cluster_map[idx] = int(cluster_lookup[key])
        else:
            cluster_map[idx] = 0
            missing.append(idx)
    return cluster_map, missing


def agent_integer_id(agent_obj, fallback: int) -> int:
    try:
        return int(str(agent_obj.id).split("_")[-1])
    except (TypeError, ValueError):
        return int(fallback)


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dims: List[int]):
        super().__init__()
        dims = [in_dim] + list(hidden_dims)
        layers = []
        for din, dout in zip(dims[:-1], dims[1:]):
            layers.extend([nn.Linear(din, dout), nn.ReLU()])
        layers.append(nn.Linear(dims[-1], out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class WorkerTransition:
    state: np.ndarray
    subgoal: int
    action: int
    reward: float


@dataclass
class ManagerTransition:
    state: np.ndarray
    subgoal: int
    reward: float


class IndependentGoalConditionedDQN(BaseLearningModel):
    """Independent IQL worker conditioned on a shared cluster subgoal."""

    def __init__(self, state_size: int, action_space_size: int, num_subgoals: int, device: torch.device, config: Dict):
        super().__init__()
        self.device = device
        self.state_size = int(state_size)
        self.action_space_size = int(action_space_size)
        self.num_subgoals = int(num_subgoals)
        self.epsilon = float(config.get("eps_init", 0.99))
        self.eps_decay = float(config.get("eps_decay", 0.998))
        self.batch_size = int(config.get("batch_size", 16))
        self.num_epochs = int(config.get("num_epochs", 1))
        self.intrinsic_reward_weight = float(config.get("intrinsic_reward_weight", 0.0))
        self.goal_consistency_reward = float(config.get("goal_consistency_reward", 1.0))
        self.action_mask_strategy = str(config.get("action_mask_strategy", "uniform_bins"))
        self.memory = deque(maxlen=int(config.get("buffer_size", 256)))

        widths = config.get("widths", [32, 64, 32])
        hidden_dims = list(widths[:-1]) if len(widths) > 1 else [32, 64]
        if not hidden_dims:
            hidden_dims = [32, 64]

        self.q_network = MLP(self.state_size + self.num_subgoals, self.action_space_size, hidden_dims).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=float(config.get("lr", 0.003)))
        self.loss_fn = nn.MSELoss()
        self.loss: List[float] = []
        self.last_state: Optional[np.ndarray] = None
        self.last_subgoal: Optional[int] = None
        self.last_action: Optional[int] = None

    def _goal_one_hot(self, subgoals: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.one_hot(subgoals.long(), num_classes=self.num_subgoals).float()

    def _network_input(self, states: torch.Tensor, subgoals: torch.Tensor) -> torch.Tensor:
        return torch.cat([states, self._goal_one_hot(subgoals)], dim=-1)

    def _allowed_actions(self, subgoal: int) -> np.ndarray:
        if self.action_mask_strategy not in {"uniform_bins", "goal_bins"}:
            return np.arange(self.action_space_size)
        bins = np.array_split(np.arange(self.action_space_size), self.num_subgoals)
        chosen = bins[int(subgoal) % self.num_subgoals]
        return chosen if len(chosen) else np.arange(self.action_space_size)

    def _intrinsic_reward(self, subgoal: int, action: int) -> float:
        allowed = self._allowed_actions(subgoal)
        return self.goal_consistency_reward if action in set(allowed.tolist()) else 0.0

    def act(self, state, subgoal: int) -> int:
        state_np = np.asarray(state, dtype=np.float32)
        allowed = self._allowed_actions(subgoal)

        if np.random.rand() < self.epsilon:
            action = int(np.random.choice(allowed))
        else:
            state_tensor = torch.as_tensor(state_np, dtype=torch.float32, device=self.device).unsqueeze(0)
            sg_tensor = torch.tensor([subgoal], dtype=torch.long, device=self.device)
            with torch.no_grad():
                q_values = self.q_network(self._network_input(state_tensor, sg_tensor))[0]
            masked = torch.full_like(q_values, float("-inf"))
            allowed_tensor = torch.as_tensor(allowed, dtype=torch.long, device=self.device)
            masked[allowed_tensor] = q_values[allowed_tensor]
            action = int(torch.argmax(masked).item())

        self.last_state = state_np.copy()
        self.last_subgoal = int(subgoal)
        self.last_action = int(action)
        return action

    def push(self, reward: float) -> None:
        if self.last_state is None or self.last_action is None or self.last_subgoal is None:
            return
        intrinsic = self._intrinsic_reward(self.last_subgoal, self.last_action)
        shaped_reward = float(reward) + self.intrinsic_reward_weight * intrinsic
        self.memory.append(WorkerTransition(self.last_state, self.last_subgoal, self.last_action, shaped_reward))
        self.last_state = None
        self.last_subgoal = None
        self.last_action = None

    def learn(self) -> Optional[float]:
        if len(self.memory) < self.batch_size:
            return None

        losses = []
        for _ in range(self.num_epochs):
            batch = random.sample(self.memory, self.batch_size)
            states = torch.as_tensor(np.stack([b.state for b in batch]), dtype=torch.float32, device=self.device)
            subgoals = torch.as_tensor([b.subgoal for b in batch], dtype=torch.long, device=self.device)
            actions = torch.as_tensor([b.action for b in batch], dtype=torch.long, device=self.device).unsqueeze(1)
            rewards = torch.as_tensor([b.reward for b in batch], dtype=torch.float32, device=self.device).unsqueeze(1)
            q_values = self.q_network(self._network_input(states, subgoals)).gather(1, actions)
            loss = self.loss_fn(q_values, rewards)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
            self.optimizer.step()
            losses.append(float(loss.item()))

        mean_loss = float(np.mean(losses))
        self.loss.append(mean_loss)
        self.epsilon *= self.eps_decay
        return mean_loss


class ClusterManagerDQN:
    """One high-level DQN manager shared by all AVs in one cluster."""

    def __init__(self, state_size: int, num_subgoals: int, device: torch.device, config: Dict):
        self.device = device
        self.num_subgoals = int(num_subgoals)
        self.epsilon = float(config.get("manager_eps_init", config.get("eps_init", 0.99)))
        self.eps_decay = float(config.get("manager_eps_decay", config.get("eps_decay", 0.998)))
        self.batch_size = int(config.get("manager_batch_size", config.get("batch_size", 16)))
        self.num_epochs = int(config.get("manager_num_epochs", 1))
        self.manager_reward_weight = float(config.get("manager_reward_weight", 1.0))
        hidden_dims = list(config.get("manager_hidden_dims", [64, 64]))
        self.q_network = MLP(state_size, self.num_subgoals, hidden_dims).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=float(config.get("manager_lr", config.get("lr", 0.003))))
        self.loss_fn = nn.MSELoss()
        self.memory = deque(maxlen=int(config.get("manager_buffer_size", config.get("buffer_size", 256))))
        self.loss: List[float] = []

    def act(self, cluster_state: np.ndarray) -> int:
        if np.random.rand() < self.epsilon:
            return int(np.random.choice(self.num_subgoals))
        state_tensor = torch.as_tensor(cluster_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return int(torch.argmax(q_values, dim=-1).item())

    def push(self, cluster_state: np.ndarray, subgoal: int, cluster_reward: float) -> None:
        self.memory.append(ManagerTransition(np.asarray(cluster_state, dtype=np.float32).copy(), int(subgoal), self.manager_reward_weight * float(cluster_reward)))

    def learn(self) -> Optional[float]:
        if len(self.memory) < self.batch_size:
            return None
        losses = []
        for _ in range(self.num_epochs):
            batch = random.sample(self.memory, self.batch_size)
            states = torch.as_tensor(np.stack([b.state for b in batch]), dtype=torch.float32, device=self.device)
            subgoals = torch.as_tensor([b.subgoal for b in batch], dtype=torch.long, device=self.device).unsqueeze(1)
            rewards = torch.as_tensor([b.reward for b in batch], dtype=torch.float32, device=self.device).unsqueeze(1)
            q_values = self.q_network(states).gather(1, subgoals)
            loss = self.loss_fn(q_values, rewards)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
            self.optimizer.step()
            losses.append(float(loss.item()))
        mean_loss = float(np.mean(losses))
        self.loss.append(mean_loss)
        self.epsilon *= self.eps_decay
        return mean_loss


def safe_cluster_state(env, cluster_agent_ids: List[str], state_size: int) -> np.ndarray:
    """Mean-field aggregation of current observations for one cluster."""
    observations = []
    for agent_id in cluster_agent_ids:
        try:
            obs = env.observe(agent_id)
            if obs is not None:
                observations.append(np.asarray(obs, dtype=np.float32))
        except Exception:
            continue
    if not observations:
        return np.zeros(state_size, dtype=np.float32)
    return np.mean(np.stack(observations), axis=0).astype(np.float32)


def maybe_init_wandb(exp_id: str, config: Dict):
    if wandb is None:
        return None
    try:
        return wandb.init(entity="mk-hrl", project="sandbox", name=exp_id, config=config)
    except Exception as exc:
        logging.warning("W&B initialization failed; continuing without it: %s", exc)
        return None


def maybe_wandb_log(data: Dict, step: int) -> None:
    if wandb is not None and wandb.run is not None:
        wandb.log(data, step=step)


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

    ALGORITHM = "feudal_iql"
    exp_id, alg_config = args.id, args.alg_conf
    env_config, task_config = args.env_conf, args.task_conf
    network, env_seed, torch_seed = args.net, args.env_seed, args.torch_seed

    print("### STARTING EXPERIMENT ###")
    print(f"Algorithm: {ALGORITHM.upper()}")
    print(f"Experiment ID: {exp_id}")
    print(f"Network: {network}")
    print(f"Environment seed: {env_seed}")
    print(f"Torch seed: {torch_seed}")

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

    # Merge existing IQL and Feudal-HRL configs.
    with open(f"../config/algo_config/iql/{alg_config}.json", "r", encoding="utf-8") as f:
        iql_params = json.load(f)
    with open(f"../config/algo_config/feudal_hrl/{alg_config}.json", "r", encoding="utf-8") as f:
        feudal_params = json.load(f)
    with open(f"../config/env_config/{env_config}.json", "r", encoding="utf-8") as f:
        env_params = json.load(f)
    with open(f"../config/task_config/{task_config}.json", "r", encoding="utf-8") as f:
        task_params = json.load(f)

    alg_params = dict(iql_params)
    alg_params.update(feudal_params)
    params = {}
    params.update(alg_params)
    params.update(env_params)
    params.update(task_params)
    params.pop("desc", None)
    for key, value in params.items():
        globals()[key] = value

    num_subgoals = int(params.get("num_subgoals", 4))
    manager_period = int(params.get("manager_period", 4))

    custom_network_folder = f"../networks/{network}"
    phases = [1, human_learning_episodes, int(training_eps) + human_learning_episodes]
    phase_names = ["Human stabilization", "Mutation and AV learning", "Testing phase"]
    records_folder = f"../results/{exp_id}"
    plots_folder = f"../results/{exp_id}/plots"

    od_file_path = os.path.join(custom_network_folder, f"od_{network}.txt")
    with open(od_file_path, "r", encoding="utf-8") as f:
        data = ast.literal_eval(f.read())
    origins, destinations = data["origins"], data["destinations"]

    agents_csv_path = os.path.join(custom_network_folder, "agents.csv")
    if not os.path.exists(agents_csv_path):
        raise FileNotFoundError(f"Agents CSV file not found at {agents_csv_path}.")
    agents_df = pd.read_csv(agents_csv_path)
    num_agents = len(agents_df)
    os.makedirs(records_folder, exist_ok=True)
    new_agents_csv_path = os.path.join(records_folder, "agents.csv")
    agents_df.to_csv(new_agents_csv_path, index=False)
    max_start_time = agents_df["start_time"].max()

    num_machines = int(num_agents * ratio_machines)
    total_episodes = human_learning_episodes + training_eps + test_eps

    key_columns = alg_params.get("cluster_key_columns", ["start_time", "origin", "destination"])
    cluster_csv_config = alg_params.get("cluster_csv_path")
    if not cluster_csv_config:
        raise ValueError("Feudal-IQL requires `cluster_csv_path` in the Feudal HRL config.")
    cluster_csv_path = os.path.join(repo_root, cluster_csv_config)
    if not os.path.exists(cluster_csv_path):
        raise FileNotFoundError(f"Cluster CSV does not exist: {cluster_csv_path}")

    cluster_lookup, num_clusters = load_cluster_lookup(cluster_csv_path, key_columns)
    agent_cluster_map, missing_indices = build_agent_cluster_map(agents_csv_path, cluster_lookup, key_columns)

    params.update({
        "num_clusters": num_clusters,
        "algorithm": ALGORITHM,
        "network": network,
        "env_seed": env_seed,
        "torch_seed": torch_seed,
        "env_config": env_config,
        "task_config": task_config,
        "alg_config": alg_config,
        "script": script_path_for_config(__file__),
        "num_agents": num_agents,
        "num_machines": num_machines,
        "cluster_csv_resolved": cluster_csv_path,
        "cluster_unmatched_agents": len(missing_indices),
    })

    print(f"Detected clusters (including fallback 0): {num_clusters}")
    print(f"Unmatched agent rows assigned to cluster 0: {len(missing_indices)}")

    with open(os.path.join(records_folder, "exp_config.json"), "w", encoding="utf-8") as f:
        json.dump(params, f, indent=4)
    maybe_init_wandb(exp_id, params)

    env = TrafficEnvironment(
        seed=env_seed,
        create_agents=False,
        create_paths=True,
        save_detectors_info=False,
        agent_parameters={
            "new_machines_after_mutation": num_machines,
            "human_parameters": {
                "model": human_model,
                "alpha": human_alpha,
                "beta": human_beta,
                "beta_randomness": human_beta_randomness,
                "deterministic": human_deterministic,
            },
            "machine_parameters": {
                "behavior": av_behavior,
                "observation_type": observations,
            },
        },
        environment_parameters={"save_every": save_every},
        simulator_parameters={
            "network_name": network,
            "custom_network_folder": custom_network_folder,
            "sumo_type": "sumo",
            "simulation_timesteps": max_start_time,
        },
        plotter_parameters={
            "phases": phases,
            "phase_names": phase_names,
            "smooth_by": smooth_by,
            "plot_choices": plot_choices,
            "records_folder": records_folder,
            "plots_folder": plots_folder,
        },
        path_generation_parameters={
            "origins": origins,
            "destinations": destinations,
            "number_of_paths": number_of_paths,
            "beta": path_gen_beta,
            "num_samples": num_samples,
            "path_gen_workers": path_gen_workers,
            "visualize_paths": False,
        },
    )

    env.start()
    env.reset()
    print_agent_counts(env)
    pbar = tqdm(total=total_episodes, desc="Human learning")
    for _ in range(human_learning_episodes):
        env.step()
        pbar.update()

    env.mutation(disable_human_learning=not should_humans_adapt, mutation_start_percentile=-1)
    print_agent_counts(env)
    obs_size = env.observation_space(env.possible_agents[0]).shape[0]

    agent_to_cluster = {}
    cluster_to_agent_ids = defaultdict(list)
    for idx, agent_obj in enumerate(env.machine_agents):
        raw_id = agent_integer_id(agent_obj, idx)
        c_id = int(agent_cluster_map.get(raw_id, 0))
        aid = str(agent_obj.id)
        agent_to_cluster[aid] = c_id
        cluster_to_agent_ids[c_id].append(aid)

    for agent_obj in env.machine_agents:
        agent_obj.model = IndependentGoalConditionedDQN(
            state_size=obs_size,
            action_space_size=agent_obj.action_space_size,
            num_subgoals=num_subgoals,
            device=device,
            config=params,
        )
    agent_lookup = {str(agent.id): agent for agent in env.machine_agents}

    active_clusters = sorted(cluster_to_agent_ids.keys())
    cluster_managers = {
        c_id: ClusterManagerDQN(obs_size, num_subgoals, device, params)
        for c_id in active_clusters
    }
    print("Active AV clusters:", {c: len(cluster_to_agent_ids[c]) for c in active_clusters})

    os.makedirs(plots_folder, exist_ok=True)
    current_cluster_goal = {c_id: 0 for c_id in active_clusters}
    cluster_goal_state = {}

    pbar.set_description("AV learning Feudal-IQL")
    for episode in range(training_eps):
        env.reset()
        manager_step = episode == 0 or episode % manager_period == 0

        if manager_step:
            for c_id in active_clusters:
                c_state = safe_cluster_state(env, cluster_to_agent_ids[c_id], obs_size)
                cluster_goal_state[c_id] = c_state
                current_cluster_goal[c_id] = cluster_managers[c_id].act(c_state)

        episode_rewards, episode_travel_times, worker_losses = [], [], []
        cluster_rewards = defaultdict(list)

        for agent_id in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            model = agent_lookup[agent_id].model
            model.push(reward)

            if termination or truncation:
                reward_f = float(reward)
                episode_rewards.append(reward_f)
                c_id = agent_to_cluster.get(agent_id, 0)
                cluster_rewards[c_id].append(reward_f)
                episode_travel_times.append(float(info["travel_time"]) if isinstance(info, dict) and "travel_time" in info else -reward_f)
                if episode % update_every == 0:
                    loss = model.learn()
                    if loss is not None:
                        worker_losses.append(loss)
                action = None
            else:
                c_id = agent_to_cluster.get(agent_id, 0)
                action = model.act(observation, current_cluster_goal.get(c_id, 0))
            env.step(action)

        manager_losses = []
        if manager_step:
            for c_id in active_clusters:
                rewards_c = cluster_rewards.get(c_id, [])
                if not rewards_c:
                    continue
                cluster_managers[c_id].push(cluster_goal_state[c_id], current_cluster_goal[c_id], float(np.mean(rewards_c)))
                if episode % update_every == 0:
                    loss = cluster_managers[c_id].learn()
                    if loss is not None:
                        manager_losses.append(loss)

        log_data = {
            "episode": human_learning_episodes + episode,
            "training/reward_sum": float(np.sum(episode_rewards)) if episode_rewards else 0.0,
            "training/reward_mean": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
            "training/travel_time_mean": float(np.mean(episode_travel_times)) if episode_travel_times else 0.0,
            "training/travel_time_sum": float(np.sum(episode_travel_times)) if episode_travel_times else 0.0,
            "training/worker_loss": float(np.mean(worker_losses)) if worker_losses else 0.0,
            "training/manager_loss": float(np.mean(manager_losses)) if manager_losses else 0.0,
            "training/worker_epsilon_mean": float(np.mean([a.model.epsilon for a in env.machine_agents])),
            "training/manager_epsilon_mean": float(np.mean([m.epsilon for m in cluster_managers.values()])),
        }
        for c_id in active_clusters:
            log_data[f"subgoal/cluster_{c_id}"] = int(current_cluster_goal[c_id])
        maybe_wandb_log(log_data, step=human_learning_episodes + episode)

        if episode % plot_every == 0:
            env.plot_results()
        pbar.update()

    # Deterministic testing.
    for agent in env.machine_agents:
        agent.model.epsilon = 0.0
        agent.model.q_network.eval()
    for manager in cluster_managers.values():
        manager.epsilon = 0.0
        manager.q_network.eval()

    pbar.set_description("Testing Feudal-IQL")
    for episode in range(test_eps):
        env.reset()
        if episode == 0 or episode % manager_period == 0:
            for c_id in active_clusters:
                c_state = safe_cluster_state(env, cluster_to_agent_ids[c_id], obs_size)
                current_cluster_goal[c_id] = cluster_managers[c_id].act(c_state)

        episode_rewards, episode_travel_times = [], []
        for agent_id in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            if termination or truncation:
                reward_f = float(reward)
                episode_rewards.append(reward_f)
                episode_travel_times.append(float(info["travel_time"]) if isinstance(info, dict) and "travel_time" in info else -reward_f)
                action = None
            else:
                c_id = agent_to_cluster.get(agent_id, 0)
                action = agent_lookup[agent_id].model.act(observation, current_cluster_goal.get(c_id, 0))
            env.step(action)

        maybe_wandb_log({
            "episode": human_learning_episodes + training_eps + episode,
            "testing/reward_sum": float(np.sum(episode_rewards)) if episode_rewards else 0.0,
            "testing/reward_mean": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
            "testing/travel_time_mean": float(np.mean(episode_travel_times)) if episode_travel_times else 0.0,
            "testing/travel_time_sum": float(np.sum(episode_travel_times)) if episode_travel_times else 0.0,
        }, step=human_learning_episodes + training_eps + episode)
        pbar.update()

    pbar.close()
    env.plot_results()

    worker_loss_records = []
    for agent in env.machine_agents:
        for iteration, loss_value in enumerate(agent.model.loss, start=1):
            worker_loss_records.append({"iteration": iteration, "agent_id": agent.id, "loss": loss_value})
    save_loss_records(records_folder, worker_loss_records, columns=["iteration", "agent_id", "loss"])

    manager_rows = []
    for c_id, manager in cluster_managers.items():
        for iteration, loss_value in enumerate(manager.loss, start=1):
            manager_rows.append({"iteration": iteration, "cluster_id": c_id, "manager_loss": loss_value})
    pd.DataFrame(manager_rows).to_csv(os.path.join(records_folder, "manager_losses.csv"), index=False)

    env.stop_simulation()
    clear_SUMO_files(os.path.join(records_folder, "SUMO_output"), os.path.join(records_folder, "episodes"), remove_additional_files=True)
    run_metrics_analysis(exp_id, results_folder="../results")

    if wandb is not None and wandb.run is not None:
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
