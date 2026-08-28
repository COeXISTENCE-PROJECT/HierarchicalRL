"""Feudal MAPPO HRL experiment script for URB-style single-step route decisions."""
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
import torch.nn.functional as F
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
from utils import (  # type: ignore
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

# --- TUTAJ ZACZYNA SIĘ TWÓJ KOD ---
# @dataclass
# class Transition:
# ... i tak dalej, aż do samego dołu ...

from dataclasses import dataclass
import torch.nn.functional as F

@dataclass
class Transition:
    state: np.ndarray          # Lokalna obserwacja (dla Aktorów)
    global_state: np.ndarray   # Stan klastra (dla Krytyka)
    subgoal: int
    action: int
    manager_log_prob: float
    controller_log_prob: float
    extrinsic_reward: float
    intrinsic_reward: float
    manager_step: bool

class CentralizedCritic(nn.Module):
    def __init__(self, obs_dim: int, hidden_dims: list):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        return self.net(global_state)

class ClusterMAPPOAgent(BaseLearningModel):
    def __init__(
        self,
        state_size: int,
        action_space_size: int,
        config: dict,
        device: torch.device,
        cluster_id: int = 0,
    ):
        super().__init__()
        self.device = device
        self.cluster_id = int(cluster_id)
        self.action_space_size = int(action_space_size)
        self.num_subgoals = int(config["num_subgoals"])
        
        # Hyperparametry PPO i Feudal
        self.manager_period = int(config["manager_period"])
        self.batch_size = int(config["batch_size"])
        self.gamma = float(config.get("gamma", 0.99))
        self.gae_lambda = float(config.get("gae_lambda", 0.95))
        
        self.manager_clip_eps = float(config["manager_clip_eps"])
        self.controller_clip_eps = float(config["controller_clip_eps"])
        self.intrinsic_reward_weight = float(config["intrinsic_reward_weight"])
        self.manager_reward_weight = float(config["manager_reward_weight"])
        
        self.deterministic = False
        self.memory: List[Transition] = []
        self.loss: List[Dict[str, float]] = []
        
        # --- SIECI WSPÓŁDZIELONE DLA CAŁEGO KLASTRA ---
        self.manager = FeudalManager(
            obs_dim=state_size,
            num_subgoals=self.num_subgoals,
            hidden_dims=config["manager_hidden_dims"],
        ).to(self.device)
        
        self.controller = FeudalController(
            obs_dim=state_size,
            action_dim=self.action_space_size,
            num_subgoals=self.num_subgoals,
            hidden_dims=config["controller_hidden_dims"],
            subgoal_embed_dim=int(config["subgoal_embed_dim"]),
        ).to(self.device)

        # Nowość: Krytyk dla całego klastra
        self.critic = CentralizedCritic(
            obs_dim=state_size, 
            hidden_dims=config.get("critic_hidden_dims", [64, 64])
        ).to(self.device)

        # Wspólny optymalizator dla wszystkich trzech sieci
        self.optimizer = build_mlp_optimizer(
            nn.ModuleList([self.manager, self.controller, self.critic]), 
            float(config["manager_lr"]) # Możesz rozdzielić LR w razie potrzeby
        )

    def _to_tensor(self, state: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

    def _build_uniform_subgoal_mask(self, subgoal: int) -> torch.Tensor:
        # Analogicznie do starego kodu
        bins = np.array_split(np.arange(self.action_space_size), self.num_subgoals)
        rotated = (subgoal + self.cluster_id) % self.num_subgoals
        chosen = bins[rotated]
        mask = torch.zeros((1, self.action_space_size), dtype=torch.float32, device=self.device)
        mask[0, chosen] = 1.0
        return mask

    def push(self, transition: Transition):
        """Po prostu wrzuca gotowy obiekt do wspólnego wora klastra."""
        self.memory.append(transition)
    def act(self, local_state: np.ndarray, current_subgoal: int, manager_step: bool) -> tuple:
        """Wybiera akcję dla konkretnego agenta na podstawie jego lokalnego stanu."""
        state_tensor = self._to_tensor(local_state)
        
        if manager_step:
            manager_out = self.manager.act(state_tensor, deterministic=self.deterministic)
            new_subgoal, manager_log_prob = manager_out.subgoal, manager_out.log_prob
        else:
            new_subgoal, manager_log_prob = current_subgoal, 0.0

        subgoal_tensor = torch.tensor([new_subgoal], dtype=torch.long, device=self.device)
        action_mask = self._build_uniform_subgoal_mask(new_subgoal)
        
        controller_out = self.controller.act(
            state_tensor, subgoal_tensor, action_mask=action_mask, deterministic=self.deterministic
        )
        
        return int(new_subgoal), manager_log_prob, int(controller_out.action), float(controller_out.log_prob)

    def compute_gae(self, rewards, values, next_value):
        """Oblicza GAE na podstawie ocen Krytyka."""
        advantages = []
        gae = 0
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * next_value - values[step]
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages.insert(0, gae)
            next_value = values[step]
        return torch.tensor(advantages, dtype=torch.float32, device=self.device)

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
            
        batch = self.memory[:]
        self.memory.clear()
        
        # Przygotowanie tensorów
        local_states = torch.as_tensor(np.stack([b.state for b in batch]), dtype=torch.float32, device=self.device)
        global_states = torch.as_tensor(np.stack([b.global_state for b in batch]), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor([b.action for b in batch], dtype=torch.long, device=self.device)
        subgoals = torch.as_tensor([b.subgoal for b in batch], dtype=torch.long, device=self.device)
        old_ctrl_log_probs = torch.as_tensor([b.controller_log_prob for b in batch], dtype=torch.float32, device=self.device)
        
        # Sumowanie nagród
        rewards = [b.extrinsic_reward + self.intrinsic_reward_weight * b.intrinsic_reward for b in batch]
        
        # 1. Krytyk ocenia stany globalne
        values = self.critic(global_states).squeeze()
        # W uproszczeniu dla PPO zakładamy next_value = 0 dla końca paczki
        advantages = self.compute_gae(rewards, values.tolist(), next_value=0.0)
        returns = advantages + values.detach()
        
        # Normalizacja Advantage
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 2. Aktualizacja Controllera (Mikro)
        action_masks = torch.cat([self._build_uniform_subgoal_mask(int(sg)) for sg in subgoals.tolist()], dim=0)
        ctrl_dist = self.controller.dist(local_states, subgoals, action_mask=action_masks)
        new_ctrl_log_probs = ctrl_dist.log_prob(actions)
        
        ratio = torch.exp(new_ctrl_log_probs - old_ctrl_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.controller_clip_eps, 1 + self.controller_clip_eps) * advantages
        controller_loss = -torch.min(surr1, surr2).mean()

        # 3. Aktualizacja Managera
        manager_batch = [b for b in batch if b.manager_step]
        if manager_batch:
            m_states = torch.as_tensor(np.stack([b.state for b in manager_batch]), dtype=torch.float32, device=self.device)
            m_subgoals = torch.as_tensor([b.subgoal for b in manager_batch], dtype=torch.long, device=self.device)
            old_m_log_probs = torch.as_tensor([b.manager_log_prob for b in manager_batch], dtype=torch.float32, device=self.device)
            m_rewards = torch.as_tensor([self.manager_reward_weight * b.extrinsic_reward for b in manager_batch], dtype=torch.float32, device=self.device)
            
            # Normalizacja nagród dla Managera
            m_advantages = (m_rewards - m_rewards.mean()) / (m_rewards.std() + 1e-8)
            
            m_dist = self.manager.dist(m_states)
            new_m_log_probs = m_dist.log_prob(m_subgoals)
            m_ratio = torch.exp(new_m_log_probs - old_m_log_probs)
            m_surr1 = m_ratio * m_advantages
            m_surr2 = torch.clamp(m_ratio, 1 - self.manager_clip_eps, 1 + self.manager_clip_eps) * m_advantages
            m_entropy = m_dist.entropy().mean()
            manager_loss = -torch.min(m_surr1, m_surr2).mean() - float(self.manager_entropy_coef) * m_entropy
        else:
            manager_loss = torch.tensor(0.0, device=self.device)

        # 4. Aktualizacja Krytyka
        critic_loss = F.mse_loss(self.critic(global_states).squeeze(), returns)

        # 5. Wspólna Optymalizacja
        total_loss = controller_loss + manager_loss + 0.5 * critic_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.manager.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.controller.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        self.loss.append({
            "manager_loss": float(manager_loss.item()),
            "controller_loss": float(controller_loss.item()), 
            "critic_loss": float(critic_loss.item())
        })

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

    ALGORITHM = "feudal_hrl_mappo"
    exp_id = args.id
    alg_config = args.alg_conf
    env_config = args.env_conf
    task_config = args.task_conf
    network = args.net
    env_seed = args.env_seed
    torch_seed = args.torch_seed

    print("### STARTING EXPERIMENT ###")
    print(f"Algorithm: {ALGORITHM.upper()}")
    print(f"Experiment ID: {exp_id}")
    print(f"Network: {network}")
    print(f"Environment seed: {env_seed}")
    print(f"Torch seed: {torch_seed}")
    print(f"Algorithm config: {alg_config}")
    print(f"Environment config: {env_config}")
    print(f"Task config: {task_config}")

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
    else:
        raise FileNotFoundError(f"Agents CSV file not found at {agents_csv_path}.")

    num_machines = int(num_agents * ratio_machines)
    total_episodes = human_learning_episodes + training_eps + test_eps

    cluster_csv_path = None
    if alg_params.get("use_cluster_embedding", False) and alg_params.get("cluster_csv_path"):
        cluster_csv_path = os.path.join(repo_root, alg_params["cluster_csv_path"])
    
    key_columns = alg_params.get("cluster_key_columns", ["start_time", "origin", "destination"])
    agent_cluster_map = {}

    if alg_params.get("use_cluster_embedding", False):
        if cluster_csv_path and os.path.exists(cluster_csv_path):
            cluster_lookup, num_clusters = load_cluster_lookup(cluster_csv_path, key_columns)
            agent_cluster_map, missing_indices = build_agent_cluster_map(agents_csv_path, cluster_lookup, key_columns)
            params["num_clusters"] = num_clusters
        else:
            raise FileNotFoundError(
                f"\n[BŁĄD HRL] 'use_cluster_embedding'jest na true, ale nie znaleziono pliku: '{cluster_csv_path}'. "
                f"Sprawdź ścieżkę w config.json!"
            )
    else:
        params["num_clusters"] = 1

    exp_config_path = os.path.join(records_folder, "exp_config.json")
    dump_config = params.copy()
    dump_config.update(
        {
            "network": network,
            "env_seed": env_seed,
            "torch_seed": torch_seed,
            "env_config": env_config,
            "task_config": task_config,
            "alg_config": alg_config,
            "script": script_path_for_config(__file__),
            "algorithm": ALGORITHM,
            "num_agents": num_agents,
            "num_machines": num_machines,
        }
    )
    with open(exp_config_path, "w", encoding="utf-8") as f:
        json.dump(dump_config, f, indent=4)

    wandb.init(
        entity="mk-hrl",
        project="sandbox",
        name=exp_id,
        config=dump_config,
    )

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
                "observation_type": "previous_agents_plus_start_time",
            },
        },
        environment_parameters={
            "save_every": save_every,
        },
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

    env.mutation(
        disable_human_learning=not should_humans_adapt,
        mutation_start_percentile=-1,
    )
    print_agent_counts(env)

    obs_size = env.observation_space(env.possible_agents[0]).shape[0]
    
    # 1. Inicjalizacja Klastrów (Jeden współdzielony model na klaster!)
    unique_clusters = set(agent_cluster_map.values())
    if not unique_clusters: # Zabezpieczenie na wypadek braku przypisań
        unique_clusters = {0}
        
    cluster_models = {}
    for c_id in unique_clusters:
        cluster_models[c_id] = ClusterMAPPOAgent(
            state_size=obs_size,
            action_space_size=env.action_space(env.possible_agents[0]).n,
            config=params,
            device=device,
            cluster_id=c_id
        )

    # Agent_lookup nie służy już do przechowywania osobnych modeli, a jedynie wskaźników do klastra
    agent_to_cluster = {}
    for idx in range(len(env.machine_agents)):
        agent_obj = env.machine_agents[idx]
        try:
            agent_int_id = int(str(agent_obj.id).split('_')[-1])
        except:
            agent_int_id = idx
        c_id = agent_cluster_map.get(agent_int_id, 0)
        agent_to_cluster[str(agent_obj.id)] = c_id

    os.makedirs(plots_folder, exist_ok=True)
    pbar.set_description("AV learning MAPPO")
    
    for episode in range(training_eps):
        env.reset()
        episode_rewards = []
        episode_travel_times = []
        
        # Słownik do trzymania tymczasowego stanu dla poszczególnych agentów (bo modele tego już nie robią)
        agent_context = {} 

        for agent_id in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            
            c_id = agent_to_cluster.get(agent_id, 0)
            model = cluster_models[c_id]

            # 2. Obliczanie Global State (Mean Field dla danego klastra)
            cluster_active_agents = [a for a in env.agents if agent_to_cluster.get(a, 0) == c_id]
            if len(cluster_active_agents) > 0:
                cluster_obs = [env.observe(a) for a in cluster_active_agents]
                global_state = np.mean(cluster_obs, axis=0)
            else:
                global_state = observation # Fallback

            # 3. Zrzut danych z poprzedniego kroku (jeśli istnieje)
            if agent_id in agent_context:
                prev = agent_context[agent_id]
                
                # Obliczanie kary za zmianę celu
                intrinsic_reward = 1.0 / max(params["num_subgoals"], 1)
                if prev.get("previous_subgoal") is not None and prev["subgoal"] != prev["previous_subgoal"]:
                    intrinsic_reward -= params.get("goal_switch_penalty", 0.0)
                
                transition = Transition(
                    state=prev["state"],
                    global_state=prev["global_state"],
                    subgoal=prev["subgoal"],
                    action=prev["action"],
                    manager_log_prob=prev["manager_log_prob"],
                    controller_log_prob=prev["controller_log_prob"],
                    extrinsic_reward=float(reward),
                    intrinsic_reward=intrinsic_reward,
                    manager_step=prev["manager_step"]
                )
                model.push(transition)

            # 4. Sprawdzenie stanu terminalnego
            if termination or truncation:
                reward = float(reward)
                episode_rewards.append(reward)
                if isinstance(info, dict) and "travel_time" in info:
                    episode_travel_times.append(float(info["travel_time"]))
                else:
                    episode_travel_times.append(-reward)
                
                # Model uczy się asynchronicznie, gdy zbierze pełen batch w pamięci klastra
                model.learn()
                action = None
                
                if agent_id in agent_context:
                    del agent_context[agent_id] # Sprzątamy
            else:
                # 5. Wybór nowej akcji
                step_count = agent_context.get(agent_id, {}).get("step_count", 0)
                manager_step = (step_count % params["manager_period"] == 0)
                current_sg = agent_context.get(agent_id, {}).get("subgoal", 0)
                
                new_subgoal, manager_log_prob, action, controller_log_prob = model.act(observation, current_sg, manager_step)
                
                # Zapisanie bieżącego kontekstu agenta (stan lokalny i globalny!)
                agent_context[agent_id] = {
                    "state": observation.copy(),
                    "global_state": global_state.copy(),
                    "subgoal": new_subgoal,
                    "previous_subgoal": current_sg if manager_step else agent_context.get(agent_id, {}).get("previous_subgoal"),
                    "action": action,
                    "manager_log_prob": manager_log_prob,
                    "controller_log_prob": controller_log_prob,
                    "manager_step": manager_step,
                    "step_count": step_count + 1
                }
            
            env.step(action)

        # Logowanie W&B i wykresy po epizodzie
        log_data = {
            "episode": human_learning_episodes + episode,
            "training/reward_sum": float(np.sum(episode_rewards)),
            "training/reward_mean": float(np.mean(episode_rewards)),
            "training/travel_time_mean": float(np.mean(episode_travel_times)),
            "training/travel_time_sum": float(np.sum(episode_travel_times)),
        }
        
        # Agregacja logów strat z modeli klastrów
        ep_c_loss = [loss["controller_loss"] for m in cluster_models.values() for loss in m.loss if m.loss]
        ep_m_loss = [loss["manager_loss"] for m in cluster_models.values() for loss in m.loss if m.loss]
        ep_crit_loss = [loss["critic_loss"] for m in cluster_models.values() for loss in m.loss if m.loss]
        
        if ep_c_loss:
            log_data.update({
                "training/manager_loss": float(np.mean(ep_m_loss)),
                "training/controller_loss": float(np.mean(ep_c_loss)),
                "training/critic_loss": float(np.mean(ep_crit_loss)),
            })
            
        wandb.log(log_data, step=human_learning_episodes + episode)

        if episode % plot_every == 0:
            env.plot_results()
        pbar.update()

    # --- FAZA TESTOWA ---
    for model in cluster_models.values():
        model.deterministic = True
        model.manager.eval()
        model.controller.eval()
        model.critic.eval()

    pbar.set_description("Testing")
    for episode in range(test_eps):
        env.reset()
        episode_rewards = []
        episode_travel_times = []
        agent_context = {}

        for agent_id in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            
            if termination or truncation:
                reward = float(reward)
                episode_rewards.append(reward)
                if isinstance(info, dict) and "travel_time" in info:
                    episode_travel_times.append(float(info["travel_time"]))
                else:
                    episode_travel_times.append(-reward)
                action = None
            else:
                c_id = agent_to_cluster.get(agent_id, 0)
                model = cluster_models[c_id]
                
                step_count = agent_context.get(agent_id, {}).get("step_count", 0)
                manager_step = (step_count % params["manager_period"] == 0)
                current_sg = agent_context.get(agent_id, {}).get("subgoal", 0)
                
                new_subgoal, _, action, _ = model.act(observation, current_sg, manager_step)
                
                agent_context[agent_id] = {
                    "subgoal": new_subgoal,
                    "step_count": step_count + 1
                }
            env.step(action)

        wandb.log(
            {
                "episode": human_learning_episodes + training_eps + episode,
                "testing/reward_sum": float(np.sum(episode_rewards)),
                "testing/reward_mean": float(np.mean(episode_rewards)),
                "testing/travel_time_mean": float(np.mean(episode_travel_times)),
                "testing/travel_time_sum": float(np.sum(episode_travel_times)),
            },
            step=human_learning_episodes + training_eps + episode,
        )
        pbar.update()

    pbar.close()
    env.plot_results()

    loss_records = []
    for c_id, model in cluster_models.items():
        for iteration, loss_value in enumerate(model.loss, start=1):
            loss_records.append(
                {
                    "iteration": iteration,
                    "cluster_id": c_id,
                    "manager_loss": loss_value["manager_loss"],
                    "controller_loss": loss_value["controller_loss"],
                    "critic_loss": loss_value["critic_loss"],
                }
            )
    save_loss_records(
        records_folder,
        loss_records,
        columns=["iteration", "cluster_id", "manager_loss", "controller_loss", "critic_loss"],
    )

    env.stop_simulation()
    clear_SUMO_files(
        os.path.join(records_folder, "SUMO_output"),
        os.path.join(records_folder, "episodes"),
        remove_additional_files=True,
    )
    run_metrics_analysis(exp_id, results_folder="../results")
    
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