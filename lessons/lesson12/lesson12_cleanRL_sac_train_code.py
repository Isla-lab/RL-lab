import argparse
import os
import random
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"))
    parser.add_argument("--env-id", type=str, default="Pendulum-v1")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--total-timesteps", type=int, default=50000)
    parser.add_argument("--buffer-size", type=int, default=int(1e6))
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-starts", type=int, default=5000)
    parser.add_argument("--policy-lr", type=float, default=3e-4)
    parser.add_argument("--q-lr", type=float, default=1e-3)
    parser.add_argument("--policy-frequency", type=int, default=2)
    parser.add_argument("--target-network-frequency", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--autotune", type=bool, default=True)
    return parser.parse_args()

def make_env(env_id, seed):
    def thunk():
        env = gym.make(env_id)
        env.action_space.seed(seed)
        return env
    return thunk

class ReplayBuffer:
    def __init__(self, capacity, obs_space, action_space, device):
        self.capacity = capacity
        self.device = device
        self.obs = np.zeros((capacity, *obs_space.shape), dtype=np.float32)
        self.next_obs = np.zeros((capacity, *obs_space.shape), dtype=np.float32)
        self.actions = np.zeros((capacity, *action_space.shape), dtype=np.float32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        self.size = 0

    def add(self, obs, next_obs, action, reward, done):
        self.obs[self.pos] = obs
        self.next_obs[self.pos] = next_obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.obs[idxs]).to(self.device),
            torch.as_tensor(self.actions[idxs]).to(self.device),
            torch.as_tensor(self.rewards[idxs]).to(self.device),
            torch.as_tensor(self.next_obs[idxs]).to(self.device),
            torch.as_tensor(self.dones[idxs]).to(self.device)
        )

class SoftQNetwork(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.fc1 = nn.Linear(np.array(envs.single_observation_space.shape).prod() + np.prod(envs.single_action_space.shape), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

LOG_STD_MAX = 2
LOG_STD_MIN = -5

class Actor(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.fc1 = nn.Linear(np.array(envs.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(envs.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(envs.single_action_space.shape))
        
        # FIXED: Changed 'env.single_action_space' to 'envs.single_action_space'
        self.register_buffer(
            "action_scale", torch.tensor((envs.single_action_space.high - envs.single_action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((envs.single_action_space.high + envs.single_action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)

        return mean, log_std

    def get_action(self, x):
        # TODO: Implement action sampling using reparameterization trick
        # NOTE: It must return (action, log_prob, mean)
        return None, None, None

if __name__ == "__main__":
    args = parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed)])
    
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    actor = Actor(envs).to(device)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    rb = ReplayBuffer(args.buffer_size, envs.single_observation_space, envs.single_action_space, device)
    
    start_time = time.time()
    obs, _ = envs.reset(seed=args.seed)
    
    episode_returns = []
    episode_lengths = []
    current_ep_return = 0.0
    current_ep_length = 0

    print(f"--- Starting SAC Training on {args.env_id} ---")
    
    for global_step in range(args.total_timesteps):
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(1)])
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            if actions is not None:
                actions = actions.detach().cpu().numpy()
            else:
                actions = np.array([envs.single_action_space.sample() for _ in range(1)]) # Fallback if get_action is unimplemented

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # Track returns manually (since no vector wrapper wrapper)
        current_ep_return += float(rewards[0])
        current_ep_length += 1

        if terminations[0] or truncations[0]:
            episode_returns.append(current_ep_return)
            episode_lengths.append(current_ep_length)
            
            avg_return = np.mean(episode_returns[-50:])
            print(f"Step: {global_step:7d} | Ep Return: {current_ep_return:7.2f} | Avg Return (last 50): {avg_return:7.2f} | Ep Length: {current_ep_length:4.0f}")
            
            current_ep_return = 0.0
            current_ep_length = 0

        # Handle real next observation for truncations
        real_next_obs = next_obs.copy()
        for idx, (term, trunc) in enumerate(zip(terminations, truncations)):
            if (term or trunc) and "final_observation" in infos:
                real_next_obs[idx] = infos["final_observation"][idx]
                
        rb.add(obs[0], real_next_obs[0], actions[0], rewards[0], terminations[0])
        obs = next_obs

        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            b_obs, b_actions, b_rewards, b_next_obs, b_dones = data
            
            # UPDATE Q-NETWORK (CRITICS)
            # TODO: Implement Q-function optimization
            
            # UPDATE POLICY (ACTOR)
            # TODO: Implement Policy optimization
            
            # UPDATE ALPHA (TEMPERATURE)
            # TODO: Implement Alpha tuning
            
            # UPDATE TARGET NETWORKS (POLYAK AVERAGING)
            # TODO: Implement target network updates

    envs.close()
    
    # Save model
    print("\n--- Saving Model ---")
    os.makedirs("models", exist_ok=True)
    model_path = f"models/sac_{args.env_id}_actor.pth"
    torch.save(actor.state_dict(), model_path)
    print(f"Success: Actor model saved to {model_path}")

    # Plotting learning curve
    print("\n--- Generating Plot ---")
    if len(episode_returns) > 0:
        plt.figure(figsize=(10, 5))
        plt.plot(episode_returns, label='Episodic Return', alpha=0.5, color='blue')
        
        window_size = min(50, len(episode_returns))
        moving_avg = np.convolve(episode_returns, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size - 1, len(episode_returns)), moving_avg, label=f'{window_size}-Episode Moving Average', color='red', linewidth=2)
        
        plt.title(f"SAC Learning Curve on {args.env_id}")
        plt.xlabel("Episode")
        plt.ylabel("Return")
        plt.legend()
        plt.grid(True)
        
        plot_filename = f"sac_learning_curve_{args.env_id}.png"
        plt.savefig(plot_filename)
        print(f"Success: Plot saved to {plot_filename}")
        plt.show()
    else:
        print("Error: No episodes completed during training.")