import argparse
import os
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

LOG_STD_MAX = 2
LOG_STD_MIN = -5

class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.single_action_space.high - env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.single_action_space.high + env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default="Hopper-v4", help="the id of the environment")
    parser.add_argument("--model-path", type=str, required=True, help="path to the trained model (.cleanrl_model file)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    env = gym.make(args.env_id, render_mode="human")
    
    # Mocking single_action_space and single_observation_space required by the exact CleanRL Agent
    env.single_action_space = env.action_space
    env.single_observation_space = env.observation_space
    
    agent = Actor(env)
    
    # Load the trained model weights
    if not os.path.exists(args.model_path):
        print(f"Error: The model path '{args.model_path}' does not exist.")
        exit(1)
        
    agent.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))
    agent.eval()

    # Run a single episode
    obs, info = env.reset()
    done = False
    total_reward = 0

    print(f"Starting test episode for environment: {args.env_id}")
    while not done:
        obs_tensor = torch.Tensor(obs).unsqueeze(0)
        
        with torch.no_grad():
            _, _, action = agent.get_action(obs_tensor) # Use mean action for testing
            
        action = action.cpu().numpy()[0]
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

    print(f"Episode finished. Total Reward: {total_reward}")
    env.close()

