import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pong_env import PongSinglePlayerEnv

# Define the policy network
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_dim)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)

def train():
    # Replace 'YourCustomPongEnv-v0' with the actual name of your custom environment
    env =  PongSinglePlayerEnv()
    input_dim = env.observation_space.shape[0]  # Should be 6 as per your observation space
    output_dim = env.action_space.n  # Should be 3 as per your action space
    policy = PolicyNetwork(input_dim, output_dim)
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)
    
    num_episodes = 1000
    gamma = 0.99  # Discount factor
    
    for episode in range(num_episodes):
        state = env.reset()
        log_probs = []
        rewards = []
        done = False
        
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            action_probs = policy(state_tensor)
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            # print(action)
            log_prob = dist.log_prob(action)
            next_state, reward, done, _ = env.step(action.item())
            
            log_probs.append(log_prob)
            rewards.append(reward)
            state = next_state
        
        # Compute discounted rewards
        discounted_rewards = []
        cumulative_reward = 0
        for r in reversed(rewards):
            cumulative_reward = r + gamma * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)
        discounted_rewards = torch.tensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)
        
        # Compute policy gradient loss
        policy_loss = []
        for log_prob, Gt in zip(log_probs, discounted_rewards):
            policy_loss.append(-log_prob * Gt)
        policy_loss = torch.stack(policy_loss).sum()
        
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        
        if episode % 50 == 0:
            total_reward = sum(rewards)
            print(f'Episode {episode}, Total Reward: {total_reward}')

if __name__ == "__main__":
    train()
