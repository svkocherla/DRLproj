import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from agents.base_agent import BaseAgent

class ActorCriticNetwork(nn.Module):
    def __init__(self, input_dim, num_actions):
        super(ActorCriticNetwork, self).__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
        self.actor = nn.Sequential(
            nn.Linear(256, num_actions),
            nn.Softmax(dim=-1)
        )
        
        self.critic = nn.Linear(256, 1)
        
    def forward(self, x):
        shared_features = self.shared_layers(x)
        action_probs = self.actor(shared_features)
        state_value = self.critic(shared_features)
        return action_probs, state_value

class ActorCriticAgent(BaseAgent):
    def __init__(self, action_space, observation_space,
                 learning_rate=3e-4,
                 gamma=0.99,
                 entropy_coef=0.01,
                 value_loss_coef=0.5,
                 max_grad_norm=0.5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        
        input_dim = observation_space.shape[0]
        num_actions = action_space.n
        
        self.network = ActorCriticNetwork(input_dim, num_actions).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        self.saved_log_probs = []
        self.saved_state_values = []
        self.saved_entropy = []
        self.saved_rewards = []
        
    def act(self, observation, greedy=False):
        state = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        action_probs, state_value = self.network(state)
        
        if greedy:
            action = torch.argmax(action_probs).item()
        else:
            m = Categorical(action_probs)
            action = m.sample()
            self.saved_log_probs.append(m.log_prob(action))
            self.saved_entropy.append(m.entropy())
            self.saved_state_values.append(state_value)
            action = action.item()
            
        return action
    
    def learn(self, reward, next_observation, done):
        self.saved_rewards.append(reward)
        
        if done:
            returns = []
            advantages = []
            R = 0
            
            for r, v in zip(reversed(self.saved_rewards), 
                          reversed([v.item() for v in self.saved_state_values])):
                R = r + self.gamma * R
                advantage = R - v
                returns.insert(0, R)
                advantages.insert(0, advantage)
            
            returns = torch.tensor(returns).to(self.device)
            advantages = torch.tensor(advantages).to(self.device)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            policy_loss = []
            value_loss = []
            entropy_loss = []
            
            for log_prob, value, R, advantage, entropy in zip(
                self.saved_log_probs, 
                self.saved_state_values,
                returns,
                advantages,
                self.saved_entropy
            ):
                policy_loss.append(-log_prob * advantage.detach())
                # Fix: Reshape value and target to match dimensions
                value_loss.append(nn.MSELoss()(value.squeeze(), torch.tensor([R]).to(self.device)))
                entropy_loss.append(-entropy)
            
            loss = (
                torch.stack(policy_loss).mean() + 
                self.value_loss_coef * torch.stack(value_loss).mean() +
                self.entropy_coef * torch.stack(entropy_loss).mean()
            )
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            self.saved_log_probs = []
            self.saved_state_values = []
            self.saved_entropy = []
            self.saved_rewards = []
    
    def reset(self):
        self.saved_log_probs = []
        self.saved_state_values = []
        self.saved_entropy = []
        self.saved_rewards = []
    
    def save_model(self, path):
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load_model(self, path):
        checkpoint = torch.load(path)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])