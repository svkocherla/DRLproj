import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from agents.base_agent import BaseAgent

class ActorCriticNet(nn.Module):
    def __init__(self, input_dim, n_actions):
        
        super(ActorCriticNet, self).__init__()
        
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        self.actor = nn.Sequential(
            nn.Linear(128, n_actions),
            nn.Softmax(dim=-1)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(128, 1)
        )
        
    def forward(self, x):
        shared_out = self.shared(x)
        return self.actor(shared_out), self.critic(shared_out)

class ACAgent(BaseAgent):
    def __init__(self, action_space, observation_space, **kwargs):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_space = action_space
        
        self.network = ActorCriticNet(
            observation_space.shape[0], 
            action_space.n
        ).to(self.device)
        self.gamma = kwargs.get('gamma', 0.99)
        self.lr = kwargs.get('lr', 0.001)
        
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)
        self.reset()
        
    def reset(self):
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.entropy = []
        
    def _preprocess_state(self, state):
        state = state.copy()
        state[0:2] /= 210
        state[2:4] /= 210
        state[4:6] /= 4.0
        return state
        
    def act(self, observation, greedy=False):
        state = self._preprocess_state(observation)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        probs, value = self.network(state)
        self.values.append(value)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        self.log_probs.append(dist.log_prob(action))
        self.entropy.append(dist.entropy())
        
        return action.item()
    
    def _compute_returns(self, rewards, done):
        returns = []
        R = 0 if done else self.values[-1].detach()
        
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
            
        returns = torch.FloatTensor(returns).to(self.device)
        return returns
    
    def learn(self, reward, next_observation, done):
        self.rewards.append(reward)
        
        if done:
            returns = self._compute_returns(self.rewards, done)
            values = torch.cat(self.values).squeeze()
        
            advantages = returns - values.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
            policy_loss = 0
            for log_prob, advantage in zip(self.log_probs, advantages):
                policy_loss -= log_prob * advantage
                
            value_loss = F.mse_loss(values, returns)
        
            entropy_loss = -torch.stack(self.entropy).mean()
            
            total_loss = (policy_loss.mean() + 
                         0.5 * value_loss + 
                         0.01 * entropy_loss)
            
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
            self.optimizer.step()
            
            self.reset()
    
    def save_model(self, path):
        torch.save(self.network.state_dict(), path)
    
    def load_model(self, path):
        self.network.load_state_dict(torch.load(path))
