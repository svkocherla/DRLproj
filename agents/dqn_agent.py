import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from agents.base_agent import BaseAgent
from agents.memory import ReplayMemory, Transition

class DQNNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQNNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)

class DQNAgent(BaseAgent):
    def __init__(self, action_space, observation_space, 
                 learning_rate=3e-4,
                 gamma=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.01,
                 epsilon_decay=0.9995,
                 memory_size=100000,
                 batch_size=128,
                 target_update=100):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_space = action_space
        self.observation_space = observation_space
        
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        
        input_dim = observation_space.shape[0]
        output_dim = action_space.n
        self.policy_net = DQNNetwork(input_dim, output_dim).to(self.device)
        self.target_net = DQNNetwork(input_dim, output_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = ReplayMemory(memory_size)
        
        self.steps_done = 0
        self.current_observation = None
        self.current_action = None
        
    def act(self, observation, greedy=False):
        self.current_observation = observation
        
        if not greedy and random.random() < self.epsilon:
            action = self.action_space.sample()
        else:
            with torch.no_grad():
                state = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state)
                action = q_values.max(1)[1].item()
        
        self.current_action = action
        return action
    
    def learn(self, reward, next_observation, done):
        # Store transition in memory
        if self.current_observation is not None:
            self.memory.push(
                self.current_observation,
                self.current_action,
                reward,
                next_observation,
                done
            )
        
        if len(self.memory) < self.batch_size:
            return
        
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        
        state_batch = torch.FloatTensor(np.array(batch.observation)).to(self.device)
        action_batch = torch.LongTensor(batch.action).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(batch.next_observation)).to(self.device)
        done_batch = torch.FloatTensor(batch.done).to(self.device)
        
        state_action_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        next_state_values = self.target_net(next_state_batch).max(1)[0].detach()
        expected_state_action_values = reward_batch + (1 - done_batch) * self.gamma * next_state_values
        
        loss = nn.MSELoss()(state_action_values, expected_state_action_values.unsqueeze(1))
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1)
        self.optimizer.step()
        
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.steps_done += 1
    
    def reset(self):
        self.current_observation = None
        self.current_action = None
    
    def save_model(self, path):
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load_model(self, path):
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])