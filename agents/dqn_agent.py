import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from agents.base_agent import BaseAgent
from agents.memory import ReplayMemory, Transition

class DQNNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent(BaseAgent):
    def __init__(self, action_space, observation_space, 
                 learning_rate=1e-4,
                 gamma=0.99,
                 epsilon_start=1.0,
                 epsilon_final=0.01,
                 epsilon_decay=10000,
                 memory_size=10000,
                 batch_size=64,
                 target_update=10):
        super().__init__(action_space, observation_space)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_actions = action_space.n
        self.obs_dim = observation_space.shape[0]
        
        # Networks
        self.policy_net = DQNNetwork(self.obs_dim, self.num_actions).to(self.device)
        self.target_net = DQNNetwork(self.obs_dim, self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Training parameters
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = ReplayMemory(memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        
        # Exploration parameters
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0
        
        # Target network update frequency
        self.target_update = target_update
        self.update_count = 0

    def get_epsilon(self):
        epsilon = self.epsilon_final + (self.epsilon_start - self.epsilon_final) * \
                 np.exp(-1. * self.steps_done / self.epsilon_decay)
        return epsilon

    def act(self, observation, greedy=False):
        epsilon = 0.01 if greedy else self.get_epsilon()
        
        if random.random() > epsilon:
            with torch.no_grad():
                state = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state)
                return q_values.max(1)[1].item()
        else:
            return random.randrange(self.num_actions)

    def learn(self, reward, next_observation, done):
        # Store transition
        self.memory.push(self.last_observation, self.last_action, 
                        reward, next_observation, done)
        
        # Update last observation
        self.last_observation = next_observation
        
        # Increment steps
        self.steps_done += 1
        
        # Check if enough samples in memory
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch and convert to tensors
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        
        state_batch = torch.FloatTensor(batch.observation).to(self.device)
        action_batch = torch.LongTensor(batch.action).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).to(self.device)
        next_state_batch = torch.FloatTensor(batch.next_observation).to(self.device)
        done_batch = torch.FloatTensor(batch.done).to(self.device)
        
        # Compute Q(s_t, a)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        
        # Compute max Q(s_{t+1}, a) for all next states
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values = self.target_net(next_state_batch).max(1)[0]
        
        # Compute expected Q values
        expected_state_action_values = (next_state_values * (1 - done_batch) * self.gamma) + reward_batch
        
        # Compute loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        
        # Update target network
        self.update_count += 1
        if self.update_count % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def reset(self):
        self.last_observation = None
        self.last_action = None