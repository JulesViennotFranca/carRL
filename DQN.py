import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class Network(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()

        self.layers = nn.ModuleList()
        layer_sizes = [input_size] + hidden_sizes + [output_size]

        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = nn.functional.relu(self.layers[i](x))

        return self.layers[-1](x) 

class ReplayBuffer():
    def __init__(self, batch_size, capacity):
        self.batch_size = batch_size
        self.memory = deque(maxlen=capacity)

    def add(self, exp):
        self.memory.append(exp)

    def sample(self):
        minibatch = random.sample(self.memory, self.batch_size)

        obs_batch = torch.stack([s1 for (s1,_,_,_,_) in minibatch])
        action_batch = torch.tensor([a for (_,a,_,_,_) in minibatch])
        reward_batch = torch.tensor([r for (_,_,r,_,_) in minibatch])
        next_obs_batch = torch.stack([s2 for (_,_,_,s2,_) in minibatch])
        done_batch = torch.tensor([d for (_,_,_,_,d) in minibatch])

        return obs_batch, action_batch, reward_batch, next_obs_batch, done_batch

    def size(self):
        return len(self.memory)
    
class DQNAgent():
    def __init__(self, obs_dim, hidden_sizes, act_dim, lr, batch_size, capacity, gamma, eps_start, eps_end, eps_decay):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay 
        self.eps = self.eps_start

        self.replay = ReplayBuffer(self.batch_size, capacity)

        self.q = Network(self.obs_dim, hidden_sizes, self.act_dim)
        self.target_q = Network(self.obs_dim, hidden_sizes, self.act_dim)
        self.target_q.load_state_dict(self.q.state_dict())
        self.optimizer = optim.Adam(self.q.parameters(), lr=lr)

    def act(self, obs):
        if random.random() < self.eps:
            action = np.random.randint(self.act_dim)
            # print("random:", action)
        else:
            with torch.no_grad():
                obs = torch.tensor(obs, dtype=torch.float32)
                obs = obs.unsqueeze(0)
                q_values = self.q(obs)
                action = torch.argmax(q_values).item()
        return action

    def learn(self):
        if self.replay.size() < self.batch_size:
            return 
        
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = self.replay.sample()
        
        q_value = self.q(obs_batch).squeeze()[range(self.batch_size), action_batch]
        with torch.no_grad():
            q_target = self.target_q(next_obs_batch).squeeze().max(dim=1)[0]
            q_target = reward_batch + (1 - done_batch.long()) * self.gamma * q_target

        loss = nn.functional.mse_loss(q_value, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_target_q()
        self.update_eps()

    def update_target_q(self):
        self.target_q.load_state_dict(self.q.state_dict())

    def update_eps(self):
        self.eps *= self.eps_decay
        self.eps = max(self.eps_end, self.eps)

    def update_replay(self, exp):
        self.replay.add(exp)
