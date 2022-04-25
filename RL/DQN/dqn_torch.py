import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym
import numpy as np
import random
from collections import namedtuple

LR= 0.01
GAMMA = 0.9
EPS = 0.9
BATCH_SIZE = 32
UPDATE_TARGET_INTERVAL = 100
CAPACITY = 2000

class Net(nn.Module):
    def __init__(self, obs_space, action_space, hidden_size = 50):
        super().__init__()
        self.fc1 = nn.Linear(obs_space, hidden_size)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(hidden_size, action_space)
        self.out.weight.data.normal_(0, 0.1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.out(x)

Transitions = namedtuple('transition', ('state', 'action', 'reward', 'next_state'))

class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.counter = 0
    
    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.counter] = Transitions(*args)
        self.counter = (self.counter + 1) % self.capacity
        
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
        
class DQN():
    def __init__(self, obs_space, action_space):
        self.action_space = action_space
        self.eval_net, self.target_net = Net(obs_space, action_space), Net(obs_space, action_space)
        self.target_net.load_state_dict(self.eval_net.state_dict())
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr = LR)
        self.loss_func = nn.MSELoss()
        self.memory = ReplayMemory(CAPACITY)
        self.learn_step_counter = 0
        self.memory_counter = 0
        
    def select_action(self, obs):
        # obs = torch.unsqueeze(obs, 0)
        if np.random.uniform() < EPS:
            action_value = self.eval_net.forward(obs)
            action = action_value.max(1)[1].view(1, 1)
        else:
            action = torch.tensor(np.random.randint(0, self.action_space)).view(1, 1)
        return action
    
    def store_transitions(self, *args):
        self.memory.push(*args)
        
    
    def learn(self):
        if self.learn_step_counter % UPDATE_TARGET_INTERVAL == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1
        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transitions(*zip(*transitions))
        batch_state = torch.cat(batch.state)
        batch_action = torch.cat(batch.action)
        batch_next_state = torch.cat(batch.next_state)
        batch_reward = torch.cat(batch.reward)
        
        q_eval = self.eval_net(batch_state).gather(1, batch_action)
        q_next = self.target_net(batch_next_state).detach()
        q_target = batch_reward.view(BATCH_SIZE, 1) + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        loss = self.loss_func(q_eval, q_target)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        

if __name__ == '__main__':
    env = gym.make('CartPole-v0').unwrapped
    state_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    
    dqn = DQN(state_space, action_space)
    
    for i_episode in range(2000):
        episode_reward = 0
        state = torch.tensor([env.reset()])
        # env.render()
        while True:
            action = dqn.select_action(state)
            next_state, reward, done, info = env.step(action=action.item())
            x, x_dot, theta, theta_dot = next_state
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            reward = torch.tensor((r1 + r2).astype(np.float32))
            episode_reward += reward.item()
            next_state = torch.tensor([next_state])
            reward = torch.tensor([reward]).view(1, 1)
            dqn.store_transitions(state, action, reward, next_state)
            state = next_state
            if len(dqn.memory) >= CAPACITY:
                dqn.learn()
                if done:
                    print("i_episode, episode_reward", i_episode, episode_reward)
                    break
            if done:
                break
env.close()