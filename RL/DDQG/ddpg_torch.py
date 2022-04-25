import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import matplotlib.pyplot as plt

class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, s):
        s = F.relu(self.linear1(s))
        s = F.relu(self.linear2(s))
        output = torch.tanh(self.linear3(s))
        return output
    
class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, s, a):
        x = torch.cat((s, a), 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        output = self.linear3(x)
        return output
    
class Agent():
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.s_dim = self.env.observation_space.shape[0]
        self.a_dim = self.env.action_space.shape[0]
        
        self.actor = Actor(self.s_dim, 256, self.a_dim)
        self.actor_target = Actor(self.s_dim, 256, self.a_dim)
        self.critic = Critic(self.s_dim + self.a_dim, 256, self.a_dim)
        self.critic_target = Critic(self.s_dim + self.a_dim, 256, self.a_dim)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr = self.actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr = self.critic_lr)
        self.buffer = []
        self.index = 0
        self.loss_func = nn.MSELoss()
        
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
    
    def act(self, s0):
        s0 = torch.FloatTensor(s0).unsqueeze(0)
        action = self.actor(s0).squeeze(0).detach().numpy()
        return action
    
    def put(self, *transition):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.index] = transition
        self.index = (self.index + 1) % self.capacity
    
    def learn(self):
        if len(self.buffer) < self.batch_size:
            return
        transitions = random.sample(self.buffer, self.batch_size)
        try:
            s, a, r, s_  = zip(*transitions)
        except TypeError:
            print('transitions', transitions)
        
        s = torch.FloatTensor(s).view(self.batch_size, -1)
        a = torch.FloatTensor(a).view(self.batch_size, -1)
        r = torch.FloatTensor(r).view(self.batch_size, -1)
        s_ = torch.FloatTensor(s_).view(self.batch_size, -1)
        
        def actor_learn():
            actions = self.actor(s)
            loss = -torch.mean(self.critic(s, actions))
            self.actor_optim.zero_grad()
            loss.backward()
            self.actor_optim.step()
            
        def critic_learn():
            a_ = self.actor_target(s_).detach()
            q_next = self.critic_target(s_, a_).detach()
            q_target = r + self.gamma * q_next
            q_eval = self.critic(s, a)
            loss = self.loss_func(q_eval, q_target)
            
            self.critic_optim.zero_grad()
            loss.backward()
            self.critic_optim.step()
        
        def soft_update(net_target, net, tau):
            for target_param, param in zip(net_target.parameters(), net.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        
        for _ in range(2):
            critic_learn()
            actor_learn()
            soft_update(self.critic_target, self.critic, tau = self.tau)
            soft_update(self.actor_target, self.actor, tau = self.tau)
        
        
if __name__ == '__main__':
    
    env = gym.make('Pendulum-v1')
    env.reset()
    # env.render()

    params = {
        'env': env,
        'gamma': 0.99,
        'actor_lr': 0.001,
        'critic_lr': 0.001,
        'tau': 0.02,
        'capacity': 10000,
        'batch_size': 32,
    }

    agent = Agent(**params)

    all_ep_r = []
    for episode in range(100):
        s0 = env.reset()
        episode_reward = 0

        for step in range(300):
            # env.render()
            a0 = agent.act(s0)
            s1, r1, done, _ = env.step(a0)
            agent.put(s0, a0, r1, s1)

            episode_reward += r1
            s0 = s1

            agent.learn()

        
        print(episode, ': ', episode_reward)
        if episode == 0:
            all_ep_r.append(episode_reward)
        else:
            all_ep_r.append(all_ep_r[-1] * 0.9 + episode_reward * 0.1)          # 平滑
            
    plt.plot(np.arange(len(all_ep_r)), all_ep_r)
    plt.savefig('001.png', format='png')
    plt.show()