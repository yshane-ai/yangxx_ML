import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
from torch.distributions import Normal
import numpy as np
import argparse
import matplotlib.pyplot as plt

# 对于连续动作，PPO采用的是随机策略，动作基于正态分布进行采样。所以Actor网络的目的就是输出正态分布的$mu$和$sigma$。

class ActorNet(nn.Module):
    def __init__(self, n_states, bound):
        super().__init__()
        self.n_states = n_states
        self.bound = bound
        self.layer = nn.Sequential(
            nn.Linear(self.n_states, 128),
            nn.ReLU()
        )
        
        self.mu_out = nn.Linear(128, 1)
        self.sigma_out = nn.Linear(128, 1)
        
    def forward(self, x):
        x = F.relu(self.layer(x))
        mu = self.bound * F.tanh(self.mu_out(x))
        sigma = F.softplus(self.sigma_out(x))
        return mu, sigma
    
# Critic网络还是和之前的一样，用来计算$v$值。
    
class CriticNet(nn.Module):
    def __init__(self, n_states):
        super().__init__()
        self.n_states = n_states
        self.layer = nn.Sequential(
            nn.Linear(self.n_states, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, x):
        v = self.layer(x)
        return v
    
class PPO():
    def __init__(self, n_states, n_actions, bound, args):
        super().__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.bound = bound
        self.lr = args.lr
        self.gamma = args.gamma
        self.epsilon = args.epsilon
        self.a_update_steps = args.a_update_steps
        self.c_update_steps = args.c_update_steps
        
        self._build()
        
    def _build(self):
        # 这里需要注意的是，虽然在choose_action()方法里使用的是actor_model，但是在update()函数里，每次更新之前都会将actor_model的参数赋给actor_old_model，所以之前所采样的数据就相当于是用actor_old_model得到的。
        self.actor_model = ActorNet(self.n_states, self.bound)
        self.actor_old_model = ActorNet(self.n_states, self.bound)
        self.actor_optim = optim.Adam(self.actor_model.parameters(), lr = self.lr)
        
        self.critic_model = CriticNet(self.n_states)
        self.critic_optim = optim.Adam(self.critic_model.parameters(), lr = self.lr)
        
    def choose_action(self, s):
        s = torch.FloatTensor(s)
        mu, sigma = self.actor_model(s)
        dist = Normal(mu, sigma)
        action = dist.sample()
        return torch.clamp(action, -self.bound, self.bound).data.numpy()
    
    def discount_reward(self, rewards, s_):
        s_ = torch.FloatTensor(s_)
        target = self.critic_model(s_).detach()
        target_list = []
        for r in rewards[::-1]:
            target = r + self.gamma * target
            target_list.append(target)
        target_list.reverse()
        target_list = torch.cat(target_list)
        return target_list
    
    def actor_learn(self, states, actions, advantages):
        # 下面这段代码所实现的就是PPO-Clip的公式。
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions).reshape(-1, 1)
        
        mu, sigma = self.actor_model(states)
        dist = Normal(mu, sigma)
        
        old_mu, old_sigma = self.actor_old_model(states)
        old_dist = Normal(old_mu, old_sigma)
        ratio = torch.exp(dist.log_prob(actions) - old_dist.log_prob(actions))
        surr = ratio * advantages.reshape(-1, 1)
        loss = -torch.mean(torch.min(surr, torch.clamp(ratio, 1-self.bound, 1+self.bound) * advantages.reshape(-1, 1)))
        
        self.actor_optim.zero_grad()
        loss.backward()
        self.actor_optim.step()
        
    def critic_learn(self, states, targets):
        states = torch.FloatTensor(states)
        v = self.critic_model(states).reshape(1, -1).squeeze(0)
        
        loss_func = nn.MSELoss()
        loss = loss_func(v, targets)
        
        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()
        
    def cal_adv(self, states, targets):
        states = torch.FloatTensor(states)
        v = self.critic_model(states).reshape(1, -1).squeeze(0)
        adv = targets - v
        return adv.detach()
    
    def update(self, states, actions, targets):
        self.actor_old_model.load_state_dict(self.actor_model.state_dict())
        advantages = self.cal_adv(states, targets)
        
        for i in range(self.a_update_steps):
            self.actor_learn(states, actions, advantages)
        for i in range(self.c_update_steps):
            self.critic_learn(states, targets)
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_episodes', type=int, default=600)
    parser.add_argument('--len_episode', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--epsilon', type=float, default=0.2)
    parser.add_argument('--c_update_steps', type=int, default=10)
    parser.add_argument('--a_update_steps', type=int, default=10)
    args = parser.parse_args()

    env = gym.make('Pendulum-v1')
    env.seed(args.seed)
    torch.manual_seed(args.seed)

    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    bound = env.action_space.high[0]

    agent = PPO(n_states, n_actions, bound, args)

    all_ep_r = []
    for episode in range(args.n_episodes):
        ep_r = 0
        s = env.reset()
        states, actions, rewards = [], [], []
        for t in range(args.len_episode):
            a = agent.choose_action(s)
            s_, r, done, _ = env.step(a)
            ep_r += r
            states.append(s)
            actions.append(a)
            rewards.append((r + 8) / 8)       # 参考了网上的做法

            s = s_

            if (t + 1) % args.batch == 0 or t == args.len_episode - 1:   # N步更新
                states = np.array(states)
                actions = np.array(actions)
                rewards = np.array(rewards)

                targets = agent.discount_reward(rewards, s_)          # 奖励回溯
                agent.update(states, actions, targets)                # 进行actor和critic网络的更新
                states, actions, rewards = [], [], []

        print('Episode {:03d} | Reward:{:.03f}'.format(episode, ep_r))

        if episode == 0:
            all_ep_r.append(ep_r)
        else:
            all_ep_r.append(all_ep_r[-1] * 0.9 + ep_r * 0.1)          # 平滑

    plt.plot(np.arange(len(all_ep_r)), all_ep_r)
    plt.show()