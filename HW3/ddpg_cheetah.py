# Spring 2024, 535514 Reinforcement Learning
# HW3: DDPG

import sys
import gym
import numpy as np
import os
import time
import random
from collections import namedtuple
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import logging
from skopt.space import Real, Integer
from skopt import gp_minimize
logging.basicConfig(filename='train.log', level=logging.DEBUG)

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

Transition = namedtuple(
    'Transition', ('state', 'action', 'mask', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class OUNoise:

    def __init__(self, action_dimension, scale=0.1, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state * self.scale    

class Actor(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space):
        super(Actor, self).__init__()
        self.action_space = action_space
        num_outputs = action_space.shape[0]

        ########## YOUR CODE HERE (5~10 lines) ##########
        # Construct your own actor network
        
        self.layer1 = nn.Linear(num_inputs, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.out_layer = nn.Linear(hidden_size, num_outputs)
        
        ########## END OF YOUR CODE ##########
        
    def forward(self, inputs):
        
        ########## YOUR CODE HERE (5~10 lines) ##########
        # Define the forward pass your actor network
        
        x = self.layer1(inputs)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        x = self.layer3(x)
        x = F.relu(x)
        action = torch.sigmoid(self.out_layer(x))
        return action * 2.0 - 1.0
        
        ########## END OF YOUR CODE ##########

class Critic(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space):
        super(Critic, self).__init__()
        self.action_space = action_space
        num_outputs = action_space.shape[0]

        ########## YOUR CODE HERE (5~10 lines) ##########
        # Construct your own critic network
        
        self.layer1 = nn.Linear(num_inputs+num_outputs, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.out_layer = nn.Linear(hidden_size, 1)
        
        ########## END OF YOUR CODE ##########

    def forward(self, inputs, actions):
        
        ########## YOUR CODE HERE (5~10 lines) ##########
        # Define the forward pass your critic network
        
        x = self.layer1(torch.cat([inputs, actions], dim=-1))
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        x = self.layer3(x)
        x = F.relu(x)
        v = self.out_layer(x)
        return v
    
        ########## END OF YOUR CODE ##########        
        

class DDPG(object):
    def __init__(self, num_inputs, action_space, gamma=0.995, tau=0.0005, hidden_size=128, lr_a=1e-4, lr_c=1e-3):

        self.num_inputs = num_inputs
        self.action_space = action_space

        self.actor = Actor(hidden_size, self.num_inputs, self.action_space)
        self.actor_target = Actor(hidden_size, self.num_inputs, self.action_space)
        self.actor_perturbed = Actor(hidden_size, self.num_inputs, self.action_space)
        self.actor_optim = Adam(self.actor.parameters(), lr=lr_a)

        self.critic = Critic(hidden_size, self.num_inputs, self.action_space)
        self.critic_target = Critic(hidden_size, self.num_inputs, self.action_space)
        self.critic_optim = Adam(self.critic.parameters(), lr=lr_c)

        self.gamma = gamma
        self.tau = tau

        hard_update(self.actor_target, self.actor) 
        hard_update(self.critic_target, self.critic)


    def select_action(self, state, action_noise=None):
        self.actor.eval()
        mu = self.actor((Variable(state)))
        mu = mu.data

        ########## YOUR CODE HERE (3~5 lines) ##########
        # Add noise to your action for exploration
        # Clipping might be needed 
        
        noise = [0.0] if action_noise is None else action_noise.noise()
        noise = torch.FloatTensor(noise)
        return torch.clamp(mu + noise, min=-1.0, max=1.0)
    
        ########## END OF YOUR CODE ##########


    def update_parameters(self, batch):
        state_batch = Variable(torch.cat([b.state for b in batch]))
        action_batch = Variable(torch.cat([b.action for b in batch]))
        reward_batch = Variable(torch.cat([b.reward for b in batch]))
        mask_batch = Variable(torch.cat([b.mask for b in batch]))
        next_state_batch = Variable(torch.cat([b.next_state for b in batch]))
        
        ########## YOUR CODE HERE (10~20 lines) ##########
        # Calculate policy loss and value loss
        # Update the actor and the critic
        
        Q = self.critic(state_batch, action_batch)
        next_action = self.actor_target(next_state_batch)
        next_Q = self.critic_target(next_state_batch, next_action)
        
        # Q_target = r+gamma*Q by TD(0)
        Q_target = reward_batch.view(-1,1) + self.gamma * next_Q * (1-mask_batch.view(-1,1))
        value_loss = F.mse_loss(Q, Q_target)
        
        # Update critic
        self.critic_optim.zero_grad()
        value_loss.backward()
        self.critic_optim.step()
        
        #update actor
        policy_loss = -(self.critic(state_batch, self.actor(state_batch))).mean()
        self.actor_optim.zero_grad()
        policy_loss.backward()
        self.actor_optim.step()
        
        ########## END OF YOUR CODE ########## 

        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        return value_loss.item(), policy_loss.item()


    def save_model(self, env_name, suffix="", actor_path=None, critic_path=None):
        local_time = time.localtime()
        timestamp = time.strftime("%m%d%Y_%H%M%S", local_time)
        if not os.path.exists('preTrained/'):
            os.makedirs('preTrained/')

        if actor_path is None:
            actor_path = "preTrained/ddpg_actor_{}_{}_{}".format(env_name, timestamp, suffix) 
        if critic_path is None:
            critic_path = "preTrained/ddpg_critic_{}_{}_{}".format(env_name, timestamp, suffix) 
        print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    def load_model(self, actor_path, critic_path):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.actor.load_state_dict(torch.load(actor_path))
        if critic_path is not None: 
            self.critic.load_state_dict(torch.load(critic_path))

def train(lr_a=1e-4, lr_c=1e-3, batch_size=128):    
    
    # Define a tensorboard writer
    writer = SummaryWriter("./tb_record_3/HalfCheetah/train-{}-{}".format(lr_a, lr_c))
    
    logging.info('lr_a = {}, lr_c = {}, batch_size = {}'.format(lr_a, lr_c, batch_size))
    
    random_seed = 10  
    env_name = 'HalfCheetah-v2'
    env = gym.make(env_name)
    env.seed(random_seed)  
    torch.manual_seed(random_seed)  
    
    num_episodes = 1000
    gamma = 0.995
    tau = 0.002
    hidden_size = 128
    noise_scale = 0.3
    replay_size = 100000
    updates_per_step = 1
    print_freq = 50
    ewma_reward = 0
    rewards = []
    ewma_reward_history = []
    total_numsteps = 0
    updates = 0

    
    agent = DDPG(env.observation_space.shape[0], env.action_space, gamma, tau, hidden_size, lr_a, lr_c)
    ounoise = OUNoise(env.action_space.shape[0])
    memory = ReplayMemory(replay_size)
    
    for i_episode in range(num_episodes):
        
        ounoise.scale = noise_scale
        ounoise.reset()
        
        state = torch.Tensor([env.reset()])

        episode_reward = 0
        
        value_loss = []
        policy_loss = []
        
        while True:
            
            ########## YOUR CODE HERE (15~25 lines) ##########
            # 1. Interact with the env to get new (s,a,r,s') samples 
            # 2. Push the sample to the replay buffer
            # 3. Update the actor and the critic
            
            total_numsteps+=1
            
            action = agent.select_action(state, ounoise)
            next_state, reward, done, _ = env.step(action.numpy()[0])
            next_state = torch.Tensor([next_state])
            memory.push(state, action, torch.Tensor([done]), next_state, torch.Tensor([reward]))
            
            if len(memory) >= batch_size and total_numsteps%updates_per_step == 0:
                batch = memory.sample(batch_size)
                v_loss, p_loss = agent.update_parameters(batch)
                value_loss.append(v_loss)
                policy_loss.append(p_loss)
            
            episode_reward += reward
            state = next_state
            if done:
                break

            ########## END OF YOUR CODE ########## 
            

        rewards.append(episode_reward)
        actor_loss = np.mean(policy_loss)
        critic_loss = np.mean(value_loss)
        t = 0
        
        state = torch.Tensor([env.reset()])
        episode_reward = 0
        while True:
            action = agent.select_action(state)

            next_state, reward, done, _ = env.step(action.numpy()[0])
            
            env.render()
            
            episode_reward += reward

            next_state = torch.Tensor([next_state])

            state = next_state
            
            t += 1
            if done:
                break
            
        rewards.append(episode_reward)
        
        # update EWMA reward and log the results
        ewma_reward = 0.05 * episode_reward + (1 - 0.05) * ewma_reward
        ewma_reward_history.append(ewma_reward)           
        if i_episode % print_freq == 0:
            print("Episode: {}, length: {}, reward: {:.2f}, ewma reward: {:.2f}, value loss: {:.2f}, policy loss: {:.2f}".format(i_episode, t, rewards[-1], ewma_reward, critic_loss, actor_loss))
            logging.info("Episode: {}, length: {}, reward: {:.2f}, ewma reward: {:.2f}, value loss: {:.2f}, policy loss: {:.2f}".format(i_episode, t, rewards[-1], ewma_reward, critic_loss, actor_loss))

        # records
        writer.add_scalar('Reward', episode_reward, i_episode)
        writer.add_scalar('EWMA Reward', ewma_reward, i_episode)
        writer.add_scalar('Critic loss', critic_loss, i_episode)
        writer.add_scalar('Actor loss', actor_loss, i_episode)
        
        if ewma_reward >= 1000:
            agent.save_model(env_name, '.pth')
            logging.info("Running reward is now {} and the total episode is {}.".format(ewma_reward, i_episode))
            return (ewma_reward+500)/(i_episode+1) # For tuning
        
    agent.save_model(env_name, '.pth')        
    logging.info("Running reward is now {} and the total episode is {}.".format(ewma_reward, i_episode))
    return (ewma_reward+500)/(i_episode+1) # For tuning
 
def hp_tune():
    search_space = [
        Real(0, 0.01, name='lr_a'),
        Real(0, 0.01, name='lr_c'),
        Integer(64, 256, name='batch_size')
    ]

    def objective(params):
        episodes_num = -train(params[0], params[1], params[2])
        return episodes_num

    result = gp_minimize(objective, search_space, n_calls=10, random_state=10)

    print("Best hyperparameters: ", result.x)
    print("Best objective value: ", result.fun)
    print("Hyperparameters tried: ", result.x_iters)
    print("Objective values at each step: ", result.func_vals)


if __name__ == '__main__':
    hp_tune()


