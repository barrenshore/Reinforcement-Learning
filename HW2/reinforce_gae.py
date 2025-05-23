# Spring 2024, 535514 Reinforcement Learning
# HW2: REINFORCE and baseline

import os
import gym
from itertools import count
from collections import namedtuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.optim.lr_scheduler as Scheduler
from torch.utils.tensorboard import SummaryWriter

# Define a useful tuple (optional)
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

# Define a tensorboard writer
writer = SummaryWriter("./tb_record_1/REINFORCE_with_GAE")
        
env = gym.make("LunarLander-v2")
random_seed = 10 
env.seed(random_seed)

class Policy(nn.Module):
    """
        Implement both policy network and the value network in one model
        - Note that here we let the actor and value networks share the first layer
        - Feel free to change the architecture (e.g. number of hidden layers and the width of each hidden layer) as you like
        - Feel free to add any member variables/functions whenever needed
        TODO:
            1. Initialize the network (including the GAE parameters, shared layer(s), the action layer(s), and the value layer(s))
            2. Random weight initialization of each layer
    """
    def __init__(self):
        super(Policy, self).__init__()
        
        # Extract the dimensionality of state and action spaces
        self.discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.observation_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n if self.discrete else env.action_space.shape[0]
        self.hidden_size = 128
        self.double()
        
        ########## YOUR CODE HERE (5~10 lines) ##########
        self.shared_layer1 = nn.Linear(self.observation_dim, self.hidden_size)
        self.shared_layer2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.action_layer = nn.Linear(self.hidden_size, self.action_dim)
        self.value_layer = nn.Linear(self.hidden_size, 1)
        ########## END OF YOUR CODE ##########
        
        # action & reward memory
        self.saved_actions = []
        self.rewards = []

    def forward(self, state):
        """
            Forward pass of both policy and value networks
            - The input is the state, and the outputs are the corresponding 
              action probability distirbution and the state value
            TODO:
                1. Implement the forward pass for both the action and the state value
        """
        
        ########## YOUR CODE HERE (3~5 lines) ##########
        x = self.shared_layer1(state)
        x = F.relu(x)
        x = self.shared_layer2(x)
        x = F.relu(x)
        action_prob = self.action_layer(x)
        state_value = self.value_layer(x)
        ########## END OF YOUR CODE ##########

        return action_prob, state_value


    def select_action(self, state):
        """
            Select the action given the current state
            - The input is the state, and the output is the action to apply 
            (based on the learned stochastic policy)
            TODO:
                1. Implement the forward pass for both the action and the state value
        """
        
        ########## YOUR CODE HERE (3~5 lines) ##########
        state = torch.Tensor(state)
        action, state_value= self.forward(state)
        m = Categorical(logits=action)
        action = m.sample()
        ########## END OF YOUR CODE ##########
        
        # save to action buffer
        self.saved_actions.append(SavedAction(m.log_prob(action), state_value))

        return action.item()


    def calculate_loss(self, gamma=0.999, lambda_ = 0.99):
        """
            Calculate the loss (= policy loss + value loss) to perform backprop later
            TODO:
                1. Calculate rewards-to-go required by REINFORCE with the help of self.rewards
                2. Calculate the policy loss using the policy gradient
                3. Calculate the value loss using either MSE loss or smooth L1 loss
        """
        
        # Initialize the lists and variables
        R = 0
        saved_actions = self.saved_actions
        policy_losses = [] 
        value_losses = [] 
        returns = []

        ########## YOUR CODE HERE (8-15 lines) ##########
        discounted_sum = 0
        for reward in reversed(self.rewards):
            discounted_sum = reward + gamma * discounted_sum
            returns.append(discounted_sum)
        returns.reverse()
        returns = torch.Tensor(returns)
        returns = (returns - returns.mean()) / (returns.std())
        returns = returns.detach()

        log_probs = [action.log_prob for action in saved_actions]
        values = [action.value for action  in saved_actions]

        advantages = GAE(gamma, lambda_, None)(self.rewards, values)
        advantages = advantages.detach()

        action_log_probs = torch.stack(log_probs, dim=0)
        values = torch.stack(values, dim=0)[:,0]

        policy_losses = -(advantages * action_log_probs).sum()
        value_losses = F.mse_loss(values, returns).sum()
        loss = policy_losses + value_losses
        ########## END OF YOUR CODE ##########
        
        return loss

    def clear_memory(self):
        # reset rewards and action buffer
        del self.rewards[:]
        del self.saved_actions[:]

class GAE:
    def __init__(self, gamma, lambda_, num_steps):
        self.gamma = gamma
        self.lambda_ = lambda_
        self.num_steps = num_steps          # set num_steps = None to adapt full batch

    def __call__(self, rewards, values, done=False):
        """
        Implement Generalized Advantage Estimation (GAE) for your value prediction
        TODO (1): Pass correct corresponding inputs (rewards, values, and done) into the function arguments
        TODO (2): Calculate the Generalized Advantage Estimation and return the obtained value
        """
        ########## YOUR CODE HERE (8-15 lines) ##########
        advantages = []
        advantage = 0
        next_value = 0
        t = 0
        for r, v in zip(reversed(rewards), reversed(values)):
            t += 1
            td_error = r + next_value * self.gamma - v
            advantage = td_error + advantage * self.gamma * self.lambda_
            next_value = v
            advantages.insert(0, advantage)
            if self.num_steps is not None and t > self.num_steps:
                break
        advantages = torch.Tensor(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std())
        return advantages
        ########## END OF YOUR CODE ##########

def train(lr=0.01, lr_decay=0.9, gamma=0.999, lambda_ = 0.999):
    """
        Train the model using SGD (via backpropagation)
        TODO (1): In each episode, 
        1. run the policy till the end of the episode and keep the sampled trajectory
        2. update both the policy and the value network at the end of episode

        TODO (2): In each episode, 
        1. record all the value you aim to visualize on tensorboard (lr, reward, length, ...)
    """
    
    # Instantiate the policy model and the optimizer
    model = Policy()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Learning rate scheduler (optional)
    scheduler = Scheduler.StepLR(optimizer, step_size=100, gamma=lr_decay)
    
    # EWMA reward for tracking the learning progress
    ewma_reward = 0
    
    # run inifinitely many episodes
    for i_episode in count(1):
        # reset environment and episode reward
        state = env.reset()
        ep_reward = 0
        t = 0

        # Uncomment the following line to use learning rate scheduler
        # scheduler.step()
        
        # For each episode, only run 9999 steps to avoid entering infinite loop during the learning process
        
        ########## YOUR CODE HERE (10-15 lines) ##########
        while True:
            t += 1
            action = model.select_action(state)
            state, reward, done, _ = env.step(action)
            ep_reward += reward
            model.rewards.append(reward)
            if done:
                break
        
        optimizer.zero_grad()
        loss = model.calculate_loss(gamma, lambda_)
        loss_ = loss.detach()
        loss.backward()
        optimizer.step()
        scheduler.step()
        model.clear_memory()
        ########## END OF YOUR CODE ##########
            
        # update EWMA reward and log the results
        ewma_reward = 0.05 * ep_reward + (1 - 0.05) * ewma_reward
        print('Episode {}\tlength: {}\treward: {}\tloss: {}\t ewma reward: {}'.format(i_episode, t, ep_reward, loss_, ewma_reward))

        #Try to use Tensorboard to record the behavior of your implementation 
        ########## YOUR CODE HERE (4-5 lines) ##########
        writer.add_scalar('training loss', loss, i_episode)
        writer.add_scalar('EWMA reward', ewma_reward, i_episode)
        writer.add_scalar('Episode rewad', ep_reward, i_episode)
        writer.add_scalar('Length', t, i_episode)
        writer.add_scalar('Learning Rate', scheduler.get_lr()[-1], i_episode)
        ########## END OF YOUR CODE ##########

        # check if we have "solved" the cart pole problem, use 120 as the threshold in LunarLander-v2
        if ewma_reward > 120:
            if not os.path.isdir("./preTrained"):
                os.mkdir("./preTrained")
            torch.save(model.state_dict(), './preTrained/GAE_LunarLander-v2_{}.pth'.format(lr))
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(ewma_reward, t))
            #break
            return i_episode*(-ewma_reward)
    return i_episode*(-ewma_reward)


def test(name, n_episodes=10):
    """
        Test the learned model (no change needed)
    """     
    model = Policy()
    
    model.load_state_dict(torch.load('./preTrained/{}'.format(name)))
    
    render = True
    max_episode_len = 10000
    R = 0
    
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        running_reward = 0
        for t in range(max_episode_len+1):
            action = model.select_action(state)
            state, reward, done, _ = env.step(action)
            running_reward += reward
            if render:
                 env.render()
            if done:
                break
        R += running_reward
        print('Episode {}\tReward: {}'.format(i_episode, running_reward))
    print('Mean Reward:{}'.format(R/n_episodes))
    env.close()


if __name__ == '__main__':
    # For reproducibility, fix the random seed
    random_seed = 10  
    lr = 0.0047
    lr_decay = 0.94
    gamma = 0.999
    lambda_ = 0.63
    env = gym.make("LunarLander-v2")
    env.seed(random_seed)
    torch.manual_seed(random_seed)  
    train(lr, lr_decay, gamma, lambda_)
    test(f'LunarLander-v2_{lr}.pth')