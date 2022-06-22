"""
Implementation of PPO using PyTorch
Works with gym style environments
For an example of how to use see main below
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions.normal import Normal


def construct_mlp(observation_size, hidden_layers, action_size):
    layers = []
    layer_sizes = [observation_size] + hidden_layers + [action_size]
    for i in range(len(layer_sizes)-1):
        layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(layer_sizes[-1], action_size))

    return nn.Sequential(*layers)


class Actor(nn.Module):

    def __init__(self, observation_size, hidden_layers, action_size, lr):
        super().__init__()
        self.pi_net = construct_mlp(observation_size, hidden_layers, action_size)
        self.optimizer = Adam(self.pi_net.parameters(), lr=lr)
        self.log_std = -0.5*torch.ones(action_size)
        # TODO: move to gpu

    def forward(self, observation):
        with torch.no_grad():
            return self.pi_net(observation)


class Critic(nn.Module):

    def __init__(self, observation_size, hidden_layers, lr):
        super().__init__()
        self.v_net = construct_mlp(observation_size, hidden_layers, 1)
        self.optimizer = Adam(self.v_net.parameters(), lr=lr)
        # TODO: move to gpu

    def forward(self, observation):
        return self.v_net(observation)


class ActorCritic():

    def __init__(self, observation_size, action_size, actor_hidden, critic_hidden, actor_lr, critic_lr):
        self.pi_actor = Actor(observation_size, actor_hidden, action_size, actor_lr)
        self.v_critic = Critic(observation_size, critic_hidden, critic_lr)

    def forward(self, observation):
        observation = torch.as_tensor(observation, dtype=torch.float32)
        return self.pi_actor.forward(observation), self.v_critic.forward(observation)

    def forward_actor_only(self, observation):
        return self.pi_actor.forward(torch.as_tensor(observation, dtype=torch.float32))

    def return_actor(self):
        return self.pi_actor


class DataBuffer():

    def __init__(self):
        self.observations = []
        self.actions = []
        self.values = []
        self.rewards = []

    def store(self, observation, action, value, reward):
        self.observations.append(observation)
        self.actions.append(action)
        self.values.append(value)
        self.rewards.append(reward)


class PPO():
    """
    The main class that should be constructed externally
    """

    def __init__(self, env, actor_hidden=[32,32], critic_hidden=[32,32], actor_lr=0.0003, critic_lr=0.001):
        self.env = env
        observation_size = env.observation_space.shape[0]
        action_size = env.action_space.shape[0]
        self.actor_critic = ActorCritic(observation_size, action_size, actor_hidden, critic_hidden, actor_lr, critic_lr)
        self.data_buffer = DataBuffer()

    def train(self, epochs = 5, steps_per_epoch = 200):
        for _ in range(epochs):
            self._train_one_epoch(steps_per_epoch)

    def _train_one_epoch(self, steps_per_epoch):
        self._collect_trajectories(steps_per_epoch)
        self._update_model()

    def _collect_trajectories(self, steps_per_epoch):
        observation = self.env.reset()
        for _ in range(steps_per_epoch):
            action, value = self.actor_critic.forward(observation)
            new_observation, reward, done, _ = self.env.step(action)
            self.data_buffer.store(observation, action, value, reward)

            observation = new_observation
            if done is True:
                observation = self.env.reset()
                done = False

    def _update_model(self):
        # TODO: update model
        pass

    def run_and_render(self, runs=3, reward_floor=-8):
        for _ in range(runs):
            observation = env.reset()
            env.render()
            done = False
            cumulative_reward = 0
            while not done:
                action = self.actor_critic.forward_actor_only(observation)
                observation, reward, done, _ = env.step(action)
                cumulative_reward += reward
                env.render()
                if cumulative_reward < reward_floor:
                    break
            env.close()

    def save_model(self):
        raise NotImplementedError


"""
Example showing how to use this PPO module below
"""
if __name__ == "__main__":
    import gym
    # from ppo import PPO

    env = gym.make("BipedalWalker-v3")

    PPO = PPO(env)
    PPO.train()
    PPO.run_and_render()

    # policy = PPO.return_actor()
