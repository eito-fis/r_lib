import os
from collections import deque

import gym
import numpy as np
import tensorflow as tf

from src.sac.sac_model import SACQNet, SACActor
from src.sac.sac_policy import SACPolicy
from src.general.policies.policy import RandomPolicy
from src.general.replay_buffers.replay_buffer import ReplayBuffer

"""
Runs the model
"""
env = gym.make("LunarLanderContinuous-v2")
obs = env.reset()
env.render()
action_space = env.action_space
actor = SACActor(state_size=env.observation_space,
                  stack_size=1,
                  action_space=env.action_space,
                  fc=(512,256),
                  conv_size=None)
actor.load_weights("data/actor_model_95000.h5")
policy = SACPolicy(action_space=env.action_space,
                        batch_size=1,
                        model=actor)

reward = 0
for i in range(100000000):
    action, _ = policy(obs[None, :])
    action = tf.squeeze(action)
    action = action * np.abs(action_space.low)
    assert action.shape == action_space.shape

    # Take step on env with action
    new_obs, rewards, done, infos = env.step(action)
    reward += rewards
    env.render()
    # env.render()
    # Store SARS(D) in replay buffer
    obs = new_obs

    if done:
        print(f"Reward: {reward}")
        reward = 0
        input()
        obs = env.reset()

