import gym

class GymEnv():
    def __init__(self, gym_env):
        self.env = gym.make(gym_env)
        self.obs_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.stack_size = 0
        self.ep_reward = 0
        self.reward = 0

    def step(self, actions):
        obs, reward, done, info = self.env.step(actions)
        self.reward += reward
        if done:
            self.ep_reward = self.reward
            self.reward = 0
        return obs, reward, done, info

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def reset(self):
        obs = self.env.reset()
        return obs
