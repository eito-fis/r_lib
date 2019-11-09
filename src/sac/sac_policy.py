import numpy as np
import tensorflow as tf

from src.general.policies.policy import Policy

# GLOBAL
# Prevent division by zero
EPS = 1e-6
# CAP the standard deviation of the actor
LOG_STD_MAX = 2
LOG_STD_MIN = -20

class SACPolicy(Policy):
    """
    SAC Policy

    Attributes:
        action_space: Action space of the policy, expected format depends on the
        action_space_type
        representining minimium and maximium values of that action.
        action_space_type: What type of action space the policy is operating in.
        Current possible values are "Discrete" and "Continuous"
            - Discrete should pass an int as action_space
            - Continuous should pass a list of tuples (min_value, max_value) as
            action_space
        batch_size: Number of actions to be generated at once
        sample_func: Function to use to determine action
        model: Model used for the policy
    """
    def __init__(self,
                 action_space=None,
                 action_space_type="Continuous",
                 batch_size=1,
                 model=None):
        super().__init__(action_space=action_space,
                         action_space_type=action_space_type,
                         batch_size=batch_size)
        self.model = model
        if action_space_type == "Discrete":
            self.sample_func = self.sample_discrete
        elif action_space_type=="Continuous":
            self.sample_func = self.sample_continuous


    def sample_discrete(self):
        """
        SAC Discrete coming soon near you...
        """
        raise NotImplementedError

    def sample_continuous(self, obs):
        """
        Samples actions using the actor network

        Returns:
            actions: List of length of obs, where each element is a list
            containing actions for all dimensions
        """
        mean, log_std = self.model(obs)
        log_std = tf.clip_by_value(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = tf.exp(log_std)

        action = mean + tf.random.normal(tf.shape(mean)) * std

        log_prob = self.gaussian_prob(mean, log_std, action)
        scaled_action, scaled_log_prob = self.squish(mean, action, log_prob)

        return scaled_action, scaled_log_prob

    def gaussian_prob(self, mean, log_std, action):
        """
        Get log probability of a value given a mean and log standard deviation
        fo a gaussian distribution
        """
        pre_sum = -0.5 * (((action - mean) / (tf.exp(log_std) + EPS)) ** 2 +
                          2 * log_std + np.log(2 * np.pi))
        log_prob = tf.reduce_sum(pre_sum, axis=1)
        return log_prob

    def squish(self, mean, action, log_prob):
        """
        Squish the action and scale the log probability accordingly
        """
        scaled_action = tf.tanh(action)
        log_prob -= tf.reduce_sum(tf.math.log(1 - scaled_action ** 2 + EPS), axis=1)
        return scaled_action, log_prob

    def __call__(self, obs):
        return self.sample_func(obs)
