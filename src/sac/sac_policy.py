import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
Normal = tfp.distributions.Normal


from src.general.policies.policy import Policy

# GLOBAL
# Prevent division by zero
EPS = 1e-6

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
            self.eval_func = self.eval_disc
            self.step_func = self.step_disc
        elif action_space_type=="Continuous":
            self.eval_func = self.eval_cont
            self.step_func = self.step_cont


    def eval_disc(self):
        """
        SAC Discrete coming to cloud engines near you...
        """
        raise NotImplementedError

    def step_disc(self):
        """
        SAC Discrete coming to cloud engines near you...
        """
        raise NotImplementedError

    def eval_cont(self, obs, flag):
        """
        Samples actions using the actor network

        Returns:
            actions: List of length of obs, where each element is a list
            containing actions for all dimensions
        """
        mean, log_std = self.model(obs)
        std = tf.exp(log_std)

        pre_squish_action = mean + std * Normal(0, 1).sample()
        squish_action = tf.math.tanh(pre_squish_action)
        action = squish_action * self.action_range

        log_prob = Normal(mean, std).log_prob(pre_squish_action) - \
                    tf.math.log(1. - squish_action**2 + EPS) - \
                    np.log(self.action_range)
        log_prob = tf.reduce_sum(log_prob, axis=1)[:, None]

        return action, log_prob

    def step_cont(self, obs, deterministic=False):
        mean, log_std = self.model(obs)
        std = tf.exp(log_std)

        pre_squish_action = mean + std * Normal(0, 1).sample()
        pre_squish_action = mean if deterministic else pre_squish_action
        squish_action = tf.math.tanh(pre_squish_action)
        action = squish_action * self.action_range

        return action.numpy()[0]

    def eval(self, obs, flag=False):
        return self.eval_func(obs, flag)

    def step(self, obs, flag=False):
        return self.step_func(obs, flag)
