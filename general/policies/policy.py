import numpy as np
import tensorflow as tf

class Policy():
    """
    General policy framework

    Attributes:
        action_space: Action space of the policy, expected format depends on the
        action_space_type
        representining minimium and maximium values of that action.
        action_space_type: What type of action space the policy is operating in.
        Current possible values are "Discrete" and "Continuous"
        batch_size: Number of actions to be generated at once
    """
    def __init__(self,
                 action_space=None,
                 action_space_type="Discrete",
                 batch_size=1):
        self.action_space = action_space
        self.action_space_type = action_space_type
        self.batch_size = batch_size
    
    def step(self):
        return NotImplementedError

class RandomPolicy(Policy):
    """
    Random Policy

    Attributes:
        action_space: Action space of the policy, expected format depends on the
        action_space_type
        action_space_type: What type of action space the policy is operating in.
        Current possible values are "Discrete" and "Continuous"
            - Discrete should pass an int as action_space
            - Continuous should pass a list of tuples (min_value, max_value) as
            action_space
        batch_size: Number of actions to be generated at once
        sample_func: Function to use to determine action
    """
    def __init__(self,
                 action_space=None,
                 action_space_type="Discrete",
                 batch_size=1):
        super.__init__(action_space=action_space,
                       action_space_type=action_space_type,
                       batch_size=batch_size)
        if action_space_type == "Discrete":
            assert isinstance(action_space, int)
            self.sample_func = self.sample_discrete
            self.dist = [[1 / action_space for _ in action_space]
                         for _ in batch_size]
        elif action_space_type=="Continuous":
            assert isinstance(action_space, list)
            self.sample_func = self.sample_continuous
            self.num_actions = len(action_space)
            self.ranges = [y - x for x,y in action_space]
            self.mins = [x for x,y in action_space]


    def sample_discrete(self):
        """
        Samples from a discrete uniform distribution

        Returns:
            actions: List of length batch_size, where each element is the index
            of the selected element
        """
        actions = tf.random.categorical(tf.math.log(self.dist), 1)
        actions = tf.squeeze(actions, axis=-1)
        return actions
        

    def sample_continuous(self):
        """
        Samples from a continuous uniform distribution

        Returns:
            actions: List of length batch_size, where each element is a list
            that holds the selected value for each dimension.
        """
        # Generate random values between 0 and 1
        actions = tf.random.uniform([self.batch_size, self.num_actions])
        # Convert random actions to appropriate ranges
        actions = actions * self.ranges + self.mins
        return actions

    def step(self):
        return self.sample_func()
