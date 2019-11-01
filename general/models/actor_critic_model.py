
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

class ActorCriticModel(tf.keras.models.Model):
    """General actor-critic model."""
    
    
    def __init__(self,
                 num_actions=None,
                 state_size=None,
                 stack_size=None,
                 actor_fc=None,
                 critic_fc=None,
                 conv_size=None):
        """
        Constructor.

        Parameters:
            num_actions: Number of logits the actor model will output
            state_size: List containing the expected size of the state
            stack_size: Numer of states we can expect in our stack
            actor_fc: Iterable containing the amount of neurons per layer for
            the actor model
            critic_fc: Iterable containing the amount of neurons per layer for
            the critic model
                ex: (1024, 512, 256) would make 3 fully connected layers, with
                1024, 512 and 256 neurons respectively
            conv_size: Iterable containing the kernel size, stride and number
            of filters for each
                       convolutional layer.
                ex: ((8, 4, 16)) would make 1 convolution layer with an 8x8
                kernel,(4, 4) stride and 16 filters
        """
        super().__init__()
        self.num_actions = num_actions

        # Get true input_size
        self.state_size = state_size[:-1] + [state_size[-1] * stack_size]

        # Build convolutional layers
        if conv_size is not None:
            if isinstance(conv_size, tuple):
                self.convs = Custom_Convs(conv_size)
            elif conv_size == "quake":
                self.convs = Quake_Block()
            else: raise ValueError("Invalid CNN Topology")
            self.flatten = layers.Flatten()
        else: self.convs = None
        
        # Build the fully connected layers for the actor and critic models
        self.actor_fc = [layers.Dense(neurons, activation="relu", name="actor_dense_{}".format(i)) for i,(neurons) in enumerate(actor_fc)]
        self.critic_fc = [layers.Dense(neurons, activation="relu", name="critic_dense_{}".format(i)) for i,(neurons) in enumerate(critic_fc)]

        # Build the output layers for the actor and critic models
        self.actor_logits = layers.Dense(num_actions, name='policy_logits')
        self.critic_out = layers.Dense(1, name='value')

    def call(self, inputs):
        # Run convs on input
        if self.convs is not None:
            conv_out = self.convs(inputs)
            dense_in = self.flatten(conv_out)
        else:
            dense_in = inputs

        # Run actor layers
        actor_dense = dense_in
        for l in self.actor_fc:
            actor_dense = l(actor_dense)
        actor_logits = self.actor_logits(actor_dense)

        # Run critic layers
        critic_dense = dense_in
        for l in self.critic_fc:
            critic_dense = l(critic_dense)
        value = self.value(critic_dense)

        return actor_logits, value

    def process_inputs(self, inputs):
        # Convert n_envs x n_inputs list to n_inputs x n_envs list if we have
        # multiple inputs
        inputs = [np.asarray(l) for l in zip(*inputs)]
        return inputs

class Custom_Convs(tf.keras.Model):
    def __init__(self, conv_size, actv="relu"):
        super().__init__(name='')

        self.convs = [layers.Conv2D(padding="same",
                                    kernel_size=k,
                                    strides=s,
                                    filters=f,
                                    activation=actv,
                                    name="conv_{}".format(i))
                      for i,(k,s,f) in enumerate(conv_size)]
    
    def call(self, x):
        for conv in self.convs:
            x = conv(x)
        return x

# Quake 3 Deepmind style convolutions
# Like dopamine but with an additional 3x3 kernel, and skip connections
class Quake_Block(tf.keras.Model):
    def __init__(self):
        super().__init__(name='')

        self.conv2A = layers.Conv2D(padding="same", kernel_size=8, strides=4, filters=32, activation="relu")
        self.conv2B = layers.Conv2D(padding="same", kernel_size=4, strides=2, filters=64, activation="relu")
        self.conv2C = layers.Conv2D(padding="same", kernel_size=3, strides=1, filters=64)
        self.activationC = layers.Activation("relu")
        self.conv2D = layers.Conv2D(padding="same", kernel_size=3, strides=1, filters=64)
        self.activationD = layers.Activation("relu")

    def call(self, x):
        x = self.conv2A(x)
        x = skip_1 = self.conv2B(x)

        x = self.conv2C(x)
        x = skip_2 = layers.add([x, skip_1])
        x = self.activationC(x)

        x = self.conv2D(x)
        x = layers.add([x, skip_2])
        x = self.activationD(x)

        return x



if __name__ == '__main__':
    model = ActorCriticModel(num_actions=4,
                             state_size=[84,84,1],
                             stack_size=4,
                             actor_fc=[32,16],
                             critic_fc=[32,16],
                             conv_size="quake")
    print(model.trainable_weights)
    input()

    ret = model.step(np.stack([[np.random.random((84, 84, 4)).astype(np.float32),
                np.random.random((26,)).astype(np.float32)] for _ in range(5)]))
    print(ret)
    print(ret[0].shape)
    print(type(ret[0]))
    print(ret[1].shape)
    print(type(ret[1]))
    ret2 = model.get_values(np.stack([[np.random.random((84, 84, 4)).astype(np.float32),
                np.random.random((26,)).astype(np.float32)] for _ in range(5)]))
    print((ret2,))
    print(type(ret2))
    print(ret2.shape)
