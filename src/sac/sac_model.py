
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from src.general.models.model import Model

class SACQNet(Model):
    """
    Soft Actor Critic Q Model

    Attributes:
        state_size: State size the model will accept
        convs: Convolutional layers of the model
        flatten: Flatten operation of the model
        fc: List containing the actor's fully connected layers
        out: Output layer of the actor
    """
    def __init__(self,
                 state_size=None,
                 stack_size=None,
                 action_space=None,
                 fc=None,
                 conv_size=None):
        """
        Constructor.

        Parameters:
            state_size: List containing the expected size of the state
            stack_size: Numer of states we can expect in our stack
            action_space: Box action space the actor will work in
            fc: Iterable containing the amount of neurons per layer for
            the actor model
                ex: (1024, 512, 256) would make 3 fully connected layers, with
                1024, 512 and 256 neurons respectively
            conv_size: Iterable containing the kernel size, stride and number
            of filters for each
                       convolutional layer.
                ex: ((8, 4, 16)) would make 1 convolution layer with an 8x8
                kernel,(4, 4) stride and 16 filters
        """
        super().__init__()

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

        self.fc = [layers.Dense(neurons, activation="activation",
                                   name=f"q_dense_{i}") for i,(neurons)
                      in enumerate(fc)]
        self.out = layers.Dense(1, name=f"q_out")

    def call(self, obs, actions):
        # Run convs on input
        if self.convs is not None:
            conv_out = self.convs(obs)
            dense_in = self.flatten(conv_out)
        else:
            dense_in = obs

        dense = tf.concat([dense_in, actions], axis=-1)
        for l in self.fc:
            dense = l(dense)
        out = self.out(dense)

        return out

class SACActor(Model):
    """
    Soft Actor Critic Actor Model

    Attributes:
        state_size: State size the model will accept
        convs: Convolutional layers of the model
        flatten: Flatten operation of the model
        fc: List containing the actor's fully connected layers
        out: Output layer of the actor
    """
    def __init__(self,
                 state_size=None,
                 stack_size=None,
                 action_space=None,
                 fc=None,
                 conv_size=None):
        """
        Constructor.

        Parameters:
            state_size: List containing the expected size of the state
            stack_size: Numer of states we can expect in our stack
            action_space: Box action space the actor will work in
            fc: Iterable containing the amount of neurons per layer for
            the actor model
                ex: (1024, 512, 256) would make 3 fully connected layers, with
                1024, 512 and 256 neurons respectively
            conv_size: Iterable containing the kernel size, stride and number
            of filters for each
                       convolutional layer.
                ex: ((8, 4, 16)) would make 1 convolution layer with an 8x8
                kernel,(4, 4) stride and 16 filters
        """
        super().__init__()

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

        # Build the layers for the actor and critic models
        self.fc = [layers.Dense(neurons, activation="relu",
                                      name=f"actor_dense_{i}")
                         for i,(neurons) in enumerate(fc)]
        self.mean = layers.Dense(len(action_space), name='actor_mean')
        self.std = layers.Dense(len(action_space), name='actor_std')

    def step(self, obs):
        # Run convs on input
        if self.convs is not None:
            conv_out = self.convs(obs)
            dense_in = self.flatten(conv_out)
        else:
            dense_in = obs

        # Run actor layers
        dense = dense_in
        for l in self.fc:
            dense = l(dense)
        mean = self.mean(dense)
        std = self.std(dense)

        return mean, std

class Custom_Convs(tf.keras.Model):
    """
    Custom Convolution Block
    """
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

class Quake_Block(tf.keras.Model):
    """
    Quake Block

    Convolutions used by Deepmind for Quake. Like original Nature CNN but uses
    more filters and skip connections.
    """
    def __init__(self):
        super().__init__(name='')

        self.conv2A = layers.Conv2D(padding="same", kernel_size=8, strides=4,
                                    filters=32, activation="relu")
        self.conv2B = layers.Conv2D(padding="same", kernel_size=4, strides=2,
                                    filters=64, activation="relu")
        self.conv2C = layers.Conv2D(padding="same", kernel_size=3, strides=1,
                                    filters=64)
        self.activationC = layers.Activation("relu")
        self.conv2D = layers.Conv2D(padding="same", kernel_size=3, strides=1,
                                    filters=64)
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
