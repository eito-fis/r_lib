
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

class SACModel(tf.keras.models.Model):
    """
    Soft Actor Critic model.
     
    Attributes:
        state_size: State size the model will accept
        convs: Convolutional layers of the model
        flatten: Flatten operation of the model
        actor_fc: List containing the actor's fully connected layers
        actor_out: Output layer of the actor
        q1/q2_fc: Fully connected layers for each q network
        q1/q2_out: Output layer for each q network
    """
    def __init__(self,
                 state_size=None,
                 stack_size=None,
                 action_space=None,
                 actor_fc=None,
                 critic_fc=None,
                 conv_size=None):
        """
        Constructor.

        Parameters:
            state_size: List containing the expected size of the state
            stack_size: Numer of states we can expect in our stack
            action_space: Box action space the actor will work in
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
        self.actor_fc = [layers.Dense(neurons, activation="relu",
                                      name=f"actor_dense_{i}")
                         for i,(neurons) in enumerate(actor_fc)]
        self.actor_mean = layers.Dense(len(action_space), name='actor_mean')
        self.actor_std = layers.Dense(len(action_space), name='actor_std')

        # Make Q networks
        self.q1_fc, self.q1_out = self.make_q(critic_fc, "relu", "q1")
        self.q2_fc, self.q2_out = self.make_q(critic_fc, "relu", "q2")
        # Make Q target networks 
        self.q1_t_fc, self.q1_t_out = self.make_q(critic_fc, "relu", "q1_t")
        self.q2_t_fc, self.q2_t_out = self.make_q(critic_fc, "relu", "q2_t")

    def make_q(self, critic_fc, activation, name):
        """
        Helper function to make Q networks
        """
        q_fc = [layers.Dense(neurons, activation="activation",
                                   name=f"{name}_dense_{i}") for i,(neurons)
                      in enumerate(critic_fc)]
        q_out = layers.Dense(1, name=f"{name}_out")
        return q_fc, q_out

    def call(self, obs, actions):
        # Run convs on input
        if self.convs is not None:
            conv_out = self.convs(obs)
            dense_in = self.flatten(conv_out)
        else:
            dense_in = obs

        # Run actor layers
        actor_dense = dense_in
        for l in self.actor_fc:
            actor_dense = l(actor_dense)
        actor_mean = self.actor_mean(actor_dense)
        actor_std = self.actor_std(actor_dense)

        # Run critic layers
        q1_dense = tf.concat([dense_in, actions], axis=-1)
        for l in self.q1_fc:
            q1_dense = l(q1_dense)
        q1_out = q1_out(q1_dense)

        q2_dense = tf.concat([dense_in, actions], axis=-1)
        for l in self.q2_fc:
            q2_dense = l(q2_dense)
        q2_out = q2_out(q2_dense)

        return actor_mean, actor_std, q1_out, q2_out

    def step(self, obs):
        # Run convs on input
        if self.convs is not None:
            conv_out = self.convs(obs)
            dense_in = self.flatten(conv_out)
        else:
            dense_in = obs

        # Run actor layers
        actor_dense = dense_in
        for l in self.actor_fc:
            actor_dense = l(actor_dense)
        actor_mean = self.actor_mean(actor_dense)
        actor_std = self.actor_std(actor_dense)

        return actor_mean, actor_std

    def targets(self, obs, actions):
        # Run convs on input
        if self.convs is not None:
            conv_out = self.convs(obs)
            dense_in = self.flatten(conv_out)
        else:
            dense_in = obs

        # Run target critics
        q1_t_dense = tf.concat([dense_in, actions], axis=-1)
        for l in self.q1_t_fc:
            q1_t_dense = l(q1_t_dense)
        q1_t_out = q1_t_out(q1_t_dense)

        q2_t_dense = tf.concat([dense_in, actions], axis=-1)
        for l in self.q2_t_fc:
            q2_t_dense = l(q2_t_dense)
        q2_t_out = q2_t_out(q2_t_dense)

        return  q1_t_out, q2_t_out

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
