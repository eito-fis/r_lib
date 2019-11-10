
import os
from collections import deque

import numpy as np
import tensorflow as tf

from src.sac.sac_model import SACQNet, SACActor
from src.sac.sac_policy import SACPolicy
from src.general.policies.policy import RandomPolicy
from src.general.replay_buffers.replay_buffer import ReplayBuffer

class SACAgent():
    """
    SAC Agent class. Builds and trains a model
    """
    def __init__(self,
                 train_steps=None,
                 random_steps=None,
                 train_freq=1,
                 target_update_freq=1,
                 actor_lr=0.0042,
                 q_lr=0.0042,
                 entropy_lr=0.0042,
                 gamma=0.99,
                 alpha=1,
                 tau=0.05,
                 buffer_size=50000,
                 batch_size=64,
                 gradient_steps=1,
                 env=None,
                 actor_fc=None,
                 critic_fc=None,
                 conv_size=None,
                 logging_period=25,
                 checkpoint_period=50,
                 output_dir="/tmp/sac",
                 restore_dir=None,
                 wandb=None):

        # Build environment
        self.env = env
        self.obs = self.env.reset()
        self.action_space = self.env.action_space

        # Build networks
        self.actor = SACActor(state_size=self.env.obs_space,
                              stack_size=self.env.stack_size,
                              action_space=self.env.action_space,
                              fc=actor_fc,
                              conv_size=conv_size)
        self.q1 = SACQNet(state_size=self.env.obs_space,
                          stack_size=self.env.stack_size,
                          action_space=self.env.action_space,
                          fc=critic_fc,
                          conv_size=conv_size)
        self.q2 = SACQNet(state_size=self.env.obs_space,
                          stack_size=self.env.stack_size,
                          action_space=self.env.action_space,
                          fc=critic_fc,
                          conv_size=conv_size)
        self.q1_t = SACQNet(state_size=self.env.obs_space,
                            stack_size=self.env.stack_size,
                            action_space=self.env.action_space,
                            fc=critic_fc,
                            conv_size=conv_size)
        self.q2_t = SACQNet(state_size=self.env.obs_space,
                            stack_size=self.env.stack_size,
                            action_space=self.env.action_space,
                            fc=critic_fc,
                            conv_size=conv_size)
        self.q1_t.set_weights(self.q1.get_weights())
        self.q2_t.set_weights(self.q2.get_weights())
        if restore_dir:
            models = (self.actor, self.q1, self.q2, self.q1_t, self.q2_t)
            for model, restore_file in zip(models, restore_dir):
                model.load_weights(restore_file)
            

        # Build policy, replay buffer and optimizers
        self.policy = SACPolicy(action_space=self.env.action_space,
                                batch_size=1,
                                model=self.actor)
        self.random_policy = RandomPolicy(action_space=self.env.action_space,
                                          action_space_type="Continuous",
                                          batch_size=1)
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.actor_opt = tf.keras.optimizers.Adam(actor_lr)
        self.q1_opt = tf.keras.optimizers.Adam(q_lr)
        self.q2_opt = tf.keras.optimizers.Adam(q_lr)
        self.entropy_opt = tf.keras.optimizers.Adam(entropy_lr)

        # Setup training parameters
        self.gamma = gamma
        self.tau = tau
        self.train_steps = train_steps
        self.random_steps = random_steps
        self.train_freq = train_freq
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        self.gradient_steps = gradient_steps

        # Setup entropy parameters
        self.log_alpha = tf.Variable(tf.math.log(tf.cast(alpha, tf.float32)),
                                     dtype=tf.float32)
        self.alpha = tf.exp(self.log_alpha)
        self.target_entropy = -np.prod(self.env.action_space.shape)

        # Setup logging parameters
        self.reward_queue = deque(maxlen=100)
        self.logging_period = logging_period
        self.checkpoint_period = checkpoint_period
        self.episodes = 0
        self.wandb = wandb

        # Build logging directories
        self.log_dir = os.path.join(output_dir, "logs/")
        os.makedirs(os.path.dirname(self.log_dir), exist_ok=True)
        self.checkpoint_dir = os.path.join(output_dir, "checkpoints/")
        os.makedirs(os.path.dirname(self.checkpoint_dir), exist_ok=True)

        # Build summary writer
        self.summary_writer = tf.summary.create_file_writer(self.log_dir)

    def train(self):
        """
        Trains the model
        """
        for i in range(self.train_steps):
            if i < self.random_steps:
                action = self.random_policy()
            else:
                action, _ = self.policy(np.expand_dims(self.obs, 0))
                action = tf.squeeze(action)
                action = action * np.abs(self.action_space.low)
            assert action.shape == self.action_space.shape

            # Take step on env with action
            new_obs, rewards, done, self.infos = self.env.step(action)
            self.env.render()
            # Store SARS(D) in replay buffer
            self.replay_buffer.add(self.obs, action, rewards, new_obs,
                                   float(done))
            self.obs = new_obs

            if done:
                self.reward_queue.extend([self.env.ep_reward])
                self.episodes += 1
                self.obs = self.env.reset()

            # Periodically learn
            if i % self.train_freq == 0:
                for g in range(self.gradient_steps):
                    # Don"t train if the buffer is not full enough or if we are
                    # still collecting random samples
                    if not self.replay_buffer.can_sample(self.batch_size) or \
                       i < self.random_steps:
                        break
                    self.update(i, g)

    def update(self, i, g):
        """
        Samples from the replay buffer and updates the model
        """
        # Sample and unpack batch
        batch = self.replay_buffer.sample(self.batch_size)
        b_obs, b_actions, b_rewards, b_n_obs, b_dones = batch
        # Probably don"t need to preprocess if not doing parallel actors
        #b_obs = process_inputs(b_obs)

        # Calculate loss
        b_n_actions, n_log_probs = self.policy(b_n_obs)
        q1_ts = self.q1_t(b_n_obs, b_n_actions)
        q2_ts = self.q2_t(b_n_obs, b_n_actions)
        
        # TODO Make sure you don't need to stop gradient here
        min_q_ts = tf.minimum(q1_ts, q2_ts) - self.alpha * n_log_probs
        target_q = b_rewards + (1 - b_dones) * self.gamma * min_q_ts
        with tf.GradientTape(persistent=True) as tape:
            # Q loss
            q1s = self.q1(b_obs, b_actions)
            q2s = self.q2(b_obs, b_actions)
            q1_loss = 0.5 * tf.reduce_mean((q1s - target_q) ** 2)
            q2_loss = 0.5 * tf.reduce_mean((q2s - target_q) ** 2)

            # Policy loss
            new_actions, log_probs = self.policy(b_obs)
            n_q1s = self.q1(b_obs, new_actions)
            n_q2s = self.q2(b_obs, new_actions)
            min_n_qs = tf.minimum(q1s, q2s)
            policy_loss = tf.reduce_mean(self.alpha * log_probs - min_n_qs)

            # Entropy loss
            entropy_loss = -tf.reduce_mean(self.log_alpha *
                                           tf.stop_gradient(log_probs +
                                                            self.target_entropy))
        # Calculate and apply gradients
        q1_grad = tape.gradient(q1_loss, self.q1.trainable_weights)
        self.q1_opt.apply_gradients(zip(q1_grad,
                                        self.q1.trainable_weights))
        q2_grad = tape.gradient(q2_loss, self.q2.trainable_weights)
        self.q2_opt.apply_gradients(zip(q2_grad,
                                        self.q2.trainable_weights))
        actor_grad = tape.gradient(policy_loss,
                                   self.actor.trainable_weights)
        self.actor_opt.apply_gradients(zip(actor_grad,
                                           self.actor.trainable_weights))
        entropy_grad = tape.gradient(entropy_loss, self.log_alpha)
        self.entropy_opt.apply_gradients(zip([entropy_grad], [self.log_alpha]))
        
        # Update the entropy constant
        self.alpha = tf.exp(self.log_alpha)

        # Garbage collect the GradientTape
        del tape

        # Soft updates for the target network
        # TODO: Test this
        if (i + g) % self.target_update_freq == 0:
            self.soft_update(self.q1_t, self.q1)
            self.soft_update(self.q2_t, self.q2)

        self.log(policy_loss, q1_loss, q2_loss, entropy_loss, i, g)

    def soft_update(self, q_t, q):
        """
        Soft update from q to target_q network based on self.tau
        """
        for target_param, param in zip(q_t.trainable_weights,
                                       q.trainable_weights):
            updated_param = (1 - self.tau) * target_param + self.tau * param
            tf.assign(target_param, updated_param)

    def hard_update(self, q_t, q):
        """
        Hard update from q to target_q network
        """
        for target_param, param in zip(q_t.trainable_weights,
                                       q.trainable_weights):
            target_param = param

    def log(self, policy_loss, q1_loss, q2_loss, entropy_loss, i, g):
        if len(self.reward_queue) == 0:
            avg_reward = 0
        else:
            avg_reward = sum(self.reward_queue) / len(self.reward_queue)

        print(f"Step {i} - Gradient Step {g}")
        print(f"| Episodes: {self.episodes} | Average Reward{avg_reward} |")
        print(f"| Policy Loss: {policy_loss} | Entropy Loss: {entropy_loss} |")
        print(f"| Q1 Loss: {q1_loss} | Q2 Loss: {q2_loss} |")
        print(f"| Alpha: {self.alpha} |")
        print()

        # Periodically log
        if i % self.logging_period == 0:
            with self.summary_writer.as_default():
                tf.summary.scalar("Average Reward", avg_reward, i)
                tf.summary.scalar("Policy Loss", policy_loss, i)
                tf.summary.scalar("Entropy Loss", entropy_loss, i)
                tf.summary.scalar("Q1 Loss", q1_loss, i)
                tf.summary.scalar("Q2 Loss", q2_loss, i)
                tf.summary.scalar("Alpha", self.alpha, i)
            if self.wandb != None:
                self.wandb.log({"Step": i,
                                "Average Reward": avg_reward,
                                "Policy Loss": policy_loss,
                                "Entropy Loss": entropy_loss,
                                "Q1 Loss": q1_loss,
                                "Q2 Loss": q2_loss,
                                "Alpha": self.alpha})

        # Periodically save all models
        if i % self.checkpoint_period == 0:
            self.actor_model.save(f"actor_model_{i}", self.checkpoint_dir)
            self.q1_model.save(f"q1_model_{i}", self.checkpoint_dir)
            self.q2_model.save(f"q2_model_{i}", self.checkpoint_dir)
            self.q1_t_model.save(f"q1_t_model_{i}", self.checkpoint_dir)
            self.q2_t_model.save(f"q2_t_model_{i}", self.checkpoint_dir)



if __name__ == "__main__":
    from src.a2c.wrapped_obstacle_tower_env import WrappedObstacleTowerEnv
    env_filename = "../ObstacleTower/obstacletower"
    def env_func(idx):
        return WrappedObstacleTowerEnv(env_filename,
                                       worker_id=idx,
                                       gray_scale=True,
                                       realtime_mode=True)

    print("Building agent...")
    agent = A2CAgent(train_steps=1,
                     entropy_discount=0.01,
                     value_discount=0.5,
                     learning_rate=0.00000042,
                     num_steps=5,
                     env_func=env_func,
                     num_envs=4,
                     num_actions=4,
                     actor_fc=[1024,512],
                     critic_fc=[1024,512],
                     conv_size=((8,4,32), (4,2,64), (3,1,64)),
                     output_dir="./agent_test")
    print("Agent built!")

    print("Starting train...")
    agent.train()
    print("Train done!")
