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
                 tau=0.005,
                 buffer_size=50000,
                 batch_size=256,
                 gradient_steps=1,
                 env=None,
                 actor_fc=None,
                 critic_fc=None,
                 conv_size=None,
                 norm_reward=False,
                 logging_period=25,
                 checkpoint_period=50,
                 output_dir="/tmp/sac",
                 restore_dir=None,
                 wandb=None):

        # Build environment
        self.env = env
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
        self.hard_update(self.q1_t, self.q1)
        self.hard_update(self.q2_t, self.q2)
        if restore_dir:
            models = (self.actor, self.q1, self.q2, self.q1_t, self.q2_t)
            for model, restore_file in zip(models, restore_dir):
                model.load_weights(restore_file)
            

        # Build policy, replay buffer and optimizers
        self.policy = SACPolicy(action_space=self.env.action_space,
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
        self.norm_reward = norm_reward

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
        obs = self.env.reset()
        avg_prob = 0
        for i in range(self.train_steps):
            if i < self.random_steps:
                action = self.random_policy()
            else:
                action = self.policy.step(obs[None, :])
            assert action.shape == self.action_space.shape

            # Take step on env with action
            new_obs, rewards, done, self.infos = self.env.step(action)
            # self.env.render()
            # Store SARS(D) in replay buffer
            self.replay_buffer.add(obs, action, rewards, new_obs,
                                   float(done))
            obs = new_obs

            if done:
                self.reward_queue.extend([self.env.ep_reward])
                self.episodes += 1
                obs = self.env.reset()

            # Periodically learn
            if i % self.train_freq == 0:
                for g in range(self.gradient_steps):
                    # Don"t train if the buffer is not full enough or if we are
                    # still collecting random samples
                    if not self.replay_buffer.can_sample(self.batch_size) or \
                       i < self.random_steps:
                        break
                    avg_prob = self.update(i, g, done)

    def update(self, i, g, done, reward_scale=10):
        """
        Samples from the replay buffer and updates the model
        """
        # Sample and unpack batch
        batch = self.replay_buffer.sample(self.batch_size)
        b_obs, b_actions, b_rewards, b_n_obs, b_dones = batch
        b_rewards = b_rewards[:, None]
        b_dones = b_dones[:, None]

        if self.norm_reward:
            reward = reward_scale * (reward - np.mean(reward, axis=0)) / \
                        (np.std(reward, axis=0) + 1e-6)

        # Calculate loss
        b_n_actions, n_log_probs = self.policy.eval(b_n_obs)
        q1_ts = self.q1_t(b_n_obs, b_n_actions)
        q2_ts = self.q2_t(b_n_obs, b_n_actions)
        
        min_q_ts = tf.minimum(q1_ts, q2_ts) - self.alpha * n_log_probs
        target_q = b_rewards + (1 - b_dones) * self.gamma * min_q_ts

        with tf.GradientTape() as q1_tape:
            q1s = self.q1(b_obs, b_actions)
            q1_loss = 0.5 * tf.reduce_mean((q1s - target_q) ** 2)
        q1_grad = q1_tape.gradient(q1_loss, self.q1.trainable_weights)
        self.q1_opt.apply_gradients(zip(q1_grad,
                                        self.q1.trainable_weights))

        with tf.GradientTape() as q2_tape:
            q2s = self.q2(b_obs, b_actions)
            q2_loss = 0.5 * tf.reduce_mean((q2s - target_q) ** 2)
        q2_grad = q2_tape.gradient(q2_loss, self.q2.trainable_weights)
        self.q2_opt.apply_gradients(zip(q2_grad,
                                        self.q2.trainable_weights))

        with tf.GradientTape() as actor_tape:
            new_actions, log_probs = self.policy.eval(b_obs)
            n_q1s = self.q1(b_obs, new_actions)
            n_q2s = self.q2(b_obs, new_actions)
            min_n_qs = tf.minimum(n_q1s, n_q2s)
            actor_loss = tf.reduce_mean(self.alpha * log_probs - min_n_qs)
        actor_grad = actor_tape.gradient(actor_loss,
                                   self.actor.trainable_weights)
        self.actor_opt.apply_gradients(zip(actor_grad,
                                           self.actor.trainable_weights))

        with tf.GradientTape() as alpha_tape:
            entropy_loss = -tf.reduce_mean(self.log_alpha *
                                           tf.stop_gradient(log_probs +
                                                            self.target_entropy))
        entropy_grad = alpha_tape.gradient(entropy_loss, self.log_alpha)
        self.entropy_opt.apply_gradients(zip([entropy_grad],
                                             [self.log_alpha]))
        # Update the entropy constant
        self.alpha = tf.exp(self.log_alpha)

        # Soft updates for the target network
        if (i + g) % self.target_update_freq == 0:
            self.soft_update(self.q1_t, self.q1)
            self.soft_update(self.q2_t, self.q2)

        avg_prob = 0
        if done:
            avg_prob = tf.reduce_mean(tf.exp(log_probs))
            self.log(actor_loss, avg_prob, q1_loss, q2_loss, entropy_loss,
                     i, g)
        return avg_prob

    def soft_update(self, q_t, q):
        """
        Soft update from q to target_q network based on self.tau
        """
        for target_weights, weights in zip(q_t.weights,
                                       q.weights):
            updated_weights = (1 - self.tau) * target_weights + self.tau * weights
            target_weights.assign(updated_weights)

    def hard_update(self, q_t, q):
        """
        Soft update from q to target_q network based on self.tau
        """
        for target_weights, weights in zip(q_t.weights,
                                       q.weights):
            target_weights.assign(weights)

    def log(self, actor_loss, avg_prob, q1_loss, q2_loss, entropy_loss, i, g):
        # Periodically log
        if len(self.reward_queue) == 0:
            avg_reward = 0
        else:
            avg_reward = sum(self.reward_queue) / len(self.reward_queue)

        ep_reward = self.reward_queue[-1]
        print(f"Step {i} - Gradient Step {g}")
        print(f"| Episodes: {self.episodes} |")
        print(f"| Average Reward: {avg_reward} | Ep Reward: {ep_reward} |")
        print(f"| Actor Loss: {actor_loss} | Avg Probs: {avg_prob} |")
        print(f"| Q1 Loss: {q1_loss} | Q2 Loss: {q2_loss} |")
        print(f"| Entropy Loss: {entropy_loss} |")
        print(f"| Alpha: {self.alpha} |")
        print()

        with self.summary_writer.as_default():
            tf.summary.scalar("Average Reward", avg_reward, i)
            tf.summary.scalar("Episode Reward", ep_reward, i)
            tf.summary.scalar("Actor Loss", actor_loss, i)
            tf.summary.scalar("Average Prob", avg_prob, i)
            tf.summary.scalar("Entropy Loss", entropy_loss, i)
            tf.summary.scalar("Q1 Loss", q1_loss, i)
            tf.summary.scalar("Q2 Loss", q2_loss, i)
            tf.summary.scalar("Alpha", self.alpha, i)
        if self.wandb != None:
            self.wandb.log({"Step": i,
                            "Average Reward": avg_reward,
                            "Episode Reward": ep_reward,
                            "Actor Loss": actor_loss.numpy(),
                            "Average Prob": avg_prob.numpy(),
                            "Entropy Loss": entropy_loss.numpy(),
                            "Q1 Loss": q1_loss.numpy(),
                            "Q2 Loss": q2_loss.numpy(),
                            "Alpha": self.alpha.numpy()})

        # Periodically save all models
        if i % self.checkpoint_period == 0 and i != 0:
            self.actor.save(f"actor_model_{i}", self.checkpoint_dir)
            self.q1.save(f"q1_model_{i}", self.checkpoint_dir)
            self.q2.save(f"q2_model_{i}", self.checkpoint_dir)
            self.q1_t.save(f"q1_t_model_{i}", self.checkpoint_dir)
            self.q2_t.save(f"q2_t_model_{i}", self.checkpoint_dir)



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
