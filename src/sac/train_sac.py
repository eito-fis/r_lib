import os
import argparse

import numpy as np

from src.general.envs.gym_env import GymEnv
from src.sac.sac_agent import SacAgent

def main(args,
         train_steps=100000,
         random_steps=1000,
         train_freq=1,
         target_update_freq=1,
         actor_lr=0.000042,
         q_lr=0.000042,
         entropy_lr=0.000042,
         gamma=0.99,
         alpha=1,
         tau=0.005,
         buffer_size=50000,
         batch_size=512,
         gradient_steps=1,
         actor_fc=(128, 128),
         critic_fc=(128, 128),
         conv_size=None,
         logging_period=10,
         checkpoint_period=100):

    if args.wandb:
        import wandb
        if args.wandb_name != None:
            wandb.init(name=args.wandb_name,
                       project="hexapod-sac",
                       entity="olin-robolab")
        else:
            wandb.init(project="hexapod-sac",
                       entity="olin-robolab")
        wandb.config.update({"train_steps": 100000,
                             "random_steps": 1000,
                             "train_freq": 1,
                             "target_update_freq": 1,
                             "actor_lr": 0.000042,
                             "q_lr": 0.000042,
                             "entropy_lr": 0.000042,
                             "gamma": 0.99,
                             "alpha": 1,
                             "tau": 0.005,
                             "buffer_size": 50000,
                             "batch_size": 512,
                             "gradient_steps": 1,
                             "actor_fc": (128, 128),
                             "critic_fc": (128, 128),
                             "conv_size": None})
    else: wandb = None

    env = GymEnv("LunarLander-v2")
    if render:
        env.render()

    print("Building agent...")
    agent = SACAgent(train_steps=train_steps,
                     random_steps=random_steps,
                     train_freq=train_freq,
                     target_update_freq=target_update_freq,
                     actor_lr=actor_lr,
                     q_lr=q_lr,
                     entropy_lr=entropy_lr,
                     gamma=gamma,
                     alpha=alpha,
                     tau=tau,
                     buffer_size=buffer_size,
                     batch_size=batch_size,
                     gradient_steps=gradient_steps,
                     env=env,
                     actor_fc=actor_fc,
                     critic_fc=critic_fc,
                     conv_size=conv_size,
                     logging_period=logging_period,
                     checkpoint_period=checkpoint_period,
                     output_dir=output_dir,
                     restore_dir=restore,
                     wandb=wandb)
    print("Agent built!")

    print("Strating train...")
    agent.train()
    print("Train done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train PPO')
    # Directory path arguments
    parser.add_argument(
        '--output-dir',
        type=str,
        default='/tmp/ppo')

    # File path arguments
    parser.add_argument(
        '--restore',
        type=str,
        default=None)

    # Run mode arguments
    parser.add_argument(
        '--render',
        default=False,
        action='store_true')

    # WandB flags
    parser.add_argument(
        '--wandb',
        default=False,
        action='store_true')
    parser.add_argument(
        '--wandb-name',
        type=str,
        default=None)
    args = parser.parse_args()

    #logging.getLogger().setLevel(logging.INFO)

    main(args)