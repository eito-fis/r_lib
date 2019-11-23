import os
import argparse

import numpy as np

from src.general.envs.gym_env import GymEnv
from src.sac.sac_agent import SACAgent

def main(args,
         train_steps=1000000,
         random_steps=1000,
         train_freq=1,
         target_update_freq=1,
         actor_lr=0.0001,
         q_lr=0.0001,
         entropy_lr=0.001,
         gamma=0.99,
         alpha=1,
         tau=0.005,
         buffer_size=500000,
         batch_size=256,
         gradient_steps=1,
         actor_fc=(512, 256),
         critic_fc=(512, 256),
         conv_size=None,
         logging_period=25,
         checkpoint_period=5000):

    if args.wandb:
        import wandb
        if args.wandb_name != None:
            wandb.init(name=args.wandb_name,
                       project="hexapod-sac",
                       entity="olin-robolab")
        else:
            wandb.init(project="hexapod-sac",
                       entity="olin-robolab")
        wandb.config.update({"train_steps": train_steps,
                             "random_steps": random_steps,
                             "train_freq": train_freq,
                             "target_update_freq": target_update_freq,
                             "actor_lr": actor_lr,
                             "q_lr": q_lr,
                             "entropy_lr": entropy_lr,
                             "gamma": gamma,
                             "alpha": alpha,
                             "tau": tau,
                             "buffer_size": buffer_size,
                             "batch_size": batch_size,
                             "gradient_steps": gradient_steps,
                             "actor_fc": actor_fc,
                             "critic_fc": critic_fc,
                             "conv_size": conv_size})
    else: wandb = None

    env = GymEnv("LunarLanderContinuous-v2")
    if args.render:
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
                     output_dir=args.output_dir,
                     restore_dir=args.restore,
                     wandb=wandb)
    print("Agent built!")

    print("Strating train...")
    try:
        agent.train()
    finally:
        # Make sure out environment is closed
        # PLEASE DONT HIT CTRL C TWICE
        env.close()
    print("Train done!")

    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train PPO')
    # Directory path arguments
    parser.add_argument(
        '--output-dir',
        type=str,
        default='/tmp/sac')

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
