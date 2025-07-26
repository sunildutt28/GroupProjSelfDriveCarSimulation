import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
import time
import os
import matplotlib.pyplot as plt
from car_env_FreeHandW import CarEnv

class RewardLoggerCallback(BaseCallback):
    def __init__(self, check_freq: int, verbose=0):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_rewards = []
        self.timesteps = []
        
    def _on_step(self) -> bool:
        # Collect rewards for each environment
        for i in range(len(self.locals['infos'])):
            if 'episode' in self.locals['infos'][i]:
                reward = self.locals['infos'][i]['episode']['r']
                length = self.locals['infos'][i]['episode']['l']
                self.episode_rewards.append(reward)
                self.episode_lengths.append(length)
                self.timesteps.append(self.num_timesteps)
                
        if self.num_timesteps % self.check_freq == 0:
            self._log_progress()
            
        return True
    
    def _log_progress(self):
        if len(self.episode_rewards) > 0:
            mean_reward = np.mean(self.episode_rewards[-100:])  # Last 100 episodes
            mean_length = np.mean(self.episode_lengths[-100:])
            print(f"Timestep: {self.num_timesteps}")
            print(f"Mean reward (last 100 episodes): {mean_reward:.2f}")
            print(f"Mean episode length (last 100 episodes): {mean_length:.2f}")
            print("-" * 40)

def train():
    env = None
    try:
        print("Initializing training...")
        
        # Setup environment
        env = make_vec_env(lambda: CarEnv(), n_envs=8)
        
        # Model configuration
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=0.005,
            n_steps=4096,
            batch_size=128,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            max_grad_norm=0.5,
            policy_kwargs={
                'net_arch': {
                    'pi': [256, 256],
                    'vf': [256, 256]
                }
            }
        )
        
        # Train with reward logging callback
        print("Starting training (1,000,000 steps)...")
        callback = RewardLoggerCallback(check_freq=10000)
        model.learn(
            total_timesteps=1_000_000,
            callback=callback
        )
        
        model.save("trained_car_model_FH.zip")
        print("\nTraining completed. Model saved.")
        
        # Save training data for plotting
        training_data = {
            'timesteps': callback.timesteps,
            'rewards': callback.episode_rewards,
            'lengths': callback.episode_lengths
        }
        np.save("training_data.npy", training_data)
        
    except Exception as e:
        print(f"\nTraining failed: {str(e)}")
    finally:
        if env is not None:
            env.close()
            print("Environment closed")
        else:
            print("No environment to close")

if __name__ == "__main__":
    train()
    print("Script finished")