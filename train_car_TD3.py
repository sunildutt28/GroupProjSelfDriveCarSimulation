import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from car_env_FreeHandW import CarEnv
import os
import time

class TrainCallback(BaseCallback):
    def __init__(self, check_freq=1000, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.last_time = time.time()
        
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            now = time.time()
            fps = self.check_freq / (now - self.last_time)
            self.logger.record("time/fps", fps)
            self.last_time = now
            
            if len(self.model.ep_info_buffer) > 0:
                ep_info = self.model.ep_info_buffer[-1]
                if "episode" in ep_info:
                    self.logger.record("rollout/ep_rew_mean", ep_info["episode"]["r"])
                    self.logger.record("rollout/ep_len_mean", ep_info["episode"]["l"])
        return True

def train():
    env = None  # Initialize env variable
    try:
        print("Initializing training...")
        
        # Setup environment
        # Note: TD3 doesn't support multiple envs natively like PPO, so we use n_envs=1
        # env = make_vec_env(lambda: CarEnv(), n_envs=1)
        env = CarEnv()
        
        # Create action noise for TD3 (helps with exploration)
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=0.2 * np.ones(n_actions)
        )
        
        # Model configuration
        model = TD3(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=0.001,  # Typically lower than PPO
            buffer_size=1000000,  # Replay buffer size
            batch_size=100,       # Batch size for training
            gamma=0.99,          # Discount factor
            tau=0.005,           # Target network update rate
            policy_delay=2,      # Policy update delay (TD3 specific)
            action_noise=action_noise,
            train_freq=(1, "step"),  # Update every step
            gradient_steps=1,    # Gradient steps per update
            policy_kwargs={
                'net_arch': [256, 256]  # Network architecture for both actor and critic
            }
        )
        
        # Train with debug callback
        print("Starting training (1,000,000 steps)...")
        model.learn(
            total_timesteps=500_000,
            callback=DebugCallback()
        )
        
        model.save("trained_car_model_FH_TD3.zip")
        print("\nTraining completed. Model saved.")
        
    except Exception as e:
        print(f"\nTraining failed: {str(e)}")
    finally:
        if env is not None:  # Only close if env was created
            env.close()
            print("Environment closed")
        else:
            print("No environment to close")

if __name__ == "__main__":
    train()