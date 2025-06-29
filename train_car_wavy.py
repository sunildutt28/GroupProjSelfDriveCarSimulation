import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
import time
from car_env_wavyTrackWithObstacles import CarEnv

class DebugCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.last_time = time.time()
        self.last_step = 0
        
    def _on_step(self) -> bool:
        if self.num_timesteps % 100 == 0:
            now = time.time()
            fps = (self.num_timesteps - self.last_step)/(now - self.last_time)
            print(f"Step: {self.num_timesteps} | FPS: {fps:.1f}")
            self.last_time = now
            self.last_step = self.num_timesteps
        return True

def train():
    env = None  # Initialize env variable
    try:
        print("Initializing training...")
        
        # Setup environment
        env = make_vec_env(lambda: CarEnv(), n_envs=8)
        
        # Model configuration
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            #learning_rate=2.5e-4,
            learning_rate=0.001,
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
        
        # Train with debug callback
        print("Starting training (1,000,000 steps)...")
        model.learn(
            total_timesteps=1_000_000,
            callback=DebugCallback()
        )
        
        model.save("trained_car_model_wavy.zip")
        print("\nTraining completed successfully! Model saved.")
        
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
    print("Script finished")