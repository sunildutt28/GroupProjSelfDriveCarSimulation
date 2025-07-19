import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise
import time
from car_env_FreeHandW import CarEnv

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
        # Note: TD3 doesn't support multiple envs natively like PPO, so we use n_envs=1
        env = make_vec_env(lambda: CarEnv(), n_envs=1)
        
        # Create action noise for TD3 (helps with exploration)
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=0.1 * np.ones(n_actions)
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
            total_timesteps=100_000,
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
    print("Script finished")