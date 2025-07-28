import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.callbacks import BaseCallback
import time
from car_env import CarEnv

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
        print("Initializing DDPG training...")
        
        # Setup environment (DDPG doesn't support vectorized envs natively)
        env = CarEnv()  # Single environment only
        
        # Action noise for exploration
        n_actions = env.action_space.shape[0]
        action_noise = OrnsteinUhlenbeckActionNoise(
            mean=np.zeros(n_actions),
            sigma=0.1 * np.ones(n_actions),
            theta=0.15,
            dt=0.01
        )
        
        # Model configuration
        model = DDPG(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=1e-4,          # Typically lower than PPO
            buffer_size=1_000_000,       # Larger replay buffer
            batch_size=128,              # Batch size for training
            tau=0.005,                   # Target network update rate
            gamma=0.99,                  # Discount factor
            action_noise=action_noise,
            policy_kwargs={
                'net_arch': {
                    'pi': [256, 256],    # Actor network
                    'qf': [256, 256]     # Critic network
                }
            }
        )
        
        # Train with debug callback
        print("Starting training (1,000,000 steps)...")
        model.learn(
            total_timesteps=10_000,
            callback=DebugCallback(),
            log_interval=10
        )
        
        model.save("trained_car_model_ddpg")
        print("\nDDPG training completed! Model saved.")
        
    except Exception as e:
        print(f"\nTraining failed: {str(e)}")
    finally:
        if env is not None:
            env.close()
            print("Environment closed")

if __name__ == "__main__":
    train()
    print("Script finished")