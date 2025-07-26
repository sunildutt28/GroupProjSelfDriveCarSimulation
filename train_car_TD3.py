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
    # Create environment
    env = CarEnv()
    
    # Create action noise for exploration
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.2 * np.ones(n_actions)  # Slightly higher noise for better exploration
    )
    
    # Configure TD3 model with corrected buffer settings
    model = TD3(
        "MlpPolicy",
        env,
        learning_rate=3e-4,  # More stable learning rate
        buffer_size=200000,
        batch_size=256,
        action_noise=action_noise,
        optimize_memory_usage=False,  # CHANGED: Disabled memory optimization
        policy_kwargs=dict(net_arch=[256, 256]),
        gamma=0.99,
        tau=0.005,
        train_freq=(1, "step"),
        gradient_steps=1,
        verbose=1,
        tensorboard_log="./td3_car_tensorboard/",
        replay_buffer_kwargs=dict(handle_timeout_termination=True)  # Explicitly enable timeout handling
    )
    
    # Create logs directory
    os.makedirs("./td3_car_logs/", exist_ok=True)
    
    print("Starting training...")
    model.learn(
        total_timesteps=500_000,
        callback=TrainCallback(check_freq=1000),
        log_interval=10,
        tb_log_name="TD3_run"
    )
    
    # Save model
    model.save("td3_car_model")
    print("Training completed and model saved.")

if __name__ == "__main__":
    train()