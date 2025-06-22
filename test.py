# test_env.py
from car_env import CarEnv 
    # Import from your module
import pygame

if __name__ == "__main__":
    print("Creating environment...")
    env = CarEnv(render_mode='human')
    print("Environment created")
    
    obs, _ = env.reset()
    print("Environment reset")
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        env.render()  # This should now show everything
        
        if terminated or truncated:
            obs, _ = env.reset()
    
    env.close()