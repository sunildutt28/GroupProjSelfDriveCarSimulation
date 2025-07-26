import pygame
import numpy as np
from stable_baselines3 import TD3
from car_env_FreeHandW import CarEnv

def visualize():
    # Initialize pygame
    pygame.init()
    screen_width, screen_height = 800, 800
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Self driving Car - TD3 Policy")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)
    
    # Load TD3 model
    try:
        model = TD3.load("td3_car_model")
        print("TD3 model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        pygame.quit()
        return
    
    # Create environment
    env = CarEnv(render_mode='human')
    obs, _ = env.reset()
    
    # Stats tracking
    total_episodes = 0
    successful_dropoffs = 0
    episode_reward = 0
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Get action from TD3 policy
        action, _states = model.predict(obs, deterministic=True)
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        
        # Render environment
        env.render()
        
        # Display stats
        stats = [
            f"Episode: {total_episodes}",
            f"Successes: {successful_dropoffs}",
            f"Current Reward: {episode_reward:.1f}",
            f"Action: [{action[0]:.2f}, {action[1]:.2f}]",
            f"Speed: {env.car_speed:.1f}"
        ]
        
        # Draw stats
        screen.fill((0, 0, 0, 0))  # Transparent overlay
        for i, stat in enumerate(stats):
            text = font.render(stat, True, (255, 255, 255))
            screen.blit(text, (10, 10 + i * 25))
        
        pygame.display.flip()
        clock.tick(60)
        
        # Check for episode end
        if terminated or truncated:
            if terminated and env.passenger_picked:  # Successful drop-off
                successful_dropoffs += 1
                print(f"Episode {total_episodes}: Success! Reward: {episode_reward:.1f}")
            else:
                print(f"Episode {total_episodes}: Ended. Reward: {episode_reward:.1f}")
            
            total_episodes += 1
            episode_reward = 0
            obs, _ = env.reset()
    
    env.close()
    pygame.quit()

if __name__ == "__main__":
    visualize()