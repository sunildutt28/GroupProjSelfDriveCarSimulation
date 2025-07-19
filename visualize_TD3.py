import pygame
import numpy as np
from stable_baselines3 import TD3  # Changed from PPO to TD3
from car_env_FreeHandW import CarEnv

def visualize():
    # Initialize Pygame with proper error handling
    try:
        pygame.init()
        screen_width, screen_height = 800, 600
        screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Car Racing - TD3 Visualization")  # Updated caption
        clock = pygame.time.Clock()
        
        # Initialize font with fallback
        try:
            font = pygame.font.SysFont('Arial', 24)
        except:
            font = pygame.font.Font(None, 24)  # Fallback font
    except Exception as e:
        print(f"Pygame initialization failed: {e}")
        return

    # Load TD3 model instead of PPO
    try:
        model = TD3.load("trained_car_model_FH_TD3.zip")  # Changed to TD3
        print(f"TD3 Model loaded. Expected obs shape: {model.observation_space.shape}")
        print("Policy architecture:", model.policy)
        print("Observation space:", model.observation_space)
        print("Action space:", model.action_space)
    except Exception as e:
        print(f"Error loading TD3 model: {e}")
        pygame.quit()
        return

    env = CarEnv(render_mode='human')
    
    print("Environment reset")
    
    running = True
    while running:
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Get initial observation
        obs, _ = env.reset()
        
        # Handle observation shape mismatch
        if len(obs) != model.observation_space.shape[0]:
            obs = np.resize(obs, model.observation_space.shape)
        
        # Get action from TD3 model
        action, _ = model.predict(obs, deterministic=True)
        
        # TD3 outputs continuous actions - no need for clipping unless your env requires it
        action = np.clip(action, env.action_space.low, env.action_space.high)  # Optional
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Rendering section with robust error handling
        screen.fill((0, 0, 0))  # Clear screen
        
        try:
            # Environment rendering
            env_render = env.render()
            
            if isinstance(env_render, pygame.Surface):
                # Ensure valid position
                render_x = max(0, (screen_width - env_render.get_width()) // 2)
                render_y = max(0, (screen_height - env_render.get_height()) // 2)
                screen.blit(env_render, (render_x, render_y))
            
            # Display additional info
            info_text = [
                f"Reward: {reward:.2f}",
                f"Action: {action[0]:.2f}, {action[1]:.2f}",  # Show steering/throttle
                "TD3 Policy"  # Indicate we're using TD3
            ]
            
            for i, text in enumerate(info_text):
                text_surface = font.render(text, True, (255, 255, 255))
                screen.blit(text_surface, (10, 10 + i * 30))
            
        except Exception as e:
            print(f"Rendering error: {e}")
            # Fallback rendering
            error_msg = "Rendering Error"
            try:
                error_surface = font.render(error_msg, True, (255, 0, 0))
                screen.blit(error_surface, (10, 10))
            except:
                pass  # If even error rendering fails
        
        pygame.display.flip()
        clock.tick(30)
        
        if done:
            obs, _ = env.reset()
    
    # Cleanup
    try:
        env.close()
    except:
        pass
    pygame.quit()

if __name__ == "__main__":
    visualize()