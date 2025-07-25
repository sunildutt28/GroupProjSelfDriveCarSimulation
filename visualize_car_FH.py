import pygame
import numpy as np
from stable_baselines3 import PPO
from car_env_FreeHandW import CarEnv

def visualize():
    # Initialize Pygame with proper error handling
    try:
        pygame.init()
        screen_width, screen_height = 800, 600
        screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Self driving Car - PPO Visualization")
        clock = pygame.time.Clock()
        
        # Initialize font with fallback
        try:
            font = pygame.font.SysFont('Arial', 24)
        except:
            font = pygame.font.Font(None, 24)  # Fallback font
    except Exception as e:
        print(f"Pygame initialization failed: {e}")
        return

    # Load model
    try:
        model = PPO.load("trained_car_model_FH.zip")
        print(f"Model loaded. Expected obs shape: {model.observation_space.shape}")
        print("Policy architecture:", model.policy)
        print("Observation space:", model.observation_space)
        print("Action space:", model.action_space)
    except Exception as e:
        print(f"Error loading model: {e}")
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

        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        # Handle observation shape mismatch
        if len(obs) != model.observation_space.shape[0]:
            obs = np.resize(obs, model.observation_space.shape)
        
        # Get action from model
        action, _ = model.predict(obs, deterministic=True)
        #print(f"Action: Should vary between [-1, 1] {action}")  # Should vary between [-1, 1]
        obs, reward, done, _, _ = env.step(action)

        #print(f"Observation: {obs}, Reward: {reward}, Done: {done}")
        
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
            
            # Text rendering with position validation
            #reward_text = f"Reward: {reward:.2f}"
            #text_surface = font.render(reward_text, True, (255, 255, 255))
            
            # Validate text position
            text_x = 10
            text_y = 10
            if text_x < 0 or text_y < 0:
                text_x, text_y = 0, 0
            
            #screen.blit(text_surface, (text_x, text_y))
            
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