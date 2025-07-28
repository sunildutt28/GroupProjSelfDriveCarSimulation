import pygame
import numpy as np
from stable_baselines3 import DDPG
from car_env import CarEnv

def visualize():
    # Pygame initialization with error handling
    try:
        pygame.init()
        screen_width, screen_height = 800, 600
        screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Car Racing - DDPG Visualization")
        clock = pygame.time.Clock()
        
        # Initialize font with fallback
        try:
            font = pygame.font.SysFont('Arial', 24)
        except:
            font = pygame.font.Font(None, 24)  # Fallback font
    except Exception as e:
        print(f"Pygame initialization failed: {e}")
        return

    # Load DDPG model
    try:
        model = DDPG.load("trained_car_model_ddpg.zip")  # Changed to DDPG
        print(f"DDPG Model loaded. Expected obs shape: {model.observation_space.shape}")
        print("Policy architecture:", model.policy)
        print("Observation space:", model.observation_space)
        print("Action space (Continuous):", model.action_space)  # Emphasize continuous actions
    except Exception as e:
        print(f"Error loading DDPG model: {e}")
        pygame.quit()
        return

    env = CarEnv(render_mode='human')
    print("Environment reset")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Get action from DDPG (deterministic for visualization)
        obs = env.reset()[0] if 'obs' not in locals() else obs  # Ensure obs exists
        action, _ = model.predict(obs, deterministic=True)
        
        # Execute action
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Handle observation shape mismatch (if needed)
        if len(obs) != model.observation_space.shape[0]:
            obs = np.resize(obs, model.observation_space.shape)

        # Rendering
        screen.fill((0, 0, 0))  # Clear screen
        
        try:
            env_render = env.render()
            if isinstance(env_render, pygame.Surface):
                render_x = max(0, (screen_width - env_render.get_width()) // 2)
                render_y = max(0, (screen_height - env_render.get_height()) // 2)
                screen.blit(env_render, (render_x, render_y))
            
            # Display action values (useful for continuous control)
            action_text = f"Steering: {action[0]:.2f} | Throttle: {action[1]:.2f}"
            action_surface = font.render(action_text, True, (255, 255, 255))
            screen.blit(action_surface, (10, 10))
            
        except Exception as e:
            print(f"Rendering error: {e}")
            error_surface = font.render("Rendering Error", True, (255, 0, 0))
            screen.blit(error_surface, (10, 10))
        
        pygame.display.flip()
        clock.tick(30)  # 30 FPS
        
        if terminated or truncated:
            obs, _ = env.reset()

    # Cleanup
    try:
        env.close()
    except:
        pass
    pygame.quit()

if __name__ == "__main__":
    visualize()