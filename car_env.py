import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import math
from typing import Optional, Tuple

class CarEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 60}
    
    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        self.render_mode = render_mode
        
        # Action space: [steering, acceleration]
        self.action_space = spaces.Box(
            low=np.array([-1, -1], dtype=np.float32),
            high=np.array([1, 1], dtype=np.float32),
            dtype=np.float32
        )
        
        # Observation space: 16 rays + velocity x/y + angle = 19 dimensions
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(19,),
            dtype=np.float32
        )
        
        # Track parameters
        self.track_width = 800
        self.track_height = 600  # Changed to match visualization window
        self.inner_radius = 150
        self.outer_radius = 250
        self._generate_track()
        
        # Car parameters
        self.car_length = 30
        self.car_width = 15
        self.max_speed = 5
        self.reset_car_state()
        
        # Rendering
        self.screen = None
        self.clock = None
        self.font = None
        self.ray_colors = [(255, 0, 0, 128) for _ in range(16)]  # Semi-transparent red
        
        if self.render_mode == 'human':
            self._init_render()

    def _generate_track(self):
        """Generate track boundaries as two separate polygons"""
        self.inner_track = []
        self.outer_track = []
        center_x, center_y = self.track_width//2, self.track_height//2
        
        for angle in np.linspace(0, 2*math.pi, 100):
            # Inner track
            x_inner = center_x + self.inner_radius * math.cos(angle)
            y_inner = center_y + self.inner_radius * math.sin(angle)
            self.inner_track.append((x_inner, y_inner))
            
            # Outer track
            x_outer = center_x + self.outer_radius * math.cos(angle)
            y_outer = center_y + self.outer_radius * math.sin(angle)
            self.outer_track.append((x_outer, y_outer))

    def _init_render(self):
        """Initialize rendering components"""
        pygame.init()
        self.screen = pygame.display.set_mode((self.track_width, self.track_height))
        pygame.display.set_caption("Car Racing Environment")
        self.clock = pygame.time.Clock()
        try:
            self.font = pygame.font.SysFont('Arial', 16)
        except:
            self.font = pygame.font.Font(None, 16)
        
        # Create track surface
        self.track_surface = pygame.Surface((self.track_width, self.track_height), pygame.SRCALPHA)
        self._draw_track()
        print("Render initialized")  # Debug

    def _draw_track(self):
        """Draw the track onto the track surface"""
        print("Drawing track...")  # Debug
        self.track_surface.fill((0, 0, 0, 0))  # Clear with transparent
        
        # Draw outer track (green)
        if len(self.outer_track) > 2:
            pygame.draw.polygon(self.track_surface, (100, 200, 100, 255), 
                            [(int(x), int(y)) for x, y in self.outer_track])
        
        # Draw inner track (white)
        if len(self.inner_track) > 2:
            pygame.draw.polygon(self.track_surface, (255, 255, 255, 255), 
                            [(int(x), int(y)) for x, y in self.inner_track])
        print("Track drawing complete")  # Debug

    def _draw_track(self):
        """Draw the track onto the track surface"""
        # Fill background
        self.track_surface.fill((255, 255, 255, 0))  # Transparent background
        
        # Draw outer track (green)
        pygame.draw.polygon(self.track_surface, (100, 200, 100, 255), 
                          [(int(x), int(y)) for x, y in self.outer_track])
        
        # Draw inner track (white)
        pygame.draw.polygon(self.track_surface, (255, 255, 255, 255), 
                          [(int(x), int(y)) for x, y in self.inner_track])

    def reset_car_state(self):
        """Reset car to starting position"""
        self.car_pos = np.array(
            #[self.track_width//2 - self.inner_radius, self.track_height//2],
            [self.track_width//2 -  self.inner_radius - 50 , self.track_height//2],
            dtype=np.float32
        )
        self.car_angle = math.pi/2  # Pointing upwards
        self.car_speed = 0

    def _get_observation(self):
        """Get normalized observation vector"""
        distances = []
        for angle in np.linspace(0, 2*math.pi, 16, endpoint=False):
            dist = self._raycast(self.car_pos, self.car_angle + angle)
            distances.append(dist)
        
        # Normalized velocity components
        vel_x = math.cos(self.car_angle) * self.car_speed / self.max_speed
        vel_y = math.sin(self.car_angle) * self.car_speed / self.max_speed
        
        # Normalized angle (0-1)
        norm_angle = (self.car_angle % (2*math.pi)) / (2*math.pi)
        
        return np.array([*distances, vel_x, vel_y, norm_angle], dtype=np.float32)

    def _raycast(self, pos, angle):
        """Cast a ray and return normalized distance to track boundary"""
        max_dist = 300
        step = 5
        pos = (int(pos[0]), int(pos[1]))
        
        for dist in range(0, max_dist, step):
            x = pos[0] + dist * math.cos(angle)
            y = pos[1] + dist * math.sin(angle)
            
            if not (0 <= x < self.track_width and 0 <= y < self.track_height):
                return dist / max_dist
                
            if not self._is_on_track((x, y)):
                return dist / max_dist
        return 1.0

    def _is_on_track(self, pos):
        """Check if position is within track boundaries"""
        x, y = pos
        center_x, center_y = self.track_width//2, self.track_height//2
        distance = math.sqrt((x-center_x)**2 + (y-center_y)**2)
        return self.inner_radius <= distance <= self.outer_radius

    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        self.reset_car_state()
        return self._get_observation(), {}

    def step(self, action):
        """Execute one time step
        steering, acceleration = np.clip(action, -1, 1)"""

        """Execute one time step with improved physics"""
        steering, acceleration = np.clip(action, [-1, -1], [1, 1])
        
        """# Update car state
        self.car_angle += steering * 0.1
        self.car_speed += acceleration * 0.2
        self.car_speed = np.clip(self.car_speed, 0, self.max_speed)"""

         # Improved physics
        self.car_angle += steering * 0.05  # Reduced steering sensitivity
        self.car_speed += acceleration * 0.1
        #self.car_speed = np.clip(self.car_speed, -self.max_speed/2, self.max_speed)  # Allow reverse
        self.car_speed = np.clip(self.car_speed, 0, self.max_speed)  
        
        """# Calculate new position
        new_x = self.car_pos[0] + self.car_speed * math.cos(self.car_angle)
        new_y = self.car_pos[1] + self.car_speed * math.sin(self.car_angle)
        new_pos = np.array([new_x, new_y], dtype=np.float32)"""

        # Calculate new position with momentum
        move_x = self.car_speed * math.cos(self.car_angle)
        move_y = self.car_speed * math.sin(self.car_angle)
        new_pos = np.array([
            self.car_pos[0] + move_x,
            self.car_pos[1] + move_y
        ], dtype=np.float32)
        
        # Check track boundaries
        on_track = self._is_on_track(new_pos)
        
        # Calculate reward
        if on_track:
            self.car_pos = new_pos
            forward_reward = math.cos(self.car_angle) * self.car_speed * 0.2
            self.current_reward = forward_reward + 0.1  # Store as instance variable
        else:
            self.current_reward = -2  # Store as instance variable
        
        terminated = not on_track
        truncated = False
        info = {}
        
        return self._get_observation(), self.current_reward, terminated, truncated, info
    
    def render(self):
        """Render environment"""
        if self.screen is None:
            return
            
        try:
            # Clear screen
            print("Starting render...")  # Debug
            self.screen.fill((240, 240, 240))  # Light gray background
            print("Background filled")  # Debug
            
            if not hasattr(self, 'track_surface'):
                print("Track surface missing!")
            else:
                print(f"Track surface size: {self.track_surface.get_size()}")  # Debug
                self.screen.blit(self.track_surface, (0, 0))
                print("Track drawn")  # Debug
        
            # Get integer car position
            car_pos_int = (int(self.car_pos[0]), int(self.car_pos[1]))
            
            # Draw rays
            obs = self._get_observation()
            for i, angle in enumerate(np.linspace(0, 2*math.pi, 16, endpoint=False)):
                dist = obs[i] * 300
                end_x = car_pos_int[0] + dist * math.cos(self.car_angle + angle)
                end_y = car_pos_int[1] + dist * math.sin(self.car_angle + angle)
                pygame.draw.line(
                    self.screen,
                    self.ray_colors[i],
                    car_pos_int,
                    (int(end_x), int(end_y)),
                    2
                )
            
            # Draw car
            car_surface = pygame.Surface((self.car_length, self.car_width), pygame.SRCALPHA)
            pygame.draw.rect(car_surface, (200, 0, 0, 255), (0, 0, self.car_length, self.car_width))
            
            # Rotate car
            rotated_car = pygame.transform.rotate(car_surface, -self.car_angle * 180/math.pi)
            car_rect = rotated_car.get_rect(center=car_pos_int)
            self.screen.blit(rotated_car, car_rect)
            
            # Display info - now using self.current_reward
            speed_text = self.font.render(f"Speed: {self.car_speed:.1f}", True, (0, 0, 0))
            self.screen.blit(speed_text, (10, 10))
            
            reward_text = self.font.render(f"Reward: {self.current_reward:.1f}", True, (0, 0, 0))
            self.screen.blit(reward_text, (10, 30))
            
            pygame.display.flip()
            self.clock.tick(self.metadata['render_fps'])
            
        except Exception as e:
            print(f"Rendering error: {e}")
            
    def close(self):
        """Clean up resources"""
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None