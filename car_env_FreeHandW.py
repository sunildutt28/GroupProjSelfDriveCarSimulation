import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import math
from typing import Optional
import random

class CarEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 60}

    def __init__(self, render_mode: Optional[str] = None, track_image_path="track_map.png"):
        super().__init__()
        self.render_mode = render_mode
        self.track_image_path = track_image_path

        self.action_space = spaces.Box(low=np.array([-1, -1], dtype=np.float32),
                                       high=np.array([1, 1], dtype=np.float32),
                                       dtype=np.float32)

        self.observation_space = spaces.Box(low=0, high=1, shape=(19,), dtype=np.float32)

        self.track_width = 800
        self.track_height = 800

        self.car_length = 30
        self.car_width = 15
        self.max_speed = 8
        self.reset_car_state()

        self.screen = None
        self.clock = None
        self.font = None
        self.ray_colors = [(0, 255, 0, 128) for _ in range(16)]

        self._load_track_from_image(self.track_image_path)

        if self.render_mode == 'human':
            self._init_render()

    def _init_render(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.track_width, self.track_height))
        pygame.display.set_caption("Car - Image-Based Track")
        self.clock = pygame.time.Clock()
        try:
            self.font = pygame.font.SysFont('Arial', 16)
        except:
            self.font = pygame.font.Font(None, 16)

    def _load_track_from_image(self, image_path="track_map.png"):
        if not pygame.get_init():
            pygame.init()
            pygame.display.set_mode((1, 1), flags=pygame.HIDDEN)

        image = pygame.image.load(image_path).convert()
        self.track_surface = pygame.transform.scale(image, (self.track_width, self.track_height))
        self.track_mask_surface = self.track_surface.copy()

        black = (0, 0, 0)
        threshold = (10, 10, 10)
        self.track_mask = pygame.mask.from_threshold(self.track_surface, black, threshold)

    def _is_on_track(self, pos):
        x, y = int(pos[0]), int(pos[1])
        if 0 <= x < self.track_width and 0 <= y < self.track_height:
            return self.track_mask.get_at((x, y)) != 0
        return False

    def _raycast(self, pos, angle):
        max_dist = 300
        step = 5
        for dist in range(0, max_dist, step):
            x = int(pos[0] + dist * math.cos(angle))
            y = int(pos[1] + dist * math.sin(angle))
            if not (0 <= x < self.track_width and 0 <= y < self.track_height):
                return dist / max_dist
            if self.track_mask.get_at((x, y)) == 0:
                return dist / max_dist
        return 1.0

    def reset_car_state(self):
        self.car_pos = np.array([430, 520], dtype=np.float32)
        self.car_angle = -math.pi 
        self.car_speed = 0

    def _get_observation(self):
        distances = [self._raycast(self.car_pos, self.car_angle + angle)
                     for angle in np.linspace(0, 2 * math.pi, 16, endpoint=False)]
        vel_x = math.cos(self.car_angle) * self.car_speed / self.max_speed
        vel_y = math.sin(self.car_angle) * self.car_speed / self.max_speed
        norm_angle = (self.car_angle % (2 * math.pi)) / (2 * math.pi)
        return np.array([*distances, vel_x, vel_y, norm_angle], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.reset_car_state()
        self._load_track_from_image(self.track_image_path)
        return self._get_observation(), {}

    def step(self, action):
        steering, acceleration = np.clip(action, [-0.5, -1], [0.5, 1])
        self.car_speed *= 0.9
        self.car_speed += acceleration * 0.8
        self.car_angle += steering * 0.04
        self.car_speed = np.clip(self.car_speed, 0.05, self.max_speed)

        move_x = self.car_speed * math.cos(self.car_angle)
        move_y = self.car_speed * math.sin(self.car_angle)
        new_pos = self.car_pos + np.array([move_x, move_y], dtype=np.float32)

        on_track = self._is_on_track(new_pos)

        if on_track:
            self.car_pos = new_pos
            # Base rewards
            reward = 0.05 + max(0, self.car_speed) * 0.15
            

            # --- Orientation reward ---
            # Look ahead a small distance
            lookahead_dist = 20
            forward_x = self.car_pos[0] + lookahead_dist * math.cos(self.car_angle)
            forward_y = self.car_pos[1] + lookahead_dist * math.sin(self.car_angle)

            # Sample the pixel ahead
            x, y = int(forward_x), int(forward_y)
            if 0 <= x < self.track_width and 0 <= y < self.track_height:
                direction_pixel = self.track_surface.get_at((x, y))
                # Bonus if car is heading into black road
                if direction_pixel[:3] == (0, 0, 0):
                    reward += 0.1
                else:
                    reward -= 0.1  # discourage misalignment
        else:
            reward = -10.0



        terminated = not on_track
        truncated = False
        self.current_reward = reward
        return self._get_observation(), reward, terminated, truncated, {}

    def render(self):
        if self.screen is None:
            return
        self.screen.fill((240, 240, 240))
        self.screen.blit(self.track_surface, (0, 0))
        car_pos_int = (int(self.car_pos[0]), int(self.car_pos[1]))
        obs = self._get_observation()
        for i, angle in enumerate(np.linspace(0, 2 * math.pi, 16, endpoint=False)):
            dist = obs[i] * 300
            end_x = car_pos_int[0] + dist * math.cos(self.car_angle + angle)
            end_y = car_pos_int[1] + dist * math.sin(self.car_angle + angle)
            pygame.draw.line(self.screen, self.ray_colors[i], car_pos_int, (int(end_x), int(end_y)), 2)
        car_surface = pygame.Surface((self.car_length, self.car_width), pygame.SRCALPHA)
        pygame.draw.rect(car_surface, (200, 0, 0, 255), (0, 0, self.car_length, self.car_width))
        rotated_car = pygame.transform.rotate(car_surface, -self.car_angle * 180 / math.pi)
        car_rect = rotated_car.get_rect(center=car_pos_int)
        self.screen.blit(rotated_car, car_rect)
        speed_text = self.font.render(f"Speed: {self.car_speed:.1f}", True, (0, 0, 0))
        reward_text = self.font.render(f"Reward: {self.current_reward:.1f}", True, (0, 0, 0))
        self.screen.blit(speed_text, (10, 10))
        self.screen.blit(reward_text, (10, 30))
        pygame.display.flip()
        self.clock.tick(self.metadata['render_fps'])

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
