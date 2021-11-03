import os
#os.environ["SDL_VIDEODRIVER"] = "dummy"
import pygame
import numpy as np
import sys
from pygame.locals import *
import tensorflow as tf
from PIL import Image

BLUE  = np.array([255, 255, 0])
DANGER = (255, 0, 0)
RED   = np.array([0, 255, 255])
GREEN = np.array([255, 0, 255])
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)


class Player(pygame.sprite.Sprite):
    def __init__(self):
        super(Player, self).__init__()
        self.surface = pygame.Surface((6,6))
        random_color = np.random.choice(255, size=3)
        pygame.draw.circle(self.surface, random_color, (3,3), 3)
        self.rect = self.surface.get_rect()
        self.rect.update(400-3, 400-3, 6, 6)

    def draw(self, surface):
        surface.blit(self.surface, self.rect)

    def update(self, x, y):
        self.rect.update(x - 3, y - 3, 6, 6)


class PyGameEngine:
    FPS = 30

    def __init__(self, area_size, cyclones, human=False):
        pygame.init()
        self.frames = pygame.time.Clock()
        self.display = pygame.display.set_mode((800,800))
        self.display.fill(WHITE)
        self.player = Player()
        for c in cyclones:
            color = GREEN if c.nature == 1 else RED
            pygame.draw.circle(self.display, list(255-color*c.power*0.5), c.pos*2+400, c.ray*2)
        for c in cyclones:
            color = GREEN if c.nature == 1 else DANGER
            pygame.draw.circle(self.display, color, c.pos*2+400, c.deadly_ray*2)
        pygame.draw.circle(self.display, BLACK, (400,400), area_size*2, width=3)
        self.bg = self.display.copy()

        self.human = human

        #self.img_state = pygame.surfarray.array3d(self.display)
        #new_frame = 0.299 * self.img_state[:, :, 0] + 0.587 * self.img_state[:, :, 1] + 0.114 * self.img_state[:, :, 2]
        #img = tf.image.resize(new_frame[:, :, np.newaxis], (160, 160), method='nearest').numpy().astype(
        #    np.uint8)
        #i = Image.fromarray(img[:,:,0], 'L')
        #i.save('state.png')


    def update(self, x, y):
        self.display.blit(self.bg, (0,0))
        self.player.update(400+x*2, 400+y*2)
        self.player.draw(self.display)
        pygame.draw.circle(self.display, BLACK, (400+x*2,400+y*2), 200, width=1)
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()

        self.frames.tick(PyGameEngine.FPS)
        pygame.display.update()


        if self.human:
            pressed_keys = pygame.key.get_pressed()
            if pressed_keys[K_DOWN]:
                if pressed_keys[K_LEFT]:
                    action_id = 2
                elif pressed_keys[K_RIGHT]:
                    action_id = 4
                elif pressed_keys[K_UP]:
                    action_id = 0
                else:
                    action_id = 3
            elif pressed_keys[K_RIGHT]:
                if pressed_keys[K_LEFT]:
                    action_id = 0
                elif pressed_keys[K_UP]:
                    action_id = 6
                else:
                    action_id = 5
            elif pressed_keys[K_UP]:
                if pressed_keys[K_LEFT]:
                    action_id = 8
                else:
                    action_id = 7
            elif pressed_keys[K_LEFT]:
                action_id = 1
            else:
                action_id = 0

            return action_id

    def exit(self):
        pygame.quit()

