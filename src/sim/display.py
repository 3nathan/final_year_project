from os import environ
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

import pygame
import time
import numpy as np

from util.config import Config

CONFIG = Config()

class Display(object):
    def __init__(self, W=CONFIG.DISPLAY_W, H=CONFIG.DISPLAY_H):
        pygame.init()
        self.screen = pygame.display.set_mode((W,H))
        pygame.display.set_caption("sim")
        self.surface = pygame.Surface(self.screen.get_size()).convert()
        self.running = True

    def draw_img(self, img):
        pygame.surfarray.blit_array(self.surface, img.swapaxes(0,1))
        self.screen.blit(self.surface, (0,0))

        pygame.display.flip()

    def show_video(self, frames, fps=60):
        t = 1/fps
        for frame in frames:
            self.handle_close()
            self.draw_img(frame)
            time.sleep(t)

    def handle_close(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                pygame.quit()

    def quit(self):
        self.running = False
        pygame.quit()
