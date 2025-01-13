from os import environ
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

import pygame
import time
import numpy as np

class Display(object):
    def __init__(self, W, H):
        pygame.init()
        self.screen = pygame.display.set_mode((W,H))
        pygame.display.set_caption("sim playback")
        self.surface = pygame.Surface(self.screen.get_size()).convert()
        self.running = True

    def show_img(self, img):
        self.__blit_surface__(img)

        while self.running:
            self.__handle_close__()

    def show_video(self, frames, fps=60):
        t = 1/fps
        for frame in frames:
            self.__handle_close__()
            self.__blit_surface__(frame)
            time.sleep(t)
    
    def __blit_surface__(self, img):
        pygame.surfarray.blit_array(self.surface, img.swapaxes(0,1))
        self.screen.blit(self.surface, (0,0))

        pygame.display.flip()

    def __handle_close__(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                pygame.quit()
