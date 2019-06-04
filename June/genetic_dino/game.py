import os, sys, random
#확인 필요
import numpy as np
import pygame
import copy
import matplotlib.pyplot as plt
#파일 라이브러리들 import
from generation import Generation

class Game():
    #Game 생성자
    def __init__(self):
        #pygame생성자 실행
        pygame.init()

        self.generation = Generation()

        self.population = self.generation.population

        self.game_speed = 4
        self.max_game_speed = 16
        self.high_score = 0
        self.n_gen = 0
        self.current_gen_score = 0

        self.dinos = None
        self.genomes = []

        self.screen = pygame.display.set_mode(scr_size)
        self.clock = pygame.time.Clock()
        pygame.display.set_caption('Genetic T-Rex Run')

        # self.jump_sound = pygame.mixer.Sound('static/점프경로.wav')
        # self.die_sound = pygame.mixer.Sound('static/점프경로.wav')
        # self.check_point_sound = pygame.mixer.Sound('static/점프경로.wav')
        
        self.scores = []
        #plt -> figure/axes 기능?
        self.fig = plt.figure(figsize=(int(width/100), 5))
        self.ax = plt.axes()

        plt.xlabel('Generation', fontsize=18)
        plt.ylabel('Score', fontsize=16)
        plt.show(block=False)

    def intro_screen(self):
        Dino.containers = []
        temp_dino = Dino(44, 47, self.screen)
        temp_dino.isBlinking = True
        game_start = False

        callout, callout_rect = load_image('call_out2.png', 192, 62, -1)
        callout_rect.left = width * 0.05
        callout_rect.top = height * 0.3
        