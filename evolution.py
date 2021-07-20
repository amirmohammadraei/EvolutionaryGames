import random
from copy import deepcopy
import csv
from player import Player
import numpy as np
from config import CONFIG
import math


class Evolution():

    def __init__(self, mode):
        self.mode = mode

    # calculate fitness of players
    def calculate_fitness(self, players, delta_xs):
        for i, p in enumerate(players):
            p.fitness = delta_xs[i]

    def mutate(self, child):

        child.nn.w1 += np.random.normal(0, 1, child.nn.w1.shape)
        child.nn.w2 += np.random.normal(0, 1, child.nn.w2.shape)

        child.nn.b1 += np.random.normal(0, 0.8, child.nn.b1.shape)
        child.nn.b2 += np.random.normal(0, 0.8, child.nn.b2.shape)
        
        return child

    def generate_new_population(self, num_players, prev_players=None):

        # in first generation, we create random players
        if prev_players is None:
            return [Player(self.mode) for _ in range(num_players)]

        else:
            fitnesses = [i.fitness for i in prev_players]
            chosen = random.choices(prev_players, weights=fitnesses, cum_weights=None, k=num_players) # motanesb ba shayestegi
            childs = [self.mutate(deepcopy(i)) for i in chosen]

            return childs

    def next_population_selection(self, players, num_players):

        players.sort(key=lambda x: x.fitness, reverse=True)

        total_sum = 0
        for player in players: total_sum = total_sum + player.fitness
        maximum = players[0].fitness
        minimum = players[len(players) - 1].fitness
        average = total_sum / len(players)


        with open('answer.csv', mode='a') as answer_file:
            answer_writer = csv.writer(answer_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            answer_writer.writerow([minimum, maximum, average])
        answer_file.close()

        return players[: num_players]
