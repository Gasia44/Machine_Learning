#!/usr/bin/env python3
"""
FILL in the description of the script
"""
from __future__ import print_function
from collections import defaultdict
import argparse
import matplotlib.pyplot as plt
import numpy as np


def random_argmax(array):
    return np.random.choice(np.flatnonzero(array == array.max()))


class GreedyPlayer(object):
    def _pick(self, estimates):
        return random_argmax(estimates)

    def play(self, bandits, turns):
        """
        Plays for given number of turns and return reward history.
        """
        rewards = []
        self.estimates = np.zeros(len(bandits))
        self.chosen = np.zeros(len(bandits))

        for i in range(turns):
            optimal = self._pick(self.estimates)
            reward = bandits[optimal].sample()
            self.estimates[optimal] = ((self.estimates[optimal] * self.chosen[optimal])+ reward) / (self.chosen[optimal] +1)
            self.chosen[optimal] += 1

            rewards.append(reward)
        return rewards


class EpsilonPlayer(GreedyPlayer):
    """
    Implement a play that explores epsilon portion of picks.
    """
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def play(self, bandits, turns):
        self.i = np.zeros(turns)
        self.chosen = np.zeros(len(bandits))
        self.estimates = np.zeros(len(bandits))
        rewards = []

        for t in range(turns):
            if np.random.random() < 1 - self.epsilon:
                self.i[t] = random_argmax(self.estimates)
            else:
                self.i[t] = np.random.choice(len(bandits))

            optimal = int(self.i[t])
            reward = bandits[optimal].sample()

            self.estimates[optimal] = ((self.estimates[optimal] * self.chosen[optimal]) + reward) / (self.chosen[optimal] + 1)
            self.chosen[optimal] += 1

            rewards.append(reward)
        return rewards


class GaussianBandit(object):
    def __init__(self, mu, std):
        self.mu = mu
        self.std = std

    def sample(self):
        return np.random.normal(self.mu, self.std)


def parse_args(*argument_array):
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-bandits', type=int, default=10)
    parser.add_argument('--trials', type=int, default=100)
    parser.add_argument('--play-length', type=int, default=500)
    args = parser.parse_args(*argument_array)
    return args


def main(args):
    name_player_tuples = [('greedy', GreedyPlayer()),
                          ('e=0.5', EpsilonPlayer(0.5)),
                          ('e=0.3', EpsilonPlayer(0.3)),
                          ('e=0.1', EpsilonPlayer(0.1)),
                          ('e=0.01', EpsilonPlayer(0.01)),
                          ('e=0.03', EpsilonPlayer(0.03))]
    _all = defaultdict(list)
    for trial in range(args.trials):
        one_trial_rewards = rewards(name_player_tuples, args.n_bandits,
                                    args.play_length)
        for name, reward_hist in one_trial_rewards.items():
            _all[name].append(reward_hist)

    for name, player in name_player_tuples:
        average_rewards = np.array(_all[name]).mean(axis=0)
        plt.plot(range(args.play_length), average_rewards, label=name)
    plt.legend()
    plt.title('average rewards (mean over {} trials)'.format(args.trials))
    plt.show()


def rewards(name_player_tuples, n_bandits, play_length):
    bandits = [GaussianBandit(np.random.normal(0, 1), 1)
               for _ in range(n_bandits)]
    rtn = dict()
    for name, player in name_player_tuples:
        reward_history = player.play(bandits, play_length)
        rtn[name] = reward_history
    return rtn


if __name__ == '__main__':
    args = parse_args()
    main(args)
