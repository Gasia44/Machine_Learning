#!/usr/bin/env python3
"""
FILL in the description of the script
"""
from __future__ import print_function
import argparse
import random
import os
import argparse
import pandas as pd
import numpy as np


def parse_args(*argument_array):
    parser = argparse.ArgumentParser()
    parser.add_argument('filenames', nargs='+')
    args = parser.parse_args(*argument_array)
    return args


class RandomPlayer:
    def start(self):
        self.digit = np.random.choice(10)
        return 4 * [self.digit]

    def guess(self):
        return np.random.choice(10, 4)

    def proces_oponent_reply(self, your_guess, oponent_reply):
        pass

    def reply(self, guess):
        num_zeros = np.sum(x == self.digit for x in guess)
        return (num_zeros, num_zeros)


class LyingPlayer:
    def start(self):
        return [0, 0, 0, 0]

    def guess(self):
        return np.random.choice(10, 4)

    def proces_oponent_reply(self, your_guess, oponent_reply):
        pass

    def reply(self, guess):
        return (1, 0)


def main(args):
    for _file in args.filenames:
        module_name = os.path.splitext(_file)[0]
        Player = __import__(module_name, globals(), locals(), ['Player'], 0).Player

        p1 = Player()
        liar = LyingPlayer()

        for game in range(4):
            secret, _ = p1.start(), liar.start()

            wrong_digit = (set(range(10)) - set(secret)).pop()

            # Turn 0
            g1 = p1.guess()
            g2 = secret.copy()
            g2[np.random.choice(4)] = wrong_digit

            r1 = liar.reply(g1)
            r2 = p1.reply(g2)

            p1.proces_oponent_reply(g1, r1)
            liar.proces_oponent_reply(g2, r2)

            assert r2 == (3, 3)

            # Turn 1
            g1 = p1.guess()
            g2 = 4 * [wrong_digit]
            g2[0] = secret[1]
            g2[1] = secret[2]
            g2[3] = secret[3]

            r1 = liar.reply(g1)
            r2 = p1.reply(g2)

            p1.proces_oponent_reply(g1, r1)
            liar.proces_oponent_reply(g2, r2)

            assert r2[0] == 3
            assert r2[1] == 1 + 1 * (g2[0] == secret[0]) + 1 * (g2[1] == secret[1])

            # Turn 2
            g1 = p1.guess()
            g2 = 4 * [secret[np.random.choice(4)]]

            r1 = liar.reply(g1)
            r2 = p1.reply(g2)

            p1.proces_oponent_reply(g1, r1)
            liar.proces_oponent_reply(g2, r2)

            assert r2[0] == r2[1]

            # Turn 3
            g1 = p1.guess()
            g2 = secret
            r1 = (4, 2)
            r2 = p1.reply(g2)

            p1.proces_oponent_reply(g1, r1)
            liar.proces_oponent_reply(g2, r2)

            assert r2 == (4, 4)

        # Test against Random Player
        p1 = Player()
        rand = RandomPlayer()

        wins = 0
        for game in range(4):
            secret1, secret2 = p1.start(), rand.start()
            for turn in range(33):
                g1, g2 = p1.guess(), rand.guess()
                r1, r2 = rand.reply(g1), p1.reply(g2)
                p1.proces_oponent_reply(g1, r1), rand.proces_oponent_reply(g2, r2)
                if r1 == (4, 4):
                    wins += 1
                    break
        assert wins == 4
    print('test passed')


if __name__ == '__main__':
    args = parse_args()
    main(args)
