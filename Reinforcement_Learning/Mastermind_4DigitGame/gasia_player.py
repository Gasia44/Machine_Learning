import numpy as np
import itertools
import random


class Player:
    def start(self):
        """
        :return: secret, the 4-digit array that was chosen for the game.
        """
        self.secret = np.random.randint(0, 9 + 1, size=(4))

        self.set = np.zeros([10 * 10 * 10 * 10, 4], dtype=np.int)
        i = 0
        for combination in itertools.product(range(10), repeat=4):
            self.set[i] = np.array(combination)
            i += 1
        self.guessed = []
        return self.secret

    def guess(self):
        # TODO: construct a 4 digit array as a guess.
        if len(self.set) < 70:

            table_look = np.zeros([len(self.set), len(self.set)], dtype=np.int)
            sec_index = 0
            for sec in self.set:
                guess_index = 0
                for my_guess in self.set:
                    reply_subset = self.reply(my_guess, sec)
                    numbers = []

                    for num in self.set:
                        if self.reply(my_guess, num) == reply_subset:
                            numbers.append(num)

                    table_look[guess_index][sec_index] = len(self.set) - len(numbers)
                    guess_index += 1

                sec_index += 1

            table_sum = np.sum(table_look, axis=1)

            aa = self.set[np.argmax(table_sum)]

            while any((aa == x).all() for x in self.guessed):
                table_sum[np.argmax(table_sum)] = -1
                aa = self.set[np.argmax(table_sum)]

            return self.set[np.argmax(table_sum)]

        lout2 = [np.unique(x) for x in self.set]
        length_ = [len(y) for y in lout2]
        ccc = np.argwhere(length_ == np.max(length_))

        rand = self.set[random.choice(ccc)][0]

        while any((rand == x).all() for x in self.guessed):
            rand = self.set[random.choice(ccc)][0]

        return rand

    def proces_oponent_reply(self, your_guess, oponent_reply):
        """
        :param your_guess: guess you have made before
        :param oponent_reply: reply that oponent has given to the previous
            guess
        """
        # TODO: Write logit to process the reply oponent has given to your
        # guess.

        numbers = []
        for num in self.set:
            if self.reply(your_guess, num) == oponent_reply:
                numbers.append(num)

        self.guessed.append(your_guess)
        self.set = np.array(numbers, dtype=np.int)

        return self.set

    def reply(self, guess, secret_compare=None):
        """
        :param guess: a 4-digit array that oponent has guessed
        :return: reply how accurate the guess is with a tuple of 2 numbers
        """
        if secret_compare is None:
            secret_compare = self.secret

        reply = np.zeros(2, dtype=np.int)
        sec = np.copy(secret_compare)
        count = 0
        for i in range(len(guess)):
            if guess[i] in sec:
                sec[np.where(sec == guess[i])[0][0]] = -1
                count += 1

        reply[0] = int(count)

        reply[1] = ((guess == secret_compare).sum())
        # TODO: Given a guess reply how accurate it was.
        return reply[0], reply[1]