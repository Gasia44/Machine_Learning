#!/usr/bin/env python3
"""
Find optimal policy for the following problem:

Jack manages two locations for a nationwide car rental company.
Each day, some number of customers arrive at each location to rent
cars. If Jack has a car available, he rents it out and is credited
$10 by the national company. If he is out of cars at that location,
then the business is lost.  Cars become available for renting the
day after they are returned. To help ensure that cars are available
where they are needed, Jack can move them between the two locations
overnight, at a cost of $2 per car moved. We assume that the number
of cars requested and returned at each location are Poisson random
variables, meaning that the probability that the number is n is λ^n/n! e^{−λ},
where λ is the expected number.  Suppose λ is 3 and 4 for
rental requests at the first and second locations and 3 and 2 for
returns. To simplify the problem slightly, we assume that there can
be no more than 20 cars at each location (any additional cars are
returned to the nationwide company, and thus disappear from the
problem) and a maximum of five cars can be moved from one location
to the other in one night. We take the discount rate to be γ = 0.9
and formulate this as a continuing finite MDP, where the time steps
are days, the state is the number of cars at each location at the
end of the day, and the actions are the net numbers of cars moved
between the two locations overnight.
"""
import argparse
import itertools
from scipy.stats import poisson
from functools import lru_cache
import numpy as np
import math

# Problem Constants
FIRST_RENTAL_MEAN = 3
SECOND_RENTAL_MEAN = 4

FIRST_RETURN_MEAN = 3
SECOND_RETURN_MEAN = 2

theta = 0.0001

@lru_cache(maxsize=None)
def _poisson_pmf(k, mu):
    return poisson.pmf(k, mu)


def parse_args(*argument_array):
    parser = argparse.ArgumentParser()
    parser.add_argument('--discount-rate', type=float, default=0.9)
    parser.add_argument('--max-cars-per-location', type=int, default=20)
    parser.add_argument('--max-cars-can-move', type=int, default=4)
    parser.add_argument('--discounting-rate', type=float, default=0.9)
    args = parser.parse_args(*argument_array)
    return args


def p(new_state, reward, old_state, action):
    """
    Return p(new_state, reward | old_state, action)
    """
    COST_PER_CAR = 2

    probability = []
    state_prob = []
    reward_prob = []

    for i in range(new_state[0]+1):
        for j in range(new_state[1]+1):
            for k in range( args.max_cars_per_location + 1 - new_state[0]):
                for l in range( args.max_cars_per_location + 1 - new_state[1]):
                    s_0 = k - i
                    s_1 = j - l
                    if s_0 > 0 and s_0 < args.max_cars_per_location and s_1 > 0 and s_1 < args.max_cars_per_location:
                        first_  = _poisson_pmf(i, FIRST_RENTAL_MEAN)
                        second_ = _poisson_pmf(j, SECOND_RENTAL_MEAN)
                        third_  = _poisson_pmf(k, FIRST_RETURN_MEAN)
                        fourth_ = _poisson_pmf(l, SECOND_RETURN_MEAN)

                        reward_prob.append(reward + 10*i + 10*j)
                        probability.append( first_*second_*third_*fourth_)
                        state_prob.append((s_0, s_1))

    return probability, reward_prob, state_prob

def greedy_policy(value, state, gamma, all_available_actions):
    policy = np.zeros(value.shape, dtype=np.int)

    for s in state:
        new_state_list = []
        action_list = []


        for action in range(len(all_available_actions)):
            First = s[0] + all_available_actions[action]
            Second = s[1] - all_available_actions[action]

            if(First >= 0 and Second>=0 and First<=args.max_cars_per_location and Second<=args.max_cars_per_location):
                action_list.append(all_available_actions[action])
                new_state_list.append((First, Second))

        temp = np.zeros(len(action_list))
        for i in range(len(action_list)):
            reward = np.abs(action_list[i]) * -2
            probability, reward_prob, state_prob = p(new_state_list[i], reward, s, action_list[i])

            state_prob_0 = [item[0] for item in state_prob]
            state_prob_1 = [item[1] for item in state_prob]
            val_fun_temp = np.zeros(len(state_prob_0))

            for j in range(len(state_prob_0)):
                val_fun_temp[j] = reward_prob[j] + gamma * value[state_prob_0[j]][state_prob_1[j]]

            if probability:
                temp[i] = np.dot(probability, val_fun_temp)
        policy[s[0]][s[1]] = action_list[int(np.argmax(temp))]

    return policy


def main(args):
    all_available_actions = np.arange(-args.max_cars_can_move,
                                      1 + args.max_cars_can_move)
    policy = np.zeros((args.max_cars_per_location + 1,
                       args.max_cars_per_location + 1), dtype=np.int)
    value = np.zeros(policy.shape)


    a = np.arange(args.max_cars_per_location + 1)
    state = list(itertools.product(a, a))

    value_function = np.zeros((args.max_cars_per_location + 1, args.max_cars_per_location + 1), dtype=np.int)

    for _iter in range(5):
        # TODO: Find the value function corresponding to the starting policy

        # delta = 0.001
        # delta_zero = 0
        #while(delta > theta):
        for i in range(5):
            value.dump('iter_{}-value_i{}.dat'.format(_iter, i))
            for s in state:
                p_sum = 0

                old_value_function = value_function[s[0]][s[1]]

                action = policy[s[0]][s[1]]
                new_state = (s[0] + action, s[1] - action )
                if (new_state[0] > 0 and new_state[0] < args.max_cars_per_location and new_state[1] >0 and new_state[1]<args.max_cars_per_location):
                    probability, reward_prob, state_prob= p(new_state, np.abs(action) * -2, s, action)

                    state_prob_0 = [item[0] for item in state_prob]
                    state_prob_1 = [item[1] for item in state_prob]
                    val_fun_temp = np.zeros(len(state_prob_0))

                    for i in range(len(state_prob_0)):
                        val_fun_temp[i] = reward_prob[i] + args.discounting_rate * float(value_function[state_prob_0[i]][state_prob_1[i]])

                    value_function[s[0]][s[1]] = np.dot(probability , val_fun_temp)
                #delta = max(delta_zero, np.abs(old_value_function - value_function[s[0]][s[1]]))
                #delta_zero = delta

        # TODO: Find the greedy policy that corresponds to the value function

        policy = greedy_policy(value_function, state, args.discounting_rate, all_available_actions)

        policy.dump('policy_{}.dat'.format(_iter))

    policy.dump('final_policy.dat')


if __name__ == '__main__':
    args = parse_args()
    main(args)

# def test():
#     assert p((3, 0), -6, (0, 0), -3) < 1e-9
