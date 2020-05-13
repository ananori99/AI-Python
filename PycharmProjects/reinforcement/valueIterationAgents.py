# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).
import functools

import mdp
import util

from learningAgents import ValueEstimationAgent


class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()  # A Counter is a dict with default 0

        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        # a lot like doing expecti-max recurance for some horizon
        # start with v0 = 0, as no time left means no reward
        # given vk, do one ply of expectimax to get vk+1
        # repeat until converged

        states = mdp.getStates()
        # print(states)
        # actions = mdp.getPossibleActions(states[2])
        # print(actions)
        # tranprob = mdp.getTransitionStatesAndProbs(states[2], actions[1])
        # print(tranprob)
        # reward = mdp.getReward(states[2], actions[1], states[2])
        # print(reward)
        # term = mdp.isTerminal(states[0])
        # print(term)

        # for all states initally zero
        # for the depth of horizon
        for k in range(self.iterations):
            # Use the util.Counter class in util.py, which is a dictionary with a default value of zero.
            # Every iteration updates the values and (implicitly) the policy
            # self.values is the acumulator, meaning that at each step it can be looked to to get the vk-1 info
            # for each state, vk is the max over its actions + the time penelized vk-1
            # "Use the "batch" version of value iteration where each vector Vk is computed from a fixed vector Vk-1 (like in lecture)"
            # I think this is what you want? not undated in place?
            vk = util.Counter()
            # for each state, get the max sum over the actions
            for s in states:
                act = mdp.getPossibleActions(s)
                maxk = -99E99
                for a in act:
                    # find the expected value at that step based on probs
                    staProb = mdp.getTransitionStatesAndProbs(s, a)
                    # print(staProb)
                    # list of tuples (state, prob), we would want to fold as prob*reward(stateNow, a, Prob's state) +
                    # g*valueNow(s)
                    # print([0]+staProb)

                    sum2 = functools.reduce(lambda x, y: x + y[1] * (mdp.getReward(s, a, y[0]) + self.discount *
                                                                     self.values[y[0]]), [0] + staProb)

                    maxk = max(maxk, sum2)
                # if the value has not updated/ no actions Make sure to handle the case when a state has no available
                # actions in an MDP (think about what this means for future rewards).
                if len(act) != 0:  # no actions to take, so stay 0 in the counter
                    vk[s] = maxk

            # after that iterate is done, vk becomes the new vk-1
            self.values = util.Counter.copy(vk)
            # so, at the end of iterations vk is now vk-1

        return


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        # Q-value: take action a and act optimally afterward
        # given the state and action at this point
        staProb = self.mdp.getTransitionStatesAndProbs(state, action)
        sum = functools.reduce(lambda x, b: x + b[1] * (self.mdp.getReward(state, action, b[0]) + self.discount *
                                                        self.values[b[0]]), [0] + staProb)
        return sum

        # util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # do mini- expectimax on a
        # from slides and the fxn above, this is a lot easier in terms of q values
        # for s, find a the gives the max q value
        bestA = None
        bestQ = -999e99
        act = self.mdp.getPossibleActions(state)
        for a in act:
            q = self.computeQValueFromValues(state, a)
            if q > bestQ:
                bestA = a
                bestQ = q
        return bestA

        # util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
