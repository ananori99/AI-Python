# qlearningAgents.py
# ------------------
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

from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random, util, math


class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """

    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        # "Note that your value iteration agent does not actually learn from experience. Rather, it ponders its MDP
        # model to arrive at a complete policy before ever interacting with a real environment. self.mdp = mdp ...
        # You will now write a Q-learning agent, which does very little on construction,
        # but instead learns by trial and error from interactions with the environment
        # through its update(state, action, nextState, reward) method.
        explore = self.epsilon
        learnRate = self.alpha
        discount = self.discount
        #depth = self.itterations
        # start Q(s,a) = 0 for all s,a
        self.qvalue = util.Counter()
        # now the heavy lifting of what to do is not in init, but in update
        # update will do a sum on the weighted future state max Q values

        # DO NOT ARGMAX UTIL.COUNTER

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        # if a key s,a has not been updated, it defaults to zero
        return self.qvalue[state, action]
        # util.raiseNotDefined()

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        # Make sure that in your computeValueFromQValues and computeActionFromQValues functions, you only access Q
        # values by calling getQValue . This abstraction will be useful for question 8 when you override getQValue to
        # use features of state-action pairs rather than state-action pairs directly.

        # from the slide, this is nested in the sum for update gama*max( [ Q[s1,a1] for a1 in mdp.actions(s1) ]

        legalActions = self.getLegalActions(state)
        bestQ = -99E99

        # for this given s', find the action that has the best qvalue and return that q value
        for a in legalActions:
            if self.getQValue(state, a) > bestQ:
                bestQ = self.getQValue(state, a)

        if len(legalActions) != 0:  # no actions to take, so stay 0 in the counter
            return bestQ
        return 0.0
        # util.raiseNotDefined()

        # Q-value: take action a and act optimally afterward
        # given the state and action at this point
        staProb = self.mdp.getTransitionStatesAndProbs(state, a)
        sum = functools.reduce(lambda x, b: x + b[1] * (self.mdp.getReward(state, a, b[0]) + self.discount *
                                                        self.values[b[0]]), [0] + staProb)
        return sum

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        # For computeActionFromQValues, you should break ties randomly for better behavior. The random.choice()
        # function will help. In a particular state, actions that your agent hasn't seen before still have a Q-value,
        # specifically a Q-value of zero, and if all of the actions that your agent has seen before have a negative
        # Q-value, an unseen action may be optimal.
        # Make sure that in your computeValueFromQValues and
        # computeActionFromQValues functions, you only access Q values by calling getQValue . This abstraction will
        # be useful for question 8 when you override getQValue to use features of state-action pairs rather than
        # state-action pairs directly.
        bestA = []
        bestQ = -999e99
        # store a list of actions in order to randomly choose
        act = self.getLegalActions(state)
        for a in act:
            q = self.getQValue(state, a)
            if q > bestQ:
                bestA = [a]
                bestQ = q
            # now you want to randomly pick in the case of a tie
            if self.getQValue(state, a) == bestQ:
                bestA.append(a)

        if len(bestA) == 0:
            return None
        # if tied, random
        return random.choice(bestA)

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

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        # You can choose an element from a list uniformly at random by calling the random.choice function. You can
        # simulate a binary variable with probability p of success by using util.flipCoin(p), which returns True with
        # probability p and False with probability 1-p.
        # Note that if there are no legal actions, which is the case at
        # the terminal state, you should choose None as the action.
        if len(legalActions) == 0:
            return action
        # with probability self.epsilon, we should take a random action and
        # take the best policy action otherwise.
        coin = flipCoin(self.epsilon)
        caseSwitch ={0 : self.computeActionFromQValues(state), 1 :  random.choice(legalActions) }
        return caseSwitch[coin]


    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        # Q1[q] = sum( [ p * ( R(s,a,s1)
        # + gamma * max( [ Q[s1,a1] for a1 in mdp.actions(s1) ] )
        # max-action-q is with compute value from qvalues fxn
        pR = (1.0 - self.alpha) * self.getQValue(state, action)
        maxQ = reward + self.discount * self.computeValueFromQValues(nextState)
        self.qvalue[state, action] = pR + self.alpha * maxQ


    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05, gamma=0.8, alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self, state)
        self.doAction(state, action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """

    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        # Note: Approximate Q-learning assumes the existence of a feature function f(s,a) over state and action
        # pairs, which yields a vector f1(s,a) .. fi(s,a) .. fn(s,a) of feature values. We provide feature functions
        # for you in featureExtractors.py. Feature vectors are util.Counter (like a dictionary) objects containing
        # the non-zero pairs of features and values; all omitted features have value zero.
        # get the feature dictionary
        f = self.featExtractor.getFeatures(state, action)
        w = self.getWeights()
        #print(f)
        # then you want the sum of w[i]*f[i] for all i
        # Q-value: take action a and act optimally afterward
        # given the state and action at this point
        sum3 = sum(w[i]*f[i] for i in f.keys())
        return sum3

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        # where each weight wi is associated with a particular feature fi(s,a). In your code, you should implement
        # the weight vector as a dictionary mapping features (which the feature extractors will return) to weight
        # values. You will update your weight vectors similarly to how you updated Q-values:
        # first need diffrence, diffrence has q-maxing a' for a given s'
        legalActions = self.getLegalActions(nextState)
        bestQ = -99E99
        g = self.discount
        r = reward
        if len(legalActions) == 0:  # no actions to take, so stay 0 in the counter
            bestQ = 0
        # diff is same as normal q
        for a in legalActions:
            if self.getQValue(nextState, a) > bestQ:
                bestQ = self.getQValue(nextState, a)
        # now that you have maxQ(s',a'), you want r + g*maxQ - Q(s,a)
        diff = r + g*bestQ - self.getQValue(state, action)
        # now that you have diffrence, you want to update
        # w[i] = w[i] + a*diff*f(s,a)
        a = self.alpha
        f = self.featExtractor.getFeatures(state, action)
        w = self.weights
        #w[i] = w[i] + a * diff * f[i]) for i in f.keys()
        for i in f.keys():
            w[i] = w[i] + a * diff * f[i]
        return


    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
