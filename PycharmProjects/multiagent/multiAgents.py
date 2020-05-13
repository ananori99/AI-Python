# multiAgents.py
# --------------
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
#
# Modified by Eugene Agichtein for CS325 Sp 2014 (eugene@mathcs.emory.edu)
#
from numpy import argmax

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    #from the last project


    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        Note that the successor game state includes updates such as available food,
        e.g., would *not* include the food eaten at the successor state's pacman position
        as that food is no longer remaining.
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        currentFood = currentGameState.getFood() #food available from current state
        newFood = successorGameState.getFood() #food available from successor state (excludes food@successor)
        currentCapsules=currentGameState.getCapsules() #power pellets/capsules available from current state
        newCapsules=successorGameState.getCapsules() #capsules available from successor (excludes capsules@successor)
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # from util, useful are getGhostPositions, getCapsules, getFood, getWalls
        walls = currentGameState.getWalls()
        ghostPos = currentGameState.getGhostPositions()
        food = newFood.asList()
        # print(successorGameState) #prints map
        #print(newPos) #prints (x,y)
        # print(currentFood) #boolean table
        # print(newFood) #boolean table
        # print(currentCapsules) # array for power pelets
        # print(newCapsules)
        # print(newGhostStates) # an object, do not use pacman.py has a bunch of helpers to use instead
        #print(newScaredTimes) # array of times for each ghost, zero if not scared
        #print(walls) #boolean map
        #print(ghostPos) #array of (x,y)
        # Hint 1: As features, try the reciprocal of important values (such as distance to food or capsules) rather than
        # just the values themselves.
        # distance to ghosts
        gDist = [manhattanDistance(ghost, newPos) for ghost in ghostPos]
        #then to avoid mult by zero
        gScores = []
        for gd in gDist:
            if (gd==0):#if zero is the distance, you are eaten
                gScores.append(9999999999)
            else:
                gScores.append(1.0/gd)

        # then to get the food
        fDist = [manhattanDistance(f, newPos) for f in food]
        fScores = []
        for fd in fDist:
            if (fd == 0):  # if zero is the distance, you are on the food
                fScores.append(10)
            else:
                fScores.append(1.0/fd)

        # above encourages getting closer to food, to encourage eating the food
        eaten = len(currentFood.asList()) - len(food)
        # to fear dying
        dead = 999999999999999*int(successorGameState.isLose())
        # to want to win, note lower weight than dying
        won =  9999999999999*int(successorGameState.isWin())

        score = sum(fScores) - 5*sum(gScores) + 5*eaten - dead + won

        #print score
        return score
        # Hint 2: The evaluation function you're writing is evaluating state-action pairs; in later parts of the project,
        # you'll be evaluating states.

        return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"

        #print(self.depth)
        #suc = gameState.generateSuccessor(0,'Right')

        #print(getScore(gameState)) #these are problem
        #print(scoreEvaluationFunction(gameState)) # problem
        #print(gameState.getLegalActions(0))
        #print(gameState.getNumAgents())
        # Look-ahead agents evaluate future states whereas reflex agents evaluate actions from the current state.
        #util.raiseNotDefined()

        # Minimax Search from aima, modified

        # pacman does max

        def max_value(gameState, dep):
            if gameState.isWin():  # check for terminal nodes, for pac this is win and lose
                return self.evaluationFunction(gameState)
            if gameState.isLose():
                return self.evaluationFunction(gameState)
            # to limit depth, consider being at the desired depth to be a terminal node
            # depth zero done at start up
            nexD = dep + 1
            if nexD >= self.depth:
                return self.evaluationFunction(gameState)

            v = -9e999  # then do the regular max for pacman's legal moves to find the best
            # then consider all the moves pacman can do
            act = gameState.getLegalActions(0)
            # get a value v max for each action.
            for a in act:
                state = gameState.generateSuccessor(0, a)
                v = max(v, min_value(state, nexD, 1))
            return v


        # ghosts do min, number of ghosts won't change
        ghosts = gameState.getNumAgents() - 1




        #HERE
        def min_value(gameState, dep, agent):
            # print("gh")
            if gameState.isLose() or gameState.isWin():  # check terminal nodes
                #print("ghost")
                return self.evaluationFunction(gameState)
            v = 9e999
            # each ghost must move

            # the best moves for each ghost
            # then consider all the moves pacman can do
            act = gameState.getLegalActions(agent)

            # get a value for each action.
            states = []
            for a in act:
                states.append(gameState.generateSuccessor(agent, a))
            # children = zip(act, score)
            # then each ghost does its own min, with the ghosts getting to move in order of index
            for s in states:
                if agent >= ghosts:  # all ghosts are done and it's pacman's turn
                    vt = max_value(s, dep)
                else:  # next ghost turn
                    vt = min_value(s, dep, agent + 1)
                v = min(v, vt)
            return v

            # Body of minimax_decision starts here:
            # from slides
            # starting at current state, first move is pacman
            # get action score pairs to find max from
            # action, state = argmax(gameState.generateSuccessor(0,),

        act = gameState.getLegalActions(0)
        v = -9e999
        score = []
        for a in act:
            sta = gameState.generateSuccessor(0, a)
            # lambda ((a, s)): min_value(s))
            # then do the min nodes, from which to find the max...
            # state, depth, agent
            vt =  min_value(sta, 0, 1)
            score.append(vt)
        # find the maximum according to score
        mindx = score.index(max(score))
        action = act[mindx]

        return action




class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """


    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # Modified from min-max, based on the aima games.py and the project website
        def rm_stop(list):
            act = []
            for m in list:
                if m != 'Stop':
                    act.append(m)
            return list

        def max_value(gameState, dep, a, b):
            # max wants a higher alpha
            alpha = a
            if gameState.isWin() or gameState.isLose():  # check for terminal nodes, for pac this is win and lose
                return self.evaluationFunction(gameState)
            # to limit depth, consider being at the desired depth to be a terminal node
            # depth zero done at start up
            nexD = dep + 1
            if nexD >= self.depth:
                return self.evaluationFunction(gameState)

            v2 = -9e999  # then do the regular max for pacman's legal moves to find the best
            # then consider all the moves pacman can do
            act = rm_stop(gameState.getLegalActions(0)) # intermeadiate
            #act = []
            #for m in actI:
                #if m != 'Stop':
                 #   act.append(m)


            # get a value v max for each action.
            for ac in act:
                state = gameState.generateSuccessor(0, ac)
                #state depth agent a b
                v2 = max(v2, min_value(state, nexD, 1, alpha, b))
                if v2 > b:
                    return v2
                alpha = max(alpha, v2)  # If you can't prune, update and contine your loop
            return v2

        def min_value(gameState, dep, agent, a, b):
            if gameState.isLose() or gameState.isWin():  # check terminal nodes
                #print("ghost")
                return self.evaluationFunction(gameState)
            v3 = 9e999
            # ghosts do min, number of ghosts won't change
            ghosts = gameState.getNumAgents() - 1
            # min wants lower beta
            beta = b
            # each ghost must move
            # the best moves for each ghost
            # then consider all the moves pacman can do
            act2 = rm_stop(gameState.getLegalActions(agent))
            # get a value for each action
            for ac in act2:
                state = gameState.generateSuccessor(agent, ac)
                if agent < ghosts: # still ghost turn
                    v3 = min(v3, min_value(state, dep, agent + 1, a, beta))
                    if v3 < a:
                        return v3
                    beta = min(beta, v3) # If you can't prune, update and contine your loop
                else: #pacman's turn
                    v3 = min(v3, max_value(state, dep, a, beta))
                    if v3 < a:
                        return v3
                    beta = min(beta, v3)  # If you can't prune, update and contine your loop
            return v3

        # Body of minimax_decision starts here:
        # from slides
        # starting at current state, first move is pacman
        # get action score pairs to find max from
        # action, state = argmax(gameState.generateSuccessor(0,),

        act = rm_stop(gameState.getLegalActions(0))
        v = -9e999
        score = []
        A = -9e999
        B = 9e999
        for a in act:
            sta = gameState.generateSuccessor(0, a)
            # lambda ((a, s)): min_value(s))
            # then do the min nodes, from which to find the max...
            # state, depth, agent
            vt = min_value(sta, 0, 1, A, B)
            score.append(vt)
            if vt >= B:
                return a
            A = max(A,vt)
        # find the maximum according to score
        mindx = score.index(max(score))
        action = act[mindx]

        return action


# FROM AIMA, unchanged

            # Body of alphabeta_search starts here:
            # The default test cuts off at depth d or at a terminal state

        #cutoff_test = (cutoff_test or
                       #(lambda state, depth: depth > d or game.terminal_test(state)))
        #eval_fn = eval_fn or (lambda state: game.utility(state, player))
        #action, state = argmax(game.successors(state),
                               #lambda ((a, s)): min_value(s, -infinity, infinity, 0))
        #return action



class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        # Modified from min-max, based on the aima games.py and the project website
        # don't prune, as ghosts now choose amongst their getLegalActions uniformly at random

        # what pacman does still has not changed
        def max_value(gameState, dep):
            if gameState.isWin():  # check for terminal nodes, for pac this is win and lose
                return self.evaluationFunction(gameState)
            if gameState.isLose():
                return self.evaluationFunction(gameState)
            # to limit depth, consider being at the desired depth to be a terminal node
            # depth zero done at start up
            nexD = dep + 1
            if nexD >= self.depth:
                return self.evaluationFunction(gameState)

            v = -9e999  # then do the regular max for pacman's legal moves to find the best
            # then consider all the moves pacman can do
            act = gameState.getLegalActions(0)
            # get a value v max for each action.
            for a in act:
                state = gameState.generateSuccessor(0, a)
                v = max(v, exp_value(state, nexD, 1))
            return v


        # ghosts do min, number of ghosts won't change
        ghosts = gameState.getNumAgents() - 1




        #HERE, now these are not min nodes but get the expectation for uniform
        def exp_value(gameState, dep, agent):
            # print("gh")
            if gameState.isLose() or gameState.isWin():  # check terminal nodes
                #print("ghost")
                return self.evaluationFunction(gameState)
            v = 0.0 # instead of comparing the min, accumulate the average
            # each ghost must move

            # the best moves for each ghost
            # then consider all the moves pacman can do
            act = gameState.getLegalActions(agent)
            # uniform distribution, so probability is based on number of choices
            prob = 1.0/len(act)

            # get a value for each action.
            states = []
            for a in act:
                states.append(gameState.generateSuccessor(agent, a))
            # children = zip(act, score)
            # then each ghost does its own min, with the ghosts getting to move in order of index
            for s in states:
                if agent >= ghosts:  # all ghosts are done and it's pacman's turn
                    vt = max_value(s, dep)
                else:  # next ghost turn
                    vt = exp_value(s, dep, agent + 1)
                v = v + prob*vt # getting the average instead of minimizing
            return v

            # Body of minimax_decision starts here:
            # from slides
            # starting at current state, first move is pacman
            # get action score pairs to find max from
            # action, state = argmax(gameState.generateSuccessor(0,),

        act = gameState.getLegalActions(0)
        v = -9e999
        score = []
        for a in act:
            sta = gameState.generateSuccessor(0, a)
            # lambda ((a, s)): min_value(s))
            # then do the min nodes, from which to find the max...
            # state, depth, agent
            vt =  exp_value(sta, 0, 1)
            score.append(vt)
        # find the maximum according to score
        mindx = score.index(max(score))
        action = act[mindx]

        return action





def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
        "One way you might want to write your evaluation function is to use a linear combination of features.
        That is, compute values for features about the state that you think are important, and then combine those
        features by multiplying them by different values and adding the results together. You might decide
        what to multiply each feature by based on how important you think it is."

        obvious features are food nearby, food is more eaten, a pellet is close, a pellet is in effect, ghosts are near
        or far (depending on pellet), and that pacman is not cornered (has many actions to chose from)

        more ambitious might be ignoring ghosts/food if a wall protects you...

        How each of these is done:
        penelty / reward for food and ghosts distances is same as reflex
            could do filter on the points based on wall and direction??? or weight basesd on this?
        to encorage eating food without being a reflex agent/ successor states, just note that this evaluation is for a
            particular map. as in, food does not seem like it can increase as time progresses. only decrease or stay the
            same. because the states are judged relative to each other (and pellets do not ramdomly appear), a penelty
            for having more food on the map encourages pacman to clear his plate.
        to see if pellet is close, do a pellet distance. much like food and ghosts. Also has a flag for if a pellet was
            eaten this turn
        the other way that the flag for seeing if the ghosts are scared is from their scared timers, this also gives a
            weight based on what percentage they are through with being scared. For the scared timers, the pair of time
            and position is tracked for each ghost. the default not-scared seems to be zero, with the max seeming to be
            39? Can adjust weights if this is wrong...
            Because of the way ghost distances were calculated, it would be cumbersome to remove the scared ghosts from
            the ghost distance. however, in ghost distance each scared ghost just contributes c*1/dist...
            as such, the contribution of the scared ghosts is 2*c*1/dist. this lets a ghost that is half scared cancel
            out the gDist, and the greater than and less either diminish or overwhelm gScore
        to see that pacman is not cornered, simply check the number of legal action. If less than 2, do penelty.
            otherwise reward based on having more available moves. the penelty is the negative reciprocal, so that 1
            move is worse than 2

    """
    "*** YOUR CODE HERE ***"
    # Useful information you can extract from a GameState (pacman.py)
    newPos = currentGameState.getPacmanPosition()
    currentFood = currentGameState.getFood()  # food available from current state
    currentCapsules = currentGameState.getCapsules()  # power pellets/capsules available from current state
    currentGhostStates = currentGameState.getGhostStates()
    ScaredTimes = [ghostState.scaredTimer for ghostState in currentGhostStates]
    # from util, useful are getGhostPositions, getCapsules, getFood, getWalls
    walls = currentGameState.getWalls()
    ghostPos = currentGameState.getGhostPositions()
    food = currentFood.asList()
    pellet = currentCapsules

    gDist = [manhattanDistance(ghost, newPos) for ghost in ghostPos]
    # then to avoid mult by zero
    gScores = []
    for gd in gDist:
        if (gd == 0):  # if zero is the distance, you are eaten
            gScores.append(9999999999)
        else:
            gScores.append(1.0 / gd)

    # then to get the food
    fDist = [manhattanDistance(f, newPos) for f in food]
    fScores = []
    for fd in fDist:
        if (fd == 0):  # if zero is the distance, you are on the food
            fScores.append(10)
        else:
            fScores.append(1.0 / fd)

        # then to get the pellets, it also handles the flag for the sign change
    #pFlag = 1
    pDist = [manhattanDistance(p, newPos) for p in pellet]
    pScores = []
    for pd in pDist:
        if (pd == 0):  # if zero is the distance, you are on the food
            pScores.append(10)
            #pFlag = -1
        else:
            pScores.append(1.0 / pd)

    # above encourages getting closer to food, to encourage eating the food
    eaten = - len(food)
    # to fear dying
    dead = 999999999999999 * int(currentGameState.isLose())
    # to want to win, note lower weight than dying
    won = 9999999999999 * int(currentGameState.isWin())

    # then, to see if the ghosts are scared and from there adjust for how long the are scared
    # in case different ghosts can have different scared times, and because these lists are built in order of index
    gTup = zip(ScaredTimes, ghostPos)
    scare = 0
    # scared time
    for (ti, po) in gTup :
        if ti>0:
            scare = scare + (float(ti)/39.0)*(1.0/manhattanDistance(po, newPos))

    # to avoid being traped
    trap = 0
    moves = currentGameState.getLegalActions(0)
    move = []
    for m in moves:
        if m != 'Stop':
            move.append(m)
    #print(len(move))
    l = len(moves)
    if l < 2 : #because of Stop
        trap = l-2
    #print('foo')

    score = sum(pScores) +sum(fScores) - 5 * sum(gScores) + 5 * eaten - dead + won + 10 * scare + 10*trap
    return score

# Abbreviation
better = betterEvaluationFunction


def expirimentalEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
        "One way you might want to write your evaluation function is to use a linear combination of features.
        That is, compute values for features about the state that you think are important, and then combine those
        features by multiplying them by different values and adding the results together. You might decide
        what to multiply each feature by based on how important you think it is."

        obvious features are food nearby, food is more eaten, a pellet is close, a pellet is in effect, ghosts are near
        or far (depending on pellet), and that pacman is not cornered (has many actions to chose from)

        more ambitious might be ignoring ghosts/food if a wall protects you...

        How each of these is done:
        penelty / reward for food and ghosts distances is same as reflex
            could do filter on the points based on wall and direction??? or weight basesd on this?
        to encorage eating food without being a reflex agent/ successor states, just note that this evaluation is for a
            particular map. as in, food does not seem like it can increase as time progresses. only decrease or stay the
            same. because the states are judged relative to each other (and pellets do not ramdomly appear), a penelty
            for having more food on the map encourages pacman to clear his plate.
        to see if pellet is close, do a pellet distance. much like food and ghosts. Also has a flag for if a pellet was
            eaten this turn
        the other way that the flag for seeing if the ghosts are scared is from their scared timers, this also gives a
            weight based on what percentage they are through with being scared. For the scared timers, the pair of time
            and position is tracked for each ghost. the default not-scared seems to be zero, with the max seeming to be
            39? Can adjust weights if this is wrong...
            Because of the way ghost distances were calculated, it would be cumbersome to remove the scared ghosts from
            the ghost distance. however, in ghost distance each scared ghost just contributes c*1/dist...
            as such, the contribution of the scared ghosts is 2*c*1/dist. this lets a ghost that is half scared cancel
            out the gDist, and the greater than and less either diminish or overwhelm gScore
        to see that pacman is not cornered, simply check the number of legal action. If less than 2, do penelty.
            otherwise reward based on having more available moves. the penelty is the negative reciprocal, so that 1
            move is worse than 2

            ACTUALLY, IT SEEMS "EATING" SCARED GHOSTS DOES NOT MAKE THEM DISAPEAR, IT ONLY MAKES THEM STOP BEING SCARED
            AS SUCH IT IS NOT GOOD TO CHASE SCARED GHOSTS, ONLY TO IGNORE THEM. THE SCARED GHOSTS ARE ALREADY DOING THE
            AVOIDANCE WORK

        if ghosts are scared, care less about being traped

        INSTEAD OF LUMPING GHOSTS TOGETHER, MIGHT WANT TO CHECK STATE AND DO EACH SEPERATE

        SWITCHED TO MAP, FILTER, REDUCE FOR SPEED-UP. THESE ARE EXPLAINED IN IN-CODE COMMENTS
        GOT RID OF WHAT IFs I COULD, THAT IS WHY YOU SEE  VERY SMALL # TO AVOID ZERO DIVISION

        MAP IS FASTER FOR SMALL N THAN LIST COMPREHENSION, AS IN FOOD, GHOST, PELLET



    """
    "*** YOUR CODE HERE ***"
    # Useful information you can extract from a GameState (pacman.py)
    newPos = currentGameState.getPacmanPosition()
    currentFood = currentGameState.getFood()  # food available from current state
    currentCapsules = currentGameState.getCapsules()  # power pellets/capsules available from current state
    currentGhostStates = currentGameState.getGhostStates()
    ScaredTimes = [ghostState.scaredTimer for ghostState in currentGhostStates]
    ghostPos = currentGameState.getGhostPositions()
    food = currentFood.asList()
    pellet = currentCapsules
    pe, foo = 0, 0


    moves = list(filter(lambda x: x != 'Stop', currentGameState.getLegalActions(0)))
    l = len(moves)
    trap = 10e2*(l - 1)

    # then, to see if the ghosts are scared and from there adjust for how long the are scared
    # in case different ghosts can have different scared times, and because these lists are built in order of index
    gTup = zip(ScaredTimes, ghostPos)
    # Then filter out the ghosts that still have at least 10 seconds of being scared
    scareMinT = 10
    badG = list(filter(lambda x: x[0] < scareMinT, gTup))

    # then for the bad ghosts, see who you should ignore because of walls
    safe = list(filter(lambda x: x not in moves, ['South', 'North','East', 'West']))
    # so far, the safe list directions where a wall my protect you, now verify that the wall extends above and bellow
    # that point that is blocked
    x,y = newPos
    trueSafe = []

    for dir in safe:
        if dir == 'South' and currentGameState.hasWall(x-1, y) and currentGameState.hasWall(x-1, y+1) and currentGameState.hasWall(x-1, y-1): # south means the x-coord bellow is blocked, so you want to know if the points to the left and right of it are also wall
            trueSafe.append(dir)
        if dir == 'North' and currentGameState.hasWall(x+1, y) and currentGameState.hasWall(x+1, y+1) and currentGameState.hasWall(x+1, y-1):  # south means the x-coord bellow is blocked, so you want to know if the points to the left and right of it are also wall
            trueSafe.append(dir)
        if dir == 'East' and currentGameState.hasWall(x, y+1) and currentGameState.hasWall(x+1, y+1) and currentGameState.hasWall(x-1, y+1):  # south means the x-coord bellow is blocked, so you want to know if the points to the left and right of it are also wall
            trueSafe.append(dir)
        if dir == 'West' and currentGameState.hasWall(x, y-1) and currentGameState.hasWall(x+1, y-1) and currentGameState.hasWall(x-1, y-1):  # south means the x-coord bellow is blocked, so you want to know if the points to the left and right of it are also wall
            trueSafe.append(dir)

    # now that there is the list of where there are walls, you can see if there are ghost you can ignore...
    #So, for example for the North, the x-diffrence for pacman to a ghost is larger than y and positive
    # x,y are pacman. pos is ghosts
    def ghotDir(gx, gy) :  #don't bother with ignoring ghost that are far, as these can move and have twists and turns
        # also, this function is used for food to ignore. it would be bad to ignore a food dense other half...
        if  x-gx>0 and x-gx>1.5*abs(y-gy): # pacman above
            return 'South'
        if  gx-x>0 and gx-x>1.5*abs(y-gy): # pacman below
            return 'North'
        if y-gy>0 and y-gy>1.5*abs(x-gx): # pacman east
            return 'East'
        if gy-y > 0 and gy-y> 1.5* abs(x-gx):  # pacman west
            return 'West'
        else:
            return 'no'


    # then filter out the bad ghosts that are blocked by walls, as in not in safe
    badG = list(filter(lambda x: ghotDir(int(x[1][0]), int(x[1][1])) not in trueSafe, badG))
    bad = list(zip(*badG))
    #print(bad)

    # now for this new list of scared times that has those ghosts blocked by walls filtered out, you want to
    # map the positions to inverse manhaten,
    # and then reduce the weight on those that are still a little scared, ie (39-time)/time
    # then sum it all
    # seems like a mapreduce?
    # x is scared, y is ghost position

    if len(bad) :
        final_list = sum(lambda x, y: (39 - x) / 39 * (1 / (1e-10 + manhattanDistance(y, newPos))), bad[0], bad[1])
        # this should be relativly inconsequential, but faster than the if
        gScore = final_list
    else:
        gScore = 0



    #then, we also want to ignore food and pellets that we can't reach
    # true Safe already tells where there are walls
    def foodDir(gx, gy) :  #don't bother with ignoring ghost that are far, as these can move and have twists and turns
        # also, this function is used for food to ignore. it would be bad to ignore a food dense other half...
        dIg = 5
        if dIg>x-gx and x-gx>0 and x-gx>1.5*abs(y-gy): # pacman above
            return 'South'
        if dIg>gx-x and gx-x>0 and gx-x>1.5*abs(y-gy): # pacman below
            return 'North'
        if dIg>y-gy and y-gy>0 and y-gy>1.5*abs(x-gx): # pacman east
            return 'East'
        if dIg>gy-y and gy-y > 0 and gy-y> 1.5* abs(x-gx):  # pacman west
            return 'West'
        else:
            return 'no'

    # this is the list of food not blocked by a wall
    goodF = list(filter(lambda x: foodDir(int(x[0]), int(x[1])) not in trueSafe, food))
    if len(goodF) :
        foo = sum(list(map(lambda y: (1/(1e-10+manhattanDistance(y, newPos))), goodF)))
        # this should be relativly inconsequential, but faster than the if

    # pellets not blocked by wall
    goodP = list(filter(lambda x: foodDir(int(x[0]), int(x[1])) not in trueSafe, pellet))
    if len(goodP):
        pe = sum(list(map(lambda y: (1 / (1e-10 + manhattanDistance(y, newPos))), goodP)))
        # this should be relativly inconsequential, but faster than the if



    # above encourages getting closer to food, to encourage eating the food
    eaten = -len(food)
    eatenP = -len(currentCapsules)
    # to value eating ghosts
    # eatenG = len(gScores)
    # to fear dying
    dead = 999999999999999 * int(currentGameState.isLose())
    # to want to win, note lower weight than dying
    won = 9999999999999 * int(currentGameState.isWin())


    c = 1

    score = pe + foo - c *3* gScore + 5*eaten + c*eatenP - dead + won - 2*trap
    return score

# Abbreviation
exp = expirimentalEvaluationFunction


class ContestAgent(MultiAgentSearchAgent):
    """
      Your agent for the mini-contest
    """

    def getAction(self, gameState):
        """
          Returns an action.  You can use any method you want and search to any depth you want.
          Just remember that the mini-contest is timed, so you have to trade off speed and computation.

          Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
          just make a beeline straight towards Pacman (or away from him if they're scared!)
        """
        "*** YOUR CODE HERE ***"
        # as the ghost now chase, the assumptions of min-max and alpha-beta are now true
        # also, better function already accounts for pellets and dead ends
        # so just a-b with better function
        gameState.evalFn = exp # working in a dummy function so I don't have to worry about breaking question 5
        self.depth = 4

        # Modified from min-max, based on the aima games.py and the project website


        def max_value(gameState, dep, a, b):
            # max wants a higher alpha
            alpha = a
            if gameState.isWin() or gameState.isLose():  # check for terminal nodes, for pac this is win and lose
                return self.evaluationFunction(gameState)
            # to limit depth, consider being at the desired depth to be a terminal node
            # depth zero done at start up
            nexD = dep + 1
            if nexD >= self.depth:
                return self.evaluationFunction(gameState)

            v2 = -9e999  # then do the regular max for pacman's legal moves to find the best
            # then consider all the moves pacman can do
            act  = list(filter(lambda x: x != 'Stop', gameState.getLegalActions(0)))
            #act = rm_stop(gameState.getLegalActions(0))  # intermeadiate
            #print(rm_stop(act))

            # get a value v max for each action.
            for ac in act:
                state = gameState.generateSuccessor(0, ac)
                # state depth agent a b
                v2 = max(v2, min_value(state, nexD, 1, alpha, b))
                if v2 > b:
                    return v2
                alpha = max(alpha, v2)  # If you can't prune, update and contine your loop
            return v2

        def min_value(gameState, dep, agent, a, b):
            if gameState.isLose() or gameState.isWin():  # check terminal nodes
                # print("ghost")
                return self.evaluationFunction(gameState)
            v3 = 9e999
            # ghosts do min, number of ghosts won't change
            ghosts = gameState.getNumAgents() - 1
            # min wants lower beta
            beta = b
            # each ghost must move
            # the best moves for each ghost
            # then consider all the moves pacman can do
            act2 = list(filter(lambda x: x != 'Stop', gameState.getLegalActions(agent)))
            #act2 = rm_stop(gameState.getLegalActions(agent))
            # get a value for each action
            for ac in act2:
                state = gameState.generateSuccessor(agent, ac)
                if agent < ghosts:  # still ghost turn
                    v3 = min(v3, min_value(state, dep, agent + 1, a, beta))
                    if v3 < a:
                        return v3
                    beta = min(beta, v3)  # If you can't prune, update and contine your loop
                else:  # pacman's turn
                    v3 = min(v3, max_value(state, dep, a, beta))
                    if v3 < a:
                        return v3
                    beta = min(beta, v3)  # If you can't prune, update and contine your loop
            return v3

        # Body of minimax_decision starts here:
        # from slides
        # starting at current state, first move is pacman
        # get action score pairs to find max from
        # action, state = argmax(gameState.generateSuccessor(0,),

        act = list(filter(lambda x: x != 'Stop', gameState.getLegalActions(0)))
        score = []
        A = -9e999
        B = 9e999
        for a in act:
            sta = gameState.generateSuccessor(0, a)
            # lambda ((a, s)): min_value(s))
            # then do the min nodes, from which to find the max...
            # state, depth, agent
            vt = min_value(sta, 0, 1, A, B)
            score.append(vt)
            if vt >= B:
                return a
            A = max(A, vt)
        # find the maximum according to score
        mindx = score.index(max(score))
        action = act[mindx]

        return action





        #util.raiseNotDefined()

