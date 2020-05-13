# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called
by Pacman agents (in searchAgents.py).
"""

# ANA COMMENT: this is the final version of search.py where autograder gives 100% on questions 1- 4.
# All done with search.py


import util
#import searchAgents


# create a helper class Node (as in, Node in the Search *tree*, not the state from the original problem).
# This Node can keep track of the parent Node (guaranteed to be only one, since you are constructing a search *tree*,
# like i did on the board or in slides). Node could also keep track of actions, cumulative costs, and whatever else you
# need.
# The idea is to later use this Node object to help you recover the path (list of actions) all the way back to the start
# state.
class Node:
    def __init__(self, state, action, cost, parent):
        self.actions = []  # e.g., to store full sequence of actions to get to node from root
        self.actionFromParent = action
        self.cost = cost
        self.state = state
        self.parent = parent
        self.depth = 0
        self.path_cost = cost
        self.corners = {}
        # more code needed here to maintain actions. need special case for root
        # for root the parent should be null
        if parent is not None:
            # get the actions that the parent has done to get to where it is, and append your own action
            self.actions.extend(parent.getActions())
            self.actions.append(action)
            # more code needed here to maintain cost, depth, etc. (see lecture slides)
            # for cost add the parent's cost in order to have the total cost
            self.path_cost += parent.path_cost
            self.depth = parent.depth + 1

    def getCost(self):
        return self.path_cost

    def getState(self):
        return self.state

    def getActions(self):
        # code here to compute path back to root
        return self.actions

        # can have other useful methods, like cost updates, support methods for hashing, etc (see lecture slides)

    def __eq__(self, other):  # same states treated as same node
        return isinstance(other, Node) and self.state == other.state

    def __hash__(self):
        return hash(self.state)

    def path(self):
        """Return a list of nodes forming the path from the root to this node."""
        node, path_back = self, []
        while node:
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))

    def solution(self):
        """Return the sequence of actions to go from the root to this node."""
        return [node.action for node in self.path()[1:]]

    def expand(self, problem):
        """List the nodes reachable in one step from this node."""
        children = []
        #print "in expand for", self.getState()
        temp = problem.getSuccessors(self.state) # looks good
        #print "get succ", temp
        for child in temp:
            node = child[0]
            move = child[1]
            cost = child[2] #+ self.getCost()
            #print "n, m, c", node,move, cost
            # (self, state, action, cost, parent):
            child = Node(node, move, cost, self)
            #print child.getActions()
            children.append(child)
        return children



# noinspection PyClassHasNoInit
class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()


# New by me
class cornState:
    """
  holds info for the corner problem...
  """
    #expect a tuple and dictionary

    def __init__(self, position, foundCorn):
        self.pos = position
        self.found = foundCorn

    def __eq__(self, other):
        if other == None:
            return False
        return (self.pos == other.pos) & (cmp(self.found, other.found) == 0)

    def __hash__(self):
        tup = tuple(self.found.keys())
        return hash((self.pos, tup))

    def __getitem__(self, idx):
        return tuple.__getitem__((self.pos, self.found),idx)

    def __len__(self):
        return len(self.found)


    def copy(self):
        state = cornState(self.pos, self.found)
        return state

    def getPosition(self):
        if self.pos == None: return None
        return self.pos  # not sur what to do of this...

    def getFound(self):
        if self.found == None: return {}
        return self.found











def conerStateHandeler(state):
    # this is to handle the way state works in the corners problem
    # it gives the position, particularly for the dictionary
        ty = type(state[0])
        test = type((5,5))
        test2 = type({})
        if  ty == test :
            if type(state[1]) == test2:
                # print "corn"
                return cornState(state[0],state[1])
        else:
            return state

def conerStateGoal(state):
    # this is to handle the way state works in the corners problem
    # it gives the position, particularly for the dictionary
        ty = type(state[0])
        test = type((5,5))
        test2 = type({})
        if  ty == test :
            if type(state[1]) == test2:
                return state[1]
        else:
            return state


def graphSearch(problem, fringe):
    # tree search plus the sat of already visited states
    # store the closed set as a set
    closed = {}  # visited states
    # node initalized as (self, state, action, cost, parent)
    fringe.push(Node(problem.getStartState(), [], 1, None))


    while not fringe.isEmpty():  # while not empty
        node = fringe.pop()
        # print "pop. fringe :", len(fringe.list)
        state = node.getState()
        # print state
        control = conerStateHandeler(state)

        # print node.getState()
        if problem.isGoalState(node.getState()):
            # print "found win! "
            # print node.getActions()
            # print len(fringe.list)
            return node.getActions()
        #print state

        if control not in closed :
            closed[control] = 1
        else:
            continue
        children = node.expand(problem)
        #print "did expand"

        for child in children:
            #print "graph search ",child.getState()
            #print "push ",child, "fringe size ", len(fringe.list)
            fringe.push(child)
    #print "I did graph search and failed :( "
    #print len(fringe.list)
    return []


def graphSearchP(problem, fringe):
    # now you don't care about repeated paths
    # node initialized as (self, state, action, cost, parent)
    # push is push(self, item, priority)
    # this is the root, so it may as well be free...
    closed = {}  # set so closed defaults to expensive
    inHeap = {}  # the algorithim wants to keep track of wwho is still on the heap
    stateS = problem.getStartState()


    fringe.push(Node(stateS, [], 1, None), 0)
    inHeap[problem.getStartState()] = 0

    while not fringe.isEmpty():  # while not empty
        node = fringe.pop()
        control = node.getState()
        if problem.isGoalState(node.getState()):
            # print "found win! ", node.getActions()
            # this seems to fail when there are multiple dots, and ghosts
            return node.getActions()
        # after pop you are no longer unvisited
        closed[control] = 1
        # after pop you are off the heap, so remove from dict
        if control in inHeap:
            del closed[control]
        else:
            continue
        children = node.expand(problem)
        for child in children:
            childCost = problem.getCostOfActions(child.getActions())
            childControl = child.getState()
            if childControl not in inHeap and childControl not in closed:
                # if child not in fringe:
                fringe.push(child, childCost)  # so I had a super anoying bug in that
                # I had been using node to keep track of costs, from over looking how this would stop the agent's cost
                # functions from being able to be used... :( annoying
                inHeap[childControl] = childCost
                # else if is more expensive, replace
            elif childControl in inHeap and childCost < inHeap[childControl]:
                inHeap[childControl] = childCost
                fringe.push(child, childCost)

                # print "I did graph search and failed :( "
    return []  # Keeps saying has to return list


def graphSearchA(problem, fringe, fcost):
    # now you don't care about repeated paths
    # node initialized as (self, state, action, cost, parent)
    # push is push(self, item, priority)
    # this is the root, so it may as well be free...
    closed = {}  # set so closed defaults to expensive
    inHeap = {}  # the algorithim wants to keep track of wwho is still on the heap
    # this returns a corn type, whish is hashable
    control = conerStateHandeler(problem.getStartState())
    print "control ", control, problem.getStartState()

    fringe.push(Node(problem.getStartState(), [], 1, None), 0)
    inHeap[control] = 0

    while not fringe.isEmpty():  # while not empty
        node = fringe.pop()
        if problem.isGoalState(node.getState()):
            # print "found win! ", node.getActions()
            # this seems to fail when there are multiple dots, and ghosts
            return node.getActions()
        # after pop you are no longer unvisited
        control = conerStateHandeler(node.getState())
        closed[control] = 1
        # after pop you are off the heap, so remove from dict
        if control in inHeap:
            del closed[control]
        else:
            continue
        children = node.expand(problem)
        for child in children:
            # basically the exact same as ucs, just now has a huristic added to cost
            childCost = problem.getCostOfActions(child.getActions()) + fcost(child)
            if child.getState() not in inHeap and child.getState() not in closed:
                # if child not in fringe:
                fringe.push(child, childCost)  # so I had a super anoying bug in that
                # I had been using node to keep track of costs, from over looking how this would stop the agent's cost
                # functions from being able to be used... :( annoying
                inHeap[child.getState()] = childCost
                # else if is more expensive, replace
            elif child.getState() in inHeap and childCost < inHeap[child.getState()]:
                inHeap[child.getState()] = childCost
                fringe.push(child, childCost)

                # print "I did graph search and failed :( "
    return []  # Keeps saying has to return list


def graphSearchAOld(problem, fringe):
    # now you don't care about repeated paths
    # node initialized as (self, state, action, cost, parent)
    # push is push(self, item, priority)
    # this is the root, so it may as well be free...
    closed = {}
    fringe.push(Node(problem.getStartState(), [], 1, None))
    # Don't worry about repeated paths to a state.
    while not fringe.isEmpty():  # while not empty
        node = fringe.pop()
        if problem.isGoalState(node.getState()):
            # print "found win! ", node.getActions()
            return node.getActions()
        if node.getState() not in closed:
            closed[node.getState()] = 1
        else:
            continue
        children = node.expand(problem)
        for child in children:
            if child.getState() not in closed:
                fringe.push(child)
    # print "I did graph search and failed :( "
    return None


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other
    maze, the sequence of moves will be incorrect, so only use this for tinyMaze
    """
    from game import Directions
    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first

    Your search algorithm needs to return a list of actions that reaches
    the goal.  Make sure to implement a graph search algorithm

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())

    Start: (5, 5)
    Is the start a goal? False
    Start's successors: [((5, 4), 'South', 1), ((4, 5), 'West', 1)]


    """
    "*** YOUR CODE HERE ***"
    from util import Stack

    # create  a dictionary or set  where   you add states(as keys) that you already  explored.
    # Explored means  you retrieved  its successors
    # (self, state, action, cost, parent)
    # getting the root
    # graph search with a stack
    fringe = Stack()
    ans = graphSearch(problem, fringe)
    return ans
    # after graph search returns the path, get the directions


def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.
    """
    "*** YOUR CODE HERE ***"
    from util import Queue
    # very similar to DFS, but the data structure is now a que
    # root = problem.getStartState()
    # getting the root
    # graph search with a stack
    fringe = Queue()
    ans = graphSearch(problem, fringe)
    return ans


def uniformCostSearch(problem):
    """Search the node of least total cost first. """
    "*** YOUR CODE HERE ***"
    # very similar to the above, but now with priority que
    from util import PriorityQueue
    # very similar to DFS, but the data structure is now a que
    # getting the root
    # graph search with a priority que
    fringe = PriorityQueue()
    ans = graphSearchP(problem, fringe)
    return ans


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    from util import PriorityQueue

    # def heuristic(n):
    # return None
    # print heuristic(problem.getStartState(), problem)

    def fcost(n):
        return heuristic(n.getState(), problem)

    # fringe = util.PriorityQueueWithFunction(fcost)
    fringe = PriorityQueue()
    # I think my node implimentation was not letting the cost function work...
    ans = graphSearchA(problem, fringe, fcost)
    return ans


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
