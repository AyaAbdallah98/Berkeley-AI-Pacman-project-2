# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from math import inf
from util import Distance
import pacman

import math

def Distance(xy1, xy2):
    "Returns the Manhattan distance between points xy1 and xy2"
    return math.sqrt(((xy1[0] - xy2[0]))**2 + ((xy1[1] - xy2[1]))**2)

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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        "*** YOUR CODE HERE ***"
        # Useful information you can extract from a GameState (pacman.py)
        from util import manhattanDistance
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        newGhostPositions = successorGameState.getGhostPositions()
        #print(successorGameState)
        #print(newPos)
        #print(newFood)
        #print(newGhostStates)
        #print(newScaredTimes)
        #print(newGhostPositions[0])

        #the distance between the new position and the available food
        eval1 = 0
        dist_food = [Distance(newPos, food) for food in newFood.asList()]
        #print(dist_food)
        if len(dist_food):
            eval1 = 4/min(dist_food)      #to get the nearest food first

        #put penality if pacman stop to prevent random stopping
        if action == Directions.STOP:
            eval1 -= 8

        #distance from the ghost
        eval2 = 0
        minx = inf
        for i in range(len(newGhostStates)):
           dist_ghost = Distance(newPos, newGhostStates[i].getPosition())
           if dist_ghost == newPos:  # will reach the ghost
            eval2 += dist_ghost
           elif dist_ghost < minx:    #to take into consideration the nearest ghost only in case of multiple ghosts
             minx = dist_ghost
        dist_ghost = minx
        if dist_ghost < 5:
          eval2 +=dist_ghost
        else:
            eval2 +=5

        #num_food = len(newFood.asList())
        for i in newScaredTimes:
          if  i == 0:
            score = eval1 + 2*eval2 +  successorGameState.getScore()
          elif i != 0:
            score =eval1 + successorGameState.getScore()
        return score

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
from math import inf
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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        '''Minimax implementation.'''
        def max_value(depth, gameState):
            if depth == self.depth or gameState.isLose() or gameState.isWin():
                return self.evaluationFunction(gameState)
            max_num = -inf
            for action in gameState.getLegalActions(0):
                max_num = max(max_num, min_value(depth, gameState.generateSuccessor(0, action),1))
            return max_num

        def min_value(depth, gameState,agentIndex):
            if depth == self.depth or gameState.isLose() or gameState.isWin():
                return self.evaluationFunction(gameState)
            min_num = +inf
            if agentIndex == gameState.getNumAgents()-1:
                depth+=1
            for action in gameState.getLegalActions(agentIndex):
                if agentIndex < gameState.getNumAgents()-1:
                  min_num = min(min_num, min_value(depth, gameState.generateSuccessor(agentIndex, action), agentIndex+1))
                else:
                  min_num = min(min_num, max_value(depth, gameState.generateSuccessor(agentIndex, action)))
            return min_num

        best_action, current_value = None, None
        for action in gameState.getLegalActions(0):
            action_value = min_value(0, gameState.generateSuccessor(0, action),1)
            if current_value is None or current_value < action_value:
               best_action = action
               current_value = action_value
        return best_action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def max_value(depth, gameState, alpha, beta):
            if depth == self.depth or gameState.isLose() or gameState.isWin():
                return self.evaluationFunction(gameState)
            max_num = -inf
            for action in gameState.getLegalActions(0):
                max_num = max(max_num, min_value(depth, gameState.generateSuccessor(0, action),1, alpha, beta))
                if max_num > beta: return max_num
                alpha = max(alpha, max_num)
            return max_num

        def min_value(depth, gameState,agentIndex, alpha, beta):
            if depth == self.depth or gameState.isLose() or gameState.isWin():
                return self.evaluationFunction(gameState)
            min_num = +inf
            if agentIndex == gameState.getNumAgents()-1:
                depth+=1
            for action in gameState.getLegalActions(agentIndex):
                if agentIndex < gameState.getNumAgents()-1:
                  min_num = min(min_num, min_value(depth, gameState.generateSuccessor(agentIndex, action), agentIndex+1, alpha, beta))
                else:
                  min_num = min(min_num, max_value(depth, gameState.generateSuccessor(agentIndex, action), alpha, beta))
                if min_num < alpha: return min_num
                beta = min(beta, min_num)
            return min_num

        alpha = -inf
        beta = inf
        best_action, current_value = None, None
        for action in gameState.getLegalActions(0):
            action_value = min_value(0, gameState.generateSuccessor(0, action),1, alpha, beta)
            if current_value is None or current_value < action_value:
               best_action = action
               current_value = action_value

            if action_value > beta:
                return best_action
            alpha = max(alpha, action_value)
        return best_action


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
        def max_value(depth, gameState):
            if depth == self.depth or gameState.isLose() or gameState.isWin():
                return self.evaluationFunction(gameState)
            max_num = -inf
            for action in gameState.getLegalActions(0):
                max_num = max(max_num, min_value(depth, gameState.generateSuccessor(0, action),1))
            return max_num

        def min_value(depth, gameState,agentIndex):
            if depth == self.depth or gameState.isLose() or gameState.isWin():
                return self.evaluationFunction(gameState)

            if agentIndex == gameState.getNumAgents()-1:
                depth+=1
            allactions = gameState.getLegalActions(agentIndex)
            if len(allactions) != 0:
                prop = 1.0 / len(allactions)
            min_num = 0
            for action in gameState.getLegalActions(agentIndex):
                if agentIndex < gameState.getNumAgents()-1:
                  min_num += prop*min_value(depth, gameState.generateSuccessor(agentIndex, action), agentIndex+1)
                else:
                  min_num += prop*max_value(depth, gameState.generateSuccessor(agentIndex, action))
            return min_num
        allactions = gameState.getLegalActions(1)
        if len(allactions) != 0:
            prop = 1.0 / len(allactions)
        best_action, current_value = None, None
        for action in gameState.getLegalActions(0):
            action_value = prop*min_value(0, gameState.generateSuccessor(0, action),1)
            if current_value is None or current_value < action_value:
               best_action = action
               current_value = action_value
        return best_action


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    current_pos = currentGameState.getPacmanPosition()
    current_food = currentGameState.getFood()
    ghost_pos = currentGameState.getGhostPositions()
    currentGhostStates = currentGameState.getGhostStates()
    currentScaredTimes = [ghostState.scaredTimer for ghostState in currentGhostStates]
    # the distance between the new position and the available food
    eval1 = 0
    dist_food = [Distance(current_pos, food) for food in current_food.asList()]
    # print(dist_food)
    if len(dist_food):
        eval1 = 4 / min(dist_food)  # to get the nearest food first

    #power pellets have a big contribution
    num_pellets = len(currentGameState.getCapsules())
    all_scared = currentScaredTimes

    # distance from the ghost
    eval2 = 0
    minx = inf
    for i in range(len(ghost_pos)):
        dist_ghost = Distance(current_pos, ghost_pos[i])
        if dist_ghost == current_pos:  # will reach the ghost
            eval2 += dist_ghost
        elif dist_ghost < minx:  # to take into consideration the nearest ghost only in case of multiple ghosts
            minx = dist_ghost
    dist_ghost = minx
    if dist_ghost < 5:
        eval2 += dist_ghost
    else:
        eval2 += 5

    # num_food = len(newFood.asList())
    for i in currentScaredTimes:
        if i == 0:
            score = eval1 - num_pellets + currentScaredTimes[0] + 2*eval2 + currentGameState.getScore()
        elif i != 0:
            score = eval1 - num_pellets + currentScaredTimes[0] + currentGameState.getScore()
    return score


# Abbreviation
better = betterEvaluationFunction
