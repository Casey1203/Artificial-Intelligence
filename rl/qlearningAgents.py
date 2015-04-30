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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

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
		self.q_val = util.Counter()#maintain a dict type qValue, key is (state, action) pair


	def getQValue(self, state, action):
		"""
		  Returns Q(state,action)
		  Should return 0.0 if we have never seen a state
		  or the Q node value otherwise
		"""
		return self.q_val[(state, action)]

	def computeValueFromQValues(self, state):
		"""
		  Returns max_action Q(state,action)
		  where the max is over legal actions.  Note that if
		  there are no legal actions, which is the case at the
		  terminal state, you should return a value of 0.0.
		"""
		#return max Qvalue
		if len(self.getLegalActions(state)) == 0:
			return 0.0
		maxQ = float("-inf")
		for action in self.getLegalActions(state):
			maxQ = max(maxQ, self.getQValue(state, action))
		return maxQ

	def computeActionFromQValues(self, state):
		"""
		  Compute the best action to take in a state.  Note that if there
		  are no legal actions, which is the case at the terminal state,
		  you should return None.
		"""
		"*** YOUR CODE HERE ***"
		if len(self.getLegalActions(state)) == 0:
			return None
		bestAction = []
		maxQ = float("-inf")
		for action in self.getLegalActions(state):
			if self.getQValue(state, action) > maxQ:
				maxQ = self.getQValue(state, action)
				bestAction = [action]
			elif self.getQValue(state, action) == maxQ:
				bestAction.append(action)# many action may cause the Q value max, just randomly choose one
		return random.choice(bestAction)

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
		if len(legalActions) == 0:# no legal action
			return None
		if util.flipCoin(self.epsilon):#random
			return random.choice(legalActions)
		else:#optimal
			return self.computeActionFromQValues(state)
		return action

	def update(self, state, action, nextState, reward):
		"""
		  The parent class calls this to observe a
		  state = action => nextState and reward transition.
		  You should do your Q-Value update here

		  NOTE: You should never call this function,
		  it will be called on your behalf
		"""
		# update one qValue under certain state,action when this pair comes
		sample = reward + self.discount * self.computeValueFromQValues(nextState)
		self.q_val[(state, action)] = (1 - self.alpha) * self.getQValue(state, action) + self.alpha * sample


	def getPolicy(self, state):
		return self.computeActionFromQValues(state)

	def getValue(self, state):
		return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
	"Exactly the same as QLearningAgent, but with different default parameters"

	def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
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
		self.weights = util.Counter()#don't change qValue as before, but weights, which indirectly change qValue

	def getWeights(self):
		return self.weights

	def getQValue(self, state, action):
		"""
		  Should return Q(state,action) = w * featureVector
		  where * is the dotProduct operator
		"""
		weight = self.getWeights()
		featureVector = self.featExtractor.getFeatures(state, action)
		return self.weights * featureVector
	def update(self, state, action, nextState, reward):
		"""
		   Should update your weights based on transition
		"""
		difference = (reward + self.discount * self.computeValueFromQValues(nextState)) - self.getQValue(state, action)
		feature = self.featExtractor.getFeatures(state, action)#f(s,a)

		#self.weights = self.getWeights() + self.alpha * difference * feature
		for key in feature.keys():# update the value of weight whose key satisfy that it is in the feature vector.
			self.weights[key] = self.getWeights()[key] + self.alpha * difference * feature[key]

	def final(self, state):
		"Called at the end of each game."
		# call the super-class final method
		PacmanQAgent.final(self, state)

		# did we finish training?
		if self.episodesSoFar == self.numTraining:
			# you might want to print your weights here for debugging
			"*** YOUR CODE HERE ***"
			pass
