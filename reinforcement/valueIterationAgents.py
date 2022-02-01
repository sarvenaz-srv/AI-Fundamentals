# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


from ctypes import sizeof
import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
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
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for i in range(self.iterations):
            tempvalues=util.Counter()
            states = self.mdp.getStates()
            for state in states:
                actions = self.mdp.getPossibleActions(state)
                maxValue = - 999999
                for action in actions:
                    sumValue = 0.0
                    transitions = self.mdp.getTransitionStatesAndProbs(state, action)
                    for t in transitions:
                        sumValue += t[1] * (self.mdp.getReward(state, action, t[0])+self.discount*self.values[t[0]])
                    if(sumValue>maxValue):
                        maxValue = sumValue
                if maxValue != -999999:
                    tempvalues[state] = maxValue

            for state in states:
                self.values[state] = tempvalues[state]


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
        transitions = self.mdp.getTransitionStatesAndProbs(state,action)
        sumValue = 0.0
        for t in transitions:
            sumValue+= t[1]*(self.mdp.getReward(state, action, t[0]) + self.discount*self.values[t[0]])
        return sumValue
        #util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        actions  = self.mdp.getPossibleActions(state)
        maxValue = -999999
        argMax = None
        for action in actions:
            val = self.computeQValueFromValues(state, action)
            if val>maxValue:
                maxValue = val
                argMax = action
        return argMax
        # util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        count = len(states)
        for i in range(self.iterations):
            state = states[i%count]
            actions = self.mdp.getPossibleActions(state)
            maxValue = -999999
            for action in actions:
                transitions = self.mdp.getTransitionStatesAndProbs(state,action)
                sumValue = 0.0
                for t in transitions:
                    sumValue += t[1] * (self.mdp.getReward(state, action, t[0]) + self.discount * self.values[t[0]])
                if sumValue>maxValue:
                    maxValue = sumValue
            if maxValue != -999999:
                self.values[state] = maxValue


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        pq = util.PriorityQueue()
        states = self.mdp.getStates()
        predecessors = dict()
        for state in states:
            if not self.mdp.isTerminal(state):
                actions = self.mdp.getPossibleActions(state)
                for action in actions:
                    transitions = self.mdp.getTransitionStatesAndProbs(state, action)
                    for t in transitions:
                        if not t[0] in predecessors:
                            predecessors[t[0]] = set()
                        predecessors[t[0]].add(state)
        
        for state in states:
            if( not self.mdp.isTerminal(state)):
                actions = self.mdp.getPossibleActions(state)
                maxValue = -999999
                for action in actions:
                    val = self.getQValue(state,action)
                    if val > maxValue:
                        maxValue = val
                diff = abs(maxValue - self.values[state])
                pq.push(state, -diff)
        
        for i in range(self.iterations):
            if not pq.isEmpty():
                s = pq.pop()
                if not self.mdp.isTerminal(s):
                    value = -999999
                    actions = self.mdp.getPossibleActions(s)
                    for action in actions:
                        value = max(value, self.getQValue(s, action))
                    self.values[s] = value
                    for p in predecessors[s]:
                        if not self.mdp.isTerminal(p):
                            actions = self.mdp.getPossibleActions(p)
                            maxValue = -999999
                            for action in actions:
                                val = self.getQValue(p, action)
                                if val > maxValue:
                                    maxValue = val
                            diff = abs(maxValue - self.values[p])
                            if diff > self.theta:
                                pq.update(p, -diff)


                

