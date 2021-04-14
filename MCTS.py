import logging
import math
from time import perf_counter
import time

import numpy as np

EPS = 1e-8

log = logging.getLogger(__name__)


class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Ls = {}  # stores game.getLegalMoves for board s

    def getActionProb(self, canonicalBoard, deterministic = True,maxtime=0,details=False):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            pi: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]
        """
        if maxtime != 0:
            simulations = 0
            for i in range(2):
                self.search(canonicalBoard)
                simulations+=1
            # TK if a neural network is used for mcts for the first time after loading, it takes some time
            # to be initialized. Therefore 2 simulations are performed before the timer starts to initialize
            # the neural network first.
            start_time = time.perf_counter()
            while time.perf_counter() - start_time<maxtime and simulations<self.args.numMCTSSims:
                self.search(canonicalBoard)
                simulations += 1
        else:
            for i in range(self.args.numMCTSSims):
                self.search(canonicalBoard)
            simulations = self.args.numMCTSSims

        s = self.game.stringRepresentation(canonicalBoard)
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game.getActionSize())]

        # TK when the deterministic approach is used, only the best action has a probability of 1 and all else are 0
        if deterministic == True:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            pi = [0] * len(counts)
            pi[bestA] = 1
            if details == True:
                Qs = [self.Qsa[(s,a)] if (s,a) in self.Qsa else 0 for a in range(self.game.getActionSize())]
                return pi,Qs,simulations
            return pi

        # TK otherwise all of them are distributed according to their N(s,a)
        counts_sum = float(sum(counts))
        pi = [x / counts_sum for x in counts]
        if details == True:
            Qs = [self.Qsa[(s,a)] if (s,a) in self.Qsa else 0 for a in range(self.game.getActionSize())]
            return pi,Qs,simulations
        return pi

    def search(self, canonicalBoard):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current canonicalBoard
        """

        s = self.game.stringRepresentation(canonicalBoard)


        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(canonicalBoard, 1)
        if self.Es[s] != 0:
            # terminal node
            return -self.Es[s]

        if s not in self.Ps:
            # leaf node
            self.Ps[s], v = self.nnet.predict(canonicalBoard)   
            legal = self.game.getLegalMoves(canonicalBoard, 1)
            self.Ps[s] = self.Ps[s] * legal  # masking illegal moves
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:    
                self.Ps[s] /= sum_Ps_s  # renormalize
            else:
                # if all legal moves were masked make all legal moves equally probable

                # NB! All legal moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.   
                log.error("All legal moves were masked, doing a workaround.")
                self.Ps[s] = self.Ps[s] + legal
                self.Ps[s] /= np.sum(self.Ps[s])

            self.Ls[s] = legal
            self.Ns[s] = 0
            return -v
        
        legal = self.Ls[s]
        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in range(self.game.getActionSize()):
            if legal[a]:
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + self.Ps[s][a] * math.sqrt(self.Ns[s]) / (
                            1 + self.Nsa[(s, a)])
                else:
                    u = self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)  # Q = 0 ?

                if u > cur_best:
                        cur_best = u
                        best_act = a

        a = best_act
        next_s, next_player = self.game.getNextState(canonicalBoard, 1, a)
        next_s = self.game.getCanonicalForm(next_s, next_player)

        v = self.search(next_s)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1

        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return -v
