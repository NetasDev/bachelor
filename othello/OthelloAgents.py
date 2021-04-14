import numpy as np
import math
from time import perf_counter
from .OthelloGame import OthelloGame
from MCTS import MCTS



class RandomAgent():
    def __init__(self, game):
        self.game = game

    def play(self, board,deterministic =True):
        a = np.random.randint(self.game.getActionSize())
        legal = self.game.getLegalMoves(board, 1)
        while legal[a]!=1:
            a = np.random.randint(self.game.getActionSize())
        return a
    
    def reset(self):
        return


class HumanAgent():
    def __init__(self, game):
        self.game = game
        self.name = "Human"

    def play(self, board,deterministic=True):
        # display(board)
        legal = self.game.getLegalMoves(board, 1)
        for i in range(len(legal)):
            if legal[i]:
                print("[", int(i/self.game.n), int(i%self.game.n), end="] ")
        while True:
            input_move = input()
            input_a = input_move.split(" ")
            if len(input_a) == 2:
                try:
                    x,y = [int(i) for i in input_a]
                    if ((0 <= x) and (x < self.game.n) and (0 <= y) and (y < self.game.n)) or \
                            ((x == self.game.n) and (y == 0)):
                        a = self.game.n * x + y if x != -1 else self.game.n ** 2
                        if legal[a]:
                            break
                except ValueError:
                    # Input needs to be an integer
                    'Invalid integer'
            print('Illegal move')
        return a

    def reset(self):
        return


class GreedyAgent():
    def __init__(self, game):
        self.game = game
        self.name = "Greedy"

    def reset(self):
        return

    def play(self, board,deterministic=True):
        legal = self.game.getLegalMoves(board, 1)
        candidates = []
        for a in range(self.game.getActionSize()):
            if legal[a]==0:
                continue
            nextBoard, _ = self.game.getNextState(board, 1, a)
            score = self.game.getScore(nextBoard, 1)
            candidates += [(score, a)]
        
        candidates.sort(reverse=True)
        random_indice = np.random.choice(range(len(candidates)))
        return candidates[random_indice][1]

class MinimaxAgent():
    def __init__(self,game,depth,vFunction=0,maxtime=0):
        self.maxtime = maxtime
        self.game = game
        self.depth = depth
        self.alpha = -math.inf
        self.beta = math.inf
        if vFunction==1: self.name= "Minimax matrix value"
        elif vFunction==2: self.name= "Minimax combined value"
        else: self.name = "Minimax disc value"
        self.vFunction = vFunction

    def play(self,board,deterministic=True,details = False):
        #if details is True, additionally to the selected action the action value and depth searched will be returned

        legal = self.game.getLegalMoves(board, 1)
        candidates = [] # TK candidates are a tuple of (score , action)
        depth_searched = 0

        # TK if no maxtime is set, normal minimax search is used to get candidates
        if self.maxtime == 0:
            depth_searched = self.depth
            for a in range(self.game.getActionSize()):
                if legal[a]==0:
                    continue
                nextBoard, _ = self.game.getNextState(board, 1, a)
                score = self.minimax(nextBoard,-1,self.depth,self.alpha,self.beta)
                candidates += [(score, a)]
        else:
        #TK otherwise iterative deepening is used
            start = perf_counter()
            endtime = start + self.maxtime
            for i in range(0,self.depth):
                
                temp_candidates = []
                for a in range(self.game.getActionSize()):
                    if legal[a]==0:
                        continue
                    nextBoard, _ = self.game.getNextState(board, 1, a)
                    score = self.minimax(nextBoard,-1,i,self.alpha,self.beta,endtime=endtime)
                    temp_candidates += [(score, a)]
                if perf_counter() < endtime:
                    candidates = temp_candidates
                    depth_searched = i
                else:
                    depth_searched = i-1
                    break
        
        candidates.sort(reverse=True)
        best_indices = [i for i,x in enumerate(candidates) if x[0]==candidates[0][0]]
        
        
        if deterministic == False:
            #if probabilistic approach is used, the sum of all candidates with positive action value are added.
            #then one action is picked with p(action) = action value/sum positive action values
            total, i = 0,0
            for candidate in candidates:
                if candidate[0]>0:
                    if candidate[0] == math.inf:
                        #TK if a win is guaranteed, this action is always taken
                        if details == True:
                            return (candidate[1],candidates[0],depth_searched)
                        return candidate[1]
                    else:
                        total += candidate[0]
                else:
                    break
                i = i+1
            if i != 0:
                chances  = []
                for j in range(i):
                    chances.append(candidates[j][0]/total)
                random_indice = np.random.choice(range(i),p=chances)

                if details == True:
                    return (candidates[random_indice][1],candidates[random_indice][0],depth_searched)
                return candidates[random_indice][1]
            #TK if no actions had positive action values, deterministic approach is used even before turn Threshold.

        #TK deterministic approach
        random_best_indice = np.random.choice(best_indices)
        if details == True:
            return (candidates[random_best_indice][1],candidates[random_best_indice][0],depth_searched)
        return candidates[random_best_indice][1]

    def reset(self):
        return
    
    def minimax(self,board,player,depth,alpha,beta,endtime=0):

        if endtime != 0:
            if perf_counter()>endtime:
                return 0
        legal = self.game.getLegalMoves(board,player)
        gameEnd = self.game.getGameEnded(board,1)
        if depth == 0 or gameEnd!=0:
            if gameEnd == 1:
                return math.inf #TK win
            if gameEnd == -1:
                return -math.inf #TK loose
            if gameEnd !=0: #TK draw
                return 0

            if depth == 0:
                if self.vFunction==1:
                    self.game.getMatrixValue(board,1)
                if self.vFunction==2:
                    return self.game.getCombinedValue(board,1)
                return self.game.getDiscValue(board,1)
        
        #TK Maximizing step
        if player == 1:
            score = -math.inf
            for a in range(self.game.getActionSize()):
                # TK only legal actions are considered
                if legal[a]==0:
                    continue
                nextBoard, _ = self.game.getNextState(board,1,a)
                score = max(score,self.minimax(nextBoard,-player,depth-1,alpha,beta,endtime=endtime))
                alpha = max(alpha,score)
                if alpha >= beta:
                    break
            return score

        #Tk Minimizing step
        if player == -1:
            score = math.inf
            for a in range(self.game.getActionSize()):
                # TK only legal actions are considered
                if legal[a]==0:
                    continue
                nextBoard, _ = self.game.getNextState(board,-1,a)
                score = min(score,self.minimax(nextBoard,-player,depth-1,alpha,beta,endtime=endtime))
                beta = min(beta,score)
                if beta <= alpha:
                    break
            return score

class NeuralAgent():
    def __init__(self,game,nnet,args,maxtime=0,name="Neural Agent"):
        self.game = game
        self.args = args
        self.nnet = nnet
        self.mctsStart = MCTS(game,nnet,args)
        self.mcts = MCTS(game,nnet,args)
        self.name = name
        self.maxtime = maxtime

    def reset(self):
        self.mcts = self.mctsStart


    def play(self,board,deterministic=True, details = False):
        #The neural agent gets pi from MCTS and then samples from it. If deterministic = True, the return pi will be 1 
        #in one entry and 0 otherwise.
        #If details is set to true, the action value of the chosen action and the number of simulations done is also returned

        if details:
            pi,Qs,simulations = self.mcts.getActionProb(board,deterministic=deterministic,details=True,maxtime=self.maxtime)
            choice = np.random.choice(range(len(pi)),p=pi)
            return choice, float(Qs[choice]),simulations
        else:
            prob = self.mcts.getActionProb(board,deterministic=deterministic,maxtime=self.maxtime)
            choice = np.random.choice(range(len(prob)),p=prob)
            return choice
        

    


    

        

    




