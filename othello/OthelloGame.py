from __future__ import print_function
import sys
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
from Game import Game
from othello.OthelloLogic import Board
import numpy as np
from time import perf_counter

class OthelloGame(Game):
    square_content = {
        -1: "X",
        +0: "-",
        +1: "O"
    }   

    @staticmethod
    def getSquarePiece(piece):
        return OthelloGame.square_content[piece]

    def __init__(self, n):
        self.n = n

    def getInitBoard(self):
        # return initial board (numpy board)
        b = Board(self.n)
        return np.array(b.pieces)

    def getBoardSize(self):
        # (a,b) tuple
        return (self.n, self.n)

    def getActionSize(self):
        # return number of actions
        return self.n*self.n + 1

    def getNextState(self, board, player, action):
        # if player takes action on board, return next (board,player)
        # action must be a legal move
        if action == self.n*self.n:
            return (board, -player)
        b = Board(self.n)
        b.pieces = np.copy(board)
        move = (int(action/self.n), action%self.n)
        b.execute_move(move, player)
        return (b.pieces, -player)

    def getLegalMoves(self, board, player):
        # return a fixed size binary vector
        legal = [0]*self.getActionSize()
        b = Board(self.n)
        b.pieces = np.copy(board)
        legalMoves =  b.get_legal_moves(player)
        if len(legalMoves)==0:
            legal[-1]=1
            return np.array(legal)
        for x, y in legalMoves:
            legal[self.n*x+y]=1
        return np.array(legal)

    def getGameEnded(self, board, player):
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        # player = 1
        b = Board(self.n)
        b.pieces = np.copy(board)
        if b.has_legal_moves(player):
            return 0
        if b.has_legal_moves(-player):
            return 0
        dif = b.countDiff(player)
        if  dif > 0:
            return 1
        if dif == 0:
            return -0.05
        return -1

    def getCanonicalForm(self, board, player):
        # return state if player==1, else return -state if player==-1
        return player*board

    def getSymmetries(self, board, pi):
        # mirror, rotational
        assert(len(pi) == self.n**2+1)  # 1 for pass
        pi_board = np.reshape(pi[:-1], (self.n, self.n))
        l = []

        for i in range(1, 5):
            for j in [True, False]:
                newB = np.rot90(board, i)
                newPi = np.rot90(pi_board, i)
                if j:
                    newB = np.fliplr(newB)
                    newPi = np.fliplr(newPi)
                l += [(newB, list(newPi.ravel()) + [pi[-1]])]
        return l

    def stringRepresentation(self, board):
        return board.tostring()

    def stringRepresentationReadable(self, board):
        board_s = "".join(self.square_content[square] for row in board for square in row)
        return board_s

    def getScore(self, board, player):

        b = Board(self.n)
        b.pieces = np.copy(board)
        return b.countDiff(player)

    def action_to_move(self,action):
        if action == None:
            return None
        if int(action/self.n)>=self.n:
            return (self.n,0)
        move = (int(action/self.n), action%self.n)
        return move

    def move_to_action(self,move):
        action = self.n*move[0]+move[1]
        return action


######################################################################################################
    #TK the following functions are the value functions and other functions used by the value functions.
    #For most of them, the bachelor paper explained how they are calculated.

    def getCombinedValue(self,board,player):
        player_moves = self.getLegalMoves(board,player)
        opponent_moves = self.getLegalMoves(board,-player)

        vdisc = self.getDiscValue(board,player)
        vmob  = self.getMobilityValue(board,player,player_moves,opponent_moves)
        vcorner = self.getCornerValue(board,player,player_moves,opponent_moves)
        vstability = self.getStabilityValue(board,player)
        #print("coin: "+str(vdisc)+" mobility: "+str(vmob)+ " corner: "+ str(vcorner) + " stability: "+ str(vstability))

        return 0.25*(vdisc+vmob+vcorner+vstability)

    def get_corner_score(self,board,player,legalMoves):
        score = 0
        corner_squares = ((0,0),(0,self.n-1),(self.n-1,0),(self.n-1,self.n-1))
        for x,y in corner_squares:
            if board[x][y]==player:
                score += 1
            if legalMoves[self.move_to_action((x,y))] == 1:
                score += 0.5
        return score

    def getCornerValue(self,board,player,player_moves,opponent_moves):

        player_corner_score = self.get_corner_score(board,player,player_moves)
        opponent_corner_score = self.get_corner_score(board,player*-1,opponent_moves)
        if(player_corner_score+opponent_corner_score)!=0:
            return (player_corner_score-opponent_corner_score)/(player_corner_score+opponent_corner_score)
        return 0

    def getDiscValue(self,board,player):
        player_score = sum(board[board==player])
        opponent_score = -sum(board[board==-player])
        return (player_score-opponent_score)/(player_score+opponent_score)

    def getMobilityValue(self,board,player,player_moves,opponent_moves):
        pamv = len(player_moves[player_moves==1])
        oamv = len(opponent_moves[opponent_moves==1])

        empty_neighbours = []
        if len(np.argwhere(board==-player))>0:
            for pos in np.argwhere(board==-player):
                empty_neighbours += self.get_empty_neighbours(board,pos)
                unique_empty_neighbours = set(empty_neighbours)
            ppmv = len(unique_empty_neighbours)
        else:
            ppmv = 0
        empty_neighbours = []
        if len(np.argwhere(board==player))>0:
            for pos in np.argwhere(board==player):
                empty_neighbours += self.get_empty_neighbours(board,pos)
                unique_empty_neighbours = set(empty_neighbours)
            opmv = len(unique_empty_neighbours)
        else:
            opmv = 0
        if  ((pamv +0.5*ppmv) + (oamv +0.5*opmv))!=0:
            value = ((pamv +0.5*ppmv) - (oamv +0.5*opmv)) / ((pamv +0.5*ppmv) + (oamv +0.5*opmv))
        else:
            value = 0
        return value

    def get_empty_neighbours(self,board,position):
        i = position[0]
        j = position[1]
        empty_neighbours = []
        for x in range(max(0,i-1),min(i+2,self.n)):
            for y in range(max(0,j-1),min(j+2,self.n)):
                if (x != i or y !=j):
                    if board[x][y]==0:
                        empty_neighbours.append((x,y))
        return empty_neighbours

            
    def getEdgeStabilityMatrix(self,board,player):
        ## TK returns a matrix with 1 on every position of a stable edge disc

        n = self.n
        corners = ((0,0),(n-1,0),(n-1,n-1),(0,n-1))
        upper_edge = []
        lower_edge = []
        left_edge = []
        right_edge = []

        for i in range(1,n-1):
            upper_edge.append((i,0))
            right_edge.append((n-1,i))
            lower_edge.append((i,n-1))
            left_edge.append((0,i))
            
        edges = [upper_edge,right_edge,lower_edge,left_edge]
        alignment = [(1,0),(0,1),(1,0),(0,1)]
        stability_matrix = np.zeros((n,n))
        for corner in corners:
            if board[corner[0]][corner[1]]!=0:
                stability_matrix[corner[0]][corner[1]] = 1
        
        i = 0
        for edge in edges:
            temp_list = edge+ [corners[i%4]] +[corners[(i+1)%4]]
            full_row = True
            #TK if every square on an edge is filled, all coins on these squares are stable
            for square in (temp_list):
                if board[square[0]][square[1]]==0:
                    full_row = False
                    break
            if full_row:
                for square in (temp_list):
                    stability_matrix[square[0]][square[1]]=1

            #TK Discs next to stable discs of the same color are stable aswell
            #To check this, the squares of the edge are checked one after another in one direction.
            #Each square is compared to the previous one and if the previous one has the same color
            #and is stable, this disc is also stable.
            #The same thing is than done for the over direction. Once no more discs are changed from
            #unstable to stable, the loop ends.
            changed = True
            while changed == True:
                changed = False
                for field in edge:
                    prev_field = (field[0]-alignment[i][0],field[1]-alignment[i][1])
                    if stability_matrix[field[0]][field[1]]==0:
                        if board[prev_field[0]][prev_field[1]]!=0 and board[prev_field[0]][prev_field[1]] == board[field[0]][field[1]]:
                            if stability_matrix[prev_field[0]][prev_field[1]]==1:
                                stability_matrix[field[0]][field[1]] = 1
                                changed = True
                for field in edge:
                    prev_field = (field[0]+alignment[i][0],field[1]+alignment[i][1])
                    if stability_matrix[field[0]][field[1]]==0:
                        if board[prev_field[0]][prev_field[1]]!=0 and board[prev_field[0]][prev_field[1]] == board[field[0]][field[1]]:
                            if stability_matrix[prev_field[0]][prev_field[1]]==1:
                                stability_matrix[field[0]][field[1]] = 1
                                changed = True
            i +=1
        return stability_matrix

    def getStabilityValue(self,board,player):
        edge_stability_matrix = self.getEdgeStabilityMatrix(board,player)
        stable_coins = board*edge_stability_matrix
        player_stable_coins = sum(stable_coins[stable_coins==player])
        opponent_stable_coins = -sum(stable_coins[stable_coins==-player])

        if(player_stable_coins+opponent_stable_coins)!=0:
            return (player_stable_coins-opponent_stable_coins)/(player_stable_coins+opponent_stable_coins)
        return 0


    def getMatrixValue(self,board,maxplayer):
        if self.n == 8:
            weights = np.array(([ 4,-3, 2, 2, 2, 2,-3, 4],
                                [-3,-4,-1,-1,-1,-1,-4,-3],
                                [ 2,-1, 1, 0, 0, 1,-1, 2],
                                [ 2,-1, 0, 1, 1, 0,-1, 2],
                                [ 2,-1, 0, 1, 1, 0,-1, 2],
                                [ 2,-1, 1, 0, 0, 1,-1, 2],
                                [-3,-4,-1,-1,-1,-1,-4,-3],  
                                [ 4,-3, 2, 2, 2, 2,-3, 4]))
            if maxplayer==1:
                return np.sum(weights*board)/112
            return -np.sum(weights*board)/112
            # 112 is the maximum difference possible with 56 for player 1 and -56 for player 2
        if self.n == 6:
            weights = np.array(([ 5,-3, 3, 3,-3, 5],
                                [-3,-4,-1,-1,-4,-3],
                                [ 3,-1, 1, 1,-1, 3],
                                [ 3,-1, 1, 1,-1, 3],
                                [-3,-4,-1,-1,-4,-3],  
                                [ 5,-3, 3, 3,-3, 5]))
            if maxplayer==1:
                return np.sum(weights*board)/96
            return -np.sum(weights*board)/96

            # 96 is the maximum difference possible with 48 for player 1 and -48 for player 2
