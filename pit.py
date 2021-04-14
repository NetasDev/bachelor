import Arena
import numpy as np
from MCTS import MCTS
from othello.OthelloGame import OthelloGame
from othello.keras.NNet import NNetWrapper as nn
from othello.OthelloInteractiveBoard import InteractiveBoard
from othello.OthelloLogic import Board
from othello.OthelloAgents import *
import pygame
import os
import wandb

import numpy as np
from utils import *


"""
TK all the comments and explanations in this class were written by me.
"""

"""
TK To play a game of Othello, a OthelloGame object has to be created. 
The parameter of this game is the desired board size.
"""
game6 = OthelloGame(6)
game8 = OthelloGame(8)
"""
TK Agents have to be created to play the game. All agents require as parameter the game created.
These agents can be:
A human agent if someone wants to play against other agents themselves
"""
human6 = HumanAgent(game6)
human8 = HumanAgent(game8)
"""
Tk The minimax agent needs to be given a depth.
Optionally a vfunction can be chosen. Vfunction has to be
1 matrix value, 2 for combined value or anything else for disc value, which is 
also the default setting.

A maxtime in seconds can be set to instead perform minimax for a given time frame.
If a maxtime is set, the set depth is instead used as a maximum depth to which can be searched.
"""

minimaxDV6 = MinimaxAgent(game6,2)
minimaxDV8 = MinimaxAgent(game8,2)
minimaxMV8 = MinimaxAgent(game8,2,vFunction=1)
minimaxCV8 = MinimaxAgent(game8,2,vFunction=2)


minimaxDV8_timed = MinimaxAgent(game8,100,maxtime=1)

"""
TK In Order to create a neural Network player with MCTS a NNetWrapper has to be created and this wrapper has to load the net 
from the target folder with the given filename.
Also a dotdict including atleast numMCTSSims (the number of MCTS Simulations done each turn)
has to be created. A maxtime can be given aswell in which case MCTS simulations are performed as long as the time allows
or at max numMCTSSims are done.

The trained networks are in ./trained network/
"""
network6x6 = nn(game6)
network6x6.load_checkpoint(folder="./trained network/6x6/24h/",filename="weights")
args = dotdict({'numMCTSSims': 50})
neural6x6 = NeuralAgent(game6,network6x6,args)

network8x8 = nn(game8)
network8x8.load_checkpoint(folder="./trained network/8x8/24h/",filename="weights")
args = dotdict({'numMCTSSims': 50})
neural8x8 = NeuralAgent(game8,network8x8,args)

args_timed = dotdict({'numMCTSSims': 5000})
neural8x8_timed = NeuralAgent(game8,network8x8,args,maxtime=1)
"""
Tk Once the Agent are created, they can be matched against each other 1vs1 by creating an arena Object
This Object needs both players and the game to be played to be initialized.
Additionally a number the turnTreshhold can be set here make the players choose an action probabilisticly for the first turns 
and always choosing the best calculated action after the turnThreshold is exceeded.
"""
#arena = Arena.Arena(neural6x6,minimaxDV6,game6,turnThreshold=8)
"""
TK Afterwards the function playGames can be used to play X games between the two players.
X has to be a multiple of 2
The variable Display can be set to True to have an interactive board shown on screen.
This Board shows the current board, the possible moves of the current player und more information about the match.
A save folder can be set to save the games in a folder. (If there already are games in that folder, the new games are added)
"""

"""
player1wins,player2wins,draws = arena.playGames(4,display=True)
print("player 1 wins: "+ str(player1wins))
print("player 2 wins: "+ str(player2wins))
print("draws : " +str(draws))
"""

"""
# 8x8 board against minimax with disc value
arena = Arena.Arena(neural8x8,minimaxDV8,game8,turnThreshold=15)
player1wins,player2wins,draws = arena.playGames(4,display=False)
print("player 1 wins: "+ str(player1wins))
print("player 2 wins: "+ str(player2wins))
print("draws : " +str(draws))
"""
"""
# 8x8 board with display off and against minimax with matrix value
arena = Arena.Arena(neural8x8,minimaxMV8,game8,turnThreshold=15)
player1wins,player2wins,draws = arena.playGames(4,display=False)
print("player 1 wins: "+ str(player1wins))
print("player 2 wins: "+ str(player2wins))
print("draws : " +str(draws))
"""

"""
# Timed agents on 8x8 board and save in /test_games/ folder
arena = Arena.Arena(neural8x8_timed,minimaxDV8_timed,game8,turnThreshold=15)
player1wins,player2wins,draws = arena.playGames(4,display=True,save="./test_games/")
print("player 1 wins: "+ str(player1wins))
print("player 2 wins: "+ str(player2wins))
print("draws : " +str(draws))
"""
"""
# To play against the neural agent yourself on an 8x8 board
arena = Arena.Arena(human8,neural8x8_timed,game8,turnThreshold=15)
player1wins,player2wins,draws = arena.playGames(4,display=True,save="./test_games/")
print("player 1 wins: "+ str(player1wins))
print("player 2 wins: "+ str(player2wins))
print("draws : " +str(draws))
"""
"""
TK Saved games can be loaded can be loaded by using the static load method of InteractiveBoard.
The path of the game to be loaded has to be given to the function and it returns the InteractiveBoard object
saved for that game. For example when loading game 4 of the evaluation games on an 6x6 board
between neural agent and minimax with combined value:
"""

"""
InBoard = InteractiveBoard.load("./evaluation_games/6x6/minimax/combined value/game4")
InBoard.show_replay()
"""

"""
TK Tournaments between X players can be played by using the static function of the Arena
The row of the result are the wins/draws of this player against the opponent in that column
"""
"""
agents = []
agents.append(neural8x8)
agents.append(minimaxDV8)
agents.append(minimaxCV8)

Arena.Arena.play_tournament(agents,4,game8,15,savefolder="./test_tournament/")
"""
#Since the games are InteractiveBoard objects, they can be directly acsessed to extract data
#For example to get the board at turn 15 of the game
"""
InBoard = InteractiveBoard.load("./evaluation_games/6x6/minimax/combined value/game4")
print(InBoard.board_history[15])
"""
########### 


#TK This is how the average data over all turns was extracted from the played games
"""
all_mcts_simulations = 0
evaluation_mcts = np.zeros((140,3))
evaluation_minimax = np.zeros((140,3))
average_board_value = np.zeros((140,4))
disc_difference = np.zeros((140,2))
wins,losses,draws = 0,0,0

for i in range(1000):
    path = "./evaluation_games/6x6/minimax/combined value/game"+str(i)
    if os.path.isfile(path+".pkl"):
        Inboard = InteractiveBoard.load(path)
        neural = 0

        if Inboard.player1_name == "Neural agent":
            neural = 1
        else:
            neural = -1
        for j in range(len(Inboard.board_history)):
            # TK all 4 different values of the board calculated for the neural agent, as well as the score difference
            # and the number of times games reached this turn (to calculate the average)
            
            disc_difference[j][1] += 1 
            disc_difference[j][0] += Inboard.game.getScore(Inboard.board_history[j],neural)
            average_board_value[j][0]+= Inboard.game.getDiscValue(Inboard.board_history[j],neural)
            average_board_value[j][1]+= Inboard.game.getCornerValue(Inboard.board_history[j],neural,Inboard.game.getLegalMoves(Inboard.board_history[j],neural),Inboard.game.getLegalMoves(Inboard.board_history[j],-neural))
            average_board_value[j][2]+= Inboard.game.getMobilityValue(Inboard.board_history[j],neural,Inboard.game.getLegalMoves(Inboard.board_history[j],neural),Inboard.game.getLegalMoves(Inboard.board_history[j],-neural))
            average_board_value[j][3]+= Inboard.game.getStabilityValue(Inboard.board_history[j],neural)

        for j in range(len(Inboard.action_history)):
            if Inboard.prediction_history[j][0]==neural:
                evaluation_mcts[j+1][0] += 1
                evaluation_mcts[j+1][1] +=Inboard.prediction_history[j][1]
                evaluation_mcts[j+1][2] +=Inboard.prediction_history[j][2]
            if Inboard.prediction_history[j][0]==-neural:
                evaluation_minimax[j+1][0] += 1
                if Inboard.prediction_history[j][1] == math.inf: # TK 
                    evaluation_minimax[j+1][1] += 1
                elif Inboard.prediction_history[j][1] == -math.inf:
                    evaluation_minimax[j+1][1] += -1
                else:
                    evaluation_minimax[j+1][1] +=Inboard.prediction_history[j][1]
                    
                if Inboard.prediction_history[j][2]<10:
                    evaluation_minimax[j+1][2] +=Inboard.prediction_history[j][2]
                else:
                    evaluation_minimax[j+1][2] += 10

        a = Inboard.game.getGameEnded(Inboard.board_history[len(Inboard.board_history)-1],neural)
        if a ==1:
            wins+=1    
        if a ==-1:
            losses+=1
        if a == -0.05:
            draws += 1
            
print(evaluation_mcts)

print(str(wins) + " "+str(losses)+" "+str(draws))
"""
"""
wandb.init(project="neural vs minimax")

for j in range(len(disc_difference)):
    if disc_difference[j][1]>0:
        if evaluation_mcts[j][0]>0 and evaluation_minimax[j][0]>0:
            wandb.log({"Average disk difference":disc_difference[j][0]/disc_difference[j][1],"turn":j,
                        "Neural prediction":evaluation_mcts[j][1]/evaluation_mcts[j][0],     
                        "MCTS sims":evaluation_mcts[j][2]/evaluation_mcts[j][0],
                        "Minimax prediction":evaluation_minimax[j][1]/evaluation_minimax[j][0],
                        "Depth searched":evaluation_minimax[j][2]/evaluation_minimax[j][0],
                        "Disc value":average_board_value[j][0]/disc_difference[j][1],
                        "Corner value":average_board_value[j][1]/disc_difference[j][1],
                        "Mobility value":average_board_value[j][2]/disc_difference[j][1],
                        "stability value":average_board_value[j][3]/disc_difference[j][1]})
"""



