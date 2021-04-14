import logging
import pandas as pd
import numpy as np
import os
import wandb

from tqdm import tqdm
from othello.OthelloInteractiveBoard import InteractiveBoard
from othello.keras.NNet import NNetWrapper as nn
from othello.OthelloInteractiveBoard import InteractiveBoard
from othello.OthelloLogic import Board
from othello.OthelloAgents import *

from utils import *

log = logging.getLogger(__name__)


class Arena():
    """
    An Arena class where any 2 agents can be pit against each other.
    """

    def __init__(self, player1, player2, game, turnThreshold=3,display=False):
        """
        Input:
            player 1,2: two functions that takes board as input, return action
            game: Game object

        see othello/OthelloAgents.py for an example. See pit.py for pitting
        human players/other baselines with each other.
        """
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display
        self.turnThreshold = turnThreshold

    def playGame(self, verbose=False,save=False,display=False):
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """

        # TK if the display is active, the game is played using the pygame loop in Interactive board instead.
        # all returns are the same as when using the normal function
        if display == True:
            InBoard = InteractiveBoard(self.game,self.player1,self.player2)
            gameResult = InBoard.play_game(details=True,turnThreshold=self.turnThreshold)
            if save != False:
                i = 1
                while os.path.isfile(save+"game"+str(i)+".pkl"):
                    i=i+1
                InBoard.save(save+"game"+str(i))
            return gameResult

        self.player1.reset()
        self.player2.reset()

        if save!= False:
            InBoard = InteractiveBoard(self.game,self.player1,self.player2)
            InBoard.board_history.append(InBoard.board)

        players = [self.player2, None, self.player1]
        curPlayer = 1
        board = self.game.getInitBoard()
        turn = 0
        while self.game.getGameEnded(board, curPlayer) == 0:
            turn += 1
            if save == False:
                action = players[curPlayer + 1].play(self.game.getCanonicalForm(board, curPlayer),turn>=self.turnThreshold)
            else:
                action, value, numCalcs = players[curPlayer + 1].play(self.game.getCanonicalForm(board, curPlayer),turn>=self.turnThreshold,details=True)

            legal = self.game.getLegalMoves(self.game.getCanonicalForm(board, curPlayer), 1)

            if legal[action] == 0:
                log.error(f'Action {action} is not legal!')
                log.error(f'failed player {players[curPlayer+1].name}')
                log.debug(f'legal = {legal}')
                assert legal[action] > 0
            board, curPlayer = self.game.getNextState(board, curPlayer, action)

            if save != False:
                InBoard.board_history.append(board)
                InBoard.action_history.append((curPlayer*-1,action))
                InBoard.prediction_history.append((curPlayer*-1,value,numCalcs))

        if save != False:
            i = 1
            while os.path.isfile(save+"game"+str(i)+".pkl"):
                i=i+1
            InBoard.save(save+"game"+str(i))
        return curPlayer * self.game.getGameEnded(board, curPlayer)

    def playGames(self, num, verbose=False,display = False,save = False):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """

        num = int(num / 2)
        oneWon = 0
        twoWon = 0
        draws = 0
        i=1

        for _ in tqdm(range(num), desc="Arena.playGames (1)"):
            

            gameResult = self.playGame(verbose=verbose,save=save,display=display)
            if gameResult == 1:
                oneWon += 1
            elif gameResult == -1:
                twoWon += 1
            else:
                draws += 1
            i+=1

        self.player1, self.player2 = self.player2, self.player1
        i = 1

        for _ in tqdm(range(num), desc="Arena.playGames (2)"):

            gameResult = self.playGame(verbose=verbose,save=save,display=display)
            if gameResult == -1:
                oneWon += 1
            elif gameResult == 1:
                twoWon += 1
            else:
                draws += 1
            i+=1
        return oneWon, twoWon, draws

    @staticmethod
    def play_tournament(players,num_matches,game,turnThreshold,savefolder=False):
        # TK A tournament is played between x players, and each player is matched up against all other x-1 players.
        # The results are printed. If a savefolder is set, both the results and all games played are save in this folder.
        wins = np.zeros((len(players),len(players)),dtype=int)
        draws = np.zeros((len(players),len(players)),dtype=int)
        names = []

        for i in range(0,len(players)):
            print(i)
            names.append(players[i].name)
            j = i+1
            while j<=len(players)-1:
                arena = Arena(players[i],players[j],game,turnThreshold=turnThreshold)
                if savefolder!= False:
                    save = savefolder +"/"+ players[i].name +" VS " +players[j].name+"/"
                else:
                    save = False
                wins[i][j],wins[j][i],draws[i][j] = arena.playGames(num_matches,save=save)
                draws[j][i] = draws[i][j]
                #print(players[i].name +" played against "+players[j].name + " with " + str(wins[i][j])+" wins and " + str(wins[j][i])+ "losses")
                j = j+1

        df = pd.DataFrame(wins,columns=names,index=names)
        df2 = pd.DataFrame(draws,columns=names,index=names)

        print("wins:")
        print(df)
        print("draws:")
        print(df2)

        if savefolder!= False:
            df.to_csv(r""+savefolder+"/wins.csv")
            df2.to_csv(r""+savefolder+"/draws.csv")

    @staticmethod
    def play_previous_generations(player,folder,num_matches,game,turnThreshold,savefolder=False):
        # TK One agent is matched up against all previous generations of networks in a folder.
        # In order for this function to work, the names of the previous generations have to be
        # exactly as they are created by the training cycle in coach.
        # The results are printed. If a savefolder is set, both the results and all games played are save in this folder.

        
        wandb.init(project="8x8 against previous iterations")
        wins=[]
        losses=[]
        draws=[]
        names =[]
        i = 0
        generation =0
        while i <=70:
            print(folder+"checkpoint_"+str(i))
            if os.path.isfile(folder+"checkpoint_"+str(i)):
                print("loading generation "+str(generation))
                network = nn(game)
                network.load_checkpoint(folder= folder,filename="checkpoint_"+str(i))
                args = dotdict({'numMCTSSims':player.args.numMCTSSims,'cpuct':player.args.cpuct})
                neuralplayer = NeuralAgent(game,network,args)
                neuralplayer.name = "Neural Network Generation "+str(generation)
                names.append(neuralplayer.name)
                if savefolder!= False:
                    save = savefolder +"/"+ neuralplayer.name+"/"
                else:
                    save = False
                arena = Arena(player,neuralplayer,game,turnThreshold=turnThreshold)
                winsG,lossesG,drawsG = arena.playGames(num_matches,save=save)
                wins.append(winsG)
                losses.append(lossesG)
                draws.append(drawsG)
                wandb.log({'Wins':winsG,'Losses':lossesG,'Draws':drawsG,
                'Winrate':winsG/(winsG+drawsG+lossesG),'Winrate including draws':(winsG+0.5*drawsG)/(winsG+drawsG+lossesG)})
                generation = generation+1
            i = i+1


        df = pd.DataFrame(wins,columns=player.name,index=names)
        df2 = pd.DataFrame(wins,columns=player.name,index=names)

        print("wins:")
        print(df)
        print("draws:")
        print(df2)

        if savefolder!= False:
            df.to_csv(r""+savefolder+"/wins.csv")
            df2.to_csv(r""+savefolder+"/draws.csv")

