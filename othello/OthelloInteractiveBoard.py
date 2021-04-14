import pygame
import pygame.freetype
import os
import pickle
import time
from .OthelloAgents import *

# TK InteractiveBoard is an object used to save and load games and also to play games
# The play function was not used for test games, it is only used as a tool to save or load games for analyzing or to play
# casual games. It could be used for testing, but there is no real reason to see the board while testing. 

# The play function does the same process as the play function used in arena, just within a pygame loop.

# Save and load can be used to save or load Inboard objectives and show_replay can be used to show a replay of an already played game.

class InteractiveBoard():
    def __init__(self,game,player1,player2):
        self.player1 = player1
        self.player2 = player2
        self.player1_name = self.player1.name
        self.player2_name = self.player2.name
        self.game = game
        self.size = self.game.n
        self.side_screen_size = 500
        self.screen_size = 1000
        self.square_size = 1000/self.size
        self.space = self.square_size/12
        self.board= self.game.getInitBoard()

        self.player_to_move = 1                 #TK active player
        self.board_history = []                 #all board states within a game are saved
        self.action_history = []                #all actions aswell
        self.prediction_history = []            #when the details option is used, whenever an action is saved, the action value
                                                #and number of MCTS sims that were used to decide on that action are saved aswell in the prediction history
        self.move = 0                           

    def get_last_action(self,player):
        for i in range(len(self.action_history)):
            if self.action_history[len(self.action_history)-i-1][0] == player:
                return self.action_history[len(self.action_history)-i-1][1]
        return None
    
    def get_last_prediction(self,player):
        for i in range(len(self.prediction_history)):
            if self.prediction_history[len(self.prediction_history)-i-1][0] == player:
                return self.prediction_history[len(self.prediction_history)-i-1][1]
        return None

    def get_predictions_of_player(self,player):
        prediction_history = []
        for i in range(len(self.prediction_history)):
            if self.prediction_history[i][0] == player:
                prediction_history[i][0] = self.prediction_history[i][1]
                prediction_history[i][1] = self.prediction_history[i][2]
        return prediction_history

    def get_last_numCalcs(self,player):
        for i in range(len(self.prediction_history)):
            if self.prediction_history[len(self.prediction_history)-i-1][0] == player:
                return self.prediction_history[len(self.prediction_history)-i-1][2]
        return None
    
    def human_players_turn(self):
        if(self.player_to_move == 1 and isinstance(self.player1,HumanAgent)) or (self.player_to_move==-1 and isinstance(self.player2,HumanAgent)):
            return True
        return False

    def draw_field(self,screen):
        screen.fill((0,110,0))
        for row in range(self.size+2):
            pygame.draw.line(screen,(0,0,0),(row*self.square_size,0),(row*self.square_size,self.screen_size),width=4)
            pygame.draw.line(screen,(0,0,0),(0,row*self.square_size),(self.screen_size,row*self.square_size),width=4)
    
    def get_field_at_mouse_pos(self,pos):
        x,y = pos
        row = int(y /self.square_size)
        col = int(x /self.square_size)
        return row,col

    def draw_side_board(self,screen):
        #get all information needed and draw the side board
        Game_Font = pygame.freetype.Font(None,24)
        pygame.draw.rect(screen,(255,255,255),(self.screen_size,0,self.side_screen_size+100,self.screen_size))
        Game_Font.render_to(screen,(self.screen_size+1/5*self.side_screen_size,150),"Black:",(0,0,0))
        Game_Font.render_to(screen,(self.screen_size+1/5*self.side_screen_size,200),self.player1_name,(0,0,0))
        Game_Font.render_to(screen,(self.screen_size+1/5*self.side_screen_size,250),"Last Move:",(0,0,0))
        Game_Font.render_to(screen,(self.screen_size+1/5*self.side_screen_size+150,250),str(self.game.action_to_move(self.get_last_action(1))),(0,0,0))
        Game_Font.render_to(screen,(self.screen_size+1/5*self.side_screen_size,300),"Score:",(0,0,0))
        Game_Font.render_to(screen,(self.screen_size+1/5*self.side_screen_size+100,300),str(self.game.getScore(self.board,1)),(0,0,0))
    
        if self.get_last_prediction(1)!=None:
            Game_Font.render_to(screen,(self.screen_size+1/5*self.side_screen_size,350),"Action value:",(0,0,0))
            Game_Font.render_to(screen,(self.screen_size+1/5*self.side_screen_size+180,350),str(self.get_last_prediction(1))[:6],(0,0,0))
            Game_Font.render_to(screen,(self.screen_size+1/5*self.side_screen_size,400),"Depth/MCTS sims:",(0,0,0))
            Game_Font.render_to(screen,(self.screen_size+1/5*self.side_screen_size+230,400),str(self.get_last_numCalcs(1)),(0,0,0))


        Game_Font.render_to(screen,(self.screen_size+1/5*self.side_screen_size,550),"White:",(0,0,0))
        Game_Font.render_to(screen,(self.screen_size+1/5*self.side_screen_size,600),self.player2_name,(0,0,0))
        Game_Font.render_to(screen,(self.screen_size+1/5*self.side_screen_size,650),"Last Move:",(0,0,0))
        Game_Font.render_to(screen,(self.screen_size+1/5*self.side_screen_size+150,650),str(self.game.action_to_move(self.get_last_action(-1))),(0,0,0))
        Game_Font.render_to(screen,(self.screen_size+1/5*self.side_screen_size,700),"Score:",(0,0,0))
        Game_Font.render_to(screen,(self.screen_size+1/5*self.side_screen_size+150,700),str(self.game.getScore(self.board,-1)),(0,0,0))

        if self.get_last_prediction(-1)!=None:
            Game_Font.render_to(screen,(self.screen_size+1/5*self.side_screen_size,750),"Action value:",(0,0,0))
            Game_Font.render_to(screen,(self.screen_size+1/5*self.side_screen_size+180,750),str(self.get_last_prediction(-1))[:6],(0,0,0))
            Game_Font.render_to(screen,(self.screen_size+1/5*self.side_screen_size,800),"Depth/MCTS sims:",(0,0,0))
            Game_Font.render_to(screen,(self.screen_size+1/5*self.side_screen_size+230,800),str(self.get_last_numCalcs(-1)),(0,0,0))
        

    def draw_board(self,screen):
        # draw all discs
        for i in range(self.size):
            for j in range(self.size):
                x = self.square_size*(i + 0.5)
                y = self.square_size*(j + 0.5)
                if self.board[i][j]==1:
                    pygame.draw.circle(screen,(0,0,0),(x,y),self.square_size/2-self.space/1.1)
                    pygame.draw.circle(screen,(0,0,0),(x,y),self.square_size/2-self.space)
                if self.board[i][j]==-1:
                    pygame.draw.circle(screen,(0,0,0),(x,y),self.square_size/2-self.space/1.1)
                    pygame.draw.circle(screen,(255,255,255),(x,y),self.square_size/2-self.space)
        if self.game.getGameEnded(self.board,self.player_to_move) == 0:
            #draw all potential moves
            moves = self.game.getLegalMoves(self.board, self.player_to_move)
            for n in range(len(moves)):
                if moves[n]==1:
                    x = (int(n/self.size)+0.5)*self.square_size
                    y = ((n%self.size)+0.5)*self.square_size
                    if int(n/self.size) == self.game.n:
                        pygame.draw.circle(screen,(40,40,40),(x,y),self.square_size/2-self.space/1.15)
                        pygame.draw.circle(screen,(255,255,255),(x,y),self.square_size/2-self.space)
                        Game_Font = pygame.freetype.Font(None,24)
                        Game_Font.render_to(screen,(x-24,y-12),"Skip",(0,0,0))
                        break
                    pygame.draw.circle(screen,(0,0,0),(x,y),self.square_size/2-self.space/1.5)
                    pygame.draw.circle(screen,(0,110,0),(x,y),self.square_size/2-self.space)
            
            ####

    def play_game(self,turnThreshold,details=True):
        pygame.init()
        self.board_history.append(self.board)
        self.player1.reset()
        self.player2.reset()

        turn = 1
        turn_over = False
        screen = pygame.display.set_mode((self.screen_size+self.side_screen_size+100,self.screen_size))
        """+self.side_screen_size"""
        self.draw_field(screen)
        self.draw_side_board(screen)
        self.draw_board(screen)
        pygame.display.update()
        run = True
        while run:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if self.human_players_turn():
                        pos = pygame.mouse.get_pos()
                        row,col = self.get_field_at_mouse_pos(pos)
                        moves = self.game.getLegalMoves(self.board,self.player_to_move)
                        for i in range(len(moves)):
                            if moves[i]==1 and self.game.action_to_move(i)==(col,row):
                                self.board, self.player_to_move = self.game.getNextState(self.board,self.player_to_move,i)
                                self.board_history.append(self.board)
                                self.action_history.append((self.player_to_move*-1,i))
                                turn += 1
                                turn_over = True
                                    

            if not self.human_players_turn() and turn_over==False:
                if self.player_to_move ==1:
                    if details == True:
                        action, value, numCalcs = self.player1.play(self.game.getCanonicalForm(self.board,self.player_to_move),turn>=turnThreshold,details=True)
                        self.prediction_history.append((self.player_to_move,value,numCalcs))
                    else:
                        action = self.player1.play(self.game.getCanonicalForm(self.board,self.player_to_move),turn>=turnThreshold)
                else:
                    if details == True:
                        action, value, numCalcs = self.player2.play(self.game.getCanonicalForm(self.board,self.player_to_move),turn>=turnThreshold,details=True)
                        self.prediction_history.append((self.player_to_move,value,numCalcs))
                    else:
                        action = self.player2.play(self.game.getCanonicalForm(self.board,self.player_to_move),turn>= turnThreshold)

                self.board,self.player_to_move = self.game.getNextState(self.board,self.player_to_move,action)
                self.board_history.append(self.board)
                self.action_history.append((self.player_to_move*-1,action))
                turn += 1

            self.draw_field(screen)
            self.draw_side_board(screen)
            self.draw_board(screen)
            pygame.display.update()
            if self.game.getGameEnded(self.board,self.player_to_move) !=0:
                time.sleep(3)
                break
            turn_over = False

        pygame.quit()
        return self.player_to_move * self.game.getGameEnded(self.board, self.player_to_move)
    
    def save(self,path):
        self.player1 = None
        self.player2 = None
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path+'.pkl','wb') as output:
            pickle.dump(self,output,pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path):
        with open(path+'.pkl','rb') as input:
            return pickle.load(input)

    def show_replay(self):
        pygame.init()
        game = self.game
        board = self.board_history[0]
        move_his = self.action_history
        prediction_history = self.prediction_history 
        #InBoard = InteractiveBoard(board,game,len(board))
        #FPS = 60
        run = True
        screen = pygame.display.set_mode((self.screen_size+self.side_screen_size+100,self.screen_size))
        """+self.side_screen_size,self.screen_size"""
        while run:

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        if self.move > 0:
                            self.move = self.move -1
                            self.player_to_move *= -1
                        
                    if event.key == pygame.K_RIGHT:
                        if self.move < len(self.board_history)-1:
                            self.move = self.move +1
                            self.player_to_move *= -1
                    if event.key == pygame.K_1:

                        pygame.image.save(screen,"testfile.jpg")
                    
            self.board = self.board_history[self.move]
            self.action_history = move_his[:self.move]
            self.prediction_history = prediction_history[:self.move]
            
            self.draw_field(screen)
            self.draw_side_board(screen)
            self.draw_board(screen)
            pygame.display.update()
        pygame.quit()

            
        

     