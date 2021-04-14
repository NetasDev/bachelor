# Selfplay Network Training for Turn-Based Games

This project tries to examine selfplay based learing for neural networks by documenting the training process and testing
the trained networks against each other and against the Minimax Algorithm.

This Project is based on the code of the alpha-zero-general Project at https://github.com/suragnair/alpha-zero-general
All Credit for the base code belongs to Surag Nair and the Contributors listed in his repository

I started of from that repository. I changed some of the code, for example, making changes so that mcts is reset between each game,
actually returning a small negative z value on a draw instead of counting it as a win, properly adding this z value to the trainings 
examples, execute MCTS within a given time frame, names for each agent, etc. Than wandb documentations were added to document all
information of the training process, which might be relevant.

I then implemented the minimax agent and minimax function (for both see othello/othelloAgents.py) and the value functions (see OthelloGame.py). 

Afterwards i implemented the OthellInteractiveBoard class to have a tool for analyzing
games. This included saving and loading games, and all relevant informations they provided. This includs for each turn the board, 
actions selected by an agent and the action value and depth searched/mcts simulations performed to get that action value.
InteractiveBoard has its name because it is also used to play games or look at replays of games by visualizing it via pygame.
When a humanAgent plays, moves can be made by simply clicking on a square. In replays it is possible to switch between turns by
using the left and right arrow keys.

Additionally functions were added to:
play games between X players instead of two by matching one player against all other X-1 players
matching a player against all generations created in a folder in training
analyze games
test the functions used

A general explanation of how functions can be used (including examples) is given in pit.py.
Comments made by myself in the code are marked with my initals TK to be distingushable from comments of the base repository.
The only thing i did change in previously existing comments is the name of variables that I changed.

To run the code, a conda environment is used. By using conda and the command "conda env create -f environment.yml" an 
environment is created. In this environment the commands "pip install pygame" and "pip install wandb" have to be 
executed. 

Also since the trained networks are big files (>100mb) they are stored through git lfs. This means
that after cloning this repository git lfs pull has to be used to get these trained networks.

MCTS can be found in MCTS.py, minimax and neural agents in othello/OthelloAgents.py, the training cycle in Coach.py,
the neural net in othello/keras/OthelloNet.py and the game loop in Arena.py.
A network can be trained by setting hyperparameters in main.py and then using python main.py
All other functions are presented in pit.py.

