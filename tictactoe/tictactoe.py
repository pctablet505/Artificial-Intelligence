"""
Tic Tac Toe Player
"""

import math
from copy import deepcopy

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    
    count_x=0
    count_o=0
    for x in board:
        for y in x:
            if y==X:
                count_x+=1
            elif y==O:
                count_o+=1
    if count_o<count_x:
        return O
    else:
        return X
    


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    moves=set()
    for i in range(3):
        for j in range(3):
            if board[i][j]==EMPTY:
                moves.add((i,j))
    return moves


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    i,j=action
    if board[i][j]!=EMPTY:
        raise Exception('Invalid Move')
    else:
        new_board=deepcopy(board)
        new_board[i][j]=player(new_board)
    return new_board


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    for i in range(3):
        if board[i]==[X,X,X]:
            return X
        elif board[i]==[O,O,O]:
            return O
        column=[board[j][i] for j in range(3)]
        if column==[X,X,X]:
            return X
        elif column==[O,O,O]:
            return O
    diagonal1=[board[i][i] for i in range(3)]
    diagonal2=[board[i][2-i] for i in range(3)]
    if diagonal1==[X,X,X] or diagonal2==[X,X,X]:
            return X
    elif diagonal1==[O,O,O] or diagonal2==[O,O,O]:
        return O
    return None
        
            


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    if winner(board) is not None:
        return True
    else:
        for x in board:
            for y in x:
                if y==EMPTY:
                    return False
    return True


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    won=winner(board)
    if won==X:
        return 1
    elif won==O:
        return -1
    else:
        return 0

def max_value(board,alpha=float('-inf'),beta=float('inf')):
    if terminal(board):
        return utility(board)
    v=float('-inf')
    for action in actions(board):
        value=min_value(result(board, action),alpha,beta)
        if value>v:
            v=value
            alpha=max(alpha,v)
            if beta<=alpha:
                break        
    return v
def min_value(board,alpha=float('-inf'),beta=float('inf')):
    if terminal(board):
        return utility(board)
    v=float('inf')
    for action in actions(board):
        value=max_value(result(board, action),alpha,beta)
        if value<v:
            v=value
            alpha=min(alpha,v)
            if beta<=alpha:
                break        
    return v
    
def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    current_player=player(board)
    if terminal(board):
        return None
    
    best_move=None
    
    if current_player==X:
        v=float('-inf')
        for action in actions(board):
            action_value=min_value(result(board, action))
            if action_value>v:
                v=action_value
                best_move=action
    elif current_player==O:
         v=float('inf')
         for action in actions(board):
             action_value=max_value(result(board, action))
             if action_value<v:
                 v=action_value
                 best_move=action
    return best_move
    
