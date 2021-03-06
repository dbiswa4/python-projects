#!/usr/bin/env python
# nrooks.py : Solve the N-Rooks problem!
# D. Crandall, August 2016
# Updated by Zehua Zhang, January 2017
#
# The N-rooks problem is: Given an empty NxN chessboard, place N rooks on the board so that no rooks
# can take any other, i.e. such that no two rooks share the same row or column.

import sys

# Count # of pieces in given row
def count_on_row(board, row):
    return sum( board[row] ) 

# Count # of pieces in given column
def count_on_col(board, col):
    return sum( [ row[col] for row in board ] ) 

def is_prev_dia_danger(board, row, col, total_pieces):
    if col == 0:
        return 0
    if row > 0 and board[row - 1 ][col - 1]:
        return 1
    if row < N - 1 and board[row + 1 ][col - 1]:
        return 1

    # 1 is danger
    # 0 is not danger
    return 0

# Count total # of pieces on board
def count_pieces(board):
    return sum([ sum(row) for row in board ] )

# Return a string with the board rendered in a human-friendly format
def printable_board(board, piece='R', start_row=0, end_row=0):
    return "\n".join([ " ".join([ piece if col else "_" for col in row ]) for row in board[start_row: end_row]])

# Add a piece to the board at the given position, and return a new board (doesn't change original)
def add_piece(board, row, col):
    '''
    For a column do the following.
    Take zero or more rows as per value of 'row'
    '''
    return board[0:row] + [board[row][0:col] + [1,] + board[row][col+1:]] + board[row+1:]


#Final Implementation - more generic, it works for both rooks and queens
# Get list of successors of given board state

def successors3(board, total_pieces, eval_dia_danger_flag=0):
    #print 'successors()3 - get list of successors...'
    #print 'find list of successors for board : ', board
    successors_boards = []
    dia_danger_flag = False
    for c in range(0, count_pieces(board) + 1):
        if count_on_col(board, c) == 0:
            for r in range(0, total_pieces):
                if eval_dia_danger_flag:
                    dia_danger_flag = is_prev_dia_danger(board, r, c, total_pieces)
                if board[r][c] < 1 and count_on_row(board, r) == 0 and not dia_danger_flag and count_pieces(board) < total_pieces:
                    successors_boards.append(add_piece(board, r, c))

    return successors_boards

# check if board is a goal state
def is_goal(board, total_pieces):
    '''
    all([0,0]) False, all([0,1]) False, all([1,1]) True
    '''
    return count_pieces(board) == total_pieces and \
        all( [ count_on_row(board, r) <= 1 for r in range(0, total_pieces) ] ) and \
        all( [ count_on_col(board, c) <= 1 for c in range(0, total_pieces) ] )

# Solve n-rooks!
def solve(initial_board, total_pieces, eval_dia_danger_flag=0):
    print 'solve()...\n'
    print 'total_pieces : ', total_pieces
    fringe = [initial_board]

    #print 'fringe : ', fringe
    #print 'fringe len : ', len(fringe)

    while len(fringe) > 0:
        print '> 0 fringe len : ', len(fringe)
        #print 'Top board in fringe : ', fringe[len(fringe) - 1 ]
        #print 'Take the top board from fringe and find list of successors, foreach successor evaluate whether it is goal state\n'
        for s in successors3(fringe.pop(0), total_pieces, eval_dia_danger_flag):
            #print 'Current successor board : ', s
            if is_goal(s, total_pieces):
                return(s)
            #print 'Current successor not a goal board, hence add to the fringe'
            fringe.append(s)
    return False

# This is N, the size of the board. It is passed through command line arguments.
N = int(sys.argv[1])
Q = int(sys.argv[2])
R = int(sys.argv[3])

# The board is stored as a list-of-lists. Each inner list is a row of the board.
# A zero in a given square indicates no piece, and a 1 indicates a piece.
initial_board = [[0]*N]*N
print ("Starting from initial board:\n" + printable_board(initial_board) + "\n\nLooking for solution...\n")

#Solve for rooks. Diagonal constraint set to false, which is the later argument of solve()
solution = solve(initial_board, R, 0)

#Solve for Queens. Diagonal constraint set to true, which is the later argument of solve()
initial_board = solution
solution = solve(initial_board, N, 1)
print 'solution: \n', solution
#Replace by proper characters
print (printable_board(solution, 'R', 0, R) if solution else "Sorry, no solution found. :(")
print '\n'
print (printable_board(solution, 'Q', R, N) if solution else "Sorry, no solution found. :(")

#Combine rooks and queens matrix
print '\n\n\nFinal Solution:'
print "\n".join([printable_board(solution, 'R', 0, R), printable_board(solution, 'Q', R, N)])


