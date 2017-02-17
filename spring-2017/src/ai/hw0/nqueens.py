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
    '''
    e.g.
    board = [[0, 1], [1, 1]]
    board[0] = [0, 1]
    sum(board[0]) = 1
    '''
    return sum( board[row] ) 

# Count # of pieces in given column
def count_on_col(board, col):
    return sum( [ row[col] for row in board ] ) 

def count_on_dia(board, row, col):
    total = 0
    for c in range(col+1, N):
        print 'c : ', c
        row += 1
        print 'row : ', row
        total += board[row][c]
        if row == N:
            return total
    return False

def is_prev_dia_danger(board, row, col):
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
def printable_board(board):
    return "\n".join([ " ".join([ "Q" if col else "_" for col in row ]) for row in board])

# Add a piece to the board at the given position, and return a new board (doesn't change original)
def add_piece(board, row, col):
    '''
    For a column do the following.
    Take zero or more rows as per value of 'row'

    '''
    #print 'add_piece()...'
    #print board[0:row] + [board[row][0:col] + [1,] + board[row][col+1:]] + board[row+1:]

    return board[0:row] + [board[row][0:col] + [1,] + board[row][col+1:]] + board[row+1:]


# Get list of successors of given board state
def successors(board):
    print 'successors() - get list of successors...'
    print 'find list of successors for board : ', board
    return [ add_piece(board, r, c) for r in range(0, N) for c in range(0,N) ]

#2nd implementation
# Get list of successors of given board state
def successors2(board):
    successors_boards = []
    for c in range(0, N):
        for r in range(0, N):
            if board[r][c] < 1 and count_pieces(board) < N:
                successors_boards.append(add_piece(board, r, c))
    return successors_boards

#3rd implementation
# Get list of successors of given board state
def successors3(board):
    print 'successors()3 - get list of successors...'
    print 'Find list of successors for board : ', board
    successors_boards = []
    for c in range(0, count_pieces(board) + 1):
        if count_on_col(board, c) == 0:
            for r in range(0, N):
                if board[r][c] < 1 and count_on_row(board, r) == 0 and count_pieces(board) < N:
                    successors_boards.append(add_piece(board, r, c))

    return successors_boards

#4 Get list of successors of given board state
def successors4(board):
    print 'successors()4 - get list of successors...'
    print 'Find list of successors for board : ', board
    successors_boards = []
    for c in range(0, count_pieces(board) + 1):
        if count_on_col(board, c) == 0:
            for r in range(0, N):
                #not is_prev_dia_danger(board, r, c) and
                if board[r][c] < 1 and count_on_row(board, r) == 0 and not is_prev_dia_danger(board, r, c) and count_pieces(board) < N:
                    successors_boards.append(add_piece(board, r, c))

    return successors_boards

# check if board is a goal state
def is_goal(board):
    '''
    all([0,0]) False, all([0,1]) False, all([1,1]) True
    '''
    return count_pieces(board) == N and \
        all( [ count_on_row(board, r) <= 1 for r in range(0, N) ] ) and \
        all( [ count_on_col(board, c) <= 1 for c in range(0, N) ] )

# Solve n-rooks!
def solve(initial_board):
    print 'solve()...\n'
    fringe = [initial_board]

    print 'fringe : ', fringe
    print 'fringe len : ', len(fringe)

    while len(fringe) > 0:
        print '> 0 fringe len : ', len(fringe)
        print 'Top board in fringe : ', fringe[len(fringe) - 1 ]
        print 'Take the top board from fringe and find list of successors, foreach successor evaluate whether it is goal state\n'
        #for s in successors( fringe.pop() ):
        #for s in successors2(fringe.pop()):
        #for s in successors3(fringe.pop()):
        #for s in successors3(fringe.pop(0)):
        for s in successors4(fringe.pop()):
            #print 'Current successor board : ', s
            if is_goal(s):
                return(s)
                #print (printable_board(s) if s else "Sorry, no solution found. :(")
            #print 'Current successor not a goal board, hence add to the fringe'
            fringe.append(s)
    return False

# This is N, the size of the board. It is passed through command line arguments.
N = int(sys.argv[1])
#Q = int(sys.argv[2])
#R = int(sys.argv[3])

# The board is stored as a list-of-lists. Each inner list is a row of the board.
# A zero in a given square indicates no piece, and a 1 indicates a piece.
initial_board = [[0]*N]*N
print ("Starting from initial board:\n" + printable_board(initial_board) + "\n\nLooking for solution...\n")

#Trying to find all possible solutions
solution = solve(initial_board)
print (printable_board(solution) if solution else "Sorry, no solution found. :(")
#solve(initial_board)
print "Reached end of the program"


