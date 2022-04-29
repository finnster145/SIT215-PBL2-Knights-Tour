# Import library used for array manipulation
import numpy as np


#   This Function defines the potential moves for a knight and
#   will check if the move is legal, it will return the updated move.
def DetermineMoves(x,y,n):
    pos_x = (2, 1, 2, 1, -2, -1, -2, -1)
    pos_y = (1, 2, -1, -2, 1, 2, -1, -2)
    potential = []
    for i in range(8):
        if x+pos_x[i] >= 0 and x+pos_x[i] < n and y+pos_y[i] >= 0 and y+pos_y[i] < n and chess_board[x+pos_x[i]][y+pos_y[i]] == 0:
            potential.append([x+pos_x[i], y+pos_y[i]])

    return potential



# This function reads in the size of the board and the starting coordinates.
def Solution(x,y,n,chess_board):
    # Confirming if the initial starting point aligns with the size of the board
    if x<n and y<n:
        x = x
        y = y
        chess_board[x][y] = 1
        tracker = 2
        
        #    A for loop that will constantly iterate until all the chessboard squares have been filled up.
        for i in range((n*n)-1):
            potential = DetermineMoves(x,y,n)
            
            if len(potential)==0:
                print("No solution found from ",x,",",y)
                break
            
            minimum = potential[0]
            # This is the implementation of the Warnsdorf Heuristic. It seeks to move to the cell with minimal degrees
            # The minimal degree cells are the cells on the outside. In the output you'll notice that the majority of the large
            # numbers are on the outside whereas the smaller numbers are on the inside.
            for p in potential:
                if len(DetermineMoves(p[0],p[1],n)) <= len(DetermineMoves(minimum[0],minimum[1],n)):
                    minimum = p
            x = minimum[0]
            y = minimum[1]
            chess_board[x][y] = tracker
            tracker += 1
            
        # Once complete it will confirm the solution and print out the chess_board as seen on line 54.
        if tracker==n*n+1:
            print("Solution found!")
            
    else:
        print("insert correct initial points")
        
    print(np.array(chess_board))
    
# The next few lines are defining the board and accepting parameters for N, X and Y.
n=int(input("How many rows/columns wouold you like?  "))

a = str(n - 1)

chess_board = [[0 for i in range(n)] for j in range(n)]

# initial position
x=int(input("What is the starting x position for your knight? (Max value = " + a + ")  "))
y=int(input("What is the starting y position for your knight? (Max value = " + a + ")  "))

Solution(x,y,n,chess_board)

# resetting the chessboard to recompile this cell
chess_board = [[0 for i in range(n)] for j in range(n)]
