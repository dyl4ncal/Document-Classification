'''
Created on Mar. 9, 2019

@author: Dylan

This program uses a K-Nearest Neighbor learning algorithm which stores all instances correspond to training 
data points in n-dimensional space. When an unknown discrete data is received, it analyzes the closest k 
number of instances saved (nearest neighbors)and returns the most common class as the prediction and for 
real-valued data it returns the mean of k nearest neighbors.

__label__2 indicates that comment is positive
__label__1 indicates that comment is negative
'''
import numpy as np
import os
import sys


with np.load(os.path.join(sys.path[0], "Data/", "train.npz")) as train_1:
    trian_data=train_1['train']

with np.load("C:/Users/SKY/Dropbox/Marker/371materials/binary classification task/test.npz") as test_1:
    test_data=test_1['test']


'''   
import numpy as np
with np.load("C:/Users/SKY/Dropbox/Marker/371materials/binary classification task/data_numpy.npz") as data:
    train_data=data['train']
    test_data=data['test']
    
#np.savez("train.npz", train=train_data)
#np.savez("test.npz", test=test_data)
''' '''
with np.load("original_numpy.npz") as data:
    positive=data['positive_samples']
    negative=data['negative_samples']
'''

'''#Prints the main menu.
def print_menu():
    print("""\n---------Sudoku Puzzle Solver: Main Menu---------      
\n1. Solve Sudoku using Back-Tracking Algorithm
2. Solve Sudoku using Forward-Checking Algorithm
3. Terminate Program\n""")'''

#def execute_classification_algorithm():
        
        
#Begin running the program by calling the XXXXXX() function.
#if __name__ == "__main__":
    #XXXXXx()
        

