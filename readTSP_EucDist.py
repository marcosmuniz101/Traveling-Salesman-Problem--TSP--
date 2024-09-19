#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 11:51:27 2024

@author: dgrimes
"""

from math import sqrt
from scipy.spatial import distance_matrix
# 
# Read instance in tsp format
def readInstance(fName):
    file = open(fName, 'r')
    size = int(file.readline())
    inst = {}
    for i in range(size):
        line=file.readline()
        (myid, x, y) = line.split()
        inst[int(myid)] = (int(x), int(y))
    file.close()
    return inst

# Compute Euclidean distance between pair of (x,y) coords
def euclideanDistance(cityA, cityB):
    #return sqrt( (cityA[0]-cityB[0])**2 + (cityA[1]-cityB[1])**2 )
    ##Rounding nearest integer
    return round( sqrt( (cityA[0]-cityB[0])**2 + (cityA[1]-cityB[1])**2 ) )