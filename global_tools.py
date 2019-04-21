#!/usr/bin/env python

from __future__ import division
from __future__ import unicode_literals

__author__ = 'Bin Ouyang & Fengyu_xie'
__version__ = 'Dev'

import numpy as np
from operator import mul
from itertools import permutations,product
from functools import partial,reduce

##################################
## General tools that will be frequently cross referenced
##################################
def _GCD(a,b):
	""" The Euclidean Algorithm """
    a = abs(a)
    b = abs(b)
    while a:
        a, b = b%a, a
    return b    
        
def _GCD_List(lst):
	""" Finds the GCD of numbers in a list.
	Input: List of numbers you want to find the GCD of
		E.g. [8, 24, 12]
	Returns: GCD of all numbers, 4 
	"""
    return reduce(_GCD, lst)

def _get_bits(structure):
    """
    Helper method to compute list of species on each site.
    Includes vacancies
    """
    all_bits = []
    for site in structure:
        bits = []
        for sp in sorted(site.species_and_occu.keys()):
            bits.append(str(sp))
        if site.species_and_occu.num_atoms < 0.99:
            bits.append("Vacancy")
       #bits.append("Vacancy")
        all_bits.append(bits)
    return all_bits

def _GetIonChg(ion):
    """
    This tool function helps to read the charge from a given specie(in string format).
    """
    #print(ion)
    if ion[-1]=='+':
        return int(ion[-2]) if ion[-2].isdigit() else 1
    elif ion[-1]=='-':
        #print(ion[-2])
        return int(-1)*int(ion[-2]) if ion[-2].isdigit() else -1
    else:
        return 0

def _factors(n):
    """
    This function take in an integer n and computes all integer multiplicative factors of n

    """
    return set(reduce(list.__add__,
                      ([i, n // i] for i in range(1, int(n ** 0.5) + 1) if n % i == 0)))

def _Get_Hermite_Matricies(Num):
    """
    This function take in an integer and computes all
    Hermite normal matricies with determinant equal to this integer
    All random real matrices can be transformed into a upper-triangle matrix by unitary
    transformations.
    Note:
    clustersupercell.supercell_from_sc does not take an np.matrix! It takes a matrix-like
    3*3 list!!!
    """
    Mats = []; Factors = list(_factors(Num)); Factors *= 3;
    for Perm in set(permutations(Factors, 3)):
        if reduce(mul, Perm) == Num:
            Mat = np.array([[Perm[0], 0, 0], [0, Perm[1], 0], [0, 0, Perm[2]]])
            Perms2 = set(permutations(np.tile(np.arange(Perm[2]), 2), 2))
            Num_list = np.arange(Perm[1]);
            for Num2 in Num_list:
                for Perm2 in Perms2:
                    Mat[0, 1] = Num2; Mat[0:2, 2] = Perm2; LMat = Mat.tolist();
                    if LMat not in Mats: Mats.append(LMat);
    return Mats;

def _mat_mul(mat1,mat2):
    A = np.matrix(mat1)
    B = np.matrix(mat2)
    return (A*B).tolist()

def _FindSpecieSite(specie,occuDict):
    for site in occuDict:
        if specie in occuDict[site]: return site

def _Modify_Specie(specie):
    if not specie[-2].isdigit():
        specie = specie[:-1]+'1'+specie[-1]
    return specie

def _Back_Modify(specie):
    if specie[-2]=='1':
        specie = specie[:-2]+specie[-1]
    return specie

def _Is_Neutral_Occu(occu,specieChgDict):
    totalChg = 0
    for site in occu:
        for specie in site:
            totalChg += site[specie]*specieChgDict[specie]
    return abs(totalChg)<0.001

def _RepresentsInt(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False
