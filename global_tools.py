#!/usr/bin/env python

from __future__ import division
from __future__ import unicode_literals

__author__ = 'Bin Ouyang & Fengyu_xie'
__version__ = 'Dev'

import numpy as np
from operator import mul
from itertools import permutations,product
from functools import partial,reduce
import os

##################################
## General tools that will be frequently cross referenced.
## Do not add underscore before since that makes them private!!
##################################
def GCD(a,b):
    """ The Euclidean Algorithm """
    a = abs(a)
    b = abs(b)
    while a:
        a, b = b%a, a
    return b    
        
def GCD_List(lst):
    """ Finds the GCD of numbers in a list.
	Input: List of numbers you want to find the GCD of
		E.g. [8, 24, 12]
	Returns: GCD of all numbers, 4 
    """
    return reduce(GCD, lst)

def get_bits(structure):
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

def GetIonChg(ion):
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

def factors(n):
    """
    This function take in an integer n and computes all integer multiplicative factors of n

    """
    return set(reduce(list.__add__,
                      ([i, n // i] for i in range(1, int(n ** 0.5) + 1) if n % i == 0)))

def Get_Hermite_Matricies(Num):
    """
    This function take in an integer and computes all
    Hermite normal matricies with determinant equal to this integer
    All random real matrices can be transformed into a upper-triangle matrix by unitary
    transformations.
    Note:
    clustersupercell.supercell_from_sc does not take an np.matrix! It takes a matrix-like
    3*3 list!!!
    """
    Mats = []; Factors = list(factors(Num)); Factors *= 3;
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

def mat_mul(mat1,mat2):
    A = np.matrix(mat1)
    B = np.matrix(mat2)
    return (A*B).tolist()

def FindSpecieSite(specie,occuDict):
    for site in occuDict:
        if specie in occuDict[site]: return site

def Modify_Specie(specie):
    if not specie[-2].isdigit():
        specie = specie[:-1]+'1'+specie[-1]
    return specie

def Back_Modify(specie):
    if specie[-2]=='1':
        specie = specie[:-2]+specie[-1]
    return specie

def Is_Neutral_Occu(occu,specieChgDict):
    totalChg = 0
    for site in occu:
        for specie in site:
            totalChg += site[specie]*specieChgDict[specie]
    return abs(totalChg)<0.001

def RepresentsInt(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False

def Reversed(pair):
    return [pair[1],pair[0]]

def Write_MAXSAT_input(soft_bcs,soft_ecis,bit_inds,sc_size=None,conserve_comp=None,sp_names=None,hard_marker=1000000000000,eci_mul=1000000):
    print('Preparing MAXSAT input file.')
    soft_cls = []
    hard_cls = []
    N_sites=len(bit_inds)

    for site_id in range(N_sites):
        hard_cls.extend([[hard_marker]+[int(-1*id_1),int(-1*id_2)] for id_1,id_2 in combinations(bit_inds[site_id],2)])
        #Hard clauses to enforce sum(specie_occu)=1

    if conserve_comp:
        sublat_sp_ids = [[]]*len(N_sites)
        
        #Marks variables on the same sublattices and are corresponding to the same specie.
        site_id = 0
        while site_id<N_sites:
            for sp,sp_id in enumerate(bit_inds[site_id]):
                if len(sublat_sp_ids[site_id//sc_size])<len(bit_inds[site_id]):
                    sublat_sp_ids[site_id//sc_size].append([])
                sublat_sp_ids[site_id//sc_size][sp].append(sp_id)
            site_id+=1

        comp_scale = int(sc_size/round(sum(conserve_comp[0].values())))
        scaled_composition = [{sp:sublat[sp]*comp_scale for sp in sublat} for sublat in conserve_comp]
        for sl_id,sublat in enumerate(sublat_sp_ids):
            for sp_id,specie_ids in enumerate(sublat):
                sp_name = sp_names[sl_id][sp_id] #Put in the input instrad
                hard_cls.extend([[hard_marker]+\
                                 [int(-1*b_id) for b_id in combo]+\
                                 [int(b_id) for b_id in specie_id if b_id not in combo] \
                                 for combo in combinations(specie_ids,scaled_composition[sl_id][sp_name])])
        #Hard clauses to enforce composition consistency,scales with O(2^N). Updated Apr 12 19
    
    all_eci_sum = 0
    for b_cluster,eci in zip(soft_bcs, soft_ecis):
        if int(eci*eci_mul)!=0: 
    #2016 Solver requires that all weights >=1 positive int!! when abs(eci)<1/eci_mul, cluster will be ignored!
            if eci>0:
                clause = [int(eci*eci_mul)]
                all_eci_sum += int(eci*eci_mul)  
        #MAXSAT 2016 series only takes in integer weights
                for b_id in b_cluster:
                    clause.append(int(-1*b_id))
    #Don't worry about the last specie for a site. It is take as a referecne specie, 
    #thus not counted into nbits and combos at all!!!
                soft_cls.append(clause)
            else:
                clauses_to_add = []
                for i in range(len(b_cluster)):
                    clause = [int(-1*eci*eci_mul),int(b_cluster[i])]
                    for j in range(i+1,len(b_cluster)):
                        clause.append(int(-1*b_cluster[j]))
                    clauses_to_add.append(clause)
                soft_cls.extend(clauses_to_add)

    print('Summation of all ecis:',all_eci_sum)
    all_cls = hard_cls+soft_cls
    #print('all_cls',all_cls)
        
    num_of_vars = sum([len(line) for line in blk.bit_inds])
    num_of_cls = len(all_cls)
    maxsat_input = 'c\nc Weighted paritial maxsat\nc\np wcnf %d %d %d\n'%(num_of_vars,num_of_cls,hard_marker)
    for clause in all_cls:
        maxsat_input+=(' '.join([str(lit) for lit in clause])+' 0\n')
    f_maxsat = open('maxsat.wcnf','w')
    f_maxsat.write(maxsat_input)
    f_maxsat.close()
    print('maxsat.wcnf written.')

def Call_Maxsat(solver='ccls_akmaxsat',MAXSAT_PATH='./solvers/',MAXSAT_CUTOFF = 600):
    COMPLETE_MAXSAT = ['akmaxsat','ccls_akmaxsat']
    INCOMPLETE_MAXSAT = ['CCLS2015']

    rand_seed = random.randint(1,100000)
    print('Callsing MAXSAT solver. Using random seed %d.'%rand_seed)
    os.system('cp ./maxsat.wcnf '+MAXSAT_PATH)
    os.chdir(MAXSAT_PATH)
    MAXSAT_CMD = './'+solver+' ./maxsat.wcnf'
    if self.solver in INCOMPLETE_MAXSAT:
        MAXSAT_CMD += ' %d %d'%(rand_seed,MAXSAT_CUTOFF)
    MAXSAT_CMD += '> maxsat.out'
    print(MAXSAT_CMD)
    os.system(MAXSAT_CMD)
    os.chdir('..')
    os.system('cp '+MAXSAT_PATH+'maxsat.out'+' ./maxsat.out')
    print('MAXSAT solution found!')

def Read_Maxsat():
    maxsat_res = []
    with open('./maxsat.out') as f_res:
        lines = f_res.readlines()
        for line in lines:
            if line[0]=='v':
                maxsat_res = [int(num) for num in line.split()[1:]]
    sorted(maxsat_res,key=lambda x:abs(x))
    return maxsat_res

