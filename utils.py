#!/usr/bin/env python

from __future__ import division
from __future__ import unicode_literals

__author__ = 'Bin Ouyang & Fengyu_xie'
__version__ = 'Dev'

import numpy as np
from operator import mul
from itertools import permutations,product,combinations
from functools import partial,reduce
import os
import json

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
        for sp in sorted(site.species.keys()):
            bits.append(str(sp))
        if site.species.num_atoms < 0.99:
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
    return pair[::-1]
        
def Is_Anion_Site(site):
    for sp in site.species.keys():
        if GetIonChg(sp)<0:
            return True
    return False

def write_vasp_inputs(Str,VASPDir,functional='PBE',num_kpoints=25,additional_vasp_settings=None, strain=((1.01,0,0),(0,1.05,0),(0,0,1.03)) ):
    # This is a somewhat strange input set. Essentially the matgen input set (PBE+U), but with tigher
    # convergence.
    # This is also a somewhat outdated and convoluted way to generate VASP inputs but it should work fine.
    # These changes to the default input set give much better results.
    # Do not increaes the EDIFF to make it converge faster!!!
    # If convergence is too slow, reduce the K-points
    # This is still using PBE+U with matgen U values though. Need to use MITCompatibility (after the run)
    # to apply oxygen corrections and such.
    # In other expansions that rely on SCAN or HSE, the corrections are different - no O correction for example
    # In additional_vasp_settings, you can add to, or modify the default VASPsettings.
    VASPSettings={"ALGO": 'VeryFast',"ISYM": 0, "ISMEAR": 0, "EDIFF": 1e-6, "NELM": 400, "NSW": 1000, "EDIFFG": -0.02,
                     'LVTOT': False, 'LWAVE': False, 'LCHARG': False, 'NELMDL': -6, 'NELMIN': 8,
                     'LSCALU': False, 'NPAR': 2, 'NSIM': 2, 'POTIM': 0.25, 'LDAU': True};

    if additional_vasp_settings:
        for key in additional_vasp_settings:
            VASPSettings[key]=additional_vasp_settings[key]
            print('Changed {} setting to {}.'.format(key,additional_vasp_settings[key]))

    if not os.path.isdir(VASPDir):os.mkdir(VASPDir);

    # Joggle the lattice to help symmetry broken relaxation. You may turn it off by setting strain=None
    if strain:
         deformation = Deformation(strain)
         defStr = deformation.apply_to_structure(Str)

    #Str=Structure(StrainedLatt,Species,FracCoords,to_unit_cell=False,coords_are_cartesian=False);
    VIO=MITRelaxSet(defStr,potcar_functional = functional); VIO.user_incar_settings=VASPSettings;
    VIO.incar.write_file(os.path.join(VASPDir,'INCAR'));
    VIO.poscar.write_file(os.path.join(VASPDir,'POSCAR'));
    Kpoints.automatic(num_kpoints).write_file(os.path.join(VASPDir,'KPOINTS'));
    # Use PAW_PBE pseudopotentials, cannot use PBE_52, this does not exist on ginar!
    # NOTE: For the POTCARs to work, you need to set up the VASP pseudopotential directory as per the
    # pymatgen instructions, and set the path to them in .pmgrc.yaml located in your home folder.
    # The pymatgen website has instuctrions for how to do this.
    POTSyms=VIO.potcar_symbols;
    for i, Sym in enumerate(POTSyms):
        if Sym == 'Zr': POTSyms[i]='Zr_sv';
    Potcar(POTSyms,functional=functional).write_file(os.path.join(VASPDir,'POTCAR'));



def Write_MAXSAT_input(soft_bcs,soft_ecis,bit_inds,maxsat_fin='maxsat.wcnf',\
                       MAXSAT_PATH='./solvers/',sc_size=None,conserve_comp=None,\
                       sp_names=None,hard_marker=1000000000000,eci_mul=1000000):
    print('Preparing MAXSAT input file.')
    soft_cls = []
    hard_cls = []
    N_sites=len(bit_inds)

    for site_id in range(N_sites):
        hard_cls.extend([[hard_marker]+[int(-1*id_1),int(-1*id_2)] for id_1,id_2 in combinations(bit_inds[site_id],2)])
        #Hard clauses to enforce sum(specie_occu)=1

    if conserve_comp and sc_size:
        
        #Marks variables on the same sublattices and are corresponding to the same specie.
        #We assume that supercell is already sorted for bit_inds.
        #print(bit_inds)
        sublats = [bit_inds[i*sc_size:(i+1)*sc_size] for i in range(0,N_sites//sc_size)]
        #print(sublats)
        bits_sp_sublat = [[[sublat[s_id][sp_id] for s_id in range(len(sublat))] \
                            for sp_id in range(len(sublat[0]))] for sublat in sublats]        
        
        #print(bits_sp_sublat)
        comp_scale = int(sc_size/round(sum(conserve_comp[0].values())))
        #print('sublat_sp_ids',sublat_sp_ids,'sp_names',sp_names)
        scaled_composition = [{sp:sublat[sp]*comp_scale for sp in sublat} for sublat in conserve_comp]

        print("Solving under conserved composition:",scaled_composition,"Supercell size:",sc_size)
        #scaled_composition)

        for sl_id,sublat in enumerate(bits_sp_sublat):
            for sp_id,specie_ids in enumerate(sublat):
                sp_name = sp_names[sl_id][sp_id] #Put in the input instead
                for n_occu in range(sc_size+1):
                    if n_occu != scaled_composition[sl_id][sp_name]:
                         hard_cls.extend([[hard_marker]+\
                                 [int(-1*b_id) for b_id in combo]+\
                                 [int(b_id) for b_id in specie_ids if b_id not in combo] \
                                 for combo in combinations(specie_ids,n_occu)])
        #Hard clauses to enforce composition consistency,scales with O(2^N). Updated Apr 12 19
    if not sc_size:
       print('Warning: supercell size not given. Skipping composition constraints.')   

    all_eci_sum = 0
    for b_cluster,eci in zip(soft_bcs, soft_ecis):
        if int(eci*eci_mul)!=0: 
    #2016 Solver requires that all weights >=1 positive int!! when abs(eci)<1/eci_mul, cluster will be ignored!
            if eci>0:
                clause = [int(eci*eci_mul)]
                all_eci_sum+=int(eci*eci_mul)
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
                    all_eci_sum+=int(-1*eci*eci_mul)
                    for j in range(i+1,len(b_cluster)):
                        clause.append(int(-1*b_cluster[j]))
                    clauses_to_add.append(clause)
                soft_cls.extend(clauses_to_add)

    print('Soft clusters converted!')
    if all_eci_sum > hard_marker:
        print('Hard clauses marker might be too small. You may consider using a bigger value.')

    all_cls = hard_cls+soft_cls
    #print('all_cls',all_cls)
        
    num_of_vars = sum([len(line) for line in bit_inds])
    num_of_cls = len(all_cls)
    maxsat_input = 'c\nc Weighted paritial maxsat\nc\np wcnf %d %d %d\n'%(num_of_vars,num_of_cls,hard_marker)
    for clause in all_cls:
        maxsat_input+=(' '.join([str(lit) for lit in clause])+' 0\n')
    f_maxsat = open(MAXSAT_PATH+maxsat_fin,'w')
    f_maxsat.write(maxsat_input)
    f_maxsat.close()
    print('{} written into.'.format(maxsat_fin,MAXSAT_PATH))

def Call_MAXSAT(maxsat_fin='maxsat.wcnf',maxsat_fout='maxsat.out',solver='CCEHC-incomplete',\
                MAXSAT_PATH='./solvers/',maxsat_file='maxsat.mson',MAXSAT_CUTOFF = 200,entry_name='default'):
    COMPLETE_MAXSAT = ['CCEHC_to_akmaxsat','ccls_akmaxsat',\
                       'open-wbo-gluc','open-wbo-riss']
    INCOMPLETE_MAXSAT = ['CCLS-incomplete','CCEHC-incomplete']

    print('Calling MAXSAT solver.')
    os.chdir(MAXSAT_PATH) 
    MAXSAT_CMD = './'+solver+' ./'+maxsat_fin
    if solver in INCOMPLETE_MAXSAT:
        print("Warning: using incomplete solver. Global optimacy not guaranteed!")
    if solver in COMPLETE_MAXSAT:
        print("Warning: using complete solver. Time cost might be intractable. Good luck!")
    MAXSAT_CMD += '> '+maxsat_fout
    print(MAXSAT_CMD)
    os.system(MAXSAT_CMD)
    os.chdir('..')
    #Saving maxsat calculations into mson.
    entry = {}
    with open(MAXSAT_PATH+maxsat_fin) as fin:
        s_in = fin.read()
    with open(MAXSAT_PATH+maxsat_fout) as fout:
        s_o = fout.read()
    entry['name']=entry_name
    entry['i']=s_in
    entry['o']=s_o
    if os.path.isfile(maxsat_file):
        try:
            with open(maxsat_file) as f:
                l = json.load(f)
        except:
            print("Warning: maxsat file is abnormal. Check before usage.")
            l = []
        l.append(entry)
        with open(maxsat_file,'w') as f:
            json.dump(l,f)
    else:
        l = []
        l.append(entry)
        with open(maxsat_file,'w') as f:
            json.dump(l,f)
    
    print('MAXSAT finished!')

def Read_MAXSAT(MAXSAT_PATH='./solvers/',maxsat_fout='maxsat.out'):
    maxsat_res = []
    with open(MAXSAT_PATH+maxsat_fout) as f_res:
        lines = f_res.readlines()
        for line in lines:
            if line[0]=='v':
                maxsat_res = [int(num) for num in line.split()[1:]]
    sorted(maxsat_res,key=lambda x:abs(x))
    return maxsat_res

