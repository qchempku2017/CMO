#!/usr/bin/env python

from __future__ import division
from __future__ import unicode_literals

__author__ = 'Bin Ouyang & Fengyu_xie'
__version__ = 'Dev'


import os
import sys
import argparse
import json
import time
from pymatgen.io.cif import CifParser
from monty.json import MSONable

from generator_tools,analyzer_tools,runner_tools import *

###################################
#One example of OXRange we use is given below:
#ox_ranges = {'Mn': {(0.5, 1.5): 2,
#            (1.5, 2.5): 3,
#            (2.5, 3.4): 4,
#            (3.4, 4.05): 3,
#            (4.05, 5.0): 2}};
#The way it works is really emperical but we did not find a generalized way to
#correlate charge states with magnetism. Maybe we can automate a self consistent routine
#to check the check balance during VASP caculation loading and pick a best one. Or just 
#Pre-calculate it and make it into a data file like json or yaml (works like the DFT+U paprameters);
##################################

class CEAutojob(MSONable):
    """
        An abstract Cluster Expansion automated step. Can be constructed by specifiying job type and reading setting.mson file. 
        Has 4 different steps:
        1, generator: enumerate structures, do sampling, and update vasp calculation folder. writes a prim file.
        2, analyzer: read, analyze vasp calculation results, updates ce.mson and calcdata.mson.
        3, runner: submit calculations to ginar, and keep logging of the calculational status.
        4, gs_solver: find ground state, add gs into vasp_dir, fire a calculation. Call analyzer after success.

        The CE improvement sequence should be:
        generator(cycle 1):N structures -> runner -> analyzer -> gs_solver: N_hull structures -> runner -> analyzer -> RMSE and 
        termination conditions -> generator(cycle 2)

        Runner is the pivot. Still looking at it.
    """
 if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', help="Load vasp runs", type=str, default=None, nargs='+')
    parser.add_argument('--vasp', help="JSON files with saved VASP outputs", type=str, nargs='+')
    parser.add_argument('--ce', help="JSON file of current CE", type=str, default=None)
    parser.add_argument('--fit', help="Fit CE based on VASP outputs", action='store_true')
    parser.add_argument('--prim', help="cif file of primitive cell(for CE fitting)", type=str, default=None)
    parser.add_argument('--getmc', help="Generate MC structures from this CE", action='store_true')
    parser.add_argument('--supercellnum',help="Number of skewed and unskewed supercells to sample",type=int,default=10)
    parser.add_argument('--transmat',help="Transformation matrix to prim, if it is possible to make your prim more symmetric",\
                       type=int,nargs='+',default=None)
    parser.add_argument('--compounds',help="Compounds to enumerate. Example:--compunds='LiCoO2,LiF,Li2CoO2'",\
                       type=str, default=None)
    parser.add_argument('--enforceoccu',help='''Minimum required occupation fraction for each site. Site refered to by
                      index. Example: --enforceoccu='{"0":0.0,"1":1.0,"2":1.0,"3":1.0}'. ''',type=str,default=None)
    parser.add_argument('--samplestep',help="Sampling step of sublattice sites.",type=int,default=1)
    parser.add_argument('--scs', help="Max supercell matrix determinant to be enumerated(must be integer)",\
                       type=int)
    parser.add_argument('--tlst', help="Temperature list", type=int, nargs='+');
    parser.add_argument('--complst', help="Composition list to be simulated[xNb,xF]", type=str,nargs='+');
    parser.add_argument('--vaspinputs', help="Generate compatible VASP inputs for this directory", type=str, default=None)
    parser.add_argument('--vasprun',help="Run vasp for all structures under this directory",type=str,default=None)
    parser.add_argument('-o', help="Output filename (json file or dir)", default=None)
    args = parser.parse_args()
    import time

    start_time = time.time()
    
    if args.vaspinputs: gen_vasp_inputs(args.vaspinputs)
    
    elif args.vasprun:
        run_vasp(args.vasprun)

    elif args.load:
        DirLst = args.load;
        load_data(DirLst, args.o, wmg=args.mg)

    elif args.fit:
        primData = CifParser(args.prim)
        prim = primData.get_structures()[0]
        fit_ce(args.vasp, prim, args.o)

    elif args.getmc:
        primData = CifParser(args.prim)
        prim = primData.get_structures()[0]

        enforceoccu=json.loads(args.enforceoccu)
        enforceoccuNew = {}
        for key in enforceoccu:
            enforceoccuNew[int(key)]=float(enforceoccu[key])

        compounds = args.compounds.split(',')
        supercells = []
        if len(args.transmat)==9:
            tm=args.transmat
            transmat=[[tm[0],tm[1],tm[2]],[tm[3],tm[4],tm[5]],[tm[6],tm[7],tm[8]]]
            print("Using pre-transformation matrix:",transmat)
        else:
            print("Warning: transformation matrix wasn't set. Using no pre-transformation to primitive cell!")
            transmat=None
        supercells.extend(Supercells_From_Compounds(args.scs,prim.get_sorted_structure(),compounds,\
                              enforceoccuNew,args.samplestep,args.supercellnum,transmat))
        #supercells is now a list of (sc,ro,composition)
        if not args.ce:
            get_mc_structs(args.ce, args.vasp, args.o, supercells,Prim=prim.get_sorted_structure())
        else:
            get_mc_structs(args.ce, args.vasp, args.o, supercells)

    print("--- %s seconds ---" % (time.time() - start_time)) 
