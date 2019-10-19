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

from generator_tools import *
from analyzer_tools import *
from runner_tools import *
#from gs_tools import * temporarily deprecated
from utils import *

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
        generator(cycle 1) of a pool:N structures -> runner -> analyzer -> gs_solver: N_hull structures -> runner -> analyzer 
        -> RMSE and termination conditions -> generator(cycle 2)

        Runner is the pivot. Still looking at it. Using old stuff as alternative.
    """
    def __init__(self,\
                 n_sc_select=10,transmat=None,compaxis=None,enforce_occu=None,sample_step=1,\
                 ce_radius=None,max_de=100,max_ew=3,sm_type='pmg_sm',ltol=0.2,stol=0.15,angle_tol=5,fit_solver='cvxopt_l1',basis='01',weight='unweighted'
                 ):
        

if __name__ == "__main__":

    print("--- %s seconds ---" % (time.time() - start_time)) 
