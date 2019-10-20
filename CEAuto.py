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
from generator_tools import *
from analyzer_tools import *
#from runner_tools import *
from gs_tools import * 
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

class CEAutojob(object):
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
                 n_sc_select=10,transmat=None,compaxis=None,enforce_occu=None,sample_step=1,vasp_settings='vasp_settings.mson',max_sc_size=64,\
                 ce_radius=None,max_de=100,max_ew=3,sm_type='pmg_sm',ltol=0.2,stol=0.15,angle_tol=5,fit_solver='cvxopt_l1',basis='01',weight='unweighted'\
                 ):
        """
            All parameters are optional, but becareful with enforce_occu!

            1st row: generator options. Settings for vasp should be written in another file: vasp_settings.mson
              n_sc_select: How many skewed and unskewed supercells to select from enumeration.
              transmat: Transformation matrix applied to construct a more symmetric prim cell. Recommended when system can have better symmetry. For 
                        example, in disordered rock-salt's 2-site prim cell, you can use [[1,1,0],[1,0,1],[0,1,1]]
              compaxis: The composition axis to project your chemical formula(Li0.1CoO2 -> 0.05Li2O + 1CoO2, then compaxis=['Li2O','CoO2']). 
                        If none, will not do projection;
              enforce_occu: Minimum occupation fraction on each site, in dictionary. For example, in disordered rock-salt, if you have no constraints 
                            on cation sites while prohibiting anion vacancies, you should use: [0.0,1.0];
              sampling_step: For each sublattice, we will flip sampling_step number of sites into another specie each step of enumeration;
              vasp_settings: File name that contains vasp_setting.
              max_sc_size: maimum enumerated supercell size each cell.

            2nd row: analyzer options.
              ce_radius: cluster selection setting. In CEAuto all clusters within a diameter range are selected. For example, if ce_radius =
                         {2:5.0, 3:4.0, 4:3.0}
              max_de, max_ew: maximum dielectric and ewald term parameters, used to filter electrostatically unstable structures that ususally fail in 
                              DFT calculation.
              sm_type: structure matcher type. By default using pymatgen vanilla matcher, but anion framework is highly recommended;
              ltol,stol,angle_tol: structure matcher parameters. See pymatgen documentations.
              fit_solver: solver used in CE fitting. By default using an l1 regularized optimizer.
              basis: basis used to computer cluster functions. By defualt, using delta basis (and this is the only supported one in cluster expansion.)
              weight: weighting method used to improve fitting.
        """
        self.n_sc_select = n_sc_select  
        self.transmat = transmat        
        self.compaxis = compaxis        
        self.enforce_occu = enforce_occu
        self.sample_step = sample_step
        self.vasp_settings = vasp_settings
        self.max_sc_size = max_sc_size

        self.ce_radius = ce_radius
        self.max_de = max_de
        self.max_ew = max_ew
        self.sm_type = sm_type
        self.ltol = ltol
        self.stol = stol
        self.angle_tol = angle_tol
        self.fit_solver = fit_solver
        self.basis = basis
        self.weight = weight

    @property
    def status(self):
        """
        This property function looks into the status of prim.cif, options.mson (user provided, compulsory), vasp_settings.mson (user provided, optional),
        generator.mson (by generator), vasp_run directory, calcdata.mson, ce.mson, analyzer.mson (by analyzer), gs.mson (by gs_solver)
        It returns a dictionary concerning the status of these files and directories, which will be used to decide the behaviours of self.run_ce(),
        allowing break-point continuation.
        """
        #Nothing at all
        s = 0
        
        #Initial set up finished but generator has not been run yet, then run generator
        if os.path.isfile('prim.cif') and os.path.isfile('options.mson'):
            s = 1 
        else:
            return s
        #Generator already ran before, but DFT has not finished
        if os.path.isfile('generator.mson') and os.path.isdir('vasp_run'):   
            s = 2
        else:
            return s
        
        all_calculated=True

        _is_vasp_calc = lambda fs: 'POSCAR' in fs and 'INCAR' in fs and 'KPOINTS' in fs and 'POTCAR' in fs
        _is_finished = lambda fs: 'CONTCAR' in fs and 'OUTCAR' in fs
        if os.path.isdir('vasp_run'):
            for root,dirs,files in os.walk('vasp_run'):
                if _is_vasp_calc(files) and (not 'accepted' in files) and (not 'failed' in files):
                    if not _is_finished(files): 
                        all_calculated=False
                        break

        #Next calculation step has all been finished. If this is true, then call to run_ce will start from an analyzer run,
        #regardless of analyzer status, if CE has not converged.
        if all_calculated:
            s = 3
        else:
            return s
       
        #Everything is there, and CE has converged.Do nothing
        if os.path.isfile('ce.mson') and os.path.isfile('calcdata.mson') and os.path.isfile('rmse_cvs.mson'):
            with open('rmse_cvs.mson','r') as cvs_record:
                rmse_cvs = json.load(cvs_record)
            cvs = rmse_cvs['cvs']
            if len(cvs)<3: 
                return s
            elif abs(cvs[-1]-cvs[-2])<=1E-4 and abs(cvs[-1]-cvs[-3])<5E-4:
                s = 4
            else:
                return s

        return s

    def _analyzer_call(self):
        ana = CalcAnalyzer(ce_radius = self.ce_radius)
        ana.fit_ce()
        ana.write_files()
        #Appending new cv score and RMSE
        if os.path.isfile('rmse_cvs.mson'):
            with open('rmse_cvs.mson','r') as rc_in:
                rmse_cvs=json.load(rc_in) 
        else:
            rmse_cvs = {'rmse':[],'cvs':[]}

        rmse_cvs['rmse'].append(ana.ECIG.rmse)
        rmse_cvs['cvs'].append(ana.ECIG.cv)
        with open('rmse_cvs.mson','w') as rc_out:
            json.dump(rmse_cvs,rc_out)  

    def _generator_call(self):
        if os.path.isfile("generator.mson"):
        #### Start from here later

    def _submission_runs(self):
        """
        This submits all uncalculated entree to the computation cluster and do DFT.
        """

    def run_ce(self,refit_only=False):
        """
        Run a user defined cluster expansion job.
        refit_only: If true, only run analyzer once, without adding new DFT samples.
        """
        #steps before entering cycle
        init_stat = self.status
        if refit_only:
            print("Doing refit only.")
            if init_stat >=3: # Must be at least 3 for an analyzer run to be possible.
                os.rename('rmse_cvs.mson','rmse_cvs_old.mson')
                self._analyzer_call()
                print("Cluster expansion refitted!")
                return
            else:
                raise ValueError("Stucture sample pool not finished yet!")
                
        if init_stat == 0:
            raise ValueError("Initial setup not finished, can't do anything!")
        elif init_stat == 4:
            print("Warning: Already have a converged cluster expansion. If you want to refit with other parameters, \
                   rewrite the analyzer options in options.mson, re-initialize this object, and call fit_ce function\
                   under refit_only=True. Otherwise I will do nothing.")
        elif init_stat == 
        
    def run_gs(self):
        """
        This calls the gs solver.
        """

if __name__ == "__main__":

    print("--- %s seconds ---" % (time.time() - start_time)) 
