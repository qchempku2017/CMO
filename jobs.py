#!/usr/bin/env python

"""
    This file defines several tasks that can be performed by CEAuto, inclusing CE fitting, canonical GS solver,
    Monte-Carlo thermodynamics (canonical and semi-grand-canonical), etc.
"""

from __future__ import division
from __future__ import unicode_literals

__author__ = 'Fengyu Xie'
__version__ = 'Dev'

import os
import sys
import json
import time
import yaml

from pymatgen.io.cif import CifParser
from generator_tools import *
from analyzer_tools import *
#from runner_tools import *
from gs_tools import * 
from cluster_expansion.utils import *
from collections import OrderedDict

class CEJob(object):
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
                 n_sc_select=10,transmat=None,compaxis=None,merge_sublats=None,max_vacs=None,\
                 sample_step=1,vasp_settings='vasp_settings.mson',max_sc_size=64,\

                 ce_radius=None,max_de=100,max_ew=3,sm_type='pmg_sm',ltol=0.2,\
                 stol=0.15,angle_tol=5,fit_solver='cvxopt_l1',basis='01',weight='unweighted',\

                 gen_file = 'generator.mson', ana_file = 'analyzer.mson', data_file = 'calcdata.mson',\
                 ce_file = 'ce.mson', run_dir = 'vasp_run', prim_file = 'prim.cif',\
                 rmse_file = 'rmse_cvs.mson',sub_temp = 'sub_template.txt',\

                 precommand="", vaspcommand = "mpiexec.hydra -n $NSLOTS pvasp.5.4.4.intel >> vasp.out",\
                 postcommand="", checking_interval = 120\
                 ):
        """
            All parameters are optional, but be careful with max_vacs, and I highly recommend you to set this up!
            Also, you'd better set up the vaspcommand option manually.

            1st row: generator options. Settings for vasp should be written in another file: vasp_settings.mson
              n_sc_select: 
                  How many skewed and unskewed supercells to select from enumeration.
              transmat: 
                  Transformation matrix applied to construct a more symmetric prim cell. Recommended when system can have better symmetry. For 
                  example, in disordered rock-salt's 2-site prim cell, you can use [[1,1,0],[1,0,1],[0,1,1]]
              compaxis: 
                  The composition axis to project your chemical formula(Li0.1CoO2 -> 0.05Li2O + 1CoO2, then compaxis=['Li2O','CoO2']). 
                  If none, will not do projection;
              merge_sublats:
                  Merging selected sites in primitive cell into sublattices. A list of lists.
              max_vacs: 
                  Maximum vacancy fraction on each sublattice, in list form. If merge_sublats = None, then each site in prim is an independent sublat.
                  For example, in disordered rock-salt, if you have no constraints on cation sites while prohibiting anion vacancies, 
                  you should use: [1.0,0.0];
              sampling_step: 
                  For each sublattice, we will flip (sampling_step) number of sites into another specie each step of enumeration;
              vasp_settings: 
                  File name that contains vasp_setting dictionary.
              max_sc_size: 
                  maimum enumerated supercell size each cell.

            2nd row: analyzer options.
              ce_radius: 
                  cluster selection setting. In CEAuto all clusters within a diameter range are selected. For example, if ce_radius =
                  {2:5.0, 3:4.0, 4:3.0}
              max_de, max_ew: 
                  maximum dielectric and ewald term parameters, used to filter electrostatically unstable structures that ususally fail in 
                  DFT calculation.
              sm_type: 
                  structure matcher type. By default using pymatgen vanilla matcher, but anion framework is highly recommended;
              ltol,stol,angle_tol: 
                  structure matcher parameters. See pymatgen documentations.
              fit_solver: 
                  solver used in CE fitting. By default using an l1 regularized optimizer.
              basis: 
                  basis used to computer cluster functions. By defualt, using delta basis (and this is the only supported one in cluster expansion.)
              weight: 
                  weighting method used to improve fitting.

           3rd row:related setting and data files used and written by CEAuto.

           4th row:commands used in the runner.
               checking_interval: 
                   time interval between two checks of the computer queue status. If all submissions are finished, go to next step. Otherwise continue waiting.
        """
        self.n_sc_select = n_sc_select  
        self.transmat = transmat        
        self.compaxis = compaxis
        self.merge_sublats = merge_sublats        
        self.max_vacs = max_vacs
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

        self.gen_file = gen_file
        self.ana_file = ana_file
        self.ce_file = ce_file
        self.data_file = data_file
        self.run_dir = run_dir
        self.prim_file = prim_file
        self.rmse_file = rmse_file

        self.sub_temp = sub_temp
        self.precommand = precommand
        self.vaspcommand = vaspcommand
        self.postcommand = postcommand
        self.checking_interval = checking_interval

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
        if os.path.isfile(self.prim_file):
            s = 1 
        else:
            return s
        #Generator already ran before, but DFT has not finished
        if os.path.isfile(self.gen_file) and os.path.isdir(self.run_dir):   
            s = 2
        else:
            return s
        
        all_calculated=True

        _is_vasp_calc = lambda fs: 'POSCAR' in fs and 'INCAR' in fs and 'KPOINTS' in fs and 'POTCAR' in fs
        _is_finished = lambda fs: 'CONTCAR' in fs and 'OUTCAR' in fs
        if os.path.isdir(self.run_dir):
            for root,dirs,files in os.walk(self.run_dir):
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
        if os.path.isfile(self.ce_file) and os.path.isfile(self.data_file) and os.path.isfile(self.rmse_file):
            with open(self.rmse_file,'r') as cvs_record:
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
        print('#### Analyzer Call ####')
        if os.path.isfile(self.ana_file):
            ana = CalcAnalyzer.from_settings(setting_file=self.ana_file)
        else:
            ana = CalcAnalyzer(vaspdir = self.run_dir, \
                               prim_file = self.prim_file, \
                               calc_data_file = self.data_file, \
                               ce_file = self.ce_file,\
                               ce_radius = self.ce_radius,\
                               max_de = self.max_de,\
                               max_ew = self.max_ew,\
                               sm_type = self.sm_type,\
                               ltol = self.ltol,\
                               stol = self.stol,\
                               angle_tol = self.angle_tol,\
                               solver = self.fit_solver,\
                               basis = self.basis,\
                               weight = self.weight)
                             
        ana.fit_ce()
        ana.write_files()
        if not os.path.isfile(self.ana_file):
            ana.write_settings(settings_file=self.ana_file)
        #Appending new cv score and RMSE
        if os.path.isfile(self.rmse_file):
            with open(self.rmse_file,'r') as rc_in:
                rmse_cvs=json.load(rc_in) 
        else:
            rmse_cvs = {'rmse':[],'cvs':[]}

        rmse_cvs['rmse'].append(ana.ECIG.rmse)
        rmse_cvs['cvs'].append(ana.ECIG.cv)
        with open(self.rmse_file,'w') as rc_out:
            json.dump(rmse_cvs,rc_out)  

    def _generator_call(self):
        print('#### Generator Call ####')
        if os.path.isfile(self.gen_file):
            gen = StructureGenerator.from_settings(setting_file=self.gen_file)
        else:
            gen = StructureGenerator(prim_file = self.prim_file,\
                                     merge_sublats = self.merge_sublats,\
                                     outdir = self.run_dir,\
                                     max_vacs = self.max_vacs,\
                                     sample_step=self.sample_step,\
                                     max_sc_size = self.max_sc_size,\
                                     sc_selec_enum=self.n_sc_select,\
                                     comp_axis=self.compaxis,\
                                     transmat = self.transmat,\
                                     ce_file = self.ce_file,\
                                     vasp_settings = self.vasp_settings)

        gen.generate_structures()
        if not os.path.isfile(self.gen_file):
            gen.write_settings(settings_file=self.gen_file)

    def _run_calcs(self):
        """
        This submits all uncalculated entree to the computation cluster and do DFT.
        We create the submission script using userprovided, self.sub_temp file, and submit them to user's queue.
        Make sure your submission script and the commands correct. Our program does not check it!
        After submission, our program will check the status of all the calculations your submitted under the name
        *jobname*, at an iterval of self.checking_interval, until all the calculations are finished.
        By default, all the parameters are only based on an SGE queueing system. You may install other APIs and  
        modify the code to support PBS or other queueing systems.
        """
        print('#### Calculation Call ####')
        import re
        import stat
        
        #prepare the submission script
        with open(self.sub_temp) as temp_file:
            template = temp_file.read()
        cwd = os.getcwd()
        jobname = os.path.split(cwd)[-1]+'_ce'
        script = re.sub('\{\*jobname\*\}',jobname,template) 
        script = re.sub('\{\*precommand\*\}',self.precommand,script)   
        script = re.sub('\{\*vaspcommand\*\}',self.vaspcommand,script)
        script = re.sub('\{\*postcommand\*\}',self.postcommand,script)
        
        absRunDir = os.path.join(cwd,self.run_dir)
        parentDir = cwd

        POSDirs=[];

        _is_VASP_Input = lambda files: ('INCAR' in files) and \
                     ('POSCAR' in files) and ('POTCAR' in files)\
                     and ('KPOINTS' in files)

        for Root,Dirs,Files in os.walk(absRunDir):
            if _is_VASP_Input(Files) and 'fm.0' in Root:
                if os.path.isfile(os.path.join(Root,'OUTCAR')):
                    if 'accepted' in Files or 'failed' in Files:
                        continue
                # Kicking out completed calculations.
                POSDirs.append(Root);

        for Root in POSDirs:
            runRoot = os.path.abspath(Root)
            os.chdir(runRoot)
            print("Submitting VASP for {}".format(os.getcwd()))
            ### sub.sh is architecture dependent, and should be provided by the user.
            ### After integrating custodian we shouldn't be using it anymore!
            sub_file = os.path.join(os.getcwd(),'sub.sh')
            with open(sub_file,'w') as fout:
                fout.write(script)
            st = os.stat(sub_file)
            os.chmod(sub_file, st.st_mode | stat.S_IEXEC)
            
            os.system('qsub sub.sh')
            print("Submitted VASP for {}".format(os.getcwd()))
            os.chdir(parentDir)
        print('Submissions done, waiting for calculations.')
        
        while True:
            #try:
            import qstat #only for SGE queueing system
            q,j = qstat.qstat()
            all_jobs = q+j
            #print("Current jobs:\n",q,j)
            all_Finished = True
            for job in all_jobs:
                if type(job)== OrderedDict or type(job)==dict and job['JB_name'] == jobname:
                    all_Finished = False
                    break
            if all_Finished:
                print("Calculations done.")
                break
            #except:
            #    print('Queue status not normal, continuing.')
            #    continue
            print('Time:',time.time())
            time.sleep(self.checking_interval)

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
                os.rename(self.rmse_file,self.rmse_file+'.old')
                self._analyzer_call()
                print("Cluster expansion refitted! Old rmse and cv results copied to .old file.")
                return
            else:
                raise ValueError("Stucture sample pool not finished yet!")
                
        if init_stat == 0:
            raise ValueError("No primitive cell file provided, can't do anything!")
        elif init_stat == 4:
            print("Warning: Already have a converged cluster expansion. If you want to refit with other parameters, \
                   rewrite the analyzer options in options.mson, re-initialize this object, and call fit_ce function\
                   under refit_only=True. Otherwise I will do nothing.")
        else:
            if init_stat <= 1:
                self._generator_call()
            if init_stat <= 2:
                self._run_calcs()
            if init_stat <= 3:
                self._analyzer_call()

        while self.status != 4:
        #### Iterate until converged.
            self._generator_call()
            self._run_calcs()
            self._analyzer_call()
           
    @classmethod
    def from_dict(cls,d):
        n_sc_select=d['n_sc_select'] if 'n_sc_select' in d else 10
        transmat=d['transmat'] if 'transmat' in d else None
        compaxis=d['compaxis'] if 'compaxis' in d else None
        merge_sublats = d['merge_sublats'] if 'merge_sublats' in d else None       
        max_vacs=d['max_vacs'] if 'max_vacs' in d else None
        sample_step=d['sample_step'] if 'sample_step' in d else 1
        vasp_settings=d['vasp_settings'] if 'vasp_settings'in d else 'vasp_settings.mson'
        max_sc_size=d['max_sc_size'] if 'max_sc_size' in d else 64

        ce_radius=d['ce_radius'] if 'ce_radius' in d else None
        max_de=d['max_de'] if 'max_de' in d else 100
        max_ew=d['max_ew'] if 'max_ew' in d else 3
        sm_type=d['sm_type'] if 'sm_type' in d else 'pmg_sm'
        ltol=d['ltol'] if 'ltol' in d else 0.2
        stol=d['stol'] if 'stol'in d else 0.15
        angle_tol=d['angle_tol'] if 'angle_tol' in d else 5
        fit_solver=d['fit_solver'] if 'fit_solver' in d else 'cvxopt_l1'
        basis=d['basis'] if 'basis' in d else '01'
        weight=d['weight'] if 'weight' in d else 'unweighted'

        gen_file=d['gen_file'] if 'gen_file' in d else 'generator.mson'
        ana_file=d['ana_file'] if 'ana_file' in d else 'analyzer.mson'
        data_file=d['data_file'] if 'data_file' in d else 'calcdata.mson'
        ce_file=d['ce_file'] if 'ce_file' in d else 'ce.mson'
        run_dir=d['run_dir'] if 'run_dir' in d else 'vasp_run'
        #gs_dir=d['gs_dir'] if 'gs_dir' in d else 'gs_run'
        prim_file=d['prim_file'] if 'prim_file' in d else 'prim.cif'
        rmse_file=d['rmse_file'] if 'rmse_file' in d else 'rmse_cvs.mson'
        sub_temp=d['sub_temp'] if 'sub_temp' in d else 'sub_template.txt'

        precommand=d['precommand'] if 'precommand' in d else ""
        vaspcommand=d['vaspcommand'] if 'vaspcommand' in d else\
                    "mpiexec.hydra -n $NSLOTS pvasp.5.4.4.intel" #This command is for ginar only
        postcommand=d['postcommand'] if 'postcommand' in d else ""
        checking_interval=d['checking_interval'] if 'checking_interval' in d else 60
        
        return cls(n_sc_select=n_sc_select,\
                   transmat=transmat,\
                   compaxis=compaxis,\
                   merge_sublats = merge_sublats,\
                   max_vacs = max_vacs,\
                   sample_step=sample_step,\
                   vasp_settings=vasp_settings,\
                   max_sc_size=max_sc_size,\

                   ce_radius=ce_radius,\
                   max_de=max_de,max_ew=max_ew,\
                   sm_type=sm_type,\
                   ltol=ltol, stol=stol,angle_tol=angle_tol,\
                   fit_solver=fit_solver,\
                   basis=basis,weight=weight,\

                   gen_file = gen_file,\
                   ana_file = ana_file,\
                   data_file = data_file,\
                   ce_file = ce_file,\
                   run_dir = run_dir,\
                   #gs_dir = gs_dir,\
                   prim_file = prim_file,\
                   rmse_file = rmse_file,\
                   sub_temp = sub_temp,\

                   precommand=precommand,\
                   vaspcommand=vaspcommand,\
                   postcommand=postcommand,\
                   checking_interval=checking_interval\
                  )

    @classmethod
    def from_options(cls,options_file='ce_options.yaml'):
        with open(options_file,'r') as fin:
            d=yaml.safe_load(fin)
        return cls.from_dict(d)

    def as_dict(self):
        return {'n_sc_select':self.n_sc_select,\
                'transmat':self.transmat,\
                'compaxis':self.compaxis,\
                'merged_sublats':self.merged_sublats,\
                'max_vacs':self.max_vacs,\
                'sample_step':self.sample_step,\
                'vasp_settings':self.vasp_settings,\
                'max_sc_size':self.max_sc_size,\

                'ce_radius':self.ce_radius,\
                'max_de':self.max_de,\
                'max_ew':self.max_ew,\
                'sm_type':self.sm_type,\
                'ltol':self.ltol,'stol':self.stol,'angle_tol':self.angle_tol,\
                'fit_solver':self.fit_solver,\
                'basis':self.basis,
                'weight':self.weight,\

                'gen_file':self.gen_file,\
                'ana_file':self.ana_file,\
                'data_file':self.data_file,\
                'ce_file':self.ce_file,\
                'run_dir':self.run_dir,\
                #'gs_dir':self.gs_dir,\
                'prim_file':self.prim_file,\
                'rmse_file':self.rmse_file,\
                'sub_temp':self.sub_temp,\

                'precommand':self.precommand,\
                'vaspcommand':self.vaspcommand,\
                'postcommand':self.postcommand,\
                'checking_interval':self.checking_interval\
               }

    def write_options(self,options_file='ce_options.yaml'):
        with open(options_file,'w') as fout:
            yaml.dump(self.as_dict(),fout)



class GSCanonicalJob(object):
    def __init__(self,ce_file='ce.mson',calc_data_file='calcdata.mson',gs_dir='gs_run'):
        self.ce_file = ce_file
        self.calc_data_file = calc_data_file
        self.gs_dir = gs_dir

    def run_gs(self):
        """
        This calls the canonical gs solver and gives ground states on a hull.
        """
        gss_on_hull = solvegs_for_hull(ce_file=self.ce_file,\
                                       calc_data_file=self.data_file,\
                                       outdir = self.gs_dir)

        with open(self.gs_file,'w') as gs_out:
            json.dump(gss_on_hull,gs_out)

    def as_dict(self):
        return {'ce_file':self.ce_file,
                'calc_data_file':self.calc_data_file,
                'gs_dir':self.gs_dir
               }

    @classmethod
    def from_dict(cls,d):
        ce_file=d['ce_file'] if 'ce_file' in d else 'ce.mson'
        calc_data_file=d['calc_data_file'] if 'calc_data_file' in d else 'calcdata.mson'
        gs_dir=d['gs_dir'] if 'gs_dir' in d else 'gs_run'

        return cls(ce_file=ce_file,\
                   calc_data_file=calc_data_file,\
                   gs_dir=gs_dir)

    @classmethod
    def from_file(cls,filename='gs_options.yaml'):
        with open(filename,'r') as fin:
            return cls.from_dict(yaml.safe_load(fin))
