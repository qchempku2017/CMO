#!/usr/bin/env python

from __future__ import division
from __future__ import unicode_literals

__author__ = 'Bin Ouyang & Fengyu_xie'
__version__ = 'Dev'


import os
import sys
import json
#from monty.json import MSONable
import random
from copy import deepcopy
import numpy as np
import numpy.linalg as la

from pymatgen.analysis.structure_matcher import StructureMatcher, ElementComparator
from pymatgen.analysis.local_env import CrystalNN
from cluster_expansion.eci_fit import EciGenerator
from cluster_expansion.ce import ClusterExpansion
from pymatgen.io.vasp.inputs import *
from pymatgen.io.vasp.outputs import *
from pymatgen.io.cif import CifParser
from pymatgen import Structure
from pymatgen.analysis.elasticity.strain import Deformation
from pymatgen.core.sites import PeriodicSite
from monty.json import MSONable
from monty.serialization import dumpfn
## note for dummies: dumpfn in monty.serialization will automatically convert all memebers in a dict into
## json acceptable class. So if you see errors like: 'ndarray' not serializable in json, and can't find why,
## just get over it and use dumpfn.

from OxData import OXRange #This should be a database like file


#### Private tools ####
def _assign_ox_states(struct,magmoms):
    """
    Assign oxidation states based on magnetic moments taken from the OUTCAR.
    Reference magnetic moments obtained by looking at a bunch of structures.

    DOES NOT CHECK THAT THE ASSIGNMENT IS CHARGE BALANCED!

    Currently assinging based on an OXRange dict file, but does not ensure 
    charge balance!!

    Args:
        Str: structure
        Mag: list of magnetic moments for the sites in s
    """
    # Oxidation states corresponding to a range of magnetic moments (min, max)
    ###OXRange imported from OxData.py
    # Talk to Tina Chen about stable Oxstate assignment.
    DefaultOx={'Li':1,'F':-1,'O':-2, 'Mg':2, 'Ca':2 }
    OxLst=[];
    # Restricted Ox state for O to 2 here, but actually can be one. There is currently
    # no good method to assign since O magmoms are usually highly inaccurate. We have 
    # to do that.

    for site_id, site in enumerate(struct):
        Assigned=False;
        if site.species_string in OXRange.keys():
            for (MinMag,MaxMag),MagOx in OXRange[site.species_string].items():
                if magmoms[site_id]>=MinMag and magmoms[site_id]<MaxMag:
                    OxLst.append(MagOx); Assigned=True; break;
        elif site.species_string in DefaultOx.keys():
            OxLst.append(DefaultOx[site.species_string]); Assigned=True;
        if not Assigned:
            print("Cant assign ox states for site={}, mag={}".\
                    format(Site,Mag[SiteInd])); assert Assigned;

    struct.add_oxidation_state_by_site(OxLst);
    return struct
 
#### Public Tools ####
class CalcAnalyzer(object):

    def __init__(self, vaspdir='vasp_run', prim_file='prim.cif',calc_data_file='calcdata.mson',ce_file='ce.mson',ce_radius=None,\
                 max_de=100,max_ew=3, sm_type='pmg_sm', ltol=0.2, stol=0.15, angle_tol=5, vor_tol=1e-3, solver='cvxopt_l1',\
                 basis='01',weight='unweighted'):
        self.calcdata = {}
        self.vaspdir = vaspdir
        self.calc_data_file = calc_data_file
        self.ce_file = ce_file
        self.solver = solver
        self.weight = weight
        self.sm_type = sm_type
        self.ltol = ltol
        self.stol = stol
        self.angle_tol = angle_tol
        self.vor_tol = vor_tol
        self.basis = basis
        

        self.prim = CifParser(prim_file).get_structures()[0]

        if os.path.isfile(calc_data_file):
            with open(calc_data_file) as Fin:
                self.calcdata = json.load(Fin)
        else:
            print('No previous calculation data found. Building new.')
            self.calcdata['prim']=self.prim.as_dict()
            self.calcdata['compositions']={}
                       
        if os.path.isfile(ce_file):
            with open(ce_file) as Fin:
                ce_dat_old = json.load(Fin)
            self.ce = ClusterExpansion.from_dict(ce_dat_old['cluster_expansion'])
            self.max_de = ce_dat_old['max_dielectric']
            self.max_ew = ce_dat_old['max_ewald']

        else:
            if not ce_radius:
                d_nns = []
                for i,site1 in enumerate(self.prim):
                    d_ij = []
                    for j,site2 in enumerate(self.prim):
                        if j<i: continue;
                        if j>i:
                            d_ij.append(site1.distance(site2))
                        if j==i:
                            d_ij.append(min([self.prim.lattice.a,self.prim.lattice.b,self.prim.lattice.c]))
                    d_nns.append(min(d_ij))
                d_nn = max(d_nns)
    
                ce_radius = {}
                # Default cluster radius
                ce_radius[2]=d_nn*4.0
                ce_radius[3]=d_nn*2.0
                ce_radius[4]=d_nn*2.0

            self.max_de = max_de
            self.max_ew = max_ew
            self.ce_radius = ce_radius

            self.ce = ClusterExpansion.from_radii(self.prim, ce_radius,sm_type = self.sm_type,\
                                     ltol=self.ltol, stol=self.stol, angle_tol=self.angle_tol,\
                                     vor_tol=self.vor_tol, supercell_size='volume',use_ewald=True,\
                                     use_inv_r=False,eta=None, basis=self.basis);
    
        #self.max_deformation = max_deformation
        #print("Scanning vasprun for new data points.")
        

    def fit_ce(self):
        """
        Inputs:
            1, data_file: name of the mson file that stores the primitive cell for the calculation,
            compisitional axis (if any), all the structures (input and relaxed), their energies 
            and compositions. (cluster_expansion object, ecis, and ground state solver informations 
            will be saved in another mson file, named as ce_file.) These structures are already 
            assigned charges, and are deduplicated.
               Recorded in one dictionary.
            2, ce_file: a file to store cluster expansion info, gs info, ecis, etc.
            3, ce_radius: Max cluster radius set up. Only required when no existing ce is present.
            4, max_deformation: parameters to set up CE.structure_matcher object
        Outputs:
            None. The ce_data file will be updated.

        """
        print("Loading data from {}".format(self.vaspdir))
        self._load_data()
        print("Updating cluster expansion.")
        #Use crystal nearest neighbor analyzer to find nearest neighbor distance, and set cluster radius according to it.
        
        ValidStrs = []
        energies = []
        for comp in self.calcdata['compositions']:
            for entry in self.calcdata['compositions'][comp]:
                ValidStrs.append(Structure.from_dict(entry['relaxed_deformed']))
                energies.append(entry['total_energy'])
    
        #print('ValidStrs',ValidStrs,'len',len(ValidStrs))
        #print('energies',energies,'len',len(energies))
        ## These should already have been deduplicated
    
       # Fit expansion, currently only support energy/free energy expansion. If you want to expand other properties, 
        # You have to write on your own.
        if self.weight=='unweighted':
            self.ECIG=EciGenerator.unweighted(cluster_expansion=self.ce, structures=ValidStrs,\
                                     energies = energies,\
                                     max_dielectric=self.max_de, max_ewald=self.max_ew, \
                                     solver = self.solver);
        elif self.weight == 'e_above_hull':
            self.ECIG=EciGenerator.weight_by_e_above_hull(cluster_expansion=self.ce, structures=ValidStrs,\
                                     energies = energies,\
                                     max_dielectric=self.max_de, max_ewald=self.max_ew, \
                                     solver = self.solver);
        elif self.weight == 'e_above_comp':
            self.ECIG=EciGenerator.weight_by_e_above_comp(cluster_expansion=self.ce, structures=ValidStrs,\
                                     energies = energies,\
                                     max_dielectric=self.max_de, max_ewald=self.max_ew, \
                                     solver = self.solver);
        
        else:
            raise ValueError('Weighting option not implemented!')

        print("RMSE: {} eV/prim, num of points: {}.".format(self.ECIG.rmse,len(ValidStrs))); 

    
    def _load_data(self):
        """    
        This function parses existing vasp calculations, does mapping check, assigns charges and writes into the calc_data file 
        mentioned in previous functions. What we mean by mapping check here, is to see whether a deformed structure can be mapped
        into a supercell lattice and generates a set of correlation functions in clustersupercell.corr_from_structure.
        
        We plan to do modify corr_from_structure from using pymatgen.structurematcher to a grid matcher, which will ensure higher 
        acceptance for DFT calculations, but does not necessarily improve CE hamitonian, since some highly dipoled and deformed 
        structures might have poor DFT energy, and even SABOTAGE CE!
        """    
        # Every key in self.calcdata['compositions'] is a composition, and each composition contains a list of dict entrees.
        # relaxed_structure, input_structure, magmoms, total_energy. 
    
        _is_vasp_calc = lambda fs: 'POSCAR' in fs and 'CONTCAR' in fs and 'OSZICAR' in fs and 'OUTCAR' in fs
        # Load VASP runs from given directories
    
        _did_ax_decomp = False
        n_matched = 0
        n_inputs = 0
        for root,dirs,files in os.walk(self.vaspdir):
            if _is_vasp_calc(files) and (not 'accepted' in files) and (not 'failed' in files):
                print("Loading VASP run in {}".format(root));
                parent_root = os.path.join(*root.split(os.sep)[0:-1])
                parent_parent_root = os.path.join(*root.split(os.sep)[0:-2])
                #print('parent {}'.format(parent_root))
                #print('parent parent {}'.format(parent_parent_root))
                with open(os.path.join(parent_parent_root,'composition_by_site')) as compfile:
                    composition = json.load(compfile)
                    compstring = json.dumps(composition)
    
                if compstring not in self.calcdata['compositions']:
                    self.calcdata['compositions'][compstring]=[]
    
                relaxed_struct = Poscar.from_file(os.path.join(root,'CONTCAR')).structure
                input_struct = Poscar.from_file(os.path.join(parent_root,'POSCAR')).structure
    
                strict_sm = StructureMatcher(stol=0.1, ltol=0.1, angle_tol=1, comparator=ElementComparator())
                _is_unique = True
                for entry in self.calcdata['compositions'][compstring]:
                    entry_struct = Structure.from_dict(entry['relaxed_structure'])
                    if strict_sm.fit(entry_struct,relaxed_struct):
                        _is_unique = False
                        break
                if not _is_unique:
                    print("Entry {} already in calculation data file. Passing.".format(root))
                    open(os.path.join(root,'failed'),'a').close()
                    continue
                n_inputs += 1
    
                # Note: the input_struct here comes from the poscar in upper root, rather than fm.0, so 
                # it is not deformed.
    
                # Rescale volume to that of unrelaxed structure, this will lead to a better mapping back. 
                # I changed it to a rescaling tensor
                relaxed_lat_mat = np.matrix(relaxed_struct.lattice.matrix)
                input_lat_mat = np.matrix(input_struct.lattice.matrix)
                o2i_deformation = Deformation(input_lat_mat.T*relaxed_lat_mat.I.T)
                relaxed_deformed = o2i_deformation.apply_to_structure(relaxed_struct)
                #print(relaxed_deformed,input_struct)
    
                # Assign oxidation states to Mn based on magnetic moments in OUTCAR
                Out=Outcar(os.path.join(root,'OUTCAR')); Mag=[];
                for SiteInd,Site in enumerate(relaxed_struct.sites):
                    Mag.append(np.abs(Out.magnetization[SiteInd]['tot']));
                relaxed_deformed = _assign_ox_states(relaxed_deformed,Mag);
                relaxed_struct = _assign_ox_states(relaxed_struct,Mag)
                #input_struct = _assign_ox_states(input_struct,Mag)
                #print("relaxed_struct {}".format(relaxed_struct))
                # Throw out structures where oxidation states don't make sense
                #print('Input Struct:',input_struct)
                #print('Relaxed deformed:',relaxed_deformed)
    
                if np.abs(relaxed_struct.charge)>0.01:
                    print(relaxed_struct)
                    print("Instance {} not charge balanced .. skipping".format(root))
                    open(os.path.join(root,'failed'),'a').close()
                    #Add a status marker to directory.
                    continue
    
                # Get final energy from OSZICAR or Vasprun. Vasprun is better but OSZICAR is much
                # faster and works fine is you separately check for convergence, sanity of
                # magnetic moments, structure geometry
                with open(os.path.join(root, 'OUTCAR')) as outfile:
                    outcar_string = outfile.read()
                if 'reached required accuracy' not in outcar_string:
                    print('Instance {} did not converge to required accuracy. Skipping.'.format(root))
                    open(os.path.join(root,'failed'),'a').close()
                    continue
                TotE=Oszicar(os.path.join(root, 'OSZICAR')).final_energy;
                # Checking convergence
               
                # Checking whether structure can be mapped to corr function.
                # This is out deformation tolerance.     
                try:
                #print(self.ce.as_dict())
                    self.ce.corr_from_structure(relaxed_deformed)
                except:
                    print("Instance {} too far from lattice. Skipping.".format(root))
                    open(os.path.join(root,'failed'),'a').close()
                    continue
    
                new_entry = {}
                new_entry['input_structure']=input_struct.as_dict()
                new_entry['relaxed_structure']=relaxed_struct.as_dict()
                new_entry['relaxed_deformed']=relaxed_deformed.as_dict()
                new_entry['total_energy']=TotE
                new_entry['magmoms']=Mag
    
                if os.path.isfile(os.path.join(parent_parent_root,'axis')):
                    _did_ax_decomp = True
                    with open(os.path.join(parent_parent_root,'axis')) as axisfile:
                        axis = json.load(axisfile)
                    if 'axis' not in new_entry:
                        new_entry['axis']=axis
                else:
                    if _did_ax_decomp:
                        print('Selected axis decomposition, but instance {} is not axis decomposed!'.format(root))
                        new_entry['axis'] = None
    
                self.calcdata['compositions'][compstring].append(new_entry)
                open(os.path.join(root,'accepted'),'a').close()
                n_matched += 1
    
        # Data already deduplicated!
        print('{}/{} structures matched in this run. Parsed vasp data saved into {}.'.format(n_matched,n_inputs,self.calc_data_file))
 
    def write_files(self):
        with open(self.calc_data_file,'w') as Fout:
            json.dump(self.calcdata,Fout)
        #with open(self.ce_file,'w') as Fout:
            #d = self.ECIG.as_dict()
            #for key,val in d.items():
            #    print('key {} is of type {}'.format(key,type(val)))
            #json.dump(d,Fout)
            # For any msonable, use dumpfn to save your time!
        dumpfn(self.ECIG,self.ce_file)    
    
    @classmethod
    def from_settings(cls,setting_file='analyzer_settings.mson'):
        if os.path.isfile(setting_file):
            with open(setting_file,'r') as fs:
                settings = json.load(fs)
        else:
            settings = {}
        return cls.from_dict(settings)

    @classmethod
    def from_dict(cls,settings):
        if 'vaspdir' in settings: vaspdir = settings['vaspdir']; 
        else: vaspdir = 'vasp_run';
    
        if 'prim_file' in settings: prim_file = settings['prim_file'];
        else: prim_file = 'prim.cif';
    
        if 'calc_data_file' in settings: calc_data_file = settings['calc_data_file'];
        else: calc_data_file = 'calcdata.mson';
    
        if 'ce_file' in settings: ce_file = settings['ce_file'];
        else: ce_file = 'ce.mson';
    
        if 'ce_radius' in settings: ce_radius = settings['ce_radius'];
        else: ce_radius = None;
    
        if 'max_de' in settings: max_de = settings['max_de'];
        else: max_de = 100;
    
        if 'max_ew' in settings: max_ew = settings['max_ew'];
        else: max_ew = 3;
    
        if 'sm_type' in settings: sm_type = settings['sm_type'];
        else: sm_type = 'pmg_sm';
    
        if 'ltol' in settings: ltol = settings['ltol'];
        else: ltol = 0.2;
    
        if 'stol' in settings: stol = settings['stol'];
        else: stol = 0.15;
    
        if 'angle_tol' in settings: angle_tol = settings['angle_tol'];
        else: angle_tol = 5;
    
        if 'vor_tol' in settings: vor_tol = settings['vor_tol'];
        else: vor_tol = 1e-3;

        if 'solver' in settings: solver = settings['solver'];
        else: solver = 'cvxopt_l1';

        if 'basis' in settings: basis = settings['basis'];
        else: basis = '01';

        if 'weight' in settings: weight = settings['weight'];
        else: weight = 'unweighted'

        return cls(vaspdir=vaspdir,prim_file=prim_file,calc_data_file=calc_data_file,ce_file = ce_file, ce_radius=ce_radius,\
                   max_de = max_de, max_ew = max_ew, sm_type = sm_type, ltol = ltol, stol = stol, angle_tol = angle_tol,\
                   vor_tol = vor_tol, solver = solver, basis = basis, weight = weight)

    def as_dict(self):
        settings = {}
        settings['vaspdir'] = self.vaspdir
        settings['prim_file'] = self.prim_file
        settings['calc_data_file'] = self.calc_data_file
        settings['ce_file'] = self.ce_file
        settings['ce_radius'] = self.ce_radius
        settings['max_de'] = self.max_de
        settings['max_ew'] = self.max_ew
        settings['sm_type'] = self.sm_type
        settings['ltol'] = self.ltol
        settings['stol'] = self.stol
        settings['angle_tol'] = self.angle_tol
        settings['vor_tol'] = self.vor_tol
        settings['solver'] = self.solver
        settings['basis'] = self.basis
        settings['weight'] = self.weight
        return settings
    
    def write_settings(self,settings_file='analyzer_settings.mson'):
        print('Writing anlyzer settings to {}'.format(settings_file))
        with open(settings_file,'w') as fout:
            json.dump(self.as_dict(),fout)
