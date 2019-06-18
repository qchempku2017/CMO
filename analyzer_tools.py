#!/usr/bin/env python

from __future__ import division
from __future__ import unicode_literals

__author__ = 'Bin Ouyang & Fengyu_xie'
__version__ = 'Dev'


import os
import sys
import json
from monty.json import MSONable
import random
from copy import deepcopy
import numpy as np
import numpy.linalg as la

from pymatgen.analysis.structure_matcher import StructureMatcher, ElementComparator
from pymatgen.analysis.local_env import CrystalNN
from pyabinitio.cluster_expansion.eci_fit import EciGenerator
from pyabinitio.cluster_expansion.ce import ClusterExpansion
from pymatgen.io.vasp.inputs import *
from pymatgen.io.vasp.outputs import *
from pymatgen.io.cif import CifParser
from pymatgen import Structure
from pymatgen.analysis.elasticity.strain import Deformation
from pymatgen.core.sites import PeriodicSite
from monty.json import MSONable

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
class CalcAnalyzer(MSONable):

    def __init__(self, vaspdir='vasp_run', prim_file='prim.cif',calc_data_file='calcdata.mson',ce_file='ce.mson',ce_radius=None,max_de=100,max_ew=3,max_deformation={'ltol':0.2,'stol':0.15,'angle_tol':5}):
        self.calcdata = {}
        self.vaspdir = vaspdir
        self.calc_data_file = calc_data_file
        self.ce_file = ce_file
        self.max_deformation = max_deformation

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
            self.max_ewald = max_ew

            self.ce = ClusterExpansion.from_radii(self.prim, ce_radius,ltol=max_deformation['ltol'], \
                                     stol=max_deformation['stol'], angle_tol=max_deformation['angle_tol'],\
                                     supercell_size='volume',use_ewald=True,use_inv_r=False,eta=None);
    
            self.max_deformation = max_deformation

            self._load_data()

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
        self.ECIG=EciGenerator.unweighted(cluster_expansion=self.ce, structures=ValidStrs,\
                                     energies = energies,\
                                     max_dielectric=self.max_de, max_ewald=self.max_ew);
    
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
            if _is_vasp_calc(files):
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
                    continue
    
                # Get final energy from OSZICAR or Vasprun. Vasprun is better but OSZICAR is much
                # faster and works fine is you separately check for convergence, sanity of
                # magnetic moments, structure geometry
                with open(os.path.join(root, 'OUTCAR')) as outfile:
                    outcar_string = outfile.read()
                if 'reached required accuracy' not in outcar_string:
                    print('Instance {} did not converge to required accuracy. Skipping.'.format(root))
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
                n_matched += 1
    
        # Data already deduplicated!
        print('{}/{} structures matched in this run. Parsed vasp data saved into {}.'.format(n_matched,n_inputs,self.calc_data_file))
 
    def write_files(self):
        with open(self.calc_data_file,'w') as Fout:
            json.dump(self.calcdata,Fout)
        with open(self.ce_file,'w') as Fout:
            json.dump(self.ECIG.as_dict(),Fout)
