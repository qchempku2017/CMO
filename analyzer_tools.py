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

from OxData import OXRange #This should be a database like file


def fit_ce(calc_data_file='calcdata.mson', ce_file='ce.mson', ce_radius=None):
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
        Outputs:
            None. The ce_data file will be updated.

    """
    calc_data = json.load(open(calc_data_file,'r'))

    #Use crystal nearest neighbor analyzer to find nearest neighbor distance, and set cluster radius according to it.
    
    ValidStrs = []
    energies = []
    for comp in calc_data['compositions']:
        for entry in calc_data['compositions'][comp]:
            ValidStrs.append(Structure.from_dict(entry['relaxed structure']))
            energies.append(entry['total_energy'])

    ## These should already have been deduplicated

    prim = Structure.from_dict(calc_data['prim'])
    cnn = CrystalNN()
    all_nn_d = []
    for i in range(len(prim)):
        nns = cnn.find_nn_info(prim,i)
        nn_d = min([nn['site'].distance(prim[i]) for nn in nns])
        all_nn_d.append(nn_d)
    d_nn=max(all_nn_d)

    if not ce_radius:
        ce_radius = {}
        # Default cluster radius
        ce_radius[2]=d_nn*4.0
        ce_radius[3]=d_nn*2.0
        ce_radius[4]=d_nn*2.0
    
    if os.path.isfile(ce_file):
        with open(ce_file,'r') as ce_old_stream:
            ce_data_old = json.load(ce_old_stream)
        CE = ClusterExpansion.from_dict(ce_data_old['cluster_expansion'])
        max_de = ce_data_old['max_dielectric']
        max_ew = ce_data_old['max_ewald']
    else:

        CE=ClusterExpansion.from_radii(prim, ce_radius,ltol=0.15, \
                                     stol=0.2, angle_tol=2,supercell_size='volume',
                                     use_ewald=True,use_inv_r=False,eta=None);
        max_de = 100
        max_ew = 3
    # Fit expansion, currently only support energy/free energy expansion. If you want to expand other properties, 
    # You have to write on your own.

    ECIG=EciGenerator.unweighted(cluster_expansion=CE, structures=ValidStrs,\
                                 energies = energies,\
                                 max_dielectric=max_de, max_ewald=max_ew);

    print("RMSE: {} eV/prim".format(ECIG.rmse)); 
    
    ce_data_new = ECIG.as_dict()
    if 'rmse_log' not in ce_data_new:
        ce_data_new['rmse_log']=[]

    ce_data_new['rmse_log'].append({'num_struct':len(ValidStrs),'rmse':ECIG.rmse})

    with open(ce_file,'w') as ce_stream:
        json.dump(ce_data_new,ce_stream)

    print('{} updated.'.format(ce_file))
    
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
    DefaultOx={'Li':1,'F':-1,'O':-2, 'Mg':2, 'Ca':2 }; OxLst=[];
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
    return Str;

def load_data(primfile='prim.cif', calc_data_file='calcdata.mson', vaspdir='vasp_run',\
               max_deformation={'ltol':0.2,'stol':0.1,'angle_tol':5}):
    """
    Args:
        vaspdir: List of directories to search for VASP runs
        prim_file: primitive cell file.
        ce_data_file: output database file.
        max_deformation: max tolerable deformation between input 
                         and output structure. Any relaxation over
                         this criteria should be dropped!

    This function parses existing vasp calculations, does mapping check, assigns charges and writes into the calc_data file 
    mentioned in previous functions. What we mean by mapping check here, is to see whether a deformed structure can be mapped
    into a supercell lattice and generates a set of correlation functions in clustersupercell.corr_from_structure.
    
    We plan to do modify corr_from_structure from using pymatgen.structurematcher to a grid matcher, which will ensure higher 
    acceptance for DFT calculations, but does not necessarily improve CE hamitonian, since some highly dipoled and deformed 
    structures might have poor DFT energy, and even SABOTAGE CE!
    """

    prim = CifParser(primfile).get_structures()[0]
    calc_data_dict = {}

    calc_data_dict['prim'] = ce.structure.as_dict()
    calc_data_dict['compositions']={}
    
    if os.path.isfile(calc_data_file):
        print("Loading existing calculation datas.")
        with open(calc_data_file) as old_calc:
            calc_data_dict = json.load(old_calc)

    # Every key of calc_data_dict['compositions'] is a composition, and each composition contains a list of dict entrees.
    # relaxed_structure, input_structure, supercell_matrix, magmoms, total_energy. 

    _is_vasp_calc = lambda fs: 'INCAR' in fs and 'POSCAR' in fs and 'KPOINTS' in fs and 'POTCAR' in fs and \
                               'CONTCAR' in fs and 'OSZICAR' in fs and 'OUTCAR' in fs
    # Load VASP runs from given directories

    _did_ax_decomp = False
    for root,dirs,files in os.walk(vaspdir):
        if _is_vasp_calc(files):
            try:
                print("Loading VASP run in {}".format(root));
                parent_root = os.path.join(*root.split(os.sep)[0:-1])
                parent_parent_root = os.path.join(*root.split(os.sep)[0:-1])

                relaxed_struct = Poscar.from_file(os.path.join(root,'CONTCAR')).structure
                input_struct = Poscar.from_file(os.path.join(parent_root,'POSCAR')).structure
                # Note: the input_struct here comes from the poscar in upper root, rather than fm.0, so 
                # it is not deformed.

                # Rescale volume to that of unrelaxed structure, this will lead to a better mapping back. 
                # I changed it to a rescaling tensor
                relaxed_lat_mat = np.matrix(relaxed_struct.lattice.matrix)
                input_lat_mat = np.matrix(input_struct.lattice.matrix)
                o2i_deformation = Deformation(input_lat_mat.T*relaxed_lat_mat.I.T)
                relaxed_deformed = o2i_deformation.apply_to_structure(relaxed_struct)

                # Assign oxidation states to Mn based on magnetic moments in OUTCAR
                Out=Outcar(os.path.join(root,'OUTCAR')); Mag=[];
                for SiteInd,Site in enumerate(relaxed_struct.sites):
                    Mag.append(np.abs(Out.magnetization[SiteInd]['tot']));
                relaxed_struct=_assign_ox_states(relaxed_struct,Mag);
                # Throw out structures where oxidation states don't make sense
                if np.abs(relaxed_struct.charge)>0.01:
                    print(relaxed_struct);print("Not charge balanced .. skipping");continue;

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

                with open(os.path.join(parent_parent_root,'composition_by_site')) as compfile:
                    composition = json.load(compfile)
                    compstring = json.dumps(composition)
                with open(os.path.join(parent_root,'supermat')) as supermat_file:
                    supermat = json.load(supermat_file)
                
                if compstring not in calc_data_dict['compositions']:
                    calc_data_dict['compositions'][compstring]=[]

                # Checking site deformation and mapping.
                sm = StructureMatcher(stol=max_deformation['stol'], ltol=max_deformation['ltol'],\
                                      angle_tol=max_deformation['angle_tol'], comparator=ElementComparator())

                # This is out deformation tolerance.
                if not sm.fit(relaxed_deformed, input_struct):
                    continue

                strict_sm = StructureMatcher(stol=0.1, ltol=0.1, angle_tol=1, comparator=ElementComparator())
                _is_unique = True
                for entry in calc_data_dict['compositons'][compstring]:
                    entry_struct = Structure.from_dict(entry['relaxed_structure'])
                    if strict_sm.fit(entry_struct,relaxed_struct):
                        _is_unique = False
                        break

                if _is_unique:
                    new_entry = {}
                    new_entry['input_structure'].append(input_struct.as_dict())
                    new_entry['relaxed_structure'].append(relaxed_struct.as_dict())
                    new_entry['total_energy'].append(TotE)
                    new_entry['magmoms'].append(Mag)
                    new_entry['supercell_matrices'].append(supermat)
                    if 'axis' in files:
                        _did_ax_decomp = True
                        with open(os.path.join(parent_parent_root,'axis')) as axisfile:
                            
                            axis = json.load(axisfile)
                        if 'axis' not in new_entry:
                            new_entry['axis']=axis
                    else:
                        if _did_ax_decomp:
                            print('Selected axis decomposition, but instance {} is not axis decomposed!'.format(root))
                            new_entry['axis'] = None
                    calc_data_dict['compositions'][compstring].append(new_entry)

            except: print("\tParsing error - calculation not finished?")
        else:
            print('\tParsing error - calculation not finished?')

    # Data already deduplicated!
    with open(calc_data_file,"w") as Fid: json.dump(calc_data_dict,Fid);
    print('Parsed vasp data saved into {}'.format(calc_data_file))

