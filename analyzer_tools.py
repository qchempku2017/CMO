#!/usr/bin/env python

from __future__ import division
from __future__ import unicode_literals

__author__ = 'Bin Ouyang & Fengyu_xie'
__version__ = 'Dev'


import os
import sys
import argparse
import json
import random
from copy import deepcopy
import numpy as np
import numpy.linalg as la
from pymatgen.analysis.structure_matcher import StructureMatcher, ElementComparator
from pymatgen.analysis.loacl_env import CrystalNN
from pyabinitio.cluster_expansion.eci_fit import EciGenerator
from pyabinitio.cluster_expansion.ce import ClusterExpansion
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from itertools import groupby
from pymatgen.io.vasp.sets import MITRelaxSet
from pymatgen.io.vasp.inputs import *
from pymatgen.io.vasp.outputs import *
from pymatgen.io.cif import *
from pymatgen import Structure
from pymatgen.core.periodic_table import Specie
from pymatgen.core.composition import Composition
from pymatgen.core.sites import PeriodicSite
from itertools import permutations,product
from operator import mul
from functools import partial,reduce
import multiprocessing
from mc import *
from OxData import OXRange #This should be a database like file
import collections

default_radius = {2:7, 3:4.1, 4:4.1}

def _mapping_and_screening(data_file):
    """
        May relaxed structures appears too far off input structures. Filter them out and update the 
        database. Also deduplicate here since two diffrent inputs might result in a same output.
    """


def _fit_ce(calc_data_file, ce_file, ce_radius=None):
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
    prim = Structure.from_dict(calc_data['prim'])

    #Use crystal nearest neighbor analyzer to find nearest neighbor distance, and set cluster radius according to it.
    
    ValidStrs = []
    for CalcInd, Calc in enumerate(calc_data['relaxed_structures']):
        Str=Structure.from_dict(Calc);
        ValidStrs.append(Str);
    ## These should already have been deduplicated

    cnn = CrystalNN()
    nn = cnn.find_nn_info(ValidStrs[0],0)[0]['site']
    d_nn = ValidStrs[0][0].distance(nn)

    if not ce_radius:
        ce_radius = {}
        ce_radius[2]=d_nn*4
        ce_radius[3]=d_nn*2
        ce_radius[4]=d_nn*2
    
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
                                 energies = calc_data['total_eneriges'],\
                                 max_dielectric=max_de, max_ewald=max_ew);

    print("RMSE: {} eV/prim".format(ECIG.rmse)); 
    
    ce_data_new = ECIG.as_dict()
    ce_data_new['rmse_current']=ECIG.rmse
    with open(ce_file,'w') as ce_stream:
        json.dump(ce_data_new,ce_stream)

    print('{} updated.'.format(ce_file))
    

def _dedup(calc_data_dict):
    """
    Deduplicate list of structures by structure matching. Only comparing elements, not species.
    """
    unique_data_dict={}
    unique_data_dict['prim'] = calc_data_dict['prim']
    unique_data_dict['input_structures'] = []
    unique_data_dict['relaxed_structures'] = []
    unique_data_dict['total_energies'] = []
    unique_data_dict['magmoms'] = []
    unique_data_dict['compositions'] = []
    unique_data_dict['supercell_matrices'] = []

    sm = StructureMatcher(stol=0.1, ltol=0.1, angle_tol=1, comparator=ElementComparator())
    for entry_i, str_data in enumerate(calc_data_dict['relaxed_structures']):
        is_unique=True;
        struct = Structure.from_dict(str_data)
        for entry_j, ustr_data in enumerate(unique_data_dict['relaxed_structures']):
            ustruct = Structure.from_dict(ustr_data)
            if sm.fit(struct,ustruct):
                is_unique = False
                break
        if is_unique:
            unique_data_dict['input_structures'].append(calc_data_dict['input_structures'][entry_i])
            unique_data_dict['relaxed_structures'].append(calc_data_dict['relaxed_structures'][entry_i])
            unique_data_dict['total_energies'].append(calc_data_dict['total_energies'][entry_i])
            unique_data_dict['magmoms'].append(calc_data_dict['magmoms'][entry_i])
            unique_data_dict['compositions'].append(calc_data_dict['compositions'][entry_i])
            unique_data_dict['supercell_matrices'].append(calc_data_dict['supercell_matrices'][entry_i])

    print('Selected {} out of {} structures after deduplication.'.format(len(unique_data_dict['compositions']),\
          len(calc_data_dict['compositions'])))

    return unique_data_dict;

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
    DefaultOx={'Li':1,'F':-1,'O':-2}; OxLst=[];
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

def _load_data(vaspdir,ce_file,calc_data_file):
    """
    Args:
        LoadDirs: List of directories to search for VASP runs
        OutFile: Savefile
    This function parses existing vasp calculations, does mapping check, assigns charges and writes into the calc_data file 
    mentioned in previous functions. What we mean by mapping check here, is to see whether a deformed structure can be mapped
    into a supercell lattice and generates a set of correlation functions in clustersupercell.corr_from_structure.
    
    We plan to do modify corr_from_structure from using pymatgen.structurematcher to a grid matcher, which will ensure higher 
    acceptance for DFT calculations, but does not necessarily improve CE hamitonian, since some highly dipoled and deformed 
    structures might have poor DFT energy, and even SABOTAGE CE!
    """

    calc_data_dict = {}
    # Load VASP runs from given directories
    for Root,Dirs,Files in os.walk(vaspdir):
        if "OSZICAR" in Files and 'CONTCAR' in Files and 'OUTCAR' in Files and '3.double_relax' in Root:
            try:
                Name=os.sep.join(Root.split(os.sep)[0:-1]);
                Str=Poscar.from_file(os.path.join(Root,"CONTCAR")).structure
                ValidComp=True;
                    for Ele in Str.composition.element_composition.elements:
                        if str(Ele) not in ['Li', 'Mn', 'O', 'F']: ValidComp=False;break;
                    if not ValidComp: continue;
                    print("Loading VASP run in {}".format(Root));
                    # Rescale volume to that of unrelaxed structure
                    OStr=Poscar.from_file(os.path.join(os.sep.join(Root.split(os.sep)[0:-2]),"POSCAR")).structure
                    OVol=OStr.volume; VolScale=(OVol/Str.volume)**(1.0/3.0);
                    Str=Structure(lattice=Lattice(Str.lattice.matrix*VolScale),
                            species=[Site.specie for Site in Str.sites],
                            coords=[Site.frac_coords for Site in Str.sites]);
                    # Assign oxidation states to Mn based on magnetic moments in OUTCAR
                    Out=Outcar(os.path.join(Root,'OUTCAR')); Mag=[];
                    for SiteInd,Site in enumerate(Str.sites):
                        Mag.append(np.abs(Out.magnetization[SiteInd]['tot']));
                    Str=_assign_ox_states(Str,Mag);
                    # Throw out structures where oxidation states don't make sense
                    if np.abs(Str.charge)>0.1:
                        print(Str);print("Not charge balanced .. skipping");continue;
                    # Get final energy from OSZICAR or Vasprun. Vasprun is better but OSZICAR is much
                    # faster and works fine is you separately check for convergence, sanity of
                    # magnetic moments, structure geometry
                    if VASPRUN: TotE=float(Vasprun(os.path.join(Root, "vasprun.xml")).final_energy);
                    else: TotE=Oszicar(os.path.join(Root, 'OSZICAR')).final_energy;
                    # Make sure run is converged
                    with open(os.path.join(Root,'OUTCAR')) as OutFile: OutStr=OutFile.read();
                    assert "reached required accuracy" in OutStr; Comp=Str.composition;
                    Data.append({'s':Str.as_dict(),'toten':TotE,'comp':Comp.as_dict(),\
                        'path':Root,'name':Name});
                except: print("\tParsing error - not converged?")
    # Deduplicate data
    UData = _dedup(Data);
    with open(OutFile,"w") as Fid: Fid.write(json.dumps(UData));

