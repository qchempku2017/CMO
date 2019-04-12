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

###################################
#One example of OXRange we use is given below:
ox_ranges = {'Mn': {(0.5, 1.5): 2,
            (1.5, 2.5): 3,
            (2.5, 3.4): 4,
            (3.4, 4.05): 3,
            (4.05, 5.0): 2}};
#The way it works is really emperical but we did not find a generalized way to
#correlate charge states with magnetism. Maybe we can automate a self consistent routine
#to check the check balance during VASP caculation loading and pick a best one. Or just 
#Pre-calculate it and make it into a data file like json or yaml (works like the DFT+U paprameters);
##################################

VASPRUN = False

def fit_ce(InFileLst,Prim,OutFile):
    # Load calculations from list of json savefiles
    CalcData=[];
    for InFile in InFileLst:
        with open(InFile, 'r') as Fid:CalcData.extend(json.loads(Fid.read()));
    # Define cluster expansion, basis set, Have to do volume scaling in this configuration
    # could also do num_sites since there are no vacancies here
    CE=ClusterExpansion.from_radii(prim, {2: 7, 3: 4.1, 4: 4.1},ltol=0.15, \
                                     stol=0.2, angle_tol=2,supercell_size='volume',
                                     use_ewald=True,use_inv_r=False,eta=None);
    # Filter structures to those that map to the lattice
    ValidStrs = []
    for CalcInd, Calc in enumerate(CalcData):
        print('{}/{} ({})'.format(CalcInd,len(CalcData),Calc['name']))
        try:
            Str=Structure.from_dict(Calc['s']); CE.corr_from_structure(Str);
            ValidStrs.append(Calc);
        except: print("\tToo far off lattice, throwing out."); continue;
    print("{}/{} structures map to the lattice".format(len(valid_structs), len(calc_data)))
    # Fit expansion
    ECIG=EciGenerator.unweighted(cluster_expansion=CE,
                                 structures=[Structure.from_dict(Calc['s']) for Calc in ValidStrs],
                                 energies=[Calc['toten'] for Calc in ValidStrs],\
                                 max_dielectric=100,max_ewald=3);
    print("RMSE: {} eV/prim".format(ECIG.rmse)); dumpfn(ECIG,OutFile);

def _dedup(StrLst):
    """
    Deduplicate list of structures by structure matching.
    """
    for i, Str in enumerate(SLst):
        try: Str['s'] = Structure.from_dict(Str['s']);
        except: print("Error!");print(Str['s']);raise ValueError("Dedup error");
    ULst=[];
    SM=StructureMatcher(stol=0.1, ltol=0.1, angle_tol=1, comparator=ElementComparator())
    for i, Str in enumerate(SLst):
        Unique=True;
        for j, UStr in enumerate(ULst):
            if sm.fit(Str['s'], UStr['s']): Unique=False; break;
        if Unique: ULst.append(UStr);
    for UStr in ULst: UStr['s'] = UStr['s'].as_dict();
    return ULst;

def _assign_ox_states(Str,Mag):
    """
    Aassign oxidation states based on magnetic moments taken from the OUTCAR.
    Reference magnetic moments obtained by looking at a bunch of structures.

    DOES NOT CHECK THAT THE ASSIGNMENT IS CHARGE BALANCED!
    Args:
        Str: structure
        Mag: list of magnetic moments for the sites in s
    """
    # Oxidation states corresponding to a range of magnetic moments (min, max)
    ###OXRange imported from OxData.py
    DefaultOx={'Li':1,'F':-1,'O':-2}; OxLst=[];
    for SiteInd,Site in enumerate(Str.Sites):
        Assigned=False;
        if Site.species_string in OXRange.keys():
            for (MinMag,MaxMag),MagOx in OXRange[Site.species_string].items():
                if Mag[SiteInd]>=MinMag and Mag[SiteInd]<MaxMag:
                    OxLst.append(MagOx); Assigned=True; break;
        elif Site.species_string in DefaultOx.keys():
            OxLst.append(DefaultOx[Site.species_string]); Assigned=True;
        if not Assigned:
            print("Cant assign ox states for site={}, mag={}".\
                    format(Site,Mag[SiteInd])); assert Assigned;
    Str.add_oxidation_state_by_site(OxLst);
    return Str;

def load_data(LoadDirs,OutFile):
    """
    Args:
        LoadDirs: List of directories to search for VASP runs
        OutFile: Savefile
    """
    Data=[];
    # Load VASP runs from given directories
    for LoadDir in LoadDirs:
        for Root,Dirs,Files in os.walk(LoadDir):
            if "OSZICAR" in Files and 'CONTCAR' in Files and 'OUTCAR' in Files\
                    and '3.double_relax' in Root:
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

def _get_ind_groups(Bits,Cations,Anions):
    """
    Define sublattices for monte carlo flips
    """
    i1 = [i for i, b in enumerate(Bits) if sorted(b) == sorted(Cations)];
    i2 = [i for i, b in enumerate(Bits) if sorted(b) == sorted(Anions)];
    return i1, i2;

def get_mc_structs(CEFile,CalcFiles,OutDir,SCLst,Prim=None,TLst=[500, 1500, 10000]):
    '''For CE sampling using MC, use three set of temperature, merge this with LocalOrdering code
       CEFile: directory of CE Mson data file
       CalcFiles: directory of vasp calculation Mson data files
       OutDir: directory to write outputs
       SCLst: a list contaning enumerated SC's and RO pairs.
       Prim: primitive cell read from cif file
       TLst: temprature list to do MC enumeration on
       useX: a list of compounds of which we may want to calculate component.
    '''
    print('#### MC Initialization ####')
    if CEFile:
        # Load cluster expansion
        with open(CEFile,'r') as Fid: ECIG=EciGenerator.from_dict(json.loads(Fid.read()));
        CalcData=[];
        if CalcFiles:
            for CalcFile in CalcFiles:
                with open(CalcFile,'r') as Fid: CalcData.extend(json.loads(Fid.read()));
        else: print("Not checking versus previous calculations")
        CE=ECIG.ce; ECIs=deepcopy(ECIG.ecis); print('ce information:'); print(ce.structure);
    else:
        # No existing cluster expansion - use electrostatics only
        CalcData=[];
        CE=ClusterExpansion.from_radii(Prim,{2: 1},ltol=0.3,stol=0.2,angle_tol=2,\
                                       supercell_size='num_sites',use_ewald=True,use_inv_r=False,eta=None);
        ecis=np.zeros(CE.n_bit_orderings+1); ecis[-1]=1;
        print('Primitive cell read from CIF file:\n',Prim)

    mc_structs={};

    for SC,RO,composition,sites_WorthToExpand in SCLst:
        compositionFrac = {}
        totMole = sum(composition.values())
        for compound in composition:
            compositionFrac[compound]=float(composition[compound])/totMole
        print("Processing composition:\n",compositionFrac,'\nSupercell:\n',SC,'\nsize:\n',int(round(np.abs(np.linalg.det(SC)))))

        clusSC=CE.supercell_from_matrix(SC);
        #print(clusSC.supercell)
        #print(clusSC.bits)
        # Define cation/anion sublattices
        ions=[]
        for subLat in RO:
            for specie in RO[subLat]:
                if specie not in ions: ions.append(specie)
        #cations = [_Back_Modify(ion) for ion in ions if (ion[-1]=='+' or (ion[-1]!='+' and ion[-1]!='-'))]
        #cations = cations
        #anions = [_Back_Modify(ion) for ion in ions if ion[-1]=='-']
        #anions = anions 
        #print('cations',cations,'anions',anions)

        Bits=clusSC.bits;
        scs = int(round(np.linalg.det(SC)))
        #print('supercell',clusSC.supercell)
        # generate a list of groups of sites to swap between! We have known which sites are partially occupied,
        # so we only need to figure out how pymatgen make a group of supercell sites from a primitive cell site.
        # Looks like it simply just replicate sites one by one! 
        indGrps=[list(range(i*scs,(i+1)*scs)) for i in range(len(RO)) if sites_WorthToExpand[i]];
        #Note: indGrps should be generated from a clusSC supercell!!!!!

        #print('indGrps',indGrps,'RO',RO)         

        # Replace species according to RO
        randSites = []
        for i,site in enumerate(Prim):
             randSite = PeriodicSite(RO[i],site.frac_coords,Prim.lattice,properties=site.properties)
             randSites.append(randSite)
        randStr = Structure.from_sites(randSites)
            
        # Get electrostatics enumeration guess
        order=OrderDisorderedStructureTransformation(algo=2);

        randStr.make_supercell(SC)

        #print('randStr',randStr)
        #print('RO',RO);
        #print('indGrps',indGrps)

        # A lot of things crash is the structure is actually ordered and the MC has zero degrees of freedom
        # If strange errors come up in this section, thats probably the cause - sometimes due to roundoff
        # error, this check thinks the structure isnt ordered, while in reality is it, and then stuff crashes
        # Currently, checks if the composition is actually LiF or MnO and doesnt run MC on that
        #if np.abs(xf - 1.0) > 0.001 and np.abs(xm2 - 1.0) > 0.001:
        randStr =order.apply_transformation(randStr)
        #print('randStr:',randStr)
        #print('ceStr:',clusSC.supercell)
        # Simulated annealing for better guess at ground state
        # You may want to change the number of MC flips for each temperature
        #print(type(clusSC))
        init_occu = clusSC.occu_from_structure(randStr)
        sa_occu = simulated_anneal(ecis=ecis, cluster_supercell=clusSC, occu=init_occu, ind_groups=indGrps,
                                   n_loops=20000, init_T=5100, final_T=100, n_steps=20)

        # Integrate Wenxuan's solver here, and abandon MC annealing.#

        compStr = str()
        for compound in compositionFrac:
            compStr+=(compound+'_{0:.3f}_'.format(compositionFrac[compound]))
        compStr = compStr[:-1]
        if compStr not in mc_structs:
            mc_structs[compStr]=[]

        # Add approximate ground state to set of MC structures
        # Format as (structure, temperature) - for ground state, temperature is "0"
        mc_structs[compStr].append((clusSC.structure_from_occu(sa_occu),0))

        for T in TLst:
            print("\t T = {}K".format(T))
            # Equilibration run
            # Play around with the number of MC flips in the run - the current number is very arbitrary
	    # We can try to implement VO et.al's sampling method here!

            occu, _, _, _ = run_T(ecis=ecis, cluster_supercell=clusSC, occu=deepcopy(sa_occu), 
                                 T=T, n_loops=100000, ind_groups=indGrps, n_rand=2, check_unique=False)

            # Production run
            # Same comment about number of flips - very arbitrary right now
            occu, min_occu, min_e, rand_occu = run_T(ecis=ecis, cluster_supercell=clusSC, occu=deepcopy(occu),
                                 T=T, n_loops=200000, ind_groups=indGrps, n_rand=6, check_unique=True) 
            
            # Check that returned random structures
            # are all different
	    # Save best structure and a few random structures from the production run
            mc_structs[compStr].append((clusSC.structure_from_occu(min_occu),T))
            for rand, rand_e in rand_occu:
                mc_structs[compStr].append((clusSC.structure_from_occu(rand),T))

    # Deduplicate - first versus previously calculated structures, then versus structures within this run
    print('Deduplicating random structures.')
    calculated_structures = []

    for calc_i, calc in enumerate(CalcData):
        calculated_structures.append(Structure.from_dict(calc['s']))

    unique_structs = {}
    unqCnt = 0
    sm = StructureMatcher(ltol=0.3, stol=0.3, angle_tol=5, comparator=ElementComparator())
    for compStr,structs in mc_structs.items():
        if compStr not in unique_structs:
            unique_structs[compStr] = []
        for struct,T in structs:
            unique = True
            for ostruct in calculated_structures:
                if sm.fit(struct, ostruct):
                    unique = False
                    break
            for ostruct,T in unique_structs[compStr]:
                if sm.fit(struct, ostruct):
                    unique = False
                    break
            if unique:
                unique_structs[compStr].append((struct,T))
                unqCnt += 1
    print('Obtained %d unique occupied random structures.'%unqCnt)

    # Save structures
    print('#### MC Final Saving ####')
    if not os.path.isdir(OutDir):
        os.mkdir(OutDir)
        print(OutDir,' does not exist. Created.')
    for compStr,structs in unique_structs.items():
        for i, (struct,T) in enumerate(structs):
            compDir = compStr
            compPathDir = os.path.join(OutDir,compDir)
            if not os.path.isdir(compPathDir): os.mkdir(compPathDir)
            structDir = os.path.join(compPathDir,str(i))
            if not os.path.isdir(structDir): os.mkdir(structDir)
            Poscar(struct.get_sorted_structure()).write_file(os.path.join(structDir,'POSCAR'))
    print('Saving of %s successful.'%OutDir)

def _write_vasp_inputs(Str,VASPDir):
    # This is a somewhat strange input set. Essentially the matgen input set (PBE+U), but with tigher
    # convergence.
    # This is also a somewhat outdated and convoluted way to generate VASP inputs but it should work fine.
    # These changes to the default input set give much better results.
    # Do not increaes the EDIFF to make it converge faster!!!
    # If convergence is too slow, reduce the K-points
    # This is still using PBE+U with matgen U values though. Need to use MITCompatibility (after the run)
    # to apply oxygen corrections and such.
    # In other expansions that rely on SCAN or HSE, the corrections are different - no O correction for example
    VASPSettings={"ALGO": 'VeryFast',"ISYM": 0, "ISMEAR": 0, "EDIFF": 1e-6, "NELM": 400, "NSW": 1000, "EDIFFG": -0.02,
                     'LVTOT': False, 'LWAVE': False, 'LCHARG': False, 'NELMDL': -6, 'NELMIN': 8,
                     'LSCALU': False, 'NPAR': 2, 'NSIM': 2, 'POTIM': 0.25, 'LDAU': True};
    if not os.path.isdir(VASPDir):os.mkdir(VASPDir);
    # Joggle the lattice to help symmetry broken relaxation
    FracCoords=[Site.frac_coords for Site in Str.sites];
    Species=[Site.specie for Site in Str.sites]; Latt=Str.lattice;
    StrainedLatt=Lattice.from_lengths_and_angles([Latt.a*1.01,Latt.b*1.05,Latt.c*1.03],
                                                [Latt.alpha,Latt.beta,Latt.gamma]);
    Str=Structure(StrainedLatt,Species,FracCoords,to_unit_cell=False,coords_are_cartesian=False);
    VIO=MITRelaxSet(Str,potcar_functional='PBE'); VIO.user_incar_settings=VASPSettings;
    VIO.incar.write_file(os.path.join(VASPDir,'INCAR'));
    VIO.poscar.write_file(os.path.join(VASPDir,'POSCAR'));
    Kpoints.automatic(25).write_file(os.path.join(VASPDir,'KPOINTS'));
    # Use PAW_PBE pseudopotentials, cannot use PBE_52, this does not exist on ginar!
    # NOTE: For the POTCARs to work, you need to set up the VASP pseudopotential directory as per the
    # pymatgen instructions, and set the path to them in .pmgrc.yaml located in your home folder.
    # The pymatgen website has instuctrions for how to do this.
    POTSyms=VIO.potcar_symbols;
    for i, Sym in enumerate(POTSyms):
        if Sym == 'Zr': POTSyms[i]='Zr_sv';
    Potcar(POTSyms,functional='PBE').write_file(os.path.join(VASPDir,'POTCAR'));

def run_vasp(RunDir):
    """
    Run vasp for all structures under RunDir.
    """
    absRunDir = os.path.abspath(RunDir)
    parentDir = os.path.dirname(absRunDir)
    POSDirs=[];
    _is_VASP_Input = lambda files: ('INCAR' in files) and \
                     ('POSCAR' in files) and ('POTCAR' in files)\
                     and ('KPOINTS' in files)

    for Root,Dirs,Files in os.walk(RunDir):
        if _is_VASP_Input(Files) and 'fm' in Root: POSDirs.append(Root);
    for Root in POSDirs:
        runRoot = os.path.abspath(Root)
        os.chdir(runRoot)
        print("Submitting VASP for {}".format(os.getcwd()))
        os.system("qsub "+parentDir+"/sub.sh")
        print("Submitted VASP for {}".format(os.getcwd()))
        os.chdir(parentDir)
    print('Submission works done.')


def gen_vasp_inputs(SearchDir):
    """
    Search through directories, find POSCARs and generate FM VASP inputs.
    """
    POSDirs=[];
    for Root,Dirs,Files in os.walk(SearchDir):
        for File in Files:
            if File == 'POSCAR' and 'fm' not in Root: POSDirs.append([Root,File]);
    for [Root,File] in POSDirs:
        print("Writing VASP inputs for {}/{}".format(Root,File));
        Str=Poscar.from_file(os.path.join(Root,File)).structure
        VASPDir= os.path.join(Root,'fm.0'); _write_vasp_inputs(Str,VASPDir);

def _get_converged_E_at_T(T,ECIs=None,cluster_supercell=None, init_occu=None, init_e=None, ind_groups=None):
    """
    Utility function for running MC in parallel load from LocalStructureAnalyzer
    """
    print("\t Running T = {}K".format(T))
    occu, min_occu, min_e, energies = run_T(ecis=ecis,
                                            cluster_supercell=cluster_supercell,
                                            occu=deepcopy(init_occu),
                                            T=T,
                                            n_loops=2000000,
                                            ind_groups=ind_groups,
                                            n_rand=0,
                                            check_unique=False,
                                            sample_energies=10000)
    return [T, energies, min_e, min_occu]

def _GetIonChg(ion):
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

def _factors(n):
    """
    This function take in an integer n and computes all integer multiplicative factors of n

    """
    return set(reduce(list.__add__,
                      ([i, n // i] for i in range(1, int(n ** 0.5) + 1) if n % i == 0)))

def _Get_Hermite_Matricies(Num):
    """
    This function take in an integer and computes all
    Hermite normal matricies with determinant equal to this integer
    All random real matrices can be transformed into a upper-triangle matrix by unitary
    transformations.
    Note:
    clustersupercell.supercell_from_sc does not take an np.matrix! It takes a matrix-like
    3*3 list!!!
    """
    Mats = []; Factors = list(_factors(Num)); Factors *= 3;
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

def _mat_mul(mat1,mat2):
    A = np.matrix(mat1)
    B = np.matrix(mat2)
    return (A*B).tolist()

def _Enumerate_SC(maxDet,prim,nSk=1,nRect=1,transmat=None):
    '''
    Enumerate all possible supercell matrices and pick 10 random unskewd scs 
    and 10 skewed scs from enumeration.
    '''
    print('#### Supercell Enumeration ####')
    scs=[]
    for det in range(int(maxDet/4),maxDet+1,int(maxDet/4)):
        scs.extend(_Get_Hermite_Matricies(det))
    print('Generated %d supercell matrices with max determinant %d'%(len(scs),maxDet))
    #print('Supercell Matrices:\n',scs)
    print('Picking %d random skew supercells and %d random rectangular supercells.'%(nSk,nRect))
    _is_diagonal = lambda sc: (sc[0][1]==0 and sc[0][2]==0 and sc[1][2]==0)
    scs_sk = [sc for sc in scs if not _is_diagonal(sc)]
    scs_re = [sc for sc in scs if _is_diagonal(sc)]
    selected_scs = random.sample(scs_sk,nSk)+random.sample(scs_re,nRect)
    #print("scs before trans:",selected_scs)
    if transmat:
        selected_scs=[_mat_mul(sc,transmat) for sc in selected_scs]
    return selected_scs

def _FindSpecieSite(specie,occuDict):
    for site in occuDict:
        if specie in occuDict[site]: return site

def _Modify_Specie(specie):
    if not specie[-2].isdigit():
        specie = specie[:-1]+'1'+specie[-1]
    return specie

def _Back_Modify(specie):
    if specie[-2]=='1':
        specie = specie[:-2]+specie[-1]
    return specie

def _Is_Neutral_Occu(occu,specieChgDict):
    totalChg = 0
    for site in occu:
        for specie in site:
            totalChg += site[specie]*specieChgDict[specie]
    return totalChg==0

def Supercells_From_Compounds(maxSize,prim,compounds,enforceOccu=None,sampleStep=1,supercellnum=1,transmat=None):
    #Warning: Currently assumes a specie only occupies one site.
    '''
    In this function supercell replacement maps are no longer enumerated by site occupation, but
    by compound component ratios.
        maxSize: determinant of supercell matrix that is to be enumerated
        primData: primitive cell data. Generated by pymatgen.io.cif.CifParser(). Initially a pure compound, for 
                  instance, LiNiO2 in /Examples
        occuDict: a Dict, specified by the user to indicate which sublattice should be occupied by which species.
                  usually in the form like {'sublattice1':['specie1','specie2',...],...}. For ex, {'Li+':['Mn3+',
                  'Mn4+'],...}(1,Specie name in occudict.keys() are only a symbol that mark sites in lattice. 2,
                  Do not use 'Vac' as an occupation specie, and every site in cif should be filled! 3,By specifying
                  occupation number as <0.99 in initial CIF, you tell this program that you want to expand this site
                  represented by some specie string!)
        enforceOccu: Sometimes you want a sublattice to be always fully occupied or at least half occupied. Use this 
                     parameter and input a dict, such as {'Li+':0.333,...}.
        sampleStep: Sometimes you don't want to change site occupation one by one. Like, if you have 24 A sites, you 
                   wanna sample every 6 Li+ occupation so you will have less possibility to handle with and less 
                   memory consumption. We recommend you use common factor(gcd) of 'nSites', or charge balance may
                   not be easily satisfied!!
    Charge balance ensured.
    The list returned will be a list of tuples in the form (SC,RO,comp) now.
    '''
    #Preparation steps
    SCenum = _Enumerate_SC(maxSize,prim,supercellnum,supercellnum,transmat)

    specieChgDict={}
    
    occuDicts = []
    for site in prim:
        siteSpecies = site.species_string.split(',')
        #print('siteSpecies',siteSpecies)
        if len(siteSpecies)>1 or (':' in siteSpecies[0]):
            siteOccu = dict([[s.strip() for s in specieoccu.split(':')] for specieoccu in siteSpecies])
        else:
            siteOccu = {siteSpecies[0]:1.00}
        #print(siteOccu)
        siteOccuMod = {}
        for specie in siteOccu:
            specieMod = _Modify_Specie(specie)
            if specieMod not in specieChgDict: specieChgDict[specieMod]=_GetIonChg(specieMod)
            siteOccuMod[specieMod]=float(siteOccu[specie])
        occuDicts.append(siteOccuMod)
    #print(occuDicts)

    compSpecieNums = {}
    # Get representative specie in a compound to make a calculation of composition from site enumeration easier.
    compUniqSpecies = {}
    for compStr in compounds:
        comp = Composition(compStr)
        compSpecieNums[compStr]={}
        try:
            compChgDict = comp.oxi_state_guesses()[0]
        except:
            raise ValueError('Cannot use compound with non-integer valence as base compound!')
        compNumDict = comp.get_el_amt_dict()
        for specie in compChgDict:
            specieStr = specie+str(abs(int(compChgDict[specie])))+('+' if compChgDict[specie]>=0 else '-')
            compSpecieNums[compStr][specieStr]=compNumDict[specie]

    # print(compSpecieNums,specieChgDict)
    # print(compounds)
    for compStr in compounds:
        for specie in compSpecieNums[compStr]:
            specieIsUniq = True
            for otherCompStr in compounds:
                if ((specie in compSpecieNums[otherCompStr]) and compStr!=otherCompStr): 
                    #print('Specie',specie,'in',compStr,'is not unique')
                    specieIsUniq = False
            if specieIsUniq:
                compUniqSpecies[compStr]=specie
                break

    for compound in compounds:
        if compound not in compUniqSpecies:
             print('Specified reference compound',compound,'Does not have a unique specie. Exiting!')
             sys.exit()
                         
    uniqSpecieComps = {val:key for key,val in compUniqSpecies.items()}
    #print('uniqSpecieComps',uniqSpecieComps)

    print('#### Occupation enumeration ####')
    SCLst = []
    for SC in SCenum:
        SCSize = np.abs(np.linalg.det(SC))
        numSites = collections.Counter()
        for site in range(len(prim)):
            numSites[site]+=int(round(SCSize))

    # Enumeration by site, not by compounds' chemical formulas
    # We will need to provide a site-by-site replacement method and abandon pymatgen replace_species
        poss_Occu_Sites = {}
        for site in numSites:
            poss_Sp_Nums = {}
            for specie in occuDicts[site]:  
               poss_Sp_Nums[specie]=list(range(0,int(numSites[site])+1,sampleStep))
            keys,values = zip(*poss_Sp_Nums.items())
            allOccu_for_Site = [dict(zip(keys,v)) for v in itertools.product(*values) if (((enforceOccu and\
            sum(v)/numSites[site]+0.01>enforceOccu[site]) or (not enforceOccu)) and (sum(v)/numSites[site]\
            -0.01<1.0))]
            #print(allOccu_for_Site)
            poss_Occu_Sites[site] = allOccu_for_Site
        ks,vs = zip(*poss_Occu_Sites.items())
        allOccu_for_Sites = [dict(zip(ks,v)) for v in itertools.product(*vs) if _Is_Neutral_Occu(v,specieChgDict) ]

        #print(allOccu_for_Sites)
        #Calculate compositions and fractional occupation for each site.
        allCompositions = []
        allFracOccu_for_Sites = []
        occus_WorthToExpand = []
        for occu in allOccu_for_Sites:
            occuComposition=collections.Counter()
            #These two counters are used to prevent unreasonably generated occupations that cannot be matched with any compound.
            specieStat_from_Occu=collections.Counter()
            specieStat_from_Compounds=collections.Counter()
            fracOccu = {}
            for site in occu:
                #print(occu)
                siteFracOccu = {}
                for specie in occu[site]:
                    siteFracOccu[specie]=occu[site][specie]/numSites[site]
                    specieStat_from_Occu[specie]+=occu[site][specie]
                    if specie in uniqSpecieComps:
                        corrCompound = uniqSpecieComps[specie]
                        occuComposition[corrCompound]+=occu[site][specie]/compSpecieNums[corrCompound][specie]
                        #print(occuComposition)
                fracOccu[site]=siteFracOccu
            for compound in occuComposition:
                for specie in compSpecieNums[compound]:
                    specieStat_from_Compounds[specie]+=compSpecieNums[compound][specie]*occuComposition[compound]
            specieNumMatch = True
            for specie in specieStat_from_Occu:
                if abs(specieStat_from_Occu[specie]-specieStat_from_Compounds[specie])>0.01:
                    specieNumMatch = False
                    break

            #Check whether the generated composition is a fully occupied and pure compound, and avoid to do CE on that.
            sites_WorthToExpand = []
            for site in fracOccu:
                site_WorthToExpand = True
                for specie in fracOccu[site]:
                    if abs(fracOccu[site][specie]-1.00)<0.001:
                        site_WorthToExpand=False
                        break
                sites_WorthToExpand.append(site_WorthToExpand)
                
            worthToExpand = reduce(lambda x,y:x or y,sites_WorthToExpand)
                    
            #print('fracOccu',fracOccu,'Composition',occuComposition)
            if specieNumMatch and worthToExpand:
                allFracOccu_for_Sites.append(fracOccu)
                allCompositions.append(occuComposition)
                occus_WorthToExpand.append(sites_WorthToExpand)
                #print(occuComposition,'expanded')

        SCLst.extend(zip([SC]*len(allCompositions),allFracOccu_for_Sites,allCompositions,occus_WorthToExpand))

        print('Generated %d compositions for supercell '%len(allCompositions),SC,'.')

    #print(SCLst)
    return SCLst

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
