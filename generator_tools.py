#!/usr/bin/env python

from __future__ import division
from __future__ import unicode_literals

__author__ = 'Bin Ouyang & Fengyu_xie'
__version__ = 'Dev'

import json
from monty.json import MSONable
import os
import sys
import random
from copy import deepcopy
import numpy as np
import numpy.linalg as la

from pymatgen.analysis.structure_matcher import StructureMatcher, ElementComparator
from pyabinitio.cluster_expansion.ce import ClusterExpansion
from pymatgen.io.vasp.sets import MITRelaxSet
from pymatgen.io.vasp.inputs import *
from pymatgen.io.vasp.outputs import *
from pymatgen.io.cif import *
from pymatgen import Structure
from pymatgen.core.periodic_table import Specie
from pymatgen.core.composition import Composition
from pymatgen.core.sites import PeriodicSite
from pymatgen.analysis.elasticity.strain import Deformation

from itertools import permutations,product
from operator import mul
from functools import partial,reduce
import multiprocessing
import collections

from mc import *
from global_tools import *

##################################
## Less general tools that are not cross refered by other modules
##################################
def _get_ind_groups(Bits,Cations,Anions):
    """
    Define sublattices for monte carlo flips
    """
    i1 = [i for i, b in enumerate(Bits) if sorted(b) == sorted(Cations)];
    i2 = [i for i, b in enumerate(Bits) if sorted(b) == sorted(Anions)];
    return i1, i2;

def _Enumerate_SC(maxDet,prim,nSk=1,nRect=1,transmat=None):
    '''
    Enumerate all possible supercell matrices and pick 10 random unskewd scs 
    and 10 skewed scs from enumeration.
    '''
    print('#### Supercell Enumeration ####')
    scs=[]
    trans_size = int(abs(np.linalg.det(transmat)))
    for det in range(int(maxDet/4),maxDet+1,int(maxDet/4)):
        scs.extend(_Get_Hermite_Matricies(int(det/trans_size)))
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

def _get_mc_structs(ce_file,outdir='vasp_run',SCLst,Prim=None,all_axis=None,TLst=[500, 1500, 10000]):
    '''For CE sampling using MC, use three set of temperature, merge this with LocalOrdering code
       ce_file: directory of CE Mson data file
       outdir: directory to write outputs
       SCLst: a list contaning enumerated SC's and RO pairs.
       Prim: primitive cell read from cif file
       TLst: temprature list to do MC enumeration on
       useX: a list of compounds of which we may want to calculate component.
    '''
    print('#### MC Initialization ####')
    calculated_structures = []
    if ce_file:
        # Load cluster expansion
        with open(ce_file,'r') as Fid: cedata = json.load(Fid);
        _was_generated = lambda x: 'POSCAR' in x and not 'KPOINTS' in x and not 'INCAR' in x and not 'POTCAR' in x
        if os.path.isdir(outdir):
            print("Checking previously enumerated structures.")
            for root,dirs,files in os.walk(outdir):
                if _was_generated(files):
                    calculated_structures.append(Poscar.from_file(os.join(root,'POSCAR').structure))
        else: 
            print("Not checking versus previous calculations")
        CE=ClusterExpansion.from_dict(cedata['cluster_expansion']); 
        ECIs=cedata['ecis']; 
        print('ce information:'); print(ce.structure);
        Prim = ce.structure

    else:
        # No existing cluster expansion, we are building form start - use electrostatics only
        print("Not checking versus previous calculations")
        CE=ClusterExpansion.from_radii(Prim,{2: 1},ltol=0.3,stol=0.2,angle_tol=2,\
                                       supercell_size='num_sites',use_ewald=True,use_inv_r=False,eta=None);
        ecis=np.zeros(CE.n_bit_orderings+1); ecis[-1]=1;
        print('Primitive cell read from CIF file:\n',Prim)

    mc_structs={};
    if all_axis:
        ro_axis_strings = {}

    sc_ro_pair_id = 0
    for SC,RO,sites_WorthToExpand in SCLst:
        print("Processing composition:\n",RO,'\nSupercell:\n',SC,'\nsize:\n',int(round(np.abs(np.linalg.det(SC)))))
        clusSC=CE.supercell_from_matrix(SC);
        #print(clusSC.supercell)
        #print(clusSC.bits)
        # Define cation/anion sublattices
        ions=[]
        for subLat in RO:
            for specie in sublat:
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

        # Integrate Wenxuan's solver here, and abandon MC annealing. (In the future.)#
        RO_int = [{specie:int(round(site[specie]*scs)) for specie in site} for site in RO] 
        #convert frac occupation back to integers.
        RO_string = json.dumps(RO_int)
        if RO_string not in mc_structs:
            mc_structs[RO_string]=[]
        
        if all_axis:
            axis_string = json.dumps(all_axis[sc_ro_pair_id])
            if RO_string not in ro_axis_strings:
                ro_axis_strings[RO_string] = axis_string

        # Add approximate ground state to set of MC structures
        # Format as (structure, temperature) - for ground state, temperature is "0"
        mc_structs[RO_string].append((clusSC.structure_from_occu(sa_occu),0))

        for T in TLst:
            print("Doing MC under T = {}K".format(T))
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
            mc_structs[RO_string].append((clusSC.structure_from_occu(min_occu),T))
            for rand, rand_e in rand_occu:
                mc_structs[RO_string].append((clusSC.structure_from_occu(rand),T))

        sc_ro_pair_id += 1

    # Deduplicate - first versus previously calculated structures, then versus structures within this run
    print('Deduplicating random structures.')

    unique_structs = {}
    unqCnt = 0
    sm = StructureMatcher(ltol=0.3, stol=0.3, angle_tol=5, comparator=ElementComparator())
    for RO_string,structs in mc_structs.items():
        if RO_string not in unique_structs:
            unique_structs[RO_string] = []
        for struct,T in structs:
            unique = True
            for ostruct in calculated_structures:
                if sm.fit(struct, ostruct):
                    unique = False
                    break
            for ostruct,T in unique_structs[RO_string]:
                if sm.fit(struct, ostruct):
                    unique = False
                    break
            if unique:
                unique_structs[RO_string].append((struct,T))
                unqCnt += 1
    print('Obtained %d unique occupied random structures.'%unqCnt)

    # Save structures
    print('#### MC Final Saving ####')
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
        print(outdir,' does not exist. Created.')

    RO_id = 0
    for RO_string,structs in unique_structs.items():
        RODir = 'Composition{}'.format(RO_id)
        compPathDir = os.path.join(outdir,RODir)
        if not os.path.isdir(compPathDir): os.mkdir(compPathDir)
        occu_file_path = os.path.join(compPathDir,'composition_by_site')
        if not os.path.isfile(occu_file_path):
            with open(occu_file_path,'w') as occufile:
                occufile.write(RO_string)
        if all_axis:
            axis_file_path = os.path.join(compPathDir,'axis')
            if not os.path.isfile(axis_file_path):
                with open(axis_file_path,'w') as axisfile:
                    axisfile.write(ro_axis_strings[RO_string])
        for i, (struct,T) in enumerate(structs):
            structDir = os.path.join(compPathDir,str(i))
            if not os.path.isdir(structDir): os.mkdir(structDir)
            Poscar(struct.get_sorted_structure()).write_file(os.path.join(structDir,'POSCAR'))
        RO_id += 1

    print('Saving of %s successful. Writing VASP input files later.'%outdir)

def _write_vasp_inputs(Str,VASPDir,functional='PBE',num_kpoints=25,additional_vasp_settings=None, strain=((1.01,0,0),(0,1.05,0),(0,0,1.03)) ):
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
            VASPSettings[key]=additional_vasp_settings
            print('Changed {} setting.'.format(key))

    if not os.path.isdir(VASPDir):os.mkdir(VASPDir);

    # Joggle the lattice to help symmetry broken relaxation. You may turn it off by setting strain=None
    if strain:
         deformation = Deformation(strain)
         Str = Deformation.apply_to_structure(Str)

    Str=Structure(StrainedLatt,Species,FracCoords,to_unit_cell=False,coords_are_cartesian=False);
    VIO=MITRelaxSet(Str,potcar_functional = functional); VIO.user_incar_settings=VASPSettings;
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

def _gen_vasp_inputs(SearchDir):
    """
    Search through directories, find POSCARs and generate FM VASP inputs.
    """
    POSDirs=[];
    for Root,Dirs,Files in os.walk(SearchDir):
        for File in Files:
            if File == 'POSCAR' and 'fm.0' not in Root and 'INCAR' not in Root: POSDirs.append([Root,File]);
    for [Root,File] in POSDirs:
        print("Writing VASP inputs for {}/{}".format(Root,File));
        Str=Poscar.from_file(os.path.join(Root,File)).structure
        VASPDir= os.path.join(Root,'fm.0'); _write_vasp_inputs(Str,VASPDir);

def _generate_axis_ref(compounds):
    """
    Here we do axis decomposition for a chemical formula. Not necessary for complexed systems.
    Inputs:
        formula: chemical formula, represented in dictionary form: {'Li+':8,'Co3+':8,'O2-':16}
        compounds: a list of compound axis in string form: ['LiCoO2','CoO2']
    Outputs:
        A dict that gives the molar ratio of each compound: {'LiCoO2':1.0, 'CoO2':0}
    """
    ###Preprocessing###
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
             print('Can not generate axis. Specified reference compound {} does not have a unique specie. Exiting!'\
                    .format(compound))
             sys.exit()
                         
    uniqSpecieComps = {val:key for key,val in compUniqSpecies.items()}
    #print('uniqSpecieComps',uniqSpecieComps)

    return compSpecieNums, compUniqSpecies, uniqSpecieComps

def _axis_decompose(comSpecieNums, compUniqSpecies,uniqSpecieComps, occu, sc_size):
    occuComposition=collections.Counter()
    #These two counters are used to prevent unreasonably generated occupations that cannot be matched with any compound.
    specieStat_from_Occu=collections.Counter()
    specieStat_from_Compounds=collections.Counter()

    for s,site in enumerate(occu):
        #print(occu)
        for specie in occu[s]:
            specieStat_from_Occu[specie]+=occu[s][specie]
            if specie in uniqSpecieComps:
                corrCompound = uniqSpecieComps[specie]
                occuComposition[corrCompound]+=occu[s][specie]/compSpecieNums[corrCompound][specie]
                #print(occuComposition)
    for compound in occuComposition:
        for specie in compSpecieNums[compound]:
            specieStat_from_Compounds[specie]+=compSpecieNums[compound][specie]*occuComposition[compound]
    specieNumMatch = True
    for specie in specieStat_from_Occu:
        if abs(specieStat_from_Occu[specie]-specieStat_from_Compounds[specie])>0.01:
            specieNumMatch = False
            break
    if not specieNumMatch:
        print('Axis decomposition failed due to mismatch of number of species. Please check your axis compound selection carefully.')
        return

    tot_mol = sum(occuComposition.keys())
    axis = {compound:occuComposition[compound]/tot_mol for compound in occuComposition}
    return axis

def _supercells_from_occus(maxSize,prim,enforceOccu=None,sampleStep=1,supercellnum=10,\
                               comp_axis=None,transmat=[[1,0,0],[0,1,0],[0,0,1]]):
    #Warning: Currently assumes a specie only occupies one site.
    '''
    Inputs:
        maxSize: determinant of supercell matrix that is to be enumerated
        prim: primitive cell data. Generated by pymatgen.io.cif.CifParser(). Initially a pure compound, for 
                  instance, LiNiO2 in /Examples
        compaxis: a list of compound names. If specified, the program will caculate the composition in compound ratio,
                  but ususally not used since we don't think its necessary nor applicable in complexed disordered 
                  rocksalt systems.
        enforceOccu: Sometimes you want a sublattice to be always fully occupied or at least half occupied. Use this 
                     parameter and input a dict, such as {'Li+':0.333,...}.
        sampleStep: Sometimes you don't want to change site occupation one by one. Like, if you have 24 A sites, you 
                   wanna sample every 6 Li+ occupation so you will have less possibility to handle with and less 
                   memory consumption. We recommend you use common factor(gcd) of 'nSites', or charge balance may
                   not be easily satisfied!!
        transmat: Transfromation matrix to the primitive cell, before you form supercells.
    Charge balance ensured by conditional checking.
    Outputs:
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
    
    print('#### Occupation enumeration ####')
    SCLst = []
    SC_sizes = [int(abs(np.linalg.det(SC))) for SC in SCenum] 

    # Enumeration by site, not by compounds' chemical formulas
    # We will need to provide a site-by-site replacement method and abandon pymatgen replace_species
    for sc_id,sc in enumerate(SCenum):
        poss_Occu_Sites = []
        for s,site in enumerate(prim):
            poss_Sp_Nums = {}
            for specie in occuDicts[s]:  
               poss_Sp_Nums[specie]=list(range(0,SC_sizes[sc_id]+1,sampleStep))
            keys,values = zip(*poss_Sp_Nums.items())
            allOccu_for_Site = [dict(zip(keys,v)) for v in itertools.product(*values) if (((enforceOccu and\
            sum(v)/SC_sizes[sc_id]+0.01>enforceOccu[site]) or (not enforceOccu)) and (sum(v)/SC_sizes[sc_id]\
            -0.01<1.0))]
            #print(allOccu_for_Site)
            poss_Occu_Sites.append(allOccu_for_Site)
        
        allOccu_for_Sites = [list(site_combo) for site_combo in itertools.product(*poss_Occu_Sites) \
                             if _Is_Neutral_Occu(site_combo,specieChgDict) ]

        #print(allOccu_for_Sites)
        #Calculate compositions and fractional occupation for each site.
        if comp_axis:
            print('Axis decomposition selected, reference compounds:',comp_axis)
            compSpecieNums, compUniqSpecies, uniqSpecieComps = _generate_axis_ref(comp_axis)
            all_axis = []

        allFracOccu_for_Sites = []
        occus_WorthToExpand = []
        for occu in allOccu_for_Sites:
            #Check whether the generated composition is a fully occupied and pure compound, and avoid to do CE on that.
            #Also generate replacement table
            sites_WorthToExpand = []
            fracOccu = []
            for site_occu in occu:
                fracOccu.append({specie:site_occu[specie]/SC_sizes[sc_id] for specie in site_occu})
            for site_fracoccu in fracOccu:
                site_WorthToExpand = True
                for specie in site_fracoccu:
                    if abs(site_fracoccu[specie]-1.00)<0.001:
                        site_WorthToExpand=False
                        break
                sites_WorthToExpand.append(site_WorthToExpand)
                
            worthToExpand = reduce(lambda x,y:x or y,sites_WorthToExpand)
                    
            #print('fracOccu',fracOccu,'Composition',occuComposition)
            if worthToExpand:
                allFracOccu_for_Sites.append(fracOccu)
                if comp_axis:
                    all_axis.append(_axis_decompose(comSpecieNums, compUniqSpecies,uniqSpecieComps, \
                                           occu, SC_sizes[sc_id]))
                occus_WorthToExpand.append(sites_WorthToExpand)
                #print(occuComposition,'expanded')

        SCLst.extend(zip([SC]*len(allCompositions),allFracOccu_for_Sites,occus_WorthToExpand))

        print('Generated %d compositions for supercell '%len(allCompositions),SC,'.')

    #print(SCLst)
    if comp_axis:
        return SCLst,all_axis
    else:
        return SCLst,None

class StructureGenerator(MSONable):
    def __init__(prim, outdir='vasp_run', enforced_occu = None, sample_step=1, max_sc_size = 64, sc_selec_num = 10, comp_axis=None, transmat=[[1,0,0],[0,1,0],[0,0,1]],ce_file = None):
        """
        prim: The structure to build a cluster expasion on. In our current version, prim must be a pyabinitio.core.Structure object, and each site in p
              rim is considered to be the origin of an independent sublattice. In MC enumeration and GS solver, composition conservation is done on 
              each sublattice, and no specie exchange is allowed between sublattices. For example, Li+ might occupy both O and T sites in spinel 
              structures, they can be swapped between O and T without breaking conservation of composition, but in our program this kind of flipping 
              will not be allowed. If you want to consider Li+ distribution over O and T, you should enumerate Li+ on O sublattice and T sublattice 
              independently.
        outdir: output directory. Recommended == vasp_dir
        enforced_occu: This specifies the lowest total occupation ratio of a sublattice. In the form of: [0.0,1.0,1.0,1.0], which means site #2,3,4 
              must be fully occupied while site 1 has no special constraint.
        sample_step: Enumeration step of species on sublattices. sample_step = 2 means that 2 atoms will be changed at each step during occupation 
              enumeration.
        max_sc_size: Maximum supercell size to enumerate, in determinant of supercell. 
        sc_selec_enum: Number of skewed and unskewed supercells to randomly select from enumeration.
        comp_axis: We provide a fucntion here to transform an occupation representation into composition in compound ratio, for your convedience to 
              drawing phase diagrams. By default the composition is not decomposed since your systems can oftenly get too complexed. The occupation
              will be stored in CEfile and also under all subdirectories of calculations.
              Form of an occupation to store: [{'Li+':2,'Co3+':2,'Co4+':6,'Vac':6},{'O2-':16}]
        transmat: Sometimes your primitive cell is not symmetric enough, just like the case in the 2-site rhombohedral primitive cell of rock salt.
              You can apply a transformation before shape enumeration.
        ce_file: the file that contains a full record of the current cluster expansion work, including ce.as_dict, structures, ecis, etc.
        """

        self.prim = prim
        if self.enforced_occu:
            print("Occupation on each site at least:",enforced_occu)
        self.enforced_occu = enforced_occu
        self.sample_step=sample_step
        self.max_sc_size=max_sc_size
        self.sc_selec_num=sc_selec_num
        self.comp_axis = comp_axis
        self.transmat = transmat
        print("Using transformation matrix {}".format(transmat))
        self.ce_file = ce_file
        self.outdir =  outdir
        if not os.path.isfile('prim'):
            print("Primitive POSCAR file 'prim' doesn't exist. Creating one.")
            Poscar(prim,comment='PRIMITIVE CELL FILE').write_file('prim')
        
    def generate_structures(self):
        sc_ro,all_axis =  _supercells_from_occus(self.max_sc_size, self.prim.get_sorted_structure(), self.enforced_occu,\
                                        self.sample_step, self.sc_selec_num, self.comp_axis, self.transmat)
        _get_mc_structs(self.ce_file,self.outdir,sc_ro,Prim=self.prim,all_axis=all_axis,TLst=[500, 1500, 10000])
        
    def write_structures(self):
        _gen_vasp_inputs(self.outdir)

####
# I/O interface for saving only, from_dict method not required
####
    def as_dict(self):
        return {'prim':self.prim.as_dict(),\
                'enforced_occu':self.enforced_occu,\
                'comp_axis':self.comp_axis,\
                'sample_step':self.sample_step,\
                'max_sc_size':self.max_sc_size,\
                'sc_selec_num':self.sc_selec_num,\
                'transmat':self.transmat,\
                'ce_file':self.ce_file,
                'outdir':self.outdir
                '@module':self.__class__.__module__,\
                '@class':self.__class__.__name__\
               }

