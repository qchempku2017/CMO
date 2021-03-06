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
from cluster_expansion.ce import ClusterExpansion
from pymatgen.io.vasp.sets import MITRelaxSet
from pymatgen.io.vasp.inputs import *
from pymatgen.io.vasp.outputs import *
from pymatgen.io.cif import *
from pymatgen import Structure
from pymatgen.core.periodic_table import Specie
from pymatgen.core.composition import Composition
from pymatgen.core.sites import PeriodicSite
from pymatgen.analysis.elasticity.strain import Deformation
from pymatgen.core.lattice import Lattice
from pymatgen.transformations.standard_transformations import OrderDisorderedStructureTransformation

from itertools import permutations,product
from operator import mul
from functools import partial,reduce
import multiprocessing
import collections

from cluster_expansion.mc import *
from cluster_expansion.utils import *
from selector_tools import *

"""
Update Notes:
2-3-20 Fengyu Xie: From now on, we will store the supercell matrix from enumeration step, to prevent
inaccuracy and inefficiency of matrix generation by pymatgen.StructureMatcher.
"""
##################################
## Less general tools that are not cross refered by other modules
##################################
def _is_proper_sc(sc,prim):
    newmat = np.array(prim.lattice.matrix)@np.array(sc)
    latt = Lattice(newmat)
    angles = sorted([latt.alpha,latt.beta,latt.gamma])
    if np.linalg.cond(newmat)<=10 and not(angles[-1]<20 and angles[-2]<10):
        return True
    else:
        return False
    #Mat can't be too skewed or too prolonged!

def _Enumerate_SC(maxDet,prim,n_select=1,transmat=None):
    '''
    Enumerate all possible orthogonal supercell matrices. Abandon all skewed ones
    to avoid numerical instability in structure matcher.
    '''
    print('#### Supercell Enumeration ####')
    scs=[]
    trans_size = int(round(abs(np.linalg.det(transmat)))) if transmat else 1

    for det in range(int(maxDet/4),maxDet+1,int(maxDet/4)):
        scs.extend(Get_Diag_Matrices(int(det/trans_size)))
    print('Generated %d supercell matrices with max determinant %d'%(len(scs),maxDet))
    scs = [sc for sc in scs if _is_proper_sc(sc,prim)]
    print('Picking %d random rectangular supercells.'%(n_select))

    ns = min(n_select,len(scs))

    selected_scs = random.sample(scs,ns)
    #print("scs before trans:",selected_scs)
    if transmat:
        selected_scs=[mat_mul(sc,transmat) for sc in selected_scs]
    return selected_scs

def _was_generated(x):
    return 'POSCAR' in x and not 'KPOINTS' in x and not 'INCAR' in x and not 'POTCAR' in x

def _get_mc_structs(SCLst,CE,ecis,merge_sublats=None,TLst=[500, 1500, 10000],compaxis=None,outdir='vasp_run'):
    '''This function checks the previous calculation directories when called. If no previous calculations, it
       generates the intial pool. If there are previous calculations, add new sampled structures based on 
       structure selection rule.
       For CE sampling using MC, use three set of temperature, merge this with LocalOrdering code.
       ce_file: directory of CE Mson data file
       outdir: directory to write outputs
       SCLst: a list contaning enumerated SC's and RO pairs.
       TLst: temprature list to do MC enumeration on
       useX: a list of compounds of which we may want to calculate component.
       compaxis: a list of compound names. If specified, the program will caculate the composition in compound ratio,
                but ususally not used since we don't think its necessary nor applicable in complexed disordered 
                rocksalt systems.
    '''
    print('#### MC Initialization ####')
    Prim = CE.structure
    print('SM type:',CE.sm_type)
    calculated_structures = {}

    if os.path.isdir(outdir):
        print("Checking previously enumerated structures.")
        for root,dirs,files in os.walk(outdir):
            if _was_generated(files):
                parentdir = os.path.join(*root.split(os.sep)[0:-1])
                with open(os.path.join(parentdir,'composition_by_site')) as RO_file:
                    RO_old = json.load(RO_file)
                    RO_old_string = json.dumps(RO_old)
                if RO_old_string not in calculated_structures:
                    calculated_structures[RO_old_string]=[]
                calculated_structures[RO_old_string].append(Poscar.from_file(os.path.join(root,'POSCAR')).structure)
                # struct_id = int(root.split(os.sep)[-1])
    else: 
        print("No previous calculations, generating the initial pool.")
 
    mc_structs={};
    if compaxis:
        ro_axis_strings = {}

    sc_ro_pair_id = 0

    if merge_sublats is None:
        sublat_list = [[i] for i in range(len(Prim))]
    else:
        sublat_list = merge_sublats

    for SC,RO,sublats_WorthToExpand in SCLst:
        #len(RO)==len(sublats_WorthToExpand)==len(merge_sublats) if merge_sublats else ==len(SC)

        print("Processing composition:\n",RO,'\nSupercell:\n',SC,'\nsize:\n',int(round(np.abs(np.linalg.det(SC)))))
        clusSC=CE.supercell_from_matrix(SC);

        Bits=clusSC.bits;
        scs = int(round(np.abs(np.linalg.det(SC))))
        # generate a list of groups of sites to swap between! We have known which sites are partially occupied,
        # so we only need to figure out how pymatgen make a group of supercell sites from a primitive cell site.
        # Looks like it simply just replicate sites one by one! 
        # indGrps=[list(range(i*scs,(i+1)*scs)) for i in range(len(RO)) if sites_WorthToExpand[i]];

        indGrps = [list(range(i*scs*len(sublat),(i+1)*scs*len(sublat))) for i,sublat in enumerate(sublat_list)\
                  if sublats_WorthToExpand[i]]

        RO_int = [{specie:int(round(RO[i][specie]*scs*len(sublat))) for specie in RO[i]} for i,sublat in enumerate(sublat_list)]
        #species will now be swapped within a 'sublattice', instead of the replicates of a site.

        #Note: indGrps should be generated from a clusSC supercell!!!!!
        #print('indGrps',indGrps,'RO',RO)         

        # Replace species according to RO
        randSites = []
        for i,sublat in enumerate(sublat_list):
            for s_id in sublat:
                site = Prim[s_id] 
                randSite = PeriodicSite(RO[i],site.frac_coords,Prim.lattice,properties=site.properties)
                randSites.append(randSite)
        randStr = Structure.from_sites(randSites)
            
        # Get electrostatics enumeration guess
        order=OrderDisorderedStructureTransformation(algo=2);

        randStr.make_supercell(SC)

        randStr =order.apply_transformation(randStr)

        #print('randStr:\n',randStr,'\nce prim:\n',CE.structure)
        # Simulated annealing for better guess at ground state
        # You may want to change the number of MC flips for each temperature

        init_occu = clusSC.occu_from_structure(randStr)
        #print("Starting occupation:", init_occu)
        sa_occu = simulated_anneal(ecis=ecis, cluster_supercell=clusSC, occu=init_occu, ind_groups=indGrps,
                                   n_loops=20000, init_T=5100, final_T=100, n_steps=20)
        print("MC ground state acquired, analyzing composition.")
        
        # Axis decomposition
        if compaxis:
            axis = _axis_decompose(compaxis,RO_int)
        #convert frac occupation back to integers.
        sp_list = []
        for sublat_occu in RO_int:
            sp_list.extend(sublat_occu.values())
        _gcd = GCD_List(sp_list)
        RO_reduced_int=[{sp:sublat_occu[sp]//_gcd for sp in sublat_occu} for sublat_occu in RO_int]

        #Reduce occupation numbers by GCD.
        RO_string = json.dumps(RO_reduced_int)
        print("Reduced occupation:", RO_string)
        if RO_string not in mc_structs:
            mc_structs[RO_string]=[]
        
        #REFERED AXIS DECOMPOSITION IS WRONG. FIX THIS!

        if compaxis:
            axis_string = json.dumps(axis)
            if RO_string not in ro_axis_strings:
                ro_axis_strings[RO_string] = axis_string
            print("Axis composition: ",axis_string)

        # Add approximate ground state to set of MC structures
        # Format as (structure, temperature) - for ground state, temperature is "0"
        # print("GS structure:",clusSC.structure_from_occu(sa_occu))
        mc_structs[RO_string].append((clusSC.structure_from_occu(sa_occu),0))
        print("MC GS added to the preset.")

        for T in TLst:
            print("Doing MC sampling under T = {}K".format(T))
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
                mc_structs[RO_string].append((clusSC.structure_from_occu(rand),SC))

        sc_ro_pair_id += 1

    # Deduplicate - first compared to previously calculated structures, then compared to structures within this run
    print('Deduplicating random structures.')

    unique_structs = {}
    unqCnt = 0
    sm = StructureMatcher(ltol=0.3, stol=0.3, angle_tol=5, comparator=ElementComparator())
    for RO_string,structs in mc_structs.items():
        if RO_string not in unique_structs:
            unique_structs[RO_string] = []

        for struct,matrix in structs:
            unique = True
            if RO_string in calculated_structures:
                for ostruct in calculated_structures[RO_string]:
                    if sm.fit(struct, ostruct):
                        unique = False
                        break
            if unique:
                for ostruct,matrix in unique_structs[RO_string]:
                    if sm.fit(struct, ostruct):
                        unique = False
                        break
            if unique:
                try:
                    #Check if structure matcher works for this structure. If not, abandon.
                    cs=CE.supercell_from_matrix(matrix)
                    corr=cs.corr_from_structure(struct)
                    unique_structs[RO_string].append((struct,matrix))
                    unqCnt += 1
                except:
                    continue

    print('Obtained %d unique occupied random structures.'%unqCnt)

    if compaxis:
        return unique_structs,ro_axis_strings
    else:
        return unique_structs,None

def _write_mc_structs(unique_structs,ro_axis_strings,outdir='vasp_run'):
    # Save structures
    print('#### MC Final Saving ####')
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
        print(outdir,' does not exist. Created.')
        load_old = False
    else:
        print('Old calculations found. Appending to old calculations.')
        load_old = True

    old_str_max_ids = {}
    old_ro_ids = {}
    if load_old:
        for root,dirs,files in os.walk(outdir):
            if _was_generated(files):
                parentdir = os.path.join(*root.split(os.sep)[0:-1])
                with open(os.path.join(parentdir,'composition_by_site')) as RO_file:
                    RO_old = json.load(RO_file)
                    RO_old_string = json.dumps(RO_old)
                if RO_old_string not in old_str_max_ids:
                    old_str_max_ids[RO_old_string]=max([int(idx) for idx in os.listdir(parentdir) if RepresentsInt(idx)])

        import re

        for dirname in os.listdir(outdir):
            with open(os.path.join(outdir,dirname,'composition_by_site')) as RO_file:
                RO_old = json.load(RO_file)
                RO_old_string = json.dumps(RO_old)
            ro_id = int(re.search(r'\d+', dirname).group())
            if RO_old_string not in old_ro_ids:
                old_ro_ids[RO_old_string]=ro_id

        old_max_RO_id = max(list(old_ro_ids.values()))    

    else:
        old_max_RO_id = 0
    
    new_RO_id = 0
    for RO_string,structs in unique_structs.items():
        if RO_string in old_ro_ids:
            RODir = 'Composition{}'.format(old_ro_ids[RO_string])
        else:
            RODir = 'Composition{}'.format(new_RO_id+old_max_RO_id+1)
            old_ro_ids[RO_string]=new_RO_id+old_max_RO_id+1
            new_RO_id +=1

        compPathDir = os.path.join(outdir,RODir)
        if not os.path.isdir(compPathDir): os.mkdir(compPathDir)

        occu_file_path = os.path.join(compPathDir,'composition_by_site')
        if not os.path.isfile(occu_file_path):
            with open(occu_file_path,'w') as occufile:
                occufile.write(RO_string)
        if ro_axis_strings:
            axis_file_path = os.path.join(compPathDir,'axis')
            if not os.path.isfile(axis_file_path):
                with open(axis_file_path,'w') as axisfile:
                    axisfile.write(ro_axis_strings[RO_string])
        for i, (struct,matrix) in enumerate(structs):
            if RO_string in old_str_max_ids:
                structDir = os.path.join(compPathDir,str(i+old_str_max_ids[RO_string]+1))
            else:
                structDir = os.path.join(compPathDir,str(i))

            if not os.path.isdir(structDir): os.mkdir(structDir)
            Poscar(struct.get_sorted_structure()).write_file(os.path.join(structDir,'POSCAR'))
            with open(os.path.join(structDir,'matrix'),'w') as mat_file:
                json.dump(list(matrix),mat_file)

    print('Saving of %s successful.'%outdir)

def _gen_vasp_inputs(SearchDir='vasp_run',functional='PBE', num_kpoints=25,add_vasp_settings=None, strain=((1.01,0,0),(0,1.05,0),(0,0,1.03)) ):
    """
    Search through directories, find POSCARs and generate FM VASP inputs.
    """
    POSDirs=[];
    for Root,Dirs,Files in os.walk(SearchDir):
        if 'POSCAR' in Files and 'fm.0' not in Dirs and 'INCAR' not in Files: POSDirs.append(Root);
    for Root in POSDirs:
        print("Writing VASP inputs for {}".format(Root));
        Str=Poscar.from_file(os.path.join(Root,'POSCAR')).structure
        VASPDir= os.path.join(Root,'fm.0'); 
        write_vasp_inputs(Str,VASPDir,functional,num_kpoints,add_vasp_settings,strain);

def _generate_axis_ref(compounds):
    """
    Here we do axis decomposition for a chemical formula. Not necessary for complexed systems.
    Inputs:
        compounds: a list of compound axis in string form: ['LiCoO2','CoO2']
    Outputs:
        compSpecieNums: a dict recording how many species a compound has in its formula: {'CoO2':['Co4+':1,'O2-':2]} 
        compUniqSpecies: a dict recording the 'marker specie' to a compound:{'LiCoO2':'Co3+','CoO2':'Co4+'}
        uniqSpecieComps: reversed dict of compUniqSpecies.
    """
    ###Preprocessing###
    compSpecieNums = {}
    # Get representatEADME.mdie in a compound to make a calculation of composition from site enumeration easier.
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

def _axis_decompose(compounds, occu):
    #print("Axis decompose")
    compSpecieNums, compUniqSpecies,uniqSpecieComps = _generate_axis_ref(compounds)
    occuComposition=collections.Counter()
    #These two counters are used to prevent unreasonably generated occupations that cannot be matched with any compound.
    specieStat_from_Occu=collections.Counter()
    specieStat_from_Compounds=collections.Counter()
    
    for s,sublat in enumerate(occu):
        #print(occu)
        for specie in occu[s]:
            if specie in uniqSpecieComps:
                corrCompound = uniqSpecieComps[specie]
                occuComposition[corrCompound]+=occu[s][specie]/compSpecieNums[corrCompound][specie]
                #print(occuComposition)
    for compound in occuComposition:
        for specie in compSpecieNums[compound]:
            specieStat_from_Compounds[specie]+=compSpecieNums[compound][specie]*occuComposition[compound]
    for sublat in occu:
        for specie in sublat:
            specieStat_from_Occu[specie]+=sublat[specie]
    
    #print(specieStat_from_Occu,specieStat_from_Compounds)
    specieNumMatch = True
    for specie in specieStat_from_Occu:
        if abs(specieStat_from_Occu[specie]-specieStat_from_Compounds[specie])>0.01:
            specieNumMatch = False
            break
    #print(specieNumMatch)
    if not specieNumMatch:
        print('Axis decomposition failed due to mismatch of number of species. Please check your axis compound selection carefully.')
        return

    tot_mol = sum(occuComposition.values())

    axis = {compound:occuComposition[compound]/tot_mol for compound in occuComposition}
    return axis

def _supercells_from_occus(maxSize,prim,maxVacs=None,sampleStep=1,supercellnum=10,\
                           transmat=[[1,0,0],[0,1,0],[0,0,1]],merge_sublats=None):
    #Warning: Currently assumes a specie only occupies one site.
    '''
    Inputs:
        maxSize: determinant of supercell matrix that is to be enumerated
        prim: primitive cell data. Generated by pymatgen.io.cif.CifParser(). Initially a pure compound, for 
                  instance, LiNiO2 in /Examples
        maxVacs: maximum vacancy percentage on a site. This is a list, with len(maxVacs)==len(sublat_list)
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
    SCenum = _Enumerate_SC(maxSize,prim,supercellnum,transmat)

    specieChgDict={}
    
    occuDicts = []
    if merge_sublats is None:
        sublat_list = [[i] for i in range(len(prim))]
    else:
        sublat_list = merge_sublats

    for sublat in sublat_list:
        sublat_sites_occu = []
        for s_id in sublat:
            siteSpecies = prim[s_id].species_string.split(',')
            #print('siteSpecies',siteSpecies)
            if len(siteSpecies)>1 or (':' in siteSpecies[0]):
                siteOccu = dict([[s.strip() for s in specieoccu.split(':')] for specieoccu in siteSpecies])
            else:
                siteOccu = {siteSpecies[0].strip():1.00}
            #print(siteOccu)
            siteOccuMod = {}
            for specie in siteOccu:
                specieMod = Modify_Specie(specie)
                if specieMod not in specieChgDict: specieChgDict[specieMod]=GetIonChg(specieMod)
                siteOccuMod[specieMod]=float(siteOccu[specie])
            sublat_sites_occu.append(siteOccuMod)

        sublatOccu = {}
        for siteOccu in sublat_sites_occu:
            for specie in siteOccu:
                if specie not in sublatOccu:
                    sublatOccu[specie]=0.0
                sublatOccu[specie]+=siteOccu[specie]/len(sublat_sites_occu)
        occuDicts.append(sublatOccu)
    #Now the occupation in written in sublattices form.
    #print("Occupation Enum:",occuDicts)
    
    print('#### Occupation enumeration ####')
    SCLst = []
    SC_sizes = [int(round(abs(np.linalg.det(SC)))) for SC in SCenum] 

    # Enumeration now by sublattices, not by compounds' chemical formulas
    for sc_id,sc in enumerate(SCenum):
        #print(sc,SC_sizes[sc_id])
        poss_Occu_Sublats = []
        for sl_id,sublat in enumerate(sublat_list):
            poss_Sp_Nums = {}
            for specie in occuDicts[sl_id]:  
               poss_Sp_Nums[specie]=list(range(0,SC_sizes[sc_id]*len(sublat)+1,sampleStep))
            keys,values = zip(*poss_Sp_Nums.items())
            allOccu_for_Sublat = [dict(zip(keys,v)) for v in itertools.product(*values) if ((maxVacs is not None and\
            float(sum(v))/(SC_sizes[sc_id]*len(sublat))>=1.0-maxVacs[sl_id]) or maxVacs is None) and \
            float(sum(v))/(SC_sizes[sc_id]*len(sublat))<=1.0]
            #print(allOccu_for_Site)
            poss_Occu_Sublats.append(allOccu_for_Sublat)

        #print(poss_Occu_Sites)
        allOccu_for_Sublats = [list(sublat_combo) for sublat_combo in itertools.product(*poss_Occu_Sublats) \
                             if Is_Neutral_Occu(sublat_combo,specieChgDict) ]

        allFracOccu_for_Sublats = []
        occus_WorthToExpand = []

        for occu in allOccu_for_Sublats:
            #Check whether the generated composition is a fully occupied and pure compound, and avoid to do CE on that.
            #Also generate replacement table
            sublats_WorthToExpand = []
            fracOccu = []
            for sl_id,sl_occu in enumerate(occu):
                fracOccu.append({specie:float(sl_occu[specie])/(SC_sizes[sc_id]*len(sublat_list[sl_id])) \
                                 for specie in sl_occu})
            #print('fracOccu',fracOccu)

            for sublat_fracoccu in fracOccu:
                sublat_WorthToExpand = True
                for specie in sublat_fracoccu:
                    if abs(sublat_fracoccu[specie]-1.00)<0.001:
                        sublat_WorthToExpand=False
                        break
                if sublat_WorthToExpand:
                    sublat_WorthToExpand = (sum(list(sublat_fracoccu.values())) > 0.0)

                sublats_WorthToExpand.append(sublat_WorthToExpand)
            #print(sites_WorthToExpand)
    
            worthToExpand = reduce(lambda x,y:x or y,sublats_WorthToExpand)
                    
            #print('worthToExpand',worthToExpand)
            if worthToExpand:
                allFracOccu_for_Sublats.append(fracOccu)
                #print("allFracOccu_for_sites",allFracOccu_for_Sites)
                occus_WorthToExpand.append(sublats_WorthToExpand)
                #We will no longer do axis decomposition here
                                                 
        SCLst.extend(zip([sc]*len(allFracOccu_for_Sublats),allFracOccu_for_Sublats,occus_WorthToExpand))

        print('Generated %d compositions for supercell '%len(allFracOccu_for_Sublats),sc,'.')

    #print('SCLst',SCLst)
    return SCLst

class StructureGenerator(MSONable):
    def __init__(self,prim_file='prim.cif', merge_sublats = None, \
                 outdir='vasp_run', max_vacs = None, sample_step=1, max_sc_size = 64,\
                 sc_selec_enum = 10, comp_axis=None, transmat=None,ce_file = 'ce.mson',vasp_settings='vasp_settings.mson'):
        """
        prim: The structure to build a cluster expasion on. In our current version, prim must be a pyabinitio.core.Structure object, and each site in p
              rim is considered to be the origin of an independent sublattice. In MC enumeration and GS solver, composition conservation is done on 
              each sublattice, and no specie exchange is allowed between sublattices. For example, Li+ might occupy both O and T sites in spinel 
              structures, they can be swapped between O and T without breaking conservation of composition, but in our program this kind of flipping 
              will not be allowed. If you want to consider Li+ distribution over O and T, you should enumerate Li+ on O sublattice and T sublattice 
              independently.

        merge_sublats: if not none, should be a list of lists, in which each list stands for the site indices in prim that are going to be considered
              a same sublattice. The indices should be 0-based. If none, each site in prim will be considered an independent sublattice.
              If sites are considered a same sublattice, in the MC enumeration, they will be of the same index group to swap atoms in between.
              Atoms will not be swapped to outside of the ind group

        outdir: output directory. Recommended == vasp_dir


        sample_step: Enumeration step of species on sublattices. sample_step = 2 means that 2 atoms will be changed at each step during occupation 
              enumeration.

        max_sc_size: Maximum supercell size to enumerate,(volume of the supercell). 

        sc_selec_enum: Number of skewed and unskewed supercells to randomly select from enumeration.

        comp_axis: We provide a fucntion here to transform an occupation representation into composition in compound ratio, for your convedience to 
              drawing phase diagrams. By default the composition will not be decomposed to avoid over-complication. The occupation will be stored 
              in Calcdata file and also under all the subdirectories of calculations.
              Example of a stored occupation dict to store: [{'Li+':2,'Co3+':2,'Co4+':6,'Vac':6},{'O2-':16}] each dict term is the occupation
              of a subdirectory.
              if decomposed, generates a new axis dict, for example: {'CaO':0.5,'MgO':0.5}

        transmat: Sometimes your primitive cell is not symmetric enough, just like the case in the 2-site rhombohedral primitive cell of rock salt.
              You can apply a transformation before doing enumeration over supercells.

        ce_file: the file that contains a full record of the current cluster expansion work, is the output of ECIGenerator.as_dict()

        vasp_settings: setting parameters for vasp calcs. Is in dictionary form. Keys are 'functional','num_kpoints','additional_vasp_settings'
              (in dictionary form), 'strain'(in matrix or list form). If not given, will use default settings in MITRelaxset.
        """

        if os.path.isfile(prim_file):
            self.prim = CifParser(prim_file).get_structures()[0]
        else:
            raise ValueError("Primitive cell file can not be found, stopping!")

        if merge_sublats is not None:
            print("Merging the following sites into sublattices: {}".format(merge_sublats))
        self.merge_sublats = merge_sublats

#        if an_sublats is not None:
#            print("Selecting the following sublattices as the anion framework: {}".format(an_sublats))
#        self.an_sublats = an_sublats

        if max_vacs is not None:
            print("Vacancies on each site are at most:",max_vacs)
        self.max_vacs = max_vacs

        self.sample_step=sample_step
        self.max_sc_size=max_sc_size
        self.sc_selec_enum=sc_selec_enum
        self.comp_axis = comp_axis
        self.transmat = transmat
        #print("Using transformation matrix {}".format(transmat))
        self.ce_file = ce_file

        if ce_file and os.path.isfile(ce_file):
            # Load cluster expansion
            with open(ce_file,'r') as Fid: cedata = json.load(Fid);
            self.ce=ClusterExpansion.from_dict(cedata['cluster_expansion']); 
            self.ecis=cedata['ecis']; 
            print('Previous CE information:'); print(self.ce.structure);

        else:
            # No existing cluster expansion, we are building form start - use electrostatics only
            # This CE object is a bogus one, containing only electrostatics.
            print("Not checking previous cluster expansion, using ewald as sampling criteria.")
            self.ce=ClusterExpansion.from_radii(self.prim,{2: 1}, ltol=0.2,stol=0.15,angle_tol=5,\
                                       supercell_size='num_sites',use_ewald=True,use_inv_r=False,\
                                       eta=None)
            #Here we use pmg_sm as structure matcher.
            self.ecis=np.zeros(self.ce.n_bit_orderings+1); self.ecis[-1]=1;

        self.outdir =  outdir
        self._pool = []
        self._mat_pool = []
      
        self.vasp_file = vasp_settings
        if os.path.isfile(vasp_settings):
            print("Applying VASP settings in {}.".format(vasp_settings))
            with open(vasp_settings) as vs_in:
                self.vasp_settings=json.load(vs_in)
        else:
            print("Applying CEAuto default VASP settings.")
            self.vasp_settings = None
        self._sc_ro = None

    @property
    def sc_ro(self):
        # Enumerated supercells and compositions.
        if not self._sc_ro:
            self._sc_ro =  _supercells_from_occus(self.max_sc_size, self.prim, maxVacs=self.max_vacs,\
                                        sampleStep=self.sample_step, supercellnum=self.sc_selec_enum, \
                                        transmat=self.transmat,merge_sublats=self.merge_sublats)
        return self._sc_ro
    #Share the same set of sc_ro across project.

    def generate_structures(self):
        """
            Does not check if ce.mson is generated! So please make sure that, before two StructureGenerator calls,
            make an analyzer call!
        """
        #This part is for debug only
        if os.path.isfile('pool.temp') and os.path.isfile('ro_axis.temp'):
            with open('pool.temp') as temp_file:
                _unique_structs_d = json.load(temp_file)
            _unique_structs = {}
            _unique_structs = {comp:[(Structure.from_dict(struct),matrix) for struct,matrix in _unique_structs_d[comp]]\
                               for comp in _unique_structs_d}
            with open('ro_axis.temp') as temp_file:
                _ro_axis_strings = json.load(temp_file)
        #
        else:
            _unique_structs,_ro_axis_strings = _get_mc_structs(self.sc_ro,self.ce,self.ecis,merge_sublats=self.merge_sublats,\
                                                               TLst=[500, 1500, 10000],\
                                                               compaxis= self.comp_axis,outdir=self.outdir)
            with open('pool.temp','w') as temp_file:
                _unique_structs_d = {comp:[(struct.as_dict(),matrix) for (struct,matrix) in _unique_structs[comp]] \
                                     for comp in _unique_structs}
                json.dump(_unique_structs_d,temp_file)
            with open('ro_axis.temp','w') as temp_file:
                json.dump(_ro_axis_strings,temp_file)

        ss = StructureSelector(self.ce) #Using Nystrom selection by default

        _unique_structs_buff = []
        for comp in _unique_structs:
            _unique_structs_buff.extend([(comp,struct,matrix) for struct,matrix in _unique_structs[comp]])
        _pool = [val for key,val,mat in _unique_structs_buff]
        _mat_pool = [mat for key,val,mat in _unique_structs_buff]

        if len(self._pool)==0:
            n_init = min(3*len(_unique_structs),len(_pool))
            print("Initializing CE with {} chosen structures.".format(n_init))
            selected_inds = ss.initialization(_pool,mat_pool=_mat_pool,n_init=n_init)
        else:
            n_add = len(_unique_structs)
            print("Updating CE with {} chosen structures.".format(n_add))
            selected_inds = ss.select_new(self._pool,_pool,\
                                          old_mat_pool=self._mat_pool,\
                                          new_mat_pool=_mat_pool,\
                                          n_probe=n_add)

        self._pool.extend([_pool[idx] for idx in selected_inds])
        self._mat_pool.extend([_mat_pool[idx] for idx in selected_inds])

        #print('self._pool',self._pool)
        _unique_structs_selected = {}
        for idx in selected_inds:
            comp = _unique_structs_buff[idx][0]
            struct = _unique_structs_buff[idx][1]
            mat = _unique_structs_buff[idx][2]
            if comp not in _unique_structs_selected:
                _unique_structs_selected[comp]=[]
            _unique_structs_selected[comp].append((struct,mat))

        _write_mc_structs(_unique_structs_selected,_ro_axis_strings,outdir=self.outdir)
        self._write_vasp_inputs()
        
        self.as_file()
        os.remove('pool.temp')
        os.remove('ro_axis.temp')

    def _write_vasp_inputs(self):
        if self.vasp_settings:
            if 'functional' in self.vasp_settings:
                functional = self.vasp_settings['functional']
            else:
                funtional = 'PBE'
            if 'num_kpoints' in self.vasp_settings:
                num_kpoints = self.vasp_settings['num_kpoints']
            else:
                num_kpoints = 25
            if 'additional_vasp_settings' in self.vasp_settings:
                additional = self.vasp_settings['additional_vasp_settings']
            else:
                additional = None
            if 'strain' in self.vasp_settings:
                strain = self.vasp_settings['strain']
            else:
                strain = ((1.01,0,0),(0,1.05,0),(0,0,1.03))
            _gen_vasp_inputs(self.outdir,functional,num_kpoints,additional,strain)
        else:
            _gen_vasp_inputs(self.outdir)

####
# I/O interface for saving only, from_dict method not required
####
    @classmethod
    def from_file(cls,setting_file='generator.mson'):
        if os.path.isfile(setting_file):
            with open(setting_file,'r') as fs:
                settings = json.load(fs)
        else:
            settings = {}
        return cls.from_dict(settings)

    @classmethod
    def from_dict(cls,d):
        if 'prim_file' in d: prim_file = d['prim_file'];
        else: prim_file = 'prim.cif'; 

        if 'merge_sublats' in d: merge_sublats = d['merge_sublats']
        else: merge_sublats = None
 
#        if 'an_sublats' in d: an_sublat = d['an_sublats']
#        else: an_sublats = None

        if 'max_vacs' in d: max_vacs = d['max_vacs'];
        else: max_vacs = None;

        if 'comp_axis' in d: comp_axis = d['comp_axis'];
        else: comp_axis = None;

        if 'sample_step' in d: sample_step = d['sample_step'];
        else: sample_step = 1;

        if 'max_sc_size' in d: max_sc_size = d['max_sc_size'];
        else: max_sc_size = 64;

        if 'sc_selec_enum' in d: sc_selec_enum = d['sc_selec_enum'];
        else: sc_selec_enum = 10;

        if 'transmat' in d: transmat = d['transmat'];
        else: transmat = None;

        if 'ce_file' in d: ce_file = d['ce_file'];
        else: ce_file = 'ce.mson';

        if 'outdir' in d: outdir = d['outdir'];
        else: outdir = 'vasp_run';

        if 'vasp_settings' in d: vasp_settings = d['vasp_settings'];
        else: vasp_settings = 'vasp_settings.mson';

        #if 'n_select' in d: n_select = d['n_select'];
        #else: n_select = 1;
        
        generator = cls(prim_file=prim_file, merge_sublats = merge_sublats,\
                        #an_sublats = an_sublats,\
                        max_vacs=max_vacs, comp_axis=comp_axis, sample_step=sample_step,\
                        max_sc_size=max_sc_size, sc_selec_enum=sc_selec_enum, transmat=transmat, ce_file=ce_file,\
                        outdir=outdir, vasp_settings=vasp_settings)

        if 'sc_ro' in d:
            generator._sc_ro = d['sc_ro']
        if 'pool' in d:
            generator._pool = [Structure.from_dict(s_d) for s_d in d['pool']]
        if 'mat_pool' in d:
            generator._mat_pool = d['mat_pool']

        return generator

    def as_dict(self):
        return {'prim':self.prim.as_dict(),\
                'merge_sublats':self.merge_sublats,\
                 #'an_sublats':self.an_sublats,\
                'max_vacs':self.max_vacs,\
                'comp_axis':self.comp_axis,\
                'sample_step':self.sample_step,\
                'max_sc_size':self.max_sc_size,\
                'sc_selec_enum':self.sc_selec_enum,\
                'transmat':self.transmat,\
                'ce_file':self.ce_file,\
                'outdir':self.outdir,\
                'vasp_settings':self.vasp_file,\
                #'n_select':self.n_select,\
                'sc_ro':self.sc_ro,\
                'pool':[s.as_dict() for s in self._pool],\
                'mat_pool':[list(m) for m in self._mat_pool],\
                '@module':self.__class__.__module__,\
                '@class':self.__class__.__name__\
               }
    
    def as_file(self,settings_file='generator.mson'):
        print("Writing generator settings to {}".format(settings_file))
        with open(settings_file,'w') as fout:
            json.dump(self.as_dict(),fout)
