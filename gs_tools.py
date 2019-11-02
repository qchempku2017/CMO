from __future__ import division

from pyabinitio.cluster_expansion.ce import ClusterExpansion,Cluster,SymmetrizedCluster

from monty.json import MSONable
import json
import numpy as np
import math
import random
import os
from copy import deepcopy
from operator import mul
from functools import reduce
import random

from itertools import combinations,permutations,chain
from pymatgen.analysis.ewald import EwaldSummation
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer, SymmOp
from pymatgen.util.coord import coord_list_mapping_pbc
from pymatgen import Structure,PeriodicSite
from pymatgen.analysis.structure_matcher import StructureMatcher

from utils import *
from ewald_tools import *
from block_tools import CEBlock

from gurobi import *

__author__ = "Fengyu Xie"
__version__ = "2019.0.0"

SITE_TOL = 1e-6

""" 
In this file we will provide socket functions to ground state solver I/O. 
For number of hard clauses issue, we firstly try to reduce the problem size
down to 16 sites.(for UB)
"""
"""
A temporary ver that only provides UB convergence. Debugging LB.
"""

"""
Notes:
    In this version I separated LB interation and UB iteration.
    UB based on supercell, LB based on block.
    I changed the ewald correction in UB (it was intially wrong, only pairs
    within SC should be corrected, should remove periodic boundary condition
    introduced by ClusterSupercell.generate_mapping)
    Then I estimated the ewald correction to LB by peridic ewald value. Not
    mathematically proven.

    ClusterExpansion class already have all the enerigies normalized before fitting.

    Using only gurobi as an optimizer now.
"""

#####
# Tool functions
#####

#####
#Class functions
#####

class GScanonical(MSONable):
    """
    Defines a ground state problem for a generalized ising system. Uses 0/1(cluster counting) formalism.
    """

    def __init__(self, ce, eci, composition, transmat=[[1,0,0],[0,1,0],[0,0,1]],maxsupercell=200, \
                 max_block_range = 1, num_split=10, n_iniframe=20):
        """
        Args:
            ce: a cluster expansion object that you wish to solve the GS of.
            eci: eci's of clusters in ce
            maxsupercell: maximum supercell size. Enumerated matrices with det|M|>maxsupercell will be abandoned.
                          WE ONLY ENUMERATE DIAGONAL MATRICES HERE, based on an assumption that non-diagonal
                          can be captured by larger diagonal periodicity.
                          max enumerated sc diameter is chosen based on max interaction distance in ce divided by
                          primitive cell dimensions
            max_block_range: maximum block extension range. This determines how far an image you want to split your
                             clusters to.(Unit: supercell dimensions)
            selec: number of enumerated supercell matrices to select.
            num_split: number of Symmetrized clusters to split. When num_split = 10, only sc's with top 10 ECI absolute values will be splitted.
            n_iniframe: number of sampled structures to intialize LP in lowerbound problem.
            composition: a list of dictionaries that represents species occupation ratio on each site. For example:
                         [{'Li+':8,'Co3+':6,'Co4+':2},{'O2-':13,'F-':3}]
                         Compositions are read from the CE file constructed by the CE generator.
                         Note: The canonical ensemble is established on SUBLATTICES, so there can't be specie exchange
                               between sublattices. If you want to consider that, you have to do that in the enumerator
                               part!!
        """
        self.ce = ce
        if self.ce.basis != '01':
            raise NotImplementedError("GS solver not implemented for other basis.")
        self.eci = eci
        
        self._bclus_original = None
        self._ecis_original = None
        self._bclus_corrected = None
        self._ecis_corrected = None
        self._bit_inds = None
        self._bclus_ewald = None
        self._ecis_ewald = None

        self.maxsupercell = maxsupercell
        self.max_block_range = max_block_range
        self.num_of_sizes = num_of_sizes
        self.selec = selec
        
        sp_list = []
        for site_occu in composition:
            sp_list.extend(site_occu.values())
        self._comp_step = GCD_List(sp_list)
        # Find the enumeration step for composition, and this will be the minimum enumerated supercell size.
        self.composition=[{sp:site_occu[sp]//self._comp_step for sp in site_occu} for site_occu in composition]
        self.formula_size = int(round(sum(self.composition[0].keys())))        

        self.e_lower = None
        self.str_upper = None
        self.e_upper = None
        self.solved = False  #call function 'solve' to give a value to this
        self.use_ewald = self.ce.use_ewald
        self.use_inv_r = self.ce.use_inv_r

        if self.use_ewald:
            print("Warning: Ewald correction required! But I can not guarantee an LB solution. ")

        self._enumlist = None
        self.transmat = transmat
        if self.transmat != [[1,0,0],[0,1,0],[0,0,1]]:
            print("Using transformation matrix:",self.transmat,"in enumeration.")

        self.num_split = num_split
        self.n_iniframe = n_iniframe
        prim = self.ce.structure
        if self.transmat != [[1,0,0],[0,1,0],[0,0,1]]:
            prim = prim.make_supercell(self.transmat)
        max_radius = max([max([sc.max_radius for sc in self.ce.clusters[sz]]) for sz in self.ce.clusters])
        self.enum_d = max( int(max_radius/prim.lattice.a)+1, int(max_radius/prim.lattice.b)+1,\
                           int(max_radius/prim.lattice.c)+1 )
        self.a = prim.lattice.a
        self.b = prim.lattice.b
        self.c = prim.lattice.c
        # GS structures have to contain at least all types of cluster interactions

####
# Callable interface
####       
    def solve(self):
        if not(self.solved):
            self.solved = self._iterate_supercells()
        if self.solved:
            print("Current ground state solved. Energy: {}, structure: {}.".format(self.e_upper,self.str_upper))
    
    @property
    def enumlist(self):
        """
            Here we only enumerate diagonal matrices, because off-diagonal perioditicites should be captured in
            diagonal supercells.
        """
        if self._enumlist is None:
            _enumlist=[]
            # enumerate 64 supercell matrices
            for i in range(self.enum_d, self.enum_d+4):
                for j in range(self.enum_d,self.enum_d+4):
                    for k in range(self.enum_d,self.enum_d+4):
                        if i*j*k<=self.maxsupercell and (i*j*k)%self.formula_size==0: 
                            #supercell shouldn't be too large, and should be able to host a formula. 
                            ijk_buff = sorted([(i,self.a),(j,self.b),(k,self.c)],key=lambda x:x[1])
                            ijk = [v[0] for v in ijk_buff]
                            _enumlist.append([[ijk[0],0,0],[0,ijk[1],0],[0,0,ijk[2]]])
                            #longest axis are iterated first to make faster convergence!
            if self.transmat: 
                self._enumlist=[mat_mul(sc,self.transmat) for sc in self._enumlist]            

            print("Enumerated supercells generated!")
        #else:    print("use existing enumlist");
        return self._enumlist
 
    @property
    def bclus_original(self):
        if (not self._bclus_original) or (not self._ecis_original) or (not self._bit_inds):
            self._bclus_original = []
            self._ecis_original = []
            self._bit_inds = []

            for mat in self.enumlist:
                clus_sup = self.ce.supercell_from_matrix(mat)

                #Generate a refenrence table for MAXSAT var_id to specie_id. Note: MAXSAT var_id starts from 1
                bit_inds = get_bit_inds(clus_sup.supercell)
                self._bit_inds.append(bit_inds)
                #print(bit_inds)               
                #Here we generate a mapping from clusters to site combos, which will be turned into clauses.
                clusters = self.ce.clusters
                eci = self.eci
                eci_new = {size:[eci[(sc.sc_b_id):(sc.sc_b_id+len(sc.bit_combos))] for sc in clusters[size]] \
                               for size in clusters}
                b_clusters = []
                eci_return = []
                # eci array in danill's code: 
                #eci[0] is zero cluster
                #eci[-1] is ewald cluster
                #sc_b_id starts from 1
                eci_0 = eci[0]
                for sc,sc_inds in clus_sup.cluster_indices:
                    for i,combo_orbit in enumerate(sc.bit_combos):
                        for j,combo in enumerate(combo_orbit):
                            b_clusters.extend([[bit_inds[site][combo[s]] for s,site in enumerate(sc_ind)]\
                                                  for sc_ind in sc_inds])
                            #combo_id = sum([len(orbit) for orbit in sc.bit_combos[:i]])+j
                            bit_multip = sc.multiplicity*len(combo_orbit)
                            eci_return.extend([eci_new[len(sc.bits)][sc.sc_id-clusters[len(sc.bits)][0].sc_id][i]/bit_multip\
                                                   for sc_ind in sc_inds])
                            #one eci for one orbit
                self._bclus_original.append(b_clusters)
                self._ecis_original.append(eci_return)
        else:
            return self._bclus_original
    
    @property
    def ecis_original(self):
        if (not self._bclus_original) or (not self._ecis_original) or (not self._bit_inds):
            bclus_original = self.bclus_original
        return self._ecis_original

    @property
    def bit_inds(self):
        if (not self._bclus_original) or (not self._ecis_original) or (not self._bit_inds):     
            bclus_original = self.bclus_original
        return self._bit_inds

    @property
    def bclus_ewald(self):
        if (not self._bclus_ewald) or (not self._ecis_ewald):
            self._bclus_ewald = []
            self._ecis_ewald = []
            if self.use_ewald:
                for mat in self.enumlist:
                    bclus_ewald,eci_ewald = ewald_correction(self.ce,mat)
                    self._bclus_ewald.append(bclus_ewald)
                    self._ecis_ewald.append(eci_ewald)
        return self._bclus_ewald

    @property
    def ecis_ewald(self):
        if (not self._ecis_ewald) or (not self._bclus_ewald):           
            bclus_ewald = self.bclus_ewald
        return self._ecis_ewald

    @property
    def bclus_corrected(self):
        """
            Returns a list of ewald corrected bit_clusters, each element for a matrix in self.enumlist.
        """

        if (not self._bclus_corrected) or (not self._ecis_corrected):
            self._bclus_corrected = []
            self._ecis_corrected = []

            for m,mat in enumerate(self.enumlist):
                #Doing real corrections to ecis
                b_cluster_ew = self.bclus_ewald[m].copy()
                eci_return_ew = self.ecis_ewald[m].copy()
                b_clusters = self.bclus_original[m].copy()
                eci_return = self.ecis_original[m].copy()

                for b_cluster_ew,b_eci_ew in zip(b_clusters_ew,eci_return_ew):
                    _in_b_clusters = False
                    for bc_id,b_cluster in enumerate(b_clusters):
                        if len(b_cluster)!=len(b_cluster_ew):
                            continue
                        if len(b_cluster)==2:
                            if b_cluster == b_cluster_ew or Reversed(b_cluster)==b_cluster_ew:                        
                                eci_return[bc_id]=eci_return[bc_id] + 2*b_eci_ew
                            #*2 because a pair only appear in b_clusters_ew for once, but should be summed twice
                                _in_b_clusters = True
                                break
                        elif len(b_cluster)==1:
                            if b_cluster == b_cluster_ew:
                                eci_return[bc_id]=eci_return[bc_id] + b_eci_ew
                                _in_b_clusters = True
                                break
                    if not _in_b_clusters:
                        b_clusters.append(b_cluster_ew)
                        eci_return.append(b_eci_ew if len(b_cluster_ew)==1 else b_eci_ew*2)

                self._bclus_corrected.append(b_clusters)
                self._ecis_corrected.append(eci_return)

        return self._bclus_corrected

    @property
    def ecis_corrected(self):
        if (not self._ecis_corrected) or (not self._bclus_corrected):
            bclus_corrected = self.bclus_corrected
        return self._ecis_corrected
####
# Private tools for the class
#### 
    def _iterate_supercells(self):
        for mat_id,mat in enumerate(self.enumlist):
            print("Solving on supercell matrix:",mat)
            #try:
            cur_e_upper,cur_str_upper=self._solve_upper(mat_id)
            print("Current GS upper-bound: %f"%cur_e_upper)
            cur_e_lower=self._solve_lower(mat_id)
            print("Current GS lower_bound: %f"%cur_e_lower)
            if (self.e_upper is None and self.e_lower is None) \
                or abs(self.e_upper-self.e_lower)>abs(cur_e_upper-cur_e_lower):
            #self.e_lower = cur_e_lower
                old_e_upper = self.e_upper
                old_e_lower = self.e_lower
                self.e_upper = cur_e_upper
                self.e_lower = cur_e_lower
                self.str_upper = cur_str_upper
                if not(old_e_upper is None) and not(old_e_lower is None) and abs(self.e_upper-self.e_lower)<0.001:
                    print('UB and LB converged on matrix:',mat)
                    return True
                else:
                    print("Current UB:{}, current LB:{}, not converged!".format(self.e_upper,self.e_lower))
           # except:
           #     cur_e_upper=None
           #     cur_str_upper=None
           #     cur_e_lower=None
           #     self.e_upper = None
           #     self.e_lower = None
           #     self.str_upper = None
           #     print("GS UB or LB for {} not found! Skipping".format(mat))           
        return False
            
    def _solve_upper(self,mat_id):
        """
            Now using gurobi.
        """
        b_clusters_new = self.bclus_corrected[mat_id]
        ecis_new = self.ecis_corrected[mat_id]
       
        num_of_vars = max(chain(*bit_ind))
        #switched to gurobi
        m = Model()
        x = m.addVars(num_of_vars,vtype=GRB.BINARY)

        # Add hard clauses(sum(x_site)<=1)
        self._add_hard_constrs(m,x,mat_id)
        
        # Add canonical constraints
        self._add_ca_constrs(m,x,mat_id)

        # Add objective
        obj = GenExpr()
        for bclus,eci in zip(b_clusters_new,ecis_new):
            clause = 1
            for bit in bclus:
                clause = clause*x[bit-1]
            obj += eci*clause
        m.setObjective(obj,GRB.MINIMIZE)
        m.setParam(GRB.Param.TimeLimit, 1800)

        #Update to register all the new variables and constraints.
        m.update()
        m.optimize()
        
        maxsat_res = [(x_id+1) if x[x_id].x else -(x_id+1) for x_id in x]
        maxsat_res = sorted(maxsat_res,key=lambda x:abs(x))
        cs = self.ce.supercell_from_matrix(self.enumlist[mat_id])
        cs_bits = get_bits(cs.supercell)

        upper_sites = []
        #print(bit_ind)
        for s,site in enumerate(bit_ind):
            should_be_ref = True
            st = cs.supercell[s]
            for v_id,var_id in enumerate(site):
            #For all variables on a site, only one could be true. If no hard cluases fail.
                if maxsat_res[var_id-1]>0:
                    upper_sites.append(PeriodicSite(cs_bits[s][v_id],st.frac_coords,st.lattice))
                    should_be_ref = False
                    break
            if should_be_ref:
                upper_sites.append(PeriodicSite(cs_bits[s][-1],st.frac_coords,st.lattice))

        upper_str = Structure.from_sites(upper_sites)
        #print("UB structure:",upper_str)
        #upper_cs = self.ce.supercell_from_structure(upper_str)
        upper_corr = cs.corr_from_structure(upper_str)
        upper_e = np.dot(np.array(self.eci),upper_corr)
        return upper_e,upper_str

    def _solve_lower(self,mat_id):
        """
    As known, the energy of a periodic structure can be expressed as shown in Wenxuan's 16 paper (PRB 94.134424, 2016), formula 6a. 
    The block is mathematically the summbation of cluster interaction terms within a supercell (periodicity) and within 
    an environment around a supercell. The energy of the whole structure is the sum of all block energies.
    Here we try to find the minimal block energy with LP. all variables within block are allowed to relax during parametized MAXSAT.
    
    block_range: the largest distance to split your clusters away. for ex,if you have a cubic supercell with 3 dimensions L*L*L 
                 block_range = 3 means you shall only split the interactions in the block to the supercells 3* L away. This is to
                 make sure that you don't generate too many variables for MAXSAT.
    Warning: 1, For systems with ce.use_ewald, I currently don't have theoretically rigorous idea to correct ewald energy in blocks.
                If I want to take the ewald energy accuratly into account, I have to extend the block to infinitly large because the
                ewald term is infinitly ranged.
                Currently I'm assuming that no geometric frustration appears in the LB state so I can still use the correction for 
                upperbound. The electrostatic parts ARE NOT EXTENDED! This is not mathematically proved. One must show that all 
                ionic structures should have no less energy per unit cell than a periodic one, for the approximation above to be 
                rigorous.
             2, Will try cluster tree opt later.
    (Note: total structure energy is usually normalized with a factor N (size of supercell). Danill has already done this in pyabinitio,
           so don't normalize again!)
        """

        blk = CEBlock(self.ce,self.enumlist[mat_id], self.eci, self.composition, bclus_ewald = self.bclus_ewald[mat_id],\
                      eci_ewald = self.ecis_ewald[mat_id], block_range=self.max_block_range, n_iniframe=self.n_iniframe)         

        #### Calling Gurobi ####
        lower_e = blk.solve()
        return lower_e

    def _add_hard_constrs(self,m,x,mat_id):
        bit_ind = self.enumlist[mat_id]
        for site in bit_ind:
            hard = LinExpr()
            for bit in site:
                hard+=x[bit-1]
            m.addConstr(hard <= 1)

    def _add_ca_constrs(self,m,x,mat_id):
        bit_ind = self.enumlist[mat_id]
        sc_size = int(round(np.abs(np.linalg.det(self.enumlist[mat_id]))))
        scale = sc_size//self.formula_size

        scaled_composition = [{sp:sublat[sp]*comp_scale for sp in sublat} for sublat in self.composition]
        specie_names = [[str(sp) for sp in sorted(sublat.species_and_occu.keys())] for sublat in self.ce.structure]
        sublats = [bit_ind[i*sc_size:(i+1)*sc_size] for i in range(0,N_sites//sc_size)]
        bits_sp_sublat = [[[sublat[s_id][sp_id] for s_id in range(len(sublat))] \
                            for sp_id in range(len(sublat[0]))] for sublat in sublats]
        for sublat_id,sublat in enumerate(bits_sp_sublat):
            for sp_id,sp_bits in enumerate(sublat):
                sp_name = specie_names[sublat_id][sp_id]
                ca_expression = LinExpr()
                for bit in sp_bits:
                    ca_expression+=x[bit-1]
                m.addConstr(ca_expression = scaled_composition[sublat_id][sp_name])

####
# I/O interface
####
    @classmethod
    def from_dict(cls,d):
        ce = ClusterExpansion.from_dict(d['cluster_expansion'])
        gs_socket= cls(ce,d['ecis'],d['composition']) 
        #essential terms for initialization
        if 'maxsupercell' in d:
            gs_socket.maxsupercell=d['maxsupercell']
        if 'max_block_range' in d:
            gs_socket.max_block_range=d['max_block_range']

        if 'num_split' in d:
            gs_socket.num_split = d['num_split']
        if 'n_iniframe' in d:
            gs_socket.n_iniframe = d['n_iniframe']
        if 'e_lower' in d:
            gs_socket.e_lower=d['e_lower']        
        if 'e_upper' in d:
            gs_socket.e_upper=d['e_upper']
        #Only upper-bound solver gives a bit ordering
        if 'str_upper' in d:
            gs_socket.str_upper=d['str_upper']
        if 'transmat' in d:
            gs_socket.transmat=d['transmat']
        if 'enumlist' in d:
            gs_socket._enumlist=d['enumlist']
        if 'bclus_original' in d and 'ecis_original' in d and 'bit_inds' in d and 'bclus_ewald' in d\
            and 'ecis_ewald' in d:
            gs_socket._bclus_original = d['bclus_original']
            gs_socket._ecis_original = d['ecis_original']
            gs_socket._bit_inds = d['bit_inds']
            gs_socket._bclus_ewald = d['bclus_ewald']
            gs_socket._ecis_ewald = d['ecis_ewald']
        if ('e_lower' in d) and ('e_upper' in d) and np.abs(d['e_lower']-d['e_upper'])<=0.001 and ('str_upper' in d):
            gs_socket.solved=True
            #print("Loaded instance already has a solved GS. GS energy: %f"%(gs_socket.e_upper))
            #print("Under composition:",gs_socket.composition)
        else:
            print("Loaded cluster expansion needs GS solution. Use self.solve() method to generate one.")
        return gs_socket
    
    def as_dict(self):
        if not self.solved:
            print("Warning: Your ground state has not been solved yet. If you haven't tried, please go back.")
        return {'cluster_expansion':self.ce.as_dict(),\
                'ecis':self.eci,\
                'composition':self.composition,\
                'maxsupercell':self.maxsupercell,\
                'max_block_range':self.max_block_range,\
                'num_split':self.num_split,\
                'n_iniframe':self.n_iniframe,\
                'e_lower':self.e_lower,\
                'e_upper':self.e_upper,\
                'str_upper':self.str_upper.as_dict(),\
                'transmat':self.transmat,\
                'enumlist':self.enumlist,\
                'bclus_original':self.bclus_original,\
                'ecis_original':self.ecis_original,\
                'bit_inds':self.bit_inds,\
                'bclus_ewald':self.bclus_ewald,\
                'ecis_ewald':self.ecis_ewald,\
                '@module':self.__class__.__module__,\
                '@class':self.__class__.__name__\
               }

    def as_file(self,gs_solver_file='gs_solver.mson'):
        with open(gs_solver_file,'w') as fout:
            json.dump(self.as_dict(),fout)

    @classmethod
    def from_file(cls,gs_solver_file='gs_solver.mson'):
        with open(gs_solver_file,'r') as fin:
            gsd = json.load(fin)
        return cls.from_dict(gsd)
####
# Canonical to Semi-grand
####

def solvegs_for_hull(ce_file='ce.mson',calc_data_file='calcdata.mson',outdir = 'gs_run',share_enumlist=True):
    """
    Here we generate all the canonical ground states for the hull in calc_data_file, and store them in gs_file.
    Output:
        A boolean variable. Indicating whether the GS has converged for hull. Currently only judging from energy.
    """
    if not(os.path.isdir(outdir)):
        os.makedirs(outdir)        
    #creating a folder to store previous gs calculations.    

    if not(os.path.isfile(ce_file)) or not(os.path.isfile(calc_data_file)):
        print("No valid cluster expansion and calculation datas detected, can't solve GS!")
        return

    # Will use existing settings to keep setting same across hull.
    with open(calc_data_file) as calc_data:
        calcdata = json.load(calc_data)
    with open(ce_file) as ce_data:
        cedata = json.load(ce_data)

    ecis = cedata['ecis']
    ce = ClusterExpansion.from_dict(cedata['cluster_expansion'])
    gss = {}

    for compstring in calcdata['compositions']:
        composition=json.loads(compstring)
        filename = os.path.join(outdir,compstring)
        if len(os.listdir(outdir)) and share_enumlist:
            print("Using existing generator settings.")
            gs_socket=GScanonical.from_file(os.path.join(outdir,os.listdir(outdir)[-1]))
            #clearing out previous results, only using old settings and el-corrections.
            gs_socket.solved = False
            gs_socket.e_upper = None
            gs_socket.e_lower = None
            gs_socket.str_upper = None
        else:
            gs_socket = GScanonical(ce,ecis,composition)
           
        print("Solving for composition:",compstring)
        gs_socket.solve()
        gss[compstring]={}
        gss[compstring]['e_upper']=gs_socket.e_upper
        gss[compstring]['e_lower']=gs_socket.e_lower
        gss[compstring]['str_upper']=gs_socket.str_lower
        gss[compstring]['solved']=gs_socket.solved
        gs_socket.as_file(filename)
        #if gs_socket.solved:

    print("Solved GS structures on {} hull points.".format(len(gss)))

    return gss

# We don't update gss into structures any more.
#def writegss_to_vasprun(gs_file='gs.mson',vasprun='vasp_run',vs_file='vasp_settings.mson'):
#    sm = StructureMatcher(ltol=0.2, stol=0.15, angle_tol=5, comparator=ElementComparator())
#    
#    calculated_structures = {}
#    targetdirs = {}
#    maxids = {}
#    _was_generated = lambda x: 'POSCAR' in x and not 'KPOINTS' in x and not 'INCAR' in x and not 'POTCAR' in x
#    if os.path.isdir(vasprun):
#        print("Checking previously enumerated structures.")
#        for root,dirs,files in os.walk(vasprun):
#            if _was_generated(files):
#                parentdir = os.path.join(*root.split(os.sep)[0:-1])
#                with open(os.join(parentdir,'composition_by_site')) as comp_file:
#                    composition_old = json.load(comp_file)
#                    compstring_old = json.dumps(composition)
#                if compstring_old not in calculated_structures:
#                    calculated_structures[compstring_old]=[]
#                calculated_structures[compstring_old].append(Poscar.from_file(os.join(root,'POSCAR').structure)) 
#                if compstring_old not in targetdirs:
#                    targetdirs[compstring_old]=parentdir
#                if compstring_old not in maxids:
#                    maxids[compstring_old]=max([int(idx) for idx in os.listdir(parentdir) if RepresentsInt(idx)])
#    else: 
#        print("Previous calculation or generator setting file missing. Exiting")
#        return
#    
#    with open(gs_file) as gs_in:
#        gss = json.load(gs_in)
#    
#    if os.path.isfile(vs_file):
#        with open(vs_file) as vs_in:
#            vasp_settings = json.load(vs_in)
#    else:
#        vasp_settings = None
#
#    for compstring in gss:
#        _unique = True
#        if compstring in calculated_structures:
#            gstruct = Structure.from_dict(gss[compstring]['gs_structure'])
#            for ostruct in calculated_structures[compstring]:
#                if sm.fit(ostruct,gstruct):
#                    print("GS for composition:\n{}\nalready calculated. Skipping.".format(compstring))
#                    _unique = False
#                    break
#        if _unique:
#            targetdir = targetdirs[compstring]
#            writedir = os.path.join(targetdir,str(maxids[compstring]+1))
#            if not os.path.isdir(writedir): os.mkdir(writedir)
#            Poscar(gstruct.get_sorted_structure()).write_file(os.path.join(writedir,'POSCAR'))
#            vaspdir = os.path.join(writedir,'fm.0')
#            if vasp_settings:
#                print("Applying VASP settings",vasp_settings)
#                if 'functional' in vasp_settings:
#                    functional = vasp_settings['functional']
#                else:
#                    functional = 'PBE'
#                if 'num_kpoints' in vasp_settings:
#                    num_kpoints = vasp_settings['num_kpoints']
#                else:
#                    num_kpoints = 25
#                if 'additional_vasp_settings' in vasp_settings:
#                    additional = vasp_settings['additional_vasp_settings']
#                else:
#                    additional = None
#                if 'strain' in vasp_settings:
#                    strain = vasp_settings['strain']
#                else:
#                    strain = ((1.01,0,0),(0,1.05,0),(0,0,1.03))
#            
#                write_vasp_inputs(gstruct,vaspdir,functional,num_kpoints,additional,strain)
#            else:
#                print("Using CEAuto default VASP settings.")
#                write_vasp_inputs(gstruct,vaspdir)
#
#            print('New GS written to {}.'.format(writedir))

    
    
