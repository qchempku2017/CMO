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

from itertools import combinations,permutations
from pymatgen.analysis.ewald import EwaldSummation
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer, SymmOp
from pymatgen.util.coord import coord_list_mapping_pbc
from pymatgen import Structure,PeriodicSite
from pymatgen.analysis.structure_matcher import StructureMatcher

from global_tools import *
from block_tools import CEBlock

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

    def __init__(self, ce, eci, composition, transmat=[[1,0,0],[0,1,0],[0,0,1]],maxsupercell=16, \
                 max_block_range = 1,num_of_sizes=4, selec=20, hard_marker=1000000000000000, \
                 eci_mul=1000000,solver='CCEHC-incomplete',\
                 num_split=10,n_iniframe=20):
        """
        Args:
            ce: a cluster expansion object that you wish to solve the GS of.
            eci: eci's of clusters in ce
            maxsupercell: maximum supercell sizes
            max_block_range: maximum block extension range. This determines how far an image you want to split your
                             clusters to.(Unit: supercell dimensions)
            num_of_sizes: number of supercell sizes to be enumerated. When num_of_sizes = 4 while maxsupercell = 16,
                          enumerated sizes are [4,8,12,16]
            selec: number of enumerated supercell matrices to select.
            solver: the MAXSAT solver used to solve the upper bound problem. Default: CCEHC-incomplete.
                    (We use incomplete solvers to make the time consumption tractable for MAXSAT)
            hard_marker: the marking number of hard clauses in MAXSAT. Should be sufficiently large (>sum(ECI)).
            eci_mul: since many MAXSAT only take integer weights, we multiply and floor ECI's with this to make integers.
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
        self.eci = eci
        self._bclus_corrected = None
        self._ecis_corrected = None
        self._bit_inds = None

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

        self.solver = solver
        self.e_lower = None
        self.str_upper = None
        self.e_upper = None
        self.solved = False  #call function 'solve' to give a value to this
        self.use_ewald = self.ce.use_ewald
        self.use_inv_r = self.ce.use_inv_r
        self._enumlist = None
        self.transmat = transmat
        if self.transmat != [[1,0,0],[0,1,0],[0,0,1]]:
            print("Using transformation matrix:",self.transmat,"in enumeration.")

        self.hard_marker = hard_marker
        self.eci_mul = eci_mul
        self.num_split = num_split
        self.n_iniframe = n_iniframe
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
            Here we only enumerate upper-triangle matrices!
        """
        if not(self._enumlist):
            _enumlist=[]
            scale = int(abs(np.linalg.det(self.transmat)))

            min_size = sum(self.composition[0].values())
            enumrange = list(range(int(self.maxsupercell/self.num_of_sizes),self.maxsupercell+1,\
                              int(self.maxsupercell/self.num_of_sizes)))
            enumrange = [size for size in enumrange if size>min_size]
            # Size can't be too small in case it can't host a composition ratio.
            if len(enumrange)==0:
                print('No enumerated size acceptable. You should consider adjusting your compostion enumeration\
                       step in generator, or adjust maxsupercell to a proper value (Warning: a value >16 not \
                       recommended.)')
                return

            for size in enumrange:
                print("Enumerating for size %d"%size)
                _enumlist.extend(Get_Hermite_Matricies(int(size/scale)))

            print("Randomly picking supercell matrices.")
            self._enumlist=random.sample(_enumlist,self.selec)
            if self.transmat: 
                self._enumlist=[mat_mul(sc,self.transmat) for sc in self._enumlist]
            self._enumlist=sorted(self._enumlist,key=lambda a:(abs(np.linalg.det(a)),\
                                 np.linalg.norm(a[0]),np.linalg.norm(a[1]),np.linalg.norm(a[2])))
            print("Enumerated supercells generated!")
        #else:    print("use existing enumlist");
        return self._enumlist
    
    @property
    def bclus_corrected(self):
        """
            Returns a list of ewald corrected bit_clusters, each element for a matrix in self.enumlist.
        """

        if (not self._bclus_corrected) or (not self._ecis_corrected) or (not self._bit_inds):
            self._bclus_corrected = []
            self._ecis_corrected = []
            self._bit_inds = []

            if self.use_ewald:
                print("Ewald correction required!")

            for mat in self.enumlist:
                clus_sup = self.ce.supercell_from_matrix(mat)

                #Generate a refenrence table for MAXSAT var_id to specie_id. Note: MAXSAT var_id starts from 1
                bit_inds = []
                b_id = 1
                for i,site in enumerate(clus_sup.supercell):
                    site_bit_inds = []
                    for specie_id in range(len(site.species_and_occu)-1):
                    #-1 since a specie on the site is taken as reference
                        site_bit_inds.append(b_id)
                        b_id+=1
                    bit_inds.append(site_bit_inds)
                print('%d variables will be used in MAXSAT.'%(b_id-1))
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
                for sc,sc_inds in clus_sup.cluster_indices:
                    for i,all_combo in enumerate(sc.bit_combos):
                        for combo in all_combo:
                            b_clusters.extend([[ bit_inds[site][combo[s]] for s,site in enumerate(sc_ind)]\
                                                  for sc_ind in sc_inds])
                            eci_return.extend([eci_new[len(sc.bits)][sc.sc_id-clusters[len(sc.bits)][0].sc_id][i]\
                                                   for sc_ind in sc_inds])

                if self.use_ewald:
                    #Here we do electro-static correction. Reference zero energy state is the one that all sites 
                    #are occupied by reference compound.
                    print("Making up all ewald interactions for supercell:",mat)

                    ew_str = Structure.from_sites([PeriodicSite('H+',s.frac_coords,s.lattice) for s in clus_sup.supercell])
                    H = EwaldSummation(ew_str,eta=self.ce.eta).total_energy_matrix

                    #Ewald energy E_ew = (q+r)*H*(q+r)'. I used a stupid way to get H but quite effective.
                    supbits = get_bits(clus_sup.supercell)
                    r = np.array([GetIonChg(bits[-1]) for bits in supbits])
                    chg_bits = [[GetIonChg(bit)-GetIonChg(bits[-1]) for bit in bits[:-1]] for bits in supbits]
                    H_r = np.dot(H,r)
                    
                    b_clusters_ew = []
                    eci_return_ew = []

                    if not self.use_inv_r:
                        eci_ew = eci[-1]

                        for i in range(len(bit_inds)):
                            for j in range(i,len(bit_inds)):
                                for k in range(len(bit_inds[i])):
                                    for l in range((k if i==j else 0),len(bit_inds[j])):
                                        if bit_inds[i][k]!=bit_inds[j][l]:
                                            bit_a = bit_inds[i][k]
                                            bit_b = bit_inds[j][l]
                                            b_clusters_ew.append([bit_a,bit_b]) 
                                            eci_return_ew.append(eci_ew* (chg_bits[i][k]*chg_bits[j][l]*H[i][j]))
                                        else:
                                            bit = bit_inds[i][k]
                                            b_clusters_ew.append([bit])
                                            eci_return_ew.append(eci_ew*(chg_bits[i][k]**2*H[i][i]+2*chg_bits[i][k]*H_r[i]))

                    else:
                        #When using inv_r, an independent ewald sum is generated for each specie-specie pair, and the sums are
                        #considered components of corr
                        N_sp = sum([len(site.species_and_occu) for site in clus_sup.supercell])
                        N_eweci = N_sp+N_sp*(N_sp-1)//2
                        eci_ew = eci[-N_eweci:]
                        
                        equiv_sites = []
                        for sc,inds in clus_sup.cluster_indices:
                            if len(sc.bits)>1:
                                break
                            equiv_sites.append(inds[:,0])

                        sp_list = []
                        sp_id = 0
                        for sublat in equiv_sites:
                            sublat_sp_list = []
                            for specie_id in bit_inds[sublat[0]]:
                                sublat_sp_list.append(sp_id)
                                sp_id += 1
                            sp_list.extend([sublat_sp_list]*len(sublat))
                       
                        for i in range(len(bit_inds)):
                            for j in range(i,len(bit_inds)):
                                for k in range(len(bit_inds[i])):
                                    for l in range((k if i==j else 0),len(bit_inds[j])):
                                        if bit_inds[i][k]!=bit_inds[j][l]:
                                            bit_a = bit_inds[i][k]
                                            bit_b = bit_inds[j][l]
                                            b_clusters_ew.append([bit_a,bit_b])
                                            id_a = sp_list[i][k]
                                            id_b = sp_list[j][l]
                                            id_abpair = id_a*(2*N_sp-id_a-1)//2 + id_b - id_a -1 # Serial id of a,b pair in eci_ew list.
                                            eci_return_ew.append(eci_ew[N_sp+id_abpair]* (chg_bits[i][k]*chg_bits[j][l]*H[i][j]))
                                        else: #Point terms
                                            bit = bit_inds[i][k]
                                            b_clusters_ew.append([bit])
                                            id_bit = sp_list[i][k]
                                            point_eci = eci_ew[id_bit]*chg_bits[i][k]**2*H[i][i]
                                            for m in range(len(bit_inds)):
                                                id_a = id_bit
                                                id_b = sp_list[m][-1] #id of the reference specie
                                                id_abpair = id_a*(2*N_sp-id_a-1)//2 + id_b -id_a -1
                                                #Calculate H_r term with weight!
                                                point_eci += 2*chg_bits[i][k]*H[i][m]*r[m]*eci_ew[N_sp+id_abpair]
                                            eci_return_ew.append(point_eci)
                                            # eci_return_ew.append(eci_ew*(chg_bits[i][k]**2*H[i][i]+2*chg_bits[i][k]*H_r[i]))
                    #Corrections
                    for b_cluster_ew,b_eci_ew in zip(b_clusters_ew,eci_return_ew):
                        _in_b_clusters = False
                        for bc_id,b_cluster in enumerate(b_clusters):
                            if len(b_cluster)>2:
                                continue
                            #b_cluster won't duplicate. Trust Danill.
                            if b_cluster == b_cluster_ew or Reversed(b_cluster)==b_cluster_ew:                        
                                eci_return[bc_id]=eci_return[bc_id] + b_eci_ew*2
                                #*2 because a pair only appear in b_clusters_ew for once, but should be summed twice
                                _in_b_clusters = True
                                break
                        if not _in_b_clusters:
                            b_clusters.append(b_cluster_ew)
                            eci_return.append(b_eci_ew*2)
        
                self._bclus_corrected.append(b_clusters)
                self._ecis_corrected.append(eci_return)

        return self._bclus_corrected

    @property
    def ecis_corrected(self):
        if (not self._ecis_corrected) or (not self._bclus_corrected) or (not self._bit_inds):
            bclus_corrected = self.bclus_corrected
        return self._ecis_corrected

    @property
    def bit_inds(self):
        if (not self._ecis_corrected) or (not self._bclus_corrected) :
            bclus_corrected = self.bclus_corrected
        return self._bit_inds

####
# Private tools for the class
#### 
    def _iterate_supercells(self):
        for mat_id,mat in enumerate(self.enumlist):
            #Here we will convert problems into MAXSAT and LP standard inputs, solve and analyze the outputs
            print("Solving on supercell matrix:",mat)
            sc_size  = int(round(np.abs(np.linalg.det(self.enumlist[mat_id]))))
            compstat = sum(self.composition[0].values())

            if sc_size%compstat !=0:
                print("Supercell {} can not have composition {}. Skipping.".format(self.enumlist[mat_id],self.composition))
                continue

            #print("Warning: Solving UB only.")
            #try:
            cur_e_upper,cur_str_upper=self._solve_upper(mat_id)
            print("Current GS upper-bound: %f"%cur_e_upper)
            cur_e_lower=self._solve_lower(mat_id)
            print("Current GS lower_bound: %f"%cur_e_lower)
            if (self.e_upper is None and self.e_lower is None) or abs(self.e_upper-self.e_lower)>abs(cur_e_upper-cur_e_lower):
                #self.e_lower = cur_e_lower
                old_e_upper = self.e_upper
                old_e_lower = self.e_lower
                self.e_upper = cur_e_upper
                self.e_lower = cur_e_lower
                self.str_upper = cur_str_upper
                if not(old_e_upper is None) and not(old_e_lower is None) and abs(self.e_upper-self.e_lower)<0.001:
                    print('UB and LB converged on matrix:',mat)
                    return True
            #except:
            #    cur_e_upper=None
            #    cur_str_upper=None
            #    print("Current GS upper-bound not found! Skipping")
            

        return False

    def _electrostatic_correction(self,mat_id):
        """
            This part generates bit_clusters and ecis to further convert into MAXSAT clauses from a 
        ClusterSupercell object.
            For each species on each site of the cluster_supercell, a unique,1-based b_id is assigned. For ex.
            0.0 0.0 0.0 Au(1) Cu(2) Vac(3)
            0.0 0.5 0.0 Zn(4) Mn(5)
            ...
            bit_clusters are represented as tupled combinations of b_id's,such as (1) and (3,2,5). 
            Symmetry anlyzer will generate all symetrically equivalent b_clusters from symmetrized_bit_clusters(bit_combos) in pyabinitio,
        and assign an eci to them each as ECI_sc/multiplicity.
            If the self.use_ewald, will add the corresponding ewald matrix element to bit_clusters' ecis. If self.use_inv_r, a list of
        such matrix elements will be added. Finally, will remove the ewald related term(s).
            At the beginning of initialization, the self.enumlist and ewald corrections would already have been done. You gotta save a lot of time.
            mat_id: current supercell matrix id.
        """
        
        return self.bclus_corrected[mat_id],self.ecis_corrected[mat_id],self.bit_inds[mat_id]
            
    def _solve_upper(self,mat_id):
        """
            Warning: hard_marker is the weight assigened to hard clauses. Should be chosen as a big enough number!
        """
        #### Input Preparation ####
        b_clusters_new,ecis_new,site_specie_ids=self._electrostatic_correction(mat_id)
        #print(site_specie_ids)

        sc_size = int(round(np.abs(np.linalg.det(self.enumlist[mat_id]))))
        specie_names = [[str(sp) for sp in sorted(sublat.species_and_occu.keys())] for sublat in self.ce.structure]
        #dict.keys() gives a special generator called 'Keyview', not list, and thus can not be indexed. 
        #convert into list first!
        Write_MAXSAT_input(b_clusters_new,ecis_new,site_specie_ids,sc_size=sc_size,conserve_comp=self.composition,\
                           sp_names=specie_names, hard_marker=self.hard_marker, eci_mul=self.eci_mul)

        #### Calling MAXSAT ####
        Call_MAXSAT(solver=self.solver)

        #### Output Processing ####
        maxsat_res = Read_MAXSAT()
        if len(maxsat_res)==0:
            print("Error:Solution not found for supercell {}!".format(self.enumlist[mat_id]))

        cs = self.ce.supercell_from_matrix(self.enumlist[mat_id])
        cs_bits = get_bits(cs.supercell)

        upper_sites = []
        print(site_specie_ids)
        for s,site in enumerate(site_specie_ids):
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

        clus_sup = self.ce.supercell_from_matrix(self.enumlist[mat_id])
        blk = CEBlock(clus_sup, self.eci, self.composition, block_range=self.max_block_range,hard_marker=self.hard_marker,eci_mul=self.eci_mul,num_of_sclus_tosplit=self.num_split,n_iniframe=self.n_iniframe)           
        #### Calling Gurobi ####
        lower_e = blk.solve()
        return lower_e

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
        if 'num_of_sizes' in d:
            gs_socket.num_of_sizes=d['num_of_sizes']
        if 'selec' in d:
            gs_socket.selec = d['selec']
        if 'solver' in d:
            gs_socket.solver = d['solver']
        if 'hard_marker' in d:
            gs_socket.hard_marker = d['hard_marker']
        if 'eci_mul' in d:
            gs_socket.eci_mul = d['eci_mul']
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
        if 'enumlist=' in d:
            gs_socket._enumlist=d['enumlist']
        if 'bclus_corrected' in d and 'ecis_corrected' in d and 'bit_inds' in d:
            gs_socket._bclus_corrected = d['bclus_corrected']
            gs_socket._ecis_corrected = d['ecis_corrected']
            gs_socket._bit_inds = d['bit_inds']
        if ('e_lower' in d) and ('e_upper' in d) and np.abs(d['e_lower']-d['e_upper'])<=0.001 and ('str_upper' in d):
            gs_socket.solved=True
            print("Loaded instance already has a solved GS. GS energy: %f"%(gs_socket.e_upper))
            print("Under composition:",gs_socket.composition)
        else:
            print("Loaded cluster expansion needs GS solution. Use self.solve() method to generate one.")
        return gs_socket
    
    def as_dict(self):
        if not self.solved:
            print("Your ground state has not been solved yet, go back.")
            return
        return {'cluster_expansion':self.ce.as_dict(),\
                'ecis':self.eci,\
                'composition':self.composition,\
                'maxsupercell':self.maxsupercell,\
                'max_block_range':self.max_block_range,\
                'num_of_sizes':self.num_of_sizes,\
                'selec':self.selec,\
                'solver':self.solver,\
                'hard_marker':self.hard_marker,\
                'eci_mul':self.eci_mul,\
                'num_split':self.num_split,\
                'n_iniframe':self.n_iniframe,\
                'e_lower':self.e_lower,\
                'e_upper':self.e_upper,\
                'str_upper':self.str_upper.as_dict(),\
                'transmat':self.transmat,\
                'enumlist':self.enumlist,\
                'bclus_corrected':self.bclus_corrected,\
                'ecis_corrected':self.ecis_corrected,\
                'bit_inds':self.bit_inds,\
                '@module':self.__class__.__module__,\
                '@class':self.__class__.__name__\
               }

    def write_settings(self,gs_settings='gs_settings.mson',use_same_setting=True):
        if use_same_setting and not os.path.isfile(gs_settings):
            print("Using same setting across the hull, but setting file is not written. Writing into {}".format(gs_settings))
            settings = {'maxsupercell':self.maxsupercell,\
                        'max_block_range':self.max_block_range,\
                        'num_of_sizes':self.num_of_sizes,\
                        'selec':self.selec,\
                        'enumlist':self.enumlist,\
                        'solver':self.solver,\
                        'hard_marker':self.hard_marker,\
                        'eci_mul':self.eci_mul,\
                        'num_split':self.num_split,\
                        'n_iniframe':self.n_iniframe,\
                       }
            with open(gs_settings,'w') as fout:
                json.dump(settings,fout)


####
# Canonical to Semi-grand
####

def solvegs_for_hull(ce_file='ce.mson',calc_data_file='calcdata.mson',gs_setting_file='gs_settings.mson',gs_file = 'gs.mson',share_enumlist=True):
    """
    Here we generate all the canonical ground states for the hull in calc_data_file, and store them in gs_file.
    Output:
        A boolean variable. Indicating whether the GS has converged for hull. Currently only judging from energy.
    """
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

        if os.path.isfile(gs_setting_file):
            print("Using existing generator settings from {}.".format(gs_setting_file))
            with open(gs_setting_file) as fin:
                gs_settings = json.load(fin)
        else:
            gs_settings = {}

        gs_socket = GScanonical(ce,ecis,composition)
        if 'maxsupercell' in gs_settings:
            gs_socket.maxsupercell=gs_settings['maxsupercell']
        if 'max_block_range' in gs_settings:
            gs_socket.max_block_range=gs_settings['max_block_range']
        if 'num_of_sizes' in gs_settings:
            gs_socket.num_of_sizes=gs_settings['num_of_sizes']
        if 'selec' in gs_settings:
            gs_socket.selec=gs_settings['selec']
        if 'solver' in gs_settings:
            gs_socket.solver=gs_settings['solver']
        if 'enumlist' in gs_settings and share_enumlist:
            gs_socket._enumlist=gs_settings['enumlist']
        if 'hard_marker' in gs_settings:
            gs_socket.hard_marker = gs_settings['hard_marker']
        if 'eci_mul' in gs_settings:
            gs_socket.eci_mul = gs_settings['eci_mul']
        if 'num_split' in gs_settings:
            gs_socket.num_split = gs_settings['num_split']
        if 'n_iniframe' in gs_settings:
            gs_socket.n_iniframe = gs_settings['n_iniframe']
 
        if not os.path.isfile(gs_setting_file):
            print("GS solver setting not recorded. Saving.")
            gs_socket.write_settings(gs_setting_file)
              
        gss[compstring]={}
        print("Solving for composition:",compstring)
        gs_socket.solve()
        #if gs_socket.solved:
        gss[compstring]['gs_structure']=gs_socket.str_upper.as_dict()
        gss[compstring]['gs_energy']=gs_socket.e_upper
        gss[compstring]['solved']=gs_socket.solved

    if os.path.isfile(gs_file):
        with open(gs_file) as gs_in:
            gss_old = json.load(gs_in)
        _gs_converged = True
        for compstring in gss:
            if 'gs_energy' not in gss_old[compstring] or 'gs_energy' not in gss[compstring] or \
                    abs(gss_old[compstring]['gs_energy']-gss[compstring]['gs_energy'])>0.001:
                _gs_converged = False
                break
    else:
         _gs_converged = False

    with open(gs_file,'w') as gs_out:
        json.dump(gss,gs_out)

    print("Solved GS structures on {} hull points.".format(len(gss)))

    return _gs_converged

def writegss_to_vasprun(gs_file='gs.mson',vasprun='vasp_run',vs_file='vasp_settings.mson'):
    sm = StructureMatcher(ltol=0.2, stol=0.15, angle_tol=5, comparator=ElementComparator())
    
    calculated_structures = {}
    targetdirs = {}
    maxids = {}
    _was_generated = lambda x: 'POSCAR' in x and not 'KPOINTS' in x and not 'INCAR' in x and not 'POTCAR' in x
    if os.path.isdir(vasprun):
        print("Checking previously enumerated structures.")
        for root,dirs,files in os.walk(vasprun):
            if _was_generated(files):
                parentdir = os.path.join(*root.split(os.sep)[0:-1])
                with open(os.join(parentdir,'composition_by_site')) as comp_file:
                    composition_old = json.load(comp_file)
                    compstring_old = json.dumps(composition)
                if compstring_old not in calculated_structures:
                    calculated_structures[compstring_old]=[]
                calculated_structures[compstring_old].append(Poscar.from_file(os.join(root,'POSCAR').structure)) 
                if compstring_old not in targetdirs:
                    targetdirs[compstring_old]=parentdir
                if compstring_old not in maxids:
                    maxids[compstring_old]=max([int(idx) for idx in os.listdir(parentdir) if RepresentsInt(idx)])
    else: 
        print("Previous calculation or generator setting file missing. Exiting")
        return
    
    with open(gs_file) as gs_in:
        gss = json.load(gs_in)
    
    if os.path.isfile(vs_file):
        with open(vs_file) as vs_in:
            vasp_settings = json.load(vs_in)
    else:
        vasp_settings = None

    for compstring in gss:
        _unique = True
        if compstring in calculated_structures:
            gstruct = Structure.from_dict(gss[compstring]['gs_structure'])
            for ostruct in calculated_structures[compstring]:
                if sm.fit(ostruct,gstruct):
                    print("GS for composition:\n{}\nalready calculated. Skipping.".format(compstring))
                    _unique = False
                    break
        if _unique:
            targetdir = targetdirs[compstring]
            writedir = os.path.join(targetdir,str(maxids[compstring]+1))
            if not os.path.isdir(writedir): os.mkdir(writedir)
            Poscar(gstruct.get_sorted_structure()).write_file(os.path.join(writedir,'POSCAR'))
            vaspdir = os.path.join(writedir,'fm.0')
            if vasp_settings:
                print("Applying VASP settings",vasp_settings)
                if 'functional' in vasp_settings:
                    functional = vasp_settings['functional']
                else:
                    functional = 'PBE'
                if 'num_kpoints' in vasp_settings:
                    num_kpoints = vasp_settings['num_kpoints']
                else:
                    num_kpoints = 25
                if 'additional_vasp_settings' in vasp_settings:
                    additional = vasp_settings['additional_vasp_settings']
                else:
                    additional = None
                if 'strain' in vasp_settings:
                    strain = vasp_settings['strain']
                else:
                    strain = ((1.01,0,0),(0,1.05,0),(0,0,1.03))
            
                write_vasp_inputs(gstruct,vaspdir,functional,num_kpoints,additional,strain)
            else:
                print("Using CEAuto default VASP settings.")
                write_vasp_inputs(gstruct,vaspdir)

            print('New GS written to {}.'.format(writedir))

    
    
