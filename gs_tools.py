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

__author__ = "Fengyu Xie"
__version__ = "2019.0.0"

SITE_TOL = 1e-6
MAXSAT_PATH = './solvers/'
MAXSAT_CUTOFF = 600
COMPLETE_MAXSAT = ['akmaxsat','ccls_akmaxsat']
INCOMPLETE_MAXSAT = ['CCLS2015']
""" 
In this file we will provide socket functions to ground state solver I/O. 
For number of hard clauses issue, we firstly try to reduce the problem size
down to 16 sites.
"""

#####
# Tool functions
#####
def _map_symops(exp_str,symops):
    """
        Generate a mapping to indices of symmetry manipulations.
    """
    symops_mapping = []
    exp_sites = [site.frac_coords for site in exp_str]
    for symop in symops:
        new_sites = [site for site in symop.operate_multi(exp_sites)]
        #print("New:",new_sites)
        #print("Old:",exp_sites)
        symops_mapping.append(coord_list_mapping_pbc(new_sites,exp_sites,atol=SITE_TOL))
    return symops_mapping

def _modify_symops(symops_old,symops_sup,supmat):
    """
        Here we combine all rotations and spires from symops_old with all translations from symops_sup to give symops_new.
    """
    symops_prim = []
    for symop in symops_old:
       #R'=S^-1*R*S, t'=S^-1*t
       rot = (supmat.T).I * np.matrix(symop.rotation_matrix)* (supmat.T)
       t = (supmat.T).I* np.matrix(symop.translation_vector).T
       rot = rot.tolist()
       t = np.squeeze(t.T.tolist())
       symop_prim = SymmOp.from_rotation_and_translation(rot,t.tolist())
       symops_prim.append(symop_prim)
    symops_trans = [symop for symop in symops_sup if np.array_equal(symop.rotation_matrix,np.identity(3))]
    symops_new = []
    for symop_prim in symops_prim:
        for symop_trans in symops_trans:
            symop_new = symop_prim*symop_trans
            symops_new.append(symop_new)
    print("Number of symmetry operations in the supercell:",len(symops_new))
    return symops_new

def _make_up_twobodies(ce_old,eci_old,clus_sup):
    """
        In this part we will find all pair clusters with in a cluster supercell, and make up those not
        covered in previous ce.clusters[2]. Returned results formated in dictionaries.
    """
    
    symops_sup = SpacegroupAnalyzer(clus_sup.supercell).get_symmetry_operations()
    #The command above does not give our desired symmetry correctly, and a lot of degeneracies will be broken!
    #I will try to directly add all the displacement vectors to the old symops!
    symops_old = ce_old.symops
    symops_new = _modify_symops(symops_old,symops_sup,np.matrix(clus_sup.supercell_matrix))
    exp_sites = [site for site in clus_sup.supercell\
                if site.species_and_occu.num_atoms < 0.99 or len(site.species_and_occu) > 1]
    exp_str = Structure.from_sites(exp_sites)
    #print("exp_str",exp_str)
    bits = get_bits(exp_str)
    nbits = np.array([len(b) - 1 for b in bits])

    #ce_test = ClusterExpansion.from_radii(exp_str,{2:7.0})
    #print("test cluster expansion",ce_test.clusters[2],len(ce_test.clusters[2]))
    #print("initial cluster expansion",ce_old.clusters[2],"num of pair:",len(ce_old.clusters[2]))

    # Reinitiate. Don't need to modify to new symmetry
    clusters_new = {}
    for size in ce_old.clusters:
        clusters_new[size]=[]
        for sc in ce_old.clusters[size]:
            new_sc = SymmetrizedCluster(sc.base_cluster,sc.bits,ce_old.symops)
            clusters_new[size].append(new_sc) 
    #print(clusters_new[2])

    #eci_old converted into a dictionary that has the same shape as ce.clusters
    eci_new = {size:[eci_old[(sc.sc_b_id-1):(sc.sc_b_id-1+len(sc.bit_combos))]\
               for sc in ce_old.clusters[size]] for size in ce_old.clusters}
    
    non_relavant_pairs = []
    #print("Establishing all non relavant pairs.")
    pair_record = []
    print("Mapping symmetry operations.")
    symops_mapping = _map_symops(exp_str,symops_new)
    #print("Mapped %d symmetry operations"%len(symops_mapping),symops_mapping)
    for i in range(len(exp_str)):
        pair_record.append([])
        for j in range(i+1,len(exp_str)):
            pair_record[-1].append(j)
    
    print("Finding all geometrically non-equivalent pairs.")
    accepted_pairs = []
    for i,i_j in enumerate(pair_record):
        while i_j:
            j=i_j[0]
            for symop in symops_mapping:
                if not(i==symop[i] and j==symop[j]) and not(i==symop[j] and j==symop[i]):
                #Only remove other pairs that are identical.
                    if symop[i] in pair_record[symop[j]]:
                        pair_record[symop[j]].remove(symop[i])
                    if symop[j] in pair_record[symop[i]]:
                        pair_record[symop[i]].remove(symop[j])
            accepted_pairs.append((i,j))
            print("Accepted pair:",(i,j))
            # print("Pair record:", pair_record)
            del(i_j[0])

    max_pair_old = max([pair.base_cluster.max_radius for pair in clusters_new[2]])
    for pair in accepted_pairs:
        #print("checking pair",pair)
        #print([exp_str[site] for site in pair])
        #Need to transform clusters in supercells back before adding them into cluster expansion!
        #We actually don't need too forcibly add internal translations to supercells, since they are included in
        #ClusterSupercell.coord_list_mapping_pbc!
        pair_sup_coord = [exp_str[site].frac_coords for site in pair]
        pair_prim_coord = np.dot(pair_sup_coord,clus_sup.supercell_matrix)

        pair_c = Cluster([site for site in pair_prim_coord],exp_str.lattice)
        pair_sc = SymmetrizedCluster(pair_c,[np.arange(nbits[i]) for i in pair],ce_old.symops)

        if pair_c.max_radius > max_pair_old:
            """
                 Just add pair without double check. All are symetrically inequivalent since we have done dedup before.
            """
            print("Adding sym-cluster:",pair,"Radius:",pair_c.max_radius)
            clusters_new[2].append(pair_sc)
            #print("pair_c:",pair_c)
            #print('pair equiv:',pair_sc._equiv,'number:',len(pair_sc._equiv))
            #print('pair multiplicity:',pair_sc.multiplicity)
            #print('Num of self symmetry:',len(pair_sc.cluster_symops))
            eci_new[2].append([0]*len(pair_sc.bit_combos))

    if ce_old.use_inv_r:
        eci_new['ew']=eci_old[-len(clus_sup.partial_ems):]
    else:
        eci_new['ew']=eci_old[-1]
    print('Added %d pair clusters.'%(len(clusters_new[2])-len(ce_old.clusters[2])))

    ce_new = ClusterExpansion(structure=ce_old.structure, expansion_structure=ce_old.expansion_structure,\
                              symops=ce_old.symops, clusters= clusters_new,\
                              ltol=ce_old.ltol, stol=ce_old.stol, angle_tol=ce_old.angle_tol,\
                              supercell_size=ce_old.supercell_size,\
                              use_ewald=False, use_inv_r=False, eta=None)
    #Here we disable ewald in ce_new in order to make our life easier in later energy summations.
    #print('eci_new',eci_new,len(eci_new))
    #print('ce_new',ce_new.clusters)
    return ce_new,eci_new

#####
#Class functions
#####

class GScanonical(MSONable):
    """
    Defines a ground state problem for a generalized ising system. Uses 0/1(cluster counting) formalism.
    """

    def __init__(self, ce, eci, composition, maxsupercell=16, num_of_sizes=4, selec=20, ubsolver='ccls_akmaxsat'):
        """
        Args:
            ce: a cluster expansion object that you wish to solve the GS of.
            eci: eci's of clusters in ce
            maxsupercell: maximum supercell sizes
            num_of_sizes: number of supercell sizes to be enumerated.
            selec: number of supercell matrix to selec.
            ubsolver: the MAXSAT solver used to solve the upper bound problem. Default: ccls_akmaxsat.
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
        self._ces_new = None
        self.maxsupercell = maxsupercell
        self.num_of_sizes = num_of_sizes
        self.selec = selec
        
        sp_list = []
        for site_occu in composition:
            sp_list.extend(site_occu.values())
        self._comp_step = GCD_List(sp_list)
        # Find the enumeration step for composition, and this will be the minimum enumerated supercell size.
        self.composition=[{sp:site_occu[sp]//self._comp_step for sp in site_occu} for site_occu in composition]

        self.ubsolver = ubsolver
        self.e_lower = None
        self.str_upper = None
        self.e_upper = None
        self.solved = False  #call function 'solve' to give a value to this
        self.use_ewald = self.ce.use_ewald
        self.use_inv_r = self.ce.use_inv_r
        self._enumlist = None
        self.transmat = [[1,0,0],[0,1,0],[0,0,1]]
    
####
# Callable interface
####
    def set_transmat(self,transmat):
        self.transmat=transmat
        print("Using transformation matrix:",transmat,"after enumeration.")
       
    def solve(self):
        if not(self.solved):
            self.solved=self._iterate_supercells()
            if not(self.solved):
                print("Failed to find the GS after enumeration. You may want to increase enumeration scale.")
                return False              
        print("Found solution to GS problem! GS energy: %f."%self.e_lower)
        return True
    
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
            if len(enumrange)==0:
                print('No enumerated size acceptable. You should consider adjusting your compostion enumeration\
                       step in generator, or adjust maxsupercell to a proper value (Warning: a value >16 not \
                       recommended.)')
                return

            for size in enumrange:
                print("Enumerating for size %d"%size)
                _enumlist.extend(Get_Hermite_Matrices(int(size/scale)))

            print("Randomly picking supercell matrices.")
            self._enumlist=random.sample(_enumlist,self.selec)
            if self.transmat: 
                self._enumlist=[mat_mul(sc,self.transmat) for sc in self._enumlist]
            self._enumlist=sorted(self._enumlist,key=lambda a:(abs(np.linalg.det(a)),\
                                 np.linalg.norm(a[0]),np.linalg.norm(a[1]),np.linalg.norm(a[2])))
            print("Enumerated supercells generated!")
        return self._enumlist
    
    @property
    def bclus_corrected(self):
        """
            Returns a list of ewald corrected bit_clusters, each element for a matrix in self.enumlist.
        """

        if (not self._bclus_corrected) or (not self._ecis_corrected) or (not self._bit_inds) or (not self._ces_new):
            self._bclus_corrected = []
            self._ecis_corrected = []
            self._bit_inds = []
            self._ces_new = []
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

                #Ewald correction, as well as chemical potential integration.
                if self.use_ewald:
                    print("Making up all pair interactions for supercell:",mat)

                    #Reset symops of all Symmetrized clusters, reset ce.structure, make up two body clusters
                    ce_new, eci_new = _make_up_twobodies(self.ce,self.eci,clus_sup)
                    #print('ce_new',ce_new.clusters)
                    #print('eci_new',eci_new)

                    clusters_new = ce_new.clusters

                    #ce_new.structure == ce_old.structure!
                    clus_sup_new = ce_new.supercell_from_matrix(mat)

                    ew_str = Structure.from_sites([PeriodicSite('H+',s.frac_coords,s.lattice) for s in clus_sup_new.supercell])
                    H = EwaldSummation(ew_str,eta=self.ce.eta).total_energy_matrix

                    #Ewald energy E_ew = (q+r)*H*(q+r)'. I used a stupid way to get H but quite effective.
                    supbits = get_bits(clus_sup_new.supercell)
                    r = np.array([GetIonChg(bits[-1]) for bits in supbits])
                    chg_bits = [[GetIonChg(bit)-GetIonChg(bits[-1]) for bit in bits[:-1]] for bits in supbits]
                    b_clusters = []
                    eci_return = []

                    if not self.use_inv_r:
                        eci_ew = eci_new['ew']
                        H_r = np.dot(H,r)
                        #Here we do electro-static correction. Reference zero energy state is the one that all sites are occupied by reference compound. Relative chemical potential of species will also be integrated here.
                        for sc,sc_inds in clus_sup_new.cluster_indices:
                            for i,all_combo in enumerate(sc.bit_combos):
                                #An item in bit_combos store an orbit of equivalent bit combinations, not just one.
                                #len(sc_inds[item][1])=clustersupercell.size*sc.multiplicity,
                                #sc.multiplicity=len(sc.equivalent_clusters)
                                #sc.multiplicity*len(sc.cluster_symops)=len(ce.symops)
                                #Thus, all bit_combo in bit_combo[item] should be included, not just bit_combo[item][0]!
                                for combo in all_combo:
                                    b_clusters.extend([[bit_inds[sc_ind[s]][combo[s]] for s in range(len(sc_ind))]\
                                                  for sc_ind in sc_inds])
                                    if len(sc.bits)==1:     
                                        eci_return.extend([eci_new[1][sc.sc_id-clusters_new[1][0].sc_id][i]+\
                                                      eci_ew*(chg_bits[sc_ind[0]][combo[0]]*H_r[sc_ind[0]]+\
                                                              chg_bits[sc_ind[0]][combo[0]]**2*H[sc_ind[0]][sc_ind[0]]*2)\
                                                           for sc_ind in sc_inds]) 
                                    
                                    elif len(sc.bits)==2:
                                        eci_return.extend([eci_new[2][sc.sc_id-clusters_new[2][0].sc_id][i]+\
                                                      eci_ew*chg_bits[sc_ind[0]][combo[0]]*chg_bits[sc_ind[1]][combo[1]]*\
                                                      H[sc_ind[0]][sc_ind[1]] for sc_ind in sc_inds])
                                    else:
                                        eci_return.extend([eci_new[len(sc.bits)][sc.sc_id-clusters_new[len(sc.bits)][0].sc_id][i]\
                                                      for sc_ind in sc_inds])

                    else:
                        #When using inv_r, an independent ewald sum is generated for each specie, and they each are fitted into an ECI.
                        print("I'm still thinking about use_inv_r cases. Not available yet!")
                        raise NotImplementedError

                else:
                    ce_new = self.ce
                    clusters_new = self.ce.clusters
                    eci_new = {size:[eci[(sc.sc_b_id-1):(sc.sc_b_id-1+len(sc.bit_combos))] for sc in clusters_new[size]] \
                               for size in clusters_new}
                    clus_sup_new = self.ce.supercell_from_matrix(mat)
                    for sc,sc_inds in clus_sup_new.cluster_indices:
                        for i,all_combo in enumerate(sc.bit_combos):
                            for combo in all_combo:
                                b_clusters.extend([[bit_inds[site][combo[s]] for s,site in enumerate(sc_ind)]\
                                                  for sc_ind in sc_inds])
                                if len(sc.bits)==1:     
                                    eci_return.extend([eci_new[1][sc.sc_id-clusters_new[1][0].sc_id][i]\
                                                       for sc_ind in sc_inds])
                                else:
                                    eci_return.extend([eci_new[len(sc.bits)][sc.sc_id-clusters_new[len(sc.bits)][0].sc_id][i]\
                                                   for sc_ind in sc_inds])

                self._bclus_corrected.append(b_clusters)
                self._ecis_corrected.append(eci_return)
                self._ces_new.append(ce_new)

        return self._bclus_corrected

    @property
    def ecis_corrected(self):
        if (not self._ecis_corrected) or (not self._bclus_corrected) or (not self._bit_inds) or (not self._ces_new):
            bclus_corrected = self.bclus_corrected
        return self._ecis_corrected

    @property
    def bit_inds(self):
        if (not self._ecis_corrected) or (not self._bclus_corrected) or (not self._bit_inds) or (not self._ces_new):
            bclus_corrected = self.bclus_corrected
        return self._bit_inds

    @property
    def ces_new(self):
        if (not self._ecis_corrected) or (not self._bclus_corrected) or (not self._bit_inds) or (not self._ces_new):
            bclus_corrected = self.bclus_corrected
        return self._ces_new
####
# Private tools for the class
#### 
    def _iterate_supercells(self):
        for mat_id,mat in enumerate(self.enumlist):
            #Here we will convert problems into MAXSAT and LP standard inputs, solve and analyze the outputs
            print("Solving on supercell matrix:",mat)
            cur_e_upper,cur_str_upper=self._solve_upper(mat_id)
            cur_e_lower==self._solve_lower(mat_id)
            print("Current GS upper-bound: %f"%cur_e_upper)
            print("Current GS lower_bound: %f"%cur_e_lower)
            if abs(self.e_lower-self.e_upper)<abs(cur_e_lower-cur_e_upper):
                self.e_lower = cur_e_lower
                self.e_upper = cur_e_upper
                self.str_upper = cur_str_upper
            if abs(self.e_lower-self.e_upper)<0.001:
                return True
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
            Warning: still a piece of pseudocode!!!!!
        """
        
        return self.bclus_corrected[mat_id],self.ecis_corrected[mat_id],self.bit_inds[mat_id]
            
    def _solve_upper(self,mat_id,hard_marker=1000000000000000000,eci_mul=1000000000):
        """
            Warning: hard_marker is the weight assigened to hard clauses. Should be chosen as a big enough number!
        """
        #### Input Preparation ####
        b_clusters_new,ecis_new,site_specie_ids=self._electrostatic_correction(mat_id)
        N_sites = len(site_specie_ids)
        soft_cls = []
        hard_cls = []

        for site_id in range(N_sites):
            hard_cls.extend([[hard_marker]+[int(-1*id_1),int(-1*id_2)] for id_1,id_2 in combinations(site_specie_ids[site_id],2)])
        #Hard clauses to enforce sum(specie_occu)=1

        sublat_sp_ids = [[]]*len(self.ce.structure)
        sc_size = int(round(self.enumlist[mat_id]))
        
        site_id = 0
        while site_id<N_sites:
            for sp,sp_id in enumerate(site_specie_ids[site_id]):
                if len(sublat_ps_ids[site_id//sc_size])<len(site_species_ids[site_id]):
                    sublat_ps_ids[site_id//sc_size].append([])
                sublat_ps_ids[site_id//sc_size][sp].append(sp_id)
            site_id+=1

        comp_scale = int(sc_size/round(sum(self.composition[0].values())))
        scaled_composition = [{sp:sublat[sp]*comp_scale for sp in sublat} for sublat in self.composition]
        for sl_id,sublat in enumerate(sublat_ps_ids):
            for sp_id,specie_ids in enumerate(sublat):
                sp_name = self.ce.structure[sl_id].species_and_occu.key()[sp_id]
                hard_cls.extend([[hard_marker]+\
                                 [int(-1*b_id) for b_id in combo]+\
                                 [int(b_id) for b_id in specie_id if b_id not in combo] \
                                 for combo in combinations(specie_ids,scaled_composition[sl_id][sp_name])])
        #Hard clauses to enforce composition consistency,scales with O(2^N). Updated Apr 12 19
        
        all_eci_sum = 0
        for b_cluster,eci in zip(b_clusters_new,ecis_new):
            if int(eci*eci_mul)!=0: 
            #2016 Solver requires that all weights >=1 positive int!! when abs(eci)<1/eci_mul, cluster will be ignored!
                if eci>0:
                    clause = [int(eci*eci_mul)]
                    all_eci_sum += int(eci*eci_mul)  
            #MAXSAT 2016 series only takes in integer weights
                    for b_id in b_cluster:
                        clause.append(int(-1*b_id))
        #Don't worry about the last specie for a site. It is take as a referecne specie, 
        #thus not counted into nbits and combos at all!!!
                    soft_cls.append(clause)
                else:
                    clauses_to_add = []
                    for i in range(len(b_cluster)):
                        clause = [int(-1*eci*eci_mul),int(b_cluster[i])]
                        for j in range(i+1,len(b_cluster)):
                            clause.append(int(-1*b_cluster[j]))
                        clauses_to_add.append(clause)
                    soft_cls.extend(clauses_to_add)

        print('Summation of all ecis:',all_eci_sum)
        all_cls = hard_cls+soft_cls
        #print('all_cls',all_cls)
        
        num_of_vars = sum([len(line) for line in site_specie_ids])
        num_of_cls = len(all_cls)
        maxsat_input = 'c\nc Weighted paritial maxsat\nc\np wcnf %d %d %d\n'%(num_of_vars,num_of_cls,hard_marker)
        for clause in all_cls:
            maxsat_input+=(' '.join([str(lit) for lit in clause])+' 0\n')
        f_maxsat = open('maxsat.wcnf','w')
        f_maxsat.write(maxsat_input)
        f_maxsat.close()
        #### Calling MAXSAT ####
        rand_seed = random.randint(1,100000)
        print('Callsing MAXSAT solver. Using random seed %d.'%rand_seed)
        os.system('cp ./maxsat.wcnf '+MAXSAT_PATH)
        os.chdir(MAXSAT_PATH)
        MAXSAT_CMD = './'+self.ubsolver+' ./maxsat.wcnf'
        if self.ubsolver in INCOMPLETE_MAXSAT:
            MAXSAT_CMD += ' %d %d'%(rand_seed,MAXSAT_CUTOFF)
        MAXSAT_CMD += '> maxsat.out'
        print(MAXSAT_CMD)
        os.system(MAXSAT_CMD)
        os.chdir('..')
        os.system('cp '+MAXSAT_PATH+'maxsat.out'+' ./maxsat.out')
        print('MAXSAT solution found!')

        #### Output Processing ####
        maxsat_res = []
        with open('./maxsat.out') as f_res:
            lines = f_res.readlines()
            for line in lines:
                if line[0]=='v':
                    maxsat_res = [int(num) for num in line.split()[1:]]
        sorted(maxsat_res,key=lambda x:abs(x))

        ce_new = self.ces_new[mat_id]
        cs = ce_new.supercell_from_matrix(self.enumlist[mat_id])
        cs_bits = get_bits(cs.supercell)
        upper_sites = []
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
        upper_cs = self.ce.supercell_from_structure(upper_str)
        upper_corr = upper_cs.corr_from_structure(upper_str)
        upper_e = np.dot(np.array(self.eci),upper_corr)
        return upper_e,upper_str


    def _solve_lower(self,mat_id):
        #### Input Preparation ####

        #### Calling Gurobi ####

        #### Output Processing ####
        raise NotImplementedError


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
        if 'num_of_sizes' in d:
            gs_socket.num_of_sizes=d['num_of_sizes']
        if 'selec' in d:
            gs_socket.selec = d['selec']
        if 'ubsolver' in d:
            gs_socket.ubsolver = d['ubsolver']
        if 'e_lower' in d:
            gs_socket.e_lower=d['e_lower']        
        if 'lastsupermat' in d:
            gs_socket.lastsupermat=d['lastsupermat']
        if 'e_upper' in d:
            gs_socket.e_upper=d['e_upper']
        #Only lower-bound solver gives a bit ordering
        if 'str_upper' in d:
            gs_socket.str_upper=d['str_upper']
        if 'transmat' in d:
            gs_socket.transmat=d['transmat']
        if 'bclus_corrected' in d and 'ecis_corrected' in d and 'bit_inds' in d and 'ces_new' in d:
            gs_socket._bclus_corrected = d['bclus_corrected']
            gs_socket._ecis_corrected = d['ecis_corrected']
            gs_socket._bit_inds = d['bit_inds']
            gs_socket._ces_new = d['ces_new']
        if ('e_lower' in d) and ('e_upper' in d) and np.abs(d['e_lower']-d['e_upper'])<=0.001 and ('str_upper' in d):
            gs_socket.solved=True
            print("Loaded instance already has a solved GS. GS energy: %f"%(gs_socket.e_upper))
            print("Under composition:",gs_socket.composition)
        else:
            print("Loaded cluster expansion needs GS solution. Use self.solve() method to generate one.")
        return gs_socket
    
    def as_dict(self):
        if not self.solved:
            print("Your ground state has not been solved yet, solving for you.")
            self.solved = self.solve()
        if not self.solved:
            print("We have tried our best but we only got these. You may want to tune your calculation parameters.")
        return {'cluster_expansion':self.ce.as_dict(),\
                'ecis':self.eci,\
                'composition':self.composition,\
                'maxsupercell':self.maxsupercell,\
                'num_of_sizes':self.num_if_sizes,\
                'selec':self.selec,\
                'ubsolver':self.ubsolver,\
                'e_lower':self.e_lower,\
                'e_upper':self.e_upper,\
                'str_upper':self.str_upper.as_dict(),\
                'transmat':self.transmat,\
                'bclus_corrected':self.bclus_corrected,\
                'ecis_corrected':self.ecis_corrected,\
                'bit_inds':self.bit_inds,\
                'ces_new':self.ces_new,\
                '@module':self.__class__.__module__,\
                '@class':self.__class__.__name__\
               }

####
# Canonical to Semi-grand
####

def _solvegs_for_hull(ce_file='ce.mson',calc_data_file='calcdata.mson',gen_settings='generator_settings.mson',gs_file = 'gs.mson',\
                      gs_settings={}):
    """
    Here we generate all the canonical ground states for the hull in calc_data_file, and store them in gs_file.
    Output:
        A boolean variable. Indicating whether the GS has converged for hull. Currently only judging from energy.
    """
    if not(os.path.isfile(calc_data_file)):
        print("No valid calulations detected, can't solve GS!")
        return

    else:
        with open(calc_data_file) as calc_data:
            calcdata = json.load(calc_data)
        with open(ce_file) as ce_data:
            cedata = json.load(ce_data)

        ecis = cedata['ecis']
        ce = ClusterExpansion.from_dict(cedata['cluster_expansion'])
        gss = {}

        for compstring in calcdata:
            composition=json.loads(compstring)
            gs_socket = GScanonical(ce,ecis,composition)
            if 'maxsupercell' in gs_settings:
                gs_socket.maxsupercell=gs_settings['maxsupercell']
            if 'num_of_sizes' in gs_settings:
                gs_socket.num_of_sizes=gs_settings['num_of_sizes']
            if 'selec' in gs_settings:
                gs_socket.selec=gs_settings['selec']
            if 'ubsolver' in gs_settings:
                gs_socket.ubsolver=gs_settings['ubsolver']
              
            gss[compstring]={}
            if gs_socket.solve():
                gss[compstring]['gs_structure']=gs_socket.str_upper.as_dict()
                gss[compstring]['gs_energy']=gs_socket.e_upper

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

        print("Solved GS structures on {} hull points.").format(len(gss))

        return _gs_converged

def _writegss_to_vasprun(gs_file='gs.mson',vasprun='vasp_run',vs_file='vasp_settings.mson'):
    sm = StructureMatcher(ltol=0.3, stol=0.3, angle_tol=5, comparator=ElementComparator())
    
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
                    targetdirs[compstring_old]=root.split(os.sep)[-1]
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
            targetdir = os.path.join(vasprun,targetdirs[compstring])
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
            
                _write_vasp_inputs(gstruct,vaspdir,functional,num_kpoints,additional,strain)
            else:
                print("Using CEAuto default VASP settings.")
                _write_vasp_inputs(gstruct,vaspdir)

            print('New GS written to {}.'.format(writedir))

    
    
