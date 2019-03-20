from pyabinitio.cluster_expansion.ce import ClusterExpansion,Cluster,SymmetrizedCluster
from monty.json import MSONable
import json
import numpy as np
import math
import random
from operator import mul
from functools import reduce
import random
from itertools import combinations,permutations
from pymatgen.analysis.ewald import EwaldSummation
from pymatgen import Structure,PeriodicSite

__author__ = "Fengyu Xie"
__version__ = "2019.0.0"

MAXSAT_PATH = './solvers/'
MAXSAT_CUTOFF = 600
COMPLETE_MAXSAT = ['akmaxsat','ccls_akmaxsat']
INCOMPLETE_MAXSAT = ['CCLS2015']
"In this file we will provide socket functions to ground state solver I/O."


#####
# Tool functions
#####
def _factors(n):
    """
    This function take in an integer n and computes all integer multiplicative factors of n

    """
    return set(reduce(list.__add__,
                      ([i, n // i] for i in range(1, int(n ** 0.5) + 1) if n % i == 0)))

def _enumerate_mat(Num):
    Mats = []; Factors = list(_factors(Num)); Factors *= 3;
    for Perm in set(permutations(Factors, 3)):
        if reduce(mul, Perm) == Num:
            Mat = np.array([[Perm[0], 0, 0], [0, Perm[1], 0], [0, 0, Perm[2]]])
            Perms2 = set(permutations(np.tile(np.arange(Perm[2]), 2), 2))
            Num_list = np.arange(Perm[1]);
            for Num2 in Num_list:
                for Perm2 in Perms2:
                    Mat[0, 1] = Num2; Mat[0:2, 2] = Perm2; LMat = Mat.tolist();
                    if LMat not in Mats: Mats.append(LMat)
    return Mats

def _matmul(mat1,mat2):
    A = np.matrix(mat1)
    B = np.matrix(mat2)
    return (A*B).tolist()

def get_bits(structure):
    """
    Helper method to compute list of species on each site.
    Includes vacancies
    """
    all_bits = []
    for site in structure:
        bits = []
        for sp in sorted(site.species_and_occu.keys()):
            bits.append(str(sp))
        if site.species_and_occu.num_atoms < 0.99:
            bits.append("Vacancy")
       #bits.append("Vacancy")
        all_bits.append(bits)
    return all_bits

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

def _make_up_twobodies(symops,clusters,eci,cs):
    """
        In this part we will find all pair clusters with in a cluster supercell, and make up those not
        covered in previous ce.clusters[2]. Returned results formated in dictionaries.
    """
    exp_sites = [site for site in cs.supercell if site.species_and_occu.num_atoms < 0.99 or len(site.species_and_occu) > 1]
    exp_str = Structure.from_sites(exp_sites)
    bits = get_bits(exp_str)
    nbits = np.array([len(b) - 1 for b in bits])
    clusters_new = clusters
    eci_new = {size:[eci[(sc.sc_b_id-1):(sc.sc_b_id-1+len(sc.bit_combos))] for sc in clusters[size]] for size in clusters}
    #last_pair_id = clusters[3][0].sc_b_id -1
    for pair in combinations(list(range(len(exp_str))),2):
        #print("checking pair",pair)
        #print([exp_str[site] for site in pair])
        pair_c = Cluster([exp_str[site].frac_coords for site in pair],exp_str.lattice)
        pair_sc = SymmetrizedCluster(pair_c,[np.arange(nbits[i]) for i in pair],symops)
        if pair_sc not in clusters_new[2]:
            print("Adding sym-cluster:",pair_sc)
            clusters_new[2].append(pair_sc)
            eci_new[2].append([0]*len(pair_sc.bit_combos))
    if cs.use_inv_r:
        eci_new['ew']=eci[-len(cs.partial_ems):]
    else:
        eci_new['ew']=eci[-1]
    print('Added %d pair clusters.'%(len(clusters[2])-len(clusters_new[2])))
    return clusters_new,eci_new

#####
#Class functions
#####

class GSsemigrand(MSONable):
    """
    Defines a ground state problem for a generalized ising system. Uses 0/1 formalism.
    """

    def __init__(self, ce, eci,maxsupercell=100, num_of_sizes=4, selec=20 , ubsolver='ccls_akmaxsat', miu_bars=None):
        """
        Args:
            ce: a cluster expansion object that you wish to solve the GS of.
            eci: eci's of clusters in ce
            maxsupercell: largest cutoff supercell determinant.
            num_of_sizes: number of supercell sizes to be enumerated.
            selec: number of supercells to select after enumeration
            ubsolver: the MAXSAT solver used to solve the upper bound problem. Default: ccls_akmaxsat.
            miu_bars: this solver socket works under semi grand canonical ensemble. miu_bar is the relative chemical potential of species recorded in a                      dict. Notice: currently we don't convert into compositional axis representation here. Please do it somewhere else! The default 
                      references for miu_bar are the last species on each site, so miu_bars here is supposed to have the same shape and size as ce.nbit                      s. By default, all relative chemical potentials are set to zeroes.
        """
        self.ce = ce
        self.eci = eci
        self._bclus_corrected = None
        self._ecis_corrected = None
        self.maxsupercell = maxsupercell
        self.num_of_sizes = num_of_sizes
        self.selec = selec
        self.miu_bars=miu_bars
        self.ubsolver = ubsolver
        self.e_lower = None
        self.str_upper = None
        self.e_upper = None
        self.solved = False  #call function 'solve' to give a value to this
        self.transmat = None
        self.use_ewald = self.ce.use_ewald
        self.use_inv_r = self.ce.use_inv_r
        self._enumlist = None
    
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
            for size in range(int(self.maxsupercell/self.num_of_sizes),self.maxsupercell+1,\
                             int(self.maxsupercell/self.num_of_sizes)):
                print("Enumerating for size %d"%size)
                _enumlist.extend(_enumerate_mat(size))
            print("Randomly picking supercell matrices.")
            self._enumlist=random.sample(_enumlist,self.selec)
            if self.transmat: 
                self._enumlist=[_matmul(sc,self.transmat) for sc in self._enumlist]
            self._enumlist=sorted(self._enumlist,key=lambda a:(abs(np.linalg.det(a)),\
                                 np.linalg.norm(a[0]),np.linalg.norm(a[1]),np.linalg.norm(a[2])))
        print("Enumerated supercells generated!")
        return self._enumlist
    
    @property
    def bclus_corrected(self):
        if (not self._bclus_corrected) and (not self._ecis_corrected):
            self._bclus_corrected = []
            self._ecis_corrected = []
            if self.use_ewald:
                print("Ewald correction required.")
                for mat in self.enumlist:
                    print("Making up all pair interactions.")
                    clus_sup = self.ce.supercell_from_matrix(mat)
                    clusters_new, eci_new = _make_up_twobodies(self.ce.symops,self.ce.clusters,self.eci,clus_sup)
                    ce_new = ClusterExpansion(structure=self.ce.structure, expansion_structure=self.ce.expansion_structure,\
                                              symops=self.ce.symops, clusters= clusters_new,\
                                              ltol=self.ce.ltol, stol=self.ce.stol, angle_tol=self.ce.angle_tol,\
                                              supercell_size=self.ce.supercell_size,\
                                              use_ewald=self.use_ewald, use_inv_r=self.use_inv_r, eta=self.ce.eta)
                    clusters_new = ce_new.clusters
                    clus_sup_new = ce_new.supercell_from_matrix(clus_sup.supercell_matrix)
                    point_clus_indices = clus_sup_new.cluster_indices[clusters_new[1][0].sc_id:clusters_new[2][0].sc_id]
                    pair_clus_indices = clus_sup_new.cluster_indices[clusters_new[2][0].sc_id:clusters_new[3][0].sc_id]
                    ew_str = Structure.from_sites([PeriodicSite('H+',s.frac_coords,s.lattice) for s in clus_sup_new.supercell])
                    H = EwaldSummation(ew_str,eta=self.ce.eta).total_energy_matrix
                    #Ewald energy E_ew = q*H*q'.
                    supbits = get_bits(clus_sup_new.supercell)
                    r = np.array([_GetIonChg(bits[-1]) for bits in supbits])
                    chg_bits = [[_GetIonChg(bit)-_GetIonChg(bits[-1]) for bit in bits[:-1]] for bits in supbits]
                    b_clusters = []
                    eci_return = []

                    if not self.use_inv_r:
                        eci_ew = eci_new['ew']
                        H_r = H*r
                        #Here we do electro-static correction. Reference zero energy state is the one that all sites are occupied by reference compound. Relative chemical potential of species will also be integrated here.
                        for sc,sc_inds in clus_sup_new.cluster_indices:
                            for i,combo in enumerate(sc.bit_combos):
                                b_clusters.extend([[bit_inds[site][combo[s]] for s,site in enumerate(sc_ind)] for sc_ind in sc_inds])
                                if len(sc.bits)==1:     
                                    eci_return.extend([eci_new[1][sc.sc_id-clusters_new[1][0].sc_id][i]+\
                                                      eci_ew*(chg_bits[sc_ind[0]][combo[0]]*H_r[sc_ind[0]]+\
                                                              chg_bits[sc_ind[0]][combo[0]]**2*H[sc_ind[0]][sc_ind[0]]*2)+\
                                                      (self.miu_bars[sc_ind[0]][combo[0]] if self.miu_bars else 0)\
                                                      for sc_ind in sc_inds]) 
                                    #miu_bars is how we introduce chem potential.
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

                    
                    self._bclus_corrected.append(b_clusters)
                    self._ecis_corrected.append(eci_return)

            else:
                for mat in self.enumlist:
                    ce_new = self.ce
                    clusters_new = self.ce.clusters
                    eci_new = {size:[eci[(sc.sc_b_id-1):(sc.sc_b_id-1+len(sc.bit_combos))] for sc in clusters_new[size]] \
                           for size in clusters_new}
                    clus_sup_new = self.ce.supercell_from_matrix(mat)
                    for sc,sc_inds in clus_sup_new.cluster_indices:
                        for i,combo in enumerate(sc.bit_combos):
                            b_clusters.extend([[bit_inds[site][combo[s]] for s,site in enumerate(sc_ind)] for sc_ind in sc_inds])
                            if len(sc.bits)==1:     
                                eci_return.extend([eci_new[1][sc.sc_id-clusters_new[1][0].sc_id][i]+\
                                                  (self.miu_bars[sc_ind[0]][combo[0]] if self.miu_bars else 0)\
                                                  for sc_ind in sc_inds])
                            else:
                                eci_return.extend([eci_new[len(sc.bits)][sc.sc_id-clusters_new[len(sc.bits)][0].sc_id][i]\
                                                  for sc_ind in sc_inds])

                    self._bclus_corrected.append(b_clusters)
                    self._ecis_corrected.append(eci_return)

        return self._bclus_corrected

    @property
    def ecis_corrected(self):
        if (not self._ecis_corrected) and (not self._bclus_corrected):
            bclus_corrected = self.bclus_corrected
        return self._ecis_corrected

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
        #Here we find MAXSAT variable indices for all specie on sites.
        bit_inds = []
        b_id = 1
        clus_sup = self.ce.supercell_from_matrix(self.enumlist[mat_id])
        for i,site in enumerate(clus_sup.supercell):
            site_bit_inds = []
            for specie_id in range(len(site.species_and_occu)-1):#-1 since a specie on the site is taken as reference
                site_bit_inds.append(b_id)
                b_id+=1
            bit_inds.append(site_bit_inds)
        print('%d variables will be used in MAXSAT.'%(b_id-1))

        #use a new cluster expansion object to store all 2-body interations covered
        #in the supercell
        
        return self.bclus_corrected[mat_id],self.ecis_corrected[mat_id],bit_inds
            
    def _solve_upper(self,mat_id,hard_marker=10000):
        """
            Warning: hard_marker should be chosen as a big enough number!
        """
        #### Input Preparation ####
        cs = self.ce.supercell_from_matrix(self.enumlist(mat_id))
        cs_bits = get_bits(cs.supercell)
        b_clusters_new,ecis_new,site_specie_ids=self._electrostatic_correction(mat_id)
        soft_cls = []
        hard_cls = []
        for site_id in range(len(cs.supercell)):
            hard_cls.extend([[int(10000)]+[int(-1*id_1),int(-1*id_2)] for id_1,id_2 in combinations(site_specie_ids[site_id],2)])
        #Hard clauses to enforce sum(specie_occu)=1
        for b_cluster,eci in zip(b_clusters_new,ecis_new):
            clause = [eci]
            for b_id in b_cluster:
                clause.append(int(-1*b_id))
            #Don't worry about the last specie for a site. It is take as a referecne specie, thus not counted into nbits and combos at all!!!
            soft_cls.append(clause)
        all_cls = hard_cls+soft_cls
        #_modify(all_cls,site_specie_ids)
        num_of_vars = sum([len(line) for line in site_specie_ids])
        num_of_cls = len(all_cls)
        maxsat_input = 'c\nc Weighted paritial maxsat\nc\np wcnf %d %d %d'%(num_of_vars,num_of_cls,hard_marker)
        for clause in all_cls:
            maxsat_input+=([str(lit) for lit in clause].join(' ')+' 0\n')
        f_maxsat = open('maxsat.wcnf','w')
        f_maxsat.write(maxsat_input)
        f_maxsat.close()
        #### Calling MAXSAT ####
        rand_seed = random.randint(1,100000)
        print('Callsing MAXSAT solver. Using random seed %d.'%rand_seed)
        MAXSAT_CMD = MAXSAT_PATH+self.ubsolver+' ./maxsat.wcnf'
        if self.ubsolver in INCOMPLETE_MAXSAT:
            MAXSAT_CMD += ' %d %d'%(rand_seed,MAXSAT_CUTOFF)
        os.sys(MAXSAT_CMD)
        print('MAXSAT solution found!')
        #### Output Processing ####
        with open('./maxsat.out') as f_res:
            lines = f_res.readlines()
            for line in lines:
                if line[0]=='v':
                    maxsat_res = [int(num) for num in line.split()[1:]]
        upper_sites = []
        for s,site in enumerate(site_specie_ids):
            should_be_ref = True
            for s_id,specie_id in enumerate(site):
            #For all variables on a site, only one could be true. If no hard cluases fail.
                if maxsat_res[specie_id-1]>0:
                    st = cs.supercell[s]
                    upper_sites.append([PeriodicSite(cs_bits[s][s_id],st.frac_coord,st.lattice)])
                    should_be_ref = False
                    break
            if should_be_ref:
                upper_sites.append([PeriodicSite(cs_bits[s][-1],st.frac_coord,st.lattice)])
        upper_str = Structure.from_sites(upper_sites)
        upper_corr = cs.corr_from_structure(upper_str)
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
        gs_socket= cls(ce,d['ecis']) 
        #essential terms for initialization
        if 'maxsupercell' in d:
            gs_socket.maxsupercell=d['maxsupercell']
        if 'num_of_sizes' in d:
            gs_socket.num_of_sizes=d['num_of_sizes']
        if 'selec' in d:
            gs_socket.selec = d['selec']
        if 'ubsolver' in d:
            gs_socket.ubsolver = d['ubsolver']
        if 'miu_bars' in d:
            gs_socket.miu_bars = d['miu_bars']
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
        if ('e_lower' in d) and ('e_upper' in d) and np.abs(d['e_lower']-d['e_upper'])<=0.001 and ('str_upper' in d):
            gs_socket.solved=True
            print("Loaded cluster expansion already has a solved GS. GS energy: %f"%(gs_socket.e_upper))
            print("Under external chemical potential:",gs_socket.miu_bars)
        else:
            print("Loaded cluster expansion needs GS solution. Use self.solve() method to generate one.")
        return gs_socket
    
    def as_dict(self):
        return {'cluster_expansion':self.ce.as_dict(),\
                'ecis':self.eci,\
                'maxsupercell':self.maxsupercell,\
                'num_of_sizes':self.num_if_sizes,\
                'selec':self.selec,\
                'ubsolver':self.ubsolver,\
                'miu_bars':self.miu_bars,\
                'e_lower':self.e_lower,\
                'e_upper':self.e_upper,\
                'str_upper':self.str_upper.as_dict(),\
                'transmat':self.transmat,\
                '@module':self.__class__.__module__,\
                '@class':self.__class__.__name__\
               }
