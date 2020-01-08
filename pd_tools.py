from __future__ import division

from collections import defaultdict
import numpy as np
import random

from pymatgen import Structure
from pymatgen.core.sites import PeriodicSite
from pymatgen.transformations.standard_transformations import OrderDisorderedStructureTransformation

from copy import deepcopy
import os
from sklearn.linear_model import LinearRegression

from scipy import interpolate

import matplotlib.pyplot as plt
import sys
cwd=os.path.dirname(os.path.realpath(__file__))
sys.path.append(cwd)

from cluster_expansion.ce import ClusterExpansion

"""
12-2-19 First version. Remember to update CEAuto generator after this, add sublat_merge_rule option to enable sublattice merging.
For the theoretical basis of this work, read:
A van de Walle & M Asta 2002 Modelling Simul Mater Sci Eng 10 521
"""
__author__ = 'Fengyu Xie'

def merge_unions(regions):
    """
        Give a set of regions, returns the union of them.
        ie: [(2,4),(9,13),(6,12)] -> [(2,4),(6,13)]
    """
    if len(regions)<=1:
        return regions

    regions = sorted(regions)
    union_regions = [regions[0]]
    
    for lb,ub in regions[1:]:#make sure that lb<=ub
        if lb<=union_regions[-1][1]:
            #has intersection
            if ub>=union_regions[-1][1]:
                union_regions[-1][1]=ub
        else:
            #No intersection, a new section emerges
                union_regions.append([lb,ub])
    return union_regions

def get_bits(structure):
    """
    Helper method to compute list of species on each site.
    Includes vacancies
    """
    all_bits = []
    for site in structure:
        bits = []
        for sp in sorted(site.species.keys()):
            bits.append(str(sp))
        if site.species.num_atoms < 0.99:
            bits.append("Vacancy")
        all_bits.append(bits)
    return all_bits

def get_n_bits(prim,sublat_merge_rule = None):
    """
        Returns an n_bits list with the same shape as merge_sublats.
    """
    n_bits = []
    if sublat_merge_rule is None:
        for sublat in prim:
            n_bits.append(list(range(len(sublat.species))))
            if sublat.species.num_atoms<0.99:
                n_bits[-1].append(len(sublat.species))
            #Vac will be taken as the last specie
    else:
        for group in sublat_merge_rule:
            n_bits.append(list(range(len(prim[group[0]].species))))
            if prim[group[0]].species.num_atoms<0.99:
                n_bits[-1].append(len(prim[0].species))
    return n_bits

def get_indgrps(prim,matrix,merge_sublats=None):
    """
       Generate the index groups for MC manipulation.
    """
    scs = int(round(abs(np.linalg.det(matrix))))
    if merge_sublats is not None:
        sublat_list = merge_sublats
    else:
        sublat_list = [[i] for i in range(len(prim))]
    sublats_WorthToExpand = []
    for sublat in sublat_list:
        WorthToExpand = True
        sublat_species = prim[sublat[0]].species
        if type(sublat_species)==str or len(sublat_species)<=1:
            WorthToExpand = False
        else:
            for specie in sublat_species:
                if abs(sublat_species[specie]-1.00)<0.001:
                    WorthToExpand = False
                    break
        sublats_WorthToExpand.append(WorthToExpand)

    indGrps = [list(range(i*scs*len(sublat),(i+1)*scs*len(sublat))) for \
               i,sublat in enumerate(sublat_list) if sublats_WorthToExpand[i]]

    return indGrps

def get_sublat_id(id_in_sc,sc_size, sublat_merge_rule = None):
    """
        When merge_rule = None, no merging ,which means each site in a primitive cell will be considered a 
        different sublattice (index group) when filpping.
        If any merging rule, should have the form like: [[0,1],[2,3]] so site 0 and site 1 in prim will now 
        be considered as a same sublattice. So are site 2 and site 3.
    """
    id_in_prim = id_in_sc//sc_size #This is determined by the way pymatgen make_supercell
    if sublat_merge_rule is None:
        return id_in_prim
    else:
        for idx,merge_group in sublat_merge_rule:
            if id_in_prim in merge_group:
                return idx
        return None

def get_init_occu(clus_sup):
    """
        Generates a random inital occupation.
    """
    rand_str = clus_sup.supercell
    order = OrderDisorderedStructureTransformation(algo=2)
    rand_str = order.apply_transformation(rand_str)
    return clus_sup.occu_from_structure(rand_str)

def get_flip(inds, occu, n_bits, sc_size, sublat_merge_rule = None):
    """
    Flip a single specie.
    inds: site indices of the same sublattice.
    n_bits: num bits. For example: ['Li+','Mn2+','Mn3+'],['O2-','F-']
            then: [0,1,2],[0,1]
    """
    i = random.choice(inds)
    sl_id = get_sublat_id(i,sc_size,sublat_merge_rule)
    sp_avail = [sp for sp in n_bits[sl_id] if occu[i] != sp]
    if len(sp_avail)==0: return []
    sp_flip = random.choice(sp_avail)
    return [(i, sp_flip)]

def get_x(occu, n_bits,sc_size,sublat_merge_rule=None):
    """
        Get x composition values from an occupation state list given by the monte carlo part.
    Inputs:
        occu: len(occu)=num_of_sites, not num_of_sublats
        n_bits: the n_bits table given by get_n_bits
        sc_size: size of the supercell, in interger
    Outputs:
        x: Number of each species in a sublattice, excluding the last specie.
    """
    x = []
    for sublat in n_bits:
        x.append([0]*(len(sublat)-1))
    #the last specie in a list is taken as reference

    for i,sp in enumerate(occu):
        idx = get_sublat_id(i,sc_size, sublat_merge_rule)
        if sp<len(n_bits[idx])-1:
            x[idx][sp]+=1

    return x
#Check this tomorrow!

def get_dx(flip, occu, n_bits, sc_size, sublat_merge_rule = None):
    flipped_site = flip[0][0]
    sp_before = occu[flipped_site]
    sp_after = flip[0][1]
    flipped_sublat_id = get_sublat_id(flipped_site,sc_size, sublat_merge_rule)    

    dx = []
    for sublat in n_bits:
        dx.append([0]*len(sublat))
    #dx has the same shape as n_bits, but x does not.

    if sp_before < len(n_bits[flipped_sublat_id]):
        dx[flipped_sublat_id][sp_before]-=1

    if sp_after < len(n_bits[flipped_sublat_id]):
        dx[flipped_sublat_id][sp_after]+=1

    return dx

def update_x(x,dx):
    x_new = deepcopy(x)
    for i,sublat in enumerate(x):
        for j,specie in enumerate(x[i]):
            x_new[i][j] += dx[i][j]
    return x_new

def normalize_x(x,sc_size,sublat_merge_rule=None):
    normed_x = deepcopy(x)
    for s_id,sublat in enumerate(x):
        for sp_id,sp_num in enumerate(sublat):
            if sublat_merge_rule is None:
                N = sc_size
            else:
                N = sc_size*len(sublat_merge_rule[s_id])
            normed_x[s_id][sp_id]=float(normed_x[s_id][sp_id])/N
    return normed_x

def get_dphi(de,mu,dx):
    """
        Input:
            miu: relative chemical potentials, by each sublattice.
                 has the same shape as x, but not dx
    """
    dmx = 0
    for s_id,sublat in enumerate(mu):
        for sp_id,mu_sp in enumerate(sublat):
            dmx += dx[s_id][sp_id]*mu[s_id][sp_id]
    return de-dmx
    

def estimate_from_series(Qs,p=0.001,z_a = 2.576):
    ### See A van de Walle and M Asta 2002 Modelling Simul. Mater. Sci. Eng. 10 521
    """
       Fix confidence level to 99%
    """
    #print('est_e')
    L = len(Qs)
    if L == 0 :
        return None,None,None

    step = max(L//100,1)
    #select steps carefully so that we won't be wasting time.
    for t1 in range(1,L,step):
        Q_sl1 = np.array(Qs)[t1-1:(t1+L)//2] 
        Q_sl2 = np.array(Qs)[(t1+L)//2:L]
        diff = np.abs(np.average(Q_sl1)-np.average(Q_sl2))
        if diff>p:
            continue
        Q_sl = np.array(Qs)[t1-1:L]     
        corr_Qsl = np.correlate(Q_sl,Q_sl,mode='full')
        corr_Qsl = corr_Qsl[corr_Qsl.size//2:]
        if corr_Qsl.size<2:
            continue
        V = np.average(Q_sl*Q_sl)-np.average(Q_sl)**2

        x = np.arange(corr_Qsl.size)
        y = np.log(corr_Qsl)
        x_av = np.average(x)
        y_av = np.average(y)
        xy_av = np.average(x*y)
        x2_av = np.average(x*x)
        slope = (x_av*y_av - xy_av)/(x_av**2-x2_av)
        rou = np.exp(-1.0*slope)

        var_Q = V/Q_sl.size*(1+rou)/(1-rou)
        if V/Q_sl.size*(1+rou)/(1-rou)<(p/z_a)**2:
            #Converged
            return np.average(Q_sl), var_Q, t1
    return None, None, None
    
def estimate_x(xs,t1):
    #print('est_x')
    x_sample = xs[t1-1:]
    x_av = [[0]*len(sublat) for sublat in x_sample[0]]
    var_x = [[0]*len(sublat) for sublat in x_sample[0]]

    for sl_id,sublat in enumerate(x_av):    
        for sp_id,sp_num in enumerate(sublat):
            x_series = []
            for x in x_sample:
                x_series.append(float(x[sl_id][sp_id]))
            x_series = np.array(x_series)
            corr_x = np.correlate(x_series,x_series,mode='full')
            corr_x = corr_x[corr_x.size//2:]
            V = np.average(x_series*x_series)-np.average(x_series)**2
            reg = LinearRegression()
            reg.fit(np.arange(corr_x.size).reshape(-1,1),np.log(corr_x))
            rou = np.exp(-1.0*reg.coef_[0])
            v_x = V/x_series.size*(1+rou)/(1-rou)
            var_x[sl_id][sp_id]=v_x
            x_av[sl_id][sp_id]=np.average(x_series)

    return x_av,var_x

#Semi-grand canonical ensemble tools. Currently only useful for 2 components.
def run_T_miu(ecis, cluster_supercell, occu, T, miu, ind_groups, sublat_merge_rule = None, max_loops=1000000, e_prec=0.001):
    """
    miu: relative chemical potentials, for each sublattice.
         For example, we have two sublattices, each occupied with 
         ['Li+','Mn2+','Mn3+'], ['O2-','F-'], then miu should take the 
         form: [[0.1,-0.5],[0.8]], etc. The composition of a supercell
         should also have similar form.
    e_prec: target energy precision in eV. By default, 0.001.
    """
    ind_groups_non_empty = []
    for ind_group in ind_groups:
        if len(ind_group) > 0:
            ind_groups_non_empty.append(deepcopy(ind_group))
    ind_groups = ind_groups_non_empty
    if len(ind_groups) == 0:
        raise ValueError("MC: all index groups are empty, so there are no valid flips!")

    k = 8.617332e-5
    kt = k * T
    beta = 1.0 / kt
    n_bits = get_n_bits(cluster_supercell.cluster_expansion.structure)

    corr = cluster_supercell.corr_from_occupancy(occu)
    e = np.dot(corr, ecis) * cluster_supercell.size
    x = get_x(occu, n_bits,cluster_supercell.size,sublat_merge_rule)

    energies = [e]
    xs = [x]    
    print('Init e: {}, init x: {}'.format(e,x))

    n_loops = 0
    equilibriated = False
    while n_loops < max_loops*cluster_supercell.size:
        chosen_group = random.choice(ind_groups)
        flip = get_flip(chosen_group, occu, n_bits,cluster_supercell.size,sublat_merge_rule)
        d_corr, new_occu = cluster_supercell.delta_corr(flip, occu, debug=False)
        #print('\nd_corr',d_corr,'\noccu',occu,'\nnew_occu',new_occu)
        de = np.dot(d_corr, ecis) * cluster_supercell.size
        dx = get_dx(flip, occu, n_bits,cluster_supercell.size,sublat_merge_rule)
        dphi = get_dphi(de,miu,dx)
        p = np.exp(-beta * dphi)

        if np.random.rand() <= p:
            corr += d_corr
            occu = new_occu
            e += de
            x = update_x(x,dx)
            #print(e,x)
            energies.append(e)
            xs.append(x)
        
        if n_loops%(10000*cluster_supercell.size) == 0 and n_loops>=(max_loops*cluster_supercell.size//100):
            #Check exit every 10000 steps, after minimum mc runs of 200000 steps, to avoid numerical instability,
            #And lower computational costs.
            estimated_e,var_e,t1 = estimate_from_series(energies)
            if estimated_e is not None:
                equilibriated = True
                estimated_x,var_x = estimate_x(xs,t1)
                break
        n_loops+=1

    if n_loops == max_loops and not equilibriated:
        print("Warning: Maximum possible filpping steps reached, but still not converged.")
        print("T:{}, mu: {}".format(T,miu))
        return estimated_e,None,x,None,occu
    else:
        return estimated_e,var_e,estimated_x,var_x,occu

def detect_pt_from_series(Qs,Cs,k=3,z_a=2.576):
    """
    This detects whether a phase transition happens at the last point of a control series.
    Gives a boolean.
    By default, choosing k=3, confidence level 99%
    """
    #Currently seem to find premature PT at high temp. Wondering why
    if len(Qs)<k+3:
        return False
    C = np.array(Cs[:-1]).reshape(-1,1)
    xs = np.array([Cs[-1]**j for j in range(0,k+1)])
    ys = Qs[-1]
    X = np.hstack([C**j for j in range(0,k+1)])
    y = np.array(Qs[:-1])
    n = len(Qs)-1
    
    a = np.linalg.inv(X.T@X)@X.T@y
    s2 = np.linalg.norm(y-X@a)**2/(n-k-1)
    V = s2*np.linalg.inv(X.T@X)
    dQ = abs(ys-np.dot(xs,a))
    v = np.dot(xs,V@xs)
    if dQ >= z_a*np.sqrt(v+s2):
        return True
    else:
        return False

"""
    From here on, only applicable to 2 component systems.
"""
def scalar_to_matrix(mu,prim,sublat_merge_rule=None):
    """
    This converts any scalar into matrix, with the same shape as miu.
    """
    prim_bits = get_bits(prim)
    miu = []
    if sublat_merge_rule is not None:
        sublat_list = sublat_merge_rule
    else:
        sublat_list = [[i] for i in range(len(prim_bits))]
    for sublat in sublat_list:
        sl_bits = prim_bits[sublat[0]]
        if len(sl_bits)==2:
            miu.append([mu])
        elif len(sl_bits)==1:
            miu.append([])
        else:
            raise ValueError("The system is not 2-components!")
    return miu


def matrix_to_scalar(mat):
    """
        2 comp systems only.
    """
    for sublat in mat:
        if len(sublat)==1:
            return sublat[0]
    raise ValueError("The provided matrix is not from a two component system!")

class TwoCompPDFinder(object):
    def __init__(self,ecis,ce,matrix=[[5,0,0],[0,5,0],[0,0,5]],\
                 min_mu=-1,max_mu=1,mu_step=0.01,\
                 T0=500,T1=5000,n_beta_steps=100,sublat_merge_rule=None,\
                 z_a=2.576,k=3,fine_search_range=6):
        """
            This class is a generator of the T-x phase diagram of a 2 component system.
            Extension to higher dimension is possible, by reimplementing the plot_pbs
            method, and replacing the matrix_to_scalar method and the scalar_to_matrix method.
            But high dimensional phase diagram is not recommended here.
            Inputs:
                ecis: 
                    ecis of the cluster expansion of the system
                ce: 
                    cluster expansion object
                matrix: 
                    supercell matrix. Default: 5*5*5 cubic
                min_mu,max_mu: 
                    initial mu searching range.(unit: eV)
                mu_step: 
                    step length of searching mu.(unit:eV)
                    This value should not be set as small as possible. Computational cost,
                    as well as the difficulty of convergence within the metastable region,
                    can be serious problem if mu_step is too small.
                T0: 
                    minimum searching temperature (unit K).
                T1: 
                    maximum searching temperature.
                ### T0, T1, max_mu and min_mu varies between systems drastically, so make
                ### to set them up manually before you start! 
                n_beta_steps: 
                    number of beta searching steps.
                sublat_merge_rule: 
                    see generator_tools.py documentation. This is the same thing as 
                    generator.merge_sublats
                z_a: 
                    confidence factor. Default confidence level set to 99%. You can lower 
                    it if the MC runs get hard to converge, or the phase boundary tracker 
                    die too young before they are adequately extended.
                k: 
                    the order of the polynomial fit used to detect phase transition.
                    Higher orders not recommended, as they tend to over-extend a boundary.
                fine_search_range: 
                    fine searching factor, used by extend_pb. This method will try
                    to find a new phase boudary point within a mu range centered on the 
                    predicted mu. range = range(mu_pred - fine_search_range * mu_step,
                    mu_pred + fine_search_range * mu_step, mu_step) 
                    If you are still unable to find any phase transition after increasing
                    this, then your beta value might have overshoot the congruent point.
                    That will be fine. 
        """
        self.ecis = ecis
        prim = ce.structure
        #The composition of prim must be modified to suit the scs
        self.matrix = np.array(matrix)
        self.scs = int(round(abs(np.linalg.det(matrix))))
        
        self.sublat_merge_rule = sublat_merge_rule      
        
        sites = []
        #Modify site compositions, and accordingly, the ce object
        for st_original in prim:
            if st_original.species.num_atoms < 0.99:
                n_species = len(st_original.species)+1
            else:
                n_species = len(st_original.species)
            new_comp = {sp:float(self.scs//n_species)/self.scs for sp in\
                        st_original.species}
            st_mod = PeriodicSite(new_comp,st_original.frac_coords,prim.lattice,\
                                  properties = st_original.properties)
            sites.append(st_mod)
        self.prim = Structure.from_sites(sites)

 
        sites_to_expand = [site for site in self.prim if site.species.num_atoms < 0.99 \

                            or len(site.species) > 1]
        exp_structure = Structure.from_sites(sites_to_expand)
        self.ce = ClusterExpansion(structure=self.prim,expansion_structure=exp_structure,\
                                  symops=ce.symops,clusters=ce.clusters,ltol=ce.ltol,\
                                  stol=ce.stol,angle_tol=ce.angle_tol,\
                                  supercell_size=ce.supercell_size,use_ewald=ce.use_ewald,\
                                  use_inv_r=ce.use_inv_r,eta=ce.eta,basis=ce.basis,\
                                  sm_type=ce.sm_type)

        self.clus_sup = self.ce.supercell_from_matrix(matrix)

        self.min_mu = min_mu
        self.max_mu = max_mu
        self.mu_step = mu_step
        
        self.T0 = T0; self.T1 = T1;
        kb = 8.617332e-5    #eV unit
        self.beta0 = 1/(kb*T0); self.beta1 = 1/(kb*T1)
        self.n_beta_steps = n_beta_steps
        self.beta_step = abs(self.beta0-self.beta1)/n_beta_steps
        self.fine_search_range = fine_search_range
    
        self.ind_groups = get_indgrps(self.prim,self.matrix,self.sublat_merge_rule)

        self.cur_occu = get_init_occu(self.clus_sup)
        #Store occu to make later MC runs equilibriate faster.
        
        self.z_a = z_a
        self.k = k
        self._pbs = None

    def find_pb(self,min_mu,max_mu,T):
        """
        This function trys to find the starting points of phase boundaries, under fixed T,
        within (min_mu,max_mu,range)
        By default, starting from 3000K, doing cubic regression under 99% confidence level.
        Inputs:
            max_mu: upper bound of chempot searching range
            min_mu: lower bound of chempot searching range
            mu_step: chempot search step
            T: temperature of the line search
            z_a: confidence level of declaring a phase transition, 99% by default
            k: polynomial order of fitting a phase boundary. 3 by default.
        Outputs:
            All the statistically prominent phase boundaries. 
            (T, mu_pb, x_alpha,x_gamma,dmu/dbeta)
        """
        kb = 8.617332e-5
        beta0=1/(kb*T)
        print('Searching phase boundaries under {} k, between {} eV, {} eV.'.format(T,min_mu,max_mu))

        mu_range = np.arange(min_mu,max_mu,self.mu_step)
        #energies = []
        xs = []
        mus = []
        energies = []
        pbs = []
        #This stores the starting points of phase boundaries.
        for mu in mu_range:
            print('Running mu: {} ev'.format(mu))
            miu = scalar_to_matrix(mu,self.prim,self.sublat_merge_rule)
            e,_,x_lst,vx_lst,new_occu=run_T_miu(self.ecis, self.clus_sup, self.cur_occu, T, miu, \
                                         self.ind_groups, self.sublat_merge_rule)
            
            #Use normalized quantities
            print('Energy : {} eV'.format(e))
            x_lst = normalize_x(x_lst,self.scs,self.sublat_merge_rule)
            x = matrix_to_scalar(x_lst)
            vx = matrix_to_scalar(vx_lst)
            
            #Only untransformed phase points will be saved into series
            if detect_pt_from_series(energies+[e],mus+[mu],k=self.k,z_a=self.z_a):
                print("Phase transition detected, refining.")
                cnt = 0
                mu_ub = mu
                mu_lb = mu-self.mu_step
                e_gamma = e
                x_gamma = x
                #refine location of PB for twice using bisection. Can reach 0.01eV precision.
                while cnt<3:
                    cnt+=1
                    mu_test = (mu_ub+mu_lb)/2
                    miu = scalar_to_matrix(mu_test,self.prim,self.sublat_merge_rule)
                    e,_,x_lst,vx_lst,new_occu = run_T_miu(self.ecis, self.clus_sup, self.cur_occu,\
                                                     T, miu, self.ind_groups, \
                                                     self.sublat_merge_rule)


                    x_lst = normalize_x(x_lst,self.scs,self.sublat_merge_rule)
                    x = matrix_to_scalar(x_lst)
                    vx = matrix_to_scalar(vx_lst)
              
                    if detect_pt_from_series(energies+[e],mus+[mu_test],k=self.k,z_a=self.z_a):
                        mu_ub= mu_test
                        e_gamma = e
                        x_gamma = x
                    else:
                        mu_lb= mu_test
                        xs.append(x)
                        mus.append(mu_test)
                        energies.append(e)
                        self.cur_occu = new_occu
    
                e_alpha = energies[-1]
                x_alpha = xs[-1]

                if abs(x_alpha-x_gamma)/np.sqrt(vx) > self.z_a:  
                    mu_mid = (mu_ub+mu_lb)/2
                    dpb = (e_gamma-e_alpha)/(beta0*(x_gamma-x_alpha))- mu_mid/beta0
                    #position and tangent of the discovered phase boundary point.
                    #Gamma: high mu phase;
                    #Alpha: low mu phase.
                    pbs.append((T, mu_mid,x_alpha,x_gamma,dpb))
                    print('PT point: ',(T,mu_mid,x_alpha,x_gamma,dpb))
    
                    #clear out, restart from another phase region.
                    xs = []
                    mus = []
                    energies = []
                else:
                    #PT is bogus, will be treated as if no PT was detected.
                    xs.append(x)
                    mus.append(mu)
                    energies.append(e)
                    self.cur_occu = new_occu  
            else:
                xs.append(x)
                mus.append(mu)
                energies.append(e)
                self.cur_occu = new_occu
    
        if len(pbs)==0:
            print("No phase transition can be found under T = {} K, mu = {}~{} eV!".\
                   format(T,min_mu,max_mu))
        print('Detected phase boundary points:\n',pbs)
        return pbs

    def extend_pb(self,pb_line,k=None, z_a = None): #Is this gramatically correct?
        """
        return pb point at higher beta. returns None if the pb line is dead.
        Here we are actually doing a much smaller line seach over mu,
        near the next predicted pb point.
        When two neighboring phases have same x, will terminate and returns
        None; when a legal new pb point is truly discovered, we will return 
        this point.
        """
        kb = 8.617332e-5
        if k is None:
            k = self.k
        if z_a is None:
            z_a = self.z_a

        pb_end = pb_line[-1]
        last_T = pb_end[0]; last_mu = pb_end[1]; last_dpb = pb_end[4]
        
        old_x_alphas = [pb[2] for pb in pb_line]
        old_x_gammas = [pb[3] for pb in pb_line]
        old_betas = [1/(kb*pb[0]) for pb in pb_line]
  
        last_bt = 1/(kb*last_T)
        
        pred_bt = last_bt+self.beta_step
        pred_T = 1/(kb*pred_bt)
        pred_mu = last_mu+self.beta_step*last_dpb
        #Technical difficulty: How to set up this range ...
        #Try to seach for phase transition in a small range of mu near the predicted mu
        
        min_mu = pred_mu-self.mu_step*self.fine_search_range
        max_mu = pred_mu+self.mu_step*self.fine_search_range

        print("Fine searching:")
        new_pbs = self.find_pb(min_mu,max_mu,pred_T)
        #This should be able to tolerate a prediction deviation within +-2 mu_step
        if len(new_pbs)==0:
            print("Warning: No phase boundary found within the chosen find search range\
                   around T={}, mu={}. You may need to increase n_beta_steps or fine_seach_\
                   range. By default, this is treated as if the PB line is 'dead'.")
            return None

        elif len(new_pbs)>1:
            print("Warning: Multiple new phase transformations found in the fine searching range\
                   , you may need to decrease the fine_search_range parameter by a bit. By default,\
                   This is treated as if the PB line is 'dead'.")
            return None

        else:
        #If the confidence range of x_alpha and x_gamma overlaps, or a new prediction of x_alpha or 
        #x_gamma can no longer be regressed by neighter of the older series, we can declare the 'death' 
        #of a phase boundary.
            new_pb = new_pbs[0]
            new_beta = 1/(kb*new_pb[0])
            new_x_alpha = new_pb[2]
            new_x_gamma = new_pb[3]
            alpha_pred_alpha = detect_pt_from_series(old_x_alphas+[new_x_alpha],old_betas+[new_beta],\
                               k=k,z_a=z_a) 
            gamma_pred_gamma = detect_pt_from_series(old_x_gammas+[new_x_gamma],old_betas+[new_beta],\
                               k=k,z_a=z_a) 
            alpha_pred_gamma = detect_pt_from_series(old_x_alphas+[new_x_gamma],old_betas+[new_beta],\
                               k=k,z_a=z_a) 
            gamma_pred_alpha = detect_pt_from_series(old_x_gammas+[new_x_alpha],old_betas+[new_beta],\
                               k=k,z_a=z_a) 
            if alpha_pred_alpha and gamma_pred_gamma and not(gamma_pred_alpha) and not(alpha_pred_gamma):
            #Both lines can predict the new pb point well, and they do not overlap within confidence level
                return new_pb
            else:
                return None

    def walk_pb(self):
        """
            Discover, and walk active phase boundaries. Drop out dead boundaries.
            Outputs:
                a list containing all phase boundary information
        """
        print("#### Automatic Phase Boundary Tracking ####")
        if self._pbs is not None:
            return self._pbs
        bt0 = self.beta0; bt1 = self.beta1;
        dbt = self.beta_step
        beta_range = np.arange(bt1,bt0,dbt)
        active_pb_lines = []
        dead_pb_lines = []
        search_ranges = [[self.min_mu,self.max_mu]]
        active_mus = []
        
        kb = 8.617332e-5
        for beta in beta_range:
            T = 1/(kb*beta)
            print("Moving to T = {} K.".format(T))
            #Detect new active phase boundaries
            while len(search_ranges):
                mu_lb,mu_ub=search_ranges[0]
                new_pbs = self.find_pb(mu_lb,mu_ub,T)       
                if len(new_pbs):     
                    active_pb_lines.extend([[new_pb] for new_pb in new_pbs])
                    active_mus.extend([new_pb[1] for new_pb in new_pbs])
                    search_ranges.remove([mu_lb,mu_ub])
                    mu_key = lambda x:x[1]
                    active_pb_lines = sorted(active_pb_lines,key=mu_key)
                    active_mus = sorted(active_mus)          
    
            for idx,(pb_line,active_mu) in enumerate(zip(active_pb_lines,active_mus)):
                new_pb_point = self.extend_pb(pb_line) #needs to be written
                if new_pb_point is not None:
                    pb_line.append(new_pb_point)
                else:
                    #This line is dead
                    dead_pb_lines.append(pb_line)
                    active_pb_lines.remove(pb_line)
                    print("A PB line is dead at T={} K, mu={} ev".\
                          format(pb_line[-1][0],pb_line[-1][1]))
    
                    search_lb = active_mus[idx-1] if idx>=1 else min_mu 
                    search_ub = active_mus[idx+1] if idx<=len(active_mus)-2 else max_mu
                    search_ranges.append([search_lb,search_ub])
    
                    active_mus.remove(active_mu)
    
            search_ranges = merge_unions(search_ranges)  #needs  to be written
    
        self._pbs = active_pb_lines+dead_pb_lines       
        return self._pbs     
    
    def plot_pbs(self):
        if self._pbs is None:
            self._pbs = self.walk_pb()

        for pb_line in self._pbs:
            T_key = lambda x:x[0]          
            pb_line = sorted(pb_line,key=T_key)
           
            #Transform into T-X space
            x_alphas = [pt[2] for pt in pb_line]
            x_gammas = [pt[3] for pt in pb_line]
            Ts = [pt[0] for pt in pb_line]
            
            if len(pb_line)>3:
                tcka, ua = interpolate.splprep([x_alphas,Ts],s=0)
                tckg, ug = interpolate.splprep([x_gammas,Ts],s=0)
    
                T0 = min(Ts)
                T1 = max(Ts)
                dT = (T1-T0)/100
                Ts_new = np.arange(T0,T1+dT,dT)
                x_alphas_new,Ts_new = interpolate.splev(Ts_new,tcka)
                x_gammas_new,Ts_new = interpolate.splev(Ts_new,tckg)       
    
            plt.plot(x_alphas,Ts,'x',x_alphas_new,Ts_new,color='k')
            plt.plot(x_gammas,Ts,'x',x_gammas_new,Ts_new,color='k')
         
        plt.xlabel('x')  #Make sure to normalize x
        plt.ylabel('T/K')
        plt.title('T-x phase diagram')
    
        return plt

    @classmethod
    def from_dict(cls,d): 
        ecis = d['ecis'] #ecis must be a list, not an array
        ce = ClusterExpansion.from_dict(d['ce'])
        matrix = d['matrix']
        min_mu = d['min_mu'] if 'min_mu' in d else -1
        max_mu = d['max_mu'] if 'max_mu' in d else 1
        mu_step = d['mu_step'] if 'mu_step' in d else 0.01
        T0 = d['T0'] if 'T0' in d else 500
        T1 = d['T1'] if 'T1' in d else 5000
        n_beta_steps = d['n_beta_steps'] if 'n_beta_steps' in d else 50
        sublat_merge_rule = d['sublat_merge_rule'] if 'sublat_merge_rule' in d else None
        z_a = d['z_a'] if 'z_a' in d else 2.576
        k = d['k'] if 'k' in d else 3
        fine_search_range = d['fine_search_range'] if 'fine_search_range' in d else 6
        
        socket = cls(ecis=ecis,ce=ce,matrix=matrix,min_mu=min_mu,\
                     max_mu=max_mu,mu_step=mu_step,T0=T0,T1=T1,\
                     n_beta_steps=n_beta_steps,\
                     sublat_merge_rule=sublat_merge_rule,\
                     z_a=z_a,k=k,fine_search_range = fine_search_range)

        if 'pbs' in d: socket._pbs = d['pbs']
        return socket

    def as_dict(self):
        return {'ecis':self.ecis.tolist() if type(self.ecis)==np.ndarray else self.ecis,\
                'ce':self.ce.as_dict(),\
                'matrix':self.matrix.tolist(),\
                'min_mu':self.min_mu,\
                'max_mu':self.max_mu,\
                'mu_step':self.mu_step,\
                'T0':self.T0,\
                'T1':self.T1,\
                'n_beta_steps':self.n_beta_steps,\
                'sublat_merge_rule':self.sublat_merge_rule,\
                'z_a':self.z_a,\
                'k':self.k,\
                'fine_search_range':self.fine_search_range,\
                'pbs':self._pbs\
               }
