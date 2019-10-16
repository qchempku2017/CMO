__author__ = 'Fengyu Xie'
__version__ = 1.0

from gurobipy import *
from utils import *
from mc import *

from pymatgen.util.coord import lattice_points_in_supercell
from pymatgen.core.sites import PeriodicSite
from pymatgen.analysis.ewald import EwaldSummation
import numpy as np
import random

from itertools import product,chain
from utils import *

##### Tool functions #####
SITE_TOL = 1E-5
def _allclose_pbc(p1,p2):
    """
    differences in fractional coords must be integers.
    """
    dr = p1-p2
    return np.allclose(dr,np.floor(dr))

def _coord_list_mapping_blockonly(subset,superset,blkrange,atol=SITE_TOL):
    """
       This is a revised version of coord_list_mapping_pbc in pymatgen. Here we will give a mapping of contained cluster sites
       into supercell sites.
       The sites in a supercell are marked as 1,2,3,...,N_site, and corresponding sites in its neighbor with displacement 
       will be marked as site=site_in_sc+(x*(range+1)**2+y*(range+1)+z)*N_site
       Here we map cluster by cluster.
       Note: This can still be improved in efficiency.
    """
    point_inds = []
   
    for point in subset:
        if point[0]<0 or point[1]<0 or point[2]<0:
            #print("Cluster {} not contained in supercell.".format(subset))
            return None
        disp = np.floor(point)
        x = disp[0]
        y = disp[1]
        z = disp[2]
        #print(point_in_sc)

        for p_id,p in enumerate(superset):
            #print(p,point_in_sc,_allclose_pbc(p,point_in_sc))
            if _allclose_pbc(p,point):
                point_id_in_sc=p_id
                #print(point_id_in_sc)
                break
        N_site = len(superset)
        point_ind = point_id_in_sc+(x*(blkrange+1)**2+y*(blkrange+1)+z)*N_site
        point_inds.append(point_ind)
    return np.array(point_inds,dtype=np.int64)

##### class functions
class CEBlock(object):
    '''
        This is a implementation of 'block' in Wenxuan's 16 paper. Iteractive block in 18 paper will be implemented soon.

        A cluster is said to be 'contained' within a SC when all its points resides within the SC, or it has at least one
        point resides within that SC, and the rest of the points only protrude towards +x, +y, and +z direction. So I 
        revised ClusterSupercell.generate_mapping to implement this.

        A 'contained' cluster is splitted into neighbor SC's on the +x, +y and +z direction by 3 parameters labeled lambda.
        By default, only split to the nearest neighbor.

        The number of clusters to be splitted are also highly restricted. By default I only choose to split SymmetrizedClusters
        (bits included) with top 10 absolute ECI values(of course after electrostatic correction, if required).
        
        When blkrange = 1, {s} contains all variables in 0, +1x, +1y, +1z supercells and perhaps, surroundings. The variable
        indexing rule is such that in SC-0, index of variables range from 1~N_SC, and SC-1x, N_SC+1~2N_SC, so on.
        if two indices are exactly diffrent by integer times of N_SC, then they are translationally equivalent. 

        clus_sup: the cluster supercell object to bulid a block with
        eci: eci corresponding to clus_sup.ce
        composition: a composition dict in the same form as in GSSocket class.
        block_range: the distance, measured in unit of supercell parameters, that you split your variables away. default: 1,
                     only split to the neighbors.
        hard_marker: marker number of a hard clause to use in MAXSAT solver
        eci_mul: a parameter to magnify your ECI's and truncate them
        num_of_sclus_tosplit: This specifies how many SymmetrizedClusters are chosen for splitting. Select those with highest
                              absolute ECI values.
    '''
    def __init__(self, ce, matrix, eci, composition, bclus_ewald=[], eci_ewald=[],\
                 block_range=1, solver='CCEHC-incomplete', init_method = 'random'\
                 hard_marker=1000000000000000, eci_mul=1000000,n_iniframe=20,cutoff_eciabs=1E-4):

        self.ce = ce
        self.clus_sup = ce.supercell_from_matrix(matrix)
        self.sym_clusters = self.ce.symmetrized_clusters
        self.clusters = self.ce.clusters
        self.eci = eci
        self.zero_eci = self.eci[0]

        self.solver = solver
        self.init_method = init_method

        self.composition = composition
        sp_count = sum(self.composition[0].values())
        self.frac_comp = [{sp:(float(sublat[sp])/sp_count) for sp in sublat} for sublat in composition]

        self.bit_inds_sc = get_bit_inds(clus_sup.supercell)
        self.num_bits_sc = max(itertools.chain(*self.bit_inds_sc))

        self.matrix = matrix
        self.prim_to_supercell = np.linalg.inv(self.matrix)
        self.Nsite = len(clus_sup.supercell)
        self.scs = int(round(np.abs(np.linalg.det(self.matrix))))
        self.prim = clus_sup.cluster_expansion.structure

        self.blkrange = block_range
        self.n_split = (self.blkrange+1)**3-1

        self.hard_marker=hard_marker
        self.eci_mul = eci_mul
        self.num_of_sclus_tosplit = num_of_sclus_tosplit
        self.fcoords = clus_sup.fcoords
        #print('fcoords:',self.fcoords)

        self.n_iniframe = n_iniframe

        #print(bit_inds_sc)

        
        self._configs = None
        #Only bclusters with |eci|>self.cutoff_eciabs will be splitted

        self.cutoff_eciabs = cutoff_eciabs
        self._blkenergy = None
        self._lambda_param = None

        self._contained_cluster_indices = None

        self._splitted_bclusters = None
        self._splitted_ecis = None
        self._splitted_ori_id = None        
        self._original_bclusters = None
        self._original_ecis = None
 
        self._max_bit = None 
        self._max_site = None
        self._num_of_vars = None 
      
        if self.ce.use_ewald and (len(bclus_ewald)==0 or len(eci_ewald)==0):
            raise ValueError("Cluster expansion requires ewald correction, but none was given.")
        self.ewald_bclusters = bclus_ewald
        self.ewald_ecis = eci_ewald
        

#### properties ####
    @property
    def contained_cluster_indices(self):
        if self._contained_cluster_indices is None:
            print("Mapping symmetrized bit-clusters in the new rule.")
            ts = lattice_points_in_supercell(self.matrix)
            self._contained_cluster_indices = []
    
            for sc in self.sym_clusters:
                #print("Processing symmetrized cluster {}".format(sc))
                prim_fcoords = np.array([c.sites for c in sc.equivalent_clusters])
                fcoords = np.dot(prim_fcoords, self.prim_to_supercell)
                #tcoords contains all the coordinates of the symmetrically equivalent clusters
                #the indices are: [equivalent cluster (primitive cell), translational image, index of site in cluster, coordinate index]
                tcoords = fcoords[:, None, :, :] + ts[None, :, None, :]
                tcs = tcoords.shape
                tcoords_by_clusters = tcoords.reshape(-1,tcs[2],3)
                all_inds = []
                for cluster in tcoords_by_clusters:
                    inds = _coord_list_mapping_blockonly(cluster, self.fcoords, self.blkrange,\
                           atol=SITE_TOL)
                    if inds is not None:
                        all_inds.append(inds)
                self._contained_cluster_indices.append((sc, np.array(all_inds)))            
                # I revised the boundary condition of pbc mapping, to give only clusters that are 'contained' in a SC, and a cluster is 
                # no longer wrapped by periodic condition.
            print("Clusters trimmed and mapped!")
        return self._contained_cluster_indices

    @property
    def original_bclusters(self):
        """
        All the bit_clusters 'positive-contained' in the boundary of a supercell. Not splitted to their images yet.
        """
        if self._original_bclusters is None or self._original_ecis is None:
            self._original_bclusters = []
            self._original_ecis = []
            bit_inds = self.bit_inds_sc 
           
            for sc,sc_inds in self.contained_cluster_indices:
                for i,combo_orbit in enumerate(sc.bit_combos):
                    for j,combo in enumerate(combo_orbit):
                        for sc_ind in sc_inds:
                            bclus = []
                            for s,site in enumerate(sc_ind):
                            #site=site_in_sc+(x*(range+1)**2+y*(range+1)+z)*N_site
                                site_in_sc = site%self.Nsite
                                dz = (site//self.Nsite)%(self.blkrange+1)
                                dy = (site//self.Nsite)//(self.blkrange+1)%(self.blkrange+1)
                                dx = (site//self.Nsite)//(self.blkrange+1)//(self.blkrange+1)
                                bit_in_sc = bit_inds[site_in_sc][combo[s]]
                                bit = bit_in_sc+(dx*(self.blkrange+1)**2+dy*(self.blkrange+1)+dz)*self.num_bits_sc
                                bclus.append(bit)
                            self._original_bclusters.append(bclus)

                        #Remember the structure of danill's eci array? The first term should be zero clus, and sc_b_id starts from 1
                        eci_new = {size:[self.eci[(sc.sc_b_id):(sc.sc_b_id+len(sc.bit_combos))] for sc in self.clusters[size]] \
                                   for size in self.clusters}
    
                        #combo_id = sum([len(orbit) for orbit in sc.bit_combos[:i]])+j
                        #one eci for one orbit.
    
                        bit_multip = sc.multiplicity*len(combo_orbit)
                        self._original_ecis.extend([eci_new[len(sc.bits)][sc.sc_id-self.clusters[len(sc.bits)][0].sc_id][i]/bit_multip\
                                                      for sc_ind in sc_inds])
        return self._original_bclusters
    
    @property
    def original_ecis(self):
        if self._original_bclusters is None or self._original_ecis is None:
            original_bclusters = self.original_bclusters
        return self._original_ecis

    @property
     def splitted_bclusters(self):
        # Mapping rule of a site indexed i in original SC to images:(x:+x, y:+y, z:+z)(x,y,z<=extend_range):
        # new_index = i + (x*(range+1)**2+y*(range+1)+z)*n_bits_sc
        # When range=1, a clustere would be splitted into 7 images.
         if self._splitted_bclusters is None or self._splitted_ecis is None or self._splitted_ori_id is None:
             self._splitted_ori_id = []
             self._splitted_ecis = []
             self._splitted_bclusters = []
             # id of splitted clusters in self._original_bclusters. This is important because not 
             #all clusters are splitted
             # How many images will one original cluster be splitted into.
             for i,(bclus,eci) in enumerate(zip(self.original_bclusters,self.original_ecis)):
                 if abs(eci)>=self._cutoff_eciabs:
                     for x in range(self.blkrange+1):
                         for y in range(self.blkrange+1):
                            for z in range(self.blkrange+1):
                               if x==0 and y==0 and z==0:
                                   continue
                               bclus_xyzimage = [idx + (x*(self.blkrange+1)**2 + y*(self.blkrange+1) + z)* self.num_bits_sc for idx in bclus]
                               self._splitted_bclusters.append(bclus_xyzimage)
                               self._splitted_ecis.append(eci)
                     self._splitted_ori_id.append(i)
         return self._splitted_bclusters

    @property      
    def splitted_ecis(self):
        if self._splitted_bclusters is None or self._splitted_ecis is None or self._splitted_ori_id is None:
            splitted_bclusters = self.splitted_bclusters
        return self._splitted_ecis

    @property      
    def splitted_ori_id(self):
        """
        This list specifies the location of a splitted cluster's original image in self._original_bclusters
        """
        if self._splitted_bclusters is None or self._splitted_ecis is None or self._splitted_ori_id is None:
            splitted_bclusters = self.splitted_bclusters
        return self._splitted_ecis

    @property
    def max_bit(self):
        if self._max_bit is None:
            self._max_bit = max(itertools.chain(*self.splitted_bclusters))
        return self._max_bit

    @property
    def num_of_vars(self):
        if self._num_of_vars is None or self._max_site is None:
            max_bit_orimage = (self.max_bit-1)%self.num_bits_sc+1
            #bit starts from 1
            for s, site in enumerate(self.bit_inds_sc):
                if max_bit_orimage in site:
                    max_site_orimage = s
                    break
            z = self.max_bit//self.num_bits_sc%(self.blkrange+1)
            y = self.max_bit//self.num_bits_sc//(self.blkrange+1)%(self.blkrange+1)
            x = self.max_bit//self.num_bits_sc//(self.blkrange+1)//(self.blkrange+1)
            max_site = max_site_orimage+(x*(self.blkrange+1)**2+y*(self.blkrange+1)+z)*self.Nsite
            self._max_site=max_site
            self._num_of_vars = max(self.bit_inds_sc[max_site_orimage])+\
                             +(x*(self.blkrange+1)**2+y*(self.blkrange+1)+z)*self.num_bits_sc
        return self._num_of_vars

    @property
    def max_site(self):
        if self._max_site is None or self._num_of_vars is None:
            nv = self.num_of_vars
        return self._max_site

    @property
    def num_of_lambdas(self):
        return len(self.splitted_bclusters)

#### public callable methods ####
    def solve(self):
        '''
        Generate some configurations(low energy ones) and form facets with them. Optimize LP. See if block energy has converged.
        If not, update lambdas with LP results and solve MAXSAT with new lambda set, and add the newly solved block config to it,
        until E converges or no new configs emerges.
        '''
        
        print("Number of variables:",self.num_of_vars)
        if self._configs is None:
            self._configs=self._initialize()
            print("Initialized with {} configs.".format(self.n_iniframe))

        while True:
            m = Model("lambda-solving")
            lambdas = m.addVars(self.num_of_lambdas,,lb=0.0, ub=1.0) #Refer to help(Model.addVars) for more info, add all lambdas here
            E = m.addVar(vtype=GRB.CONTINUOUS,name="E",lb=-GRB.INFINITY, ub=GRB.INFINITY)
            m.update()
            m.setObjective(E,GRB.MAXIMIZE)
   
            # weight of all original clusters shouldn't be less than 0. 'Hard' constraints.
            all_hard = self._set_hard_expressions(lambdas)
            #print("Lambdas:",self.num_of_lambdas)
            for hard in all_hard:
                #print("Hard:\n",hard)
                m.addConstr(hard<=1.0)

            # E = sum(J*Clus), to maximize E, use E-sum(J*clus)<=0, 'soft' constraints.
            for config in self._configs:
                soft_expr = self._config_to_soft_expression(config,lambdas)
                #print("Soft:\n",soft_expr)
                m.addConstr(E<=soft_expr)

            m.update()           
            m.optimize()
            self._lambda_param = [lambdas[v_id].x for v_id in lambdas]
            blkenergy = m.objVal
            if self._blkenergy is not None:
                if abs(self._blkenergy-blkenergy)<0.001:
                    print("Lowerbound for composition {} converged.".format(self.composition))
                    break

            maxsat_bclus,maxsat_ecis=self._form_maxsat()
            self._Write_MAXSAT_input_forblk(maxsat_bclus,maxsat_ecis)

            Call_MAXSAT(solver=self.solver)
            new_config = Read_MAXSAT()[:self.num_of_vars]

            if new_config in self._configs:
                print("Lowerbound for composition {} converged.".format(self.composition))
                break

            self._blkenergy = blkenergy
            self._configs.append(new_config)
        return self._blkenergy
            


#### private tools ####

    def _initialize(self):
        """
        a_config = tuple(vars_in_sc,vars_in_sc_x+1,vars_in_sc_y+1,vars_in+sc_z+1)
        a_clusterfunc_set = tuple(in_sc(ewald_corrected, if needed),in_sc_x+1,in_sc_y+1,in_sc_z+1)
        """
        if self.init_method = 'mc':
            iniconfigs = self._mc_sampling()
        elif self.init_method = 'random':
            iniconfigs = self._rand_sampling()
        else:
            raise NotImplementedError("Initialization method not implemented.")
        return iniconfigs
    
    def _rand_sampling(self):
        """
        Complete random intialization of a block
        """
        import copy
        rand_occus = []
        disorderSites = []
        for i,site in enumerate(self.prim):
            disorderSite = PeriodicSite(self.frac_comp[i],site.frac_coords,self.prim.lattice,properties=site.properties)
            disorderSites.append(disorderSite)
        disorderStr = Structure.from_sites(disorderSites)
        disorderStr.make_supercell(self.matrix)

        for i in range(self.n_iniframe):
            order = OrderDisorderedStructureTransformation(algo=2)
            randStr = copy.deepcopy(disorderStr)
            randStr = order.apply_transformation(randStr)
            rand_occus.append(self.clus_sup.occu_from_structure(randStr))
        iniconfigs = self._scoccu_to_blkconfig(rand_occus)
        return iniconfigs      
    
    def _mc_sampling(self):
        """
        Pile up mc low energy supercells to make a block.
        """
        sites_WorthToExpand = []
        for sublat in self.frac_comp:
            site_WorthToExpand = True
            for specie in sublat:
                if abs(sublat[specie]-1.00)<0.001:
                    site_WorthToExpand=False
                    break
            sites_WorthToExpand.append(site_WorthToExpand)
        indGrps= [list(range(i*self.scs,(i+1)*self.scs)) for i in range(len(self.frac_comp)) if sites_WorthToExpand[i]] 

        order = OrderDisorderedStructureTransformation(algo=2)
        randSites = []
        for i,site in enumerate(self.prim):
             randSite = PeriodicSite(self.frac_comp[i],site.frac_coords,self.prim.lattice,properties=site.properties)
             randSites.append(randSite)
        randStr = Structure.from_sites(randSites)
        randStr.make_supercell(self.matrix)
        randStr = order.apply_transformation(randStr)
        init_occu = self.clus_sup.occu_from_structure(randStr)
        #print(randStr)

        base_occu  = simulated_anneal(ecis=self.eci, cluster_supercell=self.clus_sup,occu=init_occu,ind_groups=indGrps,n_loops=20000, init_T=5100, final_T=100,n_steps=20)
        # Sampling run
        print("Sampling.") 

        occu, min_occu, min_e, rand_occus = run_T(ecis=self.eci, cluster_supercell=self.clus_sup, occu=deepcopy(base_occu),T=100,n_loops=100000, ind_groups = indGrps, n_rand=self.n_iniframe, check_unique=True)       
        #print(rand_occu)
        iniconfigs = self._scoccu_to_blkconfig(rand_occus)
        return iniconfigs
    
    def _scoccu_to_blkconfig(self,rand_occu):
        '''
        Here we transfrom a list of occupations generated by monte-carlo program into a block confirugration in MAXSAT output format, as
        we store them in self._configs.
        
        We no longer replicate low E structural units. Instead we randomly pile up low E SC units.
        '''
        #print('occu',occu)
        posvars_scs = []
        configs = []
        for rand,rand_e in rand_occu:
            posvars_sc = []
            for s_id,sp_id in enumerate(rand):
                if sp_id < len(self.bit_inds_sc[s_id]):
                    var = self.bit_inds_sc[s_id][sp_id]
                #last specie ignored
            posvars_scs.append(posvars_sc)

        n_pieces = self.num_of_vars//self.num_bits_sc
        for i in range(self.n_iniframe):
            posvars=[[num+i*self.num_bits_sc for num in random.choice(posvars_scs)]\
                                    for i in range(n_pieces)] 
            posvars=sum(posvars,[])
            config = [(b if b in posvars else -b) for b in range(1,self.num_of_vars+1)]
            configs.append(config)

        return configs
              
    def _config_to_soft_expression(self,config,lambdas):
    #### LinExpr class provided by gurobi is extrememly easy to manipulate and modify! For example:
    #    cond = LinExpr() //then we have an empty linear expression.
    #    cond += x+y //the variable x is linked into the linear expression, and the expression becomes x+y. Adress linked, not the values!
    #    And also:
    #    cond += vec[2]+3*vec[3]
        o_clusfuncs,s_clusfuncs,e_clusfuncs=self._config_to_clusfuncs(config)
        #print(config)
        soft_expr = LinExpr()
        #ewald_term = 0
        #clus_term = self.zero_eci*self.scs
        soft_expr.add(self.zero_eci*self.scs)
        for i,clusfunc in enumerate(s_clusfuncs):
            if clusfunc!=0:
                soft_expr.add(clusfunc*self.splitted_ecis[i]*lambdas[i])
        for i,clusfunc in enumerate(e_clusfuncs):
            if clusfunc!=0:
                soft_expr.add(clusfunc*self.ewald_ecis[i])
                #ewald_term+=self.ewald_ecis[i]
        for i,clusfunc in enumerate(o_clusfuncs):
            # When this cluster is a splitted.
            if i in self.splitted_ori_id:
                pos = self.splitted_ori_id.index(i)
                extended_ids = list(range(pos*self.n_split,(pos+1)*self.n_split))
                hard_expr = LinExpr()
                for e_id in extended_ids:
                    hard_expr.add(lambdas[e_id])
                if clusfunc!=0:
                    soft_expr.add(clusfunc*self.original_ecis[i]*(1-hard_expr))
                    #clus_term += self.original_ecis[i]
            # When this cluster is not splitted.
            else:
                if clusfunc!=0:
                    soft_expr.add(clusfunc*self.original_ecis[i])
                    #clus_term += self.original_ecis[i]

        #print("ewald term for sc:",ewald_term,"clus term for sc:",clus_term,"scs",self.scs)

        #Regularize!
        return soft_expr/self.scs


    def _config_to_clusfuncs(self,config):
        "Calculate cluster functions based on configuration of the block."

        original_clusfuncs = [reduce( (lambda x,y:x*y), [(1 if config[bit-1]>0 else 0) for bit in bclus])\
                               for bclus in self.original_bclusters]
        splitted_clusfuncs = [reduce( (lambda x,y:x*y), [(1 if config[bit-1]>0 else 0) for bit in bclus])\
                               for bclus in self.splitted_bclusters]  
        ewald_clusfuncs = [reduce( (lambda x,y:x*y), [(1 if config[bit-1]>0 else 0) for bit in bclus])\
                               for bclus in self.ewald_bclusters]        
        return original_clusfuncs, splitted_clusfuncs, ewald_clusfuncs


    def _set_hard_expressions(self,lambdas):
        all_hard_exprs = []
        #for id0,(bclus,eci) in enumerate(zip(self._original_bclusters,self._original_ecis)):
        for pos,id0 in enumerate(self.splitted_ori_id):
            #pos = self._splitted_ori_id.index(id0)
            extended_ids = list(range(pos*self.n_split,(pos+1)*self.n_split))
            hard_expr = LinExpr()
            for e_id in extended_ids:
                hard_expr.add(lambdas[e_id])
            all_hard_exprs.append(hard_expr)
        return all_hard_exprs
 
    def _form_maxsat(self):
        '''
        This generates a set of bclus and their corresponding ecis, to feed into global_tools.Write_to_MAXSAT function.
        '''
        maxsat_bclusters = []
        maxsat_ecis = []
        for id0,(bclus,eci) in enumerate(zip(self.original_bclusters,self.original_ecis)):
            maxsat_bclusters.append(bclus)
            if id0 in self.splitted_ori_id:
                pos = self.splitted_ori_id.index(id0)
                extended_ids = list(range(pos*self.n_split,(pos+1)*self.n_split))
            # id of bclustet's images in neighboring supercells.
                splitted_sum = 0
                for ext_id in extended_ids:
                    splitted_sum+=self._lambda_param[ext_id]
                maxsat_ecis.append((1-splitted_sum)*eci)
                for ext_id in extended_ids:
                    maxsat_bclusters.append(self.splitted_bclusters[ext_id])
                    maxsat_ecis.append(self.splitted_ecis[ext_id]*self._lambda_param[ext_id])
            else:
                maxsat_ecis.append(eci)
            # Please, please do not merge the two cycles! Or you will breaking the corresponding relation between bclus and eci!

        for bclus_ew,eci in zip(self.ewald_bclusters,self.ewald_ecis):
            _in_b_clusters = False
            for bc_id,bclus in enumerate(maxsat_bclusters):
                if len(bclus)>2:
                    continue
                if bclus_ew == bclus or bclus_ew == Reversed(bclus):
                    if len(bclus_ew)==2:
                        maxsat_ecis[bc_id]=maxsat_ecis[bc_id]+2*eci
                        _in_b_clusters = True
                        break
                    elif len(bclus_ew)==1:
                        maxsat_ecis[bc_id]=maxsat_ecis[bc_id]+eci
                        _in_b_clusters = True
                        break
                if not _in_b_clusters:
                    if len(bclus_ew)==2:
                        maxsat_bclusters.append(bclus_ew)
                        maxsat_ecis.append(eci*2)
                    elif len(bclus_ew)==1:
                        maxsat_bclusters.append(bclus_ew)
                        maxsat_ecis.append(eci)
        return maxsat_bclusters,maxsat_ecis

    def _Write_MAXSAT_input_forblk(self,soft_bcs,soft_ecis):
        print('Preparing MAXSAT input file for LB convergence')
        soft_cls = []
        hard_cls = []
        
        for site_id in range(self.max_site+1):
            dz = (site_id//self.Nsite)%(self.blkrange+1)
            dy = (site_id//self.Nsite)//(self.blkrange+1)%(self.blkrange+1)
            dx = (site_id//self.Nsite)//(self.blkrange+1)//(self.blkrange+1)
            site_in_sc = site_id % self.Nsite
            bit_inds = self.bit_inds_sc
            for id_1,id_2 in combinations(bit_inds[site_in_sc],2):
                id1_img = id_1 + (dx*(self.blkrange+1)**2+dy*(self.blkrange+1)+dz)*self.num_bits_sc
                id2_img = id_2 +(dx*(self.blkrange+1)**2+dy*(self.blkrange+1)+dz)*self.num_bits_sc
                # Maximum bit appeared in hard clauses.
                hard_cls.append([self.hard_marker]+[int(-1*id1_img),int(-1*id2_img)])
            #Hard clauses to enforce sum(specie_occu)=1
       
        all_eci_sum = 0
        for b_cluster,eci in zip(soft_bcs, soft_ecis):
            if int(eci*self.eci_mul)!=0: 
        #2016 Solver requires that all weights >=1 positive int!! when abs(eci)<1/eci_mul, cluster will be ignored!
                if eci>0:
                    clause = [int(eci*self.eci_mul)]
                    all_eci_sum+=int(eci*self.eci_mul)
            #MAXSAT 2016 series only takes in integer weights
                    for b_id in b_cluster:
                        clause.append(int(-1*b_id))
        #Don't worry about the last specie for a site. It is take as a referecne specie, 
        #thus not counted into nbits and combos at all!!!
                    soft_cls.append(clause)
                else:
                    clauses_to_add = []
                    for i in range(len(b_cluster)):
                        clause = [int(-1*eci*self.eci_mul),int(b_cluster[i])]
                        all_eci_sum+=int(-1*eci*self.eci_mul)
                        for j in range(i+1,len(b_cluster)):
                            clause.append(int(-1*b_cluster[j]))
                        clauses_to_add.append(clause)
                    soft_cls.extend(clauses_to_add)
    
        print('Soft clusters converted!')
        if all_eci_sum > self.hard_marker:
            print('Hard clauses marker might be too small. You may consider using a bigger value.')
    
        all_cls = hard_cls+soft_cls
        #print('all_cls',all_cls)
            
        num_of_cls = len(all_cls)
        maxsat_input = 'c\nc Weighted paritial maxsat\nc\np wcnf %d %d %d\n'%(self.num_of_vars,num_of_cls,self.hard_marker)
        for clause in all_cls:
            maxsat_input+=(' '.join([str(lit) for lit in clause])+' 0\n')
        f_maxsat = open('maxsat.wcnf','w')
        f_maxsat.write(maxsat_input)
        f_maxsat.close()
        print('maxsat.wcnf written.')
