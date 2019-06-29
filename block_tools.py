__author__ = 'Fengyu Xie'
__version__ = 1.0

from gurobipy import *
from global_tools import *
from mc import *

from pymatgen.util.coord import lattice_points_in_supercell
from pymatgen.core.sites import PeriodicSite
from pymatgen.analysis.ewald import EwaldSummation
import numpy as np

from itertools import product

##### Tool functions #####
SITE_TOL = 1E-5
def _allclose_pbc(p1,p2):
    pbc = np.array(list(product([-1.0,0.0,1.0],[-1.0,0.0,1.0],[-1.0,0.0,1.0])))
    for dr in pbc:
        if np.allclose(dr,p1-p2):
            return True
    return False

def _coord_list_mapping_blockonly(subset,superset,blkrange,atol=SITE_TOL):
    '''
       This is a revised version of coord_list_mapping_pbc in pymatgen. Here we will give a mapping of contained cluster sites
       into supercell sites.
       The sites in a supercell are marked as 1,2,3,...,N_site, and corresponding sites in its neighbor with displacement 
       will be marked as site=site_in_sc+(x*(range+1)**2+y*(range+1)+z)*N_site
       Here we map cluster by cluster.
       Note: This can still be improved in efficiency.
    '''
    point_inds = []
    #print(subset)
    #print(superset)
    for point in subset:
        disp = np.floor(point)
        x = disp[0]
        y = disp[1]
        z = disp[2]
        if point[0]<0 or point[1]<0 or point[2]<0:
            #print("Cluster {} not contained in supercell.".format(subset))
            return None
        point_in_sc = point-disp
        #print(point_in_sc)

        for p_id,p in enumerate(superset):
            #print(p,point_in_sc,_allclose_pbc(p,point_in_sc))
            if _allclose_pbc(p,point_in_sc):
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
    def __init__(self, clus_sup, eci, composition, block_range=1,hard_marker=1000000000000000,eci_mul=1000000,num_of_sclus_tosplit=10,n_iniframe=20):

        self.cesup = clus_sup
        self.ce = clus_sup.cluster_expansion
        self.sym_clusters = self.ce.symmetrized_clusters
        self.clusters = self.ce.clusters
        self.eci = eci
        self.composition = composition
        sp_count = sum(self.composition[0].values())
        self.frac_comp = [{sp:(float(sublat[sp])/sp_count) for sp in sublat} for sublat in composition]

        self.matrix = clus_sup.supercell_matrix
        self.prim_to_supercell = np.linalg.inv(self.matrix)
        self.Nsite = len(clus_sup.supercell)
        self.scs = int(round(np.abs(np.linalg.det(self.matrix))))
        self.prim = clus_sup.cluster_expansion.structure

        self.blkrange = block_range
        self.hard_marker=hard_marker
        self.eci_mul = eci_mul
        self.num_of_sclus_tosplit = num_of_sclus_tosplit
        self.use_ewald = self.cesup.cluster_expansion.use_ewald
        self.use_inv_r = self.cesup.cluster_expansion.use_inv_r
        self.fcoords = clus_sup.fcoords
        self.n_iniframe = n_iniframe
        bit_inds_sc = []
        b_id = 1
        for i,site in enumerate(clus_sup.supercell):
            site_bit_inds = []
            for specie_id in range(len(site.species_and_occu)-1):
            #-1 since a specie on the site is taken as reference
                site_bit_inds.append(b_id)
                b_id+=1
                bit_inds_sc.append(site_bit_inds)
        self.bit_inds_sc = bit_inds_sc
        self.num_bits_sc = b_id-1

        #later we have anther variable self.bit_inds
        if self.use_ewald:
            print("Finding pairs corresponding to ewald interactions.")

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

                for i in range(len(bit_inds_sc)):
                    for j in range(i,len(bit_inds_sc)):
                        for k in range(len(bit_inds_sc[i])-1):
                            for l in range((k if i==j else 0),len(bit_inds_sc[j])-1):
                                if bit_inds_sc[i][k]!=bit_inds_sc[j][l]:
                                    bit_a = bit_inds_sc[i][k]
                                    bit_b = bit_inds_sc[j][l]
                                    b_clusters_ew.append([bit_a,bit_b]) 
                                    eci_return_ew.append(eci_ew* (chg_bits[i][k]*chg_bits[j][l]*H[i][j]))
                                else:
                                    bit = bit_inds_sc[i][k]
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
                       
                for i in range(len(bit_inds_sc)):
                    for j in range(i,len(bit_inds_sc)):
                        for k in range(len(bit_inds_sc[i])-1):
                            for l in range((k if i==j else 0),len(bit_inds_sc[j])-1):
                                if bit_inds_sc[i][k]!=bit_inds_sc[j][l]:
                                    bit_a = bit_inds_sc[i][k]
                                    bit_b = bit_inds_sc[j][l]
                                    b_clusters_ew.append([bit_a,bit_b])
                                    id_a = sp_list[i][k]
                                    id_b = sp_list[j][l]
                                    id_abpair = id_a*(2*N_sp-id_a-1)//2 + id_b - id_a -1 # Serial id of a,b pair in eci_ew list.
                                    eci_return_ew.append(eci_ew[N_sp+id_abpair]* (chg_bits[i][k]*chg_bits[j][l]*H[i][j]))
                                else: #Point terms
                                    bit = bit_inds_sc[i][k]
                                    b_clusters_ew.append([bit])
                                    id_bit = sp_list[i][k]
                                    point_eci = eci_ew[id_bit]*chg_bits[i][k]**2*H[i][i]
                                    for m in range(len(bit_inds_sc)):
                                        id_a = id_bit
                                        id_b = sp_list[m][-1] #id of the reference specie
                                        id_abpair = id_a*(2*N_sp-id_a-1)//2 + id_b -id_a -1
                                        #Calculate H_r term with weight!
                                        point_eci += 2*chg_bits[i][k]*H[i][m]*r[m]
                                    eci_return_ew.append(point_eci)
            self._ewald_bclusters = b_clusters_ew
            self._ewald_ecis = eci_return_ew
        
        self._configs = []
        self._num_of_lambdas = None
        self._blkenergy = None
        self._lambda_param = None

        self._splitted_bclusters = []
        self._splitted_ecis = []        
        self._original_bclusters = []
        self._original_ecis = []
        

#### public socket ####
    def solve(self):
        '''
        Generate some configurations(low energy ones) and form facets with them. Optimize LP. See if block energy has converged.
        If not, update lambdas with LP results and solve MAXSAT with new lambda set, and add the newly solved block config to it,
        until E converges or no new configs emerges.
        '''
        self._configs=self._initialize()
        #self._num_of_lambdas=len(self._splitted_bclusters)
        while True:
            m = Model("lambda-solving")
            lambdas = m.addVars(self._num_of_lambdas,ub=1.0) #Refer to help(Model.addVars) for more info, add all lambdas here
            E = m.addVar(vtype=GRB.CONTINUOUS,name="E")
            m.setObjective(E,GRB.MAXIMIZE)
   
            # weight of all original clusters shouldn't be less than 0. 'Hard' constraints.
            all_hard = self._set_hard_expressions(lambdas)
            for hard in all_hard:
                m.addConstr(hard,sense=GRB.LESS_EQUAL,rhs=1.0)

            # E = sum(J*Clus), to maximize E, use E-sum(J*clus)<=0, 'soft' constraints.
            for config in self._configs:
                soft_expr = self._config_to_soft_expression(config,lambdas)
                m.addConstr(soft_expr, GRB.LESS_EQUAL, E)
            
            m.optimize()
            self._lambda_param = [v.x for v in m.getValues() if v.varName != "E"]
            blkenergy = m.objVal
            if self._blkenergy:
                if abs(self._blkenergy-blkenergy)<0.001:
                    print("Lowerbound for composition {} converged.".format(self.composition))
                    break

            maxsat_bclus,maxsat_ecis=self._form_maxsat()
            self._Write_MAXSAT_input_forblk(maxsat_bclus,maxsat_ecis)

            Call_MAXSAT()
            new_config = Read_MAXSAT()[:self.num_of_vars]

            if new_config in self._configs:
                print("Lowerbound for composition {} converged.".format(self.composition))
                break

            self._blkenergy = blkenergy
            self._configs.append(new_config)
        return self._blkenergy
            
#### private tools ####
    # a_config = tuple(vars_in_sc,vars_in_sc_x+1,vars_in_sc_y+1,vars_in+sc_z+1)
    # a_clusterfunc_set = tuple(in_sc(ewald_corrected, if needed),in_sc_x+1,in_sc_y+1,in_sc_z+1)

    def _initialize(self):
    # Set initial configurations of {s} using MC method.
        print("Mapping symmetrized bit-clusters in the new rule.")
        ts = lattice_points_in_supercell(self.matrix)
        self.contained_cluster_indices = []

        for sc in self.sym_clusters:
            print("Processing symmetrized cluster {}".format(sc))
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
            self.contained_cluster_indices.append((sc, np.array(all_inds)))            
            # I revised the boundary condition of pbc mapping, to give only clusters that are 'contained' in a SC, and a cluster is 
            # no longer wrapped by periodic condition.
        print("Clusters trimmed and mapped!")
         
        self._cutoff_eciabs = sorted(self.eci,key=lambda x:abs(x))[-self.num_of_sclus_tosplit]
        #bclusters with abs(eci)<_cutoff_eciabs will not be splitted.

        bit_inds = self.bit_inds_sc 
       
        for sc,sc_inds in self.contained_cluster_indices:
            for i,all_combo in enumerate(sc.bit_combos):
                for combo in all_combo:
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
                    #self._original_bclusters.extend([[ bit_inds[site][combo[s]] for s,site in enumerate(sc_ind)]\
                    #                              for sc_ind in sc_inds])  
                    #need to map extended 'site' back, and need to do the splitting here.
                    eci_new = {size:[self.eci[(sc.sc_b_id-1):(sc.sc_b_id-1+len(sc.bit_combos))] for sc in self.clusters[size]] \
                               for size in self.clusters}

                    self._original_ecis.extend([eci_new[len(sc.bits)][sc.sc_id-self.clusters[len(sc.bits)][0].sc_id][i]\
                                                  for sc_ind in sc_inds])

        #### Splitting and extension. Trivial. ####
        # Mapping rule of a site indexed i in original SC to supercell:(x:+x, y:+y, z:+z)(x,y,z<=extend_range):
        # new_index = i + (x*(range+1)**2+y*(range+1)+z)*n_bits_sc
        # When range=1, a clustere would be splitted into 7 images.

        for bclus,eci in zip(self._original_bclusters,self._original_ecis):
            if abs(eci)>self._cutoff_eciabs:
                for x in range(self.blkrange+1):
                    for y in range(self.blkrange+1):
                       for z in range(self.blkrange+1):
                          if x==0 and y==0 and z==0:
                              continue
                          bclus_xyzimage = [idx + (x*(self.blkrange+1)**2 + y*(self.blkrange+1) + z)* self.num_bits_sc for idx in bclus]
                          self._splitted_bclusters.append(bclus_xyzimage)
                          self._splitted_ecis.append(eci)

        self.num_of_vars = max([max(bclus) for bclus in self._splitted_bclusters])
        #Max image bit index
        #Must extend len(bit_inds) to self.num_of_vars. This is required by MAXSAT solvers.                    

        print("Initializing 20 frames with lowest CE-MC energy!")
        self._num_of_lambdas = len(self._splitted_bclusters)
        #preparing MC
        
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
        init_occu = self.cesup.occu_from_structure(randStr)
        print(randStr)

        base_occu  = simulated_anneal(ecis=self.eci, cluster_supercell=self.cesup,occu=init_occu,ind_groups=indGrps,n_loops=20000, init_T=5100, final_T=100,n_steps=20)
        # Sampling run
        print("Sampling.")
        occu, min_occu, min_e, rand_occu = run_T(ecis=self.eci, cluster_supercell=self.cesup, occu=deepcopy(base_occu),T=100,n_loops=100000, ind_groups = indGrps, n_rand=self.n_iniframe, check_unique=True)       

        #print(rand_occu)
        iniconfigs = [self._scoccu_to_blkconfig(rand) for rand, rand_e in rand_occu]
        return iniconfigs
    
    def _scoccu_to_blkconfig(self,occu):
        '''
        Here we transfrom an occupation list generated by monte-carlo program into a block confirugration in MAXSAT output format, as
        we store them in self._configs.
        '''
        base_config=list(range(1,self.num_of_vars+1))
        config=[-v for v in base_config]
        for s_id,sp_id in enumerate(occu):
            print(occu)
            print(self.bit_inds_sc)
            if sp_id >= len(self.bit_inds_sc[s_id]):
                var =  self.bit_inds_sc[s_id][-1]
            else:
                var = self.bit_inds_sc[s_id][sp_id]
            for dx in range(self.blkrange+1):
                for dy in range(self.blkrange+1):
                    for dz in range(self.blkrange+1):
                        occupied_id = var-1+(dx*(self.blkrange+1)**2+dy*(self.blkrange+1)+dz)*self.num_bits_sc
                        if occupied_id < self.num_of_vars:
                            config[occupied_id]=-1*config[occupied_id]
        return config
               
    def _config_to_soft_expression(self,config,lambdas):
    #### LinExpr class provided by gurobi is extrememly easy to manipulate and modify! For example:
    #    cond = LinExpr() //then we have an empty linear expression.
    #    cond += x+y //the variable x is linked into the linear expression, and the expression becomes x+y. Adress linked, not the values!
    #    And also:
    #    cond += vec[2]+3*vec[3]
        o_clusfuncs,s_clusfuncs,e_clusfuncs=self._config_to_clusfuncs(config)
        soft_expr = LinExpr()
        for i,clusfunc in enumerate(s_clusfuncs):
            soft_expr.add(clusfunc*self._splitted_ecis[i]*lambdas[i])
        for i,clusfunc in enumerate(e_clusfuncs):
            soft_expr.add(clusfunc*self._ewald_ecis[i])
        for i,clusfunc in enumerate(o_clusfuncs):
            extended_ids = list( range(i*(self.blkrange+1)**3+1, (i+1)*(self.blkrange+1)**3) )
            hard_expr = LinExpr()
            for e_id in extended_ids:
                hard_expr.add(lambdas[e_id])
            soft_expr.add(clusfunc*self._original_ecis[i]*(1-hard_expr))
        return soft_expr

    def _config_to_clusfuncs(self,config):
    # Calculate cluster functions based on configuration of the block.
        original_clusfuncs = [reduce( (lambda x,y:x*y), [(1 if config[bit-1]>0 else 0) for bit in bclus])\
                               for bclus in self._original_bclusters]
        splitted_clusfuncs = [reduce( (lambda x,y:x*y), [(1 if config[bit-1]>0 else 0) for bit in bclus])\
                               for bclus in self._splitted_bclusters]  
        ewald_clusfuncs = [reduce( (lambda x,y:x*y), [(1 if config[bit-1]>0 else 0) for bit in bclus])\
                               for bclus in self._ewald_bclusters]        
        return original_clusfuncs, splitted_clusfuncs, ewald_clusfuncs

    def _set_hard_expressions(self,lambdas):
        all_hard_exprs = []
        for id0,(bclus,eci) in enumerate(zip(self._original_bclus,self._original_ecis)):
             extended_ids = list( range(id0*(self.blkrange+1)**3+1, (id0+1)*(self.blkrange+1)**3) )
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
        for id0,(bclus,eci) in enumerate(zip(self._original_bclusters,self._original_ecis)):
            maxsat_bclusters.append(bclus)
            extended_ids = list( range(id0*(self.blkrange+1)**3+1, (id0+1)*(self.blkrange+1)**3) )
            # id of bclustet's images in neighboring supercells.
            splitted_sum = 0
            for ext_id in extended_ids:
                splitted_sum+=self._splitted_ecis[ext_id]
            maxsat_ecis.append(1-splitted_sum)
            # Please, please do not merge the two cycles! Or you will breaking the corresponding relation between bclus and eci!
            for ext_id in extended_ids:
                maxsat_bclusters.append(self._splitted_bclusters[ext_id])
                maxsat_ecis.append(self._splitted_ecis[ext_id])

        for bclus_ew,eci in zip(self._ewald_bclusters,self._ewald_ecis):
            _in_b_clusters = False
            for bc_id,bclus in enumerate(maxsat_bclusters):
                if len(b_cluster)>2:
                    continue
                if bclus_ew == bclus or bclus_ew == Reversed(bclus):
                    maxsat_ecis[bc_id]=maxsat_ecis[bc_id]+2*eci
                    _in_b_clusters = True
                    break
                if not _in_b_clusters:
                    maxsat_bclusters.append(bclus_ew)
                    maxsat_ecis.append(eci*2)

        return maxsat_bclusters,maxsat_ecis

    def _Write_MAXSAT_input_forblk(self,soft_bcs,soft_ecis):
        print('Preparing MAXSAT input file.')
        soft_cls = []
        hard_cls = []
        max_sc_num = (self.num_of_vars-1)//self.num_bits_sc
        max_site_num = (max_sc_num+1)*self.Nsite-1
        max_bit = -1
    
        for site_id in range(max_site_num+1):
            dz = (site_id//self.Nsite)%(self.blkrange+1)
            dy = (site_id//self.Nsite)//(self.blkrange+1)%(self.blkrange+1)
            dx = (site_id//self.Nsite)//(self.blkrange+1)//(self.blkrange+1)
            site_in_sc = site_id % self.Nsite
            for id_1,id_2 in combinations(bit_inds[site_in_sc],2):
                id1_img = id_1 + (dx*(self.blkrange+1)**2+dy*(self.blkrange+1)+dz)*self.num_bits_sc
                id2_img = id_2 +(dx*(self.blkrange+1)**2+dy*(self.blkrange+1)+dz)*self.num_bits_sc
                # Maximum bit appeared in hard clauses.
                max_bit = max(max_bit,id1_img,id2_img)
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
                        all_eci_sum+=int(-1*eci*eci_mul)
                        for j in range(i+1,len(b_cluster)):
                            clause.append(int(-1*b_cluster[j]))
                        clauses_to_add.append(clause)
                    soft_cls.extend(clauses_to_add)
    
        print('Soft clusters converted!')
        if all_eci_sum > self.hard_marker:
            print('Hard clauses marker might be too small. You may consider using a bigger value.')
    
        all_cls = hard_cls+soft_cls
        #print('all_cls',all_cls)
            
        num_of_vars = max_bit
        num_of_cls = len(all_cls)
        maxsat_input = 'c\nc Weighted paritial maxsat\nc\np wcnf %d %d %d\n'%(num_of_vars,num_of_cls,hard_marker)
        for clause in all_cls:
            maxsat_input+=(' '.join([str(lit) for lit in clause])+' 0\n')
        f_maxsat = open('maxsat.wcnf','w')
        f_maxsat.write(maxsat_input)
        f_maxsat.close()
        print('maxsat.wcnf written.')
