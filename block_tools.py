__author__ = 'Fengyu Xie'
__version__ = 1.0

from gurobipy import *

##### Tool functions #####

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
        block_range: the distance, measured in unit of supercell parameters, that you split your variables away. default: 1,
                     only split to the neighbors.
        hard_marker: marker number of a hard clause to use in MAXSAT solver
        eci_mul: a parameter to magnify your ECI's and truncate them
        num_of_sclus_tosplit: This specifies how many SymmetrizedClusters are chosen for splitting. Select those with highest
                              absolute ECI values.
    '''
    def __init__(clus_sup, eci,block_range=1,hard_marker=1000000000000000,eci_mul=1000000,num_of_sclus_tosplit=10):

        self.cesup = clus_sup
        self.eci = eci
        self.matrix = clus_sup.supercell_matrix
        self.blkrange = block_range
        self.hard_marker=hard_marker
        self.eci_mul = eci_mul
        self.num_of_sclus_tosplit = num_of_sclus_tosplit
        self.use_ewald = self.cesup.use_ewald
        self.use_inv_r = self.cesup.use_inv_r

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

        #later we have anther variable self.bit_inds
        self.ewald_bclusters = []
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
        self.ewald_bclusters = zip(b_clusters_ew,eci_return_ew)
        
        self._configs = []

#### public socket ####
    def solve(self):
        self._configs=self._initialize_configs()
        while True:
            m = Model("lambda-solving")
            m.addVars() #Refer to help(Model.addVars) for more info, add all lambdas here
            m.addVar(vtype=GRB.CONTINUOUS,name="E")
            m.setObjective(E,GRB.MAXIMIZE)
            for config in self._configs:
                m.addConstraint(self._config_to_constraint(config))
            m.optimize()
            self._lambda_param = m.
            
#### private tools ####
    # a_config = tuple(vars_in_sc,vars_in_sc_x+1,vars_in_sc_y+1,vars_in+sc_z+1)
    # a_clusterfunc_set = tuple(in_sc(ewald_corrected, if needed),in_sc_x+1,in_sc_y+1,in_sc_z+1)
    def _sc_to_config(self):
               

    def _initialize_frames(self):
    # Set initial configurations of {s} using MC method
        print("Initializing 20 frames with lowest CE-MC energy!")
        return num_clus_to_split,iniconfigs

    def _config_to_constraint(config):
    
    def _config_to_energy(config):

    def _extend_sc_to_config(sc_occu):






        frame_min_configs = []
        #CEBlock is a new class I made to represent block in lowerbound solution. It has different _generate_mapping from ClusterSupercell.
        #Since clusters are too many in number, I only choose the clusters with top 10 ECI to split.
        for lbds,frame in blk.lambda_frames:
            print("Solving splitting frame under lambdas:", lbds)
            N_sites = blk.size
           #### Calling MAXSAT ####
            rand_seed = random.randint(1,100000)
            print('Callsing MAXSAT solver. Using random seed %d.'%rand_seed)
            os.system('cp ./maxsat.wcnf '+MAXSAT_PATH)
            os.chdir(MAXSAT_PATH)
            MAXSAT_CMD = './'+self.solver+' ./maxsat.wcnf'
            if self.solver in INCOMPLETE_MAXSAT:
                MAXSAT_CMD += ' %d %d'%(rand_seed,MAXSAT_CUTOFF)
            MAXSAT_CMD += '> maxsat.out'
            print(MAXSAT_CMD)
            os.system(MAXSAT_CMD)
            os.chdir('..')
            os.system('cp '+MAXSAT_PATH+'maxsat.out'+' ./maxsat.out')
            print('MAXSAT solution found!')

            #### MAXSAT Output Processing ####
            maxsat_res = []
            with open('./maxsat.out') as f_res:
                lines = f_res.readlines()
                for line in lines:
                    if line[0]=='v':
                        maxsat_res = [int(num) for num in line.split()[1:]]
            sorted(maxsat_res,key=lambda x:abs(x))
            if maxsat_res not in frame_min_configs:
                frame_min_configs.append(maxsat_res)
                # Coarse grained results might overlap

        for config in frame_min_configs:
 
