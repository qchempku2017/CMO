from gurobipy import *

##### Tool functions #####

##### class functions
class CEBlock(object):
    '''
        This is a implementation of 'block' in Wenxuan's 16 paper. Iteractive block in 18 paper will be implemented soon.
    '''
    def __init__(clus_sup, eci,block_range=1,hard_marker=1000000000000000000,eci_mul=1000000000,grain_den=4):
        self.cesup = clus_sup
        self.eci = eci
        self.matrix = clus_sup.supercell_matrix
        self.blkrange = block_range
        self.hard_marker=hard_marker
        self.eci_mul = eci_mul
        self.grain_den = grain_den
        self.use_ewald = self.cesup.use_ewald
        self.use_inv_r = self.cesup.use_inv_r

        self._frames = None
        self._lambda_grid = None
        self._frame_min_configs = None

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
        #Here we do electro-static correction. Reference zero energy state is the one that all sites 
        #are occupied by reference compound.
            print("Making up all ewald interactions")

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










        frame_min_configs = []
        #CEBlock is a new class I made to represent block in lowerbound solution. It has different _generate_mapping from ClusterSupercell.
        #Since clusters are too many in number, I only choose the clusters with top 10 ECI to split.
        for lbds,frame in blk.lambda_frames:
            print("Solving splitting frame under lambdas:", lbds)
            N_sites = blk.size
            soft_cls = []
            hard_cls = []

            for site_id in range(N_sites):
                hard_cls.extend([[hard_marker]+[int(-1*id_1),int(-1*id_2)] for id_1,id_2 in combinations(blk.bit_inds[site_id],2)])
                #Hard clauses to enforce sum(specie_occu)=1
        
            all_eci_sum = 0
            for b_cluster,eci in frame:
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
        
            num_of_vars = sum([len(line) for line in blk.bit_inds])
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
 
