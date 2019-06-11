from gurobipy import *

class CEBlock(object):
    def __init__(clus_sup, block_range=1,hard_marker=1000000000000000000,eci_mul=1000000000,grain_den=4):
        self.cesup = clus_sup
        self.matrix = clus_sup.supercell_matrix
        



        b_clusters_old = []
        eci_old = []

        for sc,sc_inds in clus_sup.cluster_indices:
            for i,all_combo in enumerate(sc.bit_combos):
                for combo in all_combo:
                    b_clusters_old.extend([[ bit_inds[site][combo[s]] for s,site in enumerate(sc_ind)]\
                                         for sc_ind in sc_inds])
                    eci_old.extend([eci_new[len(sc.bits)][sc.sc_id-clusters[len(sc.bits)][0].sc_id][i]\
                                         for sc_ind in sc_inds])
                    
        # Separate electrostatic(don't extend) and non-electrostatic part(extend)
        if self.ce.use_ewald:
            b_clusters = self.b_clus_corrected[mat_id]
            eci_new = self.ecis_corrected[mat_id]
            N_ewonly = len(ecis_new)-len(eci_old)
            b_clusters_ew = b_clusters[-N_ewonly:]
            eci_ew = eci_new[-N_ewonly:]
            for clus_id,clus in enumerate(b_clusters[:-N_ewonly]):
                if eci_old[clus_id]!=eci_new[clus_id]:
                    b_clusters_ew.append(clus)
                    eci_ew.append(eci_new[clus_id]-eci_old[clus_id])






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
 
