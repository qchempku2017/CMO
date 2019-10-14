bit_inds = get_bit_inds(cesup1.supercell)
clusters = ce.clusters
b_clusters = []
for sc,sc_inds in cesup1.cluster_indices:
    for i,all_combo in enumerate(sc.bit_combos):
        for combo in all_combo:
            b_clusters.extend([[ bit_inds[site][combo[s]] for s,site in enumerate(sc_ind)]\
                                  for sc_ind in sc_inds])

cnts = {}
for sc,sc_inds in cesup1.cluster_indices:
    for c_ind in sc_inds:
        if c_ind.to_list() not in cnts:
            cnts[c_ind.tolist()]=1
        else:
            cnts[c_ind.tolist()]+=1
