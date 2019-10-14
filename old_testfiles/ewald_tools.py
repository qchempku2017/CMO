from pymatgen.analysis.ewald import EwaldSummation
from pymatgen import Structure
from pymatgen.core.sites import PeriodicSite
from utils import *

def ewald_correction(ce,supmat,ecis):
    """
     Decompose electrostatic interactions in a supercell into 2 body
     and point cluster terms.
    """

    print("Making up all ewald interactions for supercell:",supmat)
    cs = ce.supercell_from_matrix(supmat)

    ew_str = Structure.from_sites([PeriodicSite('H+',s.frac_coords,s.lattice) for s in cs.supercell])
    H = EwaldSummation(ew_str,eta=ce.eta).total_energy_matrix
 
    #Ewald energy E_ew = (q+r)*H*(q+r)'. 1/2 already absorbed I used a stupid way to get H but quite effective.
    supbits = cs.bits
    r = np.array([GetIonChg(bits[-1]) for bits in supbits])
    chg_bits = [[GetIonChg(bit)-GetIonChg(bits[-1]) for bit in bits[:-1]] for bits in supbits]
    H_r = np.dot(H,r)
    ewald_0 = np.dot(r,np.dot(H,r))
    
    ewald_clusters = []
    ewald_interactions = []
    bit_inds = get_bit_inds(cs.supercell)
    #bit_inds is 1 based     

    if not ce.use_inv_r:
        eci_ew = ecis[-1]
        for i in range(len(bit_inds)):
            for j in range(len(bit_inds)):
                if i!=j:
                    for k in range(len(bit_inds[i])):
                        for l in range(len(bit_inds[j])):
                            bit_a = bit_inds[i][k]
                            bit_b = bit_inds[j][l]
                            ewald_clusters.append([bit_a,bit_b]) 
                            ewald_interactions.append(eci_ew*chg_bits[i][k]*chg_bits[j][l]*H[i][j])
                else:
                    for k in range(len(bit_inds[i])):
                        bit = bit_inds[i][k]
                        ewald_clusters.append([bit])
                        ewald_interactions.append(eci_ew*(chg_bits[i][k]**2*H[i][i]+H_r[i]*chg_bits[i][k]*2))
    else:
        #When using inv_r, an independent ewald sum is generated for each specie-specie pair, and the sums are
        #considered components of corr, and also the original ewald summation.
        N_sp = sum([len(site.species_and_occu) for site in clus_sup.cluster_expansion.structure])
        N_eweci = 1+N_sp+N_sp*(N_sp-1)//2 #original ewald term, then species, then specie pairs.
        eci_ew = ecis[-N_eweci:]
        
        for sc,inds in cs.cluster_indices:
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
            for j in range(len(bit_inds)):
                if i!=j:
                    for k in range(len(bit_inds[i])):
                        for l in range(len(bit_inds[j])):
                            bit_a = bit_inds[i][k]
                            bit_b = bit_inds[j][l]
                            ewald_clusters.append([bit_a,bit_b])
                            id_a = sp_list[i][k]
                            id_b = sp_list[j][l]
                            
                            id_abpair = id_a*(2*N_sp-id_a-1)//2 + id_b - id_a -1 # Serial id of a,b pair in eci_ew list.
                            ewald_interactions.append(eci_ew[1+N_sp+id_abpair]* (chg_bits[i][k]*chg_bits[j][l]*H[i][j]))
                else: #Point terms
                    for k in range(len(bit_inds[i])):
                        bit = bit_inds[i][k]
                        b_clusters_ew.append([bit])
                        id_bit = 1+sp_list[i][k]
                        point_eci = eci_ew[id_bit]*chg_bits[i][k]**2*H[i][i]
                        for m in range(len(bit_inds)):
                            id_a = id_bit
                            id_b = sp_list[m][-1] #id of the reference specie
                            id_abpair = id_a*(2*N_sp-id_a-1)//2 + id_b -id_a -1
                            #Calculate H_r term with weight!
                            point_eci += 2*chg_bits[i][k]*H[i][m]*r[m]*eci_ew[1+N_sp+id_abpair]
                        ewald_interactions.append(point_eci)
        
    return ewald_clusters,ewald_interactions
