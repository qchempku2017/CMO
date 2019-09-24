from pymatgen import Structure
from pymatpro.hamiltonian.pairwise import PairwiseHamiltonian
from pymatgen.analysis.structure_matcher import StructureMatcher, OrderDisorderElementComparator

import numpy as np

class PairwiseCEHamiltonian(PairwiseHamiltonian):
    
    def __init__(self, structure, cluster_expansion, ecis):
        self.structure = structure
        self.ce = cluster_expansion
        self.ecis = ecis
        self.energies = self._calculate_energies()

    
    def _calculate_energies(self):
        energies = np.zeros((len(self.structure), len(self.structure)))
        
        
        dm = self.structure.distance_matrix
        
        non_dup_sites = []
        blocked = []
        for i, site in enumerate(self.structure):
            if i in blocked:
                continue
            non_dup_sites.append(site)
            blocked.extend(np.where(dm[i] < 0.001)[0])
        
        cs = self.ce.supercell_from_structure(Structure.from_sites(non_dup_sites))
        sm = StructureMatcher(primitive_cell=False,
                              attempt_supercell=False,
                              allow_subset=True,
                              scale=True,
                              comparator=OrderDisorderElementComparator())

        aligned = sm.get_s2_like_s1(cs.supercell, self.structure)
        dists = aligned.lattice.get_all_distances(aligned.frac_coords, cs.supercell.frac_coords)
        lw_mapping = np.argmin(dists, axis=-1)

        bits = cs.bits
        exp_inds = np.where(np.sum(aligned.distance_matrix < 0.001, axis=-1) > 1)[0]

        for i in exp_inds:
            for j in exp_inds:
                if lw_mapping[i] == lw_mapping[j]:
                    continue
                occu = np.copy(cs.nbits)
                occu[lw_mapping[i]] = bits[lw_mapping[i]].index(str(aligned[i].specie))
                occu[lw_mapping[j]] = bits[lw_mapping[j]].index(str(aligned[j].specie))
                energies[i][j] = np.dot(cs.corr_from_occupancy(occu), self.ecis)

        return energies