from __future__ import division

from collections import defaultdict
from pymatgen import Structure, PeriodicSite, Lattice
from pymatgen.analysis.structure_matcher import StructureMatcher, OrderDisorderElementComparator
from pymatgen.analysis.ewald import EwaldSummation
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer, SymmOp
from pymatgen.util.coord import is_coord_subset, lattice_points_in_supercell, coord_list_mapping,\
            coord_list_mapping_pbc, is_coord_subset_pbc
from monty.json import MSONable
from warnings import warn

import itertools
import numpy as np

import sys
import os
cwd = os.path.dirname(os.path.realpath(__file__))
sys.path.append(cwd)
# This is to tell python to find module util and ce_utils under the same directory of this file.

from utils import *
from ce_utils import delta_corr_single_flip

SITE_TOL = 1e-6

SYMMETRY_ERROR = ValueError("Error in calculating symmetry operations. Try using a "
                            "more symmetrically refined input structure. "
                            "SpacegroupAnalyzer(s).get_refined_structure().get_primitive_structure() "
                            "usually results in a safe choice")
#### Ortho basis
def calc_M(bits):
    M_tot = []
    for element in bits:
        M_tot.append(len(element))
    M_tot = np.array(M_tot)
    return M_tot

def sigma2gamma(sigma, alpha, M):
    if alpha %2==1:
        gamma = -np.cos(2* np.pi * np.ceil(alpha/2)* sigma / M)
    elif alpha %2 ==0:
        gamma = -np.sin(2* np.pi* np.ceil(alpha/2) * sigma /M)

    return gamma

#### Cluster functions other than 0/1 representation ####
def cluster_function_ortho(c_occu, combos, Ms):
    """
    combos: consits of symmetry equivalent bits, we need to average all symmetry equivalent bits
    * Plus 1 to combos to make it compatible with reduced species ordering from 1 to M-1
    * combos ranges from 0 to M-2 for reduced species originally
    """
    """
    CALPHAD: Computer Coupling of Phase Diagrams and Thermochemistry 33 (2009) 266–27
    """
    combos += 1
    bits_N = len(combos)
    for i_bits, bits in enumerate(combos):
        for i_bit, bit in enumerate(np.nditer(bits)):
            M = Ms[i_bit]
            if i_bit == 0:
                corr_bits = sigma2gamma(sigma= c_occu[:,i_bit], alpha= bit, M= M)
                # sigma: site variable(spin representation), gamma: site function
                # alpha: cluster element, equivalent with bit
            else:
                corr_bits *= sigma2gamma(sigma=c_occu[:,i_bit], alpha= bit, M=M)
        if i_bits == 0:
            corr_tot = corr_bits / bits_N
        else:
            corr_tot += corr_bits / bits_N

    return np.average(corr_tot)

#### supplementary tools
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
       #bits.append("Vacancy")
        all_bits.append(bits)
    return all_bits

#### Objects ####
class Cluster(MSONable):
    """
    An undecorated (no occupancies) cluster with translational symmetry
    """

    def __init__(self, sites, lattice):
        """
        Args:
            sites: list of frac coords for the sites
            symops: list of symops from pymatgen.symmetry
            lattice: pymatgen Lattice object
        """
        sites = np.array(sites)
        centroid = np.average(sites, axis=0)
        shift = np.floor(centroid)
        self.centroid = centroid - shift
        self.sites = sites - shift
        self.lattice = lattice
        self.c_id = None

    def assign_ids(self, c_id):
        """
        Method to recursively assign ids to clusters after initialization.
        """
        self.c_id = c_id
        return c_id + 1

    @property
    def size(self):
        return len(self.sites)

    @property
    def max_radius(self):
        coords = self.lattice.get_cartesian_coords(self.sites)
        all_d2 = np.sum((coords[None, :, :] - coords[:, None, :]) ** 2, axis=-1)
        return np.max(all_d2) ** 0.5

    def __eq__(self, other):
        if self.sites.shape != other.sites.shape:
            return False
        other_sites = other.sites + np.round(self.centroid - other.centroid)
        return is_coord_subset(self.sites, other_sites, atol=SITE_TOL)

    def __str__(self):
        points = str(np.round(self.sites,2)).replace("\n", " ").ljust(len(self.sites) * 21)
        return "Cluster: id: {:<3} Radius: {:<4.3} Points: {} Centroid: {}".format(self.c_id,
                                                                                   self.max_radius,
                                                                                   points,
                                                                                   np.round(self.centroid,2))

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def from_sites(sites):
        return Cluster([s.frac_coords for s in sites], sites[0].lattice)


class SymmetrizedCluster(MSONable):
    """
    Cluster with translational and structure symmetry. Also includes the possible orderings
    on the cluster
    """
    def __init__(self, base_cluster, bits, structure_symops):
        """
        Args:
            base_cluster: a Cluster object.
            bits: list describing the occupancy of each site in cluster. For each site, should
                    be the number of possible occupancies minus one. i.e. for a 3 site cluster,
                    each of which having one of Li, TM, or Vac, bits should be
                    [[0, 1], [0, 1], [0, 1]]. This is because the bit combinations that the
                    methodology *seems* to be missing are in fact linear combinations of other smaller
                    clusters. With least squares fitting, it can be verified that reintroducing these
                    bit combos doesn't improve the quality of the fit (though Bregman can do weird things
                    because of the L1 norm).
                    In any case, we know that pairwise ECIs aren't sparse in an ionic system, so
                    not sure how big of an issue this is.
            structure_symops: list of symmetry operations for the base structure
        """
        self.base_cluster = base_cluster
        self.bits = bits
        self.structure_symops = structure_symops
        self.sc_id = None
        self.sc_b_id = None
        #lazy generation of properties
        self._equiv = None
        self._symops = None
        self._bit_combos = None

    @property
    def equivalent_clusters(self):
        """
        Returns symmetrically equivalent clusters
        """
        if self._equiv:
            return self._equiv
        equiv = [self.base_cluster]
        for symop in self.structure_symops:
            new_sites = symop.operate_multi(self.base_cluster.sites)
            c = Cluster(new_sites, self.base_cluster.lattice)
            if c not in equiv:
                equiv.append(c)
        self._equiv = equiv
        if len(equiv) * len(self.cluster_symops) != len(self.structure_symops):
            raise SYMMETRY_ERROR
        return equiv

    @property
    def bit_combos(self):
        """
        List of arrays, each array is of symmetrically equivalent bit orderings
        """
        if self._bit_combos is not None:
            return self._bit_combos
        #get all the bit symmetry operations
        bit_ops = []
        for _, bitop in self.cluster_symops:
            if bitop not in bit_ops:
                bit_ops.append(bitop)
        all_combos = []
        for bit_combo in itertools.product(*self.bits):
            if bit_combo not in itertools.chain(*all_combos):
                bit_combo = np.array(bit_combo)
                new_bits = []
                for b_o in bit_ops:
                    new_bit = tuple(bit_combo[np.array(b_o)])
                    if new_bit not in new_bits:
                        new_bits.append(new_bit)
                all_combos.append(new_bits)
        self._bit_combos = [np.array(x, dtype=np.int) for x in all_combos] # this shouldn't be an array
        return self._bit_combos

    @property
    def cluster_symops(self):
        """
        Symmetry operations that map a cluster to its periodic image.
        each element is a tuple of (pymatgen.core.operations.Symop, mapping)
        where mapping is a tuple such that
        Symop.operate(sites) = sites[mapping] (after translation back to unit cell)
        """
        if self._symops:
            return self._symops
        self._symops = []
        for symop in self.structure_symops:
            new_sites = symop.operate_multi(self.base_cluster.sites)
            c = Cluster(new_sites, self.base_cluster.lattice)
            if self.base_cluster == c:
                c_sites = c.sites + np.round(self.base_cluster.centroid - c.centroid)
                self._symops.append((symop, tuple(coord_list_mapping(self.base_cluster.sites, c_sites, atol=SITE_TOL))))
        if len(self._symops) * self.multiplicity != len(self.structure_symops):
            raise SYMMETRY_ERROR
        return self._symops

    @property
    def max_radius(self):
        return self.base_cluster.max_radius

    @property
    def sites(self):
        return self.base_cluster.sites

    @property
    def multiplicity(self):
        return len(self.equivalent_clusters)

    def assign_ids(self, sc_id, sc_b_id, start_c_id):
        """
        Args:
            sc_id: symmetrized cluster id
            sc_b_id: start bit ordering id
            start_c_id: start cluster id

        Returns:
            next symmetrized cluster id, next bit ordering id, next cluster id
        """
        self.sc_id = sc_id
        self.sc_b_id = sc_b_id
        c_id = start_c_id
        for c in self.equivalent_clusters:
            c_id = c.assign_ids(c_id)
        return sc_id+1, sc_b_id + len(self.bit_combos), c_id

    def __eq__(self, other):
        #when performing SymmetrizedCluster in list, this ordering stops the equivalent structures from generating
        return self.base_cluster in other.equivalent_clusters

    def __str__(self):
        return "SymmetrizedCluster: id: {:<4} bit_id: {:<4} multiplicity: {:<4} symops: {:<4}" \
            " {}".format(self.sc_id, self.sc_b_id, self.multiplicity, len(self.cluster_symops), self.base_cluster)

    def __repr__(self):
        return self.__str__()


class ClusterExpansion(object):
    """
    Holds lists of SymmetrizedClusters and ClusterSupercells. This is probably the class you're looking for
    and should be instantiating. You probably want to generate from ClusterExpansion.from_radii, which will
    auto-generate the symmetrized clusters, unless you want more control over them.
    """

    def __init__(self, structure, expansion_structure, symops, clusters, \
                 sm_type='pmg_sm', ltol=0.2, stol=0.1, angle_tol=5,\
                 supercell_size='num_sites', use_ewald=False, use_inv_r=False, eta=None, basis = '01'):
        """
            Args:
                structure:
                    disordered structure to build a cluster expansion for. Typically the primitive cell
                radii:
                    dict of {cluster_size: max_radius}. Radii should be strictly decreasing.
                    Typically something like {2:5, 3:4}
                sm_type:
                    The structure matcher type that you wish to use in structure matching. Can choose from 
                    pymatgen default (pmg_sm), anion framework (an_frame)
                ltol, stol, angle_tol, supercell_size: parameters to pass through to the StructureMatcher, 
                    when sm_type == 'pmg_sm' or 'an_frame'
                    Structures that don't match to the primitive cell under these tolerances won't be included
                    in the expansion. Easiest option for supercell_size is usually to use a species that has a
                    constant amount per formula unit.
                use_ewald:
                    whether to calculate the ewald energy of each structure and use it as a feature. Typically
                    a good idea for ionic materials.
                use_inv_r:
                    experimental feature that allows fitting to arbitrary 1/r interactions between specie-site
                    combinations.
                eta:
                    parameter to override the EwaldSummation default eta. Usually only necessary if use_inv_r=True
                basis: 
                    Basis to use in cluster expansion. Currently can be 'ortho' or '01', plan to add 'chebyshev'.
            """

        if use_inv_r and eta is None:
            warn("Be careful, you might need to change eta to get properly "
                 "converged electrostatic energies. This isn't well tested")

        self.structure = structure
        self.expansion_structure = expansion_structure
        self.symops = symops

        # test that all the found symmetry operations map back to the input structure
        # otherwise you can get weird subset/superset bugs
        fc = self.structure.frac_coords
        for op in self.symops:
            if not is_coord_subset_pbc(op.operate_multi(fc), fc, SITE_TOL):
                raise SYMMETRY_ERROR

        self.supercell_size = supercell_size
        self.use_ewald = use_ewald
        self.eta = eta
        self.use_inv_r = use_inv_r
        
        self.sm_type=sm_type
        self.stol = stol
        self.ltol = ltol
        #self.vor_tol = vor_tol
        self.basis = basis

        if self.sm_type == 'pmg_sm' or self.sm_type == 'an_frame':
            self.angle_tol = angle_tol
            self.sm = StructureMatcher(primitive_cell=False,
                                   attempt_supercell=True,
                                   allow_subset=True,
                                   scale=True,
                                   supercell_size=self.supercell_size,
                                   comparator=OrderDisorderElementComparator(),
                                   stol=self.stol,
                                   ltol=self.ltol,
                                   angle_tol=self.angle_tol)
       # elif self.sm_type == 'an_dmap':
       #     print("Warning: Delaunay matcher only applicable for close packed anion framework!")
       #     try:
       #         from delaunay_matcher import DelaunayMatcher
       #         self.sm = DelauneyMatcher()
       #     # At leaset three methods are required in Delauney Matcher: match, mapping and supercell matrix finding.
       #     except:
       #         pass
       # I abandoned delaunay because it is not stable with respect to distortion.
        else:
            raise ValueError('Structure matcher not implemented!')

        self.clusters = clusters

        # assign the cluster ids
        n_clusters = 1
        n_bit_orderings = 1
        n_sclusters = 1
        for k in sorted(self.clusters.keys()):
            for y in self.clusters[k]:
                n_sclusters, n_bit_orderings, n_clusters = y.assign_ids(n_sclusters, n_bit_orderings, n_clusters)
        self.n_sclusters = n_sclusters
        self.n_clusters = n_clusters
        self.n_bit_orderings = n_bit_orderings
        self._supercells = {}


    @classmethod
    def from_radii(cls, structure, radii,\
                   sm_type = 'pmg_sm', ltol=0.2, stol=0.1, angle_tol=5,\
                   supercell_size='volume',use_ewald=False, use_inv_r=False, eta=None, basis = '01'):
        """
        Args:
            structure:
                disordered structure to build a cluster expansion for. Typically the primitive cell
            radii:
                dict of {cluster_size: max_radius}. Radii should be strictly decreasing.
                Typically something like {2:5, 3:4}
            ltol, stol, angle_tol, supercell_size: parameters to pass through to the StructureMatcher.
                Structures that don't match to the primitive cell under these tolerances won't be included
                in the expansion. Easiest option for supercell_size is usually to use a species that has a
                constant amount per formula unit.
            use_ewald:
                whether to calculate the ewald energy of each structure and use it as a feature. Typically
                a good idea for ionic materials.
            use_inv_r:
                experimental feature that allows fitting to arbitrary 1/r interactions between specie-site
                combinations.
            eta:
                parameter to override the EwaldSummation default eta. Usually only necessary if use_inv_r=True
        """
        symops = SpacegroupAnalyzer(structure).get_symmetry_operations()
        #get the sites to expand over
        sites_to_expand = [site for site in structure if site.species.num_atoms < 0.99 \
                            or len(site.species) > 1]
        expansion_structure = Structure.from_sites(sites_to_expand)
        clusters = cls._clusters_from_radii(expansion_structure, radii, symops)
        return cls(structure=structure, expansion_structure=expansion_structure, symops=symops, \
                   sm_type = sm_type, ltol=ltol, stol=stol, angle_tol=angle_tol,\
                   clusters=clusters, supercell_size=supercell_size, use_ewald=use_ewald, \
                   use_inv_r=use_inv_r,eta=eta, basis=basis)

    @classmethod
    def _clusters_from_radii(cls, expansion_structure, radii, symops):
        """
        Generates dictionary of size: [SymmetrizedCluster] given a dictionary of maximal cluster radii and symmetry
        operations to apply (not necessarily all the symmetries of the expansion_structure)
        """
        bits = get_bits(expansion_structure)
        nbits = np.array([len(b) - 1 for b in bits])
        # nbits = np.array([len(b) for b in bits])

        new_clusters = []
        clusters = {}
        for i, site in enumerate(expansion_structure):
            new_c = Cluster([site.frac_coords], expansion_structure.lattice)
            new_sc = SymmetrizedCluster(new_c, [np.arange(nbits[i])], symops)
            if new_sc not in new_clusters:
                new_clusters.append(new_sc)
        clusters[1] = sorted(new_clusters, key = lambda x: (np.round(x.max_radius,6), -x.multiplicity))

        all_neighbors = expansion_structure.lattice.get_points_in_sphere(expansion_structure.frac_coords, [0.5, 0.5, 0.5],
                                    max(radii.values()) + sum(expansion_structure.lattice.abc)/2)

        for size, radius in sorted(radii.items()):
            new_clusters = []
            for c in clusters[size-1]:
                if c.max_radius > radius:
                    continue
                for n in all_neighbors:
                    p = n[0]
                    if is_coord_subset([p], c.sites, atol=SITE_TOL):
                        continue
                    new_c = Cluster(np.concatenate([c.sites, [p]]), expansion_structure.lattice)
                    if new_c.max_radius > radius + 1e-8:
                        continue
                    new_sc = SymmetrizedCluster(new_c, c.bits + [np.arange(nbits[n[2]])], symops)
                    if new_sc not in new_clusters:
                        new_clusters.append(new_sc)
            clusters[size] = sorted(new_clusters, key = lambda x: (np.round(x.max_radius,6), -x.multiplicity))
        return clusters

    def supercell_matrix_from_structure(self, structure):
        if self.sm_type == 'pmg_sm': 
            sc_matrix = self.sm.get_supercell_matrix(structure, self.structure)
        elif self.sm_type == 'an_frame':
            prim_an_sites = [site for site in self.structure if Is_Anion_Site(site)]


            prim_an = Structure.from_sites(prim_an_sites)

            s_an_fracs = []
            s_an_sps = []
            latt = structure.lattice
            for site in structure:
                if Is_Anion_Site(site):
                    s_an_fracs.append(site.frac_coords)
                    s_an_sps.append(site.specie)

            scaling = ((len(s_an_sps)/len(prim_an))/(structure.volume/self.structure.volume))**(1/3.0)
            s_an_latt = Lattice.from_parameters(latt.a * scaling, latt.b * scaling, latt.c * scaling, \
                                                        latt.alpha, latt.beta, latt.gamma)
            structure_an = Structure(s_an_latt,s_an_sps,s_an_fracs,to_unit_cell =False, coords_are_cartesian=False)
            #print('Structure:',structure)
            #print('Structure_an:',structure_an)
            #print('Prim_an:',prim_an)

            sc_matrix = self.sm.get_supercell_matrix(structure_an, prim_an)
        else:
            raise ValueError("Structure Matcher type not implemented!")
        if sc_matrix is None:
            raise ValueError("Supercell couldn't be found")
        if np.linalg.det(sc_matrix) < 0:
            sc_matrix *= -1
        return sc_matrix

    def supercell_from_structure(self, structure):
        sc_matrix = self.supercell_matrix_from_structure(structure)
        return self.supercell_from_matrix(sc_matrix)


    def supercell_from_matrix(self, sc_matrix):
        sc_matrix = tuple(sorted(tuple(s) for s in sc_matrix))
        # print(sc_matrix)
        if sc_matrix in self._supercells:
            cs = self._supercells[sc_matrix]
        else:
            cs = ClusterSupercell(sc_matrix, self)
            self._supercells[sc_matrix] = cs
        return cs

#    def corr_from_external(self, structure, sc_matrix):
#        cs = self.supercell_from_matrix(self, sc_matrix)
#        return cs.corr_from_structure(structure)
#    This function will be integrated into supercell_from_structure.

    def corr_from_structure(self, structure):
        """
        Given a structure, determines which supercell to use,
        and gets the correlation vector
        """
        cs = self.supercell_from_structure(structure)
        return cs.corr_from_structure(structure)

    def base_energy(self, structure):
        sc = self.supercell_from_structure(structure)
        occu = sc.occu_from_structure(structure)
        be = sc._get_ewald_eci(occu)[0] #* sc.size
        return be

    def refine_structure(self, structure):
        sc_matrix = self.supercell_matrix_from_structure(structure)
        sc = self.supercell_from_matrix(sc_matrix)
        occu = sc.occu_from_structure(structure)
        return sc.structure_from_occu(occu)

#    def refine_structure_external(self, structure, sc_matrix):
#        cs = self.supercell_from_matrix(sc_matrix)
#        occu = cs.occu_from_structure(structure)
#        return cs.structure_from_occu(occu)

    def structure_energy(self, structure, ecis):
        cs = self.supercell_from_structure(structure)
        return cs.structure_energy(structure, ecis)

    @property
    def symmetrized_clusters(self):
        """
        Yields all symmetrized clusters
        """
        for k in sorted(self.clusters.keys()):
            for c in self.clusters[k]:
                # print(c)
                yield c

    def __str__(self):
        s = "ClusterBasis: {}\n".format(self.structure.composition)
        for k, v in self.clusters.iteritems():
            s += "    size: {}\n".format(k)
            for z in v:
                s += "    {}\n".format(z)
        return s

    @classmethod
    def from_dict(cls, d):
        symops = [SymmOp.from_dict(so) for so in d['symops']]
        clusters = {}
        for k, v in d['clusters_and_bits'].items():
            clusters[int(k)] = [SymmetrizedCluster(Cluster.from_dict(c[0]), c[1], symops) for c in v]
        return cls(structure=Structure.from_dict(d['structure']),
                   expansion_structure=Structure.from_dict(d['expansion_structure']),
                   clusters=clusters, symops=symops, 
                   sm_type = d['sm_type'] if 'sm_type' in d else 'pmg_sm',
                   ltol=d['ltol'], stol=d['stol'], angle_tol=d['angle_tol'],
                   #vor_tol = d['vor_tol'] if 'vor_tol' in d else 1e-3,
                   supercell_size=d['supercell_size'],
                   use_ewald=d['use_ewald'], use_inv_r=d['use_inv_r'],
                   eta=d['eta'],
                   basis=d['basis'] if 'basis' in d else '01')
    # Compatible with old datas

    def as_dict(self):
        c = {}
        for k, v in self.clusters.items():
            c[int(k)] = [(sc.base_cluster.as_dict(), [list(b) for b in sc.bits]) for sc in v]
        return {'structure': self.structure.as_dict(),
                'expansion_structure': self.expansion_structure.as_dict(),
                'symops': [so.as_dict() for so in self.symops],
                'clusters_and_bits': c,
                'sm_type': self.sm_type,
                'ltol': self.ltol,
                'stol': self.stol,
                'angle_tol': self.angle_tol,
                #'vor_tol': self.vor_tol,
                'supercell_size': self.supercell_size,
                'use_ewald': self.use_ewald,
                'use_inv_r': self.use_inv_r,
                'eta': self.eta,
                'basis':self.basis,
                '@module': self.__class__.__module__,
                '@class': self.__class__.__name__}


class ClusterSupercell(object):
    """
    Calculates correlation vectors on a specific supercell lattice.
    """
    def __init__(self, supercell_matrix, cluster_expansion):
        """
        Args:
            supercell matrix: array describing the supercell, e.g. [[1,0,0],[0,1,0],[0,0,1]]
            cluster_expansion: ClusterExpansion object
        """
        self.basis = cluster_expansion.basis
        self.supercell_matrix = np.array(supercell_matrix)
        self.prim_to_supercell = np.linalg.inv(self.supercell_matrix)
        self.cluster_expansion = cluster_expansion

        self.supercell = cluster_expansion.structure.copy()
        self.supercell.make_supercell(self.supercell_matrix)
        self.size = int(round(np.abs(np.linalg.det(self.supercell_matrix))))

        self.bits = get_bits(self.supercell)
        self.nbits = np.array([len(b)-1 for b in self.bits])
        self.fcoords = np.array(self.supercell.frac_coords)

        self._generate_mappings()

        if self.cluster_expansion.use_ewald:
            #lazily generate the difficult ewald parts
            self.ewald_inds = []
            ewald_sites = []
            for bits, s in zip(self.bits, self.supercell):
                inds = np.zeros(max(self.nbits) + 1) - 1
                for i, b in enumerate(bits):
                    if b == 'Vacancy':
                        #inds.append(-1)
                        continue
                    inds[i] = len(ewald_sites)
                    ewald_sites.append(PeriodicSite(b, s.frac_coords, s.lattice))
                self.ewald_inds.append(inds)
            self.ewald_inds = np.array(self.ewald_inds, dtype=np.int)
            self._ewald_structure = Structure.from_sites(ewald_sites)
            self._ewald_matrix = None
            self._partial_ems = None
            self._all_ewalds = None
            self._range = np.arange(len(self.nbits))
        else:
            self._all_ewalds = np.zeros((0, 0, 0), dtype=np.float)
            self.ewald_inds = np.zeros((0, 0), dtype=np.int)

    @property
    def all_ewalds(self):
        if self._all_ewalds is None:
            if self.cluster_expansion.use_ewald:
                ms = [self.ewald_matrix]
            else:
                ms = []
            if self.cluster_expansion.use_inv_r:
                ms += self.partial_ems
            self._all_ewalds = np.array(ms)
        return self._all_ewalds

    @property
    def ewald_matrix(self):
        if self._ewald_matrix is None:
            self._ewald = EwaldSummation(self._ewald_structure,
                                         eta=self.cluster_expansion.eta)
            self._ewald_matrix = self._ewald.total_energy_matrix
        return self._ewald_matrix

    @property
    def partial_ems(self):
        if self._partial_ems is None:
            # There seems to be an issue with SpacegroupAnalyzer such that making a supercell
            # can actually reduce the symmetry operations, so we're going to group the ewald
            # matrix by the equivalency in self.cluster_indices
            equiv_sc_inds = []
            ei = self.ewald_inds
            n_inds = len(self.ewald_matrix)
            for sc, inds in self.cluster_indices:
                # only want the point terms, which should be first
                if len(sc.bits) > 1:
                    break
                equiv = ei[inds[:, 0]] # inds is normally 2d, but these are point terms
                for inds in equiv.T:
                    if inds[0] > -1:
                        b = np.zeros(n_inds, dtype=np.int)
                        b[inds] = 1
                        equiv_sc_inds.append(b)

            self._partial_ems = []
            for x in equiv_sc_inds:
                mask = x[None, :] * x[:, None]
                self._partial_ems.append(self.ewald_matrix * mask)
            for x, y in itertools.combinations(equiv_sc_inds, r=2):
                mask = x[None, :] * y[:, None]
                mask = mask.T + mask # for the love of god don't use a += here, or you will forever regret it
                self._partial_ems.append(self.ewald_matrix * mask)
        return self._partial_ems

    def _get_ewald_occu(self, occu):
        i_inds = self.ewald_inds[self._range, occu]

        # instead of this line:
        #   i_inds = i_inds[i_inds != -1]
        # just make b_inds one longer than it needs to be and don't return the last value
        b_inds = np.zeros(len(self._ewald_structure) + 1, dtype=np.bool)
        b_inds[i_inds] = True
        return b_inds[:-1]

    def _get_ewald_eci(self, occu):
        inds = self._get_ewald_occu(occu)
        ecis = [np.sum(self.ewald_matrix[inds, :][:, inds]) / self.size]

        if self.cluster_expansion.use_inv_r:
            for m in self.partial_ems:
                ecis.append(np.sum(m[inds, :][:, inds]) / self.size)

        return np.array(ecis)

    def _get_ewald_diffs(self, new_occu, occu):
        inds = self._get_ewald_occu(occu)
        new_inds = self._get_ewald_occu(new_occu)
        diff = inds != new_inds
        both = inds & new_inds
        add = new_inds & diff
        sub = inds & diff

        ms = [self.ewald_matrix]
        if self.cluster_expansion.use_inv_r:
            ms += self.partial_ems

        diffs = []
        for m in ms:
            ma = m[add]
            ms = m[sub]
            v = np.sum(ma[:, add]) - np.sum(ms[:, sub]) + \
                (np.sum(ma[:, both]) - np.sum(ms[:, both])) * 2

            diffs.append(v / self.size)

        return diffs

    def _generate_mappings(self):
        """
        Find all the supercell indices associated with each cluster
        """
        ts = lattice_points_in_supercell(self.supercell_matrix)
        self.cluster_indices = []
        self.clusters_by_sites = defaultdict(list)
        for sc in self.cluster_expansion.symmetrized_clusters:
            prim_fcoords = np.array([c.sites for c in sc.equivalent_clusters])
            fcoords = np.dot(prim_fcoords, self.prim_to_supercell)
            #tcoords contains all the coordinates of the symmetrically equivalent clusters
            #the indices are: [equivalent cluster (primitive cell), translational image, index of site in cluster, coordinate index]
            tcoords = fcoords[:, None, :, :] + ts[None, :, None, :]
            tcs = tcoords.shape
            inds = coord_list_mapping_pbc(tcoords.reshape((-1, 3)),
                                self.fcoords, atol=SITE_TOL).reshape((tcs[0] * tcs[1], tcs[2]))
            self.cluster_indices.append((sc, inds))
            #symmetrized cluster, 2d array of index groups that correspond to the cluster
            #the 2d array may have some duplicates. This is due to symetrically equivalent
            #groups being matched to the same sites (eg in simply cubic all 6 nn interactions
            #will all be [0, 0] indices. This multiplicity disappears as supercell size
            #increases, so I haven't implemented a more efficient method

            # now we store the symmetrized clusters grouped by site index in the supercell,
            # to be used by delta_corr. We also store a reduced index array, where only the
            # rows with the site index are stored. The ratio is needed because the correlations
            # are averages over the full inds array.
            for site_index in np.unique(inds):
                in_inds = np.any(inds == site_index, axis=-1)
                ratio = len(inds) / np.sum(in_inds)
                self.clusters_by_sites[site_index].append((sc.bit_combos, sc.sc_b_id, inds[in_inds], ratio))

    def structure_from_occu(self, occu):
        sites = []
        for b, o, s in zip(self.bits, occu, self.supercell):
            if b[o] != 'Vacancy':
                sites.append(PeriodicSite(b[o], s.frac_coords, self.supercell.lattice))
        return Structure.from_sites(sites)

    def corr_from_occupancy(self, occu):
        """
        Each entry in the correlation vector correspond to a particular symetrically distinct bit ordering.
        """
        corr = np.zeros(self.cluster_expansion.n_bit_orderings)
        corr[0] = 1 #zero point cluster
        occu = np.array(occu)
        if self.basis == 'ortho':
            for sc, inds in self.cluster_indices: # iteration over different shape cluster (ECIs is different !!!)
                Ms = calc_M(sc.bits) +1
                c_occu = occu[inds]
                for i, combos in enumerate(sc.bit_combos):
                    cf = cluster_function_ortho(c_occu, combos, Ms)
                    if np.abs(cf) < 1e-10: # avoid numerical instability
                        corr[sc.sc_b_id + i] = 0
                    else:
                        corr[sc.sc_b_id + i] = cf
        elif self.basis == '01':
            for sc, inds in self.cluster_indices:
                c_occu = occu[inds]
                for i, bits in enumerate(sc.bit_combos):
                    p = np.all(c_occu[None, :, :] == bits[:, None, :], axis=-1)
                    corr[sc.sc_b_id + i] = np.average(p) 
        else:
            raise ValueError('Basis not implemented!')

        if self.cluster_expansion.use_ewald:
                corr = np.concatenate([corr, self._get_ewald_eci(occu)])
        return corr

    def occu_from_structure(self, structure, return_mapping=False):
        """
        Calculates the correlation vector. Structure must be on this supercell
        """
        #calculate mapping to supercell
        sm_no_sc = StructureMatcher(primitive_cell=False,
                                    attempt_supercell=False,
                                    allow_subset=True,
                                    comparator=OrderDisorderElementComparator(),
                                    supercell_size=self.cluster_expansion.supercell_size,
                                    scale=True,
                                    ltol=self.cluster_expansion.ltol,
                                    stol=self.cluster_expansion.stol,
                                    angle_tol=self.cluster_expansion.angle_tol)

        #print('sc:\n',self.supercell_matrix,\
        #      '\nstr:\n',self.cluster_expansion.supercell_matrix_from_structure(structure))
        mapping = sm_no_sc.get_mapping(self.supercell, structure)

        if mapping is None:
            raise ValueError('Structure cannot be mapped to this supercell. Structure:{}\nSupercell:{}'.format(structure,self.supercell))

        mapping = mapping.tolist()
        #cs.supercell[mapping] = structure
        occu = np.zeros(len(self.supercell), dtype=np.int)
        for i, bit in enumerate(self.bits):
            # print('i=',i)
            # print('bit=', bit)
            #rather than starting with all vacancies and looping
            #only over mapping, explicitly loop over everything to
            #catch vacancies on improper sites
            if i in mapping:
                sp = str(structure[mapping.index(i)].specie)
            else:
                sp = 'Vacancy'
            occu[i] = bit.index(sp)
        if not return_mapping:
            return occu
        else:
            return occu, mapping

    def corr_from_structure(self, structure):
        occu = self.occu_from_structure(structure)
        return self.corr_from_occupancy(occu)

    def structure_energy(self, structure, ecis):
        return np.dot(self.corr_from_structure(structure), ecis) * self.size

    def occu_energy(self, occu, ecis):
        return np.dot(self.corr_from_occupancy(occu), ecis) * self.size

    def delta_corr(self, flips, occu, debug=False):
        """
        Returns the *change* in the correlation vector from applying a list of flips.
        Flips is a list of (site, new_bit) tuples.
        """
        new_occu = occu.copy()

        delta_corr = np.zeros(self.cluster_expansion.n_bit_orderings + len(self.all_ewalds))
        for f in flips:
            new_occu_f = new_occu.copy()
            new_occu_f[f[0]] = f[1]
            if self.basis == '01':
                delta_corr += delta_corr_single_flip(new_occu_f, new_occu,\
                                 self.cluster_expansion.n_bit_orderings,\
                                 self.clusters_by_sites[f[0]], f[0], f[1],self.all_ewalds,\
                                 self.ewald_inds, self.size)
            new_occu = new_occu_f

        if debug:
            e = self.corr_from_occupancy(new_occu) - self.corr_from_occupancy(occu)
            assert np.allclose(delta_corr, e)

        if self.basis != '01':
            delta_corr = self.corr_from_occupancy(new_occu) - self.corr_from_occupancy(occu)
        return delta_corr, new_occu
