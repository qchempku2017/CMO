from pyhull.voronoi import VoronoiTess
from pymatgen.analysis.structure_analyzer import solid_angle

import abc
import itertools
import logging
import math
import numpy as np

class CoordData(object):
    """
    Computes and stores the data used to generate the sense columns
    (such as vector and distance arrays). Since many of the columns
    require the same data, it is much faster to just calculate it
    once.
    """
    def __init__(self, coords, structure,
                 cutoff = 10, ntol = 1e-8):
        """
        Args:
            coords:
                numpy array of shape [site, timestep, axis]
            structure:
                structure, used for its lattice and occupations
            cutoff:
                radial distance cutoff for the calculations (A)
            ntol:
                numeric tolerance. Distances less than this value are
                set to infinity
        """
        self.coords = coords
        self.structure = structure
        self.cutoff = cutoff
        self.ntol = ntol
        self._vectors = None
        self._distances = None
        self._positions = None
        self._voronoi_bond_orders = None

    @property
    def vectors(self):
        '''
        returns:
            np array of shape [atomi, atomj, image, timestep, axis]
        '''
        if self._vectors is None:
            self._set_vectors_and_distances()
        return self._vectors

    @property
    def distances(self):
        '''
        returns:
            np array of shape [atomi, atomj, image, timestep]
            distances of sites on top of each other are set to np.inf
        '''
        if self._distances is None:
            self._set_vectors_and_distances()
        return self._distances

    @property
    def positions(self):
        '''
        positions of all atoms within the supercell generated according
        to the cutoff
        returns:
            np array of shape [atomj, image, timestep, axis]
        '''
        if self._positions is None:
            self._set_positions()
        return self._positions

    @property
    def voronoi_bond_orders(self):
        '''
        positions of all atoms within the supercell generated according
        to the cutoff
        returns:
            np array of shape [atomi, atomj, image, timestep]
        '''
        if self._voronoi_bond_orders is None:
            self._set_voronoi_bond_orders()
        return self._voronoi_bond_orders

    def _set_voronoi_bond_orders(self):
        '''
        approximate bond order, from purely geometric voronoi solid angles'
        returns
        def bond_order(cd):
        '''
        p = self.positions
        bond_orders = []
        o_index = p.shape[1]/2
        n_images = p.shape[1]
        for i in range(p.shape[2]):
            #[atomj * image, axis]
            vt = VoronoiTess(p[:, :, i].reshape((-1,3)))
            bo = np.zeros([p.shape[0] * p.shape[1]] * 2)
            vts = np.array(vt.vertices)
            for (k1, k2), v in vt.ridges.items():
                if (k1 % n_images != o_index) and (k2 % n_images != o_index):
                    continue
                if -10.101 in vts[v]:
                    continue
                val = solid_angle(vt.points[k1], vts[v])
                bo[k1, k2] = val
                bo[k2,k1] = val
            nshape = (p.shape[0], p.shape[0], p.shape[1], 1)
            bond_orders.append(bo[o_index::n_images].reshape(nshape))
        self._voronoi_bond_orders = np.concatenate(bond_orders, -1)

    def _set_positions(self):
        logging.info('calculating positions up to cutoff')
        l = self.structure.lattice
        recp_len = np.array(l.reciprocal_lattice.abc)
        i = np.floor(self.cutoff * recp_len / (2 * math.pi)) + 1
        offsets = np.mgrid[-i[0]:i[0]+1, -i[1]:i[1]+1, -i[2]:i[2]+1].T
        offsets = np.reshape(offsets, (-1, 3))
        #[image, axis]
        cart_offsets = l.get_cartesian_coords(offsets)
        #[atom, image, timestep, axis]
        self._positions = self.coords[:, None, :, :] + \
                          cart_offsets[None, :, None, :]

    def _set_vectors_and_distances(self):
        #[atomi, atomj, image, timestep, axis]
        vectors = self.coords[:, None, None , :, :] - \
                  self.positions[None, :, :, :, :]
        #[atomi, atomj, image, timestep]
        distances = np.sum(vectors ** 2, axis = -1) ** 0.5

        #set the atoms that are on top of each other to infinite distance
        #(but vectors to zero to avoid creating np.nan when we divide)
        vectors[distances > self.cutoff] = [0, 0, 0]
        distances[distances > self.cutoff] = np.inf
        vectors[distances < self.ntol] = [0, 0, 0]
        distances[distances < self.ntol] = np.inf
        #[atomi, atomj, image, timestep, axis]
        self._vectors = vectors
        #[atomi, atomj, image, timestep]
        self._distances = distances


class CoordSenseColumns(object):
    '''
    Abstract class to generate columns of the sense matrix using
    coord data. The abstract class defines the grouping of interactions,
    as these are identical for many different types of sense column.
    '''
    @abc.abstractmethod
    def make(self, cd):
        '''
        given coord data, returns a np array of shape [observation,
        grouping(such as a pair of species)]
        '''
        pass

    @abc.abstractmethod
    def names(self, cd):
        '''
        given coord data, return a list of names for the columns
        '''

    def pair_grouping(self, factors, cd):
        '''
        given a structure with species A, B, C, and factors of the shape
        (atomi, atomj, image, timestep, coord), creates an array of shape
        (atomi * timestep * coord, nspecies ** 2), where the first dimension
        is equal to the number of observations, and the second is the number
        of different types of factors given by the SenseColumns object. The
        orders are given by looping over atomi, then timestep, then coord
        for the rows, and by AA, AB, AC, BA, BB, BC... assuming that A,B,C
        appear in that order when sorted by np.unique
        '''
        sps, inv = np.unique(cd.structure.species, return_inverse = True)
        grps = [np.where(i == inv)[0] for i in range(len(sps))]
        fs = factors.shape
        #(atomi, timestep, coord, nspecies)
        X = np.zeros((fs[0], fs[3], fs[4], len(sps)))
        #loop over the groups
        #(atomi, atomj, timestep, coord)
        image_sum = np.sum(factors, axis = 2)
        for n, grp in enumerate(grps):
            X[:, :, :, n] = np.sum(image_sum[:, grp], axis = 1)
        #(atomi, timestep, coord, nspecies, nspecies)
        Y = np.zeros((fs[0], fs[3], fs[4], len(sps), len(sps)))
        for n, grp in enumerate(grps):
            Y[grp, :, :, n, :] = X[grp, :, :, :]
        A = np.reshape(Y, Y.shape[:3] + (-1,))
        A = np.reshape(A, (-1,) + A.shape[3:])
        return A

    def triplet_grouping(self, factors, cd):
        '''
        given a structure with species A, B, C, and factors of the shape
        [atomi, atomj, imagej, atomk, imagek, timestep, coord], creates an array of shape
        (atomi * timestep * coord, nspecies ** 3), where the first dimension
        is equal to the number of observations, and the second is the number
        of different types of factors given by the SenseColumns object. The
        orders are given by looping over atomi, then timestep, then coord
        for the rows, and by AAA, AAB, AAC, ABA, ABB, ABC... assuming that A,B,C
        appear in that order when sorted by np.unique
        '''
        sps, inv = np.unique(cd.structure.species, return_inverse = True)
        grps = [np.where(i == inv)[0] for i in range(len(sps))]
        fs = factors.shape


        #(atomi, atomj, atomk, timestep, coord)
        image_sum = np.sum(factors, axis = 4)
        image_sum = np.sum(image_sum, axis = 2)

        #(atomi, atomj, timestep, coord, nspecies)
        X = np.zeros((fs[0], fs[1], fs[5], fs[6], len(sps)))

        #sum over atom k
        for n, grp in enumerate(grps):
            X[:, :, :, :, n] = np.sum(image_sum[:, :, grp], axis = 2)

        #sum over atom j
        #(atomi, timestep, coord, nspecies, nspecies)
        Y = np.zeros((fs[0], fs[5], fs[6], len(sps), len(sps)))
        for n, grp in enumerate(grps):
            Y[:, :, :, :, n] = np.sum(X[:, grp], axis = 1)

        #(atomi, timestep, coord, nspecies, nspecies, nspecies)
        Z = np.zeros((fs[0], fs[5], fs[6], len(sps), len(sps), len(sps)))
        for n, grp in enumerate(grps):
            Z[grp, :, :, n, :, :] = Y[grp, :, :, :, :]

        A = np.reshape(Z, Z.shape[:3] + (-1,))
        A = np.reshape(A, (-1,) + A.shape[3:])

        return A



    def pair_names(self, cd):
        '''
        returns a list of the column names for columns returned by the
        pair_grouping function
        '''
        sps = np.unique(cd.structure.species)
        n = []
        for spi, spj in itertools.product(sps, repeat = 2):
            n.append((str(self), spi, spj))
        return n


class RInvColumns(CoordSenseColumns):
    '''
    Makes sense columns of the form r_hat/||r||**n
    '''
    def __init__(self, n):
        self.n = n

    def make(self, cd):
        #r/||r|| is a unit vector in the distance direction. We are trying to fit
        # F = r/||r|| * A / ||r||**n = A * r / ||r||**(n+1)
        factors = cd.vectors / (cd.distances[:, :, :, :, None] ** (self.n + 1))
        return self.pair_grouping(factors, cd)

    def names(self, sd):
        return self.pair_names(sd)

    def __str__(self):
        return "{}, n = {}".format(self.__class__.__name__,
                                   self.n)


class ExpColumns(CoordSenseColumns):
    '''
    Makes sense columns of the form r_hat*exp(-b*||r||)
    '''
    def __init__(self, b):
        self.b = b

    def make(self, cd):
        r_hat = cd.vectors / cd.distances[:, :, :, :, None]
        factors = r_hat * np.exp(-self.b * cd.distances[:, :, :, :, None])
        return self.pair_grouping(factors, cd)

    def names(self, sd):
        return self.pair_names(sd)

    def __str__(self):
        return "{}, b = {}".format(self.__class__.__name__,
                                   self.b)


class BORInvColumns(CoordSenseColumns):
    '''
    Makes sense columns of the form B*r_hat/||r||**n
    '''
    def __init__(self, n):
        self.n = n

    def make(self, cd):
        #r/||r|| is a unit vector in the distance direction. We are trying to fit
        # F = r/||r|| * A / ||r||**n = A * r / ||r||**(n+1)
        factors = cd.voronoi_bond_orders[:, :, :, :, None] * cd.vectors / (cd.distances[:, :, :, :, None] ** (self.n + 1))
        return self.pair_grouping(factors, cd)

    def names(self, sd):
        return self.pair_names(sd)

    def __str__(self):
        return "{}, n = {}".format(self.__class__.__name__,
                                   self.n)


class BOExpColumns(CoordSenseColumns):
    '''
    Makes sense columns of the form B*r_hat*exp(-b*||r||)
    '''
    def __init__(self, b):
        self.b = b

    def make(self, cd):
        r_hat = cd.voronoi_bond_orders[:, :, :, :, None] * cd.vectors / cd.distances[:, :, :, :, None]
        factors = r_hat * np.exp(-self.b * cd.distances[:, :, :, :, None])
        return self.pair_grouping(factors, cd)

    def names(self, sd):
        return self.pair_names(sd)

    def __str__(self):
        return "{}, b = {}".format(self.__class__.__name__,
                                   self.b)


class AxilrodTellerColumns(CoordSenseColumns):

    def _calc_U(self, cd, delta = np.array([0,0,0])):
        logging.info('generating Us for shape {}'.format(cd.positions.shape))
        #[site, timestep, axis]
        c = cd.coords + delta[None, None, :]
        #[atomj, image, timestep, axis]
        p = cd.positions
        #[atomi, atomj, imagej, atomk, imagek, timestep]
        i = c[:, None, None, None, None, :, :]
        j = p[None, :, :, None, None, :, :]
        k = p[None, None, None, :, :, :, :]
        cosabc = np.zeros((p.shape[0], p.shape[0], p.shape[1], p.shape[0], p.shape[1], p.shape[2]))
        print ('allocated')
        for n in range(cosabc.shape[0]):
            print ('doing cos {}').format(n)
            cosabc[n] = np.sum((j-i[n])*(k-i[n]), axis = -1)
            cosabc[n] *= np.sum((i[n]-j)*(k-j), axis = -1)[0]
            cosabc[n] *= np.sum((i[n]-k)*(j-k), axis = -1)[0]
        #cosabc = np.sum((j-i)*(k-i), axis = -1) * np.sum((i-j)*(k-j), axis = -1) * np.sum((i-k)*(j-k), axis = -1)
        print ('done cosabc')
        a3b3c3 = np.sum((i-j) ** 2, axis = -1) ** (3/2) * np.sum((i-k) ** 2, axis = -1) ** (3/2) * np.sum((j-k) ** 2, axis = -1) ** (3/2)
        print ('updating inf')
        a3b3c3[a3b3c3 < cd.ntol] = np.inf
        return (1 + 3 * cosabc / a3b3c3) / a3b3c3

    def make(self, cd):
        U = self._calc_U(cd)
        U_deltas = []
        for delta in (np.array([0.01,0,0]), np.array([0,0.01,0]), np.array([0,0,0.01])):
            U_deltas.append(self._calc_U(cd, delta)[:, :, :, :, :, :, None])
        #[atomi, atomj, imagej, atomk, imagek, timestep, axis]
        U_deltas = np.concatenate(U_deltas, -1)
        return self.triplet_grouping((U_deltas - U[:, :, :, :, :, :, None]) * 100, cd)


    def names(self, sd):
        return
        return self.pair_names(sd)

    def __str__(self):
        return "{}".format(self.__class__.__name__)


class GaussianTripletColumns():
    def __init__(self, r_range = np.arange(0, 7, 0.5)):
        self.rrange = r_range

    def make(self, cd):
        '''
        returns factor of shape
        (atomi * timestep * coord, (nspecies + 2) choose 3 * len(r_range) ** 3)
        '''
        max_dist = max(self.rrange) + 1
        #[atomi, atomj, image, timestep]
        d = cd.distances
        #[atomj, image, timestep, axis]
        p = cd.positions

        sps, inv = np.unique(cd.structure.species, return_inverse = True)
        sp_indices = {}
        for sp in sps:
            sp_indices[sp] = np.where(np.array(cd.structure.species) == sp)[0]

        sps_combos = np.array(list(itertools.combinations_with_replacement(sps, r = 3)))

        s0 = d.shape[0] * d.shape[3] * 3
        s1 = len(sps_combos) * len(self.rrange) ** 3
        factors = np.zeros((s0, s1))
        nf = 0
        for timestep in range(d.shape[-1]):
            for a in range(d.shape[0]):
                #[atomj, image]
                for pair in np.array(list(itertools.combinations_with_replacement(sps, r = 2))):
                    #print a
                    sp_a = cd.structure[a].specie
                    #print triplet
                    b_ind = sp_indices[pair[0]]
                    c_ind = sp_indices[pair[1]]

                    #neighbors (positions)
                    i = p[a, p.shape[1]/2, timestep, :]
                    j = p[b_ind, :, timestep][d[a, b_ind, :, timestep] < max_dist]
                    k = p[c_ind, :, timestep][d[a, c_ind, :, timestep] < max_dist]

                    #we know that all a-b and a-c cutoffs are ok, but still need to check b-c
                    #shape = (j, k, coord)
                    I = (j[:, None, :] - k[None, :, :])
                    J = (k[None, :, :] - i[None, None, :])
                    K = (j[:, None, :] - i[None, None, :])

                    #print I.shape, J.shape, K.shape

                    #shape = (j, k, l, m, n)
                    I_l = np.sum(I[:, :, None, None, None, :] ** 2, axis = -1) - self.rrange[None, None, :, None, None]
                    J_m = np.sum(J[:, :, None, None, None, :] ** 2, axis = -1) - self.rrange[None, None, None, :, None]
                    K_n = np.sum(K[:, :, None, None, None, :] ** 2, axis = -1) - self.rrange[None, None, None, None, :]

                    #shape = (j, k, l, m, n)
                    expval = np.exp(-(I_l ** 2 + J_m ** 2 + K_n ** 2) / 2)
                    #get rid of contributions with I == 0
                    #[j, k]
                    Idist = np.sum(I ** 2, axis = -1)

                    ind = Idist < cd.ntol
                    if np.prod(ind.shape):
                        expval[ind] = 0


                    #[j, k, l, m, n]
                    dU_dJ = -J_m * expval
                    dU_dK = -K_n * expval

                    #shape = (j, k, coord)
                    dJ_di = J / np.sum(J ** 2, axis = -1)[:, :, None]
                    dK_di = K / np.sum(K ** 2, axis = -1)[:, :, None]

                    F = dU_dJ[:, :, :, :, :, None] * dJ_di[:, :, None, None, None, :] + dU_dK[:, :, :, :, :, None] * dK_di[:, :, None, None, None, :]

                    #[l, m, n, coord]
                    F = np.sum(np.sum(F, axis = 1), axis = 0)

                    for z, sps_combo in enumerate(sps_combos):
                        if set([t for t in pair] + [sp_a]) == set(sps_combo):
                            y_ind = z
                            break

                    triplet = list(sps_combos[z])

                    #print triplet, sp_a

                    i_ind = triplet.index(sp_a)
                    j_ind = triplet.index(pair[0])

                    if j_ind == i_ind:
                        j_ind += 1

                    k_ind = set([0,1,2]).difference(set([i_ind, j_ind])).pop()

                    actual = np.array(['l', 'm', 'n'])
                    desired = np.copy(actual)
                    desired[i_ind] = 'l'
                    desired[j_ind] = 'm'
                    desired[k_ind] = 'n'

                    if i_ind != 0:
                        F = np.swapaxes(F, 0, i_ind)
                        t = actual[0]
                        actual[0] = actual[i_ind]
                        actual[i_ind] = t


                    if (desired != actual).any():
                        F = np.swapaxes(F, j_ind, k_ind)
                        t = actual[j_ind]
                        actual[j_ind] = actual[k_ind]
                        actual[k_ind] = t
                    '''
                    print 'final', actual, desired

                    assert (actual == desired).all()
                    '''

                    F = np.rollaxis(F, -1).reshape((3, -1))

                    factors[nf:nf+3, y_ind * len(self.rrange) ** 3:(y_ind+1) * len(self.rrange) ** 3] = F

                nf += 3
        return factors

class SenseMatrix(object):
    '''
    Makes a full sensing matrix given a list of column types
    '''
    def __init__(self, sense_columns):
        self.sense_columns = sense_columns

    def make(self, sd, make_underdetermined = False):
        d = []
        for sc in self.sense_columns:
            logging.info('generating column {}'.format(sc))
            d.append(sc.make(sd))

        A = np.concatenate(d, axis = 1)

        if make_underdetermined:
            weight = np.average(A ** 2) ** 0.5
            A = np.concatenate([A, np.eye(len(d[-1])) * weight], axis = 1)

        return A

    def names(self, sd):
        '''
        returns a descriptor for each column
        '''
        n = []
        for sc in self.sense_columns:
            n.extend(sc.names(sd))
        return n
