from __future__ import division

"""
2018-09-10 - Notes from Daniil Kitchaev (dkitch@alum.mit.edu)
This version of eci_fit.py is heavily edited by Daniil Kitchaev to fiddle around with the gs_preservation fit routine,
implementing the algorithm described by Wenxuan Huang in npjCompMat (2017) and fixing numerical issues. The general
structure of eci_fit is based on code originally written by Will Richards, available in pyabinitio.

See specific notes in the gs_preserve_fit function.

The code runs and seems to give results that are reasonable with the underlying math, but it has NOT BEEN TESTED
EXTENSIVELY ENOUGH FOR PRODUCTION and there are I think serious problems with the underlying algorithm that need to
be resolved before the gs_preservation routine will be of use for practical work.

2019-09-24 - Notes from Fengyu Xie (fengyu_xie@berkeley.edu)
In this version I added a new fitting method to implement the l1/l0 method by Wenxuan Huang in 2018. Gurobi licence required!

2019-10-06 - Notes from Fengyu Xie
In this version I added principal component regression as a testing feature. Do basis orthogonalization and fit at the same
time.
"""

import os
import warnings

from collections import defaultdict
from itertools import chain

from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen import Composition, Structure
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry

import sys
cwd=os.path.dirname(os.path.realpath(__file__))
sys.path.append(cwd)
#This is to tell python where to find module compressive and ce. We will no longer use pyabinitio equivalence.
from compressive.bregman import split_bregman #split bregman not needed if using gs_preserve or cvxopt_l1
from ce import ClusterExpansion

from matplotlib import pylab as plt

import logging
import numpy as np
import numpy.linalg as la
import json
from copy import deepcopy

from gurobi import *

def _pd(structures, energies, ce):
    """
    Generate a phase diagram with the structures and energies
    """
    entries = []

    for s, e in zip(structures, energies):
        entries.append(PDEntry(s.composition.element_composition, e))

    max_e = max(entries, key=lambda e: e.energy_per_atom).energy_per_atom + 1000
    for el in ce.structure.composition.keys():
        entries.append(PDEntry(Composition({el: 1}).element_composition, max_e))

    return PhaseDiagram(entries)


def _energies_above_composition(structures, energies):
    min_e = defaultdict(lambda: np.inf)
    for s, e in zip(structures, energies):
        comp = s.composition.reduced_composition
        if e / len(s) < min_e[comp]:
            min_e[comp] = e / len(s)
    e_above = []
    for s, e in zip(structures, energies):
        comp = s.composition.reduced_composition
        e_above.append(e / len(s) - min_e[comp])
    return np.array(e_above)


def _energies_above_hull(pd, structures, energies):
    e_above_hull = []
    for s, e in zip(structures, energies):
        e_above_hull.append(pd.get_e_above_hull(PDEntry(s.composition.element_composition, e)))
    return np.array(e_above_hull)

class LightFittedEciGenerator(object):

    def __init__(self, cluster_expansion, ecis):
        """
        :param cluster_expansion: cluster expansion used to fit original EciGenerator
        :param ecis: already fitted list of ECIs from an EciGenerator
        """
        self.ce = cluster_expansion
        self.ecis = ecis

    def structure_energy(self, structure):
        return self.ce.structure_energy(structure, self.ecis)

    @classmethod
    def from_eg(cls, eg):
        """
        Make a LightFittedEciGenerator from a fitted EciGenerator object
        """
        return cls(cluster_expansion=eg.ce, ecis=eg.ecis)

    @classmethod
    def from_dict(cls, d):
        return cls(cluster_expansion=ClusterExpansion.from_dict(d['cluster_expansion']), ecis=d.get('ecis'))

    def as_dict(self):
        return {'cluster_expansion': self.ce.as_dict(),
                'ecis': self.ecis.tolist(),
                '@module': self.__class__.__module__,
                '@class': self.__class__.__name__}


class EciGenerator(object):

    _pd_input = None
    _pd_ce = None
    _e_above_hull_input = None
    _e_above_hull_ce = None

    def __init__(self, cluster_expansion, structures, energies, weights,
                 mu=None, max_dielectric=None, max_ewald=None,
                 solver='cvxopt_l1', supercell_matrices=None,
                 ecis=None, feature_matrix=None):
        """
        Fit ECIs to a cluster expansion. This init function takes in all possible arguments,
        but its much simpler to use one of the factory classmethods below,
        e.g. EciGenerator.unweighted

        Args:
            cluster_expansion: A ClusterExpansion object
            structures: list of Structure objects
            energies: list of total (non-normalized) energies
            weights: list of weights for the optimization.
            mu: mu to use in the split_bregman, otherwise optimal value is calculated
                by CV optimization;
                When using L1 opt, mu is one single float number
                When using L1/L0, mu = [mu0,mu1]
                When using pcr, mu = pcr singular value cutoff threshold.
            max_dielectric: constrain the dielectric constant to be positive and below the
                supplied value (note that this is also affected by whether the primitive
                cell is the correct size)
            max_ewald: filter the input structures to only use those with low electrostatic
                energies (no large charge separation in cell). This energy is referenced to the lowest
                value at that composition. Note that this is before the division by the relative dielectric
                 constant and is per primitive cell in the cluster exapnsion -- 1.5 eV/atom seems to be a
                 reasonable value for dielectric constants around 10.
            solver: solver, current options are cvxopt_l1, bregman_l1, gs_preserve
            supercell_matrices, ecis, feature_matrix: Options used by from_dict to speed up
                initialization from json. It shouldn't ever be necessary to specify these.
        """
        self.ce = cluster_expansion
        self.solver = solver
        self.max_dielectric = max_dielectric
        self.max_ewald = max_ewald
        
        self.items = []
        supercell_matrices = supercell_matrices or [None] * len(structures)
        fm_rows = feature_matrix or [None] * len(structures)
        for s, e, m, w, fm_row in zip(structures, energies, supercell_matrices, weights, fm_rows):
            try:
                # TODO: Caching m and sc speeds up load times a lot
                if m is None:
                    m = self.ce.supercell_matrix_from_structure(s)
                sc = self.ce.supercell_from_matrix(m)
                if fm_row is None:
                    fm_row = self.ce.corr_from_structure(s)
            except Exception:
                print('Unable to match {} with energy {} to supercell'.format(s.composition, e))
                logging.debug('Unable to match {} with energy {} to supercell'.format(s.composition, e))
                if self.ce.supercell_size not in ['volume', 'num_sites', 'num_atoms'] \
                        and s.composition[self.ce.supercell_size] == 0:
                    logging.warn('Specie {} not in {}'.format(self.ce.supercell_size, s.composition))
                continue
            self.items.append({'structure': s,
                               'energy': e,
                               'weight': w,
                               'supercell': sc,
                               'features': fm_row,
                               'size': sc.size}) 
        # TODO: Careful about sc.size - depending on the prim may be inconsistent with E per atom vs per prim

        if self.ce.use_ewald and self.max_ewald is not None and False:
            if self.ce.use_inv_r:
                raise NotImplementedError('Can not use inv_r with max ewald yet')

            min_e = defaultdict(lambda: np.inf)
            for i in self.items:
                c = i['structure'].composition.reduced_composition
                if i['features'][-1] < min_e[c]:
                    min_e[c] = i['features'][-1]

            items = []
            for i in self.items:
                r_e = i['features'][-1] - min_e[i['structure'].composition.reduced_composition]
                if r_e > self.max_ewald:
                    logging.debug('Skipping {} with energy {}, ewald energy is {}'
                                  ''.format(i['structure'].composition, i['energy'], r_e))
                else:
                    items.append(i)
            self.items = items

        logging.info("Matched {} of {} structures".format(len(self.items),
                                                          len(structures)))


        # do least squares for comparison. Not very useful. Muted, added as a fitting method instead.
        # x = np.linalg.lstsq(self.feature_matrix, self.normalized_energies,rcond=None)[0]
        # ls_err = np.dot(self.feature_matrix, x) - self.normalized_energies
        # logging.info('least squares rmse: {}'.format(np.sqrt(np.sum(ls_err ** 2) / len(self.feature_matrix))))

        # calculate optimum mu
        self.mu = mu 
        self.ecis = np.array(ecis) if ecis is not None else None
        self._cv = None
        self._rmse = None

        if self.mu is None or self.ecis is None:
            logging.warn('Model not generated yet. Call self.generate().')   
            #print('Model not generated yet. Call self.generate().')   


#### Important update: ecis no longer generated in __init__. Call this ####
#### to generate ecis ####
    def generate(self, min_mu = -1, max_mu = 6):
        if self.mu is None:
            print('Finding optimum mu')
            self.mu = self.get_optimum_mu(self.feature_matrix, self.normalized_energies, self.weights, min_mu = min_mu, max_mu = max_mu)
            # Remeber to change the mu searching range for pcr solver
            # Recommended PCR min_mu = -5, max_mu = 0
            logging.info("Opt mu: {}".format(self.mu))              
        # actually fit the cluster expansion
        else:
            print('By default, using existing mu')

        if self.ecis is None:
            print("Doing actual fit")
            self.ecis = self._fit(self.feature_matrix, self.normalized_energies, self.weights, self.mu)
        else:
            print('By default, using existing fit')
            self.ecis = np.array(self.ecis)

        # calculate the results of the fitting

        logging.info('rmse (in-sample): {}, cv score(5 fold): {}, mu_opt: {}'.format(self.rmse,self.cv,self.mu))
        #logging.info('best CV score: {}'.format(np.nanargmax(self.cvs)))

    @property
    def normalized_ce_energies(self): 
        return np.dot(self.feature_matrix, self.ecis)

    @property
    def ce_energies(self): 
        return self.normalized_ce_energies * self.sizes

    @property
    def normalized_error(self):
        return self.normalized_ce_energies - self.normalized_energies
    
    @property
    def rmse(self):
        if self._rmse is None:
            self._rmse= np.average(self.normalized_error ** 2) ** 0.5
        return self._rmse

    @property
    def cv(self,k=5):
        if self._cv is None:
            A = self.feature_matrix
            f = self.normalized_energies
            weights = self.weights
            mu = self.mu

            partitions = np.tile(np.arange(k), len(f)//k+1)
            np.random.shuffle(partitions)
            partitions = partitions[:len(f)]

            ssr = 0
            ssr_uw = 0
            for i in range(k):
                ins = (partitions != i) #in the sample for this iteration
                oos = (partitions == i) #out of the sample for this iteration

                mapping = {}
                for i in range(len(f)):
                    if ins[i]:
                        mapping[len(mapping.keys())] = i

                ecis = self._fit(A[ins], f[ins], weights[ins], mu, subset_mapping=mapping, skip_gs=True)
                res = (np.dot(A[oos], ecis) - f[oos]) ** 2
                ssr += np.sum(res * weights[oos]) / np.average(weights[oos])
                ssr_uw += np.sum(res)

            self. _cv = 1 - ssr / np.sum((f - np.average(f)) ** 2)
        return self._cv

    @property
    def structures(self):
        return [i['structure'] for i in self.items]

    @property
    def energies(self):
        return np.array([i['energy'] for i in self.items])

    @property
    def weights(self):
        return np.array([i['weight'] for i in self.items])

    @property
    def supercells(self):
        return np.array([i['supercell'] for i in self.items])

    @property
    def feature_matrix(self):
        return np.array([i['features'] for i in self.items])

    @property
    def sizes(self):
        return np.array([i['size'] for i in self.items])

    @property
    def normalized_energies(self):
        return self.energies / self.sizes

    @property
    def spacegroups(self): 
        return [SpacegroupAnalyzer(s,symprec=1e-1).get_space_group_symbol() for s in self.structures]

    @classmethod
    def unweighted(cls, cluster_expansion, structures, energies, mu=None, max_dielectric=None,
                   max_ewald=None, solver='cvxopt_l1'):
        #print("unweighted got {}".format(len(structures)))
        weights = np.ones(len(structures))
        return cls(cluster_expansion=cluster_expansion, structures=structures, energies=energies,
                   mu=mu, weights=weights, max_dielectric=max_dielectric, max_ewald=max_ewald, solver=solver)

    @classmethod
    def weight_by_e_above_hull(cls, cluster_expansion, structures, energies, mu=None, max_dielectric=None,
                               max_ewald=None, temperature=2000, solver='cvxopt_l1'):
        pd = _pd(structures, energies, cluster_expansion)
        e_above_hull = _energies_above_hull(pd, structures, energies)
        weights = np.exp(-e_above_hull / (0.00008617 * temperature))

        return cls(cluster_expansion=cluster_expansion, structures=structures, energies=energies,
                   mu=mu, weights=weights, max_dielectric=max_dielectric, max_ewald=max_ewald, solver=solver)

    @classmethod
    def weight_by_e_above_comp(cls, cluster_expansion, structures, energies, mu=None, max_dielectric=None,
                               max_ewald=None, temperature=2000, solver='cvxopt_l1'):
        e_above_comp = _energies_above_composition(structures, energies)
        weights = np.exp(-e_above_comp / (0.00008617 * temperature))

        return cls(cluster_expansion=cluster_expansion, structures=structures, energies=energies,
                   mu=mu, weights=weights, max_dielectric=max_dielectric, max_ewald=max_ewald, solver=solver)

    @property
    def pd_input(self):
        if self._pd_input is None:
            self._pd_input = _pd(self.structures, self.energies, self.ce)
        return self._pd_input

    @property
    def pd_ce(self):
        if self._pd_ce is None:
            self._pd_ce = _pd(self.structures, self.ce_energies, self.ce)
        return self._pd_ce

    @property
    def e_above_hull_input(self):
        if self._e_above_hull_input is None:
            self._e_above_hull_input = _energies_above_hull(self.pd_input, self.structures, self.energies)
        return self._e_above_hull_input

    @property
    def e_above_hull_ce(self):
        if self._e_above_hull_ce is None:
            self._e_above_hull_ce = _energies_above_hull(self.pd_ce, self.structures, self.ce_energies)
        return self._e_above_hull_ce

    def get_scatterplot(self, xaxis='e_above_hull_input', yaxis='e_above_hull_ce'):
        """
        plots two attributes. Some useful pairs:
            xaxis='e_above_hull_input', yaxis='e_above_hull_ce'
            xaxis='normalized_energies', yaxis='normalized_ce_energies'
            xaxis='e_above_hull_input', yaxis='normalized_error'
        """
        textstr = '\n'.join((
        r'$RMSE = %.3f eV/prim$'%(self.rmse,),
        r'$CV Score = %.3f$'%(self.cv,),
        r'$N_{strs} = %d$'%len(self.items)
        ))
        plt.figure(figsize=(8.0,6.0))
        plt.scatter(self.__getattribute__(xaxis), self.__getattribute__(yaxis),label=textstr)
        fs = 18
        fst = 14
        plt.tick_params(labelsize=fst)
        plt.xlabel(xaxis,fontsize=fs)
        plt.ylabel(yaxis,fontsize=fs)
        plt.legend(fontsize=fs)
        #props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        #plt.title(textstr,fontsize=fs)

        return plt
    
    def get_eciplot(self):
        plt.close()
        fs = 18
        fst = 14
        plt.figure(figsize=(8.0,6.0))
        xs = np.arange(0,len(self.ecis))+0.5
        plt.bar(xs,self.ecis,width=1.0)
        if self.ce.use_ewald and not self.ce.use_inv_r:
            plt.bar(xs[-1:],self.ecis[-1:],width=1.0,color='r',label='Ewald eci')
        plt.tick_params(labelsize=fst)
        plt.xlabel('Cluster indices',fontsize=fs)
        plt.ylabel('ECIs eV/prim',fontsize=fs)
        if self.ce_use_ewald and not self.ce_use_inv_r:
            plt.legend(fontsize=fs)

        return plt

    def get_optimum_mu(self, A, f, weights, k=5, min_mu=-1, max_mu=6):
        """
        Finds the value of mu that maximizes the cv score
        """
        if self.solver == 'lstsq':
            assert A.shape[0]>A.shape[1]
            mu = 0
            self.cvs = [self._calc_cv_score(mu,A,f,weights,k=5)]
            return 0

        elif self.solver in ['cvxopt_l1','bregman_l1','pcr']:
            mus = list(np.logspace(min_mu, max_mu, 10))
            cvs = [self._calc_cv_score(mu, A, f, weights, k) for mu in mus]
            
            for it in range(3):
                i = np.nanargmax(cvs)
                if i == len(mus)-1:
                    if it ==0:
                        warnings.warn('Largest mu chosen. You should probably increase the basis set')
                    break
                
                mu = (mus[i] * mus[i+1]) ** 0.5
                mus[i+1:i+1] = [mu]
                cvs[i+1:i+1] = [self._calc_cv_score(mu, A, f, weights, k)]
                
                mu = (mus[i-1] * mus[i]) ** 0.5
                mus[i:i] = [mu]
                cvs[i:i] = [self._calc_cv_score(mu, A, f, weights, k)]
        
            self.mus = mus
            self.cvs = cvs
            logging.info('best cv score: {}'.format(np.nanmax(self.cvs)))
            return mus[np.nanargmax(cvs)]
        # Recommended PCR min_mu = -5, max_mu = 0

        elif self.solver == 'l1l0':
            mu0s = list(np.logspace(-5,-2, 6))
            mu1s = list(np.logspace(-2, 1, 4)) # Based on Wenxuan's empirical values.
            cvs_cur = []
            mus_cur = []

            for it in range(3):
                # Considering computational need, only fine grain the local area once.
                cvs = []
                mus = []
                for i in range(len(mu0s)):
                    cvs_i = []
                    mus_i = []
                    for j in range(len(mu1s)):
                        mu = [mu0s[i],mu1s[j]]
                        try:
                            cv = self._calc_cv_score(mu,A,f,weights,k)
                        except:
                            cv = -np.inf
                            #solution failed because of numerical instability
                        cvs_i.append(cv)
                        mus_i.append(mu)
                    cvs.append(cvs_i)
                    mus.append(mus_i)
                cvs_cur = cvs
                mus_cur = mus

                i,j = np.unravel_index(np.argmax(np.array(cvs), axis=None), np.array(cvs).shape)
                if i == len(mu0s)-1:
                    if it==0:
                        warnings.warn('Largest mu0 chosen. You should probably increase the basis set')
                    break
                if j == len(mu1s)-1:
                    if it==0:
                        warnings.warn('Largest mu1 chosen. You should probably increase the basis set')
                    break
                
                mu0_min = mu0s[i-1] if i else mu0s[i]
                mu0_max = mu0s[i+1]
                mu1_min = mu1s[i-1] if i else mu1s[i]
                mu1_max = mu1s[i+1]
                p = np.log(10)        
                mu0s = list(np.logspace(np.log(mu0_min)/p,np.log(mu0_max)/p,4))
                mu1s = list(np.logspace(np.log(mu1_min)/p,np.log(mu1_max)/p,6))
                     
            self.mus = mus_cur
            self.cvs = cvs_cur
            #print('cvs:',self.cvs,'\nmus:',self,mus)
            i_opt,j_opt = np.unravel_index(np.argmax(np.array(self.cvs), axis=None), np.array(self.cvs).shape)
            logging.info('best cv score: {}'.format(self.cvs[i_opt][j_opt]))
            return self.mus[i_opt][j_opt]

        else:
            raise ValueError("Solver not supported!")

    def _calc_cv_score(self, mu, A, f, weights, k=5):
        """
        Args:
            mu: weight of error in bregman
            A: sensing matrix (scaled appropriately)
            f: data to fit (scaled appropriately)
            k: number of partitions

        Partition the sample into k partitions, calculate out-of-sample
        variance for each of these partitions, and add them together
        """
        logging.info('starting cv score calculations for mu: {}, k: {}'.format(mu, k))
        # generate random partitions
        partitions = np.tile(np.arange(k), len(f)//k+1)
        np.random.shuffle(partitions)
        partitions = partitions[:len(f)]

        ssr = 0
        ssr_uw = 0
        for i in range(k):
            ins = (partitions != i) #in the sample for this iteration
            oos = (partitions == i) #out of the sample for this iteration

            mapping = {}
            for i in range(len(f)):
                if ins[i]:
                    mapping[len(mapping.keys())] = i

            ecis = self._fit(A[ins], f[ins], weights[ins], mu, subset_mapping=mapping, skip_gs=True)
            res = (np.dot(A[oos], ecis) - f[oos]) ** 2
            ssr += np.sum(res * weights[oos]) / np.average(weights[oos])
            ssr_uw += np.sum(res)

        logging.info('cv rms_error: {} (weighted) {} (unweighted)'.format(np.sqrt(ssr/len(f)), np.sqrt(ssr_uw/len(f))))
        cv = 1 - ssr / np.sum((f - np.average(f)) ** 2)
        return cv

    def _fit(self, A, f, weights, mu, subset_mapping=None, skip_gs=False):
        """
        Returns the A matrix and f vector for the bregman
        iterations, given weighting parameters
        """
        #print("Fit got A/f/w: {}/{}/{}".format(A.shape, f.shape, weights.shape))
        A_in = A.copy()
        f_in = f.copy()

        ecis = self._solve_weighted(A_in, f_in, weights, mu, subset_mapping=subset_mapping, skip_gs=skip_gs)

        if self.ce.use_ewald and self.max_dielectric is not None:
            if self.ce.use_inv_r:
                raise NotImplementedError('cant use inv_r with max dielectric yet')
            if ecis[-1] < 1 / self.max_dielectric:
                f_in -= A_in[:, -1] / self.max_dielectric
                A_in[:, -1] = 0
                ecis = self._solve_weighted(A_in, f_in, weights, mu, subset_mapping=subset_mapping)
                ecis[-1] = 1 / self.max_dielectric

        return ecis

    def _solve_weighted(self, A, f, weights, mu, subset_mapping=None, skip_gs=False):
        #print("Solve weighted got A/f/weights: {}/{}/{}".format(A.shape, f.shape, weights.shape))
        A_w = A * weights[:, None] ** 0.5
        f_w = f * weights ** 0.5
        #print("Aw/fw: {}/{}".format(A_w.shape, f_w.shape))

        if self.solver == 'cvxopt_l1':
            return self._solve_cvxopt(A_w, f_w, mu)
        elif self.solver == 'bregman_l1':
            return self._solve_bregman(A_w, f_w, mu)
        elif self.solver == 'gs_preserve':
            return self._solve_gs_preserve(A_w, f_w, mu, subsample_mapping=subset_mapping, skip_gs=skip_gs)
        elif self.solver == 'l1l0':
            return self._solve_l1l0(A_w, f_w, mu[0], mu[1])
        elif self.solver == 'pcr':
            return self._solve_pcr(A_w, f_w, mu)
        elif self.solver == 'lstsq':
            return np.linalg.lstsq(A_w, f_w,rcond=None)[0]

    def _solve_cvxopt(self, A, f, mu):
        """
        A and f should already have been adjusted to account for weighting
        """
        from cluster_expansion.compressive.l1regls import l1regls, solvers
        solvers.options['show_progress'] = False
        from cvxopt import matrix
        A1 = matrix(A)
        b = matrix(f * mu)
        return (np.array(l1regls(A1, b)) / mu).flatten()

    def _solve_bregman(self, A, f, mu):
        return split_bregman(A, f, MaxIt=1e5, tol=1e-7, mu=mu, l=1, quiet=True)

    def _solve_l1l0(self,A,f,mu0,mu1,M=100.0,cutoff=300):
        """
        Brute force solving by gurobi. Cutoff in 300s.
        """
        n = A.shape[0]
        d = A.shape[1]
        ATA = A.T @ A
        fTA = f.T @ A
        
        l1l0 = Model()
        w = l1l0.addVars(d,lb=-GRB.INFINITY,ub=GRB.INFINITY)
        z0 = l1l0.addVars(d,vtype=GRB.BINARY)
        z1 = l1l0.addVars(d)
        for i in range(d):
            l1l0.addConstr(M*z0[i]>=w[i])
            l1l0.addConstr(M*z0[i]>=(-1.0*w[i]))
            l1l0.addConstr(z1[i]>=w[i])
            l1l0.addConstr(z1[i]>=(-1.0*w[i]))
        # Cost function
        L = QuadExpr()
        for i in range(d):
            L = L+mu0*z0[i]
            L = L+mu1*z1[i]
            L = L-2*fTA[i]*w[i]
            for j in range(d):
                L = L+w[i]*w[j]*ATA[i][j]

        l1l0.setObjective(L,GRB.MINIMIZE)
        l1l0.setParam(GRB.Param.TimeLimit, cutoff)
        l1l0.setParam(GRB.Param.PSDTol,1e-5) # Set a larger PSD tolerance to ensure success
        l1l0.setParam(GRB.Param.OutputFlag, 0)
        # Using the default algorithm, and shut gurobi up.
        l1l0.update()
        l1l0.optimize()
        w_opt = np.array([w[v_id].x for v_id in w])
        return w_opt

    def _solve_pcr(self, A, f, mu):
        """
        Reduce cluster dimensionality using PCR method.
        """
        V,s,VT = np.linalg.svd(A.T@ A)
        d= A.shape[1]
        n= A.shape[0]
        k_cut = min(n,d)
        # If the model is not full rank, we will cut it to full rank to make it solvable.
        for k in range(min(n,d)):
            if s[k]<mu:
                k_cut = k
                break
        Vk = V[:,:k_cut]
        Ak = A@Vk
        wk = np.linalg.inv(Ak.T@Ak)@Ak.T@f
        w_opt = Vk@wk
        return w_opt
                
    def _solve_gs_preserve(self, A, f, mu, subsample_mapping, skip_gs=False):
        """
        Code notes from Daniil Kitchaev (dkitch@alum.mit.edu) - 2018-09-10

        This is a WORK IN PROGRESS based on Wenxuan's ground-state preservation fitting code.
        A, f, and mu as as in the other routines
        subsample mapping deals with the fact that weights change when fitting on a partial set (when figuring out mu)
        skin_gs gives the option of ignoring the constrained fitting part, which is helpful when figuring out mu

        In general, this code is really not production ready - the algorithm that serious numerical issues, and getting
        around them involved lots of fiddling with eigenvalue roundoffs, etc, as is commented below.

        There are also issues with the fact that constraints can be very difficult to satisfy, causing the solver to
        diverge (or just quit silently giving absurd results) - ths solution here appears to be to use MOSEK instead
        of cvxopt, and to iteratively remove constraints when they cause problems. Usually after cleaning up the data,
        everything can be fit though without removing constraints.

        At the end of the day, this algorithm seems to only be useful for niche applications because enforcing ground
        state preservation causes a giant bias in the fit and makes the error in E-above-hull highly correlated with the
        value of E-above-hull. The result is entropies are completely wrong, which is what you usually want out of a
        cluster expansion.

        So, use the code at your own risk. AFAIK, it works as described in Wenxuans paper, with various additions from
        me for numerical stability. It has not been extensively tested though or used in real projects due to the bias
        issue I described above. I think that unless the bias problem is resolved, this fitting scheme will not be
        of much practical use.
        """
        if not subsample_mapping:
            assert A.shape[0] == self.feature_matrix.shape[0]
            subsample_mapping = {}
            for i in range(self.feature_matrix.shape[0]):
                subsample_mapping[i] = i

        from cvxopt import matrix
        from cvxopt import solvers
        from pymatgen.core.periodic_table import get_el_sp
        try:
            import mosek
        except:
            raise ValueError("GS preservation fitting is finicky and MOSEK solvers are typically required for numerical stability.")
        solvers.options['show_progress'] = False
        solvers.options['MOSEK'] = {mosek.dparam.check_convexity_rel_tol: 1e-6}

        ehull = list(self.e_above_hull_input)
        structure_index_at_hull = [i for (i,e) in enumerate(ehull) if e < 1e-5]

        reduce_composition_at_hull = [
            self.structures[i].composition.element_composition.reduced_composition.element_composition for
            i in structure_index_at_hull]

        all_corr_in = np.array(self.feature_matrix)
        all_engr_in = np.array(self.normalized_energies)

        # Some structures can be degenerate in correlation space, even if they are distinct in reality. We can't
        # constrain their energies since as far as the CE is concerned, same correlation = same structure
        duplicated_correlation_set = []
        for i in range(len(all_corr_in)):
            if i not in structure_index_at_hull:
                for j in structure_index_at_hull:
                    if np.max(np.abs(all_corr_in[i] - all_corr_in[j])) < 1e-6:
                        logging.info("Structure {} ({} - {}) has the same correlation as hull structure {} ({} {})".format(i,
                                                                    self.structures[i].composition.element_composition.reduced_formula,
                                                                    self.spacegroups[i],
                                                                    j,
                                                                    self.structures[j].composition.element_composition.reduced_formula,
                                                                    self.spacegroups[j]))
                        duplicated_correlation_set.append(i)

        all_engr_in.shape = (len(all_engr_in), 1)
        f.shape = (f.shape[0], 1)

        # Adjust weights if subsample changed whats included and whats not
        weights_tmp = []
        for i in range(A.shape[0]):
            weights_tmp.append(self.weights[subsample_mapping[i]])

        subsample_mapping_inv = {}
        for i, j in subsample_mapping.items():
            subsample_mapping_inv[j] = i
        for i in duplicated_correlation_set:
            if i in subsample_mapping_inv.keys():
                weights_tmp[subsample_mapping_inv[i]] = 0


        weight_vec = np.array(weights_tmp)

        weight_matrix = np.diag(weight_vec.transpose())

        N_corr = A.shape[1]

        # Deal with roundoff error making P not positive semidefinite by using the SVD of A
        # At = USV*
        # At A = U S St Ut -> any negatives in S get squared
        # Unfortunately, this is usually not enough, so the next step is to explicitly add something small (1e-10)
        # to all eigenvalues so that eigenvalues close to zero are instead very slightly positive.
        # Otherwise, random numerical error makes the matrix not positive semidefinite, and the convex optimization
        # gets confused
        Aw = weight_matrix.dot(A)
        u, s, v = la.svd(Aw.transpose())
        Ss = np.pad(np.diag(s), ((0, u.shape[0] - len(s)),(0,0)), mode='constant', constant_values=0)
        P_corr_part = 2 * u.dot((Ss.dot(Ss.transpose()))).dot(u.transpose())
        P = np.lib.pad(P_corr_part, ((0, N_corr), (0, N_corr)), mode='constant', constant_values=0)
        P = 0.5 * (P + P.transpose())
        ev, Q = la.eigh(P)
        Qi = la.inv(Q)
        P = Q.dot(np.diag(np.abs(ev)+1e-10)).dot(Qi)

        q_corr_part = -2 * ((weight_matrix.dot(A)).transpose()).dot(f)
        q_z_part = np.ones((N_corr, 1)) / mu
        q = np.concatenate((q_corr_part, q_z_part), axis=0)

        G_1 = np.concatenate((np.identity(N_corr), -np.identity(N_corr)), axis=1)
        G_2 = np.concatenate((-np.identity(N_corr), -np.identity(N_corr)), axis=1)
        G_3 = np.concatenate((G_1, G_2), axis=0)
        h_3 = np.zeros((2 * N_corr, 1))

        # formulation is min 1/2 x'Px+ q'x s.t.: Gx<=h, Ax=b

        # P = 2 * A^T A
        # q = -2 * E^T A = q^T -> q = -2 * A^T E

        # See Wenxuan npjCompMat paper for derivation. All of the above mess is implementing this formula, plus dealing
        # with numerical issues with zero eigenvalues getting rounded off to something slightly negative

        init_vals = matrix(np.linalg.lstsq(self.feature_matrix, self.normalized_energies)[0])

        input_entries = []
        for s, e in zip(self.structures, self.energies):
            input_entries.append(PDEntry(s.composition.element_composition, e))
        max_e = max(input_entries, key=lambda e: e.energy_per_atom).energy_per_atom + 1000
        for el in self.ce.structure.composition.keys():
            input_entries.append(PDEntry(Composition({el: 1}).element_composition, max_e))
        pd_input = PhaseDiagram(input_entries)

        constraint_strings = []

        # Uncomment to save various matrices for debugging purposes
        #np.save("A.npy", A)
        #np.save("f.npy", f)
        #np.save("w.npy", weight_vec)
        #np.save("P.npy", P)
        #np.save("q.npy", q)
        #np.save("G_noC.npy", G_3)
        #np.save("h_noC.npy", h_3)

        # The next part deals with adding constraints based on on-hull/off-hull compositions
        # Once again, there are numerical errors that arise when some structures are very close in correlation space
        # or in energy, such that the solver runs into either numerical issues or something else. The solution seems
        # to be to add constraints in batches, and try the increasingly constrained fit every once in a while.
        # When the fitting fails, roll back to find the problematic constraint and remove it. Usually there isnt more
        # than one or two bad constrains, and looking at them by hand is enough to figure out why they are causing
        # problems.
        BATCH_SIZE = int(np.sqrt(len(all_corr_in)))
        tot_constraints = 0
        removed_constraints = 0
        if not skip_gs:
            for i in range(len(all_corr_in)):
                if i not in structure_index_at_hull and i not in duplicated_correlation_set:

                    reduced_comp = self.structures[i].composition.element_composition.reduced_composition.element_composition
                    if reduced_comp in reduce_composition_at_hull:  ## in hull composition

                        hull_idx = reduce_composition_at_hull.index(reduced_comp)
                        global_index = structure_index_at_hull[hull_idx]

                        G_3_new_line = np.concatenate((all_corr_in[global_index] - all_corr_in[i], np.zeros((N_corr))))

                        G_3_new_line.shape = (1, 2 * N_corr)
                        G_3 = np.concatenate((G_3, G_3_new_line), axis=0)
                        small_error = np.array(-1e-3) # TODO: This tolerance is actually quite big, but it can be reduced as needed
                        small_error.shape = (1, 1)
                        h_3 = np.concatenate((h_3, small_error), axis=0)
                        tot_constraints += 1
                        string = "{}|Added constraint from {}({} - {}) structure at hull comp".format(h_3.shape[0], reduced_comp, self.spacegroups[i], i)
                        print(string)
                        constraint_strings.append(string)

                    else:  # out of hull composition

                        comp_now = self.structures[i].composition.element_composition.reduced_composition.element_composition
                        decomposition_now = pd_input.get_decomposition(comp_now)
                        new_vector = -1.0 * all_corr_in[i]
                        for decompo_keys, decompo_values in decomposition_now.items():
                            reduced_decompo_keys = decompo_keys.composition.element_composition.reduced_composition.element_composition
                            index_1 = reduce_composition_at_hull.index(reduced_decompo_keys)
                            vertex_index_global = structure_index_at_hull[index_1]
                            new_vector = new_vector + decompo_values * all_corr_in[vertex_index_global]

                        G_3_new_line = np.concatenate((new_vector, np.zeros(N_corr)))

                        G_3_new_line.shape = (1, 2 * N_corr)
                        G_3 = np.concatenate((G_3, G_3_new_line), axis=0)

                        small_error = np.array(-1e-3)
                        small_error.shape = (1, 1)
                        h_3 = np.concatenate((h_3, small_error), axis=0)
                        tot_constraints += 1
                        string = "{}|Added constraint from {}({}) structure not at hull comp".format(h_3.shape[0], reduced_comp, i)
                        print(string)
                        constraint_strings.append(string)

                elif i in structure_index_at_hull:
                    if self.structures[i].composition.element_composition.is_element:
                        continue

                    entries_new = []
                    for j in structure_index_at_hull:
                        if not j == i:
                            entries_new.append(
                                PDEntry(self.structures[j].composition.element_composition, self.energies[j]))

                    for el in self.ce.structure.composition.keys():
                        entries_new.append(PDEntry(Composition({el: 1}).element_composition,
                                                   max(self.normalized_energies) + 1000))

                    pd_new = PhaseDiagram(entries_new)

                    comp_now = self.structures[i].composition.element_composition.reduced_composition.element_composition
                    decomposition_now = pd_new.get_decomposition(comp_now)

                    new_vector = all_corr_in[i]

                    abandon = False
                    print("Constraining gs of {}({})".format(self.structures[i].composition, self.structures[i].composition))
                    for decompo_keys, decompo_values in decomposition_now.items():
                        reduced_decompo_keys = decompo_keys.composition.element_composition.reduced_composition.element_composition
                        if not reduced_decompo_keys in reduce_composition_at_hull:
                            abandon = True
                            break

                        index = reduce_composition_at_hull.index(reduced_decompo_keys)
                        vertex_index_global = structure_index_at_hull[index]
                        new_vector = new_vector - decompo_values * all_corr_in[vertex_index_global]
                    if abandon:
                        continue

                    G_3_new_line = np.concatenate((new_vector, np.zeros(N_corr)))

                    G_3_new_line.shape = (1, 2 * N_corr)
                    G_3 = np.concatenate((G_3, G_3_new_line), axis=0)
                    small_error = np.array(-1e-3) # TODO: Same tolerance as above
                    small_error.shape = (1, 1)
                    h_3 = np.concatenate((h_3, small_error), axis=0)
                    tot_constraints += 1
                    string = "{}|Added constraint from {}({}) structure on hull, decomp".format(h_3.shape[0], comp_now, i)
                    print(string)
                    constraint_strings.append(string)

                if i % BATCH_SIZE == 0 or i == len(all_corr_in)-1:
                    valid = False
                    const_remove = 0
                    G_t = deepcopy(G_3)
                    h_t = deepcopy(h_3)
                    # Remove constraints until fit works
                    while not valid:
                        sol = solvers.qp(matrix(P), matrix(q), matrix(G_3), matrix(h_3), initvals=init_vals, solver='mosek')
                        if sol['status'] == 'optimal':
                            valid = True
                        else:
                            const_remove += 1
                            G_3 = G_t[:-1 * (const_remove),:]
                            h_3 = h_t[:-1 * (const_remove)]
                            removed_constraints += 1

                    if const_remove > 0:
                        constraint_strings.append("{}|Removed".format(G_t.shape[0] - const_remove + 1))

                    # Add constraints back in one by one and remove if they cause problems
                    for num_new in range(1, const_remove):
                        G_new_line = G_t[-1 * (const_remove - num_new),:]
                        h_new_line = h_t[-1 * (const_remove - num_new)]
                        G_new_line.shape = (1, 2 * N_corr)
                        h_new_line.shape = (1,1)
                        G_3 = np.concatenate((G_3, G_new_line), axis=0)
                        h_3 = np.concatenate((h_3, h_new_line), axis=0)
                        sol = solvers.qp(matrix(P), matrix(q), matrix(G_3), matrix(h_3), initvals=init_vals, solver='mosek')
                        removed_constraints -= 1
                        if sol['status'] != 'optimal':
                            G_3 = G_3[:-1, :]
                            h_3 = h_3[:-1]
                            removed_constraints += 1
                            constraint_strings.append("{}|Removed".format(G_t.shape[0] - const_remove + num_new + 1))
            # Uncomment for iterative saving matricex
            #np.save("G.npy", G_3)
            #np.save("h.npy", h_3)



        # Uncomment for debugging
        #np.save("G.npy", G_3)
        #np.save("h.npy", h_3)

        sol = solvers.qp(matrix(P), matrix(q), matrix(G_3), matrix(h_3), initvals=init_vals, solver='mosek')
        print("Final status: {}".format(sol['status']))
        print("Mu: {}".format(mu))
        print("Constrants: {}/{}".format(tot_constraints - removed_constraints, tot_constraints))
        ecis = np.array(sol['x'])[:N_corr, 0]

        # Uncomment for some debugging info
        #print(ecis)
        #for string in constraint_strings:
        #    print(string)
        return ecis

    def get_mu_plot(self):
        ax = plt.subplot(111)
        ax.scatter(self.mus, self.cvs)
        ax.set_xscale('log')
        return plt

    def print_ecis(self):
        corr = np.zeros(self.ce.n_bit_orderings)
        corr[0] = 1 #zero point cluster
        cluster_std = np.std(self.feature_matrix, axis=0)
        for sc in self.ce.symmetrized_clusters:
            print(sc, len(sc.bits)-1, sc.sc_b_id)
            print('bit    eci    cluster_std    eci*cluster_std')
            for i, bits in enumerate(sc.bit_combos):
                eci = self.ecis[sc.sc_b_id + i]
                c_std = cluster_std[sc.sc_b_id + i]
                print(bits, eci, c_std, eci * c_std)
        print(self.ecis)

    def structure_energy(self, structure):
        return self.ce.structure_energy(structure, self.ecis)

    @classmethod
    def from_dict(cls, d):
        generator = cls(cluster_expansion=ClusterExpansion.from_dict(d['cluster_expansion']),
                   structures=[Structure.from_dict(s) for s in d['structures']],
                   energies=np.array(d['energies']), max_dielectric=d.get('max_dielectric'),
                   max_ewald=d.get('max_ewald'), supercell_matrices=d['supercell_matrices'],
                   mu=d.get('mu'), ecis=d.get('ecis'), feature_matrix=d.get('feature_matrix'),
                   solver=d.get('solver'), weights=d['weights'])
        if 'cv' in d: generator._cv = d['cv']
        if 'rmse' in d: generator._rmse = d['rmse']
        return generator

    def as_dict(self):
        return {'cluster_expansion': self.ce.as_dict(),
                'structures': [s.as_dict() for s in self.structures],
                'energies': self.energies.tolist(),
                'supercell_matrices': [cs.supercell_matrix.tolist() for cs in self.supercells],
                'max_dielectric': self.max_dielectric,
                'max_ewald': self.max_ewald,
                'mu': self.mu,
                'ecis': self.ecis.tolist(),
                'feature_matrix': self.feature_matrix.tolist(),
                'weights': self.weights.tolist(),
                'solver': self.solver,
                'cv': self.cv,
                'rmse': self.rmse,
                '@module': self.__class__.__module__,
                '@class': self.__class__.__name__}
