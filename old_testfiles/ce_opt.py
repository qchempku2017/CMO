from __future__ import division
from __future__ import unicode_literals
import os
import sys
import argparse
import json
from copy import deepcopy
import numpy as np
import numpy.linalg as la
from itertools import permutations
from monty.serialization import loadfn, dumpfn
from math import gcd
from functools import reduce
import random
from itertools import permutations
from operator import mul
from functools import partial, reduce
import multiprocessing as mp
import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from cvxopt import matrix
from l1regls import l1regls, solvers
import itertools


def Bayes_l1_opt(A, f, mu, cov):
    Abar = np.dot(np.linalg.pinv(np.transpose(A)), (np.dot(np.transpose(A), A) + cov))

    solvers.options['show_progress'] = False
    A1 = matrix(Abar)
    b = matrix(f * mu)
    ecis = (np.array(l1regls(A1, b)) / mu).flatten()
    return ecis

def l2_opt(A, f, mu):
    m, d = A.shape
    f = f
    inv = np.linalg.pinv(np.dot(np.transpose(A), A) + mu * np.eye(d))
    ecis = np.dot(np.dot(inv, np.transpose(A)), f)
#     ecis /= mu
    return ecis

def l2_Bayessian(A, f, cov):
    m, d = A.shape
    f = f
    inv = np.linalg.pinv(np.dot(np.transpose(A), A) + cov)
    ecis = np.dot(np.dot(inv, np.transpose(A)), f)
#     ecis /= mu
    return ecis

def l1_opt(A, f, mu):


#     A = A[0:150,:]


    A_w = A
    f_w = f

    solvers.options['show_progress'] = False
    A1 = matrix(A)
    b = matrix(f * mu)
    ecis = (np.array(l1regls(A1, b)) / mu).flatten()
    return ecis


def calc_cv_score_Bayes(A, f, regu,  k=5):
        """
        Args:
            mu: weight of error in bregman
            A: sensing matrix (scaled appropriately)
            f: data to fit (scaled appropriately)
            k: number of partitions

        Partition the sample into k partitions, calculate out-of-sample
        variance for each of these partitions, and add them together
        """
        # logging.info('starting cv score calculations for mu: {}, k: {}'.format(mu, k))
        # generate random partitions
        partitions = np.tile(np.arange(k), len(f)//k+1)
        np.random.shuffle(partitions)
        partitions = partitions[:len(f)]
        ssr = 0

        for i in range(k):
            ins = (partitions != i) #in the sample for this iteration
            oos = (partitions == i) #out of the sample for this iteration

            ecis = l2_Bayessian(A=A[ins], f=f[ins],cov=regu)
#             print(A[oos])
#             print(ecis)

            res = (np.dot(A[oos], ecis) - f[oos]) ** 2
#             print(res)
            ssr += np.average(res)

        cv = ssr / k
        return cv


def calc_cv_score_l2(A, f,  k=5):
        """
        Args:
            mu: weight of error in bregman
            A: sensing matrix (scaled appropriately)
            f: data to fit (scaled appropriately)
            k: number of partitions

        Partition the sample into k partitions, calculate out-of-sample
        variance for each of these partitions, and add them together
        """
        # logging.info('starting cv score calculations for mu: {}, k: {}'.format(mu, k))
        # generate random partitions
        partitions = np.tile(np.arange(k), len(f)//k+1)
        np.random.shuffle(partitions)
        partitions = partitions[:len(f)]
        ssr = 0

        for i in range(k):
            ins = (partitions != i) #in the sample for this iteration
            oos = (partitions == i) #out of the sample for this iteration

            ecis = l2_opt(A=A[ins], f=f[ins], mu=0)
#             print(A[oos])
#             print(ecis)

            res = (np.dot(A[oos], ecis) - f[oos]) ** 2
#             print(res)
            ssr += np.average(res)

        cv = ssr / k
        return cv


def calc_cv_score_l1(A, f, mu,  k=5):
        """
        Args:
            mu: weight of error in bregman
            A: sensing matrix (scaled appropriately)
            f: data to fit (scaled appropriately)
            k: number of partitions

        Partition the sample into k partitions, calculate out-of-sample
        variance for each of these partitions, and add them together
        """
        # logging.info('starting cv score calculations for mu: {}, k: {}'.format(mu, k))
        # generate random partitions
        partitions = np.tile(np.arange(k), len(f)//k+1)
        np.random.shuffle(partitions)
        partitions = partitions[:len(f)]
        ssr = 0

        for i in range(k):
            ins = (partitions != i) #in the sample for this iteration
            oos = (partitions == i) #out of the sample for this iteration

            ecis = l1_opt(A=A[ins], f=f[ins], mu = mu)
#             print(A[oos])
#             print(ecis)

            res = (np.dot(A[oos], ecis) - f[oos]) ** 2
#             print(res)
            ssr += np.average(res)

        cv = ssr / k
        return cv

def gcv_score_l2(A, f, mu):
    m, d= A.shape
    ecis = l2_opt(A=A, f= f, mu= mu)
    rss = np.average((np.dot(A, ecis) - f)**2)
    inv = np.linalg.pinv(np.dot(np.transpose(A), A) +mu* np.eye(d))

    domi = np.trace(np.eye(m) - np.dot(np.dot(A, inv), np.transpose(A)))
    GCV = np.sqrt(m*rss) / domi
    return GCV



def get_optimal_mu_l2(A, f, weights, k=5, min_mu=-10, max_mu=0.1):
    """
    calculate the optimal mu from l2-regularized least square fitting

    regularization is uniform with mu * eye(n)

    """
    mus = list(np.logspace(min_mu, max_mu, 20))
    print(mus)
    cvs = [calc_cv_score_l2(mu, A, f, weights, k) for mu in mus]


    for _ in range(2):
        i = np.nanargmax(cvs)
        if i == len(mus)-1:
            warnings.warn('Largest mu chosen. You should probably increase the basis set')
            break

        mu = (mus[i] * mus[i+1]) ** 0.5
        mus[i+1:i+1] = [mu]
        cvs[i+1:i+1] = [calc_cv_score_l2(mu, A, f, weights, k)]

        mu = (mus[i-1] * mus[i]) ** 0.5
        mus[i:i] = [mu]
        cvs[i:i] = [calc_cv_score_l2(mu, A, f, weights, k)]

    return mus[np.nanargmax(cvs)]

def get_optimal_mu_l1(A, f, weights, k=5, min_mu=-10, max_mu=0.1):
    """
    calculate the optimal mu from l2-regularized least square fitting

    regularization is uniform with mu * eye(n)

    """
    mus = list(np.logspace(min_mu, max_mu, 20))
    print(mus)
    cvs = [calc_cv_score_l1(mu, A, f, k) for mu in mus]


    for _ in range(2):
        i = np.nanargmax(cvs)
        if i == len(mus)-1:
            warnings.warn('Largest mu chosen. You should probably increase the basis set')
            break

        mu = (mus[i] * mus[i+1]) ** 0.5
        mus[i+1:i+1] = [mu]
        cvs[i+1:i+1] = [calc_cv_score_l2(mu, A, f, weights, k)]

        mu = (mus[i-1] * mus[i]) ** 0.5
        mus[i:i] = [mu]
        cvs[i:i] = [calc_cv_score_l2(mu, A, f, weights, k)]

    return mus[np.nanargmax(cvs)]


def get_optimal_gmu_l2(A, f, weights, min_mu=-9, max_mu=0):
    """
    calculate the optimal generalized mu from l2-regularized least square fitting

    regularization is uniform with mu * eye(n)

    """
    mus = list(np.logspace(min_mu, max_mu, 10))
    print(mus)
    cvs = [gcv_score_l2(A=A,f=f,mu=mu) for mu in mus]


    for _ in range(2):
        i = np.nanargmax(cvs)
        if i == len(mus)-1:
            warnings.warn('Largest mu chosen. You should probably increase the basis set')
            break

        mu = (mus[i] * mus[i+1]) ** 0.5
        mus[i+1:i+1] = [mu]
        cvs[i+1:i+1] = [gcv_score_l2(A=A,f=f,mu=mu)]

        mu = (mus[i-1] * mus[i]) ** 0.5
        mus[i:i] = [mu]
        cvs[i:i] = [gcv_score_l2(A=A,f=f,mu=mu)]

    return mus[np.nanargmin(cvs)], cvs

def cluster_properties(ce):
    """
    return max radius and number of atoms in cluster

    input ce is cluster expansion class object
    """
    cluster_n = [0]
    cluster_r = [0]
    for sc in ce.symmetrized_clusters:
        for j in range(len(sc.bit_combos)):
            cluster_n.append( sc.bit_combos[j].shape[1])
            cluster_r.append( sc.max_radius)
    return np.array(cluster_n), np.array(cluster_r) # without ewald term

def gcv_score_Bayes(A, f, cov):
    """

    """
    m, d= A.shape
    ecis = l2_Bayessian(A=A, f= f, cov=cov)
    rss = np.average((np.dot(A, ecis) - f)**2)
    inv = np.linalg.pinv(np.dot(np.transpose(A), A) + cov)

    domi = np.trace(np.eye(m) - np.dot(np.dot(A, inv), np.transpose(A)))
    GCV = np.sqrt(m*rss) / domi
    return GCV

def get_optimal_gamma(ce, A, f):
    cluster_n, cluster_r = cluster_properties(ce=ce)
    gamma0 = np.logspace(-9, 0, 10)
    gammas = [np.append(gamma0, 0), np.logspace(-4, 0, 5), np.logspace(-4, 0, 5),
              np.linspace(0, 10, 5), np.linspace(0, 10, 5)]
    gammas = list(itertools.product(*gammas))
# test_list = np.array(test_list)
    gcvs = np.zeros(len(gammas))

    for i in range(len(gammas)):
        gamma = gammas[i]
#         print(gamma[3])
#         print(cluster_n*gamma[3])
        regu = gamma[0]*(cluster_r*gamma[1] +gamma[2] +1) **(cluster_n*gamma[3]+gamma[4])

        regu = np.append(regu, gamma[0])
#         ecis_i = l2_Bayessian(A=A, f=f, cov = np.diag(regu))

        gcv = gcv_score_Bayes(A=A, f=f, cov= np.diag(regu))
        gcvs[i] = gcv
#         print("i = {}, gcv = {}".format(i, gcv))

    print(np.min(gcvs))
    opt_gamma = gammas[np.nanargmin(gcvs)]

    regu = opt_gamma[0]*(cluster_r*opt_gamma[1] +opt_gamma[2] +1) **(cluster_n*opt_gamma[3]+opt_gamma[4])
    regu = np.append(regu, opt_gamma[0])



    return opt_gamma, regu


def regu_matrix(cluster_n, cluster_r, opt_gamma):

    regu = opt_gamma[0]*(cluster_r*opt_gamma[1] +opt_gamma[2] +1) **(cluster_n*opt_gamma[3]+opt_gamma[4])
    regu = np.append(regu, opt_gamma[0])
    return np.diag(regu)
