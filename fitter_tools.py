#!/usr/bin/env python

from __future__ import division
from __future__ import unicode_literals

__author__ = 'Bin Ouyang & Fengyu_xie'
__version__ = 'Dev'


import os
import sys
import argparse
import json
import random
from copy import deepcopy
import numpy as np
import numpy.linalg as la
from pymatgen.analysis.structure_matcher import StructureMatcher, ElementComparator
from pyabinitio.cluster_expansion.eci_fit import EciGenerator
from pyabinitio.cluster_expansion.ce import ClusterExpansion
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from itertools import groupby
from pymatgen.io.vasp.sets import MITRelaxSet
from pymatgen.io.vasp.inputs import *
from pymatgen.io.vasp.outputs import *
from pymatgen.io.cif import *
from pymatgen import Structure
from pymatgen.core.periodic_table import Specie
from pymatgen.core.composition import Composition
from pymatgen.core.sites import PeriodicSite
from itertools import permutations,product
from operator import mul
from functools import partial,reduce
import multiprocessing
from mc import *
from OxData import OXRange #This should be a database like file
import collections
