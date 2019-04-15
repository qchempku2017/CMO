#!/usr/bin/env python

from __future__ import division
from __future__ import unicode_literals

__author__ = 'Bin Ouyang & Fengyu_xie'
__version__ = 'Dev'

### Currently using simplest functions. Should integrate atomate surveillance functions soon!

import os
import sys
import json

def _run_vasp(RunDir):
    """
    Run vasp for all structures under RunDir.
    """
    absRunDir = os.path.abspath(RunDir)
    parentDir = os.path.dirname(absRunDir)
    POSDirs=[];
    if not os.path.isfile(os.path.join(parentdir,'sub.sh')):
        print("Submission script to computer clusters not provided, please provide one under job directory!")
        return

    _is_VASP_Input = lambda files: ('INCAR' in files) and \
                     ('POSCAR' in files) and ('POTCAR' in files)\
                     and ('KPOINTS' in files)

    for Root,Dirs,Files in os.walk(RunDir):
        if _is_VASP_Input(Files) and 'fm.0' in Root: POSDirs.append(Root);

    for Root in POSDirs:
        runRoot = os.path.abspath(Root)
        os.chdir(runRoot)
        print("Submitting VASP for {}".format(os.getcwd()))
        ### sub.sh is architecture dependent, and should be provided by the user.
        ### After integrating custodian we shouldn't be using it anymore!
        os.system('cp {}/sub.sh sub.sh'.format(parentDir))
        os.system('qsub sub.sh')
        print("Submitted VASP for {}".format(os.getcwd()))
        os.chdir(parentDir)
    print('Submission works done.')


