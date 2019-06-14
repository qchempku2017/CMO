#!/usr/bin/env python

import os
import sys

_is_vasp_input = lambda files: ('INCAR' in files) and ('POSCAR' in files)\
and ('POTCAR' in files) and ('KPOINTS' in files)

_have_vasp_output = lambda files: ('OUTCAR' in files)

def _is_successful_vasp(root,files):
    if not _is_vasp_input(files):
        return False
    if not _have_vasp_output(files):
        return False
    out_path = os.path.join(root,'OUTCAR')
    with open(out_path) as outfile:
        out_string = outfile.read()
        if 'reached required accuracy' not in out_string:
            print('Instance {} did not converge.'.format(root))
            return False
    return True

fails = []
n_fails = 0
n_success = 0
for root, dirs, files in os.walk('./vasp_run'):
    if _is_successful_vasp(root,files):
        print("Instance {} is successful!".format(root))
        n_success += 1
    else:
       if _is_vasp_input(files) and _have_vasp_output(files):
           fails.append(root)
           n_fails += 1

print("Failed structures:\n",fails)
print("Fail rate {}/{}".format(n_fails,n_fails+n_success))
