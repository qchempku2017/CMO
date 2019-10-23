#!/usr/bin/env python
"""
CEAuto main program.
"""

from jobs import *
import argparse

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-O','--options_file', help="A yaml file that defines all options for the CE run.",\
                        type = str, default='ce_options.yaml')
    parser.add_argument('-J','--job_type', help="Task to be preformed by CEAuto.(0=full_ce,1=fit_only,2=gs)", type = int,\
                        default='0')
    args = parser.parse_args()

    import time

    init_time = time.time()

    if args.job_type in [0,1]:
        ce_job = CEJob.from_options(args.options_file)
        ce_job.run_ce()
    elif args.job_type==2:
        gs_job = GSCanonicalJob.from_options(args.options_file)
        #Use GSoptions.yaml
        gs_job.run_gs()

    print("--- %s seconds ---" % (time.time() - init_time)) 
        
