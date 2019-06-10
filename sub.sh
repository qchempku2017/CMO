#!/bin/bash
#$ -cwd
#$ -j y
#$ -N aimd_Li3PS4
#$ -m es
#$ -V
#$ -pe impi 16
#$ -o ll_out
#$ -e ll_er
#$ -S /bin/bash

mpiexec.hydra -n $NSLOTS pvasp.5.4.4.intel >> vasp.out 
