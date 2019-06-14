#!/bin/bash
# List all the jobs
# jobid jobfolder
qstat -u fengyu_xie > jobstat;
LineNum=`wc -l jobstat | awk '{printf $1}'`
LineSta=3; #Start from the 6th lines
echo -e '================================================================';
echo -e 'Path \t Requested/Started time \t Status \tNSlot  \t JOBID';
echo -e '================================================================';
for LineInd in `seq ${LineSta} ${LineNum}`
do
    jobid=`sed -n "${LineInd}p" jobstat | awk {'printf "%i", $1'}`;
    path=`qstat -j ${jobid} | grep cwd | awk '{printf "%s \n", $2}'`;
    Status=`sed -n "${LineInd}p" jobstat | awk {'printf "%s", $5'}`;
    JobDate=`sed -n "${LineInd}p" jobstat | awk {'printf "%s", $6'}`;
    JobTime=`sed -n "${LineInd}p" jobstat | awk {'printf "%s", $7'}`;
    NSlot=`sed -n "${LineInd}p" jobstat | awk {'printf "%i", $8'}`;
 echo -e ${path} '\t' ${JobDate} ${JobTime} '\t' ${Status} '\t' ${NSlot} '\t' ${jobid};
done
rm jobstat;
