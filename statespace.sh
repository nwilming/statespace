#!/bin/sh
# embedded options to qsub - start with #PBS
# walltime: defines maximum lifetime of a job
# nodes/ppn: how many nodes (usually 1)? how many cores?

   #PBS -q batch
   #PBS -l walltime=5:00:00
   #PBS -l nodes=1:ppn=1
   #PBS -l mem=5gb

# -- run in the current working (submission) directory --
cd $PBS_O_WORKDIR

chmod g=wx $PBS_JOBNAME

# FILE TO EXECUTE
python analysis.py --freq 6. --glob-str P%02i_*freq.datamat $PBS_ARRAYID 1> jobs/$PBS_JOBID.out 2> jobs/$PBS_JOBID.err
python analysis.py --freq 26.5 --glob-str P%02i_*freq.datamat $PBS_ARRAYID 1> jobs/$PBS_JOBID.out 2> jobs/$PBS_JOBID.err
