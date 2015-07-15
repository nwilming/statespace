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

for freq in 11 22 70
do
    `printf 'Subject $i with frequency $i\n' $PBS_ARRAYID $freq`
    export QFILE=`printf 'data/P%02i_stim_all_freq_FR%i.0_Ncomp3.trajectory' $PBS_ARRAYID $freq`
    python analysis.py --freq $freq --glob-str P%02i_*stim_all_freq.datamat -s motor --suffix Ncomp3 $PBS_ARRAYID 1> jobs/$PBS_JOBID.out 2> jobs/$PBS_JOBID.err
    python analysis.py --freq $freq --glob-str P%02i_*_all_freq.datamat -s motor --load-Q $QFILE  --suffix Ncomp3 $PBS_ARRAYID 1> jobs/$PBS_JOBID.out 2> jobs/$PBS_JOBID.err
done

