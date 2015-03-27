#!/bin/bash -l
# The name of the script is myjob
#SBATCH -J testdelay_dBG 

# Only 1 hour wall-clock time will be given to this job
#SBATCH --time=01:59:59

# Number of MPI tasks.
# always ask for complete nodes (i.e. mppwidth should normally
# be a multiple of 20)
#SBATCH --nodes=1
####SBATCH -n 40

#SBATCH --error=error_file.e
#SBATCH --output=output_file.o
#load the nest module
module swap PrgEnv-cray PrgEnv-gnu
module add python
module add nest/2.2.2


#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/cfs/milner/scratch/b/berthet/modules/bcpnndopa_module/build-module-130701
echo $LD_LIBRARY_PATH
export PYTHONPATH=/pdc/vol/nest/2.2.2/lib/python2.7/site-packages:/pdc/vol/python/2.7.6-gnu/lib/python2.7/site-packages
export TMP=$(pwd)

#echo "Starting at `date`"


#rm delme_output_0
#rm error_file.e

#MULTIN=19
#COUNTER=0
#for i in `seq 0 $MULTIN`;
#do
#    for var1 in 40
#    do
temp="delme_output_$2"
echo "Job $2 running, `date` params: $1"
aprun -n 20 -N 20 -S 10 -j 1 python /cfs/milner/scratch/b/berthet/code/dopabg/main.py $1 $2 $3 > $temp 2>&1
#        COUNTER=`expr $COUNTER + 1`
#    done
#done
#wait
echo "Stopping at `date`"
