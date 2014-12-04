#!/bin/bash -l
# The name of the script is myjob
#SBATCH -J merging_data 

# Only 1 hour wall-clock time will be given to this job
#SBATCH -t 00:10:00

# Number of MPI tasks.
# always ask for complete nodes (i.e. mppwidth should normally
# be a multiple of 20)
#SBATCH -n 20

#SBATCH -e merge_error_file.e
#SBATCH -o merge_output_file.o
#load the nest module
module swap PrgEnv-cray PrgEnv-gnu
module add python
module add nest


#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/cfs/milner/scratch/b/berthet/modules/bcpnndopa_module/build-module-130701
echo $LD_LIBRARY_PATH
export PYTHONPATH=/pdc/vol/nest/2.2.2/lib/python2.7/site-packages:/pdc/vol/python/2.7.6-gnu/lib/python2.7/site-packages
export TMP=$(pwd)

#echo "Starting at `date`"


rm merge_delme_output
rm merge_error_file.e

temp="merge_delme_output"
echo "Merging, `date`"
aprun -n 10 python /cfs/milner/scratch/b/berthet/code/dopabg/MergeSpikefiles.py > $temp 2>&1
echo "Stopping at `date`"
