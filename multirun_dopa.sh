rm -r Test/
rm fig*.pdf
rm error_file.e
rm delme_output_*

N=$1
var1=40
COUNTER=0

# DO NOT FORGET TO CHANGE WALLTIME in call file


for i in `seq 0 $N`;
do 
    sbatch /cfs/milner/scratch/b/berthet/code/dopabg/dopa_bgsim.sh $var1 $COUNTER $N
    echo "job "$COUNTER " submitted"
    sleep 0.5
    COUNTER=`expr $COUNTER + 1`
done





