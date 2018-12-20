# Script for submitting speedup jobs to PBS job scheduler 
# Parallel permutation graph version

# Usage:
# $1: L
# $2: w
# $3: seed
# $4: n_iter
# $5: lr
# $6: job_name

# Submit single processor job
qsub -l procs=1 -N $6 -v L=$1,w=$2,seed=$3,n_iter=1,lr=$5,n_procs=1 run_learn_par.pbs

# Submit multi-processor jobs
for n_procs in 4 8 16 24 32
do
    qsub -l procs=$n_procs -N $6 -v L=$1,w=$2,seed=$3,n_iter=$4,lr=$5,n_procs=$n_procs run_learn_par.pbs
done
