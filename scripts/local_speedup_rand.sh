# Runs learning process for different number of processors, plots resulting speedup
# Random graph version

# Usage:
# $1: n_nodes
# $2: edge_prob
# $3: seed
# $4: n_iter
# $5: lr

mpiexec --mca mpi_warn_on_fork 0 -np 1 python3 python/main.py --n_nodes $1 --edge_prob $2 --seed $3 --n_iter 1  > out.txt
for n_proc in 2 3 4 5 6
do
    mpiexec --mca mpi_warn_on_fork 0 -np $n_proc python3 python/main.py --n_nodes $1 --edge_prob $2 --seed $3 --n_iter $4 --lr $5 >> out.txt
done
python3 python/plotSpeedup.py --n_nodes $1 --edge_prob $2 --seed $3 --n_iter $4 --lr $5