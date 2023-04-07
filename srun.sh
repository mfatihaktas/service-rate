#!/bin/bash
echo $1 $2 $3

if [ $1 = "i" ]; then
  srun --partition=main --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=4000 --time=8:00:00 --export=ALL --pty bash -i
  # srun --partition=main --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=16000 --time=12:00:00 --export=ALL --pty bash -i
  # srun --partition=main --nodes=1 --ntasks=1 --cpus-per-task=20 --mem=16000 --time=3:00:00 --export=ALL --pty bash -i

elif [ $1 = "j" ]; then
  # FILE="allocation_w_complexes/exp_prob_span_of_every_t_complexes_geq_u"
  # FILE="model/exp_random_design"
  # FILE="random_allocations/exp_prob_max_num_balls_leq_u"
  # FILE="service_rate/exp_plot_capacity_region"
  # FILE="sim/exp_mm1_stability"
  # FILE="storage_overlap/exp_design"
  FILE="storage_overlap/exp_impact_of_d"
  # FILE="storage_overlap/exp_random_design"
  # FILE="storage_search/exp_search_with_replicas_and_mds"
  # FILE="storage_opt/exp_single_obj_per_node"

  NTASKS=1
  echo "#!/bin/bash
#SBATCH --partition=main             # Partition (job queue)
#SBATCH --job-name=${FILE}
#SBATCH --nodes=${NTASKS}            # Number of nodes you require
#SBATCH --ntasks=${NTASKS}           # Total # of tasks across all nodes
#SBATCH --cpus-per-task=8            # Cores per task (>1 if multithread tasks)
#SBATCH --mem=8000                   # Real memory (RAM) required (MB)
#SBATCH --time=12:00:00              # Total run time limit (HH:MM:SS)
#SBATCH --export=ALL                 # Export your current env to the job env
#SBATCH --output=log/${FILE}.%N.%j.out
#SBATCH --error=log/${FILE}.%N.%j.err
#SBATCH -o log/srun.out

cd ${HOME}/service-rate

srun python ${PWD}/exp/${FILE}.py
  " > job_script.sh

  # rm log/*
  sbatch job_script.sh

elif [ $1 = "l" ]; then
  squeue -u mfa51

elif [ $1 = "k" ]; then
  scancel --user=mfa51 # -n learning

elif [ $1 = "node" ]; then
  sinfo --Node # --format="%n %f %c %m %G"

else
  echo "Did not match, arg= ${1}"
fi
