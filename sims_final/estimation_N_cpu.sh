#!/bin/bash
# The interpreter used to execute the script
#“#SBATCH” directives that convey submission options:
#SBATCH --job-name=estimation_N_cpu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=simfont@umich.edu
#SBATCH --time=01:00:00
#SBATCH --account=stats_dept1
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=0
#SBATCH --mem-per-cpu=12g
#SBATCH --output=/home/%u/scratch/logs/%x-%A-%a.log
# The application(s) to execute along with its input arguments and options:
module load python/3.8.7 gcc/9.2.0
source /home/simfont/naivi/bin/activate
python estimation_N_cpu.py $SLURM_ARRAY_TASK_ID