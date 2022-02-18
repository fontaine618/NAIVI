#!/bin/bash
# The interpreter used to execute the script
#“#SBATCH” directives that convey submission options:
#SBATCH --job-name=estimation_N_gpu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=simfont@umich.edu
#SBATCH --time=24:00:00
#SBATCH --account=stats_dept1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-gpu=20g
#SBATCH --output=/home/%u/scratch/logs/%x-%A-%a.log
# The application(s) to execute along with its input arguments and options:
module load python/3.8.7 gcc/9.2.0
source /home/simfont/naivi/bin/activate
python estimation_N.py 0
