#!/bin/bash
# The interpreter used to execute the script
#“#SBATCH” directives that convey submission options:
#SBATCH --job-name=nnvi_covariate_binary
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=simfont@umich.edu
#SBATCH --time=100:00:00
#SBATCH --account=stats_dept1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2000m
#SBATCH --gpus-per-node=1
#SBATCH --output=/home/%u/%x-%j.log
# The application(s) to execute along with its input arguments and options:
source /home/simfont/scratch/NNVI/venv/bin/activate
python3 main.py

