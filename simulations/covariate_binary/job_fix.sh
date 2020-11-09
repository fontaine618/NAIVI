#!/bin/bash
# The interpreter used to execute the script
#“#SBATCH” directives that convey submission options:
#SBATCH --job-name=nnvi_covariate_binary_fix
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=simfont@umich.edu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=300m
#SBATCH --time=10:00:00
#SBATCH --account=stats_dept1
#SBATCH --partition=standard
#SBATCH --output=/home/%u/%x-%j.log
# The application(s) to execute along with its input arguments and options:
source /home/simfont/scratch/NNVI/venv/bin/activate
python3 main_fix.py

