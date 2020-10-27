#!/bin/bash
# The interpreter used to execute the script
#“#SBATCH” directives that convey submission options:
#SBATCH --job-name=nnvi_missing_rate_binary
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=simfont@umich.edu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=200m
#SBATCH --time=100:00:00
#SBATCH --account=stats_dept1
#SBATCH --partition=standard
#SBATCH --output=/home/%u/%x-%j.log
# The application(s) to execute along with its input arguments and options:
source /home/simfont/scratch/NNVI/venv/bin/activate
python3 main.py

