#!/bin/bash
# The interpreter used to execute the script
#“#SBATCH” directives that convey submission options:
#SBATCH --job-name=naivi_networksize_mcmc_2000
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=simfont@umich.edu
#SBATCH --time=100:00:00
#SBATCH --account=stats_dept1
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=100g
#SBATCH --output=/home/%u/%x-%j.log
# The application(s) to execute along with its input arguments and options:
module load python/3.8.7 gcc/8.2.0
source /home/simfont/naivi/bin/activate
python3 job_mcmc_2000.py

