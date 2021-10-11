#!/bin/bash
# The interpreter used to execute the script
#“#SBATCH” directives that convey submission options:
#SBATCH --job-name=naivi_networksize_mcmc_1000
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=simfont@umich.edu
#SBATCH --time=100:00:00
#SBATCH --account=stats_dept1
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=10g
#SBATCH --output=/home/%u/%x-%j.log
# The application(s) to execute along with its input arguments and options:
source /home/simfont/naivi/bin/activate
python3 job_mcmc_1000.py

