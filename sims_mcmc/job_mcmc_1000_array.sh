#!/bin/bash
# The interpreter used to execute the script
#“#SBATCH” directives that convey submission options:
#SBATCH --job-name=naivi_networksize_mcmc_1000
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=simfont@umich.edu
#SBATCH --time=10:00:00
#SBATCH --account=stats_dept1
#SBATCH --partition=largemem
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=80g
#SBATCH --output=/home/%u/%x-%j.log
# The application(s) to execute along with its input arguments and options:
module load python/3.8.7 gcc/9.2.0
source /home/simfont/naivi/bin/activate
python3 job_mcmc_1000.py $SLURM_ARRAY_TASK_ID

