#!/bin/bash
# The interpreter used to execute the script
#“#SBATCH” directives that convey submission options:
#SBATCH --job-name=naivi_n_attributes_continuous
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=simfont@umich.edu
#SBATCH --time=2:00:00
#SBATCH --array=0
#SBATCH --account=open
#SBATCH --partition=open
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16g
# The application(s) to execute along with its input arguments and options:
module load python/3.10.4
source /work/naivi/venv/bin/activate
python -O run.py $SLURM_ARRAY_TASK_ID