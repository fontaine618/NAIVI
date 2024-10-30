#!/bin/bash
# The interpreter used to execute the script
#“#SBATCH” directives that convey submission options:
#SBATCH --job-name=naivi_cora_selection
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=simfont@umich.edu
#SBATCH --time=20:00:00
#SBATCH --array=0
#SBATCH --account=open
#SBATCH --partition=open
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=8g
# The application(s) to execute along with its input arguments and options:
module load anaconda/2023.09
conda activate naivi
python -O run.py $SLURM_ARRAY_TASK_ID