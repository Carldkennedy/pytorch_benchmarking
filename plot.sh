#!/bin/bash
#SBATCH --mem=2G
#SBATCH --cpus-per-task=2
#SBATCH --mail-user=c.d.kennedy@sheffield.ac.uk
#SBATCH --mail-type=ALL
#SBATCH --output=plot.%j.out
#SBATCH --comment=plotting
#SBATCH --time=00:00:45

module load Anaconda3/2022.10
source ../Stats.sh &&
source activate plotter &&
python ../plot.py
