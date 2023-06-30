#!/bin/bash
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=plotter-env.%j.out
#SBATCH --comment=plotter-env

module load Anaconda3/2022.10 &&
conda create --yes --name plotter python=3.10 &&
source activate plotter &&
echo "Installing pandas" &&
conda install --yes pandas &&
echo "Installing Matplotlib" &&
conda install --yes -c conda-forge matplotlib


