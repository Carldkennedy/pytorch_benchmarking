#!/bin/bash
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=fetch_data_CIFAR10.%j.out
#SBATCH --comment=fetch_data_CIFAR10

module load Anaconda3/2022.10
source activate torchbenchmark-11.8-n


# Now call python script 

python3 setup/fetch_data_CIFAR10.py