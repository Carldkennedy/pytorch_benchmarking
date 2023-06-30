#!/bin/bash
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=torchbenchmark-12.1-env.%j.out
#SBATCH --comment=torchbenmark-12.1-env

module load Anaconda3/2022.10 &&
#conda create --yes --name torchbenchmark-12.1-n python=3.10 &&
source activate torchbenchmark-12.1-n &&
conda install --yes pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch-nightly -c nvidia
