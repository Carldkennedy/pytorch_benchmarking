#!/bin/bash
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=torchbencmark-11.8-env.%j.out
#SBATCH --comment=torchbenmark-11.8-env

module load Anaconda3/2022.10 &&
conda create --yes --name torchbenchmark-11.8-n python=3.10 &&
source activate torchbenchmark-11.8-n &&
conda install --yes pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch-nightly -c nvidia
