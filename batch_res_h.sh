#!/bin/bash
#SBATCH --cpus-per-task=12
#SBATCH --partition=gpu-h100
#SBATCH --qos=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --mail-type=ALL
#SBATCH --reservation=gpu-h100
#SBATCH --mem=82G
#SBATCH --mail-user=c.d.kennedy@sheffield.ac.uk
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

module load Anaconda3/2022.10
source activate torchbenchmark-${1}-n 

batch_size=$2
model_version=$3
num_runs=${4:-10}

# Now call python script with these arguments

python3 ../ResNet.py -v $model_version -b $batch_size -n $num_runs

# python ResNet.py --model_version 50 --batch_size 256 --num_runs 10
# python ResNet.py -v $1 -b $2 -n 10

