# Benchmarking GPUs on Stanage with Pytorch

The purpose of this repository and the two scripts `setup.sh` and `submission.sh` is to enable you to benchmark 
both A100 and H100 GPUs on the Sheffield University HPC Stanage cluster. Benchmarking is performed with ResNet 18,50 and 152
trained on the CIFAR10 dataset with a sweep of batch sizes.

## Create environments and download data:

    source setup.sh
    
To avoid filling your home directory add to your `.condarc` 

    pkgs_dirs:
    - /mnt/parscratch/users/$USER/anaconda/.pkg-cache

    envs_dirs:
    - /mnt/parscratch/users/$USER/anaconda/.envs


## Submit many jobs:
    
    source submission.sh

Please edit the preamble of submission.sh to your requirements:

    ########################## Edit as required ############################
    folder_name="standard"                      # Choose a name for this run
    gpus=("a" "h")                              # options a,h
    cudas=("11.8" "12.1")                       # options 11.8,12.1
    models=("18" "50" "152")                    # options 18,50,152
    batch_sizes=("32" "64" "128" "256" "512")   # options 32,64,128,256,512
    num_runs=10                                 # (default:10)
    ########################################################################

Please change folder_name apporpriately for each submission. Any further runs with the same folder_name will output to the same directory.
A copy of all scripts will be saved along with the output.  

### Sbatch submission:
    
    batch_res_a.sh
    batch_res_h.sh
    
#### ResNet:
    
    ResNet.py

### Create plots:

    sbatch plot.sh

Calls the following scripts:

#### Collate statistics from output files:

    Stats.sh

#### Plot and each ResNet Model for all GPU and Cuda versions
        
    plot.py

## Directory structure

    pytorch_benchmarking
    |____batch_res_a.sh
    |____batch_res_h.sh
    |____plot.sh
    |____ResNet.py
    |____setup.sh
    |____submission_original.sh
    |____submission.sh
    |____setup
    | |____fetch_data_CIFAR10.py
    | |____torchbenchmark-12.1-env.47547.out
    | |____env_11.8_n.sh
    | |____fetch_data_CIFAR10.sh
    | |____env_12.1_n.sh
    | |____env_plotter.sh

## Initial Results

![](https://github.com/Carldkennedy/pytorch_benchmarking/blob/main/intialResults/ResNet_all.png?raw=True)
