#!/bin/bash

########################## Edit as required ############################
folder_name="standard"                      # Choose a name for this run
gpus=("a" "h")                              # options a,h
cudas=("11.8" "12.1")                       # options 11.8,12.1
models=("18" "50" "152")                    # options 18,50,152
batch_sizes=("32" "64" "128" "256" "512")   # options 32,64,128,256,512
num_runs=10                                 # (default:10)
########################################################################

job_ids=() # Array to store job IDs

# Attempt to create the folder and navigate into it
eval "mkdir -p $folder_name && pushd $folder_name" || { echo "Failed to create folder $folder_name. Exiting. Please choose another folder_name"; exit 1; }
echo "Created and navigated into folder: $folder_name"

# Save a copy of scripts
echo "Backing up scripts"
mkdir -p scripts && cp ../batch_res_a.sh ../batch_res_h.sh ../plot.py ../plot.sh ../ResNet.py ../Stats.sh ../submission.sh scripts/

# Submit batch jobs
echo "Submiiting batch jobs"

for gpu in "${gpus[@]}"; do
    for cuda in "${cudas[@]}"; do 

        GPU=${gpu^^} # Capitalize gpu
                
        for model in "${models[@]}"; do
        
            # Assign hours based on ResNet model (18 4hrs; 50 6hrs; 152 10hrs)
            hours=$(case $model in 18) echo 4;; 50) echo 6;; 152) echo 10;; *) echo 0;; esac)
            
            for batch_size in "${batch_sizes[@]}"; do
                jid=$(sbatch --job-name=${GPU}100_${cuda}_${model}_${batch_size} \
                --time=${hours}:00:00 ../batch_res_${gpu}.sh $cuda $batch_size $model $num_runs)
                echo $jid
                jid=$(echo $jid | tr -dc '0-9')
                job_ids+=($jid)
            done
        done
    done
done

echo "Plotting batch job will submit when all models finished"

# Construct the dependency string for the final job
dependency=""
for jid in "${job_ids[@]}"; do
  dependency+="$jid:"
done
dependency=${dependency%:}  # Remove the trailing colon

# Collate Stats and Plot once all jobs finished
sbatch --dependency=afterany:$dependency ../plot.sh $num_runs

popd
