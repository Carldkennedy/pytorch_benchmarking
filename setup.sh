#!/bin/bash

sbatch() {
/usr/bin/sbatch "$@" | cut -d " " -f4
}

jid1=$(sbatch setup/env_11.8_n.sh)
jid2=$(sbatch --dependency=afterany:$jid1 setup/env_12.1_n.sh)
jid3=$(sbatch --dependency=afterany:$jid2 setup/env_plotter.sh)
jid4=$(sbatch --dependency=afterany:$jid3 setup/fetch_data_CIFAR10.sh)

