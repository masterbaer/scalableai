#!/bin/bash

#SBATCH --job-name=kmeans_gpu                  # job name
#SBATCH --partition=gpu_4                 # queue for resource allocation
#SBATCH --time=30:00                       # wall-clock time limit  
#SBATCH --mem=30000                        # memory per node
#SBATCH --nodes=1                          # number of nodes to be used
#SBATCH --mail-type=ALL                    # Notify user by email when certain event types occur.
#SBATCH --mail-user=uivsx@student.kit.edu  # notification email address
#SBATCH --gres=gpu:1

module purge                                    # Unload all currently loaded modules.
module load devel/cuda/10.2                     # Load required modules.  
module load compiler/gnu/11.2
module load mpi/openmpi/4.1  
module load lib/hdf5/1.12.1-gnu-11.2-openmpi-4.1  
source /home/kit/stud/uivsx/scalableaivenv/bin/activate  # Activate your virtual environment.

srun python a1.py