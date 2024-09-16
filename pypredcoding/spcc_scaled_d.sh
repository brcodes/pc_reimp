#!/bin/bash
#$ -cwd
#$ -S /bin/bash
#$ -N spcc_scaled_d # job name
#$ -j n #merge stderr with stdout
#$ -V # preserve environment variables
#$ -pe smp 4 # number of cores
#$ -m e
#$ -M rogebryc@oregonstate.edu
#$ -q tarkus

/nfs3/PHARM/Brown_Lab/local/miniconda3_ppc64le/bin/python run_experiment_d.py