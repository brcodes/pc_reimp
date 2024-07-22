#!/bin/bash
#$ -cwd
#$ -S /bin/bash
#$ -N sPCC_stitch_krkU_1x10e-3_ep250_rUiters100_rinit0s. # job name
#$ -j n #merge stderr with stdout
#$ -V # preserve environment variables
#$ -pe smp 4 # number of cores
#$ -m e
#$ -M rogebryc@oregonstate.edu
#$ -q tarkus

/nfs3/PHARM/Brown_Lab/local/miniconda3_ppc64le/bin/python main10e-3.py