#!/bin/bash
#SBATCH --job-name=team_2_example_job
#SBATCH --mail-user=joehiggi@umich.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=1000m
#SBATCH --time=10:00
#SBATCH --account=siads696f25s012_class
#SBATCH --partition=standard
#SBATCH --output=/home/%u/%x-%j.log
# COMMENT:The application(s) to execute along with its input arguments and options:






