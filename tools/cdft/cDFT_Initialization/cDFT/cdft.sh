#!/bin/bash
#PBS -N cdft
#PBS -l walltime=99999:00:00
#PBS -l nodes=cpu:ppn=96
#PBS -j oe


cd /home/user/mof-spherical/test1/C2H4C2H6/
input_path="./inputCOF/"
num_processes=40
timeout=10000
#processed=0
#successful=0
num_jobs=1
#find "$input_path" -type f -name '*.dat' -print0 | xargs -0 -P $num_processes -I{} sh -c 'bash -c "my.sh {}"'

find "$input_path" -type f -name '*.dat' -print0 | xargs -0 -P $num_processes -I{} sh -c 'bash -c "source co.sh && $0 {}"' process_input 
