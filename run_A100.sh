#!/bin/bash
source /apps/profiles/modules_asax.sh.dyn
module load cuda
nvcc -lineinfo --use_fast_math -arch=sm_80 -O3 --extended-lambda fm_encode.cu -o fm_encode -lcufft
./fm_encode