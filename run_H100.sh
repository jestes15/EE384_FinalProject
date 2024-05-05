#!/bin/bash
source /apps/profiles/modules_asax.sh.dyn
module load cuda
nvcc -lineinfo --use_fast_math -arch=sm_89 -O3 --extended-lambda fm_encode.cu -o fm_encode -lcufft -DNSYS_COMP
ncu -f -o profile_output --set full --import-source yes ./fm_encode