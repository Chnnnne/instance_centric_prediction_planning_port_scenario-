#!/usr/bin/env bash

set -x
NGPUS=$1

export OMP_NUM_THREADS=2
while true
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done
echo $PORT

python -m torch.distributed.run --nproc_per_node=${NGPUS} --rdzv_endpoint=localhost:${PORT} main.py

# torchrun --nproc_per_node=${NGPUS} --rdzv_endpoint=localhost:${PORT} train.py --launcher pytorch ${PY_ARGS}


