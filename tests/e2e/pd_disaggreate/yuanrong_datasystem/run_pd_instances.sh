#!/bin/bash

MODEL_NAME=$1
HOST_IP=$2
PREFILL_PORT=$3
DECODE_PORT=$4
PREFILL_DEVICES=$5
DECODE_DEVICES=$6

if python -c "import datasystem" &> /dev/null; then
    echo "openyuanrong-datasystem is already installed"
else
    echo "Install openyuanrong-datasystem ..."
    python -m pip install openyuanrong-datasystem
fi

wait_for_server() {
    local port=$1
    timeout 1200 bash -c "
        until curl -s ${HOST_IP}:${port}/v1/completions > /dev/null; do
            sleep 1
        done" && return 0 || return 1
}

ASCEND_RT_VISIBLE_DEVICES=$PREFILL_DEVICES vllm serve $MODEL_NAME \
    --host ${HOST_IP} \
    --port ${PREFILL_PORT} \
    --max-num-batched-tokens 45000 \
    --gpu-memory-utilization 0.8 \
    --max-model-len 2048 \
    --trust-remote-code \
    --enforce-eager \
    --kv-transfer-config \
    '{"kv_connector":"YuanRongConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":2}' &

ASCEND_RT_VISIBLE_DEVICES=$DECODE_DEVICES vllm serve $MODEL_NAME \
    --host ${HOST_IP} \
    --port ${DECODE_PORT} \
    --max-num-batched-tokens 45000 \
    --gpu-memory-utilization 0.8 \
    --max-model-len 2048 \
    --trust-remote-code \
    --enforce-eager \
    --kv-transfer-config \
    '{"kv_connector":"YuanRongConnector","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":2}' &

wait_for_server ${PREFILL_PORT}
wait_for_server ${DECODE_PORT}
