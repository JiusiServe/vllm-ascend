#!/bin/bash
PROXY_SERVER_SCRIPT=$1
PROXY_HOST=$2
PROXY_PORT=$3
PREFILL_HOST=$4
DECODE_HOST=$5
PREFILL_PORT=$6
DECODE_PORT=$7

python ${PROXY_SERVER_SCRIPT} \
    --host ${PROXY_HOST} \
    --port ${PROXY_PORT} \
    --prefiller-host ${PREFILL_HOST} \
    --prefiller-port ${PREFILL_PORT} \
    --decoder-host ${DECODE_HOST} \
    --decoder-port ${DECODE_PORT} &