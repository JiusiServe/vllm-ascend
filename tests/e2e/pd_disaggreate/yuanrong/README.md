# Overview: transfer KVCache through host memory with YuanRongConnector

### Dataflow


|----------------------- prefill node ---------------------| &nbsp; |---------------------- decode node ---------------------|

Prefill Instance  -----> YuanRongConnector -----> YuanRong Data Worker -----> YuanRongConnector -----> Decode Instance

|----- kv on npu -----| &nbsp; |----- kv offload to host -----| &nbsp; |----- kv transfer by host net -----| &nbsp; |----- kv load to npu -----|

### Pros
- Network jitter and failures are handled outside of the vLLM process, better isolation and fault tolerance
- No need to allocate communication buffers on NPU, enable a larger sequence batch and throughput
- Work seamlessly with features those require offloading kvcache to host memory or SSD, like prefix cache, priority scheduling, RAG, etc.
### Cons
- Higher transfer latency compared with device-to-device transfer, not optimal for latency-sensitive scenarios





# Installation

## Install etcd
#### 1. Download the latest binaries from [etcd github releases](https://github.com/etcd-io/etcd/releases)
```
ETCD_VERSION="v3.5.12"  
wget https://github.com/etcd-io/etcd/releases/download/${ETCD_VERSION}/etcd-${ETCD_VERSION}-linux-amd64.tar.gz
```
#### 2. Unzip and install
```
tar -xvf etcd-${ETCD_VERSION}-linux-amd64.tar.gz
cd etcd-${ETCD_VERSION}-linux-amd64
# copy the binary to system
sudo cp etcd etcdctl /usr/local/bin/
```
#### 3. Verify installation
```
etcd --version
etcdctl version
```


## Install openyuanrong-datasystem
#### Install from pip (recommended):

```
pip install openyuanrong-datasystem
```

#### Or install from source:

- Refer to the openyuanrong-datasystem documentation [here](https://gitee.com/openeuler/yuanrong-datasystem)



# Deployment
## Deploy etcd
> Note: this is the minimal example to deploy etcd, more can be found at the [etcd official site](https://etcd.io/docs/current/op-guide/clustering/).

#### Deploy a single node etcd cluster at port 2379:
```
etcd \
  --name etcd-single \
  --data-dir /tmp/etcd-data \
  --listen-client-urls http://0.0.0.0:2379 \
  --advertise-client-urls http://0.0.0.0:2379 \
  --listen-peer-urls http://0.0.0.0:2380 \
  --initial-advertise-peer-urls http://0.0.0.0:2380 \
  --initial-cluster etcd-single=http://0.0.0.0:2380
```


#### Parameters:
- --name：cluster name
- --data-dir：directory to store data
- --listen-client-urls：address to listen from clients (0.0.0.0 allows access from any IP address)
- --advertise-client-urls：address advertised to clients
- --listen-peer-urls：address to listen from other nodes in the cluster
- --initial-advertise-peer-urls：address advertised to other nodes in the cluster
- --initial-cluster：initial nodes in the cluster (format: name1=peer_url1,name2=peer_url2,...)

#### Try to access the etcd cluster with the `etcdctl` command:
```
etcdctl --endpoints "127.0.0.1:2379" put key "value"
etcdctl --endpoints "127.0.0.1:2379" get key
```
etcd cluster is successfully deployed if the commands work good.

## Deploy openyuanrong-datasystem
#### Deploy a single node openyuanrong-datasystem cluster with the minimum config:
```
dscli start -w --worker_address "127.0.0.1:31501" --etcd_address "127.0.0.1:2379"
# [INFO] [  OK  ] Start worker service @ 127.0.0.1:31501 success, PID: 38100
```
openyuanrong-datasystem is deployed successful as you see the `[  OK  ]` output.

#### To safely stop and clean the openyuanrong-datasystem processes, run the command:
```
dscli stop -w --worker_address "127.0.0.1:31501"
```
#### Please refer to the [openyuanrong-datasystem gitee repo](https://gitee.com/openeuler/yuanrong-datasystem) for more information.

# Run disaggregated prefill with vLLM v1

> Note: an example script for 1P1D disaggregated prefill is available at: *vllm-ascend/tests/e2e/pd_disaggregate/yuanrong/test_yuanrong_connector.py*

#### 1. Populate the openyuanrong-datasystem worker address with environment variable:

`export DS_WORKER_ADDR=127.0.0.1:31501`

YuanRongConnector will read the openyuanrong-datasystem address from this environment variable

#### 2. Start two vLLM instances with YuanRongConnector as the backend to form a 1P1D disaggregated cluster:
```
export VLLM_USE_V1=True

# start a prefill instance on localhost:8100
ASCEND_RT_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen2.5-7B-Instruct \
    --port 8100 \
    --max-num-batched-tokens 45000 \
    --gpu-memory-utilization 0.8 \
    --trust-remote-code \
    --enforce-eager \
    --kv-transfer-config \
    '{"kv_connector":"YuanRongConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":2}' &

# start a decode instance on localhost:8200
ASCEND_RT_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen2.5-7B-Instruct \
    --port 8200 \
    --max-num-batched-tokens 45000 \
    --gpu-memory-utilization 0.8 \
    --trust-remote-code \
    --enforce-eager \
    --kv-transfer-config \
    '{"kv_connector":"YuanRongConnector","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":2}' &
```

#### 3. Start a proxy server to serve and route HTTP requests:
```
python vllm-ascend/tests/e2e/pd_disaggregate/yuanrong/simple_pd_proxy_server.py --prefiller-port 8100 --decoder-port 8200
```

#### 4. Send HTTP requests to the proxy server:
```
curl -X POST -s http://localhost:8000/v1/completions \
-H "Content-Type: application/json" \
-d '{
"model": "Qwen/Qwen2.5-7B-Instruct",
"prompt": "who is the presiden of the united states?",
"max_tokens": 50,
"temperature": 0
}'
```
