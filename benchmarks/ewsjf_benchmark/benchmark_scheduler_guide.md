# Benchmark Guide – vLLM Async Benchmark for Comparing FCFS and EWSJF Scheduling Under Varying Loads

## Overview

This document provides a complete guide for running the asynchronous vLLM benchmark script.
The benchmark is specifically designed to compare performance metrics between two scheduling strategies:

- **FCFS** – First Come First Served *(the default scheduler used by vLLM)*

- **EWSJF** – Equitable Weighted Shortest Job First

The benchmark evaluates each scheduler under increasing request rates and reports key metrics such as overall throughput, total runtime, prompt-token usage, and generated-token throughput. This allows a clear, data-driven comparison between the scheduling policies.


## The benchmark validates

The main purpose of this benchmark is to compare and evaluate the performance of two scheduling strategies—FCFS and EWSJF—under varying workload conditions. It enables a side-by-side comparison of the schedulers across different load levels (e.g., different RPS rates).

The benchmark also measures key performance metrics to facilitate this comparison:
- Throughput of prompt tokens and output tokens
- End-to-end runtime for full workload batches
- Validation of asynchronous request generation using `AsyncLLMEngine`
- System behavior under different load scenarios

## Environment Requirements

| Component       | Version / Notes                            |
|-----------------|--------------------------------------------|
| Python          | 3.9+                                       |
| Framework       | vLLM (async engine)                        |
| Dataset         | HuggingFace dataset with an "input" column |


## Install dependencies

```bash
pip install vllm datasets sortedcontainers
```

## Repository Setup

```bash
git clone <your-repo of vllm-ascend>
cd <your-repo of vllm-ascend>
pip install -e .
```

## Script Structure

Your benchmark script includes:

- **Async request pipeline**  
  `generate_async()`

- **Rate-limited request dispatcher**  
  `send_requests_with_rate_limit()`

- **Scheduler-specific execution**  
  `main_fcfs()`, `main_ewsjf()`

- **Combined mode**  
  `--mode both` → runs both schedulers sequentially

- **Metrics aggregation**  
  `metrics()`

The script prints detailed throughput after every benchmark sweep.


## Configuration

## Command-line Arguments

| Description                           | Argument          | Default                   |
|---------------------------------------|-------------------|---------------------------|
| Model name or path                    | `--model`         | `"Qwen/Qwen2.5-7B-Instruct"` |
| Tensor parallel size                  | `--tp`            | `1`                       |
| HuggingFace dataset name              | `--dataset`       | `"ChayaLevi/data-100-2000"` |
| Limit number of prompts               | `--max-prompts`   | `None`                    |
| Comma-separated request rates (e.g., 1000,500,100) | `--rates` | `"100"`                    |
| fcfs, ewsjf, or both                  | `--mode`          | `"both"`                  |
| Sampling temperature                  | `--temperature`   | `0.8`                     |
| Top-p sampling                        | `--top-p`         | `0.95`                    |
| Max output token count                | `--max-tokens`    | `100`                     |

### Example:

```bash
python benchmark.py \
  --model meta-llama/Llama-3-8B-Instruct \
  --tp 4 \
  --dataset your-dataset/name \
  --max-prompts 2000 \
  --rates 400,200,100 \
  --mode both \
  --temperature 0.8 \
  --top-p 0.95 \
  --max-tokens 64
```

## Dataset Preparation
The script loads:
```python
dataset = load_dataset(args.dataset)
prompts = list(dataset["train"]["input"])
```

The dataset must contain `"train"` split

The `"input"` column must contain text strings

Limit the number of prompts:

```bash
--max-prompts 1000
```

## Running the Benchmark
### Rate-Limited Request Sending

The benchmark uses a controlled request sender:
```python
tasks = await send_requests_with_rate_limit(
    engine,
    prompts,
    sampling_params,
    requests_per_second=rate
)
```

Features:

- Generates 1 async task per prompt

- Enforces a fixed interval based on RPS

- Prints estimated request-sending duration



## Schedulers
### 1. FCFS Mode

Run with:
```bach
--mode fcfs
```
Uses default scheduler in vLLM.

### 2. EWSJF Mode
Run with:
```bach
--mode ewsjf
```

Activated via:
```python
engine_args = AsyncEngineArgs(
    model=model_name,
    scheduler_cls="ewsjf",
    tensor_parallel_size=tensor_parallel_size
)
```

### 3. Both Schedulers
Run with:
```bach
--mode both
```
Equivalent to two full benchmark runs.


### EWSJF Scheduler Configuration

For the **EWSJF scheduler**, you can customize its behavior using a dedicated configuration file.  
The configuration file is located at:
`"vllm_ascend/core/ewsjf_scheduler/config.json"`


This file allows adjusting the following parameters:

---

### **1. `queues_config`**  
Initialization of queues with specific token-range boundaries.  
This field is a **list of dictionaries**, where each dictionary contains a `"boundaries"` key whose value is a list of **[min, max]** token limits for the queue.

**Example:**
```json
"queues_config": [
  { "boundaries": [100, 163] },
  { "boundaries": [164, 226] },
  { "boundaries": [227, 289] }
]
```

### **2. `step_size`**
Defines the range step used when generating new queues dynamically.
A larger step creates fewer, wider queues; a smaller step creates more fine-grained queues.

**Example:**
```json
"step_size": 1500
```

These parameters allow full customization of queue creation and behavior tuning for the EWSJF scheduling algorithm.


## Metrics Output

The script prints:
```bach
--- Benchmark Metrics ---
Total time: X.XX sec
Total prompts: N
Prompt tokens: T
Generated tokens: G
Requests/sec: R
Output tokens/sec: O
NOTE: No TTFT/per-request latency in this build.
```
Where:

**Requests/sec** = N / runtime

**Output tokens/sec** = total generated tokens / runtime

These metrics summarize overall system throughput under each simulated load pattern.

## Example Execution Summary
```bach
Sending 2000 requests at rate of 400 requests/sec...
Estimated time to send all requests: 5.00 seconds
Waiting for all requests to complete...

--- Benchmark Metrics ---
Total time: 8.42 sec
Total prompts: 2000
Prompt tokens: 123900
Generated tokens: 64000
Requests/sec: 237.47
Output tokens/sec: 7601.42
```

## Notes and Recommendations

- The script does not compute latency or TTFT for each request.

- For large workloads, consider increasing:

  - `--tp` (tensor parallelism)

  - batch size on datasets

  - Adjust RPS to find your hardware saturation point.