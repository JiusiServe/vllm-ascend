import json
import random
import os
import time
import asyncio
import argparse
from datasets import load_dataset
from vllm import SamplingParams
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs


os.environ["VLLM_USE_MODELSCOPE"] = "False"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


async def generate_async(engine, request_id, prompt, sampling_params):
    results_generator = engine.generate(prompt, sampling_params, request_id)
    final_output = None
    async for request_output in results_generator:
        final_output = request_output
    return final_output


async def send_requests_with_rate_limit(engine, prompts, sampling_params, requests_per_second=15):
    """Send requests at a controlled rate (requests per second)"""
    tasks = []
    interval = 1.0 / requests_per_second  # Time interval between requests (in seconds)

    for i, prompt in enumerate(prompts):
        request_id = f"request_{i}"

        # Create task
        task = asyncio.create_task(
            generate_async(engine, request_id, prompt, sampling_params)
        )
        tasks.append(task)

        # Wait between requests (except for the last one)
        if i < len(prompts) - 1:
            await asyncio.sleep(interval)

    return tasks


async def main_ewsjf(sampling_params, prompts, rates, model_name, tensor_parallel_size):
    engine_args = AsyncEngineArgs(
        model=model_name,
        scheduler_cls="ewsjf",
        tensor_parallel_size=tensor_parallel_size
    )

    for rate in rates:
        await run_engine(engine_args, prompts, sampling_params, rate, "EWSJF")


async def main_fcfs(sampling_params, prompts, rates, model_name, tensor_parallel_size):
    engine_args = AsyncEngineArgs(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size
    )

    for rate in rates:
        await run_engine(engine_args, prompts, sampling_params, rate, "FCFS")


async def main(sampling_params, prompts, rates, model_name, tensor_parallel_size, mode):
    if mode == "fcfs":
        await main_fcfs(sampling_params, prompts, rates, model_name, tensor_parallel_size)
    elif mode == "ewsjf":
        await main_ewsjf(sampling_params, prompts, rates, model_name, tensor_parallel_size)
    else:
        await main_ewsjf(sampling_params, prompts, rates, model_name, tensor_parallel_size)
        await main_fcfs(sampling_params, prompts, rates, model_name, tensor_parallel_size)


async def run_engine(engine_args, prompts, sampling_params, rate, scheduler_name):
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    start = time.time()
    print(f"Sending {len(prompts)} requests at rate of {rate} requests/sec...")
    print(f"Estimated time to send all requests: {len(prompts) / rate:.2f} seconds")
    # Send requests with rate limiting
    tasks = await send_requests_with_rate_limit(engine, prompts, sampling_params, requests_per_second=rate)
    # Wait for all requests to complete
    print("Waiting for all requests to complete...")
    outputs = await asyncio.gather(*tasks)
    end = time.time()

    duration = end - start

    await metrics(duration, prompts, outputs, scheduler_name)

    print("\nGenerated Outputs:\n" + "-" * 60)
    print("Num answers: ", len(outputs))
    print(f"\nRequest sending time: {len(prompts) / rate:.2f} seconds")
    print(f"Total execution time (until all requests complete): {duration:.2f} seconds")


async def metrics(total_runtime, prompts, outputs, scheduler_name):
    num_requests = len(prompts)
    total_generated_tokens = 0
    total_prompt_tokens = 0

    print("\n--- Aggregating Metrics ---")
    print(f"\n --- Scheduler: {scheduler_name} ---")
    for output in outputs:
        prompt_len = len(output.prompt_token_ids)
        generated_len = len(output.outputs[0].token_ids)

        total_prompt_tokens += prompt_len
        total_generated_tokens += generated_len

    requests_per_second = num_requests / total_runtime if total_runtime > 0 else 0
    output_tokens_per_second = total_generated_tokens / total_runtime if total_runtime > 0 else 0

    print("\n--- Benchmark Metrics ---")
    print(f"Total time: {total_runtime:.2f} sec")
    print(f"Total prompts: {num_requests}")
    print(f"Prompt tokens: {total_prompt_tokens}")
    print(f"Generated tokens: {total_generated_tokens}")
    print("---------------------------------------------")
    print(f"Requests/sec: {requests_per_second:.2f}")
    print(f"Output tokens/sec: {output_tokens_per_second:.2f}")
    print("NOTE: No TTFT/per-request latency in this build.")


# ------------------------------------------------------
#                    CLI INTERFACE
# ------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="General LLM Benchmark Tool using vLLM (FCFS / EWSJF)"
    )

    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="Model name or path")
    parser.add_argument("--tp", type=int, default=1,
                        help="Tensor parallel size")
    parser.add_argument("--dataset", type=str, default="ChayaLevi/data-100-2000",
                        help="HuggingFace dataset name containing 'input' column")
    parser.add_argument("--max-prompts", type=int, default=None,
                        help="Limit number of prompts")
    parser.add_argument("--rates", type=str, default="100",
                        help="Comma-separated list of request rates, e.g. 1000,500,100")
    parser.add_argument("--mode", type=str, default="both",
                        choices=["fcfs", "ewsjf", "both"],
                        help="Scheduler: fcfs / ewsjf / both")

    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=100)

    args = parser.parse_args()

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        min_tokens=1
    )

    dataset = load_dataset(args.dataset)
    prompts = list(dataset["train"]["input"])

    if args.max_prompts:
        prompts = prompts[:args.max_prompts]

    rates = [int(x) for x in args.rates.split(",")]

    asyncio.run(main(
        sampling_params,
        prompts,
        rates,
        args.model,
        args.tp,
        args.mode
    ))
