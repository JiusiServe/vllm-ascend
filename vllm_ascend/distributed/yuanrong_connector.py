# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import enum
import hashlib
import os
import threading
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

import numpy
import torch
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole)
from vllm.distributed.parallel_state import get_tp_group, get_world_group
from vllm.logger import init_logger
from vllm.utils import split_host_port
from vllm.v1.attention.backends.mla.common import MLACommonMetadata
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.v1.request import Request

from datasystem import DsTensorClient, Future

# Configuration Constants
ENABLE_PREFIX_CACHING = int(os.environ.get("USING_PREFIX_CONNECTOR", 1))
FUTURE_TIMEOUT = int(os.getenv("FUTURE_TIMEOUT", 10000))
SYNC_FUTURE_TIMEOUT = int(os.getenv("SYNC_FUTURE_TIMEOUT", 1))
SLEEP_TIMEOUT = 0.005

logger = init_logger(f"vllm.{__name__}")


class RequestStatus(enum.IntEnum):
    """Enumeration for tracking the execution status of asynchronous requests."""
    WAITING = enum.auto()  # Request is still pending execution
    TIMEOUT = enum.auto()  # Request execution timed out
    FINISHED = enum.auto()  # Request completed successfully


@dataclass
class RequestTracker:
    """
    Tracks the state of a request, specifically for managing delayed saves 
    and token scheduling.
    """
    request_id: str
    token_ids: torch.Tensor
    block_ids: tuple[List[int], ...]
    num_scheduled_tokens: int

    @staticmethod
    def from_new_request(request_id: str, token_ids: torch.Tensor,
                         block_ids: List[int],
                         num_scheduled_tokens: int) -> "RequestTracker":
        """
        Creates a new RequestTracker instance from a fresh request.

        Args:
            request_id: The unique identifier of the request.
            token_ids: The token IDs associated with the prompt.
            block_ids: The list of physical block IDs allocated.
            num_scheduled_tokens: The number of tokens initially scheduled.
        """
        return RequestTracker(request_id=request_id,
                              token_ids=token_ids,
                              block_ids=block_ids,
                              num_scheduled_tokens=num_scheduled_tokens)

    def update(self, block_ids: tuple[List[int], ...],
               num_external_scheduled_tokens: int) -> None:
        """
        Updates the tracker with newly allocated blocks and scheduled tokens.

        Args:
            block_ids: New block IDs to append to the existing list.
            num_external_scheduled_tokens: Number of additional tokens scheduled.
        """
        self.block_ids[0].extend(block_ids[0])
        self.num_scheduled_tokens += num_external_scheduled_tokens


@dataclass
class ReqMeta:
    """
    Metadata for a single request used during KV cache transfer (save/load) operations.
    """
    request_id: str
    token_ids: numpy.ndarray
    block_ids: tuple[List[int], ...]
    request_rank: int
    skip_block_num: int
    ds_cached_block_num: int
    need_save: bool

    @staticmethod
    def make_meta(request_id: str, token_ids: List[int],
                  block_ids: tuple[List[int], ...], block_size: int,
                  request_rank: int, skip_block_num: int,
                  ds_cached_block_num: int, need_save: bool) -> "ReqMeta":
        """
        Factory method to create a ReqMeta instance with aligned block calculations.

        Args:
            request_id: Unique request identifier.
            token_ids: List of token IDs.
            block_ids: List of allocated block IDs.
            block_size: The size of each KV cache block.
            request_rank: The Tensor Parallel (TP) rank assigned to handle this request.
            skip_block_num: Number of blocks to skip (already computed/loaded).
            ds_cached_block_num: Number of blocks existing in the data system.
            need_save: Whether the KV cache for this request needs to be saved.

        Returns:
            A populated ReqMeta object.
        """
        # Calculate valid tokens aligned to block boundaries
        valid_num_tokens = align_to_block_size(len(token_ids), block_size)
        valid_block_ids_count = valid_num_tokens // block_size

        return ReqMeta(request_id=request_id,
                       token_ids=numpy.array(token_ids),
                       block_ids=block_ids[0][:valid_block_ids_count],
                       request_rank=request_rank,
                       skip_block_num=skip_block_num,
                       ds_cached_block_num=ds_cached_block_num,
                       need_save=need_save)


@dataclass
class YuanRongConnectorMetadata(KVConnectorMetadata):
    """
    Metadata container for the YuanRong KV Connector, holding a batch of requests.
    """
    requests: List[ReqMeta]

    def __init__(self, tp_size: int, block_size: int):
        """
        Args:
            tp_size: Tensor Parallelism size.
            block_size: Size of the KV cache block.
        """
        self.requests = []
        self.tp_size = tp_size
        self.request_rank = 0
        self._block_size = block_size

    def add_request(self,
                    request_id: str,
                    token_ids: List[int],
                    block_ids: List[int],
                    skip_block_num: int,
                    ds_cached_block_num: int,
                    need_save: bool = True) -> None:
        """
        Adds a request to the metadata batch and assigns it a TP rank in a round-robin fashion.
        """
        request_rank = self.request_rank % self.tp_size
        self.requests.append(
            ReqMeta.make_meta(request_id=request_id,
                              token_ids=token_ids,
                              block_ids=block_ids,
                              block_size=self._block_size,
                              request_rank=request_rank,
                              skip_block_num=skip_block_num,
                              ds_cached_block_num=ds_cached_block_num,
                              need_save=need_save))
        self.request_rank = request_rank + 1


@dataclass
class ReqState:
    """Tracks the internal state of pending async save/load requests."""
    num_pending: int = -1
    finished: bool = False


class AsyncHandler:
    """
    Manages asynchronous save and load operations for KV caches.
    Handles task submission, status polling, and cleanup via an asyncio event loop.
    """

    def __init__(self, role: KVConnectorRole, task_list: List[asyncio.Task]):
        """
        Args:
            role: The role of the connector (PRODUCER, CONSUMER, or WORKER).
                  Note: In current usage, this is passed as a boolean (is_producer).
            task_list: A list to register background asyncio tasks.
        """
        self._async_save_reqs: Dict[str, ReqState] = defaultdict(ReqState)
        self._async_load_reqs: Dict[str, ReqState] = defaultdict(ReqState)

        # Determine if this handler acts as a producer based on the role flag
        self._is_producer: bool = role

        self._finished_save_reqs: asyncio.Queue = asyncio.Queue()
        self._finished_load_reqs: asyncio.Queue = asyncio.Queue()
        self._future_save_list: asyncio.Queue = asyncio.Queue()
        self._future_load_list: asyncio.Queue = asyncio.Queue()

        loop = asyncio.get_event_loop()

        # Register background tasks based on role and configuration
        if self._is_producer or ENABLE_PREFIX_CACHING:
            task_list.append(loop.create_task(self.get_save_futures_async()))

        if not self._is_producer or ENABLE_PREFIX_CACHING:
            task_list.append(loop.create_task(self.get_load_futures_async()))

    async def get_save_futures_async(self) -> None:
        """Background loop to monitor and process save futures."""
        while True:
            try:
                # Process all currently queued futures
                q_size = self._future_save_list.qsize()
                for _ in range(q_size):
                    request_id, future = self._future_save_list.get_nowait()
                    res = get_future(future)
                    req_state = self._async_save_reqs[request_id]

                    if res == RequestStatus.FINISHED:
                        logger.info(
                            f"Request: {request_id} save task finished")
                        req_state.num_pending -= 1
                        # If all tasks for this request are done and marked finished
                        if req_state.finished and req_state.num_pending == 0:
                            self._finished_save_reqs.put_nowait(request_id)
                            del self._async_save_reqs[request_id]

                    elif res == RequestStatus.WAITING or not req_state.finished:
                        # Re-queue if waiting or not fully marked finished
                        self._future_save_list.put_nowait((request_id, future))

                    else:
                        logger.error(
                            f"Request: {request_id} save future timeout/failed, result: {res}"
                        )
                        self._finished_save_reqs.put_nowait(request_id)
                        del self._async_save_reqs[request_id]

                await asyncio.sleep(SLEEP_TIMEOUT)
            except Exception as e:
                logger.error(f"Failed to process save futures: {e}")

    async def get_load_futures_async(self) -> None:
        """Background loop to monitor and process load futures."""
        while True:
            try:
                q_size = self._future_load_list.qsize()
                for _ in range(q_size):
                    request_id, future = self._future_load_list.get_nowait()
                    res = get_future(future)
                    req_state = self._async_load_reqs[request_id]

                    if res == RequestStatus.FINISHED:
                        logger.info(
                            f"Request: {request_id} load task finished")
                        req_state.num_pending -= 1
                        if req_state.num_pending == 0:
                            self._finished_load_reqs.put_nowait(request_id)
                            del self._async_load_reqs[request_id]

                    elif res == RequestStatus.WAITING:
                        self._future_load_list.put_nowait((request_id, future))

                    else:
                        logger.error(
                            f"Request: {request_id} load future timeout/failed, result: {res}"
                        )
                        self._finished_load_reqs.put_nowait(request_id)
                        del self._async_load_reqs[request_id]

                await asyncio.sleep(SLEEP_TIMEOUT)
            except Exception as e:
                logger.error(f"Failed to process load futures: {e}")
                await asyncio.sleep(SLEEP_TIMEOUT)

    def add_save_request(self, request: ReqMeta, future_num: int) -> None:
        """Registers a new save request with the expected number of futures."""
        self._async_save_reqs[request.request_id].num_pending = future_num

    def add_load_request(self, request: ReqMeta, future_num: int) -> None:
        """Registers a new load request with the expected number of futures."""
        self._async_load_reqs[request.request_id].num_pending = future_num

    def add_save_future(self, request: ReqMeta, future: Future) -> None:
        """Queues a specific save future for a request."""
        self._future_save_list.put_nowait((request.request_id, future))

    def add_load_future(self, request: ReqMeta, future: Future) -> None:
        """Queues a specific load future for a request."""
        self._future_load_list.put_nowait((request.request_id, future))

    def get_save_finished(
            self, finished_request_ids: Set[str]) -> Optional[Set[str]]:
        """
        Checks which requests have completed their save operations.
        
        Args:
            finished_request_ids: Set of requests deemed finished by the engine.
        
        Returns:
            Set of request IDs that have fully completed saving, or None.
        """
        finished_reqs = set()

        # Mark engine-finished requests as finished in our state
        for req_id in finished_request_ids:
            req_state = self._async_save_reqs.get(req_id)
            if req_state:
                req_state.finished = True
                if req_state.num_pending == 0:
                    finished_reqs.add(req_id)
                    del self._async_save_reqs[req_id]

        # Collect requests that finished asynchronously
        while not self._finished_save_reqs.empty():
            finished_reqs.add(self._finished_save_reqs.get_nowait())

        if finished_reqs:
            logger.debug(
                f"Finished save requests: {finished_reqs}, count: {len(finished_reqs)}"
            )
            return finished_reqs
        return None

    def get_load_finished(self) -> Optional[Set[str]]:
        """
        Checks which requests have completed their load operations.
        
        Returns:
            Set of request IDs that have fully completed loading, or None.
        """
        finished_reqs = set()
        while not self._finished_load_reqs.empty():
            finished_reqs.add(self._finished_load_reqs.get_nowait())

        if finished_reqs:
            logger.debug(
                f"Finished load requests: {finished_reqs}, count: {len(finished_reqs)}"
            )
            return finished_reqs
        return None


class YuanRongConnector(KVConnectorBase_V1):
    """
    Custom KV Connector implementation for YuanRong data system.
    Handles transferring KV cache blocks between vLLM GPU memory and remote storage.
    """

    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole):
        """
        Args:
            vllm_config: The main vLLM configuration object.
            role: The role this connector plays (WORKER/PRODUCER/CONSUMER).
        """
        super().__init__(vllm_config=vllm_config, role=role)

        self._block_size = vllm_config.cache_config.block_size
        self._requests_need_load: Dict[str, Request] = {}
        self.config = vllm_config.kv_transfer_config
        self.is_producer = self.config.is_kv_producer
        self.do_async_save = int(os.getenv("ASYNC_SAVE", 1))

        # Internal state for layers and caches
        self.layer_name_list: list[str] = []
        self.kv_caches: list[torch.Tensor] = []
        self.key_caches: list[torch.Tensor] = []
        self.value_caches: list[torch.Tensor] = []

        # State tracking
        self._skip_blocks: Dict[str, int] = {}
        self._ds_cached_blocks: Dict[str, int] = {}
        self._delay_save: Dict[str, RequestTracker] = {}

        # Async queues and tasks
        self._load_request_queue: asyncio.Queue[ReqMeta] = asyncio.Queue()
        self._save_request_queue: asyncio.Queue[ReqMeta] = asyncio.Queue()
        self.task_list: List[asyncio.Task] = []
        self._async_handler = None

        # Model backend flags
        self.is_ms_non_mla_type = False
        self.is_ms_mla = False
        self.is_mla = False

        self.tp_size = vllm_config.parallel_config.tensor_parallel_size

        # DataSystem Connection Configuration
        ds_worker_addr = os.getenv("DS_WORKER_ADDR", "172.17.0.4:9000")
        ip, port = split_host_port(ds_worker_addr)

        self.device = get_world_group().local_rank
        self.tp_rank = 0

        if role == KVConnectorRole.WORKER:
            self.tp_rank = get_tp_group().rank_in_group
            self.tp_group = get_tp_group()
            self.kvc_store = DsTensorClient(ip, port, self.device)
            self.kvc_store.init()

            if self.do_async_save:
                self.loop = asyncio.get_event_loop()
                self._async_handler = AsyncHandler(self.is_producer,
                                                   self.task_list)

                if ENABLE_PREFIX_CACHING or not self.is_producer:
                    self.task_list.append(
                        self.loop.create_task(self.consumer_request_task()))

                if ENABLE_PREFIX_CACHING or self.is_producer:
                    self.task_list.append(
                        self.loop.create_task(self.producer_request_task()))

                # Start event loop in a daemon thread
                thread = threading.Thread(target=self.start_event_loop,
                                          daemon=True)
                thread.start()

        elif ENABLE_PREFIX_CACHING:
            # Non-worker roles (e.g. scheduler) needing simple connectivity
            self.kvc_store = DsTensorClient(ip, port, self.device)
            self.kvc_store.init()
        else:
            self.tp_group = None

        logger.info(
            f"Initialized Datasystem: ip={ip}, port={port}, device_id={self.device}"
        )

    def start_event_loop(self):
        """Starts the async event loop execution."""
        current_thread = threading.current_thread()
        logger.info(
            f"Starting async event loop in thread: {current_thread.ident}")
        self.loop.run_until_complete(asyncio.gather(*self.task_list))
        self.loop.close()

    async def producer_request_task(self):
        """Consumer loop for handling save requests."""
        while True:
            try:
                q_size = self._save_request_queue.qsize()
                for _ in range(q_size):
                    request = self._save_request_queue.get_nowait()
                    self.do_save_request(request)
                await asyncio.sleep(SLEEP_TIMEOUT)
            except Exception as e:
                logger.error(f"producer_request_task failed: {e}")
                # Re-queue might be dangerous if error is persistent, considering simple retry logic
                # self._save_request_queue.put_nowait(request)
                await asyncio.sleep(SLEEP_TIMEOUT)

    async def consumer_request_task(self):
        """Consumer loop for handling load requests."""
        while True:
            try:
                q_size = self._load_request_queue.qsize()
                for _ in range(q_size):
                    request = self._load_request_queue.get_nowait()
                    self.do_load_kv(request)
                await asyncio.sleep(SLEEP_TIMEOUT)
            except Exception as e:
                logger.error(f"consumer_request_task failed: {e}")
                self._load_request_queue.put_nowait(request)
                await asyncio.sleep(SLEEP_TIMEOUT)

    def generate_kv_cache_token_key(self, request: ReqMeta,
                                    block_start_index: int,
                                    block_end_index: int) -> List[str]:
        """
        Generates a list of unique keys for KV cache blocks based on token content and TP rank.
        """
        if not self.is_mla:
            # Standard KV Cache: Keys are rank-specific
            external_key = "-" + str(self.tp_rank)
        else:
            # MLA (Multi-Head Latent Attention): Shared key suffix
            external_key = "-0"

        return generate_hash_sha256(block_start_index, block_end_index,
                                    request.token_ids, self._block_size,
                                    external_key)

    def start_load_kv(self, forward_context: "ForwardContext",
                      **kwargs) -> None:
        """
        Initiates the KV cache loading process.
        
        Note:
            If prefix caching is disabled and this is a producer, loading is skipped.
        """
        # Optimization: Skip load for producers without prefix caching
        if self.is_producer and not ENABLE_PREFIX_CACHING:
            return

        metadata: KVConnectorMetadata = self._get_connector_metadata()
        if not metadata.requests:
            return

        if not self.kv_caches:
            self._init_kv_caches_from_forward_context(forward_context)

        for request in metadata.requests:
            if self._async_handler is not None:
                self._load_request_queue.put_nowait(request)
            else:
                self.do_load_kv(request)

    def get_finished(
        self, finished_req_ids: Set[str]
    ) -> Tuple[Optional[Set[str]], Optional[Set[str]]]:
        """
        Retrieves lists of requests that have finished saving and loading.
        """
        finished_saved_req, finished_loaded_req = None, None
        if self._async_handler is not None:
            if self.is_producer or ENABLE_PREFIX_CACHING:
                finished_saved_req = self._async_handler.get_save_finished(
                    finished_req_ids)

            if not self.is_producer or ENABLE_PREFIX_CACHING:
                finished_loaded_req = self._async_handler.get_load_finished()

            return finished_saved_req, finished_loaded_req
        return None, None

    def get_sending_count(self) -> int:
        """Returns the number of expected send operations based on model type."""
        if self.is_mla:
            return 1
        return self.tp_size

    def do_load_kv(self, request: ReqMeta) -> None:
        """
        Executes the KV cache load operation (H2D).
        Supports both standard Key/Value split caches and MLA unified caches.
        """
        ds_cached_block_num = request.ds_cached_block_num
        skip_block_num = request.skip_block_num

        logger.debug(
            f"request: {request.request_id}, ds_cached_blocks: {ds_cached_block_num}, "
            f"skip_blocks: {skip_block_num}")

        if ds_cached_block_num == 0:
            return

        key_list = self.generate_kv_cache_token_key(request, skip_block_num,
                                                    ds_cached_block_num)
        block_id_list = request.block_ids

        if not block_id_list or not key_list:
            return

        # Handle Non-MLA (Standard split Key/Value Cache)
        if not self.is_mla:
            value_cache_key_list = [key + "-value" for key in key_list]

            if len(key_list) != len(block_id_list):
                logger.error(
                    f"mget_tensors_h2d mismatch: req {request.request_id}")

            get_timeout = 10000
            key_load_future = self.kvc_store.mget_page_attn_blockwise_h2d(
                key_list, self.key_caches, block_id_list, get_timeout)
            value_load_future = self.kvc_store.mget_page_attn_blockwise_h2d(
                value_cache_key_list, self.value_caches, block_id_list,
                get_timeout)

            if not self.do_async_save:
                get_future(key_load_future, SYNC_FUTURE_TIMEOUT)
                get_future(value_load_future, SYNC_FUTURE_TIMEOUT)
            elif self._async_handler is not None:
                self._async_handler.add_load_request(request, 2)
                self._async_handler.add_load_future(request, key_load_future)
                self._async_handler.add_load_future(request, value_load_future)

            logger.debug(
                f"mget_tensors_h2d (Split KV) success for {request.request_id}"
            )
            return

        # Handle MLA (Unified Cache)
        future = self.kvc_store.mget_page_attn_blockwise_h2d(
            key_list, self.kv_caches, block_id_list)
        if not self.do_async_save:
            get_future(future, SYNC_FUTURE_TIMEOUT)
        elif self._async_handler is not None:
            self._async_handler.add_load_request(request, 1)
            self._async_handler.add_load_future(request, future)

        logger.debug(
            f"mget_tensors_h2d (MLA) success for {request.request_id}")

    def wait_for_layer_load(self, layer_name: str) -> None:
        """Placeholder: Wait for a specific layer to finish loading."""
        return

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor,
                      attn_metadata: "AttentionMetadata", **kwargs) -> None:
        """
        Registers a KV layer to be saved. Identifies model backend type (MLA vs Standard).
        """
        if not ENABLE_PREFIX_CACHING and not self.is_producer:
            return

        if layer_name not in self.layer_name_list:
            self.layer_name_list.append(layer_name)

            # Detect Model Type
            self.is_ms_non_mla_type = isinstance(kv_layer,
                                                 tuple) and len(kv_layer) == 2
            self.is_ms_mla = os.getenv(
                "vLLM_MODEL_BACKEND",
                None) == "MindFormers" and not self.is_ms_non_mla_type
            self.is_mla = isinstance(attn_metadata,
                                     MLACommonMetadata) or self.is_ms_mla

            if self.is_mla:
                self.kv_caches.append(kv_layer)
            else:
                self.key_caches.append(kv_layer[0])
                self.value_caches.append(kv_layer[1])

    def do_save_request(self, request: ReqMeta) -> None:
        """
        Executes the KV cache save operation (D2H).
        """
        logger.debug(f"do_save_request: {request}")
        if not self.is_producer or not request.need_save:
            return

        # For MLA, usually only one rank needs to save if shared,
        # or checks specific rank alignment
        if self.is_mla and self.tp_rank != request.request_rank:
            return

        if not request.block_ids:
            return

        token_key_list = self.generate_kv_cache_token_key(
            request, 0, len(request.block_ids))

        # Handle Non-MLA
        if not self.is_mla:
            value_cache_key_list = [key + "-value" for key in token_key_list]
            key_save_future = self.kvc_store.mset_page_attn_blockwise_d2h(
                token_key_list, self.key_caches, request.block_ids)
            value_save_future = self.kvc_store.mset_page_attn_blockwise_d2h(
                value_cache_key_list, self.value_caches, request.block_ids)

            if not self.do_async_save:
                get_future(key_save_future, SYNC_FUTURE_TIMEOUT)
                get_future(value_save_future, SYNC_FUTURE_TIMEOUT)
            elif self._async_handler is not None:
                self._async_handler.add_save_request(request, 2)
                self._async_handler.add_save_future(request, key_save_future)
                self._async_handler.add_save_future(request, value_save_future)

            logger.debug(
                f"mset_tensors_d2h (Split KV) success for {request.request_id}"
            )
            return

        # Handle MLA
        future = self.kvc_store.mset_page_attn_blockwise_d2h(
            token_key_list, self.kv_caches, request.block_ids)
        if not self.do_async_save:
            get_future(future, SYNC_FUTURE_TIMEOUT)
        elif self._async_handler is not None:
            self._async_handler.add_save_request(request, 1)
            self._async_handler.add_save_future(request, future)

        logger.debug(
            f"mset_tensors_d2h (MLA) success for {request.request_id}")

    def wait_for_save(self) -> None:
        """
        Triggers the save process for any requests pending in the connector metadata.
        """
        if not self.is_producer:
            return

        connector_metadata = self._get_connector_metadata()
        if not isinstance(connector_metadata, YuanRongConnectorMetadata):
            raise ValueError(
                "connector_metadata must be instance of YuanRongConnectorMetadata"
            )

        if not connector_metadata.requests:
            return

        for request in connector_metadata.requests:
            if self._async_handler is not None:
                self._save_request_queue.put_nowait(request)
            else:
                self.do_save_request(request)

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> Tuple[int, bool]:
        """
        Calculates how many new tokens can be retrieved from the external cache.
        """
        num_computed_blocks = num_computed_tokens // self._block_size
        num_tokens_to_check = align_to_block_size(
            len(request.prompt_token_ids), self._block_size)
        prompt_blocks = num_tokens_to_check // self._block_size
        num_external_hit_tokens = 0

        # Logic for Consumer / Non-Producer Role
        if not self.is_producer:
            self._skip_blocks[request.request_id] = num_computed_blocks
            num_external_computed_tokens = len(
                request.prompt_token_ids) - num_computed_tokens - 1
            self._ds_cached_blocks[request.request_id] = prompt_blocks

            if self.do_async_save and num_external_computed_tokens > 0:
                logger.info(
                    f"req: {request.request_id}, computed: {num_computed_tokens}, "
                    f"ext_computed: {num_external_computed_tokens}")
                return num_external_computed_tokens, True

            return num_external_computed_tokens, False

        # Logic for Producer with Prefix Caching
        if ENABLE_PREFIX_CACHING:
            tokens = request.prompt_token_ids
            # Generate hash keys for the blocks we want to check
            keys = generate_hash_sha256(num_computed_blocks, prompt_blocks,
                                        numpy.array(tokens), self._block_size,
                                        "-0")

            if not keys:
                logger.info(
                    f"Req: {request.request_id}, HBM hit: {num_computed_tokens}, need load: 0"
                )
                return 0, False

            try:
                # Check existence in data store; append False as sentinel
                exists = self.kvc_store.exist(keys) + [False]
            except RuntimeError:
                logger.info(
                    f"Req: {request.request_id}, Store check failed, need load: 0"
                )
                return 0, False

            # Find first missing block
            num_external_hit_blocks = exists.index(False)
            num_external_hit_tokens = num_external_hit_blocks * self._block_size

            self._skip_blocks[request.request_id] = num_computed_blocks
            self._ds_cached_blocks[
                request.
                request_id] = num_external_hit_blocks + num_computed_blocks

            logger.info(
                f"Req: {request.request_id}, HBM hit: {num_computed_tokens}, "
                f"External hit tokens: {num_external_hit_tokens}")

            if self.do_async_save and num_external_hit_tokens > 0:
                return num_external_hit_tokens, True

        return num_external_hit_tokens, False

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: Any,  # Typed as Any to avoid circular import with KVCacheBlocks
        num_external_tokens: int
    ) -> None:
        """
        Updates internal state after the scheduler has allocated blocks.
        """
        if num_external_tokens > 0:
            block = blocks.get_unhashed_block_ids()
            self._requests_need_load[request.request_id] = (request, [block])
            logger.debug(f"Added to load queue: {request.request_id}")

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        """
        Constructs the metadata required for KV transfer based on the scheduler's output.
        Matches requests needing load/save and handles delayed save logic.
        """
        meta = YuanRongConnectorMetadata(self.tp_size, self._block_size)
        total_need_load = 0

        # Process new requests scheduled in this step
        for new_req in scheduler_output.scheduled_new_reqs:
            if new_req.req_id in self._requests_need_load:
                # Request needs loading from external source
                meta.add_request(
                    request_id=new_req.req_id,
                    token_ids=new_req.prompt_token_ids,
                    block_ids=new_req.block_ids,
                    skip_block_num=self._skip_blocks.pop(new_req.req_id, 0),
                    ds_cached_block_num=self._ds_cached_blocks.pop(
                        new_req.req_id, 0))
                total_need_load += 1
            else:
                # Logic for tracking delayed saves (Producer only)
                if self.is_producer:
                    num_scheduled_tokens = scheduler_output.num_scheduled_tokens.get(
                        new_req.req_id)
                    num_scheduled_tokens += new_req.num_computed_tokens

                    # If not all tokens are scheduled, delay the save
                    if len(new_req.prompt_token_ids) > num_scheduled_tokens:
                        self._delay_save[
                            new_req.req_id] = RequestTracker.from_new_request(
                                new_req.req_id, new_req.prompt_token_ids,
                                new_req.block_ids, num_scheduled_tokens)
                    else:
                        meta.add_request(
                            request_id=new_req.req_id,
                            token_ids=new_req.prompt_token_ids,
                            block_ids=new_req.block_ids,
                            skip_block_num=self._skip_blocks.pop(
                                new_req.req_id, 0),
                            ds_cached_block_num=self._ds_cached_blocks.pop(
                                new_req.req_id, 0))

        # Process cached (running) requests
        cached_reqs = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(cached_reqs.req_ids):
            new_block_ids = cached_reqs.new_block_ids[i]
            resumed_from_preemption = cached_reqs.resumed_from_preemption[i]

            # Handle delayed saves for running requests
            if not resumed_from_preemption:
                if req_id in self._delay_save:
                    request_tracker = self._delay_save.get(req_id)
                    num_external_scheduled_tokens = scheduler_output.num_scheduled_tokens.get(
                        req_id)
                    if request_tracker is not None:
                        request_tracker.update(new_block_ids,
                                               num_external_scheduled_tokens)

                        # Check if all tokens are now scheduled
                        if len(request_tracker.token_ids
                               ) <= request_tracker.num_scheduled_tokens:
                            del self._delay_save[req_id]
                            logger.debug(
                                f"Processing delayed save for: {request_tracker.request_id}"
                            )
                            meta.add_request(
                                request_id=request_tracker.request_id,
                                token_ids=request_tracker.token_ids,
                                block_ids=request_tracker.block_ids,
                                skip_block_num=self._skip_blocks.pop(
                                    request_tracker.request_id, 0),
                                ds_cached_block_num=self._ds_cached_blocks.pop(
                                    request_tracker.request_id, 0))

            # Handle resumed requests needing load
            if req_id in self._requests_need_load:
                request = self._requests_need_load[req_id]
                # Reconstruct token ID list for prompt
                token_ids = request.all_token_ids[:len(request.prompt_token_ids
                                                       )]
                logger.debug(
                    f"Request {request.request_id} resumed from preemption")

                meta.add_request(
                    request_id=req_id,
                    token_ids=token_ids,
                    block_ids=new_block_ids,
                    skip_block_num=self._skip_blocks.pop(req_id, 0),
                    ds_cached_block_num=self._ds_cached_blocks.pop(req_id, 0))
                total_need_load += 1

        # Process pending async load requests
        if self.do_async_save:
            for req_id, (req, block_ids) in self._requests_need_load.items():
                if not block_ids:
                    logger.debug(f"Skipping empty block load for {req_id}")
                    continue

                meta.add_request(
                    request_id=req_id,
                    token_ids=req.prompt_token_ids,
                    block_ids=block_ids,
                    skip_block_num=self._skip_blocks.pop(req_id, 0),
                    ds_cached_block_num=self._ds_cached_blocks.pop(req_id, 0),
                    need_save=False)
                total_need_load += 1

        logger.debug(f"Build Meta: total_need_load={total_need_load}, "
                     f"pending={len(self._requests_need_load)}")

        if total_need_load != len(self._requests_need_load):
            logger.error(
                f"Mismatch: need_load={total_need_load} vs pending={len(self._requests_need_load)}"
            )
            raise ValueError("Internal state mismatch in load requests")

        self._requests_need_load.clear()
        return meta

    def request_finished(
        self,
        request: "Request",
        block_ids: List[int],
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Callback when a request finishes. 
        Returns True if saving might be continuing asynchronously.
        """
        if self.is_producer:
            return bool(self.do_async_save), None
        return False, None

    def _init_kv_caches_from_forward_context(
            self, forward_context: "ForwardContext") -> None:
        """
        Initializes internal cache references from the vLLM forward context.
        Detects if the model uses MLA or standard attention.
        """
        attn_metadata = forward_context.attn_metadata

        for layer_name, attn_layer in forward_context.no_compile_layers.items(
        ):
            kv_layer = attn_layer.kv_cache[forward_context.virtual_engine]

            # Determine Backend Type
            self.is_ms_non_mla_type = isinstance(kv_layer,
                                                 tuple) and len(kv_layer) == 2
            self.is_ms_mla = (os.getenv("vLLM_MODEL_BACKEND", None)
                              == "MindFormers" and not self.is_ms_non_mla_type)
            self.is_mla = isinstance(attn_metadata,
                                     MLACommonMetadata) or self.is_ms_mla

            if layer_name not in self.layer_name_list:
                self.layer_name_list.append(layer_name)
                logger.debug(f"Init cache for layer: {layer_name}")

                if not self.is_mla:
                    self.key_caches.append(kv_layer[0])
                    self.value_caches.append(kv_layer[1])
                elif self.is_ms_mla:
                    self.kv_caches.append(kv_layer[0])
                else:
                    self.kv_caches.append(kv_layer)


# Utility Functions
def extract_number(s: str) -> Optional[int]:
    """Extracts the first integer found in a dot-separated string."""
    parts = s.split('.')
    for part in parts:
        if part.isdigit():
            return int(part)
    return None


def align_to_block_size(num_tokens: int, block_size: int) -> int:
    """
    Aligns the token count to the nearest block size boundary 
    using specific ceiling logic.
    """
    return (num_tokens + block_size - 2) // block_size * block_size


def generate_hash_sha256(block_start_index: int, block_end_index: int,
                         token_ids: numpy.ndarray, block_size: int,
                         external_key: str) -> List[str]:
    """
    Generates a list of SHA256 hash keys for a range of KV cache blocks.
    
    Args:
        block_start_index: Starting block index.
        block_end_index: Ending block index.
        token_ids: Array of token IDs.
        block_size: Size of one block.
        external_key: Suffix key (e.g., TP rank) to ensure uniqueness.
        
    Returns:
        List of hash strings.
    """
    hash_list = []
    for block_index in range(block_start_index, block_end_index):
        end_index = (block_index + 1) * block_size
        # Extract tokens belonging to this block
        input_ids = token_ids[:end_index]
        input_ids_bytes = input_ids.tobytes()

        token_hash = hashlib.sha256(input_ids_bytes).hexdigest()
        hash_list.append(token_hash + external_key)
    return hash_list


def get_future(fut: Future, timeout: int = FUTURE_TIMEOUT) -> RequestStatus:
    """
    Helper to resolve a Future object with a timeout.
    
    Returns:
        RequestStatus (FINISHED, WAITING, or TIMEOUT).
    """
    try:
        failed_list = fut.get(timeout)
    except TimeoutError:
        return RequestStatus.WAITING

    if len(failed_list) != 0:
        logger.error(f"Future returned failures: {failed_list}")
        return RequestStatus.TIMEOUT

    return RequestStatus.FINISHED
