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
import json
import os
import threading
from collections import defaultdict
from dataclasses import dataclass
from types import SimpleNamespace
from typing import (TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Set,
                    Tuple, Union, cast)

import numpy
import torch
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole)
from vllm.distributed.parallel_state import get_tp_group, get_world_group
from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.utils import logger, split_host_port
from vllm.v1.attention.backends.mla.common import MLACommonMetadata
from vllm.v1.core.kv_cache_utils import _gen_mm_extra_hash_keys
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.v1.request import Request

from datasystem import DsTensorClient, Future

# Configuration Constants
DS_WORKER_ADDR = os.getenv("DS_WORKER_ADDR", "127.0.0.1:31501")
ENABLE_PREFIX_CACHING = int(os.getenv("USING_PREFIX_CONNECTOR", "1"))
FUTURE_TIMEOUT = int(os.getenv("FUTURE_TIMEOUT", "10000"))
SYNC_FUTURE_TIMEOUT = int(os.getenv("SYNC_FUTURE_TIMEOUT", "1"))
SLEEP_TIMEOUT = 0.005


class RequestStatus(enum.IntEnum):
    """Enumeration for tracking execution states of asynchronous KV operations.
    
    This enum defines the possible states of async save/load requests
    throughout their lifecycle in the KV connector system.
    """
    WAITING = enum.auto()
    TIMEOUT = enum.auto()
    FINISHED = enum.auto()


@dataclass
class RequestTracker:
    """Tracks request state for delayed KV cache saving and token scheduling.
    
    Maintains critical information about a request's token sequence, allocated
    cache blocks, and scheduling progress to manage deferred save operations.
    """
    request_id: str
    token_ids: torch.Tensor
    block_ids: tuple[List[int], ...]
    num_scheduled_tokens: int
    mm_features: Optional[list[MultiModalFeatureSpec]] = None

    @staticmethod
    def from_new_request(
        request_id: str,
        token_ids: torch.Tensor,
        block_ids: tuple[List[int], ...],
        num_scheduled_tokens: int,
        mm_features: Optional[list[MultiModalFeatureSpec]] = None
    ) -> "RequestTracker":
        """Creates a new RequestTracker instance for a fresh request.
        
        Args:
            request_id: Unique identifier for the request
            token_ids: Tensor containing the request's token sequence
            block_ids: Tuple of allocated KV cache block ID lists
            num_scheduled_tokens: Initial count of scheduled tokens
            mm_features: Optional list of multimodal feature specifications
        
        Returns:
            Initialized RequestTracker instance
        """
        return RequestTracker(request_id=request_id,
                              token_ids=token_ids,
                              block_ids=block_ids,
                              num_scheduled_tokens=num_scheduled_tokens,
                              mm_features=mm_features)

    def update(self, block_ids: tuple[List[int], ...],
               num_external_scheduled_tokens: int) -> None:
        """Updates tracker state with newly allocated blocks and scheduled tokens.
        
        Args:
            block_ids: Newly allocated block IDs (None if no new blocks)
            num_external_scheduled_tokens: Number of additional tokens scheduled
        """
        if block_ids:
            self.block_ids[0].extend(block_ids[0])

        self.num_scheduled_tokens += num_external_scheduled_tokens


@dataclass
class ReqMeta:
    """Metadata container for KV cache transfer operations (save/load).
    
    Encapsulates all necessary information for a single request's KV cache
    transfer, including token data, block allocation, and transfer parameters.
    """
    request_id: str
    token_ids: numpy.ndarray
    block_ids: List[int]
    request_rank: int
    skip_block_num: int
    ds_cached_block_num: int
    need_save: bool
    mm_features: Optional[list[MultiModalFeatureSpec]] = None

    @staticmethod
    def make_meta(
        request_id: str,
        token_ids: List[int],
        block_ids: tuple[List[int], ...],
        block_size: int,
        request_rank: int,
        skip_block_num: int,
        ds_cached_block_num: int,
        need_save: bool,
        mm_features: Optional[list[MultiModalFeatureSpec]] = None
    ) -> "ReqMeta":
        """Factory method to create ReqMeta with block-aligned calculations.
        
        Calculates valid token count and block IDs aligned to the system's
        block size to ensure proper KV cache alignment.
        
        Args:
            request_id: Unique request identifier
            token_ids: List of token IDs for the request
            block_ids: Tuple of original block ID lists
            block_size: Size of each KV cache block
            request_rank: TP rank assigned to the request
            skip_block_num: Number of blocks to skip
            ds_cached_block_num: Number of blocks cached in datasystem
            need_save: Whether the request needs to be saved
            mm_features: Optional multimodal feature specifications
        
        Returns:
            Initialized ReqMeta instance with aligned block data
        """
        # Calculate valid token count aligned to block size
        valid_num_tokens = align_to_block_size(len(token_ids), block_size)
        valid_block_ids_count = valid_num_tokens // block_size

        return ReqMeta(request_id=request_id,
                       token_ids=numpy.array(token_ids),
                       block_ids=block_ids[0][:valid_block_ids_count],
                       request_rank=request_rank,
                       skip_block_num=skip_block_num,
                       ds_cached_block_num=ds_cached_block_num,
                       need_save=need_save,
                       mm_features=mm_features)


@dataclass
class YuanRongConnectorMetadata(KVConnectorMetadata):
    """Metadata container for YuanRong KV Connector operations.
    
    Extends the base KVConnectorMetadata to manage batches of requests with
    round-robin TP rank assignment for load balancing across tensor parallel
    workers.
    """
    requests: List[ReqMeta]

    def __init__(self, tp_size: int, block_size: int):
        """Initialize metadata container for YuanRong connector.
        
        Args:
            tp_size: Tensor parallelism size (number of TP workers)
            block_size: Size of each KV cache block
        """
        self._block_size = block_size
        self.request_rank = 0  # Counter for round-robin rank assignment
        self.requests = []
        self._tp_size = tp_size

    def add_request(
            self,
            request_id: str,
            token_ids: List[int],
            block_ids: tuple[List[int], ...],
            skip_block_num: int,
            ds_cached_block_num: int,
            need_save: bool = True,
            mm_features: Optional[list[MultiModalFeatureSpec]] = None) -> None:
        """Adds request metadata to the batch with round-robin TP rank assignment.
        
        Assigns TP ranks in a round-robin fashion to distribute load evenly
        across tensor parallel workers.
        
        Args:
            request_id: Unique request identifier
            token_ids: List of token IDs for the request
            block_ids: Tuple of block ID lists
            skip_block_num: Number of blocks to skip
            ds_cached_block_num: Number of blocks cached in datasystem
            need_save: Whether to save the request (default: True)
            mm_features: Optional multimodal feature specifications
        """
        # Assign TP rank using round-robin distribution
        request_rank = self.request_rank % self._tp_size
        self.requests.append(
            ReqMeta.make_meta(request_id=request_id,
                              token_ids=token_ids,
                              block_ids=block_ids,
                              block_size=self._block_size,
                              request_rank=request_rank,
                              skip_block_num=skip_block_num,
                              ds_cached_block_num=ds_cached_block_num,
                              need_save=need_save,
                              mm_features=mm_features))
        self.request_rank = request_rank + 1


@dataclass
class ReqState:
    """Tracks internal state of pending asynchronous save/load requests.
    
    Maintains counters for pending operations and completion status to
    coordinate async KV transfer workflows.
    """
    num_pending: int = -1  # Number of pending async operations for the request
    finished: bool = False  # Whether the request has been marked as finished


class AsyncHandler:
    """Manages asynchronous KV cache save/load operations.
    
    Coordinates background processing of async futures, tracks request states,
    and handles completion/timeout events for KV transfer operations.
    """

    def __init__(self, role: KVConnectorRole, task_list: List[asyncio.Task]):
        """Initialize async operation handler.
        
        Args:
            role: KV connector role (PRODUCER/CONSUMER/WORKER)
            task_list: List to register background async tasks
        """
        # Maps request IDs to their async state for save/load operations
        self._async_save_reqs: Dict[str, ReqState] = defaultdict(ReqState)
        self._async_load_reqs: Dict[str, ReqState] = defaultdict(ReqState)
        self._is_producer: bool = role

        # Queues for tracking completed operations
        self._finished_save_reqs: asyncio.Queue = asyncio.Queue()
        self._finished_load_reqs: asyncio.Queue = asyncio.Queue()

        # Queues for pending futures
        self._future_save_list: asyncio.Queue = asyncio.Queue()
        self._future_load_list: asyncio.Queue = asyncio.Queue()

        loop = asyncio.get_event_loop()

        # Register background tasks based on role and configuration
        if self._is_producer or ENABLE_PREFIX_CACHING:
            task_list.append(loop.create_task(self.get_save_futures_async()))

        if not self._is_producer or ENABLE_PREFIX_CACHING:
            task_list.append(loop.create_task(self.get_load_futures_async()))

    async def get_save_futures_async(self) -> None:
        """Background task to monitor and process save operation futures.
        
        Continuously polls pending save futures, updates request states,
        and handles completion/timeout events for asynchronous save operations.
        """
        while True:
            try:
                self._process_save_futures_batch()
            except Exception as e:
                logger.error("Failed to process save futures: %s", e)
            finally:
                await asyncio.sleep(SLEEP_TIMEOUT)

    def _process_save_futures_batch(self) -> None:
        """Handle a single batch of pending save futures."""
        q_size = self._future_save_list.qsize()
        for _ in range(q_size):
            request_id, future = self._future_save_list.get_nowait()
            self._handle_save_future(request_id, future)

    def _handle_save_future(self, request_id: str, future: Future) -> None:
        """Update save request state based on a future's result."""
        req_state = self._async_save_reqs.get(request_id)
        if req_state is None:
            logger.warning("Req: %s, save state missing for future",
                           request_id)
            return

        res = get_future(future)
        if res == RequestStatus.FINISHED:
            logger.info("Req: %s, Save task finished", request_id)
            req_state.num_pending -= 1
            if req_state.finished and req_state.num_pending == 0:
                self._finished_save_reqs.put_nowait(request_id)
                del self._async_save_reqs[request_id]
            return

        if res == RequestStatus.WAITING or not req_state.finished:
            self._future_save_list.put_nowait((request_id, future))
            return

        logger.error("Request: %s save future timeout/failed, result: %s",
                     request_id, res)
        self._finished_save_reqs.put_nowait(request_id)
        del self._async_save_reqs[request_id]

    async def get_load_futures_async(self) -> None:
        """Background task to monitor and process load operation futures.

        Continuously polls pending load futures, updates request states,
        and handles completion/timeout events for asynchronous load operations.
        """
        while True:
            try:
                self._process_load_futures_batch()
            except Exception as e:
                logger.error("Failed to process load futures: %s", e)
            finally:
                await asyncio.sleep(SLEEP_TIMEOUT)

    def _process_load_futures_batch(self) -> None:
        """Handle a single batch of pending load futures."""
        q_size = self._future_load_list.qsize()
        for _ in range(q_size):
            request_id, future = self._future_load_list.get_nowait()
            self._handle_load_future(request_id, future)

    def _handle_load_future(self, request_id: str, future: Future) -> None:
        """Update load request state based on a future's result."""
        req_state = self._async_load_reqs.get(request_id)
        if req_state is None:
            logger.warning("Req: %s, load state missing for future",
                           request_id)
            return

        res = get_future(future)
        if res == RequestStatus.FINISHED:
            logger.info("Req: %s, Load task finished", request_id)
            req_state.num_pending -= 1
            if req_state.num_pending == 0:
                self._finished_load_reqs.put_nowait(request_id)
                del self._async_load_reqs[request_id]
            return

        if res == RequestStatus.WAITING:
            self._future_load_list.put_nowait((request_id, future))
            return

        logger.error("Req: %s, Load future timeout/failed, result: %s",
                     request_id, res)
        self._finished_load_reqs.put_nowait(request_id)
        del self._async_load_reqs[request_id]

    def add_save_request(self, request: ReqMeta, future_num: int) -> None:
        """Register a save request with pending operation count.
        
        Args:
            request: Request metadata
            future_num: Number of async operations for this request
        """
        self._async_save_reqs[request.request_id].num_pending = future_num

    def add_load_request(self, request: ReqMeta, future_num: int) -> None:
        """Register a load request with pending operation count.
        
        Args:
            request: Request metadata
            future_num: Number of async operations for this request
        """
        self._async_load_reqs[request.request_id].num_pending = future_num

    def add_save_future(self, request: ReqMeta, future: Future) -> None:
        """Add a save operation future to the processing queue.
        
        Args:
            request: Request metadata
            future: Async future object for the save operation
        """
        self._future_save_list.put_nowait((request.request_id, future))

    def add_load_future(self, request: ReqMeta, future: Future) -> None:
        """Add a load operation future to the processing queue.
        
        Args:
            request: Request metadata
            future: Async future object for the load operation
        """
        self._future_load_list.put_nowait((request.request_id, future))

    def get_save_finished(
            self, finished_request_ids: Set[str]) -> Optional[Set[str]]:
        """Retrieve IDs of requests with completed save operations.
        
        Marks requests as finished and checks for completed async operations,
        returning IDs of fully completed save requests.
        
        Args:
            finished_request_ids: Set of request IDs marked as finished
        
        Returns:
            Set of completed save request IDs, or None if no completions
        """
        finished_reqs = set()
        # Mark requests as finished and check completion status
        for req_id in finished_request_ids:
            req_state = self._async_save_reqs.get(req_id)
            if req_state:
                req_state.finished = True
                if req_state.num_pending == 0:
                    finished_reqs.add(req_id)
                    del self._async_save_reqs[req_id]

        # Retrieve completed requests from queue
        while not self._finished_save_reqs.empty():
            finished_reqs.add(self._finished_save_reqs.get_nowait())

        if finished_reqs:
            logger.debug("Finished save requests: %s, count: %d",
                         finished_reqs, len(finished_reqs))
            return finished_reqs
        return None

    def get_load_finished(self) -> Optional[Set[str]]:
        """Retrieve IDs of requests with completed load operations.
        
        Returns IDs of requests with fully completed load operations.
        
        Returns:
            Set of completed load request IDs, or None if no completions
        """
        finished_reqs = set()
        while not self._finished_load_reqs.empty():
            finished_reqs.add(self._finished_load_reqs.get_nowait())

        if finished_reqs:
            logger.debug("Finished load requests: %s, count: %d",
                         finished_reqs, len(finished_reqs))
            return finished_reqs
        return None


class YuanRongConnector(KVConnectorBase_V1):
    """YuanRong datasystem KV cache connector implementation.
    
    Enables transfer of KV cache blocks between vLLM GPU memory and remote
    YuanRong datasystem storage, supporting both synchronous and asynchronous
    operations, prefix caching, and multimodal feature handling.
    """

    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole):
        """Initialize YuanRong KV connector.
        
        Args:
            vllm_config: Core vLLM configuration object
            role: KV connector role (WORKER/PRODUCER/CONSUMER)
        """
        super().__init__(vllm_config=vllm_config, role=role)

        self.vllm_config = vllm_config
        self._block_size = self.vllm_config.cache_config.block_size
        self._is_producer = self.vllm_config.kv_transfer_config.is_kv_producer
        self._tp_size = self.vllm_config.parallel_config.tensor_parallel_size
        self._do_async_save = int(os.getenv("ASYNC_SAVE", 1))

        # Mapping of requests needing load: request_id -> (Request object, block IDs)
        self._requests_need_load: Dict[str, tuple[Request, tuple[list[int],
                                                                 ...]]] = {}

        # Model layer and cache management
        self._layer_name_list: list[str] = []
        self._kv_caches: list[torch.Tensor] = []
        self._key_caches: list[torch.Tensor] = []
        self._value_caches: list[torch.Tensor] = []

        # Request state tracking
        self._skip_blocks: Dict[str, int] = {}
        self._ds_cached_blocks: Dict[str, int] = {}
        self._delay_save: Dict[str, RequestTracker] = {}

        # Async operation management
        self._load_request_queue: asyncio.Queue[ReqMeta] = asyncio.Queue()
        self._save_request_queue: asyncio.Queue[ReqMeta] = asyncio.Queue()
        self._task_list: List[asyncio.Task] = []
        self._async_handler = None

        # Model backend flags
        self._is_mla = False

        # Datasystem connection configuration
        ip, port = split_host_port(DS_WORKER_ADDR)
        self._device = 0
        self._tp_rank = None

        if role == KVConnectorRole.WORKER:
            # Initialize WORKER role components
            self._tp_group = get_tp_group()
            self._tp_rank = self._tp_group.rank_in_group
            self._device = get_world_group().local_rank
            self._ds_tensor_client = DsTensorClient(ip, port, self._device)
            self._ds_tensor_client.init()

            if self._do_async_save:
                self._loop = asyncio.get_event_loop()
                self._async_handler = AsyncHandler(self._is_producer,
                                                   self._task_list)
                # Register async processing tasks
                if ENABLE_PREFIX_CACHING or not self._is_producer:
                    self._task_list.append(
                        self._loop.create_task(self.consumer_request_task()))

                if ENABLE_PREFIX_CACHING or self._is_producer:
                    self._task_list.append(
                        self._loop.create_task(self.producer_request_task()))

                # Start async event loop in daemon thread
                thread = threading.Thread(target=self.start_event_loop,
                                          daemon=True)
                thread.start()
        elif ENABLE_PREFIX_CACHING:
            # Initialize datasystem client for non-WORKER roles with prefix caching
            self._ds_tensor_client = DsTensorClient(ip, port, self._device)
            self._ds_tensor_client.init()
        else:
            # Basic initialization for non-WORKER, non-caching roles
            self._tp_group = None

        logger.info(
            "YuanRongConnector initialized successfully. "
            "IP: %s, Port: %d, Device: %d", ip, port, self._device)

    @staticmethod
    def _to_block_tuple(block_ids: Any) -> tuple[list[int], ...]:
        """Normalize block IDs to a tuple of integer lists."""
        if block_ids is None:
            return tuple()

        if isinstance(block_ids, tuple):
            return tuple([list(map(int, block)) for block in block_ids])

        if isinstance(block_ids, list):
            if block_ids and isinstance(block_ids[0], (list, tuple)):
                return tuple([list(map(int, block)) for block in block_ids])
            return (list(map(int, block_ids)), )

        return (list(map(int, cast(Iterable[int], block_ids))), )

    def start_event_loop(self):
        """Start the async event loop in a dedicated thread.
        
        Runs the async task collection until all tasks complete, then closes
        the event loop.
        """
        current_thread = threading.current_thread()
        logger.info("Starting async event loop in thread: %s",
                    current_thread.ident)
        self._loop.run_until_complete(asyncio.gather(*self._task_list))
        self._loop.close()

    async def producer_request_task(self):
        """Background task for processing save requests.
        
        Consumes the save request queue and executes KV cache save operations
        for pending requests.
        """
        while True:
            try:
                q_size = self._save_request_queue.qsize()
                for _ in range(q_size):
                    request = self._save_request_queue.get_nowait()
                    self.do_save_request(request)
            except Exception as e:
                logger.error("producer_request_task failed: %s", e)
                # Re-queue request on failure (prevent loss)
                self._save_request_queue.put_nowait(request)
            finally:
                await asyncio.sleep(SLEEP_TIMEOUT)

    async def consumer_request_task(self):
        """Background task for processing load requests.
        
        Consumes the load request queue and executes KV cache load operations
        for pending requests.
        """
        while True:
            try:
                q_size = self._load_request_queue.qsize()
                for _ in range(q_size):
                    request = self._load_request_queue.get_nowait()
                    self.do_load_kv(request)
            except Exception as e:
                logger.error("consumer_request_task failed: %s", e)
                # Re-queue request on failure (prevent loss)
                self._load_request_queue.put_nowait(request)
            finally:
                await asyncio.sleep(SLEEP_TIMEOUT)

    def generate_kv_cache_token_key(self, request: ReqMeta,
                                    block_start_index: int,
                                    block_end_index: int) -> List[str]:
        """Generate unique SHA256 keys for KV cache blocks.
        
        Creates unique identifiers for KV cache blocks based on token content,
        block indices, TP rank, and multimodal features to ensure cache
        consistency and uniqueness.
        
        Args:
            request: Request metadata
            block_start_index: Starting block index for key generation
            block_end_index: Ending block index for key generation
        
        Returns:
            List of SHA256 hash keys for the specified blocks
        """
        # Use TP rank for non-MLA architectures, fixed 0 for MLA
        if not self._is_mla:
            external_key = "-" + str(self._tp_rank)
        else:
            external_key = "-0"

        return generate_hash_sha256(block_start_index, block_end_index,
                                    request.token_ids, self._block_size,
                                    external_key, request.mm_features)

    def start_load_kv(self, forward_context: "ForwardContext",
                      **kwargs) -> None:
        """Initiate KV cache loading process.
        
        Triggers loading of KV cache blocks from datasystem to GPU memory for
        eligible requests.
        
        Args:
            forward_context: Forward pass context object
            **kwargs: Additional keyword arguments
        """
        # Skip loading for producers with prefix caching disabled
        if self._is_producer and not ENABLE_PREFIX_CACHING:
            return

        # Retrieve connector metadata
        metadata: KVConnectorMetadata = self._get_connector_metadata()
        if not metadata.requests:
            return

        # Initialize KV cache references if not already done
        if not self._kv_caches:
            self._init_kv_caches_from_forward_context(forward_context)

        # Distribute load requests to processing queue or direct execution
        for request in metadata.requests:
            if self._async_handler is not None:
                self._load_request_queue.put_nowait(request)
            else:
                self.do_load_kv(request)

    def get_finished(
        self, finished_req_ids: Set[str]
    ) -> Tuple[Optional[Set[str]], Optional[Set[str]]]:
        """Retrieve IDs of requests with completed save/load operations.
        
        Gets sets of request IDs that have finished save and/or load operations
        from the async handler.
        
        Args:
            finished_req_ids: Set of request IDs marked as finished
        
        Returns:
            Tuple containing (completed save requests, completed load requests),
            with None for empty sets
        """
        finished_saved_req, finished_loaded_req = None, None
        if self._async_handler is not None:
            # Get completed save requests
            if self._is_producer or ENABLE_PREFIX_CACHING:
                finished_saved_req = self._async_handler.get_save_finished(
                    finished_req_ids)

            # Get completed load requests
            if not self._is_producer or ENABLE_PREFIX_CACHING:
                finished_loaded_req = self._async_handler.get_load_finished()

            logger.debug(
                "Finished saved requests: %s, Finished loaded requests: %s",
                finished_saved_req, finished_loaded_req)
            return finished_saved_req, finished_loaded_req
        return None, None

    def get_sending_count(self) -> int:
        """Get expected number of send operations based on model architecture.
        
        Returns the number of send operations required (1 for MLA architecture,
        TP size for non-MLA architecture).
        
        Returns:
            Number of expected send operations
        """
        if self._is_mla:
            return 1
        return self._tp_size

    def do_load_kv(self, request: ReqMeta) -> None:
        """Execute KV cache load operation (Host to Device).
        
        Loads KV cache blocks from datasystem to GPU memory, supporting both
        MLA (unified KV) and non-MLA (split Key/Value) architectures.
        
        Args:
            request: Request metadata for the load operation
        """
        ds_cached_block_num = request.ds_cached_block_num
        skip_block_num = request.skip_block_num
        logger.debug("Req: %s, ds_cached_blocks: %d, skip_blocks: %d",
                     request.request_id, ds_cached_block_num, skip_block_num)

        # Skip if no cached blocks available
        if ds_cached_block_num == 0:
            return

        # Generate cache keys for the blocks to load
        key_list = self.generate_kv_cache_token_key(request, skip_block_num,
                                                    ds_cached_block_num)
        block_id_list = request.block_ids

        if not block_id_list or not key_list:
            return

        # Handle non-MLA architecture (split Key/Value cache)
        if not self._is_mla:
            if len(key_list) != len(block_id_list):
                logger.error(
                    "Req: %s, Mismatch: key_list(len=%d) vs block_id_list(len=%d)",
                    request.request_id,
                    len(key_list),
                    len(block_id_list),
                )

            key_load_future = self._ds_tensor_client.mget_page_attn_blockwise_h2d(
                key_list, self._key_caches, block_id_list, FUTURE_TIMEOUT)
            value_cache_key_list = [key + "-value" for key in key_list]
            value_load_future = self._ds_tensor_client.mget_page_attn_blockwise_h2d(
                value_cache_key_list, self._value_caches, block_id_list,
                FUTURE_TIMEOUT)

            # Handle synchronous/asynchronous execution
            if not self._do_async_save:
                get_future(key_load_future, SYNC_FUTURE_TIMEOUT)
                get_future(value_load_future, SYNC_FUTURE_TIMEOUT)
            elif self._async_handler is not None:
                self._async_handler.add_load_request(request, 2)
                self._async_handler.add_load_future(request, key_load_future)
                self._async_handler.add_load_future(request, value_load_future)

            logger.debug("Successfully mget_h2d (Split KV) for %s",
                         request.request_id)
            return

        # Handle MLA architecture (unified KV cache)
        future = self._ds_tensor_client.mget_page_attn_blockwise_h2d(
            key_list, self._kv_caches, block_id_list)
        if not self._do_async_save:
            get_future(future, SYNC_FUTURE_TIMEOUT)
        elif self._async_handler is not None:
            self._async_handler.add_load_request(request, 1)
            self._async_handler.add_load_future(request, future)

        logger.debug("Successfully mget_h2d (MLA) for %s", request.request_id)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """YuanRongConnector does not do layerwise saving."""
        return

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor,
                      attn_metadata: "AttentionMetadata", **kwargs) -> None:
        """Register KV cache layer for transfer operations.
        
        Registers model layer KV cache references with the connector,
        distinguishing between MLA and non-MLA architectures.
        
        Args:
            layer_name: Name of the model layer
            kv_layer: KV cache tensor(s) (single tensor for MLA, tuple for non-MLA)
            attn_metadata: Attention metadata for the layer
            **kwargs: Additional keyword arguments
        """
        # Skip for non-producers with prefix caching disabled
        if not ENABLE_PREFIX_CACHING and not self._is_producer:
            return

        # Register new layers
        if layer_name not in self._layer_name_list:
            self._layer_name_list.append(layer_name)
            self._is_mla = isinstance(attn_metadata, MLACommonMetadata)

            # Register cache references based on architecture
            if self._is_mla:
                self._kv_caches.append(kv_layer)
            else:
                self._key_caches.append(kv_layer[0])
                self._value_caches.append(kv_layer[1])

    def _handle_save_futures(self, request: ReqMeta,
                             futures: list[Any]) -> None:
        """Handle synchronous or asynchronous save futures.

        Args:
            request: Request metadata object.
            futures: Futures returned by the datasystem client save calls.
        """
        if not self._do_async_save:
            for future in futures:
                get_future(future, SYNC_FUTURE_TIMEOUT)
        elif self._async_handler is not None:
            self._async_handler.add_save_request(request, len(futures))
            for future in futures:
                self._async_handler.add_save_future(request, future)

    def do_save_request(self, request: ReqMeta) -> None:
        """Execute KV cache save operation (Device to Host).
        
        Saves KV cache blocks from GPU memory to datasystem, supporting both
        MLA (unified KV) and non-MLA (split Key/Value) architectures.
        
        Args:
            request: Request metadata for the save operation
        """
        logger.debug("Req: %s, Save request", request)
        # Skip for non-producers or requests not marked for save
        if not self._is_producer or not request.need_save:
            return

        # For MLA architecture, only the assigned TP rank performs the save
        if self._is_mla and self._tp_rank != request.request_rank:
            return

        # Skip if no blocks to save
        if not request.block_ids:
            return

        token_key_list = self.generate_kv_cache_token_key(
            request, 0, len(request.block_ids))

        save_futures: list[Any] = []

        # Handle non-MLA architecture (split Key/Value cache)
        if not self._is_mla:
            key_save_future = self._ds_tensor_client.mset_page_attn_blockwise_d2h(
                token_key_list, self._key_caches, request.block_ids)
            value_cache_key_list = [key + "-value" for key in token_key_list]
            value_save_future = self._ds_tensor_client.mset_page_attn_blockwise_d2h(
                value_cache_key_list, self._value_caches, request.block_ids)

            save_futures.extend([key_save_future, value_save_future])

            self._handle_save_futures(request, save_futures)
            logger.debug("Successfully mset_d2h (Split KV) for %s",
                         request.request_id)
            return

        # Handle MLA architecture (unified KV cache)
        future = self._ds_tensor_client.mset_page_attn_blockwise_d2h(
            token_key_list, self._kv_caches, request.block_ids)
        self._handle_save_futures(request, [future])

        logger.debug("Successfully mset_d2h (MLA) for %s", request.request_id)

    def wait_for_save(self) -> None:
        """Trigger save process for pending requests.
        
        Initiates save operations for requests in the metadata batch, either
        through async queue or direct execution.
        """
        # Skip for non-producer roles
        if not self._is_producer:
            return

        connector_metadata = self._get_connector_metadata()
        if not isinstance(connector_metadata, YuanRongConnectorMetadata):
            raise ValueError(
                "The connector_metadata must be instance of YuanRongConnectorMetadata"
            )

        # Skip if no requests to save
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
        """Calculate number of tokens retrievable from external cache.
        
        Determines how many tokens can be loaded from the datasystem cache
        instead of being computed, with separate logic for producer and
        consumer roles.
        
        Args:
            request: Request object
            num_computed_tokens: Number of tokens already computed
        
        Returns:
            Tuple containing (number of external tokens, async load flag)
        """
        num_computed_blocks = num_computed_tokens // self._block_size
        num_tokens_to_check = align_to_block_size(
            len(request.prompt_token_ids), self._block_size)
        prompt_blocks = num_tokens_to_check // self._block_size
        num_external_hit_tokens = 0

        # Consumer/Non-Producer role logic
        if not self._is_producer:
            self._skip_blocks[request.request_id] = num_computed_blocks
            num_external_computed_tokens = len(
                request.prompt_token_ids) - num_computed_tokens - 1
            self._ds_cached_blocks[request.request_id] = prompt_blocks

            if self._do_async_save and num_external_computed_tokens > 0:
                logger.info(
                    "Req: %s, Computed tokens: %d, External computed tokens: %d",
                    request.request_id, num_computed_tokens,
                    num_external_computed_tokens)
                return num_external_computed_tokens, True

            return num_external_computed_tokens, False

        # Producer role with prefix caching logic
        if ENABLE_PREFIX_CACHING:
            tokens = request.prompt_token_ids
            mm_features = self._get_mm_features(request)
            keys = generate_hash_sha256(num_computed_blocks, prompt_blocks,
                                        numpy.array(tokens), self._block_size,
                                        "-0", mm_features)

            if not keys:
                logger.info(
                    "Req: %s, Total tokens %d, HBM hit tokens: %d, "
                    "External hit tokens: 0", request.request_id,
                    request.num_tokens, num_computed_tokens)
                return 0, False

            try:
                exists = self._ds_tensor_client.exist(keys) + [False]
            except RuntimeError:
                logger.info(
                    "Req: %s, Total tokens %d, HBM hit tokens: %d, "
                    "External hit tokens: 0", request.request_id,
                    request.num_tokens, num_computed_tokens)
                return 0, False

            num_external_hit_blocks = exists.index(False)
            num_external_hit_tokens = num_external_hit_blocks * self._block_size

            self._skip_blocks[request.request_id] = num_computed_blocks
            self._ds_cached_blocks[
                request.
                request_id] = num_external_hit_blocks + num_computed_blocks

            logger.info(
                "Req: %s, Total tokens %d, HBM hit tokens: %d, "
                "External hit tokens: %d", request.request_id,
                request.num_tokens, num_computed_tokens,
                num_external_hit_tokens)

            if self._do_async_save and num_external_hit_tokens > 0:
                return num_external_hit_tokens, True

        return num_external_hit_tokens, False

    def update_state_after_alloc(self, request: "Request", blocks: Any,
                                 num_external_tokens: int) -> None:
        """Update internal state after block allocation by scheduler.
        
        Records newly allocated blocks for requests that need external cache
        loading, updating the internal request tracking state.
        
        Args:
            request: Request object
            blocks: Allocated block information
            num_external_tokens: Number of external tokens to load
        """
        if num_external_tokens > 0:
            block_ids: tuple[list[int], ...] = self._to_block_tuple(
                blocks.get_unhashed_block_ids())
            self._requests_need_load[request.request_id] = (request, block_ids)
            logger.debug("Req: %s, Added to load queue", request.request_id)

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        """Construct metadata for KV transfer operations.
        
        Builds a metadata batch containing all necessary information for
        KV cache save/load operations based on scheduler output.
        
        Args:
            scheduler_output: Output from the vLLM scheduler
        
        Returns:
            Constructed YuanRongConnectorMetadata instance
        
        Raises:
            ValueError: If there's a mismatch in load request tracking
        """
        logger.debug("SchedulerOutput: %s", scheduler_output)
        meta = YuanRongConnectorMetadata(self._tp_size, self._block_size)
        total_need_load = 0

        total_need_load += self._handle_new_requests(scheduler_output, meta)
        total_need_load += self._handle_cached_requests(scheduler_output, meta)
        total_need_load += self._handle_pending_async_loads(meta)

        logger.debug("Build Meta: total_need_load=%s, pending=%s",
                     total_need_load, len(self._requests_need_load))
        if total_need_load != len(self._requests_need_load):
            logger.error("Mismatch: need_load=%s vs pending=%s",
                         total_need_load, len(self._requests_need_load))
            raise ValueError("Internal state mismatch in load requests")

        self._requests_need_load.clear()
        return meta

    def _handle_new_requests(self, scheduler_output: SchedulerOutput,
                             meta: YuanRongConnectorMetadata) -> int:
        """Handle metadata construction for newly scheduled requests."""
        total_need_load = 0
        for new_req in scheduler_output.scheduled_new_reqs:
            mm_features = self._get_mm_features(new_req)
            block_ids = self._to_block_tuple(new_req.block_ids)

            if new_req.req_id in self._requests_need_load:
                self._add_request_to_meta(meta, new_req.req_id,
                                          new_req.prompt_token_ids, block_ids,
                                          mm_features)
                total_need_load += 1
                continue

            if not self._is_producer:
                continue

            num_scheduled_tokens = scheduler_output.num_scheduled_tokens.get(
                new_req.req_id, 0)
            num_scheduled_tokens += new_req.num_computed_tokens

            # Track for delayed save if not all tokens are scheduled
            if len(new_req.prompt_token_ids) > num_scheduled_tokens:
                self._delay_save[
                    new_req.req_id] = RequestTracker.from_new_request(
                        new_req.req_id, new_req.prompt_token_ids, block_ids,
                        num_scheduled_tokens, mm_features)
                continue

            self._add_request_to_meta(meta, new_req.req_id,
                                      new_req.prompt_token_ids, block_ids,
                                      mm_features)

        return total_need_load

    def _handle_cached_requests(self, scheduler_output: SchedulerOutput,
                                meta: YuanRongConnectorMetadata) -> int:
        """Handle metadata construction for cached/suspended requests."""
        total_need_load = 0
        cached_reqs = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(cached_reqs.req_ids):
            new_block_ids = cached_reqs.new_block_ids[i]
            resumed_from_preemption = cached_reqs.resumed_from_preemption[i]

            # Handle delayed save requests
            if not resumed_from_preemption and req_id in self._delay_save:
                request_tracker = self._delay_save.get(req_id)
                num_external_scheduled_tokens = scheduler_output.num_scheduled_tokens.get(
                    req_id, 0)
                if request_tracker is not None:
                    request_tracker.update(self._to_block_tuple(new_block_ids),
                                           num_external_scheduled_tokens)
                    # Move to save queue if all tokens are scheduled
                    if len(request_tracker.token_ids
                           ) <= request_tracker.num_scheduled_tokens:
                        del self._delay_save[req_id]
                        logger.debug(
                            "Req: %s, Processing load for delayed save request",
                            request_tracker.request_id)
                        self._add_request_to_meta(meta,
                                                  request_tracker.request_id,
                                                  request_tracker.token_ids,
                                                  request_tracker.block_ids,
                                                  request_tracker.mm_features)

            # Handle resumed requests needing load
            if req_id in self._requests_need_load:
                request, _ = self._requests_need_load[req_id]
                mm_features = self._get_mm_features(request)
                logger.debug("Req: %s, Processing load for resumed request",
                             req_id)
                self._add_request_to_meta(meta, req_id,
                                          request.prompt_token_ids,
                                          self._to_block_tuple(new_block_ids),
                                          mm_features)
                total_need_load += 1

        return total_need_load

    def _handle_pending_async_loads(self,
                                    meta: YuanRongConnectorMetadata) -> int:
        """Handle pending async load requests when async save is enabled."""
        total_need_load = 0
        if not self._do_async_save:
            return total_need_load

        for req_id, (req, block_ids) in self._requests_need_load.items():
            if not block_ids:
                logger.debug("Req: %s, Skipping empty block load request",
                             req_id)
                continue

            self._add_request_to_meta(meta,
                                      req_id,
                                      req.prompt_token_ids,
                                      block_ids,
                                      self._get_mm_features(req),
                                      need_save=False)
            total_need_load += 1

        return total_need_load

    def _add_request_to_meta(
        self,
        meta: YuanRongConnectorMetadata,
        request_id: str,
        token_ids: Union[Iterable[int], numpy.ndarray[Any, Any]],
        block_ids: tuple[List[int], ...],
        mm_features: Optional[list[MultiModalFeatureSpec]] = None,
        need_save: bool = True,
    ) -> None:
        """Common helper to add a request to metadata with tracking pops."""
        skip_block_num = self._skip_blocks.pop(request_id, 0)
        ds_cached_block_num = self._ds_cached_blocks.pop(request_id, 0)
        token_id_list = [int(token_id) for token_id in token_ids]
        meta.add_request(
            request_id=request_id,
            token_ids=token_id_list,
            block_ids=block_ids,
            skip_block_num=skip_block_num,
            ds_cached_block_num=ds_cached_block_num,
            need_save=need_save,
            mm_features=mm_features,
        )

    @staticmethod
    def _get_mm_features(
            request: Any) -> Optional[list[MultiModalFeatureSpec]]:
        """Extract multimodal features from request-like objects."""
        return request.mm_features if hasattr(request, "mm_features") else None

    def request_finished(
        self,
        request: "Request",
        block_ids: List[int],
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Callback for completed requests.
        
        Notifies the connector that a request has finished processing and
        indicates if asynchronous saving is still in progress.
        
        Args:
            request: Completed request object
            block_ids: List of block IDs associated with the request
        
        Returns:
            Tuple containing (async save in progress flag, additional metadata)
        """
        logger.debug("Request finished: request=%s, block_ids=%s", request,
                     block_ids)
        # Return async save status for producer roles
        if self._is_producer:
            return bool(self._do_async_save), None
        return False, None

    def _init_kv_caches_from_forward_context(
            self, forward_context: "ForwardContext") -> None:
        """Initialize KV cache references from forward context.
        
        Extracts KV cache references from the vLLM forward context and
        initializes the connector's cache tracking structures, detecting
        the model architecture type.
        
        Args:
            forward_context: vLLM forward pass context object
        """
        attn_metadata = forward_context.attn_metadata
        no_compile_layers = forward_context.no_compile_layers
        for layer_name, attn_layer in no_compile_layers.items():
            if not hasattr(attn_layer, 'kv_cache'):
                continue

            kv_layer = attn_layer.kv_cache[forward_context.virtual_engine]
            self._is_mla = isinstance(attn_metadata, MLACommonMetadata)
            if layer_name not in self._layer_name_list:
                self._layer_name_list.append(layer_name)
                logger.debug("Init cache for layer: %s", layer_name)
                # Register cache tensors based on architecture
                if not self._is_mla:
                    self._key_caches.append(kv_layer[0])
                    self._value_caches.append(kv_layer[1])
                else:
                    self._kv_caches.append(kv_layer)


# Utility Functions
def align_to_block_size(num_tokens: int, block_size: int) -> int:
    """Align token count to nearest block size boundary.
    
    Uses ceiling division to round up the token count to the next
    block size multiple, ensuring proper cache block alignment.
    
    Args:
        num_tokens: Number of tokens to align
        block_size: Size of each cache block
        
    Returns:
        Token count aligned to block size
    """
    return (num_tokens + block_size - 2) // block_size * block_size


def generate_hash_sha256(
        block_start_index: int,
        block_end_index: int,
        token_ids: numpy.ndarray,
        block_size: int,
        external_key: str,
        mm_features: Optional[list[MultiModalFeatureSpec]] = None
) -> List[str]:
    """Generate SHA256 hash keys for KV cache blocks.
    
    Creates unique hash identifiers for each cache block based on token
    content, block indices, external keys, and multimodal features.
    
    Args:
        block_start_index: Starting block index
        block_end_index: Ending block index
        token_ids: Array of token IDs
        block_size: Size of each cache block
        external_key: External identifier (e.g., TP rank)
        mm_features: Optional multimodal feature specifications
        
    Returns:
        List of SHA256 hash keys for the specified blocks
    """
    hash_list = []
    for block_index in range(block_start_index, block_end_index):
        end_index = (block_index + 1) * block_size
        input_ids = token_ids[:end_index]
        input_ids_bytes = input_ids.tobytes()

        extra_bytes = b""
        if mm_features:
            start_token_idx = block_index * block_size
            end_token_idx = end_index
            mm_request = SimpleNamespace(mm_features=mm_features)
            mm_extra_keys, _ = _gen_mm_extra_hash_keys(mm_request,
                                                       start_token_idx,
                                                       end_token_idx, 0)
            if mm_extra_keys:
                extra_bytes = json.dumps(mm_extra_keys,
                                         separators=(',', ':')).encode("utf-8")

        # Combine all components and generate hash
        combined_bytes = input_ids_bytes + extra_bytes + external_key.encode(
            "utf-8")
        token_hash = hashlib.sha256(combined_bytes).hexdigest()
        hash_list.append(token_hash)
    return hash_list


def get_future(fut: Future, timeout: int = FUTURE_TIMEOUT) -> RequestStatus:
    """Resolve future with timeout handling.
    
    Waits for a future to complete with specified timeout, returning
    the appropriate status based on completion, timeout, or failure.
    
    Args:
        fut: Future object to resolve
        timeout: Timeout in milliseconds (default: FUTURE_TIMEOUT)
        
    Returns:
        RequestStatus indicating future outcome
    """
    try:
        failed_list = fut.get(timeout)
    except TimeoutError:
        return RequestStatus.WAITING

    if len(failed_list) != 0:
        logger.error("Future returned failures: %s" % failed_list)
        return RequestStatus.TIMEOUT

    return RequestStatus.FINISHED
