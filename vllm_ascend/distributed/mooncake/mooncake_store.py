# Standard
import os
from collections import deque
from dataclasses import dataclass

import regex as re
import torch
# Third Party
from mooncake.store import ReplicateConfig  # type: ignore
from vllm.config import VllmConfig
from vllm.distributed.ec_transfer.utils.transfer_engine import get_global_te
from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
from vllm.utils import get_ip, logger

from vllm_ascend.distributed.mooncake.config_data import MooncakeEngineKey

from .config_data import MooncakeStoreConfig
from .tensor_memory_pool import InsufficientMemoryError, TensorMemoryPool

METADATA_BYTES_LEN = 24
BASE_PORT = int(os.getenv("VLLM_BASE_PORT", "8790"))
DEFAULT_TENSOR_POOL_SIZE = 1073741824  # 1.0 GiB


@dataclass
class MooncakeTensorPoolMetadata:
    """
    Metadata element for a buffer in the tensor pool. This stores:

        key: MooncakeStore key of (key, value) pair
        addr: addr of the buffer in the tensor pool

    Those elements are maintained for zero-copy put method,
    and evicted by FIFO eviction policy.
    """

    key: str
    addrs: list[int]


class Mooncakestore():

    def __init__(self, vllm_config: VllmConfig):
        try:
            from mooncake.store import MooncakeDistributedStore  # type: ignore
        except ImportError as e:
            raise ImportError(
                "Please install mooncake by following the instructions at "
                "https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md "  # noqa: E501
                "to run vLLM with MooncakeConnector.") from e
        parallel_config = vllm_config.parallel_config
        tp_rank = get_tensor_model_parallel_rank()
        tp_size = parallel_config.tensor_parallel_size
        dp_rank = parallel_config.data_parallel_rank_local
        all_device_ids = os.getenv("ASCEND_RT_VISIBLE_DEVICES", None)
        if not all_device_ids:
            device_ids_list = list(
                range(dp_rank * tp_size, (dp_rank + 1) * tp_size))
        else:
            device_ids_list = list(map(int, all_device_ids.split(',')))
        assert len(device_ids_list) > tp_rank
        device_id = device_ids_list[tp_rank]
        self.config = MooncakeStoreConfig.load_mooncake_store_config(
            vllm_config)
        self.store = MooncakeDistributedStore()
        if self.config.protocol == "ascend" and not self.config.use_ascend_direct:
            local_hostname = get_ip() + ":" + str(BASE_PORT + int(device_id)) + \
                             ":npu_" + str(device_id)
            ret = self.store.setup(local_hostname, self.config.metadata_server,
                                   self.config.global_segment_size,
                                   self.config.local_buffer_size,
                                   self.config.protocol,
                                   self.config.device_name,
                                   self.config.master_server_address)
        elif self.config.protocol == "tcp":
            local_hostname = self.config.local_hostname
            ret = self.store.setup(local_hostname, self.config.metadata_server,
                                   self.config.global_segment_size,
                                   self.config.local_buffer_size,
                                   self.config.protocol,
                                   self.config.device_name,
                                   self.config.master_server_address)
            self.tensor_pool = TensorMemoryPool(
                max_block_size=self.config.fast_transfer_buffer_size *
                DEFAULT_TENSOR_POOL_SIZE)
            self.store.register_buffer(
                self.tensor_pool.base_address,
                self.config.fast_transfer_buffer_size *
                DEFAULT_TENSOR_POOL_SIZE)
            self.fifo_pool_queue: deque[MooncakeTensorPoolMetadata] = deque()
            self.replica_config = ReplicateConfig()
            self.replica_config.replica_num = 1
        else:
            # This is for ADXL. If ascend direct transport is on,
            # global transfer engine for an instance is required
            local_hostname = get_ip()
            transfer_engine = get_global_te(local_hostname, device_name=None)
            self.local_seg = local_hostname + ":" + str(
                transfer_engine.get_rpc_port())
            ret = self.store.setup(self.local_seg, self.config.metadata_server,
                                   self.config.global_segment_size,
                                   self.config.local_buffer_size,
                                   self.config.protocol,
                                   self.config.device_name,
                                   self.config.master_server_address,
                                   transfer_engine.get_engine())

        if ret != 0:
            msg = "Initialize mooncake failed."
            logger.error(msg)
            raise RuntimeError(msg)

    def exists(self, key: MooncakeEngineKey) -> bool:
        return self.store.is_exist(key.to_string()) == 1

    def batch_exists(self, keys: list[str]) -> list[int]:
        return self.store.batch_is_exist(keys)

    def register_buffer(self, ptr, length):
        return self.store.register_buffer(ptr, length)

    def get_batch(self, keys: list[str], addrs: list[list[int]],
                  sizes: list[list[int]], block_ids: list[int]):
        try:
            res = self.store.batch_get_into_multi_buffers(
                keys, addrs, sizes, True)
            for value in res:
                if value < 0:
                    logger.error(f"Failed to get key {keys},res:{res}")
        except Exception as e:
            logger.error(f"Failed to get key {keys}. {e}")

    def put_batch(self, keys: list[str], addrs: list[list[int]],
                  sizes: list[list[int]], block_ids: list[int]):
        try:
            config = ReplicateConfig()
            config.prefer_alloc_in_same_node = True
            res = self.store.batch_put_from_multi_buffers(
                keys, addrs, sizes, config)
            for value in res:
                if value < 0:
                    logger.error(f"Failed to put key {keys},res:{res}")
        except Exception as e:
            logger.error(f"Failed to put key {keys},error:{e}")

    def get_batch_tcp(self, keys: list[str],
                      k_caches: list[list[torch.Tensor]],
                      v_caches: list[list[torch.Tensor]],
                      block_ids: list[int]):
        if not keys:
            return
        buffer_addrs = []
        buffer_sizes = []
        sample_tensor = k_caches[0][0]
        device = sample_tensor.device
        buffer_dtype = sample_tensor.dtype
        buffer_shape = sample_tensor.shape
        block_buffer_size = sample_tensor.numel() * sample_tensor.element_size(
        )

        alloc_func = self._pool_allocate

        for k_list in k_caches:
            num_allocs = len(k_list) * 2
            current_block_addrs = [
                alloc_func(block_buffer_size) for _ in range(num_allocs)
            ]
            buffer_addrs.append(current_block_addrs)
            buffer_sizes.append([block_buffer_size] * num_allocs)
        try:
            read_bytes = self.store.batch_get_into_multi_buffers(
                keys, buffer_addrs, buffer_sizes, True)
        except Exception as e:
            logger.error("batch_get_into failed: %s", str(e))
            return

        for i, (key, addrs,
                r_byte) in enumerate(zip(keys, buffer_addrs, read_bytes)):

            if r_byte > 0:
                current_k_list = k_caches[i]
                current_v_list = v_caches[i]

                # addrs: [k0_addr, v0_addr, k1_addr, v1_addr, ...]
                addr_idx = 0
                for k_tensor, v_tensor in zip(current_k_list, current_v_list):
                    k_addr = addrs[addr_idx]
                    temp_k = self.tensor_pool.load_tensor(
                        k_addr, buffer_dtype, buffer_shape, device)
                    k_tensor.copy_(temp_k)

                    self.tensor_pool.free(k_addr)
                    addr_idx += 1

                    v_addr = addrs[addr_idx]
                    temp_v = self.tensor_pool.load_tensor(
                        v_addr, buffer_dtype, buffer_shape, device)
                    v_tensor.copy_(temp_v)

                    self.tensor_pool.free(v_addr)
                    addr_idx += 1
            else:
                for addr in addrs:
                    self.tensor_pool.free(addr)

    def put_batch_tcp(self, keys: list[str],
                      k_caches: list[list[torch.Tensor]],
                      v_caches: list[list[torch.Tensor]],
                      block_ids: list[int]):
        if not keys:
            return

        sample_tensor = k_caches[0][0]
        block_buffer_size = sample_tensor.numel() * sample_tensor.element_size(
        )

        buffer_addrs = []
        buffer_sizes = []

        for key, k_list, v_list in zip(keys, k_caches, v_caches):
            current_block_addrs = []

            for k, v in zip(k_list, v_list):
                current_block_addrs.append(self._pool_store_tensor(k))
                current_block_addrs.append(self._pool_store_tensor(v))

            self.fifo_pool_queue.append(
                MooncakeTensorPoolMetadata(key, current_block_addrs))

            buffer_addrs.append(current_block_addrs)
            buffer_sizes.append([block_buffer_size] * len(current_block_addrs))

        try:
            config = ReplicateConfig()
            config.prefer_alloc_in_same_node = True

            self.store.batch_put_from_multi_buffers(
                keys,
                buffer_addrs,
                buffer_sizes,
                config,
            )
        except Exception as e:
            logger.error(
                "Failed to put metadata for keys %s "
                "using put_batch with error %s",
                ",".join(keys),
                str(e),
            )

    def get(self, key: MooncakeEngineKey, addr: list[int], size: list[int]):
        expect_res = sum(size)
        key_str = key.to_string()
        try:
            res = self.store.batch_get_into_ascend(key_str, addr, size)
            if res[0] != expect_res:
                logger.error(f"Failed to get key: [{key_str}] .")
        except Exception:
            logger.error(f"Failed to get key: [{key_str}] .")
        return res

    def put(self, key: MooncakeEngineKey, addr: list[int], size: list[int]):
        key_str = key.to_string()
        try:
            ret = self.store.batch_put_from_ascend(key_str, addr, size)
            if ret[0] != 0:
                logger.error(f"Failed to put key {key_str}.")
        except Exception:
            logger.error(f"Failed to put key {key_str}.")

        return ret

    def close(self):
        self.store.unregister_buffer(self.tensor_pool.base_address, 1)
        self.tensor_pool.cleanup()
        self.store.close()
        logger.info("Closed the mooncake store connection")

    # ==============================
    # Tensor pool helper functions
    # ==============================

    def metadata_key(self, key: str) -> str:
        # TODO: no guarantee that there is no (k,v) with this key
        return key + "_metadata"

    def _pool_eviction(self) -> None:
        evicted_buffer = self.fifo_pool_queue.popleft()
        for addr in evicted_buffer.addrs:
            self.tensor_pool.free(addr)
        key = re.escape(evicted_buffer.key)
        count = self.store.remove_by_regex(f"^(?:{key})$")
        if count > 1:
            logger.error("count of key and meta key exceeds 2")

    def _pool_allocate(self, size: int) -> int:
        while True:
            try:
                return self.tensor_pool.allocate(size)
            except InsufficientMemoryError:
                if not self.fifo_pool_queue:
                    raise

                self._pool_eviction()

    def _pool_store_tensor(self, tensor: torch.Tensor) -> int:
        while True:
            try:
                return self.tensor_pool.store_tensor(tensor)
            except InsufficientMemoryError:
                if not self.fifo_pool_queue:
                    raise

                self._pool_eviction()
