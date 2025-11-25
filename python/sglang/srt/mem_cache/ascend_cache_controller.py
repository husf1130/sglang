from __future__ import annotations

import logging
from queue import Empty
from typing import List, Optional, Any
from itertools import chain

import torch

from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator

from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.layers.dp_attention import (
    get_attention_dp_rank,
    get_attention_tp_rank,
    get_attention_tp_size,
    is_dp_attention_enabled,
)
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool

from sglang.srt.mem_cache.hicache_storage import get_hash_str, HiCacheStorageConfig
from sglang.srt.managers.cache_controller import LayerDoneCounter
from sglang.srt.mem_cache.storage import StorageBackendFactory

logger = logging.getLogger(__name__)


def get_hash_list(token_ids: List[int], prior_hash: str = None, page_size: int = 128) -> List[str]:
    assert len(token_ids) % page_size == 0
    hashes = []
    last_hash = prior_hash
    token_groups = (token_ids[i:i + page_size] for i in range(0, len(token_ids), page_size))
    for group in token_groups:
        last_hash = get_hash_str(group, last_hash)
        hashes.append(last_hash)
    return hashes


class LoadStorageOperation:
    counter = 0

    def __init__(
        self,
        request_id: str,
        device_indices: Optional[torch.Tensor] = None,
        token_ids: Optional[List[Any]] = None,
        last_hash: Optional[str] = None,
        page_size: int = 128,
    ):
        self.request_ids = [request_id]
        self.device_indices = [device_indices] if device_indices is not None else []

        if token_ids is not None:
            self.token_ids = [token_ids]
            self.token_lens = [len(token_ids)]
            self.last_hash = [last_hash]
            hashes = get_hash_list(token_ids, last_hash, page_size)
            self.hash_keys = [hashes]
            self.hash_lens = [len(hashes)]
        else:
            self.token_ids = []
            self.token_lens = []
            self.last_hash = []
            self.hash_keys = []
            self.hash_lens = []

        self.free_device_indices = None
        self.hit_device_indices = None

        self.id = LoadStorageOperation.counter
        LoadStorageOperation.counter += 1

    @staticmethod
    def merge_ops(ops: List[LoadStorageOperation]) -> LoadStorageOperation:
        assert len(ops) > 0
        if len(ops) == 1:
            return ops[0]

        request_ids = []
        device_indices = []
        token_ids = []
        token_lens = []
        last_hash = []
        hash_keys = []
        hash_lens = []
        for op in ops:
            request_ids.extend(op.request_ids)
            device_indices.extend(op.device_indices)
            token_ids.extend(op.token_ids)
            token_lens.extend(op.token_lens)
            last_hash.extend(op.last_hash)
            hash_keys.extend(op.hash_keys)
            hash_lens.extend(op.hash_lens)
        merged_op = LoadStorageOperation("")
        merged_op.request_ids = request_ids
        merged_op.device_indices = device_indices
        merged_op.token_ids = token_ids
        merged_op.token_lens = token_lens
        merged_op.last_hash = last_hash
        merged_op.hash_keys = hash_keys
        merged_op.hash_lens = hash_lens
        return merged_op


class AscendHiCacheController:

    def __init__(
        self,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        page_size: int,
        tp_group: torch.distributed.ProcessGroup,
        storage_backend: str,
        device_id: int = 0,
    ):
        self.mem_pool_device_allocator = token_to_kv_pool_allocator
        self.mem_pool_device = token_to_kv_pool_allocator.get_kvcache()

        # self.kv_layer_ptrs: every layer ptr
        # self.kv_layer_nbytes: the byte length of each layer
        # self.kv_page_nbytes: the page byte length of each layer
        self.kv_layer_ptrs, self.kv_layer_nbytes, self.kv_page_nbytes = (
            self.mem_pool_device.get_contiguous_buf_infos()
        )

        self.page_size = page_size
        self.get_hash_str = get_hash_str
        self.device_id = device_id
        self.is_mla_model = isinstance(self.mem_pool_device, MLATokenToKVPool)

        if is_dp_attention_enabled():
            self.tp_rank = get_attention_tp_rank()
            self.tp_size = get_attention_tp_size()
            self.dp_rank = get_attention_dp_rank()
        else:
            self.tp_rank = get_tensor_model_parallel_rank()
            self.tp_size = get_tensor_model_parallel_world_size()
            self.dp_rank = 0

        # for MLA models, only one rank needs to backup the KV cache
        self.backup_skip = self.is_mla_model and self.tp_rank != 0

        self.storage_config = HiCacheStorageConfig(
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            is_mla_model=self.is_mla_model,
            is_page_first_layout=False,
            model_name=None,
            extra_config={"device_id": device_id},
        )
        try:
            self.storage_backend = StorageBackendFactory.create_backend(
                storage_backend, self.storage_config, None
            )
        except ValueError as e:
            raise ValueError(f"Failed to create storage backend: {e}") from e

        self.storage_backend.register_mem_pool_device(self.mem_pool_device)

        self.load_pages_threshold = 1
        # granularity of batch storage IO operations, in number of pages
        self.storage_batch_size = 256

        # create a new communication group for synchronizing storage operations across TP workers
        self.tp_world_size = torch.distributed.get_world_size(group=tp_group)
        if self.tp_world_size > 1:
            group_ranks = torch.distributed.get_process_group_ranks(tp_group)
            self.load_tp_group = torch.distributed.new_group(
                group_ranks, backend="gloo"
            )

        self.layer_num = self.mem_pool_device.layer_num
        self.layer_done_counter = LayerDoneCounter(self.layer_num)
        self.mem_pool_device.register_layer_transfer_counter(self.layer_done_counter)

        self.load_queue: List[LoadStorageOperation] = []

    def reset(self):
        self.load_queue.clear()

    def write(self, device_indices: torch.Tensor, origin_req_tokens: List[int]) -> int:
        if self.backup_skip:
            return 0

        try:
            hash_keys = [get_hash_list(origin_req_tokens)]
            write_results = self._memcpy_between_device_and_storage(hash_keys, [device_indices], "write")
            if self.tp_world_size > 1 and self.is_mla_model is False:
                # only mha model need all reduce
                write_results = self._allreduce_results(write_results)

            # fresh hash keys and its len get successfully
            # self._parse_success_hashes_from_l3_results(hash_keys, [len(hash_keys[0])], write_results)

            return write_results.count(1) * self.page_size
        except Empty:
            return 0

    def load(
        self,
        rid,
        new_input_tokens,
        last_hash: Optional[str] = None,
    ) -> Optional[torch.Tensor]:
        """
        Load KV caches from L3 storage to device memory.
        """
        device_indices = self.mem_pool_device_allocator.alloc(len(new_input_tokens))
        if device_indices is None:
            return None

        self.load_queue.append(LoadStorageOperation(rid, device_indices, new_input_tokens, last_hash))
        return device_indices

    def start_loading(self) -> Optional[LoadStorageOperation]:
        if len(self.load_queue) == 0:
            return None

        producer_id = self.layer_done_counter.update_producer()
        op = LoadStorageOperation.merge_ops(self.load_queue)
        self.load_queue.clear()
        producer_event = self.layer_done_counter.events[producer_id]
        producer_event.start_event.record()

        try:
            hit_group_hash_keys, hit_hash_lens, hit_token_lens = self._storage_hit_query(op)
            if self.tp_world_size > 1:
                hit_hash_lens = self._allreduce_results(hit_hash_lens)
                hit_token_lens = [x * self.page_size for x in hit_hash_lens]

            if sum(hit_hash_lens) < self.load_pages_threshold:
                # not to load storage if not enough benefits
                op.free_device_indices = op.device_indices
                logger.debug(
                    f"Revoking Load operation for request {op.request_ids} due to insufficient hits ({hit_token_lens})."
                )
                op.token_lens = [0 for _ in op.token_lens]
                return op
            else:
                hit_group_hash_keys = [group[:length] for group, length in zip(hit_group_hash_keys, hit_hash_lens)]
                device_indices = [group[:length] for group, length in zip(op.device_indices, hit_token_lens)]
                load_results = self._memcpy_between_device_and_storage(hit_group_hash_keys, device_indices, "load")
                if self.tp_world_size > 1:
                    load_results = self._allreduce_results(load_results)

                # fresh hash keys and its len get successfully
                (
                    op.hash_keys, op.hash_lens, op.token_lens
                ) = self._parse_success_hashes_from_l3_results(
                    hit_group_hash_keys, hit_hash_lens, load_results
                )
                op.free_device_indices = [
                    indices[hit_len:] for indices, hit_len in zip(op.device_indices, op.token_lens)
                ]

                op.hit_device_indices = [
                    indices[:hit_len] for indices, hit_len in zip(op.device_indices, op.token_lens)
                ]
                op.token_ids = [ids[hit_len:] for ids, hit_len in zip(op.token_ids, op.token_lens)]
                logger.debug(f"Load storage {sum(op.hash_lens)} pages for request {op.request_ids}.")
                return op

        except Empty:
            logger.error(f"Load storage {sum(op.hash_lens)} pages for request {op.request_ids}.")
            op.free_device_indices = op.device_indices
            op.token_lens = [0 for _ in op.token_lens]
            return op

    def _allreduce_results(self, results):
        results_tensor = torch.tensor(
            results, dtype=torch.int
        )
        torch.distributed.all_reduce(
            results_tensor,
            op=torch.distributed.ReduceOp.MIN,
            group=self.load_tp_group,
        )
        return results_tensor.tolist()

    def _storage_hit_query(self, operation: LoadStorageOperation) -> tuple[list[Any], list[Any], list[Any]]:
        assert len(operation.hash_keys) == len(operation.hash_lens)
        flatten_hash_keys = list(chain.from_iterable(operation.hash_keys))
        if not operation.hash_keys:
            return [], [], []

        exist_results = []
        total_len = len(flatten_hash_keys)
        for start in range(0, total_len, self.storage_batch_size):
            end = min(start + self.storage_batch_size, total_len)
            batch_hashes = flatten_hash_keys[start:end]
            hit_results = self.storage_backend.batch_exists(batch_hashes)
            exist_results.extend(hit_results)

        return self._parse_success_hashes_from_l3_results(operation.hash_keys, operation.hash_lens, exist_results)

    def _memcpy_between_device_and_storage(
        self,
        hit_group_hash_keys: List[List[str]],
        device_indices: List[torch.Tensor],
        direction: str,
    ) -> Optional[list[int]]:
        batch_memcpy = None
        if direction == "write":
            batch_memcpy = self.storage_backend.batch_set
        elif direction == "load":
            batch_memcpy = self.storage_backend.batch_get
        assert batch_memcpy is not None

        flatten_hash_keys = [key for keys in hit_group_hash_keys for key in keys]
        if not flatten_hash_keys:
            return []

        ptr_list, element_size_list = self._get_page_buffer_meta(device_indices)
        assert len(flatten_hash_keys) == len(ptr_list)
        assert len(flatten_hash_keys) == len(element_size_list)
        results = []
        total_elements = len(flatten_hash_keys)
        for start in range(0, total_elements, self.storage_batch_size):
            end = min(start + self.storage_batch_size, total_elements)
            batch_hashes = flatten_hash_keys[start:end]
            target_locations = ptr_list[start:end]
            target_sizes = element_size_list[start:end]
            memcpy_results = batch_memcpy(
                keys=batch_hashes,
                target_locations=target_locations,
                target_sizes=target_sizes
            )
            results.extend(memcpy_results)

        return results

        # flatten_hash_keys = [key for keys in hit_group_hash_keys for key in keys]
        # ptr_list, element_size_list = self._get_page_buffer_meta(device_indices)
        # assert len(flatten_hash_keys) == len(ptr_list) == len(element_size_list)
        #
        # if direction == "write":
        #     return self.storage_backend.batch_set(
        #         keys=flatten_hash_keys,
        #         target_locations=ptr_list,
        #         target_sizes=element_size_list
        #     )
        # elif direction == "load":
        #     return self.storage_backend.batch_get(
        #         keys=flatten_hash_keys,
        #         target_locations=ptr_list,
        #         target_sizes=element_size_list
        #     )

    def _parse_success_hashes_from_l3_results(
        self,
        group_hash_keys: List[List[str]],
        hash_lens: List[int],
        hash_results: List[int],
    ) -> tuple[List[Any], List[int], List[int]]:
        # 1st, group hash_results by hash_lens
        group_results: List[List[Any]] = []
        start = 0
        for length in hash_lens:
            end = start + length
            group_results.append(hash_results[start: start + length])
            start = end

        # 2nd, Extract the success hit hash keys and lengths
        hit_group_hash_keys: List[List[str]] = []
        hit_hash_lens: List[int] = []
        hit_token_lens: List[int] = []

        # for each request
        for hash_keys, results in zip(group_hash_keys, group_results):
            hit_hashes: List[str] = []
            hit_hash_len = 0
            for h, r in zip(hash_keys, results):
                if r == 1:
                    hit_hashes.append(h)
                    hit_hash_len += 1
                else:
                    break
            hit_group_hash_keys.append(hit_hashes)
            hit_hash_lens.append(hit_hash_len)
            hit_token_lens.append(hit_hash_len * self.page_size)
        return hit_group_hash_keys, hit_hash_lens, hit_token_lens

    def _get_page_buffer_meta(self, device_indices: List[torch.Tensor]):
        ptr_list = []
        element_size_list = []
        flatten_indices_tensor = torch.cat(device_indices)
        flatten_index_list = flatten_indices_tensor.tolist()
        assert len(flatten_index_list) % self.page_size == 0

        for index in range(0, len(flatten_index_list), self.page_size):
            # convert device index to page index
            page_index = flatten_index_list[index] // self.page_size

            ptrs = []
            sizes = []
            for layer_start_ptr, page_nbytes in zip(self.kv_layer_ptrs, self.kv_page_nbytes):
                layer_ptr = layer_start_ptr + page_index * page_nbytes
                ptrs.append(layer_ptr)
                sizes.append(page_nbytes)

            ptr_list.append(ptrs)
            element_size_list.append(sizes)

        return ptr_list, element_size_list
