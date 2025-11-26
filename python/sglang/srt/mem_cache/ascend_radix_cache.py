import heapq

import logging
import time
from typing import List, Optional, Tuple

import torch

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import (
    ReqToTokenPool,
)
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey, TreeNode
from sglang.srt.mem_cache.ascend_cache_controller import AscendHiCacheController, LoadStorageOperation, get_hash_list
from sglang.srt.metrics.collector import StorageMetricsCollector

logger = logging.getLogger(__name__)


class AscendHiRadixCache(RadixCache):

    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        tp_cache_group: torch.distributed.ProcessGroup,
        page_size: int,
        enable_metrics: bool,
        eviction_policy: str = "lru",
        hicache_storage_backend: Optional[str] = None,
        is_eagle: bool = False,
        device_id: int = 0
    ):
        self.kv_cache = token_to_kv_pool_allocator.get_kvcache()
        self.allocator = token_to_kv_pool_allocator
        self.enable_storage = True
        self.enable_storage_metrics = enable_metrics
        self.hicache_storage_pass_prefix_keys = False
        self.cache_controller = AscendHiCacheController(
            token_to_kv_pool_allocator,
            page_size,
            tp_cache_group,
            hicache_storage_backend,
            device_id=device_id,
        )
        if self.enable_storage_metrics:
            labels = {
                "storage_backend": hicache_storage_backend,
                "tp_rank": self.cache_controller.tp_rank,
                "dp_rank": self.cache_controller.dp_rank,
            }
            self.metrics_collector = StorageMetricsCollector(labels=labels)

        super().__init__(
            req_to_token_pool,
            token_to_kv_pool_allocator,
            page_size,
            disable=False,
            eviction_policy=eviction_policy,
            is_eagle=is_eagle,
        )

    def reset(self):
        TreeNode.counter = 0
        self.cache_controller.reset()
        super().reset()

    def clear_storage_backend(self) -> bool:
        try:
            # Check if the storage backend has a clear method (for nixl backends)
            if hasattr(self.cache_controller.storage_backend, "clear"):
                self.cache_controller.storage_backend.clear()
                return True
            else:
                logger.warning("hierarchical cache memcache store does not support clear operation.")
                return False
        except Exception as e:
            logger.error(f"Failed to clear hierarchical cache storage backend: {e}")
            return False

    def evict(self, num_tokens: int):
        leaves = self._collect_leaves_device()
        eviction_heap = [
            (self.eviction_strategy.get_priority(node), node) for node in leaves
        ]
        heapq.heapify(eviction_heap)

        num_evicted = 0
        while num_evicted < num_tokens and len(eviction_heap):
            _priority, x = heapq.heappop(eviction_heap)

            if x.lock_ref > 0:
                continue

            if not x.backuped:
                num_evicted += self._evict_regular(x)

            for child in x.parent.children.values():
                if not child.evicted:
                    break
            else:
                # all children are evicted or no children
                new_priority = self.eviction_strategy.get_priority(x.parent)
                heapq.heappush(eviction_heap, (new_priority, x.parent))

    def _evict_regular(self, node: TreeNode):
        # evict a node not initiated write to host
        self.cache_controller.mem_pool_device_allocator.free(node.value)
        num_evicted = len(node.value)
        self._delete_leaf(node)
        return num_evicted

    def load_back(
        self, rid: str, node: TreeNode, new_input_tokens: List[int], mem_quota: Optional[int] = None
    ) -> torch.Tensor:
        # todo: more loading policies

        start_time = time.perf_counter()
        last_hit_node = node
        # protect the last_hit_node from eviction
        self.inc_lock_ref(last_hit_node)

        device_indices = self.cache_controller.load(
            rid=rid,
            new_input_tokens=new_input_tokens,
            last_hash=node.get_last_hash_value()
        )

        if device_indices is None:
            self.evict(len(new_input_tokens))
            device_indices = self.cache_controller.load(
                rid=rid,
                new_input_tokens=new_input_tokens,
                last_hash=node.get_last_hash_value()
            )
        self.dec_lock_ref(last_hit_node)
        # self.inc_lock_ref(last_hit_node)

        if self.metrics_collector is not None:
            self.metrics_collector.observe_load_back_duration(
                time.perf_counter() - start_time
            )
            self.metrics_collector.increment_load_back_num_tokens(len(device_indices))
        return device_indices

    def init_load_back(
        self,
        req: Req,
        mem_quota: Optional[int] = None,
    ):
        start = time.time()

        req.ongoing_loading_l3 = False

        matched_len = len(req.prefix_indices)
        new_input_tokens = req.fill_ids[matched_len:]
        if len(new_input_tokens) <= self.page_size:
            return None

        remainder = len(new_input_tokens) % self.page_size
        if remainder == 0:
            # to avoid input tokens = 0 while hit the entire input tokens
            new_input_tokens = new_input_tokens[:(len(new_input_tokens) - self.page_size)]
        else:
            new_input_tokens = new_input_tokens[:(len(new_input_tokens) - remainder)]

        last_node = req.last_node
        if not last_node.evicted:
            loading_values = self.load_back(req.rid, last_node, new_input_tokens, mem_quota)
            if loading_values is not None:
                req.ongoing_loading_l3 = True
                # logger.debug(f"loading back {req.req_id=} {len(loading_values)} tokens for node {last_node.id}")
            else:
                logger.debug(f"init_load_back {req.req_id=} loading_values is None")

        else:
            # should not enter this branch
            while last_node.evicted:
                last_node = last_node.parent

        req.last_node = last_node

        end = time.time()
        logger.info(f"init_load_back finished, {req.req_id=}, duration {(end - start) * 1000:.3f}ms")
        return None

    def ready_to_load_cache(self, can_run_list: List[Req] = None, adder = None) -> int:
        """
        Notify the cache controller to start the KV cache loading.
        """
        start = time.time()
        operation: LoadStorageOperation = self.cache_controller.start_loading()
        if operation is not None:
            self._update_req_prefix_after_load(operation, can_run_list, adder)
        end = time.time()
        logger.info(f"ready_to_load_cache finished, duration {(end - start) * 1000:.3f}ms")
        return -1

    def _update_req_prefix_after_load(
        self,
        op: LoadStorageOperation,
        can_run_list: List[Req],
        adder,
    ):
        assert can_run_list is not None
        assert adder is not None
        load_req_list = [req for req in can_run_list if req.ongoing_loading_l3]

        total_load_length = 0
        for req, token_ids, new_indices, free_indices, hashes, length \
            in zip(load_req_list, op.token_ids, op.device_indices, op.free_device_indices, op.hash_keys, op.token_lens):
            if length > 0:
                # TODO: insert one new node into the radix tree
                req.prefix_indices = torch.cat([req.prefix_indices, new_indices])
                req.extend_input_len = len(req.fill_ids) - len(req.prefix_indices)
                prefix_len = len(req.prefix_indices)
                req.last_matched_prefix_len = prefix_len
                adder.update_prefill_budget(length, -length, 0)

                total_load_length += length
                self.cache_controller.mem_pool_device_allocator.free(free_indices)
                if self.enable_storage_metrics:
                    self.metrics_collector.log_prefetched_tokens(length)
            else:
                self.cache_controller.mem_pool_device_allocator.free(free_indices)

        logger.debug(
            f"success loading {total_load_length} tokens from l3 storage")
        return total_load_length

    def check_hicache_events(self):
        if self.enable_storage_metrics:
            self.metrics_collector.log_storage_metrics(
                self.cache_controller.storage_backend.get_stats()
            )

    def check_prefetch_progress(self, req_id: str) -> bool:
        return True

    def prefetch_from_storage(
        self,
        req_id: str,
        last_host_node: TreeNode,
        new_input_tokens: List[int],
        last_hash: Optional[str] = None,
    ):
        pass

    def insert(self, key: RadixKey, value=None, chunked=False):
        start = time.time()

        key.token_ids = self.key_convert_fn(key.token_ids)

        if len(key) == 0:
            return 0

        if self.is_eagle and value is not None:
            # Make sure the value len equal to the EAGLE bigram key len
            value = value[: len(key)]

        origin_req_tokens = key.token_ids[:]
        origin_values = value.clone()

        node = self.root_node
        child_key = self.get_child_key_fn(key)
        total_prefix_length = 0
        hash_keys = []
        while len(key) > 0 and child_key in node.children.keys():

            node = node.children[child_key]
            node.last_access_time = time.monotonic()
            prefix_len = self.key_match_fn(node.key, key)

            if prefix_len == len(node.key):
                self._inc_hit_count(node, chunked)
                total_prefix_length += prefix_len
            else:
                # partial match, split the node
                new_node = self._split_node(node.key, node, prefix_len)
                self._inc_hit_count(new_node, chunked)
                total_prefix_length += prefix_len
                node = new_node

            if self.enable_storage:
                hash_keys.extend(node.hash_value)

            key = key[prefix_len:]
            value = value[prefix_len:]

            if len(key):
                child_key = self.get_child_key_fn(key)

        if len(key):
            new_node = TreeNode()
            new_node.parent = node
            new_node.key = key
            new_node.value = value
            node.children[child_key] = new_node
            self.evictable_size_ += len(value)

            if self.enable_storage:
                last_hash = node.get_last_hash_value()
                assert (node == self.root_node) or (
                    last_hash is not None
                ), "Parent node must have a hash value with storage enabled"
                new_node.hash_value = get_hash_list(key.token_ids, last_hash, self.page_size)
                hash_keys.extend(new_node.hash_value)

            self._inc_hit_count(new_node, chunked)

        if self.enable_storage:
            self.write_storage(origin_req_tokens, origin_values, hash_keys)

        end = time.time()
        logger.info(f"insert finished, duration {(end - start) * 1000:.3f}ms")
        return total_prefix_length

    def write_storage(
        self,
        origin_req_tokens,
        device_indices,
        hash_keys: List[str],
    ):
        start = time.time()
        if len(origin_req_tokens) == 0:
            return
        assert len(origin_req_tokens) == len(device_indices)
        assert len(origin_req_tokens) == len(hash_keys) * self.page_size
        succ_num_tokens = self.cache_controller.write(device_indices, hash_keys)

        if self.enable_storage_metrics:
            self.metrics_collector.log_backuped_tokens(succ_num_tokens)

        end = time.time()
        logger.info(f"write_storage finished, duration {(end - start) * 1000:.3f}ms")

    def _inc_hit_count(self, node: TreeNode, chunked=False):
        # skip the hit count update for chunked requests
        if chunked:
            return
        node.hit_count += 1

    def _split_node(self, key: RadixKey, child: TreeNode, split_len: int):
        # child node split into new_node -> child
        new_node = TreeNode()
        new_node.children = {self.get_child_key_fn(key[split_len:]): child}
        new_node.parent = child.parent
        new_node.lock_ref = child.lock_ref
        new_node.key = child.key[:split_len]
        new_node.hit_count = child.hit_count

        # split value and host value if exists
        if child.evicted:
            new_node.value = None
        else:
            new_node.value = child.value[:split_len]
            child.value = child.value[split_len:]
        if child.backuped:
            new_node.host_value = child.host_value[:split_len]
            child.host_value = child.host_value[split_len:]

        if child.hash_value:
            new_node.hash_value = child.hash_value[: split_len // self.page_size]
            child.hash_value = child.hash_value[split_len // self.page_size:]
        child.parent = new_node
        child.key = child.key[split_len:]
        new_node.parent.children[self.get_child_key_fn(key)] = new_node
        return new_node

    def _collect_leaves_device(self):
        def is_leaf(node):
            if node.evicted:
                return False
            if node == self.root_node:
                return False
            if len(node.children) == 0:
                return True
            for child in node.children.values():
                if not child.evicted:
                    return False
            return True

        ret_list = []
        stack = [self.root_node]
        while stack:
            cur_node = stack.pop()
            if is_leaf(cur_node):
                ret_list.append(cur_node)
            else:
                for cur_child in cur_node.children.values():
                    if not cur_child.evicted:
                        stack.append(cur_child)
        return ret_list

    def release_aborted_request(self, rid: str):
        return
