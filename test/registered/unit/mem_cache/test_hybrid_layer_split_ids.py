"""Hybrid (KDA + DSA) layer-split pools map dense layer indices back to global ids.

The full-attention child of ``HybridLinearKVPool`` is addressed by dense layer
index (0..n-1), while PD/HiCache need the global model layer ids. These tests
pin that mapping, the uneven ``shard_size=4, n=10`` boundary, and the wrapper
delegation added for the split root.
"""

import unittest
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.hardware_backend.npu import memory_pool_npu
from sglang.srt.hardware_backend.npu.dsa_cache_layer_split import (
    LayerSplitNPUMLATokenToKVPool,
)
from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMLATokenToKVPool
from sglang.srt.layers.cp.utils import get_layer_shard_range
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

_MLA = dict(size=16, page_size=4, dtype=torch.bfloat16, device="cpu")
# 10 full-attention (DSA) layers at global ids 0, 2, ..., 18 -> dense 0..9.
_FULL_IDS = tuple(range(0, 20, 2))
_SHARD_SIZE = 4
# shard_size=4, n=10: the first two ranks own one extra layer.
_EXPECTED_RANGES = [(0, 3), (3, 6), (6, 8), (8, 10)]


def _recurrent_state(layer_ids):
    state = torch.zeros(len(layer_ids), 4, 8)
    return SimpleNamespace(
        get_contiguous_buf_infos=lambda: (
            [state.data_ptr()],
            [state.nbytes],
            [state[0].nbytes],
        ),
        get_state_dim_per_tensor=lambda: [8],
        get_state_layer_ids=lambda: list(layer_ids),
        get_state_slice_outer_counts=lambda: [1],
        get_state_conv_shard_groups=lambda: [None],
    )


def _split_pool(rank):
    return LayerSplitNPUMLATokenToKVPool(
        **_MLA,
        kv_lora_rank=16,
        qk_rope_head_dim=8,
        layer_num=len(_FULL_IDS),
        enable_memory_saver=False,
        layer_shard_rank=rank,
        layer_shard_size=_SHARD_SIZE,
        layer_id_mapping=_FULL_IDS,
    )


class TestHybridLayerSplitIds(CustomTestCase):
    def setUp(self):
        parallel = patch.object(
            memory_pool_npu,
            "get_parallel",
            return_value=SimpleNamespace(attn_dcp_size=1, attn_dcp_rank=0),
        )
        parallel.start()
        self.addCleanup(parallel.stop)

    def test_uneven_shard_ranges_and_shard_start(self):
        for rank in range(_SHARD_SIZE):
            start, end = get_layer_shard_range(rank, _SHARD_SIZE, len(_FULL_IDS))
            self.assertEqual((start, end), _EXPECTED_RANGES[rank])
            pool = _split_pool(rank)
            self.assertEqual(pool.layer_num, len(_FULL_IDS))
            self.assertEqual(pool.start_layer, 0)
            self.assertEqual(pool._owned_local_layer_range(), (start, end))
            # A3: layer_shard_start stays a dense (stage-local) index.
            self.assertEqual(pool.layer_shard_start, start)

    def test_kv_layer_ids_are_owned_global_ids(self):
        for rank in range(_SHARD_SIZE):
            pool = _split_pool(rank)
            start, end = _EXPECTED_RANGES[rank]
            owned = list(_FULL_IDS[start:end])
            data_ptrs, _, _ = pool.get_contiguous_buf_infos()
            ids = pool.get_kv_layer_ids()
            # k + v buffer groups repeat the owned (global) layer ids.
            self.assertEqual(len(ids), len(data_ptrs))
            self.assertEqual(ids, owned * (len(data_ptrs) // len(owned)))
            self.assertEqual(
                pool.get_kv_transfer_layer_span(len(data_ptrs)), end - start
            )
            # No indexer layers on this MLA-only pool.
            self.assertEqual(pool.get_state_layer_ids(), [])

    def test_hybrid_wrapper_delegates_split_state(self):
        rank = 1
        pool = HybridLinearKVPool(
            **_MLA,
            head_num=1,
            head_dim=16,
            full_attention_layer_ids=list(_FULL_IDS),
            mamba_pool=_recurrent_state((1, 3, 5)),
            use_mla=True,
            kv_lora_rank=16,
            qk_rope_head_dim=8,
            full_kv_pool_class=partial(
                LayerSplitNPUMLATokenToKVPool,
                layer_shard_rank=rank,
                layer_shard_size=_SHARD_SIZE,
                layer_id_mapping=_FULL_IDS,
            ),
        )
        self.assertTrue(pool.layer_shard_enabled)
        self.assertEqual(pool.layer_shard_rank, rank)
        self.assertEqual(pool.layer_shard_size, _SHARD_SIZE)
        # The dense child never sees the wrapper's global start_layer.
        self.assertEqual(pool.full_kv_pool.layer_num, len(_FULL_IDS))
        self.assertEqual(pool.full_kv_pool.start_layer, 0)
        start, end = _EXPECTED_RANGES[rank]
        self.assertEqual(pool.layer_shard_start, start)
        owned = list(_FULL_IDS[start:end])
        data_ptrs, _, _ = pool.get_contiguous_buf_infos()
        self.assertEqual(len(pool.get_kv_layer_ids()), len(data_ptrs))
        self.assertEqual(pool.get_kv_layer_ids(), owned * (len(data_ptrs) // len(owned)))

    def test_non_split_wrapper_uses_default_shard_values(self):
        pool = HybridLinearKVPool(
            **_MLA,
            head_num=1,
            head_dim=16,
            full_attention_layer_ids=list(_FULL_IDS),
            mamba_pool=_recurrent_state((1, 3, 5)),
            use_mla=True,
            kv_lora_rank=16,
            qk_rope_head_dim=8,
            full_kv_pool_class=NPUMLATokenToKVPool,
        )
        # A3: 非 split 的 full pool 没有分片属性，委托必须回落到默认值。
        self.assertFalse(pool.layer_shard_enabled)
        self.assertIsNone(pool.layer_shard_rank)
        self.assertEqual(pool.layer_shard_size, 1)
        self.assertEqual(pool.layer_shard_start, pool.start_layer)
        self.assertEqual(pool.get_kv_transfer_layer_span(6), 6)


if __name__ == "__main__":
    unittest.main()
