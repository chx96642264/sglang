"""CP-TP group sharing admits NPU GLM-5.3 prefill cache layer split (A1).

The narrow predicate (Glm5Next + NPU + split + PD prefill + mooncake backend)
lifts the split ban only; every other combination keeps the original behavior.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups.cp_tp_group_sharing_hook import (
    resolve_cp_tp_group_sharing,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_PLATFORM = "sglang.srt.arg_groups.cp_tp_group_sharing_hook.get_platform"
_MAMBAISH = "sglang.srt.arg_groups.cp_tp_group_sharing_hook.mambaish_config"

_SPLIT_BAN = "enable-dsa-cache-layer-split is not supported with CP-TP group sharing."
_DECODE_BAN = "disaggregation-mode decode is not supported with CP-TP group sharing."
_BACKEND_BAN = (
    "disaggregation-transfer-backend != mooncake is not supported with "
    "CP-TP group sharing."
)
_PP_BAN = "pp-size > 1 is not supported with CP-TP group sharing."

_GLM5 = SimpleNamespace(
    hf_config=SimpleNamespace(architectures=["Glm5NextForConditionalGeneration"])
)
_KIMI31 = SimpleNamespace(
    hf_config=SimpleNamespace(architectures=["KimiK31ForConditionalGeneration"])
)
_QWEN4 = SimpleNamespace(
    hf_config=SimpleNamespace(architectures=["Qwen4ExpForConditionalGeneration"])
)
_LINEAR = SimpleNamespace(linear_num_key_heads=64, linear_num_value_heads=64)


def _platform(*, is_npu: bool) -> SimpleNamespace:
    return SimpleNamespace(is_npu=is_npu)


def _server_args(hf=_GLM5, **kw):
    """Target prefill config T1 (TP=4, PP=1) unless overridden."""
    base = dict(
        enable_prefill_cp=True,
        tp_size=4,
        attn_cp_size=4,
        cp_strategy="interleave",
        speculative_algorithm=None,
        enable_dsa_cache_layer_split=True,
        disaggregation_mode="prefill",
        disaggregation_transfer_backend="mooncake",
        pp_size=1,
        dp_size=1,
        dcp_size=1,
        moe_dp_size=1,
        ep_size=1,
        attn_dp_size=1,
        enable_mixed_chunk=False,
        moe_a2a_backend="none",
        enable_waterfill=False,
        prefill_attention_backend=None,
        decode_attention_backend=None,
        attention_backend=None,
    )
    base.update(kw)
    return SimpleNamespace(hf_config=hf, **base)


def _run(server_args, *, is_npu=True):
    with patch(_PLATFORM, return_value=_platform(is_npu=is_npu)):
        with patch(_MAMBAISH, return_value=_LINEAR):
            # The hook reads model.hf_config.architectures; _server_args keeps
            # the model config object on server_args.hf_config.
            resolve_cp_tp_group_sharing(server_args, server_args.hf_config)


class TestNpuLayerSplitAdmitted(CustomTestCase):
    def test_t1_pp1_is_admitted(self):
        server_args = _server_args()
        _run(server_args)
        source, fields = server_args._resolved_overrides[-1]
        self.assertEqual(source, "resolve_cp_tp_group_sharing")
        self.assertTrue(fields["enable_cp_tp_group_sharing"])

    def test_bans_without_split_stay(self):
        # The backend ban rides on mooncake, not on the split predicate.
        cases = {
            "non-mooncake backend without split": _server_args(
                disaggregation_transfer_backend="ascend",
                enable_dsa_cache_layer_split=False,
            ),
        }
        for name, server_args in cases.items():
            with self.subTest(name):
                with self.assertRaisesRegex(ValueError, _BACKEND_BAN):
                    _run(server_args)


class TestNpuLayerSplitNotAdmitted(CustomTestCase):
    def test_gpu_still_rejects_split(self):
        with self.assertRaisesRegex(ValueError, _SPLIT_BAN):
            _run(_server_args(), is_npu=False)

    def test_other_models_still_reject_split(self):
        # Give each model its required cp_strategy so the strategy check does
        # not fire first and mask the split ban being asserted here.
        for hf, strategy in ((_KIMI31, "interleave"), (_QWEN4, "zigzag")):
            with self.subTest(hf.hf_config.architectures[0]):
                with self.assertRaisesRegex(ValueError, _SPLIT_BAN):
                    _run(_server_args(hf=hf, cp_strategy=strategy))

    def test_non_pd_prefill_still_rejects_split(self):
        # The split ban precedes the decode ban in the unsupported dict order.
        for mode in ("null", "decode"):
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(ValueError, _SPLIT_BAN):
                    _run(_server_args(disaggregation_mode=mode))

    def test_decode_without_split_rejected(self):
        with self.assertRaisesRegex(ValueError, _DECODE_BAN):
            _run(
                _server_args(
                    disaggregation_mode="decode",
                    enable_dsa_cache_layer_split=False,
                )
            )

    def test_non_mooncake_backend_still_rejects_split(self):
        # Not the target backend, so the narrow predicate is false.
        with self.assertRaisesRegex(ValueError, _SPLIT_BAN):
            _run(_server_args(disaggregation_transfer_backend="ascend"))

    def test_pp_gt_1_still_rejected_with_split(self):
        # pp > 1 is banned again regardless of the split predicate, so the
        # on/off-split A/B comparison stays symmetric.
        with self.assertRaisesRegex(ValueError, _PP_BAN):
            _run(_server_args(pp_size=4))


if __name__ == "__main__":
    unittest.main()
