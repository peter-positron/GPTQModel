"""Paibaker capability surface for fork-patch coordination.

Each capability is derived from a sentinel constant defined at the
implementation site (e.g., CROSS_SHARD_DEQUANT_SUPPORTED = True in
model_dequant.py).  get_caps() collects them via getattr — you cannot
forget to flip a boolean that lives next to the code it describes.

Paibaker patches read these capabilities to decide whether to apply
(fallback mode) or skip (fork already handles it).
"""

from __future__ import annotations

FORK_VERSION = "paibaker-2026.02.25+7048b470"


def get_caps() -> dict[str, bool]:
    """Return capability flags derived from implementation sentinels.

    Each key maps to a specific behavior.  True means the fork handles
    it natively; False means a paibaker monkey-patch is still needed.
    """
    caps: dict[str, bool] = {}

    # -- Cross-shard dequant (Phase 1) --
    # Sentinel: model_dequant.CROSS_SHARD_DEQUANT_SUPPORTED
    try:
        from gptqmodel.utils import model_dequant

        caps["cross_shard_dequant"] = getattr(
            model_dequant, "CROSS_SHARD_DEQUANT_SUPPORTED", False
        )
    except ImportError:
        caps["cross_shard_dequant"] = False

    # -- V1-to-V2 qzeros correction --
    # Sentinel: model_dequant.V1_TO_V2_QZEROS_SUPPORTED
    try:
        from gptqmodel.utils import model_dequant

        caps["convert_gptq_v1_to_v2"] = getattr(
            model_dequant, "V1_TO_V2_QZEROS_SUPPORTED", False
        )
    except ImportError:
        caps["convert_gptq_v1_to_v2"] = False

    # -- GPT-OSS corruption sync on save (Phase 2) --
    # Sentinel: gpt_oss.GPT_OSS_CORRUPTION_SYNC_SUPPORTED
    try:
        from gptqmodel.models.definitions import gpt_oss

        caps["gpt_oss_corruption_sync"] = getattr(
            gpt_oss, "GPT_OSS_CORRUPTION_SYNC_SUPPORTED", False
        )
    except ImportError:
        caps["gpt_oss_corruption_sync"] = False

    # -- GPT-OSS meta device init handling (Phase 2) --
    # Sentinel: gpt_oss.GPT_OSS_META_DEVICE_INIT_SUPPORTED
    try:
        from gptqmodel.models.definitions import gpt_oss

        caps["gpt_oss_meta_device_init"] = getattr(
            gpt_oss, "GPT_OSS_META_DEVICE_INIT_SUPPORTED", False
        )
    except ImportError:
        caps["gpt_oss_meta_device_init"] = False

    return caps
