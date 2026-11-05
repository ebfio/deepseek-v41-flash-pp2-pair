# patches/vllm-pp2-flag/pp_autotune.py — PP-aware FlashInfer autotune overlay
# (2026-11-05, PP-aware autotune for the two-stage pair).
#
# PROBLEM (measured, v0.31.0): under PP2, stock flashinfer_autotune() uses the
# WORLD group for broadcast/barrier, but each PP stage tunes a DIFFERENT op set
# (rank1 is the DSpark-drafter + last stage). Rank0 finishes, hits
# world.barrier() (kernel_warmup.py:480), waits for rank1's much longer tune,
# and the gloo barrier times out (~1800s default) -> EngineDeadError.
# Also only the leader saves; rank1's drafter tactics are never persisted.
#
# FIX: scope everything to the TP group (size 1 per stage here -> no cross-rank
# collectives at all), give each PP rank its own cache file, and let each rank
# save its own result. Tuning stays fully parallel per stage.
#
# Gate: VLLM_PATCH_PP_AUTOTUNE=1. Never combined with stock behavior.
#
# This file only ever executes inside the vLLM container, so its imports are
# unresolvable on the host by design. Type checkers running outside the image
# are told to skip missing-import reporting rather than flagging every line.
# pyright: reportMissingImports=false
import importlib
import logging
import os

try:
    from vllm.logger import init_logger

    logger = init_logger("vllm.pp2flag.autotune")
except Exception:  # noqa: BLE001
    logger = logging.getLogger(__name__)


def _patch():
    try:
        kw = importlib.import_module("vllm.model_executor.warmup.kernel_warmup")
        import torch
        import vllm.utils.flashinfer as fi_utils
        from flashinfer.autotuner import AutoTuner, set_autotune_process_group
        from vllm.distributed.parallel_state import get_pp_group, get_tp_group
    except Exception:
        return  # overlay must never break a boot

    def _flashinfer_autotune_pp(runner) -> None:
        tp = get_tp_group()
        pp = get_pp_group()
        tuner = AutoTuner.get()

        autotune_kwargs: dict = {}
        skip_ops = kw._flashinfer_autotune_skip_ops(runner)
        if skip_ops:
            autotune_kwargs["skip_ops"] = skip_ops

        # Per-stage, per-rank cache file: the two stages tune different ops,
        # so they must never share (or broadcast) one file.
        base = kw.resolve_flashinfer_autotune_file(runner)
        cache_path = base.with_name(f"{base.stem}.pp{pp.rank_in_group}.json")

        # Overlay v3 (2026-11-05): the cache file is keyed by the whole
        # vllm_config hash, so a hit means this exact config was tuned before
        # and the stored tactics are the ones that produced the banked numbers.
        # Loading them is enough - re-profiling would only re-select different
        # tactics (measured: 154 configs loaded, tune still re-benchmarked 7m14s
        # and the md5 moved). Skip the tune unless explicitly forced.
        cache_loaded = False
        if cache_path.exists():
            try:
                cache_loaded = tuner.load_configs(str(cache_path))
            except Exception:  # noqa: BLE001 - never break a boot
                cache_loaded = False
            if not cache_loaded:
                logger.warning(
                    "autotune cache %s present but unloadable; tuning cold",
                    cache_path,
                )

        if cache_loaded and os.environ.get(
            "VLLM_FLASHINFER_AUTOTUNE_FORCE_RETUNE"
        ) != "1":
            logger.info(
                "autotune cache loaded for this config; skipping profiling (%s)",
                cache_path,
            )
            return

        # Per-stage process group: at TP=1 this is None, so FlashInfer tunes
        # and selects tactics independently on each rank (no world averaging).
        group = tp.cpu_group if tp.world_size > 1 else None
        set_autotune_process_group(group)
        try:
            with (
                torch.inference_mode(),
                fi_utils.autotune(tune_mode=True, **autotune_kwargs),
            ):
                hisparse_enabled = (
                    runner.vllm_config.attention_config.hisparse_config is not None
                )
                if hisparse_enabled:
                    kw.autotune_hisparse_flashinfer_attention(runner)
                kw._run_flashinfer_autotune_dummy_runs(
                    runner, skip_attn=hisparse_enabled
                )
                kw.replayssm_autotune_warmup(runner)
                kw._autotune_kimi_k3_kda_qkvg(runner.get_model())
            with torch.inference_mode():
                kw._run_flashinfer_bf16_autotune_dummy_run(
                    runner, skip_ops=skip_ops, skip_attn=hisparse_enabled
                )
        finally:
            set_autotune_process_group(None)

        # No world barrier: each stage saves and proceeds independently.
        tuner.save_configs(str(cache_path))

    kw.__dict__["flashinfer_autotune"] = _flashinfer_autotune_pp


_patch()
