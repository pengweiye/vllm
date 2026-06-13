# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention layer with FlashAttention."""
from dataclasses import dataclass
import os
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from vllm import _custom_ops as ops
from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionMetadata, AttentionType,
                                              is_quantized_kv_cache)
from vllm.attention.layer import Attention
from vllm.attention.ops.merge_attn_states import merge_attn_states
from vllm.attention.utils.fa_utils import (flash_attn_supports_fp8,
                                           get_flash_attn_version,
                                           is_flash_attn_varlen_func_available)

if is_flash_attn_varlen_func_available():
    from vllm.attention.utils.fa_utils import (flash_attn_varlen_func,
                                               get_scheduler_metadata,
                                               reshape_and_cache_flash)

from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.logger import init_logger
from vllm.utils import cdiv
from vllm.v1.attention.backends.utils import (AttentionCGSupport,
                                              AttentionMetadataBuilder,
                                              CommonAttentionMetadata,
                                              get_kv_cache_layout)
from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)

# NOTE(woosuk): This is an arbitrary number. Tune it if needed.
_DEFAULT_MAX_NUM_SPLITS_FOR_CUDA_GRAPH = 16


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning_once("Ignoring invalid integer value for %s: %s", name,
                            value)
        return default


def _env_int_set(name: str,
                 default: Optional[set[int]] = None) -> Optional[set[int]]:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    if value in ("*", "all", "ALL"):
        return None
    try:
        return {int(part.strip()) for part in value.split(",") if part.strip()}
    except ValueError:
        logger.warning_once("Ignoring invalid layer list for %s: %s", name,
                            value)
        return default


class FlashAttentionBackend(AttentionBackend):

    accept_output_buffer: bool = True

    @classmethod
    def get_supported_dtypes(cls) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [32, 64, 96, 128, 160, 192, 224, 256]

    @classmethod
    def validate_head_size(cls, head_size: int) -> None:
        supported_head_sizes = cls.get_supported_head_sizes()
        if head_size not in supported_head_sizes:
            attn_type = cls.__name__.removesuffix("Backend")
            raise ValueError(
                f"Head size {head_size} is not supported by {attn_type}. "
                f"Supported head sizes are: {supported_head_sizes}. "
                "Set VLLM_ATTENTION_BACKEND=FLEX_ATTENTION to use "
                "FlexAttention backend which supports all head sizes.")

    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN_VLLM_V1"

    @staticmethod
    def get_impl_cls() -> type["FlashAttentionImpl"]:
        return FlashAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type["AttentionMetadata"]:
        return FlashAttentionMetadata

    @staticmethod
    def get_builder_cls() -> type["FlashAttentionMetadataBuilder"]:
        return FlashAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order() -> tuple[int, ...]:
        # `stride_order` indicates the permutation that gets
        # us from `get_kv_cache_shape` to the actual memory layout we want.
        cache_layout = get_kv_cache_layout()
        if cache_layout == "NHD":
            stride_order = (0, 1, 2, 3, 4)
        elif cache_layout == "HND":
            stride_order = (0, 1, 3, 2, 4)
        else:
            raise ValueError(f"Unknown cache layout format {cache_layout}.")
        return stride_order

    @staticmethod
    def get_fp8_dtype_for_flashattn(kv_cache_dtype: str) -> torch.dtype:
        if kv_cache_dtype in ("fp8", "fp8_e4m3"):
            return torch.float8_e4m3fn
        else:
            raise ValueError(f"Unrecognized FP8 dtype: {kv_cache_dtype}")


@dataclass
class FlashAttentionMetadata:
    # NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ---------------------|
    #                                   |-- query_len ---|

    num_actual_tokens: int  # Number of tokens excluding padding.
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor

    # For cascade attention.
    use_cascade: bool
    common_prefix_len: int
    cu_prefix_query_lens: Optional[torch.Tensor]
    prefix_kv_lens: Optional[torch.Tensor]
    suffix_kv_lens: Optional[torch.Tensor]

    # Optional aot scheduling
    scheduler_metadata: Optional[torch.Tensor] = None
    prefix_scheduler_metadata: Optional[torch.Tensor] = None
    max_num_splits: int = 0

    causal: bool = True


def _get_sliding_window_configs(
        vllm_config: VllmConfig) -> set[Optional[tuple[int, int]]]:
    """Get the set of all sliding window configs used in the model."""
    sliding_window_configs: set[Optional[tuple[int, int]]] = set()
    layers = get_layers_from_vllm_config(vllm_config, Attention)
    for layer in layers.values():
        assert isinstance(layer.impl, FlashAttentionImpl)
        sliding_window_configs.add(layer.impl.sliding_window)
    return sliding_window_configs


class FlashAttentionMetadataBuilder(
        AttentionMetadataBuilder[FlashAttentionMetadata]):
    # FA3:
    # Supports full cudagraphs for all cases.
    #
    # FA2:
    # For FA2, a graph is captured with max_query_len=1, (which is what we
    # capture by default for num_tokens <= max_num_seqs when there is no
    # spec-decode) then these graphs will not work for mixed prefill-decode
    # (unlike FA3). This is due to special max_query_len=1 packed-GQA handling
    # in FA2.
    # In summary if we are running with spec decodes the graphs would
    # work for mixed prefill-decode and uniform-decode. But for non-spec decodes
    # the graphs would not work for mixed prefill-decode; sorta the inverse
    # of UNIFORM_SINGLE_TOKEN_DECODE.
    # There's probably a better way to describe this using `AttentionCGSupport`
    # but for now just set it to `UNIFORM_BATCH` to get use to drop down
    # to FULL_AND_PIECEWISE.
    # TODO(luka, lucas): audit FA2 as part of:
    #  https://github.com/vllm-project/vllm/issues/22945
    cudagraph_support = AttentionCGSupport.ALWAYS \
        if get_flash_attn_version() == 3 else AttentionCGSupport.UNIFORM_BATCH

    def __init__(self, kv_cache_spec: AttentionSpec, layer_names: list[str],
                 vllm_config: VllmConfig, device: torch.device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.model_config = vllm_config.model_config
        self.parallel_config = vllm_config.parallel_config
        self.cache_config = vllm_config.cache_config
        self.compilation_config = vllm_config.compilation_config

        self.num_heads_q = self.model_config.get_num_attention_heads(
            self.parallel_config)
        self.num_heads_kv = self.model_config.get_num_kv_heads(
            self.parallel_config)
        self.kv_cache_dtype = kv_cache_spec.dtype
        self.headdim = self.model_config.get_head_size()
        self.block_size = kv_cache_spec.block_size

        self.max_num_splits = 0  # No upper bound on the number of splits.
        self.aot_schedule = (get_flash_attn_version() == 3)

        self.use_full_cuda_graph = \
            self.compilation_config.cudagraph_mode.has_full_cudagraphs()

        if self.use_full_cuda_graph and self.aot_schedule:
            self.max_cudagraph_size = self.compilation_config.max_capture_size

            if self.max_cudagraph_size > 992:
                # This condition derives from FA3's internal heuristic.
                # TODO(woosuk): Support larger cudagraph sizes.
                raise ValueError(
                    "Capture size larger than 992 is not supported for "
                    "full cuda graph.")

            self.scheduler_metadata = torch.zeros(
                vllm_config.scheduler_config.max_num_seqs + 1,
                dtype=torch.int32,
                device=self.device,
            )
            # When using cuda graph, we need to set the upper bound of the
            # number of splits so that large enough intermediate buffers are
            # pre-allocated during capture.
            self.max_num_splits = _DEFAULT_MAX_NUM_SPLITS_FOR_CUDA_GRAPH

        # Sliding window size to be used with the AOT scheduler will be
        # populated on first build() call.
        self.aot_sliding_window: Optional[tuple[int, int]] = None

    def build(self,
              common_prefix_len: int,
              common_attn_metadata: CommonAttentionMetadata,
              fast_build: bool = False) -> FlashAttentionMetadata:
        """
        fast_build disables AOT scheduling, used when there will be few
        iterations i.e. spec-decode
        """
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        max_query_len = common_attn_metadata.max_query_len
        max_seq_len = common_attn_metadata.max_seq_len
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens
        seq_lens_cpu = common_attn_metadata.seq_lens_cpu
        block_table_tensor = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping
        causal = common_attn_metadata.causal

        # the overhead of the aot schedule is not worth it for spec-decode
        aot_schedule = self.aot_schedule and not fast_build

        if self.aot_sliding_window is None:
            self.aot_sliding_window = (-1, -1)
            # For the AOT scheduler we need the sliding window value to be
            # constant for all layers to. We have to populate this on the first
            # build() call so the layers are constructed (cannot populate)
            # in __init__.
            if aot_schedule:
                sliding_window_configs = _get_sliding_window_configs(
                    self.vllm_config)
                if len(sliding_window_configs) == 1:
                    sliding_window_config = sliding_window_configs.pop()
                    if sliding_window_config is not None:
                        self.aot_sliding_window = sliding_window_config
                elif len(sliding_window_configs) > 1:
                    self.aot_schedule = False
                    aot_schedule = False

        def schedule(batch_size, cu_query_lens, max_query_len, seqlens,
                     max_seq_len, causal):
            cache_dtype = self.cache_config.cache_dtype
            if cache_dtype.startswith("fp8"):
                qkv_dtype = FlashAttentionBackend.get_fp8_dtype_for_flashattn(
                    cache_dtype)
            else:
                qkv_dtype = self.kv_cache_dtype
            if aot_schedule:
                return get_scheduler_metadata(
                    batch_size=batch_size,
                    max_seqlen_q=max_query_len,
                    max_seqlen_k=max_seq_len,
                    num_heads_q=self.num_heads_q,
                    num_heads_kv=self.num_heads_kv,
                    headdim=self.headdim,
                    cache_seqlens=seqlens,
                    qkv_dtype=qkv_dtype,
                    cu_seqlens_q=cu_query_lens,
                    page_size=self.block_size,
                    causal=causal,
                    window_size=self.aot_sliding_window,
                    num_splits=self.max_num_splits,
                )
            return None

        use_cascade = common_prefix_len > 0

        if use_cascade:
            cu_prefix_query_lens = torch.tensor([0, num_actual_tokens],
                                                dtype=torch.int32,
                                                device=self.device)
            prefix_kv_lens = torch.tensor([common_prefix_len],
                                          dtype=torch.int32,
                                          device=self.device)
            suffix_kv_lens = (seq_lens_cpu[:num_reqs] - common_prefix_len).to(
                self.device, non_blocking=True)
            prefix_scheduler_metadata = schedule(
                batch_size=1,
                cu_query_lens=cu_prefix_query_lens,
                max_query_len=num_actual_tokens,
                seqlens=prefix_kv_lens,
                max_seq_len=common_prefix_len,
                causal=False)
            scheduler_metadata = schedule(batch_size=num_reqs,
                                          cu_query_lens=query_start_loc,
                                          max_query_len=max_query_len,
                                          seqlens=suffix_kv_lens,
                                          max_seq_len=max_seq_len -
                                          common_prefix_len,
                                          causal=True)
        else:
            cu_prefix_query_lens = None
            prefix_kv_lens = None
            suffix_kv_lens = None
            prefix_scheduler_metadata = None
            scheduler_metadata = schedule(batch_size=num_reqs,
                                          cu_query_lens=query_start_loc,
                                          max_query_len=max_query_len,
                                          seqlens=seq_lens,
                                          max_seq_len=max_seq_len,
                                          causal=causal)
        # For FA3 + full cudagraph
        max_num_splits = 0
        if self.use_full_cuda_graph and scheduler_metadata is not None:
            n = scheduler_metadata.shape[0]
            self.scheduler_metadata[:n] = scheduler_metadata
            # NOTE(woosuk): We should zero out the rest of the scheduler
            # metadata to guarantee the correctness. Otherwise, some thread
            # blocks may use the invalid scheduler metadata and overwrite the
            # output buffer.
            self.scheduler_metadata[n:] = 0
            scheduler_metadata = self.scheduler_metadata[:n]

            if num_actual_tokens <= self.max_cudagraph_size:
                # NOTE(woosuk): Setting num_splits > 1 may increase the memory
                # usage, because the intermediate buffers of size [num_splits,
                # num_heads, num_tokens, head_size] are allocated. Therefore,
                # we only set num_splits when using cuda graphs.
                max_num_splits = self.max_num_splits

        attn_metadata = FlashAttentionMetadata(
            num_actual_tokens=num_actual_tokens,
            max_query_len=max_query_len,
            query_start_loc=query_start_loc,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            block_table=block_table_tensor,
            slot_mapping=slot_mapping,
            use_cascade=use_cascade,
            common_prefix_len=common_prefix_len,
            scheduler_metadata=scheduler_metadata,
            cu_prefix_query_lens=cu_prefix_query_lens,
            prefix_kv_lens=prefix_kv_lens,
            suffix_kv_lens=suffix_kv_lens,
            prefix_scheduler_metadata=prefix_scheduler_metadata,
            max_num_splits=max_num_splits,
            causal=causal)
        return attn_metadata

    def use_cascade_attention(self, *args, **kwargs) -> bool:
        return use_cascade_attention(*args, **kwargs)


class FlashAttentionImpl(AttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[list[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        logits_soft_cap: Optional[float] = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: Optional[str] = None,
        sinks: Optional[torch.Tensor] = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        elif attn_type == AttentionType.ENCODER_ONLY:
            self.sliding_window = (sliding_window - 1, sliding_window - 1)
        else:
            self.sliding_window = (sliding_window - 1, 0)
        self.kv_cache_dtype = kv_cache_dtype
        if logits_soft_cap is None:
            # In flash-attn, setting logits_soft_cap as 0 means no soft cap.
            logits_soft_cap = 0
        self.logits_soft_cap = logits_soft_cap
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        FlashAttentionBackend.validate_head_size(head_size)

        self.attn_type = attn_type
        self.vllm_flash_attn_version = get_flash_attn_version()
        if is_quantized_kv_cache(self.kv_cache_dtype) \
            and not flash_attn_supports_fp8():
            raise NotImplementedError(
                "FlashAttention does not support fp8 kv-cache on this device.")

        self.sinks = sinks
        if self.sinks is not None:
            assert self.vllm_flash_attn_version == 3, (
                "Sinks are only supported in FlashAttention 3")
            assert self.sinks.shape[0] == num_heads, (
                "Sinks must have the same number of heads as the number of "
                "heads in the layer")

        # Experimental SnapKV prototype. This intentionally does not change the
        # physical paged KV cache layout. It keeps a side compact dense cache
        # per layer, matching the original SnapKV execution shape more closely:
        # compress once after prefill, then append generated K/V to that compact
        # cache during decode.
        self.snapkv_enabled = _env_flag("VLLM_SNAPKV")
        self.snapkv_window_size = max(1, _env_int("VLLM_SNAPKV_WINDOW_SIZE",
                                                 32))
        self.snapkv_max_capacity = max(
            self.snapkv_window_size,
            _env_int("VLLM_SNAPKV_MAX_CAPACITY", 512))
        self.snapkv_kernel_size = max(1, _env_int("VLLM_SNAPKV_KERNEL_SIZE",
                                                 7))
        if self.snapkv_kernel_size % 2 == 0:
            self.snapkv_kernel_size += 1
        self.snapkv_append_capacity = max(
            1, _env_int("VLLM_SNAPKV_APPEND_CAPACITY", 1024))
        self.snapkv_debug = _env_flag("VLLM_SNAPKV_DEBUG")
        self.snapkv_debug_layers = _env_int_set("VLLM_SNAPKV_DEBUG_LAYERS",
                                                {0})
        self.snapkv_debug_decode_steps = max(
            0, _env_int("VLLM_SNAPKV_DEBUG_DECODE_STEPS", 3))
        self._snapkv_prompt_len = 0
        self._snapkv_selected_history: Optional[torch.Tensor] = None
        self._snapkv_compact_key_cache: Optional[torch.Tensor] = None
        self._snapkv_compact_value_cache: Optional[torch.Tensor] = None
        self._snapkv_compact_kv_len = 0
        self._snapkv_warned_unsupported = False
        self._snapkv_decode_debug_count = 0
        self._snapkv_fallback_reasons_logged: set[str] = set()

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: Optional[torch.Tensor] = None,
        output_scale: Optional[torch.Tensor] = None,
        output_block_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with FlashAttention.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape =
                [2, num_blocks, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        NOTE: FP8 quantization, flash-attn expect the size of
              {q,k,v}_descale to be (num_sequences, num_kv_heads).
              We use torch's .expand() to avoid duplicating values
        """
        assert output is not None, "Output tensor must be provided."

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported"
                " for FlashAttentionImpl")

        if attn_metadata is None:
            # Profiling run.
            return output

        attn_type = self.attn_type

        # IMPORTANT!
        # NOTE(woosuk): With piece-wise CUDA graphs, this method is executed in
        # eager-mode PyTorch. Thus, we need to be careful about any CPU overhead
        # in this method. For example, `view` and `slice` (or `[:n]`) operations
        # are surprisingly slow even in the case they do not invoke any GPU ops.
        # Minimize the PyTorch ops in this method as much as possible.
        # Whenever making a change in this method, please benchmark the
        # performance to make sure it does not introduce any overhead.

        num_actual_tokens = attn_metadata.num_actual_tokens

        # Handle encoder attention differently - no KV cache needed
        if attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            # For encoder attention,
            # we use direct Q, K, V tensors without caching
            return self._forward_encoder_attention(query[:num_actual_tokens],
                                                   key[:num_actual_tokens],
                                                   value[:num_actual_tokens],
                                                   output[:num_actual_tokens],
                                                   attn_metadata, layer)

        # For decoder and cross-attention, use KV cache as before
        key_cache, value_cache = kv_cache.unbind(0)

        # key and value may be None in the case of cross attention. They are
        # calculated once based on the output from the encoder and then cached
        # in KV cache.
        if (self.kv_sharing_target_layer_name is None and key is not None
                and value is not None):
            # Reshape the input keys and values and store them in the cache.
            # Skip this if sharing KV cache with an earlier attention layer.
            # NOTE(woosuk): Here, key and value are padded while slot_mapping is
            # not padded. However, we don't need to do key[:num_actual_tokens]
            # and value[:num_actual_tokens] because the reshape_and_cache_flash
            # op uses the slot_mapping's shape to determine the number of
            # actual tokens.
            reshape_and_cache_flash(
                key,
                value,
                key_cache,
                value_cache,
                attn_metadata.slot_mapping,
                self.kv_cache_dtype,
                layer._k_scale,
                layer._v_scale,
            )

        if self.kv_cache_dtype.startswith("fp8"):
            dtype = FlashAttentionBackend.get_fp8_dtype_for_flashattn(
                self.kv_cache_dtype)
            key_cache = key_cache.view(dtype)
            value_cache = value_cache.view(dtype)
            num_tokens, num_heads, head_size = query.shape
            query, _ = ops.scaled_fp8_quant(
                query.reshape(
                    (num_tokens, num_heads * head_size)).contiguous(),
                layer._q_scale)
            query = query.reshape((num_tokens, num_heads, head_size))

        if self.snapkv_enabled:
            unsupported_reason = self._snapkv_unsupported_reason(
                attn_metadata)
            if unsupported_reason is not None:
                self._snapkv_log_fallback(layer, unsupported_reason)
            elif self._snapkv_is_full_prefill(query, key, value,
                                              attn_metadata):
                self._snapkv_update_prefill_state(layer,
                                                  query[:num_actual_tokens],
                                                  key[:num_actual_tokens],
                                                  value[:num_actual_tokens])
            elif self._snapkv_can_run_decode(query, attn_metadata):
                if (key is not None and value is not None
                        and self._snapkv_forward_decode(
                            query, key[:num_actual_tokens],
                            value[:num_actual_tokens], attn_metadata, output,
                            layer)):
                    return output
            else:
                reason = self._snapkv_inactive_reason(query, key, value,
                                                      attn_metadata)
                if reason is not None:
                    self._snapkv_log_fallback(layer, reason)

        if not attn_metadata.use_cascade:
            cu_seqlens_q = attn_metadata.query_start_loc
            seqused_k = attn_metadata.seq_lens
            max_seqlen_q = attn_metadata.max_query_len
            max_seqlen_k = attn_metadata.max_seq_len
            block_table = attn_metadata.block_table
            scheduler_metadata = attn_metadata.scheduler_metadata

            descale_shape = (cu_seqlens_q.shape[0] - 1, self.num_kv_heads)

            flash_attn_varlen_func(
                q=query[:num_actual_tokens],
                k=key_cache,
                v=value_cache,
                out=output[:num_actual_tokens],
                cu_seqlens_q=cu_seqlens_q,
                max_seqlen_q=max_seqlen_q,
                seqused_k=seqused_k,
                max_seqlen_k=max_seqlen_k,
                softmax_scale=self.scale,
                causal=attn_metadata.causal,
                alibi_slopes=self.alibi_slopes,
                window_size=self.sliding_window,
                block_table=block_table,
                softcap=self.logits_soft_cap,
                scheduler_metadata=scheduler_metadata,
                fa_version=self.vllm_flash_attn_version,
                q_descale=layer._q_scale.expand(descale_shape),
                k_descale=layer._k_scale.expand(descale_shape),
                v_descale=layer._v_scale.expand(descale_shape),
                num_splits=attn_metadata.max_num_splits,
                s_aux=self.sinks,
            )
            return output

        # Cascade attention (rare case).
        cascade_attention(
            output[:num_actual_tokens],
            query[:num_actual_tokens],
            key_cache,
            value_cache,
            cu_query_lens=attn_metadata.query_start_loc,
            max_query_len=attn_metadata.max_query_len,
            cu_prefix_query_lens=attn_metadata.cu_prefix_query_lens,
            prefix_kv_lens=attn_metadata.prefix_kv_lens,
            suffix_kv_lens=attn_metadata.suffix_kv_lens,
            max_kv_len=attn_metadata.max_seq_len,
            softmax_scale=self.scale,
            alibi_slopes=self.alibi_slopes,
            sliding_window=self.sliding_window,
            logits_soft_cap=self.logits_soft_cap,
            block_table=attn_metadata.block_table,
            common_prefix_len=attn_metadata.common_prefix_len,
            fa_version=self.vllm_flash_attn_version,
            prefix_scheduler_metadata=attn_metadata.prefix_scheduler_metadata,
            suffix_scheduler_metadata=attn_metadata.scheduler_metadata,
            q_descale=layer._q_scale,
            k_descale=layer._k_scale,
            v_descale=layer._v_scale,
        )
        return output

    def _snapkv_unsupported_reason(
        self,
        attn_metadata: FlashAttentionMetadata,
    ) -> Optional[str]:
        if self.attn_type != AttentionType.DECODER:
            return f"unsupported attention type: {self.attn_type}"
        if self.kv_cache_dtype.startswith("fp8"):
            if not self._snapkv_warned_unsupported:
                logger.warning_once("SnapKV prototype does not support fp8 "
                                    "KV cache; falling back to vLLM attention.")
                self._snapkv_warned_unsupported = True
            return "fp8 KV cache is unsupported"
        if attn_metadata.use_cascade:
            return "cascade or prefix attention metadata is unsupported"
        if attn_metadata.block_table.shape[0] != 1:
            return (f"batch size is {attn_metadata.block_table.shape[0]}, "
                    "expected 1")
        if attn_metadata.query_start_loc.shape[0] != 2:
            return "query_start_loc does not describe a single request"
        if self.kv_sharing_target_layer_name is not None:
            return "KV sharing target layers are unsupported"
        if self.alibi_slopes is not None:
            return "ALiBi attention is unsupported"
        if self.sinks is not None:
            return "attention sinks are unsupported"
        return None

    def _snapkv_inactive_reason(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor],
        value: Optional[torch.Tensor],
        attn_metadata: FlashAttentionMetadata,
    ) -> Optional[str]:
        if query.shape[0] == 1:
            if self._snapkv_selected_history is None:
                return "decode skipped because no prefill state exists"
            if (self._snapkv_compact_key_cache is None
                    or self._snapkv_compact_value_cache is None):
                return "decode skipped because compact KV cache is missing"
            seq_len = int(attn_metadata.seq_lens[0].item())
            if self._snapkv_prompt_len <= 0:
                return "decode skipped because prompt_len is not initialized"
            if seq_len <= self._snapkv_prompt_len:
                return "decode skipped because seq_len has not passed prompt_len"
            return None
        if key is None or value is None:
            return "prefill skipped because key/value are missing"
        seq_len = int(attn_metadata.seq_lens[0].item())
        if seq_len != query.shape[0]:
            return (f"prefill skipped because chunked prefill is active "
                    f"(query_len={query.shape[0]}, seq_len={seq_len})")
        return None

    def _snapkv_layer_name(self, layer: torch.nn.Module) -> str:
        return str(getattr(layer, "layer_name", "<unknown>"))

    def _snapkv_layer_index(self, layer: torch.nn.Module) -> Optional[int]:
        ints: list[int] = []
        for part in self._snapkv_layer_name(layer).split("."):
            try:
                ints.append(int(part))
            except ValueError:
                continue
        if len(ints) != 1:
            return None
        return ints[0]

    def _snapkv_should_log_layer(self, layer: torch.nn.Module) -> bool:
        if not self.snapkv_debug:
            return False
        if self.snapkv_debug_layers is None:
            return True
        layer_idx = self._snapkv_layer_index(layer)
        return layer_idx in self.snapkv_debug_layers

    def _snapkv_log_fallback(self, layer: torch.nn.Module,
                             reason: str) -> None:
        if not self._snapkv_should_log_layer(layer):
            return
        layer_name = self._snapkv_layer_name(layer)
        key = f"{layer_name}:{reason}"
        if key in self._snapkv_fallback_reasons_logged:
            return
        self._snapkv_fallback_reasons_logged.add(key)
        logger.info("SnapKV fallback layer=%s reason=%s", layer_name, reason)

    def _snapkv_log_prefill(self, layer: torch.nn.Module, prompt_len: int,
                            obs_len: int, history_len: int,
                            selected_len: int) -> None:
        if not self._snapkv_should_log_layer(layer):
            return
        compression_ratio = ((selected_len + obs_len) / prompt_len
                             if prompt_len > 0 else 0.0)
        logger.info(
            "SnapKV prefill layer=%s prompt_len=%d obs_len=%d "
            "history_len=%d selected_len=%d max_capacity=%d "
            "compression_ratio=%.3f",
            self._snapkv_layer_name(layer),
            prompt_len,
            obs_len,
            history_len,
            selected_len,
            self.snapkv_max_capacity,
            compression_ratio,
        )

    def _snapkv_log_decode(self, layer: torch.nn.Module, seq_len: int,
                           prompt_len: int, selected_len: int,
                           recent_len: int, generated_len: int,
                           kv_len: int) -> None:
        self._snapkv_decode_debug_count += 1
        if not self._snapkv_should_log_layer(layer):
            return
        if self._snapkv_decode_debug_count > self.snapkv_debug_decode_steps:
            return
        logger.info(
            "SnapKV decode layer=%s step=%d seq_len=%d prompt_len=%d "
            "selected_len=%d recent_len=%d generated_len=%d kv_len=%d "
            "used=snapkv_compact_cache_decode",
            self._snapkv_layer_name(layer),
            self._snapkv_decode_debug_count,
            seq_len,
            prompt_len,
            selected_len,
            recent_len,
            generated_len,
            kv_len,
        )

    def _snapkv_is_full_prefill(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor],
        value: Optional[torch.Tensor],
        attn_metadata: FlashAttentionMetadata,
    ) -> bool:
        if key is None or value is None:
            return False
        if query.shape[0] <= 1:
            return False
        # With chunked prefill disabled, the first request step has computed
        # exactly the prompt tokens contained in this forward pass.
        return int(attn_metadata.seq_lens[0].item()) == query.shape[0]

    def _snapkv_can_run_decode(
        self,
        query: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
    ) -> bool:
        if query.shape[0] != 1:
            return False
        if self._snapkv_selected_history is None:
            return False
        if (self._snapkv_compact_key_cache is None
                or self._snapkv_compact_value_cache is None):
            return False
        seq_len = int(attn_metadata.seq_lens[0].item())
        return self._snapkv_prompt_len > 0 and seq_len > self._snapkv_prompt_len

    def _snapkv_update_prefill_state(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        prompt_len = query.shape[0]
        obs_len = min(self.snapkv_window_size, prompt_len)
        history_len = prompt_len - obs_len
        device = query.device

        if history_len <= 0:
            selected = torch.empty((self.num_heads, 0),
                                   dtype=torch.long,
                                   device=device)
            self._snapkv_prompt_len = prompt_len
            self._snapkv_selected_history = selected
            self._snapkv_init_compact_cache(key, value, selected, obs_len)
            self._snapkv_decode_debug_count = 0
            self._snapkv_log_prefill(layer, prompt_len, obs_len, history_len,
                                     selected.shape[1])
            self._snapkv_log_fallback(
                layer, "prompt too short; recent window covers full prompt")
            return

        budget = min(max(self.snapkv_max_capacity - obs_len, 0), history_len)
        if budget == history_len:
            selected = torch.arange(history_len,
                                    dtype=torch.long,
                                    device=device).expand(
                                        self.num_heads, -1).contiguous()
            self._snapkv_prompt_len = prompt_len
            self._snapkv_selected_history = selected
            self._snapkv_init_compact_cache(key, value, selected, obs_len)
            self._snapkv_decode_debug_count = 0
            self._snapkv_log_prefill(layer, prompt_len, obs_len, history_len,
                                     selected.shape[1])
            self._snapkv_log_fallback(
                layer, "prompt shorter than SnapKV capacity; no compression")
            return

        if budget == 0:
            selected = torch.empty((self.num_heads, 0),
                                   dtype=torch.long,
                                   device=device)
            self._snapkv_prompt_len = prompt_len
            self._snapkv_selected_history = selected
            self._snapkv_init_compact_cache(key, value, selected, obs_len)
            self._snapkv_decode_debug_count = 0
            self._snapkv_log_prefill(layer, prompt_len, obs_len, history_len,
                                     selected.shape[1])
            return

        # SnapKV scores historical tokens by the attention mass from the
        # observation window at the prompt tail. K is expanded to query heads so
        # each query head gets its own top-k history.
        q_obs = query[-obs_len:].transpose(0, 1).float()
        k_rep = key.repeat_interleave(self.num_queries_per_kv,
                                      dim=1).transpose(0, 1).float()
        scores = torch.matmul(q_obs, k_rep.transpose(-2, -1)) * self.scale

        q_pos = torch.arange(prompt_len - obs_len,
                             prompt_len,
                             device=device).unsqueeze(1)
        k_pos = torch.arange(prompt_len, device=device).unsqueeze(0)
        causal_mask = k_pos <= q_pos
        scores = scores.masked_fill(~causal_mask.unsqueeze(0),
                                    torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores, dim=-1, dtype=torch.float32)
        history_scores = attn[:, :, :history_len].sum(dim=1)

        kernel_size = min(self.snapkv_kernel_size,
                          history_scores.shape[-1] |
                          1)  # Keep the pooling kernel odd.
        pooled = F.avg_pool1d(history_scores.unsqueeze(1),
                              kernel_size=kernel_size,
                              padding=kernel_size // 2,
                              stride=1).squeeze(1)
        pooled = pooled[:, :history_len]
        selected = pooled.topk(budget, dim=-1).indices
        selected = selected.sort(dim=-1).values.to(torch.long)

        self._snapkv_prompt_len = prompt_len
        self._snapkv_selected_history = selected
        self._snapkv_init_compact_cache(key, value, selected, obs_len)
        self._snapkv_decode_debug_count = 0
        self._snapkv_log_prefill(layer, prompt_len, obs_len, history_len,
                                 selected.shape[1])

    def _snapkv_init_compact_cache(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        selected_history: torch.Tensor,
        obs_len: int,
    ) -> None:
        prompt_len = key.shape[0]
        selected_len = selected_history.shape[1]
        recent_start = max(0, prompt_len - obs_len)

        key_by_head = key.repeat_interleave(self.num_queries_per_kv,
                                            dim=1).transpose(0, 1)
        value_by_head = value.repeat_interleave(self.num_queries_per_kv,
                                                dim=1).transpose(0, 1)

        key_parts = []
        value_parts = []
        if selected_len > 0:
            gather_index = selected_history.unsqueeze(-1).expand(
                -1, -1, self.head_size)
            key_parts.append(key_by_head.gather(1, gather_index))
            value_parts.append(value_by_head.gather(1, gather_index))
        if obs_len > 0:
            key_parts.append(key_by_head[:, recent_start:prompt_len])
            value_parts.append(value_by_head[:, recent_start:prompt_len])

        compact_key = torch.cat(key_parts, dim=1).transpose(0, 1).contiguous()
        compact_value = torch.cat(value_parts,
                                  dim=1).transpose(0, 1).contiguous()
        compact_len = compact_key.shape[0]
        cache_capacity = compact_len + self.snapkv_append_capacity

        key_cache = torch.empty((cache_capacity, self.num_heads,
                                 self.head_size),
                                dtype=compact_key.dtype,
                                device=compact_key.device)
        value_cache = torch.empty_like(key_cache)
        key_cache[:compact_len].copy_(compact_key)
        value_cache[:compact_len].copy_(compact_value)

        self._snapkv_compact_key_cache = key_cache
        self._snapkv_compact_value_cache = value_cache
        self._snapkv_compact_kv_len = compact_len

    def _snapkv_grow_compact_cache(self) -> None:
        key_cache = self._snapkv_compact_key_cache
        value_cache = self._snapkv_compact_value_cache
        if key_cache is None or value_cache is None:
            return

        old_capacity = key_cache.shape[0]
        new_capacity = max(old_capacity * 2, old_capacity + 1)
        new_key_cache = torch.empty((new_capacity, self.num_heads,
                                     self.head_size),
                                    dtype=key_cache.dtype,
                                    device=key_cache.device)
        new_value_cache = torch.empty_like(new_key_cache)
        new_key_cache[:self._snapkv_compact_kv_len].copy_(
            key_cache[:self._snapkv_compact_kv_len])
        new_value_cache[:self._snapkv_compact_kv_len].copy_(
            value_cache[:self._snapkv_compact_kv_len])
        self._snapkv_compact_key_cache = new_key_cache
        self._snapkv_compact_value_cache = new_value_cache

    def _snapkv_append_current_kv(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> bool:
        key_cache = self._snapkv_compact_key_cache
        value_cache = self._snapkv_compact_value_cache
        if key_cache is None or value_cache is None:
            return False
        if key.shape[0] != 1 or value.shape[0] != 1:
            return False

        if self._snapkv_compact_kv_len >= key_cache.shape[0]:
            self._snapkv_grow_compact_cache()
            key_cache = self._snapkv_compact_key_cache
            value_cache = self._snapkv_compact_value_cache
            if key_cache is None or value_cache is None:
                return False

        current_key = key.repeat_interleave(self.num_queries_per_kv, dim=1)
        current_value = value.repeat_interleave(self.num_queries_per_kv, dim=1)
        key_cache[self._snapkv_compact_kv_len].copy_(current_key[0])
        value_cache[self._snapkv_compact_kv_len].copy_(current_value[0])
        self._snapkv_compact_kv_len += 1
        return True

    def _snapkv_forward_decode(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor,
        layer: torch.nn.Module,
    ) -> bool:
        selected_history = self._snapkv_selected_history
        if selected_history is None:
            return False

        prompt_len = self._snapkv_prompt_len
        seq_len = int(attn_metadata.seq_lens[0].item())
        if seq_len <= prompt_len:
            return False

        if not self._snapkv_append_current_kv(key, value):
            return False

        key_cache = self._snapkv_compact_key_cache
        value_cache = self._snapkv_compact_value_cache
        if key_cache is None or value_cache is None:
            return False

        selected_len = selected_history.shape[1]
        recent_len = min(self.snapkv_window_size, prompt_len)
        kv_len = self._snapkv_compact_kv_len
        generated_len = max(0, kv_len - selected_len - recent_len)
        if kv_len == 0:
            return False

        device = query.device
        k_dense = key_cache[:kv_len]
        v_dense = value_cache[:kv_len]
        cu_seqlens_q = torch.tensor([0, 1],
                                    dtype=torch.int32,
                                    device=device)
        cu_seqlens_k = torch.tensor([0, kv_len],
                                    dtype=torch.int32,
                                    device=device)
        descale_shape = (1, self.num_heads)

        flash_attn_varlen_func(
            q=query[:1].contiguous(),
            k=k_dense.contiguous(),
            v=v_dense.contiguous(),
            out=output[:1],
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=1,
            max_seqlen_k=kv_len,
            softmax_scale=self.scale,
            causal=False,
            window_size=(-1, -1),
            softcap=self.logits_soft_cap,
            fa_version=self.vllm_flash_attn_version,
            q_descale=layer._q_scale.expand(descale_shape),
            k_descale=layer._k_scale.expand(descale_shape),
            v_descale=layer._v_scale.expand(descale_shape),
        )
        self._snapkv_log_decode(layer, seq_len, prompt_len, selected_len,
                                recent_len, generated_len, kv_len)
        return True

    def _forward_encoder_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        layer: torch.nn.Module,
    ) -> torch.Tensor:
        """Forward pass for encoder attention without KV cache.

        Args:
            query: shape = [num_encoder_tokens, num_heads, head_size]
            key: shape = [num_encoder_tokens, num_kv_heads, head_size]
            value: shape = [num_encoder_tokens, num_kv_heads, head_size]
            output: shape = [num_encoder_tokens, num_heads, head_size]
            attn_metadata: Encoder attention metadata
            layer: The attention layer
        """
        # For encoder attention, process FP8 quantization if needed
        if self.kv_cache_dtype.startswith("fp8"):
            raise NotImplementedError(
                "quantization is not supported for encoder attention")

        # Use encoder-specific metadata for sequence information
        cu_seqlens_q = attn_metadata.query_start_loc
        cu_seqlens_k = attn_metadata.query_start_loc
        max_seqlen_q = attn_metadata.max_query_len
        max_seqlen_k = attn_metadata.max_query_len

        descale_shape = (
            cu_seqlens_q.shape[0] - 1,  # type: ignore[union-attr]
            self.num_kv_heads)

        # Call flash attention directly on Q, K, V tensors
        flash_attn_varlen_func(
            q=query,
            k=key,
            v=value,
            out=output,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.scale,
            causal=False,  # Encoder attention is bidirectional
            alibi_slopes=self.alibi_slopes,
            window_size=self.sliding_window,
            softcap=self.logits_soft_cap,
            fa_version=self.vllm_flash_attn_version,
            q_descale=layer._q_scale.expand(descale_shape),
            k_descale=layer._k_scale.expand(descale_shape),
            v_descale=layer._v_scale.expand(descale_shape),
        )

        return output


def use_cascade_attention(
    common_prefix_len: int,
    query_lens: np.ndarray,
    num_query_heads: int,
    num_kv_heads: int,
    use_alibi: bool,
    use_sliding_window: bool,
    use_local_attention: bool,
    num_sms: int,
) -> bool:
    """Decide whether to use cascade attention.

    This function 1) checks whether cascade attention is supported with the
    given configuration, and 2) heuristically decides whether using cascade
    attention can improve performance.
    """
    # Too short common prefix. Probably not worth using cascade attention.
    # We use an arbitrary threshold of 256 tokens. TODO: Tune this threshold.
    # NOTE(woosuk): This is the common case. We should return False as soon as
    # possible to avoid any unnecessary computation.
    if common_prefix_len < 256:
        return False
    # Cascade attention is currently not supported with these variants.
    if use_alibi or use_sliding_window or use_local_attention:
        return False
    # Too few queries. Probably not worth using cascade attention.
    # We use an arbitrary threshold of 8 queries. TODO: Tune this threshold.
    num_reqs = len(query_lens)
    if num_reqs < 8:
        return False

    # Heuristics to decide whether using cascade attention is beneficial.
    # 1. When FlashDecoding is not used for normal attention, cascade attention
    #    is likely to be faster since it saves memory bandwidth.
    num_queries_per_kv = num_query_heads // num_kv_heads
    # The criteria for using FlashDecoding can be found in the following link:
    # https://github.com/vllm-project/flash-attention/blob/96266b1111111f3d11aabefaf3bacbab6a89d03c/csrc/flash_attn/flash_api.cpp#L535
    use_flash_decoding = (num_queries_per_kv > 1 and not use_sliding_window
                          and not use_alibi and np.all(query_lens == 1))
    if not use_flash_decoding:
        # Use cascade attention.
        return True

    # 2. When FlashDecoding is used for normal attention, it is not clear
    #    whether cascade attention is beneficial, because FlashDecoding can
    #    launch more CTAs than cascade attention.
    #    We use a simple performance model to compare the two methods.
    #    NOTE(woosuk): The performance model is very rough and may not be
    #    accurate.
    num_tokens = num_reqs
    # NOTE(woosuk): These are default tile sizes. flash-attn might use
    # different tile sizes (e.g., 64 or 256) depending on the configuration.
    q_tile_size = 128
    kv_tile_size = 128
    num_prefix_tiles = cdiv(common_prefix_len, kv_tile_size)

    cascade_ctas = num_query_heads * cdiv(num_tokens, q_tile_size)
    cascade_waves = cdiv(cascade_ctas, num_sms)
    cascade_time = cascade_waves * num_prefix_tiles

    flash_decoding_ctas = (num_reqs * num_kv_heads *
                           cdiv(num_queries_per_kv, q_tile_size))
    flash_decoding_ctas *= num_prefix_tiles
    flash_decoding_time = cdiv(flash_decoding_ctas, num_sms)

    # Use cascade attention if it is faster than FlashDecoding.
    return cascade_time < flash_decoding_time


def cascade_attention(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cu_query_lens: torch.Tensor,
    max_query_len: int,
    cu_prefix_query_lens: torch.Tensor,
    prefix_kv_lens: torch.Tensor,
    suffix_kv_lens: torch.Tensor,
    max_kv_len: int,
    softmax_scale: float,
    alibi_slopes: Optional[torch.Tensor],
    sliding_window: tuple[int, int],
    logits_soft_cap: float,
    block_table: torch.Tensor,
    common_prefix_len: int,
    fa_version: int,
    prefix_scheduler_metadata: Optional[torch.Tensor] = None,
    suffix_scheduler_metadata: Optional[torch.Tensor] = None,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert alibi_slopes is None, ("Cascade attention does not support ALiBi.")
    # TODO: Support sliding window.
    assert sliding_window == (-1, -1), (
        "Cascade attention does not support sliding window.")

    num_tokens = query.shape[0]
    block_size = key_cache.shape[-3]
    assert common_prefix_len % block_size == 0
    num_common_kv_blocks = common_prefix_len // block_size
    assert num_common_kv_blocks > 0
    descale_shape = (cu_prefix_query_lens.shape[0] - 1, key_cache.shape[-2])

    # Process shared prefix.
    prefix_output, prefix_lse = flash_attn_varlen_func(
        q=query,
        k=key_cache,
        v=value_cache,
        cu_seqlens_q=cu_prefix_query_lens,
        seqused_k=prefix_kv_lens,
        max_seqlen_q=num_tokens,
        max_seqlen_k=common_prefix_len,
        softmax_scale=softmax_scale,
        causal=False,
        window_size=sliding_window,
        block_table=block_table[:1],
        softcap=logits_soft_cap,
        return_softmax_lse=True,
        scheduler_metadata=prefix_scheduler_metadata,
        fa_version=fa_version,
        q_descale=q_descale.expand(descale_shape)
        if q_descale is not None else None,
        k_descale=k_descale.expand(descale_shape)
        if k_descale is not None else None,
        v_descale=v_descale.expand(descale_shape)
        if v_descale is not None else None,
    )

    descale_shape = (cu_query_lens.shape[0] - 1, key_cache.shape[-2])

    # Process suffix per query.
    suffix_output, suffix_lse = flash_attn_varlen_func(
        q=query,
        k=key_cache,
        v=value_cache,
        cu_seqlens_q=cu_query_lens,
        seqused_k=suffix_kv_lens,
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_kv_len - common_prefix_len,
        softmax_scale=softmax_scale,
        causal=True,
        window_size=sliding_window,
        block_table=block_table[:, num_common_kv_blocks:],
        softcap=logits_soft_cap,
        return_softmax_lse=True,
        scheduler_metadata=suffix_scheduler_metadata,
        fa_version=fa_version,
        q_descale=q_descale.expand(descale_shape)
        if q_descale is not None else None,
        k_descale=k_descale.expand(descale_shape)
        if k_descale is not None else None,
        v_descale=v_descale.expand(descale_shape)
        if v_descale is not None else None,
    )

    # Merge prefix and suffix outputs, and store the result in output.
    merge_attn_states(output, prefix_output, prefix_lse, suffix_output,
                      suffix_lse)
