"""Variable-length packed attention via PyTorch's native SDPA.

Uses jagged NestedTensors + ``F.scaled_dot_product_attention`` to process
multiple variable-length sequences in a single fused kernel (FlashAttention
or Memory-Efficient backend selected automatically by PyTorch).  No external
``flash-attn`` package required — works with PyTorch >= 2.4.

The standard ``nn.MultiheadAttention`` path in the TransformerEncoder remains
unchanged — this module is only invoked explicitly by callers that want the
varlen optimization for packed mask paths.

Usage:
    from asymdsd.layers.flash_attention import varlen_encoder_forward

    # flat_x: (total_tokens, D), flat_pos: (total_tokens, D)
    # offsets: (num_seqs + 1,) int64 cumulative sequence lengths
    out = varlen_encoder_forward(encoder, flat_x, flat_pos, offsets)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

try:
    from torch.nested import nested_tensor_from_jagged

    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False


def _extract_qkv_weights(
    mha: nn.MultiheadAttention,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Extract Q, K, V weight matrices from nn.MultiheadAttention."""
    embed_dim = mha.embed_dim
    w = mha.in_proj_weight
    wq = w[:embed_dim]
    wk = w[embed_dim : 2 * embed_dim]
    wv = w[2 * embed_dim :]
    if mha.in_proj_bias is not None:
        b = mha.in_proj_bias
        bq = b[:embed_dim]
        bk = b[embed_dim : 2 * embed_dim]
        bv = b[2 * embed_dim :]
        bias = (bq, bk, bv)
    else:
        bias = None
    return wq, wk, wv, bias


def _apply_varlen_drop_path(
    x: torch.Tensor,
    residual: torch.Tensor,
    offsets: torch.Tensor,
    drop_p: float,
    training: bool,
) -> torch.Tensor:
    """Apply DropPath on a per-sequence basis in the varlen flat tensor.

    Matches the behavior of ``drop_path_efficient``: randomly selects a
    subset of sequences to keep, scales their residuals by 1/keep_p, and
    zeros out the rest.
    """
    if not training or drop_p == 0.0:
        return x + residual

    num_seqs = offsets.shape[0] - 1
    keep_p = 1.0 - drop_p
    keep_count = max(1, round(keep_p * num_seqs))
    keep_seq_indices = torch.randperm(num_seqs, device=x.device)[:keep_count]

    scale = torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)
    for idx in keep_seq_indices:
        start = offsets[idx]
        end = offsets[idx + 1]
        scale[start:end] = 1.0 / keep_p

    return x + residual * scale


def _varlen_block_forward(
    block: nn.Module,
    x: torch.Tensor,
    pos_enc: torch.Tensor | None,
    offsets: torch.Tensor,
    add_pos_enc: bool,
) -> torch.Tensor:
    """Run one Block through SDPA with jagged nested tensors.

    Args:
        block: A Block instance from TransformerEncoder.
        x: (total_tokens, D) flat input.
        pos_enc: (total_tokens, D) flat positional encoding, or None.
        offsets: (num_seqs + 1,) int64 cumulative lengths.
        add_pos_enc: whether to add pos_enc at this layer.
    """
    if add_pos_enc and pos_enc is not None:
        x = x + pos_enc

    drop_p = block.drop_path_p
    training = block.training

    self_attn = block.self_attn
    assert self_attn is not None

    # --- Self-attention via SDPA ---
    norm = self_attn.norm
    x_normed = norm(x)

    mha = self_attn.attn
    embed_dim = mha.embed_dim
    num_heads = mha.num_heads
    head_dim = embed_dim // num_heads

    wq, wk, wv, bias = _extract_qkv_weights(mha)

    if bias is not None:
        bq, bk, bv = bias
        q = F.linear(x_normed, wq, bq)
        k = F.linear(x_normed, wk, bk)
        v = F.linear(x_normed, wv, bv)
    else:
        q = F.linear(x_normed, wq)
        k = F.linear(x_normed, wk)
        v = F.linear(x_normed, wv)

    # (total_tokens, embed_dim) → (total_tokens, num_heads, head_dim)
    q = q.view(-1, num_heads, head_dim)
    k = k.view(-1, num_heads, head_dim)
    v = v.view(-1, num_heads, head_dim)

    # Build jagged nested tensors: (num_seqs, jagged_seq_len, num_heads, head_dim)
    q_nt = nested_tensor_from_jagged(q, offsets=offsets)
    k_nt = nested_tensor_from_jagged(k, offsets=offsets)
    v_nt = nested_tensor_from_jagged(v, offsets=offsets)

    # SDPA expects (batch, heads, seq, head_dim)
    q_nt = q_nt.transpose(1, 2)
    k_nt = k_nt.transpose(1, 2)
    v_nt = v_nt.transpose(1, 2)

    dropout_p = mha.dropout if training else 0.0
    attn_out_nt = F.scaled_dot_product_attention(
        q_nt, k_nt, v_nt, dropout_p=dropout_p
    )

    # (batch, heads, jagged, hdim) → (batch, jagged, heads, hdim) → flat values
    attn_out_nt = attn_out_nt.transpose(1, 2)
    attn_out = attn_out_nt.values()  # (total_tokens, num_heads, head_dim)
    attn_out = attn_out.reshape(-1, embed_dim)

    # Output projection + LayerScale
    attn_out = mha.out_proj(attn_out)
    attn_out = self_attn.layer_scale(attn_out)

    # Residual + DropPath (per-sequence)
    x = _apply_varlen_drop_path(x, attn_out, offsets, drop_p, training)

    # --- FFN ---
    ffn_out = block.ffn(x)
    x = _apply_varlen_drop_path(x, ffn_out, offsets, drop_p, training)

    return x


def varlen_encoder_forward(
    encoder: nn.Module,
    flat_x: torch.Tensor,
    flat_pos: torch.Tensor,
    offsets: torch.Tensor,
    max_seqlen: int | None = None,
) -> torch.Tensor:
    """Run TransformerEncoder on packed varlen sequences via native SDPA.

    This function uses the *existing* weights from the encoder's Block layers
    (nn.MultiheadAttention, FFN, LayerNorm, LayerScale) but bypasses the
    standard forward path to use PyTorch's fused SDPA kernel instead.

    Args:
        encoder: A TransformerEncoder (or TransformerModule) instance.
        flat_x: (total_tokens, embed_dim) concatenated token embeddings.
        flat_pos: (total_tokens, embed_dim) concatenated positional embeddings.
        offsets: (num_sequences + 1,) int64 cumulative sequence lengths.
        max_seqlen: unused, kept for API compatibility.

    Returns:
        (total_tokens, embed_dim) output after all layers + final norm.
    """
    add_pos_enc_every_layer = encoder.add_pos_enc

    if not add_pos_enc_every_layer:
        flat_x = flat_x + flat_pos
        pos_for_layers = None
    else:
        pos_for_layers = flat_pos

    for block in encoder.stack:
        flat_x = _varlen_block_forward(
            block,
            flat_x,
            pos_for_layers,
            offsets,
            add_pos_enc=add_pos_enc_every_layer,
        )

    flat_x = encoder.norm(flat_x)
    return flat_x


def has_relative_3d_bias(encoder: nn.Module) -> bool:
    """Check if any block in the encoder uses Relative3DBias."""
    for block in encoder.stack:
        if block.self_attn is not None and block.self_attn.relative_3d_bias is not None:
            return True
    return False


def build_cu_seqlens(
    seq_lengths: list[int],
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    """Build offsets tensor from a list of sequence lengths.

    Args:
        seq_lengths: list of individual sequence lengths.
        device: target device.

    Returns:
        offsets: (len(seq_lengths) + 1,) int64 tensor.
        max_seqlen: maximum sequence length.
    """
    offsets = torch.zeros(len(seq_lengths) + 1, dtype=torch.int64, device=device)
    for i, length in enumerate(seq_lengths):
        offsets[i + 1] = offsets[i] + length
    max_seqlen = max(seq_lengths)
    return offsets, max_seqlen


def build_cu_seqlens_from_groups(
    group_lengths: list[tuple[int, int]],
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    """Build offsets from groups of (count, seq_len) pairs.

    For packed masking: each group represents sequences of the same length
    (e.g., all 70%-masked paths have the same visible count).

    Args:
        group_lengths: [(count_0, seqlen_0), (count_1, seqlen_1), ...].
        device: target device.

    Returns:
        offsets: int64 tensor.
        max_seqlen: maximum sequence length.
    """
    offset_list = [0]
    max_seqlen = 0
    for count, seqlen in group_lengths:
        max_seqlen = max(max_seqlen, seqlen)
        for _ in range(count):
            offset_list.append(offset_list[-1] + seqlen)
    offsets = torch.tensor(offset_list, dtype=torch.int64, device=device)
    return offsets, max_seqlen
