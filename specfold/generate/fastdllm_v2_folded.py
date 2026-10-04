"""SpecFold for Fast-dLLM v2: the B+1 batched verification forward with folding.

Only ``_batched_block_forward`` is overridden; drafting, draft-logit reuse and
unmasking come from the Spiffy base class. ``SpecFoldFastDLLMV2Generation`` is the
PyTorch reference, ``SpecFoldTritonFastDLLMV2Generation`` the fused-kernel path.
With the gate disabled both reproduce the dense forward.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from specfold.generate.fastdllm_v2_generation import FastDLLMV2SpeculativeGeneration

logger = logging.getLogger(__name__)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)



def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_kv, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv * n_rep, slen, head_dim)


class SpecFoldFastDLLMV2Generation(FastDLLMV2SpeculativeGeneration):
    """Reference (PyTorch) SpecFold layer: per-position residual gate, parent QKV and
    attention state reused at folded positions with the exact chain correction, FFN
    output reused, residual streams kept per row."""

    _packed_weights = None

    def __init__(self, *args, delta: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.delta = float(delta)
        self._fold_stats: List[dict] = []

    @property
    def fold_ratio(self):
        tot = getattr(self, "_fold_draft_positions", 0)
        rec = getattr(self, "_fold_draft_recomputed", 0)
        return 1.0 - rec / tot if tot else float("nan")

    @staticmethod
    def _build_parent_map(batched_input: torch.Tensor) -> Tuple[List[Optional[int]], List[int]]:
        """par(i): the branch whose differing-position set is a size-(k-1) subset
        of draft i's. Root = main (branch 0); fallback parent = main.
        Computed on GPU with a single D2H sync (was B+1 nonzero syncs)."""
        B1 = batched_input.shape[0]
        neq = batched_input != batched_input[0:1]          # [B1, N]; row 0 all False
        counts_t = neq.sum(1)                              # [B1]
        # not_subset[i, j] = True iff diffs[j] is NOT a subset of diffs[i]
        not_subset = (neq.unsqueeze(0) & ~neq.unsqueeze(1)).any(-1)  # [B1(i), B1(j)]
        counts = counts_t.tolist()
        ns = not_subset.tolist()                           # one sync for both
        parent: List[Optional[int]] = [None] * B1
        for i in range(1, B1):
            best = 0
            for j in range(B1):
                if j != i and counts[j] == counts[i] - 1 and not ns[i][j]:
                    best = j
                    break
            parent[i] = best
        return parent, counts

    def _pack_layer_weights(self, layers) -> None:
        """Cache per-layer packed projections so QKV and gate/up each run as one
        GEMM (one weight stream, one launch) instead of three/two."""
        packed = []
        for lyr in layers:
            attn, mlp = lyr.self_attn, lyr.mlp
            w_qkv = torch.cat([attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight], dim=0)
            b_qkv = torch.cat([attn.q_proj.bias, attn.k_proj.bias, attn.v_proj.bias], dim=0)
            w_gu = torch.cat([mlp.gate_proj.weight, mlp.up_proj.weight], dim=0)
            packed.append((w_qkv, b_qkv, w_gu))
        self._packed_weights = packed

    @torch.no_grad()
    def _batched_block_forward(self, batched_input: torch.Tensor, past_key_values) -> torch.Tensor:
        B1, N = batched_input.shape
        if B1 == 1:
            return FastDLLMV2SpeculativeGeneration._batched_block_forward(
                self, batched_input, past_key_values
            )
        m_ = self.model_executor.model
        base = m_.model
        cfg = m_.config
        device = batched_input.device
        H = cfg.num_attention_heads
        Hkv = cfg.num_key_value_heads
        G = H // Hkv
        layers = base.layers
        P = int(past_key_values.get_seq_length()) if past_key_values is not None else 0
        delta = float(self.delta)

        parent, counts = self._build_parent_map(batched_input)
        order = sorted(range(B1), key=lambda i: counts[i])

        hidden = base.embed_tokens(batched_input)
        D = hidden.shape[-1]
        position_ids = torch.arange(P, P + N, device=device).unsqueeze(0)
        cos, sin = base.rotary_emb(hidden, position_ids)
        cos_n, sin_n = cos[0], sin[0]
        h = [hidden[i : i + 1] for i in range(B1)]

        for lyr in layers:
            attn = lyr.self_attn
            hd = attn.head_dim
            scaling = attn.scaling
            li = attn.layer_idx
            pk, pv = (past_key_values[li] if past_key_values is not None and len(past_key_values) > li
                      else (None, None))
            if pk is not None:
                pkh = _repeat_kv(pk, G)[0]  # [H, P, hd]
                pvh = _repeat_kv(pv, G)[0]

            qt = [None] * B1; kt = [None] * B1; vt = [None] * B1     # [H|Hkv, N, hd] roped tables
            mt = [None] * B1; Zt = [None] * B1; Ot = [None] * B1     # [H,N], [H,N], [H,N,hd]
            ft = [None] * B1                                          # ffn out [1,N,D]
            hp = [None] * B1; hn = [None] * B1

            for i in order:
                x = h[i]
                pj = parent[i]
                if pj is None:
                    rbar = torch.arange(N, device=device)
                else:
                    dn = (x[0] - h[pj][0]).norm(dim=-1)
                    pn = h[pj][0].norm(dim=-1)
                    rbar = (dn > delta * pn).nonzero(as_tuple=True)[0]
                    self._fold_draft_positions = getattr(self, "_fold_draft_positions", 0) + N
                    self._fold_draft_recomputed = getattr(self, "_fold_draft_recomputed", 0) + int(rbar.numel())
                R = int(rbar.numel())

                # tables: inherit parent, overwrite recomputed positions
                if pj is None:
                    qtab = x.new_zeros(H, N, hd); ktab = x.new_zeros(Hkv, N, hd); vtab = x.new_zeros(Hkv, N, hd)
                else:
                    qtab = qt[pj].clone(); ktab = kt[pj].clone(); vtab = vt[pj].clone()
                if R:
                    lnr = lyr.input_layernorm(x)[0, rbar]
                    qf = attn.q_proj(lnr).view(R, H, hd).transpose(0, 1)
                    kf = attn.k_proj(lnr).view(R, Hkv, hd).transpose(0, 1)
                    vf = attn.v_proj(lnr).view(R, Hkv, hd).transpose(0, 1)
                    c = cos_n[rbar].unsqueeze(0); s = sin_n[rbar].unsqueeze(0)
                    qf = qf * c + _rotate_half(qf) * s
                    kf_r = kf * c + _rotate_half(kf) * s
                    k_old = ktab[:, rbar].clone() if pj is not None else None
                    v_old = vtab[:, rbar].clone() if pj is not None else None
                    qtab[:, rbar] = qf; ktab[:, rbar] = kf_r; vtab[:, rbar] = vf
                qt[i], kt[i], vt[i] = qtab, ktab, vtab

                kh = _repeat_kv(ktab.unsqueeze(0), G)[0] if Hkv != H else ktab
                vh = _repeat_kv(vtab.unsqueeze(0), G)[0] if Hkv != H else vtab
                K = torch.cat([pkh, kh], dim=1) if pk is not None else kh   # [H, P+N, hd]
                V = torch.cat([pvh, vh], dim=1) if pv is not None else vh

                mtab = torch.full((H, N), float("-inf"), dtype=torch.float32, device=device)
                Ztab = torch.zeros(H, N, dtype=torch.float32, device=device)
                Otab = torch.zeros(H, N, hd, dtype=torch.float32, device=device)
                if pj is not None:
                    mtab.copy_(mt[pj]); Ztab.copy_(Zt[pj]); Otab.copy_(Ot[pj])

                if R:  # recomputed positions: full attention with (m,Z,O) retained
                    sc = torch.einsum("hrd,hkd->hrk", qtab[:, rbar].float(), K.float()) * scaling
                    mr = sc.max(dim=-1).values
                    ex = torch.exp(sc - mr.unsqueeze(-1))
                    Zr = ex.sum(-1)
                    Or = torch.einsum("hrk,hkd->hrd", ex, V.float()) / Zr.unsqueeze(-1)
                    mtab[:, rbar] = mr; Ztab[:, rbar] = Zr; Otab[:, rbar] = Or

                if pj is not None and R:
                    lock = torch.ones(N, dtype=torch.bool, device=device); lock[rbar] = False
                    Lx = lock.nonzero(as_tuple=True)[0]
                    if Lx.numel():
                        qL = qtab[:, Lx].float()                       # [H, L, hd]
                        koh = _repeat_kv(k_old.unsqueeze(0), G)[0] if Hkv != H else k_old
                        voh = _repeat_kv(v_old.unsqueeze(0), G)[0] if Hkv != H else v_old
                        knh = kh[:, rbar]; vnh = vh[:, rbar]
                        s_old = torch.einsum("hld,hed->hle", qL, koh.float()) * scaling
                        s_new = torch.einsum("hld,hed->hle", qL, knh.float()) * scaling
                        mo = mt[pj][:, Lx]                             # parent's row max
                        mn = torch.maximum(mo, s_new.max(-1).values)   # rescale if needed
                        f = torch.exp(mo - mn)
                        a_old = torch.exp(s_old - mn.unsqueeze(-1))
                        a_new = torch.exp(s_new - mn.unsqueeze(-1))
                        Zp = Zt[pj][:, Lx] * f
                        num = Ot[pj][:, Lx] * (Zp).unsqueeze(-1)
                        num = num - torch.einsum("hle,hed->hld", a_old, voh.float()) \
                                  + torch.einsum("hle,hed->hld", a_new, vnh.float())
                        Zn = Zp - a_old.sum(-1) + a_new.sum(-1)
                        mtab[:, Lx] = mn; Ztab[:, Lx] = Zn
                        Otab[:, Lx] = num / Zn.unsqueeze(-1)
                mt[i], Zt[i], Ot[i] = mtab, Ztab, Otab

                Ofull = Otab.permute(1, 0, 2).reshape(N, H * hd).to(x.dtype)
                hpi = x + (Ofull @ attn.o_proj.weight.t()).unsqueeze(0)
                hp[i] = hpi

                # FFN: recomputed positions run it; folded ones reuse the parent's output
                ftab = ft[pj].clone() if pj is not None else x.new_zeros(1, N, D)
                if R:
                    gu = lyr.mlp
                    z = lyr.post_attention_layernorm(hpi[0, rbar])
                    ftab[0, rbar] = gu.down_proj(F.silu(gu.gate_proj(z)) * gu.up_proj(z))
                ft[i] = ftab
                hn[i] = hpi + ftab

            h = hn

        final = base.norm(torch.cat(h, dim=0))
        return self._shift_logits(m_.lm_head(final))


class SpecFoldTritonFastDLLMV2Generation(SpecFoldFastDLLMV2Generation):
    """Fused SpecFold: delta gate + K1 QKV + flash-with-state + exact corrections
    (per-branch topo launches) + fused FFN; same math as the reference class."""

    def __init__(self, *args, fold_bucket: Optional[int] = None, **kwargs):
        super().__init__(*args, **kwargs)
        # None (default): auto-size the compaction bucket to B1*N so any draft
        # count fits (for B=8 this equals the old fixed 288).
        # An explicit int still caps it.
        self.fold_bucket = None if fold_bucket is None else int(fold_bucket)

    @torch.no_grad()
    def _batched_block_forward(self, batched_input: torch.Tensor, past_key_values) -> torch.Tensor:
        B1, N = batched_input.shape
        if B1 == 1:
            return FastDLLMV2SpeculativeGeneration._batched_block_forward(
                self, batched_input, past_key_values
            )
        import triton as _tr
        from specfold.generate.triton_kernels import (
            residual_delta_gate_kernel, src_resolve_kernel,
            fused_ln_qkv_rope_scatter_kernel, specfold_flash_attn_gqa_kernel,
            specfold_chain_correction_gqa_kernel, specfold_flash_attn_kernel,
            specfold_chain_correction_v2_kernel, fused_ffn_scatter,
        )
        m_ = self.model_executor.model
        base = m_.model
        cfg = m_.config
        device = batched_input.device
        H = cfg.num_attention_heads
        Hkv = cfg.num_key_value_heads
        layers = base.layers
        if self._packed_weights is None:
            self._pack_layer_weights(layers)
        P = int(past_key_values.get_seq_length()) if past_key_values is not None else 0
        delta = float(self.delta)

        parent, counts_l = self._build_parent_map(batched_input)
        order = [i for i in sorted(range(1, B1), key=lambda i: counts_l[i])]
        par_idx = torch.tensor([0] + [parent[i] for i in range(1, B1)], device=device, dtype=torch.int64)

        hidden = base.embed_tokens(batched_input)
        D = hidden.shape[-1]
        position_ids = torch.arange(P, P + N, device=device).unsqueeze(0)
        cos, sin = base.rotary_emb(hidden, position_ids)
        cos_c, sin_c = cos[0].contiguous(), sin[0].contiguous()
        hd = layers[0].self_attn.head_dim
        q_dim, kv_dim = H * hd, Hkv * hd
        bucket = (B1 * N) if self.fold_bucket is None else min(self.fold_bucket, B1 * N)
        TRASH = B1 * N
        BN = B1 * N
        q_buf = hidden.new_zeros(BN + 1, q_dim)
        kv_buf = hidden.new_zeros(BN + 1, 2 * kv_dim)
        O_tab = hidden.new_zeros(BN, q_dim)
        M_tab = torch.zeros(BN, H, dtype=torch.float32, device=device)
        Z_tab = torch.zeros(BN, H, dtype=torch.float32, device=device)
        ffn_buf = hidden.new_zeros(BN + 1, D)
        act_buf = hidden.new_empty(bucket, cfg.intermediate_size)
        rstd2 = torch.empty(bucket, dtype=torch.float32, device=device)
        n_flat = torch.arange(N, device=device).repeat(B1)   # position of each flat row
        # per-forward reusable gate buffers (values beyond count never read)
        rstd = torch.empty(BN, dtype=torch.float32, device=device)
        f = torch.empty(BN, dtype=torch.int64, device=device)
        rec = torch.empty(BN, dtype=torch.int8, device=device)
        rows = torch.zeros(bucket, dtype=torch.int64, device=device)
        src = torch.empty(BN, dtype=torch.int64, device=device)
        count_slab = torch.zeros(len(layers), dtype=torch.int32, device=device)
        seg = torch.zeros(BN, dtype=torch.int64, device=device)
        segcnt_slab = torch.zeros(len(layers) * B1, dtype=torch.int32, device=device)

        for lyr in layers:
            attn = lyr.self_attn
            li = attn.layer_idx
            w_qkv, b_qkv, w_gu = self._packed_weights[li]
            pk, pv = (past_key_values[li] if past_key_values is not None and len(past_key_values) > li
                      else (None, None))
            Hf = hidden.reshape(BN, D)

            count = count_slab[li : li + 1]
            segcnt = segcnt_slab[li * B1 : (li + 1) * B1]
            residual_delta_gate_kernel[(BN,)](
                Hf, par_idx, rstd, f, rec, rows, count, seg, segcnt,
                delta, B1, N, D, bucket, 1e-6, BLOCK_D=512,
            )
            src_resolve_kernel[(BN,)](f, src, N, HOPS=8)
            self._fold_draft_positions = getattr(self, "_fold_draft_positions", 0) + (B1 - 1) * N

            fused_ln_qkv_rope_scatter_kernel[( _tr.cdiv(bucket, 32), H + 2 * Hkv)](
                Hf, w_qkv, b_qkv, lyr.input_layernorm.weight, rstd, rows, count,
                cos_c, sin_c, q_buf, kv_buf,
                D, N, bucket, TRASH,
                NUM_Q_HEADS=H, NUM_KV_HEADS=Hkv, HD=hd, HALF=hd // 2,
                BLOCK_M=32, BLOCK_K=64,
            )
            if pk is not None:
                pk0, pv0, P_i = pk[0], pv[0], P
            else:
                pk0, pv0, P_i = kv_buf, kv_buf, 0
            grp = H // Hkv
            if getattr(self, "use_gqa_kernels", True):
                qp = 1
                while qp < grp:
                    qp *= 2
                specfold_flash_attn_gqa_kernel[(bucket, Hkv)](
                    q_buf, pk0, pv0, kv_buf, src, rows, count, O_tab, M_tab, Z_tab,
                    P_i, N, attn.scaling, bucket,
                    GROUP=grp, HKV=Hkv, HD=hd, BLOCK_J=64, QP=max(16, qp),
                )
                specfold_chain_correction_gqa_kernel[(BN, Hkv)](
                    q_buf, kv_buf, src, par_idx, rec, seg, segcnt, O_tab, M_tab, Z_tab,
                    N, attn.scaling,
                    GROUP=grp, HKV=Hkv, HD=hd, MAXDEPTH=8, QP=qp,
                )
            else:
                specfold_flash_attn_kernel[(bucket, H)](
                    q_buf, pk0, pv0, kv_buf, src, rows, count, O_tab, M_tab, Z_tab,
                    P_i, N, attn.scaling, bucket,
                    GROUP=grp, HKV=Hkv, HD=hd, BLOCK_J=64,
                )
                specfold_chain_correction_v2_kernel[(BN, H)](
                    q_buf, kv_buf, src, par_idx, rec, seg, segcnt, O_tab, M_tab, Z_tab,
                    N, attn.scaling,
                    GROUP=grp, HKV=Hkv, HD=hd, MAXDEPTH=8,
                )

            hp_all = torch.addmm(Hf, O_tab, attn.o_proj.weight.t())     # residual never folded
            hp_rows = hp_all[rows.clamp(max=BN - 1)]  # compacted recomputed positions
            inter = w_gu.shape[0] // 2
            fused_ffn_scatter(
                hp_rows, lyr.post_attention_layernorm.weight,
                w_gu[:inter], w_gu[inter:], lyr.mlp.down_proj.weight,
                rows, count, ffn_buf, bucket, TRASH, act_buf, rstd2,
                store_ffn_only=True,
            )
            # ffn_buf holds PURE ffn at recomputed positions; src only points at those
            hidden = (hp_all + ffn_buf[:BN][src * N + n_flat]).view(B1, N, D)

        mx, sm = torch.stack([count_slab.max(), count_slab.sum()]).tolist()
        self._fold_draft_recomputed = getattr(self, "_fold_draft_recomputed", 0) + int(sm) - len(layers) * N
        final = base.norm(hidden)
        logits = self._shift_logits(m_.lm_head(final))
        if int(mx) > bucket:
            logger.warning("SpecFold bucket overflow (%d > %d); falling back to prototype", int(mx), bucket)
            return SpecFoldFastDLLMV2Generation._batched_block_forward(self, batched_input, past_key_values)
        return logits
