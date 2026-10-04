"""Triton fused kernels for SpecFold, shared by both model families.

Architecture-agnostic: row indices, weights, bias tensors (zero for NLD) and
norm eps come in as arguments, so fastdllm_v2_folded.py and nld_folded.py call
the same kernels. Shared building blocks (compaction, fused LN+QKV+RoPE scatter,
fused FFN scatter) plus the SpecFold-specific kernels: the relative-residual
gate, the flash attention kernel that emits per-row (m, Z) softmax state, and
the order-free chain-correction kernels (exact rank-|R| attention update along
the ancestor path).
"""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def src_resolve_kernel(F_ptr, SRC_ptr, N, HOPS: tl.constexpr):
    """src[i,n] = nearest recomputing ancestor: follow f HOPS times (>= depth)."""
    pid = tl.program_id(0)
    n = pid % N
    s = tl.load(F_ptr + pid)
    for _ in range(HOPS):
        s = tl.load(F_ptr + s * N + n)
    tl.store(SRC_ptr + pid, s)


@triton.jit
def fused_ln_qkv_rope_scatter_kernel(
    H_ptr, W_ptr, BIAS_ptr, LNW_ptr, RSTD_ptr, ROWS_ptr, COUNT_ptr,
    COS_ptr, SIN_ptr,      # [N, hd] fp32-loadable
    QTAB_ptr,              # [B1*N+1, H*hd]
    KV_ptr,                # [B1*N+1, 2*Hkv*hd]  (k heads then v heads)
    D, N, bucket, trash,
    NUM_Q_HEADS: tl.constexpr, NUM_KV_HEADS: tl.constexpr,
    HD: tl.constexpr, HALF: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """One col-tile per head (BLOCK_N == HD). Epilogue: bias, RoPE (q/k),
    direct scatter into q/kv tables at rows (dead slots -> trash row)."""
    pid_m = tl.program_id(0)
    head = tl.program_id(1)            # 0..H-1 q, H..H+Hkv-1 k, then v
    count = tl.minimum(tl.load(COUNT_ptr), bucket)
    m0 = pid_m * BLOCK_M
    if m0 >= count:
        return
    offs_m = m0 + tl.arange(0, BLOCK_M)
    live = offs_m < count
    rows = tl.load(ROWS_ptr + offs_m, mask=live, other=0)
    row_store = tl.where(live, rows, trash)
    ni = rows % N
    offs_k = tl.arange(0, BLOCK_K)
    offs_n = head * HD + tl.arange(0, HD)   # cols in packed [O]
    rstd = tl.load(RSTD_ptr + rows, mask=live, other=0.0)

    acc = tl.zeros((BLOCK_M, HD), dtype=tl.float32)
    for k0 in range(0, D, BLOCK_K):
        a = tl.load(H_ptr + rows[:, None] * D + k0 + offs_k[None, :],
                    mask=live[:, None], other=0.0).to(tl.float32)
        lnw = tl.load(LNW_ptr + k0 + offs_k).to(tl.float32)
        a = a * rstd[:, None] * lnw[None, :]
        w = tl.load(W_ptr + offs_n[:, None] * D + k0 + offs_k[None, :]).to(tl.float32)
        acc += tl.dot(a.to(tl.bfloat16), tl.trans(w).to(tl.bfloat16), out_dtype=tl.float32)
    acc += tl.load(BIAS_ptr + offs_n).to(tl.float32)[None, :]

    is_v = head >= NUM_Q_HEADS + NUM_KV_HEADS
    if not is_v:
        # RoPE: out[:HALF] = x1*c - x2*s ; out[HALF:] = x2*c + x1*s
        acc2 = tl.reshape(acc, (BLOCK_M, 2, HALF))
        x1, x2 = tl.split(tl.trans(acc2, (0, 2, 1)))
        c = tl.load(COS_ptr + ni[:, None] * HD + tl.arange(0, HALF)[None, :],
                    mask=live[:, None], other=0.0).to(tl.float32)
        s = tl.load(SIN_ptr + ni[:, None] * HD + tl.arange(0, HALF)[None, :],
                    mask=live[:, None], other=0.0).to(tl.float32)
        o1 = x1 * c - x2 * s
        o2 = x2 * c + x1 * s
        acc = tl.reshape(tl.trans(tl.join(o1, o2), (0, 2, 1)), (BLOCK_M, HD))

    out_dtype = QTAB_ptr.dtype.element_ty
    hd_cols = tl.arange(0, HD)
    if head < NUM_Q_HEADS:
        ptr = QTAB_ptr + row_store[:, None] * (NUM_Q_HEADS * HD) + head * HD + hd_cols[None, :]
    else:
        kvh = head - NUM_Q_HEADS          # 0..2*Hkv-1: k heads then v heads
        ptr = KV_ptr + row_store[:, None] * (2 * NUM_KV_HEADS * HD) + kvh * HD + hd_cols[None, :]
    tl.store(ptr, acc.to(out_dtype))


@triton.jit
def rstd_kernel(X_ptr, RSTD_ptr, D, eps, BLOCK_D: tl.constexpr):
    """Per-row RMSNorm rstd for a compacted [bucket, D] matrix."""
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    sumsq = 0.0
    for d0 in range(0, D, BLOCK_D):
        x = tl.load(X_ptr + pid * D + d0 + offs).to(tl.float32)
        sumsq += tl.sum(x * x)
    tl.store(RSTD_ptr + pid, 1.0 / tl.sqrt(sumsq / D + eps))


@triton.jit
def fused_ln_gateup_silu_kernel(
    X_ptr, LNW_ptr, RSTD_ptr, WG_ptr, WU_ptr, COUNT_ptr, OUT_ptr,
    D, I, bucket,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """act = silu(LN(x) @ Wg^T) * (LN(x) @ Wu^T); one A-load, two GEMMs."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    count = tl.minimum(tl.load(COUNT_ptr), bucket)
    m0 = pid_m * BLOCK_M
    if m0 >= count:
        return
    offs_m = m0 + tl.arange(0, BLOCK_M)
    live = offs_m < count
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    rstd = tl.load(RSTD_ptr + offs_m, mask=live, other=0.0)
    accg = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    accu = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, D, BLOCK_K):
        a = tl.load(X_ptr + offs_m[:, None] * D + k0 + offs_k[None, :],
                    mask=live[:, None], other=0.0).to(tl.float32)
        lnw = tl.load(LNW_ptr + k0 + offs_k).to(tl.float32)
        a = (a * rstd[:, None] * lnw[None, :]).to(tl.bfloat16)
        wg = tl.load(WG_ptr + offs_n[:, None] * D + k0 + offs_k[None, :],
                     mask=(offs_n < I)[:, None], other=0.0).to(tl.bfloat16)
        wu = tl.load(WU_ptr + offs_n[:, None] * D + k0 + offs_k[None, :],
                     mask=(offs_n < I)[:, None], other=0.0).to(tl.bfloat16)
        accg += tl.dot(a, tl.trans(wg), out_dtype=tl.float32)
        accu += tl.dot(a, tl.trans(wu), out_dtype=tl.float32)
    act = accg * tl.sigmoid(accg) * accu
    tl.store(OUT_ptr + offs_m[:, None] * I + offs_n[None, :],
             act.to(OUT_ptr.dtype.element_ty),
             mask=live[:, None] & (offs_n < I)[None, :])


@triton.jit
def fused_down_residual_scatter_kernel(
    ACT_ptr, WD_ptr, HP_ptr, ROWS_ptr, COUNT_ptr, NH_ptr,
    I, D, bucket, trash,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    STORE_FFN_ONLY: tl.constexpr = False,
):
    """nh_table[rows] = hp + act @ Wd^T (dead slots -> trash row);
    STORE_FFN_ONLY skips the residual (pure FFN output for SpecFold)."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    count = tl.minimum(tl.load(COUNT_ptr), bucket)
    m0 = pid_m * BLOCK_M
    if m0 >= count:
        return
    offs_m = m0 + tl.arange(0, BLOCK_M)
    live = offs_m < count
    rows = tl.load(ROWS_ptr + offs_m, mask=live, other=0)
    row_store = tl.where(live, rows, trash)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, I, BLOCK_K):
        a = tl.load(ACT_ptr + offs_m[:, None] * I + k0 + offs_k[None, :],
                    mask=live[:, None], other=0.0).to(tl.bfloat16)
        w = tl.load(WD_ptr + offs_n[:, None] * I + k0 + offs_k[None, :],
                    mask=(offs_n < D)[:, None], other=0.0).to(tl.bfloat16)
        acc += tl.dot(a, tl.trans(w), out_dtype=tl.float32)
    if not STORE_FFN_ONLY:
        hp = tl.load(HP_ptr + offs_m[:, None] * D + offs_n[None, :],
                     mask=live[:, None] & (offs_n < D)[None, :], other=0.0).to(tl.float32)
        acc += hp
    tl.store(NH_ptr + row_store[:, None] * D + offs_n[None, :],
             acc.to(NH_ptr.dtype.element_ty),
             mask=live[:, None] & (offs_n < D)[None, :])


def fused_ffn_scatter(hp_rec, ln_w, w_gate, w_up, w_down, rows, count,
                      nh_tab, bucket, trash, act_buf, rstd_buf, eps_rms=1e-6,
                      store_ffn_only=False):
    """Kernel 3: post-attn RMSNorm + SwiGLU FFN + residual, scattered into
    nh_tab (dead slots -> trash row). No host sync."""
    D = hp_rec.shape[1]
    I = w_gate.shape[0]
    rstd_kernel[(bucket,)](hp_rec, rstd_buf, D, eps_rms, BLOCK_D=512)
    grid1 = (triton.cdiv(bucket, 32), triton.cdiv(I, 128))
    fused_ln_gateup_silu_kernel[grid1](
        hp_rec, ln_w, rstd_buf, w_gate, w_up, count, act_buf,
        D, I, bucket, BLOCK_M=32, BLOCK_N=128, BLOCK_K=64,
    )
    grid2 = (triton.cdiv(bucket, 32), triton.cdiv(D, 128))
    fused_down_residual_scatter_kernel[grid2](
        act_buf, w_down, hp_rec, rows, count, nh_tab,
        I, D, bucket, trash, BLOCK_M=32, BLOCK_N=128, BLOCK_K=64,
        STORE_FFN_ONLY=store_ffn_only,
    )


# ─────────────────────────── SpecFold kernels ────────────────────────────────

@triton.jit
def residual_delta_gate_kernel(
    H_ptr, PAR_ptr, RSTD_ptr, F_ptr, REC_ptr, ROWS_ptr, COUNT_ptr,
    SEG_ptr, SEGCNT_ptr,       # [B1*N] segmented per-branch lists (stride N), [B1] counters
    delta,                     # recompute threshold (relative L2)
    B1, N, D, bucket, eps_rms,
    BLOCK_D: tl.constexpr,
):
    """Gate: rec = ||x - x_par|| > delta * ||x_par|| (root rows: always); compacts the
    recomputed rows and emits rstd for RMSNorm-on-load."""
    pid = tl.program_id(0)
    b = pid // N
    n = pid % N
    par = tl.load(PAR_ptr + b)
    offs = tl.arange(0, BLOCK_D)
    sumsq = 0.0
    dsq = 0.0
    psq = 0.0
    for d0 in range(0, D, BLOCK_D):
        m = offs + d0 < D
        x = tl.load(H_ptr + (b * N + n) * D + d0 + offs, mask=m, other=0.0).to(tl.float32)
        sumsq += tl.sum(x * x)
        if b > 0:
            xp = tl.load(H_ptr + (par * N + n) * D + d0 + offs, mask=m, other=0.0).to(tl.float32)
            dd = x - xp
            dsq += tl.sum(dd * dd)
            psq += tl.sum(xp * xp)
    tl.store(RSTD_ptr + pid, 1.0 / tl.sqrt(sumsq / D + eps_rms))
    if b == 0:
        rec = True
    else:
        rec = dsq > (delta * delta) * psq
    tl.store(REC_ptr + pid, rec.to(tl.int8))
    tl.store(F_ptr + pid, tl.where(rec, b, par).to(tl.int64))
    if rec:
        slot = tl.atomic_add(COUNT_ptr, 1)
        if slot < bucket:
            tl.store(ROWS_ptr + slot, pid.to(tl.int64))
        sslot = tl.atomic_add(SEGCNT_ptr + b, 1)
        if sslot < N:
            tl.store(SEG_ptr + b * N + sslot, pid.to(tl.int64))


@triton.jit
def specfold_flash_attn_kernel(
    QTAB_ptr, PK_ptr, PV_ptr, KV_ptr, SRC_ptr, ROWS_ptr, COUNT_ptr,
    OTAB_ptr,              # [B1*N, H*hd]  scattered full O table
    MTAB_ptr, ZTAB_ptr,    # [B1*N, H]     per-row per-head softmax state
    P, N, scale, bucket,
    GROUP: tl.constexpr, HKV: tl.constexpr, HD: tl.constexpr, BLOCK_J: tl.constexpr,
):
    """Recomputed positions' flash attention, outputs scattered to the
    full table plus (m, Z) retained for children's exact corrections."""
    mm = tl.program_id(0)
    h = tl.program_id(1)
    count = tl.minimum(tl.load(COUNT_ptr), bucket)
    if mm >= count:
        return
    row = tl.load(ROWS_ptr + mm)
    b = row // N
    kvh = h // GROUP
    NH: tl.constexpr = GROUP * HKV
    dofs = tl.arange(0, HD)
    q = tl.load(QTAB_ptr + row * (NH * HD) + h * HD + dofs).to(tl.float32)
    m_i = float("-inf")
    l_i = 0.0
    acc = tl.zeros((HD,), dtype=tl.float32)
    jofs = tl.arange(0, BLOCK_J)
    for j0 in range(0, P, BLOCK_J):
        jm = j0 + jofs < P
        k = tl.load(PK_ptr + kvh * P * HD + (j0 + jofs)[:, None] * HD + dofs[None, :],
                    mask=jm[:, None], other=0.0).to(tl.float32)
        s = tl.sum(q[None, :] * k, 1) * scale
        s = tl.where(jm, s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new)
        v = tl.load(PV_ptr + kvh * P * HD + (j0 + jofs)[:, None] * HD + dofs[None, :],
                    mask=jm[:, None], other=0.0).to(tl.float32)
        acc = acc * alpha + tl.sum(p[:, None] * v, 0)
        l_i = l_i * alpha + tl.sum(p)
        m_i = m_new
    jN = tl.arange(0, 32)
    jmask = jN < N
    kvrow = tl.load(SRC_ptr + b * N + jN, mask=jmask, other=0) * N + jN
    k = tl.load(KV_ptr + kvrow[:, None] * (2 * HKV * HD) + kvh * HD + dofs[None, :],
                mask=jmask[:, None], other=0.0).to(tl.float32)
    s = tl.sum(q[None, :] * k, 1) * scale
    s = tl.where(jmask, s, float("-inf"))
    m_new = tl.maximum(m_i, tl.max(s))
    alpha = tl.exp(m_i - m_new)
    p = tl.exp(s - m_new)
    v = tl.load(KV_ptr + kvrow[:, None] * (2 * HKV * HD) + (HKV + kvh) * HD + dofs[None, :],
                mask=jmask[:, None], other=0.0).to(tl.float32)
    acc = acc * alpha + tl.sum(p[:, None] * v, 0)
    l_i = l_i * alpha + tl.sum(p)
    m_i = m_new
    tl.store(OTAB_ptr + row * (NH * HD) + h * HD + dofs, (acc / l_i).to(OTAB_ptr.dtype.element_ty))
    tl.store(MTAB_ptr + row * NH + h, m_i)
    tl.store(ZTAB_ptr + row * NH + h, l_i)


@triton.jit
def specfold_flash_attn_gqa_kernel(
    QTAB_ptr, PK_ptr, PV_ptr, KV_ptr, SRC_ptr, ROWS_ptr, COUNT_ptr,
    OTAB_ptr, MTAB_ptr, ZTAB_ptr,
    P, N, scale, bucket,
    GROUP: tl.constexpr, HKV: tl.constexpr, HD: tl.constexpr, BLOCK_J: tl.constexpr,
    QP: tl.constexpr,      # q-head tile: max(16, next_pow2(GROUP)); >=16 for tl.dot
):
    """specfold_flash_attn_kernel with GQA k/v sharing: one program per (recomputed position, kv-head)
    computes all GROUP q-heads of that kv-head, so each K/V element is loaded
    once instead of GROUP times. Same math, same outputs."""
    mm = tl.program_id(0)
    kvh = tl.program_id(1)
    count = tl.minimum(tl.load(COUNT_ptr), bucket)
    if mm >= count:
        return
    row = tl.load(ROWS_ptr + mm)
    b = row // N
    NH: tl.constexpr = GROUP * HKV
    dofs = tl.arange(0, HD)
    gofs = tl.arange(0, QP)
    gm = gofs < GROUP
    q = tl.load(QTAB_ptr + row * (NH * HD) + (kvh * GROUP + gofs)[:, None] * HD + dofs[None, :],
                mask=gm[:, None], other=0.0).to(tl.float32)          # (QP, HD)
    m_i = tl.full((QP,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((QP,), dtype=tl.float32)
    acc = tl.zeros((QP, HD), dtype=tl.float32)
    jofs = tl.arange(0, BLOCK_J)
    for j0 in range(0, P, BLOCK_J):
        jm = j0 + jofs < P
        k = tl.load(PK_ptr + kvh * P * HD + (j0 + jofs)[:, None] * HD + dofs[None, :],
                    mask=jm[:, None], other=0.0)                      # (BLOCK_J, HD)
        s = tl.dot(q.to(k.dtype), tl.trans(k), out_dtype=tl.float32) * scale
        s = tl.where(jm[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        v = tl.load(PV_ptr + kvh * P * HD + (j0 + jofs)[:, None] * HD + dofs[None, :],
                    mask=jm[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v, out_dtype=tl.float32)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_new
    jN = tl.arange(0, 32)
    jmask = jN < N
    kvrow = tl.load(SRC_ptr + b * N + jN, mask=jmask, other=0) * N + jN
    k = tl.load(KV_ptr + kvrow[:, None] * (2 * HKV * HD) + kvh * HD + dofs[None, :],
                mask=jmask[:, None], other=0.0)                       # (32, HD)
    s = tl.dot(q.to(k.dtype), tl.trans(k), out_dtype=tl.float32) * scale
    s = tl.where(jmask[None, :], s, float("-inf"))
    m_new = tl.maximum(m_i, tl.max(s, 1))
    alpha = tl.exp(m_i - m_new)
    p = tl.exp(s - m_new[:, None])
    v = tl.load(KV_ptr + kvrow[:, None] * (2 * HKV * HD) + (HKV + kvh) * HD + dofs[None, :],
                mask=jmask[:, None], other=0.0)
    acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v, out_dtype=tl.float32)
    l_i = l_i * alpha + tl.sum(p, 1)
    m_i = m_new
    tl.store(OTAB_ptr + row * (NH * HD) + (kvh * GROUP + gofs)[:, None] * HD + dofs[None, :],
             (acc / l_i[:, None]).to(OTAB_ptr.dtype.element_ty), mask=gm[:, None])
    tl.store(MTAB_ptr + row * NH + kvh * GROUP + gofs, m_i, mask=gm)
    tl.store(ZTAB_ptr + row * NH + kvh * GROUP + gofs, l_i, mask=gm)


@triton.jit
def specfold_chain_correction_gqa_kernel(
    QTAB_ptr, KV_ptr, SRC_ptr, PAR_ptr, REC_ptr,
    SEG_ptr, SEGCNT_ptr,
    OTAB_ptr, MTAB_ptr, ZTAB_ptr,
    N, scale,
    GROUP: tl.constexpr, HKV: tl.constexpr, HD: tl.constexpr, MAXDEPTH: tl.constexpr,
    QP: tl.constexpr,      # next_pow2(GROUP)
):
    """Chain-walk correction with GQA k/v sharing: one program per (folded position,
    kv-head) processes all GROUP q-heads, loading each chain entry's k/v once
    instead of GROUP times. Same telescoping math as v2."""
    pid = tl.program_id(0)
    kvh = tl.program_id(1)
    if tl.load(REC_ptr + pid) != 0:
        return
    NH: tl.constexpr = GROUP * HKV
    n = pid % N
    dofs = tl.arange(0, HD)
    gofs = tl.arange(0, QP)
    gm = gofs < GROUP
    hh = kvh * GROUP + gofs
    srow = tl.load(SRC_ptr + pid)
    q = tl.load(QTAB_ptr + (srow * N + n) * (NH * HD) + hh[:, None] * HD + dofs[None, :],
                mask=gm[:, None], other=0.0).to(tl.float32)          # (QP, HD)
    sflat = srow * N + n
    m0 = tl.load(MTAB_ptr + sflat * NH + hh, mask=gm, other=0.0)
    Z0 = tl.load(ZTAB_ptr + sflat * NH + hh, mask=gm, other=1.0)
    O0 = tl.load(OTAB_ptr + sflat * (NH * HD) + hh[:, None] * HD + dofs[None, :],
                 mask=gm[:, None], other=0.0).to(tl.float32)

    m_new = m0
    cur = (pid // N).to(tl.int64)
    for _ in range(MAXDEPTH):
        if cur != srow:
            hi = tl.load(SEGCNT_ptr + cur)
            for j in range(0, hi):
                rowj = tl.load(SEG_ptr + cur * N + j)
                kn = tl.load(KV_ptr + rowj * (2 * HKV * HD) + kvh * HD + dofs).to(tl.float32)
                m_new = tl.maximum(m_new, tl.sum(q * kn[None, :], 1) * scale)
            cur = tl.load(PAR_ptr + cur)
    f = tl.exp(m0 - m_new)
    Z = Z0 * f
    num = O0 * Z[:, None]
    cur = (pid // N).to(tl.int64)
    for _ in range(MAXDEPTH):
        if cur != srow:
            pcur = tl.load(PAR_ptr + cur)
            hi = tl.load(SEGCNT_ptr + cur)
            for j in range(0, hi):
                rowj = tl.load(SEG_ptr + cur * N + j)
                e = rowj % N
                oldrow = tl.load(SRC_ptr + pcur * N + e) * N + e
                ko = tl.load(KV_ptr + oldrow * (2 * HKV * HD) + kvh * HD + dofs).to(tl.float32)
                vo = tl.load(KV_ptr + oldrow * (2 * HKV * HD) + (HKV + kvh) * HD + dofs).to(tl.float32)
                kn = tl.load(KV_ptr + rowj * (2 * HKV * HD) + kvh * HD + dofs).to(tl.float32)
                vn = tl.load(KV_ptr + rowj * (2 * HKV * HD) + (HKV + kvh) * HD + dofs).to(tl.float32)
                ao = tl.exp(tl.sum(q * ko[None, :], 1) * scale - m_new)
                an = tl.exp(tl.sum(q * kn[None, :], 1) * scale - m_new)
                Z = Z - ao + an
                num = num - ao[:, None] * vo[None, :] + an[:, None] * vn[None, :]
            cur = pcur
    tl.store(OTAB_ptr + pid * (NH * HD) + hh[:, None] * HD + dofs[None, :],
             (num / Z[:, None]).to(OTAB_ptr.dtype.element_ty), mask=gm[:, None])
    tl.store(MTAB_ptr + pid * NH + hh, m_new, mask=gm)
    tl.store(ZTAB_ptr + pid * NH + hh, Z, mask=gm)


@triton.jit
def specfold_chain_correction_v2_kernel(
    QTAB_ptr, KV_ptr, SRC_ptr, PAR_ptr, REC_ptr,
    SEG_ptr, SEGCNT_ptr,       # segmented per-branch lists (stride N) + counters
    OTAB_ptr, MTAB_ptr, ZTAB_ptr,
    N, scale,
    GROUP: tl.constexpr, HKV: tl.constexpr, HD: tl.constexpr, MAXDEPTH: tl.constexpr,
):
    """Chain-walk correction with per-branch recomputed-entry ranges: each hop loops
    only over its own branch's entries (no global scan)."""
    pid = tl.program_id(0)
    h = tl.program_id(1)
    if tl.load(REC_ptr + pid) != 0:
        return
    NH: tl.constexpr = GROUP * HKV
    branch = pid // N
    n = pid % N
    kvh = h // GROUP
    dofs = tl.arange(0, HD)
    srow = tl.load(SRC_ptr + pid)
    q = tl.load(QTAB_ptr + (srow * N + n) * (NH * HD) + h * HD + dofs).to(tl.float32)
    sflat = srow * N + n
    m0 = tl.load(MTAB_ptr + sflat * NH + h)
    Z0 = tl.load(ZTAB_ptr + sflat * NH + h)
    O0 = tl.load(OTAB_ptr + sflat * (NH * HD) + h * HD + dofs).to(tl.float32)

    m_new = m0
    cur = (pid // N).to(tl.int64)
    for _ in range(MAXDEPTH):
        if cur != srow:
            hi = tl.load(SEGCNT_ptr + cur)
            for j in range(0, hi):
                rowj = tl.load(SEG_ptr + cur * N + j)
                kn = tl.load(KV_ptr + rowj * (2 * HKV * HD) + kvh * HD + dofs).to(tl.float32)
                m_new = tl.maximum(m_new, tl.sum(q * kn) * scale)
            cur = tl.load(PAR_ptr + cur)
    f = tl.exp(m0 - m_new)
    Z = Z0 * f
    num = O0 * Z
    cur = (pid // N).to(tl.int64)
    for _ in range(MAXDEPTH):
        if cur != srow:
            pcur = tl.load(PAR_ptr + cur)
            hi = tl.load(SEGCNT_ptr + cur)
            for j in range(0, hi):
                rowj = tl.load(SEG_ptr + cur * N + j)
                e = rowj % N
                oldrow = tl.load(SRC_ptr + pcur * N + e) * N + e
                ko = tl.load(KV_ptr + oldrow * (2 * HKV * HD) + kvh * HD + dofs).to(tl.float32)
                vo = tl.load(KV_ptr + oldrow * (2 * HKV * HD) + (HKV + kvh) * HD + dofs).to(tl.float32)
                kn = tl.load(KV_ptr + rowj * (2 * HKV * HD) + kvh * HD + dofs).to(tl.float32)
                vn = tl.load(KV_ptr + rowj * (2 * HKV * HD) + (HKV + kvh) * HD + dofs).to(tl.float32)
                ao = tl.exp(tl.sum(q * ko) * scale - m_new)
                an = tl.exp(tl.sum(q * kn) * scale - m_new)
                Z = Z - ao + an
                num = num - ao * vo + an * vn
            cur = pcur
    tl.store(OTAB_ptr + pid * (NH * HD) + h * HD + dofs, (num / Z).to(OTAB_ptr.dtype.element_ty))
    tl.store(MTAB_ptr + pid * NH + h, m_new)
    tl.store(ZTAB_ptr + pid * NH + h, Z)
