// FLASH_ATTN_EXT for one query token against a long KV (decode), as four small passes:
//   scores:  s[h][j] = scale * q_h . k_j + mask[j]          (one sub-group per 16 keys)
//   softmax: e[h][j] = exp(s[h][j] - max_j s[h][j]), z[h]   (one work-group per query head)
//   values:  partial sums of e[h][j] * v_j over key ranges  (one sub-group per range and 16 dims)
//   out:     y[h] = sum of the partials / z[h]
// A key whose mask is -inf is never read: its weight is exactly 0. With the Mach-2 QSA mask
// (all but the top-k 4-token blocks are -inf once the context is longer than top-k blocks) that
// makes the cost nearly independent of the context length, where the tile kernel reads all of K
// and V. Scores and values are products on the matrix engine with the fp32 operand (q, the
// weights) split in fp16 hi and lo rows, as the walks do; K and V are fp16 as stored. Tolerance
// class.

#include "mach1.hpp"

#include <algorithm>
#include <cstring>
#include <vector>

#define MACH1_ATTN_RMAX 16     // query heads per key head, at most: 4 DPAS of 4 heads (hi and lo rows)
#define MACH1_ATTN_RSG  8      // key ranges per work-group in the value pass
#define MACH1_ATTN1_SEGS 1024  // 256-key segments the listed one-token path can take (one work-group)
// keys from which one token goes over the list of its visible keys: with 2,048 of them visible
// the list path is 25 us per layer whatever the KV, the passes above 22 us at 4K keys, 31 at 8K,
// 45 at 28K, 96 at 128K
#define MACH1_ATTN_LIST_MIN_KV 6144

struct mach1_attn_args {
    const float      * q;      // [D, 1, Hq]
    const sycl::half * k;      // [D, n_kv, Hk]
    const sycl::half * v;
    const sycl::half * mask;   // [n_kv]
    int64_t q_s2;              // floats per query head
    int64_t k_s1, k_s2;        // halfs per key and per key head
    int64_t v_s1, v_s2;
    int     D, n_kv, Hk, R;    // R query heads per key head
    float   scale;
};

// the A operand of one DPAS: 4 values as rows 0-3 (hi) and 4-7 (lo), x = hi + lo/MACH1_XMX1_LO
static inline mach1_s8v mach1_attn_rows(const float (&x)[MACH1_ATTN_RMAX], int i) {
    mach1_s8v a;
#pragma unroll
    for (int t = 0; t < 4; ++t) {
        const sycl::half hi = (sycl::half) x[4*i + t];
        a[t]     = sycl::bit_cast<short>(hi);
        a[4 + t] = sycl::bit_cast<short>((sycl::half) ((x[4*i + t] - (float) hi)*MACH1_XMX1_LO));
    }
    return a;
}
// Lane l is key j0 + l: its 16 dims of a step are its slice of B, and the step's 16 dims of the
// query heads (lane l holds dim l, one contiguous load per head) are A. NG DPAS of 4 heads each;
// the loop has no branches: a masked lane reads the block's first key instead of its own and its
// sums are not stored, and the heads past R repeat the last one.
template <int NG>
static void mach1_attn_scores(dpct::queue_ptr stream, const mach1_attn_args a, float * S) {
    constexpr int RM = MACH1_ATTN_RMAX;
    constexpr int WG = 128;
    constexpr int U  = 2;    // steps with their loads in flight together
    const int nblk = (a.n_kv + 15)/16;
    stream->parallel_for(
        sycl::nd_range<2>(sycl::range<2>(a.Hk, (size_t) ((nblk*16 + WG - 1)/WG)*WG), sycl::range<2>(1, WG)),
        [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
            const sycl::sub_group sg = it.get_sub_group();
            const int g    = (int) it.get_group(0);
            const int j    = (int) it.get_global_id(1);
            const int lane = j & 15;
            const float m  = j < a.n_kv ? (float) a.mask[j] : -INFINITY;
            const bool  on = m > -INFINITY;
            if (!sycl::any_of_group(sg, on)) {
                return;
            }
            const int jk = on ? j : j - lane;
            const sycl::uint4 * kp = (const sycl::uint4 *) (a.k + (int64_t) jk*a.k_s1 + (int64_t) g*a.k_s2);
            const float       * qp = a.q + (int64_t) (g*a.R)*a.q_s2 + lane;
            mach1_f8v acc[NG];
#pragma unroll
            for (int i = 0; i < NG; ++i) {
                acc[i] = 0.0f;
            }
            for (int d0 = 0; d0 < a.D; d0 += 16*U) {
                sycl::uint4 k0[U], k1[U];
                float       qv[U][RM];
#pragma unroll
                for (int u = 0; u < U; ++u) {
                    k0[u] = kp[(d0 + 16*u)/8];
                    k1[u] = kp[(d0 + 16*u)/8 + 1];
#pragma unroll
                    for (int h = 0; h < 4*NG; ++h) {
                        qv[u][h] = qp[(int64_t) sycl::min(h, a.R - 1)*a.q_s2 + d0 + 16*u]*MACH1_XMX1_US;
                    }
                }
#pragma unroll
                for (int u = 0; u < U; ++u) {
                    const mach1_i8v b = { (int) k0[u][0], (int) k0[u][1], (int) k0[u][2], (int) k0[u][3],
                                          (int) k1[u][0], (int) k1[u][1], (int) k1[u][2], (int) k1[u][3] };
#pragma unroll
                    for (int i = 0; i < NG; ++i) {
                        acc[i] = mach1_dpas(acc[i], mach1_attn_rows(qv[u], i), b);
                    }
                }
            }
            if (on) {
#pragma unroll
                for (int h = 0; h < 4*NG; ++h) {
                    if (h < a.R) {
                        S[(int64_t) (g*a.R + h)*a.n_kv + j] =
                            (acc[h/4][h%4] + acc[h/4][4 + h%4]*(1.0f/MACH1_XMX1_LO))*(a.scale*(1.0f/MACH1_XMX1_US)) + m;
                    }
                }
            }
        });
}

// in place: scores -> exp(score - max); masked keys (whose scores were never written) -> 0.
// A work-item takes 4 keys per step (n_kv is a multiple of 4).
static void mach1_attn_softmax(dpct::queue_ptr stream, const mach1_attn_args a, float * S, float * sums) {
    constexpr int WG = 512;
    const int n4 = a.n_kv/4;
    const sycl::vec<uint16_t, 4> * const mask4 = (const sycl::vec<uint16_t, 4> *) a.mask;
    stream->parallel_for(
        sycl::nd_range<2>(sycl::range<2>((size_t) a.Hk*a.R, WG), sycl::range<2>(1, WG)),
        [=](sycl::nd_item<2> it) {
            const int tid = (int) it.get_local_id(1);
            sycl::float4 * s = (sycl::float4 *) (S + (int64_t) it.get_group(0)*n4*4);
            // fp16 -inf is 0xFC00; NaN masks count as masked too
            const auto visible = [&](int j) {
                const sycl::vec<uint16_t, 4> mk = mask4[j];
                return sycl::int4(mk[0] != 0xFC00 && (mk[0] & 0x7FFF) <= 0x7C00, mk[1] != 0xFC00 && (mk[1] & 0x7FFF) <= 0x7C00,
                                  mk[2] != 0xFC00 && (mk[2] & 0x7FFF) <= 0x7C00, mk[3] != 0xFC00 && (mk[3] & 0x7FFF) <= 0x7C00);
            };
            float mx = -INFINITY;
            for (int j = tid; j < n4; j += WG) {
                const sycl::int4   vis = visible(j);
                const sycl::float4 v   = s[j];
#pragma unroll
                for (int t = 0; t < 4; ++t) {
                    mx = sycl::fmax(mx, vis[t] ? v[t] : -INFINITY);
                }
            }
            mx = sycl::reduce_over_group(it.get_group(), mx, sycl::maximum<float>());
            const bool any = mx > -INFINITY;
            float sum = 0.0f;
            for (int j = tid; j < n4; j += WG) {
                const sycl::int4   vis = visible(j);
                const sycl::float4 v   = s[j];
                sycl::float4 e;
#pragma unroll
                for (int t = 0; t < 4; ++t) {
                    // (the difference of an unwritten score is not used)
                    e[t] = any && vis[t] ? sycl::exp(v[t] - mx) : 0.0f;
                    sum += e[t];
                }
                s[j] = e;
            }
            sum = sycl::reduce_over_group(it.get_group(), sum, sycl::plus<float>());
            if (tid == 0) {
                sums[it.get_group(0)] = sum;
            }
        });
}

// A work-group is RSG key ranges of one key head and one chunk of 16 dims. Lane l is dim c*16 + l:
// the 16 keys of a step at that dim (16 contiguous loads, one per key) are its slice of B, and the
// keys' weights for the query heads (lane l holds key jb + l) are A. The loop has no branches
// past the test for a fully masked step: a masked key reads the step's first key instead and
// counts as 0. The ranges' sums meet in local memory and go out as one partial per work-group.
template <int NG>
static void mach1_attn_values(dpct::queue_ptr stream, const mach1_attn_args a, const float * P, float * part,
                              const int nwg, const int rkeys) {
    constexpr int RM  = MACH1_ATTN_RMAX;
    constexpr int RSG = MACH1_ATTN_RSG;
    const int nchunk = a.D/16;
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> sh_acc(sycl::range<1>(RSG*RM*16), cgh);
        cgh.parallel_for(
            sycl::nd_range<2>(sycl::range<2>((size_t) a.Hk*nchunk, (size_t) nwg*RSG*16), sycl::range<2>(1, RSG*16)),
            [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
                float * sh = sh_acc.get_multi_ptr<sycl::access::decorated::no>().get();
                const sycl::sub_group sg = it.get_sub_group();
                const int g    = (int) it.get_group(0) / nchunk;
                const int c    = (int) it.get_group(0) % nchunk;
                const int w    = (int) it.get_group(1);
                const int tid  = (int) it.get_local_id(1);
                const int lane = tid & 15;
                const int sgi  = tid >> 4;
                const int jbeg = (w*RSG + sgi)*rkeys;
                const int jend = sycl::min(jbeg + rkeys, a.n_kv);

                mach1_f8v acc[NG];
#pragma unroll
                for (int i = 0; i < NG; ++i) {
                    acc[i] = 0.0f;
                }
                const uint16_t * vg = (const uint16_t *) a.v + (int64_t) g*a.v_s2 + c*16 + lane;
                const float    * pg = P + (int64_t) (g*a.R)*a.n_kv;
                for (int jb = jbeg; jb < jend; jb += 16) {
                    // the 16 keys' mask values, one load: lane l takes key jb + l
                    const float mv = jb + lane < a.n_kv ? (float) a.mask[jb + lane] : -INFINITY;
                    const bool  on = mv > -INFINITY;
                    if (!sycl::any_of_group(sg, on)) {
                        continue;
                    }
                    // the softmax pass left 0 for the masked keys
                    const int jp = sycl::min(jb + lane, a.n_kv - 1);
                    float pv[RM];
#pragma unroll
                    for (int h = 0; h < 4*NG; ++h) {
                        pv[h] = pg[(int64_t) sycl::min(h, a.R - 1)*a.n_kv + jp];
                    }
                    uint32_t vk[16];
#pragma unroll
                    for (int t = 0; t < 16; ++t) {
                        const bool kon = sycl::select_from_group(sg, mv, t) > -INFINITY;
                        vk[t] = vg[(int64_t) (kon ? jb + t : jb)*a.v_s1];
                        vk[t] = kon ? vk[t] : 0u;
                    }
                    mach1_i8v b;
#pragma unroll
                    for (int t = 0; t < 8; ++t) {
                        b[t] = (int) (vk[2*t] | (vk[2*t + 1] << 16));
                    }
#pragma unroll
                    for (int i = 0; i < NG; ++i) {
                        acc[i] = mach1_dpas(acc[i], mach1_attn_rows(pv, i), b);
                    }
                }
#pragma unroll
                for (int h = 0; h < 4*NG; ++h) {
                    sh[(sgi*RM + h)*16 + lane] = acc[h/4][h%4] + acc[h/4][4 + h%4]*(1.0f/MACH1_XMX1_LO);
                }
                sycl::group_barrier(it.get_group());
                if (sgi == 0) {
                    float * pd = part + ((int64_t) (g*nwg + w)*a.R)*a.D + c*16 + lane;
#pragma unroll
                    for (int h = 0; h < 4*NG; ++h) {
                        if (h < a.R) {
                            float s = 0.0f;
#pragma unroll
                            for (int q = 0; q < RSG; ++q) {
                                s += sh[(q*RM + h)*16 + lane];
                            }
                            pd[(int64_t) h*a.D] = s;
                        }
                    }
                }
            });
    });
}

static void mach1_attn_out(dpct::queue_ptr stream, const mach1_attn_args a, const float * part, const float * sums,
                           float * dst, const int nwg) {
    constexpr int WG = 256;
    const int D = a.D;
    const int R = a.R;
    const int n = a.Hk*R*D;
    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) ((n + WG - 1)/WG)*WG), sycl::range<1>(WG)),
        [=](sycl::nd_item<1> it) {
            const int i = (int) it.get_global_id(0);
            if (i >= n) {
                return;
            }
            const int h = i / D;
            const int d = i % D;
            const int g = h / R;
            const int r = h % R;
            const float * p = part + ((int64_t) (g*nwg)*R + r)*D + d;
            float s = 0.0f;
            for (int w = 0; w < nwg; ++w) {
                s += p[(int64_t) w*R*D];
            }
            const float z = sums[h];
            dst[i] = z > 0.0f ? s/z : 0.0f;
        });
}

static void mach1_attn_one_listed(ggml_backend_sycl_context & ctx, const mach1_attn_args a, float * dst, int32_t * seen);

bool ggml_sycl_mach1_attn_one(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    static const bool on = getenv("GGML_SYCL_MACH1_ATTN_ONE") == nullptr || atoi(getenv("GGML_SYCL_MACH1_ATTN_ONE")) != 0;
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    float scale = 1.0f, max_bias = 0.0f, logit_softcap = 0.0f;
    memcpy(&scale,         (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));

    const int D = (int) K->ne[0];
    if (!on || !mach1_caps(ctx.stream()).xmx || mask == nullptr || sinks != nullptr || max_bias != 0.0f || logit_softcap != 0.0f ||
        Q->ne[1] != 1 || Q->ne[3] != 1 || K->ne[3] != 1 || V->ne[0] != D || Q->ne[0] != D ||
        D % 32 != 0 || D > 1024 ||
        Q->type != GGML_TYPE_F32 || K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16 ||
        mask->type != GGML_TYPE_F16 || dst->type != GGML_TYPE_F32 || mask->ne[0] != K->ne[1] ||
        Q->ne[2] % K->ne[2] != 0 || Q->ne[2]/K->ne[2] > MACH1_ATTN_RMAX || V->ne[1] != K->ne[1] || V->ne[2] != K->ne[2] ||
        Q->nb[0] != sizeof(float) || K->nb[0] != sizeof(sycl::half) || V->nb[0] != sizeof(sycl::half) ||
        mask->nb[0] != sizeof(sycl::half) || Q->nb[2] % sizeof(float) != 0 ||
        K->nb[1] % 4 != 0 || K->nb[2] % 4 != 0 || V->nb[1] % 4 != 0 || V->nb[2] % 4 != 0 ||
        (uintptr_t) K->data % 4 != 0 || (uintptr_t) V->data % 4 != 0 || !ggml_is_contiguous(dst) ||
        K->ne[1] % 16 != 0 || K->ne[1] > (1 << 20) || (uintptr_t) mask->data % 8 != 0) {
        return false;
    }
    mach1_attn_args a;
    a.q     = (const float *) Q->data;
    a.k     = (const sycl::half *) K->data;
    a.v     = (const sycl::half *) V->data;
    a.mask  = (const sycl::half *) mask->data;
    a.q_s2  = (int64_t) (Q->nb[2]/sizeof(float));
    a.k_s1  = (int64_t) (K->nb[1]/sizeof(sycl::half));
    a.k_s2  = (int64_t) (K->nb[2]/sizeof(sycl::half));
    a.v_s1  = (int64_t) (V->nb[1]/sizeof(sycl::half));
    a.v_s2  = (int64_t) (V->nb[2]/sizeof(sycl::half));
    a.D     = D;
    a.n_kv  = (int) K->ne[1];
    a.Hk    = (int) K->ne[2];
    a.R     = (int) (Q->ne[2]/K->ne[2]);
    a.scale = scale;

    // A long KV: over the list of the visible keys, so that nothing but the count scans the KV.
    // With most keys visible (a model without a sparse mask) the list only costs its own passes,
    // ~3%: the count of the last listed call, copied back without waiting for it, says so, and
    // then only every 256th call takes the list path, to look again.
    static const bool list_on = getenv("GGML_SYCL_MACH1_ATTN_LIST") == nullptr || atoi(getenv("GGML_SYCL_MACH1_ATTN_LIST")) != 0;
    static int32_t seen_cnt = -1;
    static int     seen_kv  = 0;
    static unsigned probe   = 0;
    if (list_on && a.n_kv >= MACH1_ATTN_LIST_MIN_KV && a.n_kv <= MACH1_ATTN1_SEGS*256) {
        const bool dense = seen_cnt >= 0 && (int64_t) seen_cnt*2 > seen_kv;
        if (!dense || (probe++ & 255u) == 0) {
            seen_kv = a.n_kv;
            mach1_attn_one_listed(ctx, a, (float *) dst->data, &seen_cnt);
            return true;
        }
    }

    // at most 64 key ranges for the value pass (one wave of sub-groups for 2 key heads of 256)
    const int rkeys  = std::max(16, ((a.n_kv + 63)/64 + 15)/16*16);
    const int ranges = (a.n_kv + rkeys - 1)/rkeys;
    const int nwg    = (ranges + MACH1_ATTN_RSG - 1)/MACH1_ATTN_RSG;

    dpct::queue_ptr stream = ctx.stream();
    // declared in allocation order (the VMM pool frees in stack order)
    ggml_sycl_pool_alloc<float> s_buf(ctx.pool(), (size_t) a.Hk*a.R*a.n_kv);
    ggml_sycl_pool_alloc<float> z_buf(ctx.pool(), (size_t) a.Hk*a.R);
    ggml_sycl_pool_alloc<float> p_buf(ctx.pool(), (size_t) a.Hk*nwg*a.R*D);

    switch ((a.R + 3)/4) {
        case 1:  mach1_attn_scores<1>(stream, a, s_buf.get()); break;
        case 2:  mach1_attn_scores<2>(stream, a, s_buf.get()); break;
        case 3:  mach1_attn_scores<3>(stream, a, s_buf.get()); break;
        default: mach1_attn_scores<4>(stream, a, s_buf.get()); break;
    }
    mach1_attn_softmax(stream, a, s_buf.get(), z_buf.get());
    switch ((a.R + 3)/4) {
        case 1:  mach1_attn_values<1>(stream, a, s_buf.get(), p_buf.get(), nwg, rkeys); break;
        case 2:  mach1_attn_values<2>(stream, a, s_buf.get(), p_buf.get(), nwg, rkeys); break;
        case 3:  mach1_attn_values<3>(stream, a, s_buf.get(), p_buf.get(), nwg, rkeys); break;
        default: mach1_attn_values<4>(stream, a, s_buf.get(), p_buf.get(), nwg, rkeys); break;
    }
    mach1_attn_out(stream, a, p_buf.get(), z_buf.get(), (float *) dst->data, nwg);
    return true;
}

//
// FLASH_ATTN_EXT for a batch of query tokens against a long KV when every token sees only a small
// part of it (Mach-2 QSA at prefill: 512 blocks of 4 keys per token, whatever the context). The
// tokens of a batch do not see the same keys - together they see nearly all of the KV - so the
// dense kernel cannot be given a smaller KV; each token gets the list of its own keys instead:
//   count:   visible keys per token and 256-key segment; read back on the host, which sizes the
//            lists (cap = the longest, a multiple of 32) and leaves batches that are not sparse
//            enough to the dense kernel
//   lists:   the visible keys of every token, ascending, by prefix sums within the sub-groups
//   scores:  s[h][p] = scale * q_h . k_{L[p]} + mask      (lane = key: its 16 dims are B, the query
//            heads' 16 dims, fp16, are A by one block read per 8 heads), left as
//            w[h][p] = exp(s - m_c) in fp16 with m_c the maximum of the chunk of 16 keys, plus
//            m_c and the chunk's sum of w: the scores themselves are never stored
//   norm:    f[h][c] = exp(m_c - max_c m_c), z[h] = sum_c f[h][c] * (the chunk's sum)
//   values:  y[h] = sum_c f[h][c] * sum_{p in c} w[h][p] v_{L[p]} / z[h]   (lane = 2 dims: 16 loads
//            of V per chunk, each contiguous, are B twice; the weights are A by one block read per
//            8 heads)
// Q and the weights go into the matrix engine as fp16, K and V as stored, the sums are fp32:
// the precision class of the dense oneDNN kernel this replaces (fp16 operands, fp16 output there).
// A chunk's largest weight is 1, so small weights keep their fp16 precision next to a large one
// elsewhere in the row.
//
#define MACH1_ATTNB_SEG    256     // keys per segment of the count and of the list pass
#define MACH1_ATTNB_MIN_KV 4096    // below this the dense kernel is not worth leaving
#define MACH1_ATTNB_MIN_Q  32      // query tokens: a batch

struct mach1_attnb_args {
    const float      * q;      // [D, nt, H]
    const sycl::half * k;      // [D, n_kv, Hk]
    const sycl::half * v;
    const sycl::half * mask;   // [n_kv, nt]
    int64_t q_s1, q_s2;        // floats per token and per query head
    int64_t k_s1, k_s2;        // halfs per key and per key head
    int64_t v_s1, v_s2;
    int     D, n_kv, Hk, R, nt, cap;
    float   scale;
};

static void mach1_attnb_count(dpct::queue_ptr stream, const sycl::half * mask, const int n_kv, const int nt, const int nseg,
                              int32_t * segcnt) {
    stream->parallel_for(
        sycl::nd_range<2>(sycl::range<2>((size_t) nt*nseg, 16), sycl::range<2>(1, 16)),
        [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
            const sycl::sub_group sg = it.get_sub_group();
            const int t    = (int) it.get_group(0) / nseg;
            const int seg  = (int) it.get_group(0) % nseg;
            const int lane = (int) it.get_local_id(1);
            const sycl::half * m = mask + (int64_t) t*n_kv;
            int c = 0;
#pragma unroll
            for (int ch = 0; ch < MACH1_ATTNB_SEG/16; ++ch) {
                const int key = seg*MACH1_ATTNB_SEG + ch*16 + lane;
                c += key < n_kv && (float) m[sycl::min(key, n_kv - 1)] > -INFINITY ? 1 : 0;
            }
            c = sycl::reduce_over_group(sg, c, sycl::plus<int>());
            if (lane == 0) {
                segcnt[it.get_group(0)] = c;
            }
        });
}

// segbase[t][seg] = the visible keys of token t before the segment
static void mach1_attnb_lists(dpct::queue_ptr stream, const sycl::half * mask, const int n_kv, const int nt, const int nseg,
                              const int cap, const int32_t * segbase, int32_t * L) {
    stream->parallel_for(
        sycl::nd_range<2>(sycl::range<2>((size_t) nt*nseg, 16), sycl::range<2>(1, 16)),
        [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
            const sycl::sub_group sg = it.get_sub_group();
            const int t    = (int) it.get_group(0) / nseg;
            const int seg  = (int) it.get_group(0) % nseg;
            const int lane = (int) it.get_local_id(1);
            const sycl::half * m = mask + (int64_t) t*n_kv;
            int32_t * l = L + (int64_t) t*cap;
            int off = segbase[it.get_group(0)];
#pragma unroll
            for (int ch = 0; ch < MACH1_ATTNB_SEG/16; ++ch) {
                const int key = seg*MACH1_ATTNB_SEG + ch*16 + lane;
                const int vis = key < n_kv && (float) m[sycl::min(key, n_kv - 1)] > -INFINITY ? 1 : 0;
                const int pc  = sycl::exclusive_scan_over_group(sg, vis, sycl::plus<int>());
                if (vis) {
                    l[off + pc] = key;
                }
                off += sycl::reduce_over_group(sg, vis, sycl::plus<int>());
            }
        });
}

// Q as fp16 rows of D, one per token and head (the A operand of the scores by block reads)
static void mach1_attnb_q16(dpct::queue_ptr stream, const mach1_attnb_args a, sycl::half * Qf) {
    const int H = a.Hk*a.R;
    stream->parallel_for(
        sycl::nd_range<2>(sycl::range<2>((size_t) a.nt*H, (size_t) a.D), sycl::range<2>(1, 16)),
        [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
            const int r = (int) it.get_group(0);
            const int i = (int) it.get_global_id(1);
            Qf[(int64_t) r*a.D + i] = (sycl::half) a.q[(int64_t) (r / H)*a.q_s1 + (int64_t) (r % H)*a.q_s2 + i];
        });
}

// MC and SC: [token and key head][chunk][16], lane = query head
template <int NG>
static void mach1_attnb_scores(dpct::queue_ptr stream, const mach1_attnb_args a, const int32_t * L, const int32_t * cnt,
                               const sycl::half * Qf, sycl::half * W, float * MC, float * SC) {
    constexpr int U = 2;    // steps of 16 dims with their loads in flight together
    const int rows = a.nt*a.Hk*a.R;
    stream->parallel_for(
        sycl::nd_range<2>(sycl::range<2>((size_t) a.nt*a.Hk, (size_t) a.cap), sycl::range<2>(1, 16)),
        [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
            const sycl::sub_group sg = it.get_sub_group();
            const int tg   = (int) it.get_group(0);
            const int t    = tg / a.Hk;
            const int g    = tg % a.Hk;
            const int p    = (int) it.get_global_id(1);
            const int lane = p & 15;
            const int j    = L[(int64_t) t*a.cap + p];
            const sycl::uint4 * kp = (const sycl::uint4 *) (a.k + (int64_t) j*a.k_s1 + (int64_t) g*a.k_s2);
            mach1_f8v acc[NG];
#pragma unroll
            for (int i = 0; i < NG; ++i) {
                acc[i] = 0.0f;
            }
            for (int d0 = 0; d0 < a.D; d0 += 16*U) {
                sycl::uint4 k0[U], k1[U];
                mach1_s8v   qa[U][NG];
#pragma unroll
                for (int u = 0; u < U; ++u) {
                    k0[u] = kp[(d0 + 16*u)/8];
                    k1[u] = kp[(d0 + 16*u)/8 + 1];
#pragma unroll
                    for (int i = 0; i < NG; ++i) {
                        qa[u][i] = mach1_load_a(Qf, 2*a.D, rows, d0 + 16*u, tg*a.R + 8*i);
                    }
                }
#pragma unroll
                for (int u = 0; u < U; ++u) {
                    const mach1_i8v b = { (int) k0[u][0], (int) k0[u][1], (int) k0[u][2], (int) k0[u][3],
                                          (int) k1[u][0], (int) k1[u][1], (int) k1[u][2], (int) k1[u][3] };
#pragma unroll
                    for (int i = 0; i < NG; ++i) {
                        acc[i] = mach1_dpas(acc[i], qa[u][i], b);
                    }
                }
            }
            // a position past the token's own keys holds key 0 and gets no weight
            const float m = p < cnt[t] ? (float) a.mask[(int64_t) t*a.n_kv + j] : -INFINITY;
            float mcv = -INFINITY, scv = 0.0f;
#pragma unroll
            for (int h = 0; h < 8*NG; ++h) {
                if (h < a.R) {
                    const float      sc = acc[h/8][h%8]*a.scale + m;
                    const float      mc = sycl::reduce_over_group(sg, sc, sycl::maximum<float>());
                    const sycl::half e  = (sycl::half) (mc > -INFINITY ? sycl::exp(sc - mc) : 0.0f);
                    W[(int64_t) (tg*a.R + h)*a.cap + p] = e;
                    // the sum is of the weights as the value pass will use them
                    const float sum = sycl::reduce_over_group(sg, (float) e, sycl::plus<float>());
                    mcv = lane == h ? mc  : mcv;
                    scv = lane == h ? sum : scv;
                }
            }
            const int64_t o = ((int64_t) tg*(a.cap/16) + p/16)*16 + lane;
            MC[o] = mcv;
            SC[o] = scv;
        });
}

// one sub-group per token and key head, lane = query head: MC becomes f in place
static void mach1_attnb_norm(dpct::queue_ptr stream, const mach1_attnb_args a, float * MC, const float * SC, float * z) {
    const int nch = a.cap/16;
    stream->parallel_for(
        sycl::nd_range<2>(sycl::range<2>((size_t) a.nt*a.Hk, 16), sycl::range<2>(1, 16)),
        [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
            const int tg   = (int) it.get_group(0);
            const int lane = (int) it.get_local_id(1);
            float       * mc = MC + (int64_t) tg*nch*16 + lane;
            const float * sc = SC + (int64_t) tg*nch*16 + lane;
            float mx = -INFINITY;
            for (int c = 0; c < nch; ++c) {
                mx = sycl::fmax(mx, mc[c*16]);
            }
            float sum = 0.0f;
            for (int c = 0; c < nch; ++c) {
                const float f = mx > -INFINITY ? sycl::exp(mc[c*16] - mx) : 0.0f;
                mc[c*16] = f;
                sum += f*sc[c*16];
            }
            if (lane < a.R) {
                z[tg*a.R + lane] = sum;
            }
        });
}

template <int NG>
static void mach1_attnb_values(dpct::queue_ptr stream, const mach1_attnb_args a, const int32_t * L, const sycl::half * W,
                               const float * F, const float * z, float * dst) {
    const int rows = a.nt*a.Hk*a.R;
    stream->parallel_for(
        sycl::nd_range<2>(sycl::range<2>((size_t) a.nt*a.Hk, (size_t) a.D/2), sycl::range<2>(1, 16)),
        [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
            const sycl::sub_group sg = it.get_sub_group();
            const int tg   = (int) it.get_group(0);
            const int t    = tg / a.Hk;
            const int g    = tg % a.Hk;
            const int i    = (int) it.get_global_id(1);     // dims 2i and 2i + 1
            const int lane = i & 15;
            const uint32_t * vg = (const uint32_t *) (a.v + (int64_t) g*a.v_s2) + i;
            const int32_t  * l  = L + (int64_t) t*a.cap + lane;
            const float    * fp = F + (int64_t) tg*(a.cap/16)*16 + lane;
            mach1_f8v acc[2*NG];
#pragma unroll
            for (int n = 0; n < 2*NG; ++n) {
                acc[n] = 0.0f;
            }
            for (int p0 = 0; p0 < a.cap; p0 += 16) {
                const int   jl = l[p0];
                const float fl = fp[p0];     // lane l: the chunk's factor for query head l
                uint32_t vk[16];
#pragma unroll
                for (int c = 0; c < 16; ++c) {
                    vk[c] = vg[(int64_t) sycl::select_from_group(sg, jl, c)*(a.v_s1/2)];
                }
                // B for the even and for the odd dims: rows are the keys, in pairs
                mach1_i8v be, bo;
#pragma unroll
                for (int c = 0; c < 8; ++c) {
                    be[c] = (int) ((vk[2*c] & 0xFFFFu) | (vk[2*c + 1] << 16));
                    bo[c] = (int) ((vk[2*c] >> 16) | (vk[2*c + 1] & 0xFFFF0000u));
                }
#pragma unroll
                for (int n = 0; n < NG; ++n) {
                    const mach1_s8v wa = mach1_load_a(W, 2*a.cap, rows, p0, tg*a.R + 8*n);
                    mach1_f8v te = 0.0f, to = 0.0f;
                    te = mach1_dpas(te, wa, be);
                    to = mach1_dpas(to, wa, bo);
#pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        const float f = sycl::select_from_group(sg, fl, 8*n + r);
                        acc[2*n][r]     += f*te[r];
                        acc[2*n + 1][r] += f*to[r];
                    }
                }
            }
#pragma unroll
            for (int h = 0; h < 8*NG; ++h) {
                if (h < a.R) {
                    const float zz  = z[tg*a.R + h];
                    const float inv = zz > 0.0f ? 1.0f/zz : 0.0f;
                    *(sycl::float2 *) (dst + (int64_t) (tg*a.R + h)*a.D + 2*i) =
                        sycl::float2(acc[2*(h/8)][h%8]*inv, acc[2*(h/8) + 1][h%8]*inv);
                }
            }
        });
}

bool ggml_sycl_mach1_attn_batch(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    static const bool on = getenv("GGML_SYCL_MACH1_ATTN_BATCH") == nullptr || atoi(getenv("GGML_SYCL_MACH1_ATTN_BATCH")) != 0;
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    float scale = 1.0f, max_bias = 0.0f, logit_softcap = 0.0f;
    memcpy(&scale,         (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));

    const int D = (int) K->ne[0];
    if (!on || mask == nullptr || sinks != nullptr || max_bias != 0.0f || logit_softcap != 0.0f ||
        Q->ne[1] < MACH1_ATTNB_MIN_Q || K->ne[1] < MACH1_ATTNB_MIN_KV || K->ne[1] > (1 << 20) ||
        Q->ne[3] != 1 || K->ne[3] != 1 || V->ne[0] != D || Q->ne[0] != D || D % 32 != 0 || D > 1024 ||
        Q->type != GGML_TYPE_F32 || K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16 ||
        mask->type != GGML_TYPE_F16 || dst->type != GGML_TYPE_F32 || mask->ne[0] != K->ne[1] || mask->ne[1] != Q->ne[1] ||
        mask->ne[2] != 1 || mask->ne[3] != 1 || !ggml_is_contiguous(mask) ||
        Q->ne[2] % K->ne[2] != 0 || Q->ne[2]/K->ne[2] > 16 || V->ne[1] != K->ne[1] || V->ne[2] != K->ne[2] ||
        Q->nb[0] != sizeof(float) || K->nb[0] != sizeof(sycl::half) || V->nb[0] != sizeof(sycl::half) ||
        Q->nb[1] % sizeof(float) != 0 || Q->nb[2] % sizeof(float) != 0 ||
        K->nb[1] % 4 != 0 || K->nb[2] % 4 != 0 || V->nb[1] % 4 != 0 || V->nb[2] % 4 != 0 ||
        (uintptr_t) K->data % 4 != 0 || (uintptr_t) V->data % 4 != 0 || !ggml_is_contiguous(dst) ||
        !mach1_caps(ctx.stream()).xmx) {
        return false;
    }
    // A model whose batches are not sparse would pay for the count and its read-back at every
    // attention: after a batch that went to the dense kernel the next ones are not looked at,
    // twice as many each time, at most 31.
    static int skip = 0, skip_next = 1;
    if (skip > 0) {
        skip--;
        return false;
    }

    mach1_attnb_args a;
    a.q     = (const float *) Q->data;
    a.k     = (const sycl::half *) K->data;
    a.v     = (const sycl::half *) V->data;
    a.mask  = (const sycl::half *) mask->data;
    a.q_s1  = (int64_t) (Q->nb[1]/sizeof(float));
    a.q_s2  = (int64_t) (Q->nb[2]/sizeof(float));
    a.k_s1  = (int64_t) (K->nb[1]/sizeof(sycl::half));
    a.k_s2  = (int64_t) (K->nb[2]/sizeof(sycl::half));
    a.v_s1  = (int64_t) (V->nb[1]/sizeof(sycl::half));
    a.v_s2  = (int64_t) (V->nb[2]/sizeof(sycl::half));
    a.D     = D;
    a.n_kv  = (int) K->ne[1];
    a.Hk    = (int) K->ne[2];
    a.R     = (int) (Q->ne[2]/K->ne[2]);
    a.nt    = (int) Q->ne[1];
    a.scale = scale;
    const int H    = a.Hk*a.R;
    const int nseg = (a.n_kv + MACH1_ATTNB_SEG - 1)/MACH1_ATTNB_SEG;

    dpct::queue_ptr stream = ctx.stream();
    // declared in allocation order (the VMM pool frees in stack order)
    ggml_sycl_pool_alloc<int32_t>    seg_buf(ctx.pool(), (size_t) a.nt*nseg);
    ggml_sycl_pool_alloc<int32_t>    cnt_buf(ctx.pool());
    ggml_sycl_pool_alloc<int32_t>    l_buf  (ctx.pool());
    ggml_sycl_pool_alloc<sycl::half> q_buf  (ctx.pool());
    ggml_sycl_pool_alloc<sycl::half> w_buf  (ctx.pool());
    ggml_sycl_pool_alloc<float>      mc_buf (ctx.pool());
    ggml_sycl_pool_alloc<float>      sc_buf (ctx.pool());
    ggml_sycl_pool_alloc<float>      z_buf  (ctx.pool());

    mach1_attnb_count(stream, a.mask, a.n_kv, a.nt, nseg, seg_buf.get());
    // the one host round trip: the counts decide the list length and whether this is worth it
    std::vector<int32_t> seg((size_t) a.nt*nseg), cnt((size_t) a.nt);
    stream->memcpy(seg.data(), seg_buf.get(), seg.size()*sizeof(int32_t)).wait();
    int maxc = 0;
    for (int t = 0; t < a.nt; ++t) {
        int acc = 0;
        for (int sgm = 0; sgm < nseg; ++sgm) {
            const int c = seg[(size_t) t*nseg + sgm];
            seg[(size_t) t*nseg + sgm] = acc;
            acc += c;
        }
        cnt[t] = acc;
        maxc   = std::max(maxc, acc);
    }
    a.cap = std::max(32, (maxc + 31) & ~31);
    // the dense kernel's time goes with n_kv, this one's with cap: about even at a quarter
    // (512 tokens, 2 x 12 heads of 256: 1.9 ms each at 8K keys and 2,080 per token)
    if ((int64_t) a.cap*4 > a.n_kv) {
        skip      = skip_next;
        skip_next = std::min(2*skip_next + 1, 31);
        return false;
    }
    skip_next = 1;

    cnt_buf.alloc((size_t) a.nt);
    l_buf.alloc((size_t) a.nt*a.cap);
    q_buf.alloc((size_t) a.nt*H*D);
    w_buf.alloc((size_t) a.nt*H*a.cap);
    mc_buf.alloc((size_t) a.nt*a.Hk*a.cap);
    sc_buf.alloc((size_t) a.nt*a.Hk*a.cap);
    z_buf.alloc((size_t) a.nt*H);

    stream->memcpy(seg_buf.get(), seg.data(), seg.size()*sizeof(int32_t)).wait();
    stream->memcpy(cnt_buf.get(), cnt.data(), cnt.size()*sizeof(int32_t)).wait();
    stream->fill(l_buf.get(), (int32_t) 0, (size_t) a.nt*a.cap);
    mach1_attnb_lists(stream, a.mask, a.n_kv, a.nt, nseg, a.cap, seg_buf.get(), l_buf.get());
    mach1_attnb_q16(stream, a, q_buf.get());
    if (a.R <= 8) {
        mach1_attnb_scores<1>(stream, a, l_buf.get(), cnt_buf.get(), q_buf.get(), w_buf.get(), mc_buf.get(), sc_buf.get());
    } else {
        mach1_attnb_scores<2>(stream, a, l_buf.get(), cnt_buf.get(), q_buf.get(), w_buf.get(), mc_buf.get(), sc_buf.get());
    }
    mach1_attnb_norm(stream, a, mc_buf.get(), sc_buf.get(), z_buf.get());
    if (a.R <= 8) {
        mach1_attnb_values<1>(stream, a, l_buf.get(), w_buf.get(), mc_buf.get(), z_buf.get(), (float *) dst->data);
    } else {
        mach1_attnb_values<2>(stream, a, l_buf.get(), w_buf.get(), mc_buf.get(), z_buf.get(), (float *) dst->data);
    }
    return true;
}

//
// One query token against a long KV, over the list of its visible keys. The four passes above
// skip masked keys but still look at every key's mask (softmax) and at every 16-key step (values):
// at 130K of context that was 2 ms per token. Here the keys are counted and listed first (the
// count and list passes of the batch path, with the prefix sums on the device: for one token the
// list can be as long as the KV, so nothing has to be read back), and scores, softmax and values
// run over the list positions. The arithmetic per key is that of the passes above.
//

// seg[i]: visible keys in segment i -> those before it; cnt[0] = all of them
static void mach1_attn1_prefix(dpct::queue_ptr stream, int32_t * seg, const int nseg, int32_t * cnt) {
    constexpr int WG  = MACH1_ATTN1_SEGS;
    constexpr int NSG = WG/16;
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<int, 1> tot_acc(sycl::range<1>(NSG), cgh);
        cgh.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(WG), sycl::range<1>(WG)),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                int * tot = tot_acc.get_multi_ptr<sycl::access::decorated::no>().get();
                const sycl::sub_group sg = it.get_sub_group();
                const int tid  = (int) it.get_local_id(0);
                const int lane = tid & 15;
                const int sgi  = tid >> 4;
                const int c    = tid < nseg ? seg[sycl::min(tid, nseg - 1)] : 0;
                const int pc   = sycl::exclusive_scan_over_group(sg, c, sycl::plus<int>());
                const int tc   = sycl::reduce_over_group(sg, c, sycl::plus<int>());
                if (lane == 0) {
                    tot[sgi] = tc;
                }
                sycl::group_barrier(it.get_group());
                int before = 0, all = 0;
#pragma unroll
                for (int k = 0; k < NSG/16; ++k) {
                    const int i = k*16 + lane;
                    const int v = tot[i];
                    before += i < sgi ? v : 0;
                    all    += v;
                }
                before = sycl::reduce_over_group(sg, before, sycl::plus<int>());
                all    = sycl::reduce_over_group(sg, all, sycl::plus<int>());
                if (tid < nseg) {
                    seg[tid] = before + pc;
                }
                if (tid == 0) {
                    cnt[0] = all;
                }
            });
    });
}

// as mach1_attn_scores, lane = list position p; S[h][p], -inf past the count
template <int NG>
static void mach1_attn1_scores(dpct::queue_ptr stream, const mach1_attn_args a, const int32_t * L, const int32_t * cnt, float * S) {
    constexpr int RM = MACH1_ATTN_RMAX;
    constexpr int WG = 128;
    constexpr int U  = 2;
    const int nblk = (a.n_kv + 15)/16;
    stream->parallel_for(
        sycl::nd_range<2>(sycl::range<2>(a.Hk, (size_t) ((nblk*16 + WG - 1)/WG)*WG), sycl::range<2>(1, WG)),
        [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
            const int g    = (int) it.get_group(0);
            const int p    = (int) it.get_global_id(1);
            const int lane = p & 15;
            const int n    = cnt[0];
            if (p - lane >= n) {
                return;
            }
            const bool on = p < n;
            const int  j  = on ? L[sycl::min(p, a.n_kv - 1)] : 0;
            const float m = on ? (float) a.mask[j] : -INFINITY;
            const sycl::uint4 * kp = (const sycl::uint4 *) (a.k + (int64_t) j*a.k_s1 + (int64_t) g*a.k_s2);
            const float       * qp = a.q + (int64_t) (g*a.R)*a.q_s2 + lane;
            mach1_f8v acc[NG];
#pragma unroll
            for (int i = 0; i < NG; ++i) {
                acc[i] = 0.0f;
            }
            for (int d0 = 0; d0 < a.D; d0 += 16*U) {
                sycl::uint4 k0[U], k1[U];
                float       qv[U][RM];
#pragma unroll
                for (int u = 0; u < U; ++u) {
                    k0[u] = kp[(d0 + 16*u)/8];
                    k1[u] = kp[(d0 + 16*u)/8 + 1];
#pragma unroll
                    for (int h = 0; h < 4*NG; ++h) {
                        qv[u][h] = qp[(int64_t) sycl::min(h, a.R - 1)*a.q_s2 + d0 + 16*u]*MACH1_XMX1_US;
                    }
                }
#pragma unroll
                for (int u = 0; u < U; ++u) {
                    const mach1_i8v b = { (int) k0[u][0], (int) k0[u][1], (int) k0[u][2], (int) k0[u][3],
                                          (int) k1[u][0], (int) k1[u][1], (int) k1[u][2], (int) k1[u][3] };
#pragma unroll
                    for (int i = 0; i < NG; ++i) {
                        acc[i] = mach1_dpas(acc[i], mach1_attn_rows(qv[u], i), b);
                    }
                }
            }
#pragma unroll
            for (int h = 0; h < 4*NG; ++h) {
                if (h < a.R) {
                    S[(int64_t) (g*a.R + h)*a.n_kv + p] =
                        (acc[h/4][h%4] + acc[h/4][4 + h%4]*(1.0f/MACH1_XMX1_LO))*(a.scale*(1.0f/MACH1_XMX1_US)) + m;
                }
            }
        });
}

// in place over the listed positions (whole chunks of 16: the scores past the count are -inf)
static void mach1_attn1_softmax(dpct::queue_ptr stream, const mach1_attn_args a, const int32_t * cnt, float * S, float * sums) {
    constexpr int WG = 512;
    stream->parallel_for(
        sycl::nd_range<2>(sycl::range<2>((size_t) a.Hk*a.R, WG), sycl::range<2>(1, WG)),
        [=](sycl::nd_item<2> it) {
            const int tid = (int) it.get_local_id(1);
            const int n4  = ((cnt[0] + 15)/16)*4;
            sycl::float4 * s = (sycl::float4 *) (S + (int64_t) it.get_group(0)*a.n_kv);
            float mx = -INFINITY;
            for (int j = tid; j < n4; j += WG) {
                const sycl::float4 v = s[j];
#pragma unroll
                for (int t = 0; t < 4; ++t) {
                    mx = sycl::fmax(mx, v[t]);
                }
            }
            mx = sycl::reduce_over_group(it.get_group(), mx, sycl::maximum<float>());
            const bool any = mx > -INFINITY;
            float sum = 0.0f;
            for (int j = tid; j < n4; j += WG) {
                const sycl::float4 v = s[j];
                sycl::float4 e;
#pragma unroll
                for (int t = 0; t < 4; ++t) {
                    e[t] = any && v[t] > -INFINITY ? sycl::exp(v[t] - mx) : 0.0f;
                    sum += e[t];
                }
                s[j] = e;
            }
            sum = sycl::reduce_over_group(it.get_group(), sum, sycl::plus<float>());
            if (tid == 0) {
                sums[it.get_group(0)] = sum;
            }
        });
}

// as mach1_attn_values over the listed positions: nwg*RSG ranges share them evenly
template <int NG>
static void mach1_attn1_values(dpct::queue_ptr stream, const mach1_attn_args a, const int32_t * L, const int32_t * cnt,
                               const float * P, float * part, const int nwg) {
    constexpr int RM  = MACH1_ATTN_RMAX;
    constexpr int RSG = MACH1_ATTN_RSG;
    const int nchunk = a.D/16;
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> sh_acc(sycl::range<1>(RSG*RM*16), cgh);
        cgh.parallel_for(
            sycl::nd_range<2>(sycl::range<2>((size_t) a.Hk*nchunk, (size_t) nwg*RSG*16), sycl::range<2>(1, RSG*16)),
            [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
                float * sh = sh_acc.get_multi_ptr<sycl::access::decorated::no>().get();
                const sycl::sub_group sg = it.get_sub_group();
                const int g    = (int) it.get_group(0) / nchunk;
                const int c    = (int) it.get_group(0) % nchunk;
                const int w    = (int) it.get_group(1);
                const int tid  = (int) it.get_local_id(1);
                const int lane = tid & 15;
                const int sgi  = sycl::group_broadcast(sg, tid >> 4);
                const int n    = cnt[0];
                const int n16  = (n + 15)/16*16;
                const int rk   = sycl::max(16, ((n16 + nwg*RSG - 1)/(nwg*RSG) + 15)/16*16);
                const int jbeg = (w*RSG + sgi)*rk;
                const int jend = sycl::min(jbeg + rk, n16);

                mach1_f8v acc[NG];
#pragma unroll
                for (int i = 0; i < NG; ++i) {
                    acc[i] = 0.0f;
                }
                const uint16_t * vg = (const uint16_t *) a.v + (int64_t) g*a.v_s2 + c*16 + lane;
                const float    * pg = P + (int64_t) (g*a.R)*a.n_kv;
                for (int jb = jbeg; jb < jend; jb += 16) {
                    // the 16 keys of the step, one load: lane l takes position jb + l
                    const int jl = jb + lane < n ? L[jb + lane] : 0;
                    float pv[RM];
#pragma unroll
                    for (int h = 0; h < 4*NG; ++h) {
                        pv[h] = pg[(int64_t) sycl::min(h, a.R - 1)*a.n_kv + jb + lane];
                    }
                    uint32_t vk[16];
#pragma unroll
                    for (int t = 0; t < 16; ++t) {
                        vk[t] = vg[(int64_t) sycl::select_from_group(sg, jl, t)*a.v_s1];
                    }
                    mach1_i8v b;
#pragma unroll
                    for (int t = 0; t < 8; ++t) {
                        b[t] = (int) (vk[2*t] | (vk[2*t + 1] << 16));
                    }
#pragma unroll
                    for (int i = 0; i < NG; ++i) {
                        acc[i] = mach1_dpas(acc[i], mach1_attn_rows(pv, i), b);
                    }
                }
#pragma unroll
                for (int h = 0; h < 4*NG; ++h) {
                    sh[(sgi*RM + h)*16 + lane] = acc[h/4][h%4] + acc[h/4][4 + h%4]*(1.0f/MACH1_XMX1_LO);
                }
                sycl::group_barrier(it.get_group());
                if (sgi == 0) {
                    float * pd = part + ((int64_t) (g*nwg + w)*a.R)*a.D + c*16 + lane;
#pragma unroll
                    for (int h = 0; h < 4*NG; ++h) {
                        if (h < a.R) {
                            float s = 0.0f;
#pragma unroll
                            for (int q = 0; q < RSG; ++q) {
                                s += sh[(q*RM + h)*16 + lane];
                            }
                            pd[(int64_t) h*a.D] = s;
                        }
                    }
                }
            });
    });
}

// *seen (host memory that outlives the call) gets the count of visible keys, some time later
static void mach1_attn_one_listed(ggml_backend_sycl_context & ctx, const mach1_attn_args a, float * dst, int32_t * seen) {
    constexpr int nwg = 64/MACH1_ATTN_RSG;   // 64 ranges: one wave of sub-groups for 2 key heads of 256
    const int nseg = (a.n_kv + MACH1_ATTNB_SEG - 1)/MACH1_ATTNB_SEG;
    dpct::queue_ptr stream = ctx.stream();
    // declared in allocation order (the VMM pool frees in stack order)
    ggml_sycl_pool_alloc<int32_t> seg_buf(ctx.pool(), (size_t) nseg);
    ggml_sycl_pool_alloc<int32_t> cnt_buf(ctx.pool(), 16);
    ggml_sycl_pool_alloc<int32_t> l_buf  (ctx.pool(), (size_t) a.n_kv + 16);
    ggml_sycl_pool_alloc<float>   s_buf  (ctx.pool(), (size_t) a.Hk*a.R*a.n_kv);
    ggml_sycl_pool_alloc<float>   z_buf  (ctx.pool(), (size_t) a.Hk*a.R);
    ggml_sycl_pool_alloc<float>   p_buf  (ctx.pool(), (size_t) a.Hk*nwg*a.R*a.D);

    mach1_attnb_count(stream, a.mask, a.n_kv, 1, nseg, seg_buf.get());
    mach1_attn1_prefix(stream, seg_buf.get(), nseg, cnt_buf.get());
    stream->memcpy(seen, cnt_buf.get(), sizeof(int32_t));
    mach1_attnb_lists(stream, a.mask, a.n_kv, 1, nseg, a.n_kv, seg_buf.get(), l_buf.get());
    switch ((a.R + 3)/4) {
        case 1:  mach1_attn1_scores<1>(stream, a, l_buf.get(), cnt_buf.get(), s_buf.get()); break;
        case 2:  mach1_attn1_scores<2>(stream, a, l_buf.get(), cnt_buf.get(), s_buf.get()); break;
        case 3:  mach1_attn1_scores<3>(stream, a, l_buf.get(), cnt_buf.get(), s_buf.get()); break;
        default: mach1_attn1_scores<4>(stream, a, l_buf.get(), cnt_buf.get(), s_buf.get()); break;
    }
    mach1_attn1_softmax(stream, a, cnt_buf.get(), s_buf.get(), z_buf.get());
    switch ((a.R + 3)/4) {
        case 1:  mach1_attn1_values<1>(stream, a, l_buf.get(), cnt_buf.get(), s_buf.get(), p_buf.get(), nwg); break;
        case 2:  mach1_attn1_values<2>(stream, a, l_buf.get(), cnt_buf.get(), s_buf.get(), p_buf.get(), nwg); break;
        case 3:  mach1_attn1_values<3>(stream, a, l_buf.get(), cnt_buf.get(), s_buf.get(), p_buf.get(), nwg); break;
        default: mach1_attn1_values<4>(stream, a, l_buf.get(), cnt_buf.get(), s_buf.get(), p_buf.get(), nwg); break;
    }
    mach1_attn_out(stream, a, p_buf.get(), z_buf.get(), dst, nwg);
}
