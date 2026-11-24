// Flash attention of a prompt over a long F16 cache with the relative-position bias, on Xe2 XMX (ESIMD DPAS).
// A thread holds 16 query rows (16 tokens of one query head) and streams the keys of its KV head 16 at a time.
// The scores stay in registers, so no score matrix goes through memory (the MKL route writes and reads it twice).
#include <sycl/sycl.hpp>

#include "common.hpp"
#include "fattn.hpp"
#include "fattn-xmx.hpp"

// GGML_SYCL_SUPPORT_XMX=OFF builds and sub-group 8 AOT targets skip the kernel, as for fattn-dpas.cpp
#if !defined(GGML_SYCL_SUPPORT_XMX) || (defined(GGML_SYCL_XMX_AOT_SG) && GGML_SYCL_XMX_AOT_SG != 16)
#define GGML_SYCL_FA_XMX_BUILD 0
#else
#define GGML_SYCL_FA_XMX_BUILD 1
#endif

#if GGML_SYCL_FA_XMX_BUILD
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

namespace fattn_xmx {
constexpr int   D     = 128;       // head size
constexpr int   MB    = 2;         // 8-row DPAS blocks per thread: one K / V tile load feeds both
constexpr int   RPT   = MB * 8;    // query rows per thread
constexpr int   KT    = 16;        // keys per step
constexpr int   NC    = D / 16;    // 16-wide pieces of a row
constexpr int   WGT   = 16;        // threads per work-group, all of one KV head: they read the same K / V tiles
constexpr float LOG2E = 1.4426950408889634f;
constexpr float NEG   = -1.0e30f;  // start of the running max
}  // namespace fattn_xmx

static bool fattn_xmx_support_arch(gpu_arch arch) {
    return arch == gpu_arch::intel_gpu_bmg_g21 || arch == gpu_arch::intel_gpu_bmg_g31 ||
           arch == gpu_arch::intel_gpu_lnl_m   || arch == gpu_arch::intel_gpu_ptl_h   ||
           arch == gpu_arch::intel_gpu_ptl_u   || arch == gpu_arch::intel_gpu_wcl;
}
#endif  // GGML_SYCL_FA_XMX_BUILD

bool ggml_sycl_fattn_xmx_supported(int device, const ggml_tensor * dst) {
#if !GGML_SYCL_FA_XMX_BUILD
    GGML_UNUSED(device);
    GGML_UNUSED(dst);
    return false;
#else
    using namespace fattn_xmx;
    if (!fattn_xmx_support_arch(ggml_sycl_info().devices[device].hw_info.arch)) {
        return false;
    }
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    float logit_softcap;
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    // 2D block loads: 64-byte aligned surfaces, pitches a multiple of 16 bytes
    const auto surf_ok = [](const void * p, size_t pitch) { return (uintptr_t) p % 64 == 0 && pitch % 16 == 0; };
    return K->type == GGML_TYPE_F16 && V->type == GGML_TYPE_F16 && Q->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
           Q->ne[0] == D && K->ne[0] == D && V->ne[0] == D && Q->ne[3] == 1 && K->ne[3] == 1 && V->ne[3] == 1 &&
           K->nb[0] == 2 && V->nb[0] == 2 && Q->ne[2] % K->ne[2] == 0 && logit_softcap == 0.0f && mask &&
           mask->type == GGML_TYPE_F16 && mask->ne[2] == 1 && mask->ne[3] == 1 && mask->nb[0] == 2 &&
           surf_ok(K->data, K->nb[1]) && K->nb[2] % 64 == 0 && surf_ok(V->data, V->nb[1]) && V->nb[2] % 64 == 0 &&
           surf_ok(mask->data, mask->nb[1]) && K->ne[1] >= 32 && ggml_is_contiguous(dst);
#endif
}

void ggml_sycl_fattn_xmx(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
#if !GGML_SYCL_FA_XMX_BUILD
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
    GGML_ABORT("XMX flash attention is not built");
#else
    using namespace fattn_xmx;
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];
    const ggml_sycl_fattn_rel_bias rel = ggml_sycl_fattn_get_rel_bias(dst);

    float scale;
    memcpy(&scale, dst->op_params, sizeof(float));

    const int n_queries = Q->ne[1], n_q_heads = Q->ne[2], n_kv_heads = K->ne[2], n_kv = K->ne[1];
    const int gqa = n_q_heads / n_kv_heads;
    dpct::queue_ptr stream = ctx.stream();

    // Q to F16, scaled to base 2, [head][query][D]
    ggml_sycl_pool_alloc<sycl::half> qh(ctx.pool(), (size_t) n_q_heads * n_queries * D);
    {
        sycl::half *  qd     = qh.get();
        const float * qs     = (const float *) Q->data;
        const int64_t s1     = Q->nb[1] / 4, s2 = Q->nb[2] / 4;
        const int64_t n      = (int64_t) n_q_heads * n_queries * D;
        const float   qscale = scale * LOG2E;
        stream->parallel_for(sycl::nd_range<1>((n + 255) / 256 * 256, 256), [=](sycl::nd_item<1> it) {
            const int64_t e = it.get_global_id(0);
            if (e >= n) {
                return;
            }
            const int64_t d = e % D, q = (e / D) % n_queries, h = e / ((int64_t) D * n_queries);
            qd[e] = sycl::half(qs[d + q * s1 + h * s2] * qscale);
        });
    }

    const sycl::half * qp      = qh.get();
    const char *       kp      = (const char *) K->data;
    const char *       vp      = (const char *) V->data;
    const sycl::half * mp      = (const sycl::half *) mask->data;
    const float *      sk      = sinks ? (const float *) sinks->data : nullptr;
    float *            dp      = (float *) dst->data;
    const uint32_t     k_pitch = K->nb[1], v_pitch = V->nb[1], m_pitch = mask->nb[1];
    const size_t       k_head  = K->nb[2], v_head = V->nb[2];
    const uint32_t     m_rows  = mask->ne[1];
    const int          qtiles  = (n_queries + RPT - 1) / RPT;  // per query head
    const int          tiles   = gqa * qtiles;                 // per KV head

    // Split the keys when the threads (each a long loop over its keys) are too few to fill the GPU in even waves: about
    // 1024 fit at once with 256 registers, so 1536 threads ran as one and a half waves. Each split writes its rows
    // unnormalized with their max and sum, a second kernel combines them. At least 4096 keys per split (shorter caches lost 1%).
    static const int split_env = ggml_sycl_get_env("GGML_SYCL_FA_XMX_SPLIT", 0);
    const int64_t    thr_1     = (int64_t) n_kv_heads * tiles;
    const int        ktiles    = (n_kv + KT - 1) / KT;
    int              nsplit    = split_env > 0 ? split_env : (int) std::min<int64_t>(8, (8192 + thr_1 - 1) / thr_1);
    nsplit                     = std::max(1, std::min(nsplit, n_kv / 4096));
    const int        kt_split  = (ktiles + nsplit - 1) / nsplit;  // key tiles per split
    nsplit                     = (ktiles + kt_split - 1) / kt_split;
    const size_t     n_thr     = (size_t) thr_1 * nsplit;
    const int        q_pad     = qtiles * RPT;
    ggml_sycl_pool_alloc<float> part_buf(ctx.pool());
    float *          part      = nsplit > 1 ? part_buf.alloc((size_t) nsplit * n_q_heads * q_pad * (D + 2)) : nullptr;

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<1>((n_thr + WGT - 1) / WGT * WGT, WGT),
            sycl::ext::oneapi::experimental::properties{ sycl::ext::intel::experimental::grf_size<256> },
            [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
                using namespace sycl::ext::intel::esimd;
                using half = sycl::half;
                const int tid = it.get_global_id(0);
                const int kvh = tid / (tiles * nsplit);
                if (kvh >= n_kv_heads) {
                    return;
                }
                const int sp = (tid / tiles) % nsplit;  // the 16 threads of a work-group read the same keys
                const int t  = tid % tiles;
                const int h  = kvh * gqa + t / qtiles;  // query head
                const int q0 = (t % qtiles) * RPT;      // first token

                // the DPAS A pieces of Q come from L1 every step: in registers they took 64 of the 256 and left too few for the rest
                // rows past the tokens read zero and are not stored
                const half * qhead = qp + (size_t) h * n_queries * D;

                // the bias table of this query head
                const char * tab0 = rel.table ? rel.table + (int64_t) h * rel.tb1 : nullptr;

                simd<float, MB * NC * 8 * 16> O = 0.0f;  // [mb][c][8 rows][16]
                float mref[RPT], lsum[RPT];
#pragma unroll
                for (int R = 0; R < RPT; ++R) {
                    mref[R] = NEG;
                    lsum[R] = 0.0f;
                }
                const uint32_t * kb = (const uint32_t *) (kp + kvh * k_head);
                const half *     vb = (const half *) (vp + kvh * v_head);

                const int k_end = std::min(n_kv, (sp + 1) * kt_split * KT);
                for (int k0 = sp * kt_split * KT; k0 < k_end; k0 += KT) {
                    // S = Q K^T, [mb][8 rows][16 keys]
                    simd<float, MB * 8 * KT> S = 0.0f;
#pragma unroll
                    for (int c = 0; c < NC; ++c) {
                        simd<uint32_t, 8 * KT> kt = load_2d<uint32_t, 8, KT, 1, true, false>(kb, D * 2 - 1, n_kv - 1, k_pitch - 1, c * 8, k0);
#pragma unroll
                        for (int mb = 0; mb < MB; ++mb) {
                            S.template select<8 * KT, 1>(mb * 8 * KT) = xmx::dpas<8, 8, float, float, half, half>(
                                simd<float, 8 * KT>(S.template select<8 * KT, 1>(mb * 8 * KT)), kt.template bit_cast_view<half>().read(),
                                load_2d<half, 16, 8, 1, false, false>(qhead, D * 2 - 1, n_queries - 1, D * 2 - 1, c * 16, q0 + mb * 8));
                        }
                    }

                    // mask: the tokens' rows of this key tile, [16 rows][16 keys] like S; keys past the cache are masked here
                    simd<half, RPT * KT> mt = load_2d<half, KT, RPT, 1, false, false>(mp, n_kv * 2 - 1, m_rows - 1, m_pitch - 1, k0, q0);
                    S += simd<float, RPT * KT>(mt) * LOG2E;
                    if (k0 + KT > n_kv) {
                        simd_mask<KT> out = simd<int32_t, KT>(0, 1) >= n_kv - k0;
#pragma unroll
                        for (int R = 0; R < RPT; ++R) {
                            simd<float, KT> s = S.template select<KT, 1>(R * KT);
                            s.merge(simd<float, KT>(NEG * 2.0f), out);
                            S.template select<KT, 1>(R * KT) = s;
                        }
                    }

                    if (rel.table) {
                        // each row reads the table entries its index row names for this key tile; keys past the cache take entry 0
#pragma unroll
                        for (int R = 0; R < RPT; ++R) {
                            const int       q    = q0 + R < n_queries ? q0 + R : n_queries - 1;
                            const int32_t * irow = rel.idx + (int64_t) q * rel.ib1 + k0;
                            simd<int32_t, KT> ix;
                            if (k0 + KT <= n_kv) {
                                ix = block_load<int32_t, KT>(irow);
                            } else {
                                simd_mask<KT> in = simd<int32_t, KT>(0, 1) < n_kv - k0;
                                ix = gather<int32_t, KT>(irow, simd<uint32_t, KT>(0, 4), in);
                                ix.merge(simd<int32_t, KT>(0), !in);
                            }
                            simd<float, KT> b = gather<float, KT>((const float *) (tab0 + (int64_t) q * rel.tb2), simd<uint32_t, KT>(ix) * 4u);
                            S.template select<KT, 1>(R * KT) += b * LOG2E;
                        }
                    }

                    // online softmax in base 2: a row's max and the rescale only run when a score passes the running max
                    // a lazier max (moved only 8 past it) measured the same speed and a higher KLD
                    simd<half, MB * 8 * KT> P;
#pragma unroll
                    for (int R = 0; R < RPT; ++R) {
                        simd<float, KT> s = S.template select<KT, 1>(R * KT);
                        if ((s > mref[R]).any()) {
                            const float mx    = hmax<float>(s);
                            const float alpha = exp2(simd<float, 1>(mref[R] - mx))[0];
                            mref[R] = mx;
                            lsum[R] *= alpha;
#pragma unroll
                            for (int c = 0; c < NC; ++c) {
                                O.template select<16, 1>((((R / 8) * NC + c) * 8 + R % 8) * 16) *= alpha;
                            }
                        }
                        simd<float, KT> p = exp2(s - mref[R]);
                        lsum[R] += reduce<float>(p, std::plus<>());
                        P.template select<KT, 1>(R * KT) = convert<half>(p);
                    }

                    // O += P V
#pragma unroll
                    for (int c = 0; c < NC; ++c) {
                        simd<half, KT * 16> vt = load_2d<half, 16, KT, 1, false, true>(vb, D * 2 - 1, n_kv - 1, v_pitch - 1, c * 16, k0);
#pragma unroll
                        for (int mb = 0; mb < MB; ++mb) {
                            O.template select<8 * 16, 1>((mb * NC + c) * 8 * 16) = xmx::dpas<8, 8, float, float, half, half>(
                                simd<float, 8 * 16>(O.template select<8 * 16, 1>((mb * NC + c) * 8 * 16)), vt,
                                simd<half, 8 * KT>(P.template select<8 * KT, 1>(mb * 8 * KT)));
                        }
                    }
                }

                if (part) {
                    // rows [split][head][token]: D unnormalized values, then the max and the sum in base 2
                    float * pr = part + ((size_t) sp * n_q_heads + h) * q_pad * (D + 2);
                    simd<float, RPT> mv, lv;
#pragma unroll
                    for (int R = 0; R < RPT; ++R) {
                        mv[R] = mref[R];
                        lv[R] = lsum[R];
                        simd<float, D> orow;
#pragma unroll
                        for (int c = 0; c < NC; ++c) {
                            orow.template select<16, 1>(c * 16) = O.template select<16, 1>((((R / 8) * NC + c) * 8 + R % 8) * 16);
                        }
                        block_store<float, D>(pr + (size_t) (q0 + R) * D, orow);
                    }
                    block_store<float, RPT>(pr + (size_t) q_pad * D + q0, mv);
                    block_store<float, RPT>(pr + (size_t) q_pad * (D + 1) + q0, lv);
                    return;
                }

                // normalize; the sink logit only adds to the denominator
                const float sink2 = sk ? sk[h] * LOG2E : 0.0f;
#pragma unroll
                for (int R = 0; R < RPT; ++R) {
                    if (q0 + R < n_queries) {
                        float l = lsum[R];
                        if (sk) {
                            l += exp2(simd<float, 1>(sink2 - mref[R]))[0];
                        }
                        const float inv = l > 0.0f ? 1.0f / l : 0.0f;
                        simd<float, D> orow;
#pragma unroll
                        for (int c = 0; c < NC; ++c) {
                            orow.template select<16, 1>(c * 16) = O.template select<16, 1>((((R / 8) * NC + c) * 8 + R % 8) * 16) * inv;
                        }
                        block_store<float, D>(dp + ((size_t) (q0 + R) * n_q_heads + h) * D, orow);
                    }
                }
            });
    });

    if (part) {
        // out = sum_s 2^(m_s - M) O_s / (sum_s 2^(m_s - M) l_s + 2^(sink - M)), M the max over the splits
        const size_t n_out = (size_t) n_queries * n_q_heads * D;
        stream->parallel_for(sycl::nd_range<1>((n_out + 127) / 128 * 128, 128), [=](sycl::nd_item<1> it) {
            const size_t e = it.get_global_id(0);
            if (e >= n_out) {
                return;
            }
            const int     d  = e % D;
            const int     h  = (e / D) % n_q_heads;
            const int     q  = e / ((size_t) D * n_q_heads);
            const size_t  ss = (size_t) n_q_heads * q_pad * (D + 2);  // floats per split
            const float * p0 = part + (size_t) h * q_pad * (D + 2);
            float         M  = NEG;
            for (int s = 0; s < nsplit; ++s) {
                M = sycl::fmax(M, p0[s * ss + (size_t) q_pad * D + q]);
            }
            float o = 0.0f, l = 0.0f;
            for (int s = 0; s < nsplit; ++s) {
                const float w = sycl::exp2(p0[s * ss + (size_t) q_pad * D + q] - M);
                o += w * p0[s * ss + (size_t) q * D + d];
                l += w * p0[s * ss + (size_t) q_pad * (D + 1) + q];
            }
            if (sk) {
                l += sycl::exp2(sk[h] * LOG2E - M);
            }
            dp[((size_t) q * n_q_heads + h) * D + d] = l > 0.0f ? o / l : 0.0f;
        });
    }
#endif  // GGML_SYCL_FA_XMX_BUILD
}
