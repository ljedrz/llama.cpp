#include "attn-dec.hpp"

#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

static constexpr int ATTN_DEC_NSG = 8;  // sub-groups per work-group
static constexpr int ATTN_DEC_R   = 4;  // V^T rows a sub-group takes at once (the probabilities are loaded once for them)

// sub-group size and, when the kernel needs it, the large register file
template <typename F> struct attn_dec_kernel {
    F f;
    void operator()(sycl::nd_item<3> item) const { f(item); }
    auto get(sycl::ext::oneapi::experimental::properties_tag) const {
        return sycl::ext::oneapi::experimental::properties{ sycl::ext::intel::experimental::grf_size_automatic,
                                                            sycl::ext::oneapi::experimental::sub_group_size<WARP_SIZE> };
    }
};

// one work-group per (chunk of n_kv, KV head): the scores of the NQ query vectors of that head over the chunk, their
// max and exp-sum, and the unnormalized sum of V rows weighted by them, written as a partial for the combine kernel
template <int D, int NQ, typename mask_t, bool VROWS, bool Q8>
static void attn_dec_chunk(const ggml_sycl_attn_dec_args a, const int C, float * part, float * slm,
                           const sycl::nd_item<3> & item) {
    constexpr int NI = D / (WARP_SIZE * 8);  // groups of 8 K values per lane and row (16-byte loads for F16)
    // q8_0: a lane's 8 K values lie in one block; a sub-group's 2*WARP_SIZE V values are one block
    static_assert(!Q8 || (VROWS && QK8_0 % 8 == 0 && 2 * WARP_SIZE == QK8_0), "q8_0 layout");
    const auto    sg    = item.get_sub_group();
    const int     lane  = (int) sg.get_local_linear_id();
    const int     sgi   = (int) sg.get_group_linear_id();
    // the KV heads of a chunk are adjacent work-groups: they share the cache lines of a position (q8_0 head rows are
    // 136 bytes, so lines straddle heads)
    const int64_t hkv   = item.get_group(2);
    const int64_t split = item.get_group(1);
    const int64_t c0    = split * C;
    const int     cn    = (int) sycl::min<int64_t>(C, a.n_kv - c0);
    const int64_t G     = a.n_head / a.n_head_kv;

    float * sc = slm;               // [NQ][C] scores, then probabilities
    float * ml = slm + NQ * sycl::max(C, D);  // [NQ] max, [NQ] sum (after the scratch the position groups use)

    // query j is token j / G of head hkv*G + j % G
    float qr[NQ][NI * 8];
#pragma unroll
    for (int j = 0; j < NQ; ++j) {
        const float * qp = (const float *) (a.q + (j / G) * a.q_nb1 + (hkv * G + j % G) * a.q_nb2);
#pragma unroll
        for (int i = 0; i < NI; ++i) {
            const sycl::float4 u0 = *(const sycl::float4 *) (qp + i * WARP_SIZE * 8 + lane * 8);
            const sycl::float4 u1 = *(const sycl::float4 *) (qp + i * WARP_SIZE * 8 + lane * 8 + 4);
            qr[j][i * 8 + 0] = u0.x(); qr[j][i * 8 + 1] = u0.y(); qr[j][i * 8 + 2] = u0.z(); qr[j][i * 8 + 3] = u0.w();
            qr[j][i * 8 + 4] = u1.x(); qr[j][i * 8 + 5] = u1.y(); qr[j][i * 8 + 6] = u1.z(); qr[j][i * 8 + 7] = u1.w();
        }
    }

    // scores: a sub-group per K row, 8 values of the row per lane, two rows at a time so their loads overlap
    const char * kb = a.k + hkv * a.k_nb2;
    for (int kk0 = sgi * 2; kk0 < cn; kk0 += ATTN_DEC_NSG * 2) {
        float acc[2][NQ] = {};
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const int kk = sycl::min(kk0 + r, cn - 1);
            if constexpr (Q8) {
                // 34-byte blocks: the scale and the quants are only 2-byte aligned
                const char * kr = kb + (c0 + kk) * a.k_nb1;
#pragma unroll
                for (int i = 0; i < NI; ++i) {
                    const int        e0  = i * WARP_SIZE * 8 + lane * 8;
                    const char *     blk = kr + (e0 / QK8_0) * sizeof(block_q8_0);
                    const float      dk  = (float) *(const sycl::half *) blk;
                    const uint16_t * qs  = (const uint16_t *) (blk + sizeof(sycl::half) + e0 % QK8_0);
                    uint16_t         u[4];
#pragma unroll
                    for (int t = 0; t < 4; ++t) {
                        u[t] = qs[t];
                    }
                    float ps[NQ] = {};
#pragma unroll
                    for (int e = 0; e < 8; ++e) {
                        const float w = (float) (int8_t) (u[e / 2] >> (8 * (e % 2)));
#pragma unroll
                        for (int j = 0; j < NQ; ++j) {
                            ps[j] += w * qr[j][i * 8 + e];
                        }
                    }
#pragma unroll
                    for (int j = 0; j < NQ; ++j) {
                        acc[r][j] += dk * ps[j];
                    }
                }
            } else {
            const sycl::half * kr = (const sycl::half *) (kb + (c0 + kk) * a.k_nb1);
#pragma unroll
            for (int i = 0; i < NI; ++i) {
                const sycl::vec<uint16_t, 8> kv = *(const sycl::vec<uint16_t, 8> *) (kr + i * WARP_SIZE * 8 + lane * 8);
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const float w = (float) sycl::bit_cast<sycl::half>((uint16_t) kv[e]);
#pragma unroll
                    for (int j = 0; j < NQ; ++j) {
                        acc[r][j] += w * qr[j][i * 8 + e];
                    }
                }
            }
            }
        }
#pragma unroll
        for (int r = 0; r < 2; ++r) {
#pragma unroll
            for (int j = 0; j < NQ; ++j) {
                const float s = sycl::reduce_over_group(sg, acc[r][j], sycl::plus<float>());
                if (lane == j % WARP_SIZE && kk0 + r < cn) {
                    sc[j * C + kk0 + r] = s * a.scale;
                }
            }
        }
    }
    item.barrier(sycl::access::fence_space::local_space);

    // the mask, then max and exp-sum per query; a chunk that is masked out entirely gets weight 0 (max -inf, sum 0)
    for (int j = sgi; j < NQ; j += ATTN_DEC_NSG) {
        const char * mrow = a.mask ? a.mask + c0 * a.m_nb0 + (j / G) * a.m_nb1 + ((hkv * G + j % G) % a.mask_heads) * a.m_nb2 : nullptr;
        const float *   rt = a.rel_table ? (const float *) (a.rel_table + (hkv * G + j % G) * a.r_tb1 + (j / G) * a.r_tb2) : nullptr;
        const int32_t * ri = a.rel_idx ? a.rel_idx + c0 + (j / G) * a.r_ib1 : nullptr;
        const int32_t * rp = a.rel_kv_pos ? a.rel_kv_pos + c0 : nullptr;
        const uint32_t  rq = a.rel_q_pos ? (uint32_t) a.rel_q_pos[j / G] : 0;
        float m = -INFINITY;
        for (int kk = lane; kk < cn; kk += WARP_SIZE) {
            float s = sc[j * C + kk];
            if (mrow) {
                s += (float) *(const mask_t *) (mrow + kk * a.m_nb0);
            }
            if (rt) {
                if (ri) {
                    s += rt[ri[kk]];
                } else {
                    const uint32_t d = rq - (uint32_t) rp[kk];
                    s += rt[d < a.r_dmax ? d : a.r_dmax];
                }
            }
            if (mrow || rt) {
                sc[j * C + kk] = s;
            }
            m = sycl::max(m, s);
        }
        m = sycl::reduce_over_group(sg, m, sycl::maximum<float>());
        float l = 0.0f;
        for (int kk = lane; kk < cn; kk += WARP_SIZE) {
            const float s = sc[j * C + kk];
            const float p = (m == -INFINITY || s == -INFINITY) ? 0.0f : sycl::native::exp(sycl::max(s - m, -80.0f));
            sc[j * C + kk] = p;
            l += p;
        }
        l = sycl::reduce_over_group(sg, l, sycl::plus<float>());
        if (lane == 0) {
            ml[j]      = m;
            ml[NQ + j] = l;
        }
    }
    item.barrier(sycl::access::fence_space::local_space);

    if constexpr (VROWS) {
        // V rows per position (flash attention layout): a lane owns 2 output values, S sub-groups cover a row and the
        // PG groups of them take every PG-th position, 8 positions per iteration so their loads overlap (16 for q8_0,
        // whose positions are half the bytes)
        constexpr int S  = D / (2 * WARP_SIZE);
        constexpr int PG = ATTN_DEC_NSG / S;
        constexpr int U  = Q8 ? 16 : 8;
        static_assert(U <= WARP_SIZE, "a lane per position's scale");
        static_assert(PG == 1 || PG == 2, "position groups");
        const int    pgrp = sgi / S;
        const int    d    = (sgi % S) * 2 * WARP_SIZE + lane * 2;
        // q8_0: the sub-group's block (one scale for all lanes), then the lane's two quants
        const char * vb   = a.v + hkv * a.v_nb2 + c0 * a.v_nb1 +
                          (Q8 ? (sgi % S) * sizeof(block_q8_0) : d * sizeof(sycl::half));
        float acc[NQ][2] = {};
        for (int kk0 = pgrp; kk0 < cn; kk0 += PG * U) {
            uint32_t vv[U];
            // q8_0: one load for the scales of the U positions (lane u takes position u's), handed out by shuffles
            float    vsl = 0.0f;
            if constexpr (Q8) {
                const int kk = sycl::min(kk0 + (lane % U) * PG, cn - 1);
                vsl          = (float) *(const sycl::half *) (vb + kk * a.v_nb1);
            }
#pragma unroll
            for (int u = 0; u < U; ++u) {
                const int kk = sycl::min(kk0 + u * PG, cn - 1);
                if constexpr (Q8) {
                    vv[u] = *(const uint16_t *) (vb + kk * a.v_nb1 + sizeof(sycl::half) + lane * 2);
                } else {
                    vv[u] = *(const uint32_t *) (vb + kk * a.v_nb1);
                }
            }
#pragma unroll
            for (int u = 0; u < U; ++u) {
                const int kk = kk0 + u * PG;
                if (kk < cn) {
                    float w0, w1;
                    if constexpr (Q8) {
                        const float vs = sycl::select_from_group(sg, vsl, u);
                        w0 = vs * (float) (int8_t) (vv[u] & 0xff);
                        w1 = vs * (float) (int8_t) (vv[u] >> 8);
                    } else {
                        w0 = (float) sycl::bit_cast<sycl::half>((uint16_t) (vv[u] & 0xffff));
                        w1 = (float) sycl::bit_cast<sycl::half>((uint16_t) (vv[u] >> 16));
                    }
#pragma unroll
                    for (int j = 0; j < NQ; ++j) {
                        const float p = sc[j * C + kk];
                        acc[j][0] += w0 * p;
                        acc[j][1] += w1 * p;
                    }
                }
            }
        }
        if constexpr (PG == 2) {
            // the second position group hands its sums to the first through local memory (NQ*D floats before ml)
            item.barrier(sycl::access::fence_space::local_space);
            if (pgrp == 1) {
#pragma unroll
                for (int j = 0; j < NQ; ++j) {
                    sc[j * D + d]     = acc[j][0];
                    sc[j * D + d + 1] = acc[j][1];
                }
            }
            item.barrier(sycl::access::fence_space::local_space);
            if (pgrp == 0) {
#pragma unroll
                for (int j = 0; j < NQ; ++j) {
                    acc[j][0] += sc[j * D + d];
                    acc[j][1] += sc[j * D + d + 1];
                }
            }
        }
        if (pgrp == 0) {
#pragma unroll
            for (int j = 0; j < NQ; ++j) {
                const int64_t h = hkv * G + j % G;
                float *       o = part + ((split * a.n_head + h) * a.n_tok + j / G) * (D + 2);
                o[d]     = acc[j][0];
                o[d + 1] = acc[j][1];
            }
        }
    } else {
    // the weighted sum of V: V^T rows are contiguous in kv, 8 positions per lane, ATTN_DEC_R rows per pass
    const char * vb = a.v + hkv * a.v_nb2 + c0 * sizeof(sycl::half);
    constexpr int rows_sg = D / ATTN_DEC_NSG;
    for (int d0 = sgi * rows_sg; d0 < (sgi + 1) * rows_sg; d0 += ATTN_DEC_R) {
        float acc[ATTN_DEC_R][NQ] = {};
        for (int kk = lane * 8; kk < cn; kk += WARP_SIZE * 8) {
            float p[NQ][8];
#pragma unroll
            for (int j = 0; j < NQ; ++j) {
                const sycl::float4 p0 = *(const sycl::float4 *) (sc + j * C + kk);
                const sycl::float4 p1 = *(const sycl::float4 *) (sc + j * C + kk + 4);
                p[j][0] = p0.x(); p[j][1] = p0.y(); p[j][2] = p0.z(); p[j][3] = p0.w();
                p[j][4] = p1.x(); p[j][5] = p1.y(); p[j][6] = p1.z(); p[j][7] = p1.w();
            }
#pragma unroll
            for (int r = 0; r < ATTN_DEC_R; ++r) {
                const sycl::vec<uint16_t, 8> vv =
                    *(const sycl::vec<uint16_t, 8> *) (vb + (d0 + r) * a.v_nb1 + kk * sizeof(sycl::half));
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const float w = (float) sycl::bit_cast<sycl::half>((uint16_t) vv[e]);
#pragma unroll
                    for (int j = 0; j < NQ; ++j) {
                        acc[r][j] += w * p[j][e];
                    }
                }
            }
        }
#pragma unroll
        for (int r = 0; r < ATTN_DEC_R; ++r) {
#pragma unroll
            for (int j = 0; j < NQ; ++j) {
                const float s = sycl::reduce_over_group(sg, acc[r][j], sycl::plus<float>());
                if (lane == 0) {
                    const int64_t h = hkv * G + j % G;
                    part[((split * a.n_head + h) * a.n_tok + j / G) * (D + 2) + d0 + r] = s;
                }
            }
        }
    }
    }
    if (item.get_local_linear_id() < NQ) {
        const int     j = (int) item.get_local_linear_id();
        const int64_t h = hkv * G + j % G;
        float *       o = part + ((split * a.n_head + h) * a.n_tok + j / G) * (D + 2) + D;
        o[0] = ml[j];
        o[1] = ml[NQ + j];
    }
}

// one work-group per (token, head), a lane per output value: the chunk weights against the overall max (with the sink)
// once per work-group, then each lane sums its value over the chunks
template <int D>
static void attn_dec_combine(const ggml_sycl_attn_dec_args a, const int64_t n_split, const float * part, float * w,
                             const sycl::nd_item<3> & item) {
    const int64_t t   = item.get_group(2) % a.n_tok;
    const int64_t h   = item.get_group(2) / a.n_tok;
    const int     lid = (int) item.get_local_linear_id();

    const size_t  stride = (size_t) a.n_head * a.n_tok * (D + 2);
    const float * p0     = part + (h * a.n_tok + t) * (D + 2);

    float m = -INFINITY;
    for (int64_t s = lid; s < n_split; s += D) {
        m = sycl::max(m, p0[s * stride + D]);
    }
    m = sycl::reduce_over_group(item.get_group(), m, sycl::maximum<float>());
    if (a.sinks) {
        m = sycl::max(m, a.sinks[h]);
    }
    float l = 0.0f;
    for (int64_t s = lid; s < n_split; s += D) {
        const float ms = p0[s * stride + D];
        const float ws = ms == -INFINITY ? 0.0f : sycl::native::exp(sycl::max(ms - m, -80.0f));
        w[s] = ws;
        l += ws * p0[s * stride + D + 1];
    }
    l = sycl::reduce_over_group(item.get_group(), l, sycl::plus<float>());
    if (a.sinks) {
        l += sycl::native::exp(sycl::max(a.sinks[h] - m, -80.0f));
    }
    item.barrier(sycl::access::fence_space::local_space);

    float acc = 0.0f;
#pragma unroll 4
    for (int64_t s = 0; s < n_split; ++s) {
        acc += w[s] * p0[s * stride + lid];
    }
    a.dst[h * a.o_h + t * a.o_t + lid] = acc / l;
}

template <int D, int NQ, typename mask_t, bool VROWS, bool Q8>
static void attn_dec_launch(ggml_backend_sycl_context & ctx, const ggml_sycl_attn_dec_args & a) {
    // chunks of a multiple of 128 positions, about 2048 work-groups over the KV heads
    const int64_t c_slm   = ((int64_t) ggml_sycl_info().devices[ctx.device].smpbo / sizeof(float) - 2 * NQ) / NQ / 128 * 128;
    // caches up to 2048 positions (short contexts, the sliding-window layers): ~16 chunks per KV head, a multiple of 16
    // positions; Inkling 2 tg at depth 4096 / 32768 66.7 / 59.3 -> 67.3 / 59.9 against chunks of 128 and up
    const int64_t C       = a.n_kv <= 2048 ? std::clamp<int64_t>((a.n_kv / 16 + 15) / 16 * 16, 16, 128) :
        std::clamp<int64_t>((a.n_kv * a.n_head_kv / 2048 + 127) / 128 * 128, 128, std::min<int64_t>(1024, c_slm));
    const int64_t n_split = (a.n_kv + C - 1) / C;

    dpct::queue_ptr                stream = ctx.stream();
    ggml_sycl_pool_alloc<float>    part(ctx.pool(), n_split * a.n_head * a.n_tok * (D + 2));
    float *                        part_d = part.get();
    const ggml_sycl_attn_dec_args  p      = a;
    const int                      Ci     = (int) C;

    {
        const sycl::range<3> grid(1, n_split, a.n_head_kv);
        const sycl::range<3> block(1, 1, ATTN_DEC_NSG * WARP_SIZE);
        stream->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<float, 1> slm(sycl::range<1>(NQ * std::max<int64_t>(C, D) + 2 * NQ), cgh);
            auto f = [=](sycl::nd_item<3> item) {
                attn_dec_chunk<D, NQ, mask_t, VROWS, Q8>(p, Ci, part_d, slm.get_multi_ptr<sycl::access::decorated::no>().get(),
                                              item);
            };
            cgh.parallel_for(sycl::nd_range<3>(grid * block, block), attn_dec_kernel<decltype(f)>{ f });
        });
    }
    {
        const sycl::range<3> grid(1, 1, a.n_head * a.n_tok);
        const sycl::range<3> block(1, 1, D);
        stream->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<float, 1> w(sycl::range<1>(n_split), cgh);
            cgh.parallel_for(sycl::nd_range<3>(grid * block, block), [=](sycl::nd_item<3> item) {
                attn_dec_combine<D>(p, n_split, part_d, w.get_multi_ptr<sycl::access::decorated::no>().get(), item);
            });
        });
    }
}

template <int D, typename mask_t, bool VROWS, bool Q8 = false>
static bool attn_dec_switch_nq(ggml_backend_sycl_context & ctx, const ggml_sycl_attn_dec_args & a, int64_t nq) {
    switch (nq) {
        case 1:  attn_dec_launch<D, 1, mask_t, VROWS, Q8>(ctx, a);  return true;
        case 2:  attn_dec_launch<D, 2, mask_t, VROWS, Q8>(ctx, a);  return true;
        case 3:  attn_dec_launch<D, 3, mask_t, VROWS, Q8>(ctx, a);  return true;
        case 4:  attn_dec_launch<D, 4, mask_t, VROWS, Q8>(ctx, a);  return true;
        case 6:  attn_dec_launch<D, 6, mask_t, VROWS, Q8>(ctx, a);  return true;
        case 8:  attn_dec_launch<D, 8, mask_t, VROWS, Q8>(ctx, a);  return true;
        case 9:  attn_dec_launch<D, 9, mask_t, VROWS, Q8>(ctx, a);  return true;
        case 12: attn_dec_launch<D, 12, mask_t, VROWS, Q8>(ctx, a); return true;
        case 16: attn_dec_launch<D, 16, mask_t, VROWS, Q8>(ctx, a); return true;
        default: return false;
    }
}

bool ggml_sycl_attn_dec_supported(int64_t D, int64_t nq) {
    return (D == 128 || D == 256) &&
           (nq == 1 || nq == 2 || nq == 3 || nq == 4 || nq == 6 || nq == 8 || nq == 9 || nq == 12 || nq == 16);
}

void ggml_sycl_attn_dec(ggml_backend_sycl_context & ctx, const ggml_sycl_attn_dec_args & a) {
    const int64_t nq = a.n_tok * (a.n_head / a.n_head_kv);
    bool          ok = false;
    if (a.v_rows) {
        // flash attention: the mask is F16
        GGML_ASSERT(!a.mask || a.mask_f16);
        if (a.D == 128) {
            ok = a.kv_q8 ? attn_dec_switch_nq<128, sycl::half, true, true>(ctx, a, nq) :
                           attn_dec_switch_nq<128, sycl::half, true>(ctx, a, nq);
        } else if (a.D == 256) {
            ok = a.kv_q8 ? attn_dec_switch_nq<256, sycl::half, true, true>(ctx, a, nq) :
                           attn_dec_switch_nq<256, sycl::half, true>(ctx, a, nq);
        }
    } else if (a.kv_q8) {
        ok = false;
    } else if (a.D == 128) {
        ok = a.mask_f16 ? attn_dec_switch_nq<128, sycl::half, false>(ctx, a, nq) : attn_dec_switch_nq<128, float, false>(ctx, a, nq);
    } else if (a.D == 256) {
        ok = a.mask_f16 ? attn_dec_switch_nq<256, sycl::half, false>(ctx, a, nq) : attn_dec_switch_nq<256, float, false>(ctx, a, nq);
    }
    GGML_ASSERT(ok);
}
