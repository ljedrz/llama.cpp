// GGML_OP_MACH1_HEAD_MM and GGML_OP_MACH1_EMBED_GATHER: ports of the CUDA kernels in
// ggml-cuda/mach1.cu (mach1_head_mm_tg_kernel, mach1_embed_gather_kernel).

#include "mach1.hpp"

//
// HEAD_MM: W[r, j] = (q[r, j] - 16) * f32(gscale[r][j/64]), q the 5-bit codes packed 32 per
// 5 little-endian uint32 words. One sub-group per vocab row; lanes stride the 32-code word
// groups and each lane keeps one partial dot per token of the current token tile.
//

#define MACH1_HEAD_WG 256
#define MACH1_HEAD_NT 4

static void mach1_head_mm(dpct::queue_ptr stream, const uint8_t * qp, const sycl::half * gscale, const float * x,
                          float * dst, const int n, const int vocab, const int nt) {
    constexpr int SG   = WARP_SIZE;
    constexpr int ROWS = MACH1_HEAD_WG/SG;
    const int n_tiles = (nt + MACH1_HEAD_NT - 1)/MACH1_HEAD_NT;
    stream->parallel_for(
        sycl::nd_range<2>(sycl::range<2>(n_tiles, (size_t)((vocab + ROWS - 1)/ROWS)*MACH1_HEAD_WG),
                          sycl::range<2>(1, MACH1_HEAD_WG)),
        [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG)]] {
            const sycl::sub_group sg = it.get_sub_group();
            const int lane = (int) sg.get_local_linear_id();
            const int r    = (int) it.get_group(1)*ROWS + (int) sg.get_group_linear_id();
            if (r >= vocab) {
                return;
            }
            const int t0  = (int) it.get_group(0)*MACH1_HEAD_NT;
            const int ntt = sycl::min(MACH1_HEAD_NT, nt - t0);

            const int64_t row_bytes = (int64_t) n/8*5;
            const int     ng        = n/64;
            const uint32_t * pw = (const uint32_t *)(qp + (int64_t) r*row_bytes);
            const float    * xt = x + (int64_t) t0*n;

            float acc[MACH1_HEAD_NT];
#pragma unroll
            for (int k = 0; k < MACH1_HEAD_NT; ++k) {
                acc[k] = 0.0f;
            }
            for (int b4i = lane; b4i < n/32; b4i += SG) {
                uint32_t w5[5];
#pragma unroll
                for (int q = 0; q < 5; ++q) {
                    w5[q] = pw[b4i*5 + q];
                }
                // j = b4i*32 + i, i < 32  =>  j >> 6 == b4i >> 1
                const float gsc = (float) gscale[(int64_t) r*ng + (b4i >> 1)];
#pragma unroll
                for (int i = 0; i < 32; ++i) {
                    const int bit0 = 5*i;
                    const int wi   = bit0 >> 5;
                    const int o    = bit0 & 31;
                    uint32_t qv = w5[wi] >> o;
                    if (o > 27) {
                        qv |= w5[wi + 1] << (32 - o);   // straddles a word boundary
                    }
                    qv &= 31u;
                    const float w = (float)((int) qv - 16) * gsc;
#pragma unroll
                    for (int k = 0; k < MACH1_HEAD_NT; ++k) {
                        if (k < ntt) {
                            acc[k] += w * xt[(int64_t) k*n + b4i*32 + i];
                        }
                    }
                }
            }
#pragma unroll
            for (int k = 0; k < MACH1_HEAD_NT; ++k) {
                if (k < ntt) {
                    const float s = sycl::reduce_over_group(sg, acc[k], sycl::plus<float>());
                    if (lane == 0) {
                        dst[(int64_t)(t0 + k)*vocab + r] = s;
                    }
                }
            }
        });
}

// The same product with loads that are contiguous over the sub-group. Above, a lane owns 32-code
// groups, so each of its 32 loads of x per group has the 16 lanes on 16 different cache lines, and
// the device's rate for such gathers bounds the kernel below the memory bandwidth. Here a step is
// one scale group (64 codes, 10 words): lane l takes the 4 codes at 4l, the 20 bits from bit 20l
// of the group, out of the word they start in and (when they run past it) the next, and their 4
// x (a 256-byte run over the sub-group). 4 steps go at a time so their loads are in flight
// together.
// NTT tokens per work-group, a constant so that the loop has no branches; n_tiles tiles of them.
template <int NTT>
static void mach1_head_mm_runs(dpct::queue_ptr stream, const uint8_t * qp, const sycl::half * gscale, const float * x,
                               float * dst, const int n, const int vocab, const int n_tiles) {
    constexpr int U    = 4;
    constexpr int ROWS = MACH1_HEAD_WG/16;
    stream->parallel_for(
        sycl::nd_range<2>(sycl::range<2>(n_tiles, (size_t)((vocab + ROWS - 1)/ROWS)*MACH1_HEAD_WG),
                          sycl::range<2>(1, MACH1_HEAD_WG)),
        [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
            const sycl::sub_group sg = it.get_sub_group();
            const int tid  = (int) it.get_local_id(1);
            const int lane = tid & 15;
            const int r    = (int) it.get_group(1)*ROWS + (tid >> 4);
            if (r >= vocab) {
                return;
            }
            const int t0 = (int) it.get_group(0)*NTT;

            const int64_t row_bytes = (int64_t) n/8*5;
            const int     ng        = n/64;
            // the lane's 20 bits start at bit o of word wa; past bit 12 they end in the next word
            const uint32_t       o  = (uint32_t) (20*lane) & 31u;
            const int            wa = (5*lane) >> 3;
            const int            wb = wa + (o > 12u ? 1 : 0);
            const uint32_t     * pw = (const uint32_t *) (qp + (int64_t) r*row_bytes);
            const sycl::half   * gs = gscale + (int64_t) r*ng;
            const sycl::float4 * x4 = (const sycl::float4 *) (x + (int64_t) t0*n) + lane;
            const int            n4 = n/4;

            // the 4 weights of a word, scaled
            const auto weights = [&](uint32_t a, uint32_t b, float g) {
                // (what b leaves above bit 20 when it is not needed is masked off below)
                const uint32_t v = (a >> o) | ((b << 1) << (31u - o));
                return sycl::float4((float) ((int) (v & 31u) - 16)*g, (float) ((int) ((v >> 5) & 31u) - 16)*g,
                                    (float) ((int) ((v >> 10) & 31u) - 16)*g, (float) ((int) ((v >> 15) & 31u) - 16)*g);
            };
            const auto dot4 = [](sycl::float4 w, sycl::float4 xv) {
                return w.x()*xv.x() + w.y()*xv.y() + w.z()*xv.z() + w.w()*xv.w();
            };
            float acc[NTT];
#pragma unroll
            for (int k = 0; k < NTT; ++k) {
                acc[k] = 0.0f;
            }
            int c = 0;
            for (; c + U <= ng; c += U) {
                uint32_t     a[U], b[U];
                float        g[U];
                sycl::float4 xv[NTT][U];
#pragma unroll
                for (int j = 0; j < U; ++j) {
                    a[j] = pw[j*10 + wa];
                    b[j] = pw[j*10 + wb];
                    g[j] = (float) gs[j];
#pragma unroll
                    for (int k = 0; k < NTT; ++k) {
                        xv[k][j] = x4[k*n4 + j*16];
                    }
                }
#pragma unroll
                for (int j = 0; j < U; ++j) {
                    const sycl::float4 wv = weights(a[j], b[j], g[j]);
#pragma unroll
                    for (int k = 0; k < NTT; ++k) {
                        acc[k] += dot4(wv, xv[k][j]);
                    }
                }
                pw += U*10;
                gs += U;
                x4 += U*16;
            }
            for (; c < ng; ++c) {
                const sycl::float4 wv = weights(pw[wa], pw[wb], (float) gs[0]);
#pragma unroll
                for (int k = 0; k < NTT; ++k) {
                    acc[k] += dot4(wv, x4[k*n4]);
                }
                pw += 10;
                gs += 1;
                x4 += 16;
            }
#pragma unroll
            for (int k = 0; k < NTT; ++k) {
                const float s = sycl::reduce_over_group(sg, acc[k], sycl::plus<float>());
                if (lane == 0) {
                    dst[(int64_t)(t0 + k)*vocab + r] = s;
                }
            }
        });
}

void ggml_sycl_mach1_head_mm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);

    const ggml_tensor * qp     = dst->src[0];
    const ggml_tensor * gscale = dst->src[1];
    const ggml_tensor * x      = dst->src[2];

    GGML_ASSERT(qp->type     == GGML_TYPE_I8);
    GGML_ASSERT(gscale->type == GGML_TYPE_F16);
    GGML_ASSERT(x->type      == GGML_TYPE_F32);
    GGML_ASSERT(dst->type    == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(qp));
    GGML_ASSERT(ggml_is_contiguous(gscale));
    GGML_ASSERT(ggml_is_contiguous(x));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int n     = (int) x->ne[0];
    const int vocab = (int) qp->ne[1];
    const int nt    = (int)(x->ne[1]*x->ne[2]*x->ne[3]);

    static const bool runs = getenv("GGML_SYCL_MACH1_HEAD_RUNS") == nullptr || atoi(getenv("GGML_SYCL_MACH1_HEAD_RUNS")) != 0;
    if (runs && n % 64 == 0 && (uintptr_t) x->data % 16 == 0) {
        const uint8_t    * q  = (const uint8_t *) qp->data;
        const sycl::half * g  = (const sycl::half *) gscale->data;
        const float      * xd = (const float *) x->data;
        float            * yd = (float *) dst->data;
        // whole tiles of MACH1_HEAD_NT tokens, then the rest as one smaller tile
        const int full = nt/MACH1_HEAD_NT;
        const int rest = nt - full*MACH1_HEAD_NT;
        if (full > 0) {
            mach1_head_mm_runs<MACH1_HEAD_NT>(ctx.stream(), q, g, xd, yd, n, vocab, full);
        }
        xd += (size_t) full*MACH1_HEAD_NT*n;
        yd += (size_t) full*MACH1_HEAD_NT*vocab;
        switch (rest) {
            case 1: mach1_head_mm_runs<1>(ctx.stream(), q, g, xd, yd, n, vocab, 1); break;
            case 2: mach1_head_mm_runs<2>(ctx.stream(), q, g, xd, yd, n, vocab, 1); break;
            case 3: mach1_head_mm_runs<3>(ctx.stream(), q, g, xd, yd, n, vocab, 1); break;
            default: break;
        }
        return;
    }
    mach1_head_mm(ctx.stream(), (const uint8_t *) qp->data, (const sycl::half *) gscale->data,
                  (const float *) x->data, (float *) dst->data, n, vocab, nt);
}

//
// EMBED_GATHER: one work-item per output element; the LUT holds the source bf16 bit patterns
//

void ggml_sycl_mach1_embed_gather(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);

    const ggml_tensor * codes = dst->src[0];
    const ggml_tensor * lut   = dst->src[1];
    const ggml_tensor * ids   = dst->src[2];

    GGML_ASSERT(codes->type == GGML_TYPE_I8);
    GGML_ASSERT(lut->type   == GGML_TYPE_BF16);
    GGML_ASSERT(ids->type   == GGML_TYPE_I32);
    GGML_ASSERT(dst->type   == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(codes));
    GGML_ASSERT(ggml_is_contiguous(lut));
    GGML_ASSERT(ggml_is_contiguous(ids));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int     n_embd = (int) dst->ne[0];
    const int64_t total  = ids->ne[0]*n_embd;

    const uint8_t  * codes_p = (const uint8_t  *) codes->data;
    const uint16_t * lut_p   = (const uint16_t *) lut->data;
    const int32_t  * ids_p   = (const int32_t  *) ids->data;
    float          * dst_p   = (float          *) dst->data;

    ctx.stream()->parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t)((total + 255)/256)*256), sycl::range<1>(256)),
        [=](sycl::nd_item<1> it) {
            const int64_t gid = (int64_t) it.get_global_id(0);
            if (gid >= total) {
                return;
            }
            const int64_t tok = gid / n_embd;
            const int     j   = (int)(gid % n_embd);
            const int64_t r   = ids_p[tok];
            const int     ng  = n_embd / 64;

            const uint32_t by   = codes_p[r*(n_embd/2) + (j >> 1)];
            const uint32_t q    = (j & 1) ? (by >> 4) : (by & 0x0Fu);
            const uint32_t bits = (uint32_t) lut_p[(r*ng + (j >> 6))*16 + q] << 16;
            dst_p[gid] = sycl::bit_cast<float>(bits);
        });
}
