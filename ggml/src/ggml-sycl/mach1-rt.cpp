// GGML_OP_MACH1_RT_MM: rotated int-lattice trellis dense matmul (payload v3 NE spine),
//   W = diag(sv) . H_m/sqrt(m) . fp16(hatWr) . H_n/sqrt(n) . diag(su),
// run on the vectors as in the CPU reference and CUDA's rt_u / rt_walk / rt_out:
//   ustage: u = H_n^T (su .* x)/sqrt(n)   (one work-group per token)
//   walk:   v = hatWr u                   (one work-group per 16-row block and token tile)
//   out:    y = sv .* H_m v/sqrt(m)       (one work-group per token)
// H_d is the power-of-two FWHT for d = 2^k and the FWHT over d/r times the r-point
// Hadamard (r = 12, 20; ggml-cuda/mach1-rt-had.cuh) otherwise.

#include "mach1.hpp"

#include <array>
#if GGML_SYCL_DNNL
#include "gemm.hpp"
#endif

extern int g_ggml_sycl_enable_dnn;

#define MACH1_RT_WG    256
#define MACH1_RT_NWP   8
#define MACH1_RT_TT    8
#define MACH1_RT_WORDS 64

// row a of the r-point Hadamard: bit c set = entry (a, c) is -1
static constexpr uint32_t mach1_rt_h12[12] = {
    4094u, 1140u, 2280u, 466u, 932u, 1864u, 3728u, 3362u, 2630u, 1166u, 2332u, 570u,
};
static constexpr uint32_t mach1_rt_h20[20] = {
    1048574u, 398820u, 797640u, 546706u, 44838u, 89676u, 179352u, 358704u, 717408u, 386242u,
    772484u, 496394u, 992788u, 937002u, 825430u, 602286u, 155998u, 311996u, 623992u, 199410u,
};

// column c of the r-point Hadamard: bit a set = entry (a, c) is -1, so a radix sum over c with
// a work-item's own row a indexes the table by the (unrolled, constant) c only
template <int R>
static constexpr std::array<uint32_t, R> mach1_rt_cols(const uint32_t (&H)[R]) {
    std::array<uint32_t, R> t{};
    for (int c = 0; c < R; ++c) {
        for (int a = 0; a < R; ++a) {
            t[c] |= ((H[a] >> c) & 1u) << a;
        }
    }
    return t;
}
static constexpr std::array<uint32_t, 12> mach1_rt_h12c = mach1_rt_cols<12>(mach1_rt_h12);
static constexpr std::array<uint32_t, 20> mach1_rt_h20c = mach1_rt_cols<20>(mach1_rt_h20);

static int mach1_rt_radix(int64_t d) {
    if ((d & (d - 1)) == 0) {
        return 1;
    }
    const int64_t m12 = d/12;
    return d % 12 == 0 && (m12 & (m12 - 1)) == 0 ? 12 : 20;
}

template <int R, bool TR>
static inline void mach1_rt_radix_col(float * sh, const int M, const int b, const float sc) {
    const uint32_t * H = R == 12 ? mach1_rt_h12 : mach1_rt_h20;
    float col[R];
#pragma unroll
    for (int c = 0; c < R; ++c) {
        col[c] = sh[c*M + b];
    }
#pragma unroll
    for (int a = 0; a < R; ++a) {
        float acc = 0.0f;
#pragma unroll
        for (int c = 0; c < R; ++c) {
            const bool neg = TR ? ((H[c] >> a) & 1u) != 0u : ((H[a] >> c) & 1u) != 0u;
            acc = neg ? acc - col[c] : acc + col[c];
        }
        sh[a*M + b] = acc/sc;
    }
}

// H_d of the vector in local memory, scaled by 1/sc (sc = sqrtf(d) from the host);
// TR selects H^T for the radix factor
template <bool TR>
static inline void mach1_rt_had(float * sh, const int d, const int R, const float sc, const sycl::nd_item<1> & it) {
    const int tid = (int) it.get_local_id(0);
    const int wg  = (int) it.get_local_range(0);
    const int M   = d/R;
    for (int span = 1, ls = 0; span < M; span <<= 1, ++ls) {
        for (int b = tid; b < d/2; b += wg) {
            const int base = ((b >> ls) << (ls + 1)) + (b & (span - 1));
            const float a0 = sh[base];
            const float a1 = sh[base + span];
            sh[base]        = a0 + a1;
            sh[base + span] = a0 - a1;
        }
        sycl::group_barrier(it.get_group());
    }
    if (R == 1) {
        for (int i = tid; i < d; i += wg) {
            sh[i] = sh[i]/sc;
        }
    } else {
        // columns are disjoint, so each work-item transforms its own in place
        for (int b = tid; b < M; b += wg) {
            if (R == 12) {
                mach1_rt_radix_col<12, TR>(sh, M, b, sc);
            } else {
                mach1_rt_radix_col<20, TR>(sh, M, b, sc);
            }
        }
    }
    sycl::group_barrier(it.get_group());
}

// OUT = false: dst = H^T (side .* src)/sc;  OUT = true: dst = side .* (H src)/sc, with src + src2
// in place of src when src2 is set (the two halves of a split GEMM)
template <bool OUT>
static void mach1_rt_stage(dpct::queue_ptr stream, const float * src, const float * side, float * dst,
                           const int d, const int nt, const float * src2) {
    const int   R  = mach1_rt_radix(d);
    const float sc = sqrtf((float) d);
    const int   wg = d >= 4096 ? 1024 : MACH1_RT_WG;
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> sh_acc(sycl::range<1>(d), cgh);
        cgh.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t) nt*wg), sycl::range<1>(wg)),
            [=](sycl::nd_item<1> it) {
                float * sh = sh_acc.get_multi_ptr<sycl::access::decorated::no>().get();
                const int t   = (int) it.get_group(0);
                const int tid = (int) it.get_local_id(0);
                const float * s  = src + (int64_t) t*d;
                const float * s2 = src2 ? src2 + (int64_t) t*d : nullptr;
                for (int i = tid; i < d; i += wg) {
                    sh[i] = OUT ? (s2 ? s[i] + s2[i] : s[i]) : side[i]*s[i];
                }
                sycl::group_barrier(it.get_group());
                mach1_rt_had<!OUT>(sh, d, R, sc, it);
                float * y = dst + (int64_t) t*d;
                for (int i = tid; i < d; i += wg) {
                    y[i] = OUT ? sh[i]*side[i] : sh[i];
                }
            });
    });
}

// The same stages with one sub-group per 2^k block, for d = R*M with M = 16*E, E <= 32: lane l
// of sub-group a holds elements a*M + k*16 + l, the butterflies run across the lanes and in
// registers (mach1_sg_fwht), in the span order of mach1_rt_had, so the result is bit-identical;
// only the radix factor goes through local memory.
template <bool OUT, int R, int E>
static void mach1_rt_sg_stage(dpct::queue_ptr stream, const float * src, const float * side, float * dst,
                              const int nt, const float * src2) {
    constexpr int M  = 16*E;
    constexpr int D  = R*M;
    constexpr int WG = 16*R;
    const float sc = sqrtf((float) D);
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> sh_acc(sycl::range<1>(R > 1 ? D : 1), cgh);
        cgh.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t) nt*WG), sycl::range<1>(WG)),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                float * sh = sh_acc.get_multi_ptr<sycl::access::decorated::no>().get();
                const sycl::sub_group sg = it.get_sub_group();
                const int t    = (int) it.get_group(0);
                const int tid  = (int) it.get_local_id(0);
                const int lane = tid & 15;
                const int blk  = tid >> 4;
                const float * s  = src + (int64_t) t*D;
                const float * s2 = src2 ? src2 + (int64_t) t*D : nullptr;
                float       * y  = dst + (int64_t) t*D;

                float v[E];
#pragma unroll
                for (int k = 0; k < E; ++k) {
                    const int i = blk*M + k*16 + lane;
                    v[k] = OUT ? (s2 ? s[i] + s2[i] : s[i]) : side[i]*s[i];
                }
                mach1_sg_fwht<E>(sg, lane, v);
                if constexpr (R == 1) {
#pragma unroll
                    for (int k = 0; k < E; ++k) {
                        const int i = k*16 + lane;
                        const float o = v[k]/sc;
                        y[i] = OUT ? o*side[i] : o;
                    }
                } else {
#pragma unroll
                    for (int k = 0; k < E; ++k) {
                        sh[blk*M + k*16 + lane] = v[k];
                    }
                    sycl::group_barrier(it.get_group());
                    // entry (a, c) of H (OUT) or of H^T, as bit a of a word indexed by c
                    const uint32_t * H = OUT ? (R == 12 ? mach1_rt_h12c.data() : mach1_rt_h20c.data())
                                             : (R == 12 ? mach1_rt_h12 : mach1_rt_h20);
#pragma unroll
                    for (int k = 0; k < E; ++k) {
                        const int i = tid + k*WG;
                        const int a = i / M;
                        const int b = i % M;
                        float acc = 0.0f;
#pragma unroll
                        for (int c = 0; c < R; ++c) {
                            const bool neg = ((H[c] >> a) & 1u) != 0u;
                            const float col = sh[c*M + b];
                            acc = neg ? acc - col : acc + col;
                        }
                        const float o = acc/sc;
                        y[i] = OUT ? o*side[i] : o;
                    }
                }
            });
    });
}

// The same stage for a few tokens as two launches that spread over the device (the one above is
// one work-group per token, on a single Xe core): the 2^k blocks, one sub-group each, into dst
// (or tmp); then the radix factor, in place with one work-item per column b of a token taking its R
// values, or from tmp with one work-item per output (R times the work-items: the in-place pass has
// only M of them). Same operations in the same order as mach1_rt_sg_stage, so the result is
// bit-identical.
template <bool OUT, int R, int E>
static void mach1_rt_sg_stage_split(dpct::queue_ptr stream, const float * src, const float * side, float * dst,
                                    const int nt, const float * src2, float * tmp) {
    static_assert(R > 1, "the radix factor is what needs the second pass");
    constexpr int M = 16*E;
    constexpr int D = R*M;
    const float sc = sqrtf((float) D);
    float * mid = tmp ? tmp : dst;
    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) nt*R*16), sycl::range<1>(16)),
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            const sycl::sub_group sg = it.get_sub_group();
            const int t    = (int) it.get_group(0) / R;
            const int blk  = (int) it.get_group(0) % R;
            const int lane = (int) it.get_local_id(0);
            const float * s  = src + (int64_t) t*D;
            const float * s2 = src2 ? src2 + (int64_t) t*D : nullptr;
            float       * y  = mid + (int64_t) t*D;

            float v[E];
#pragma unroll
            for (int k = 0; k < E; ++k) {
                const int i = blk*M + k*16 + lane;
                v[k] = OUT ? (s2 ? s[i] + s2[i] : s[i]) : side[i]*s[i];
            }
            mach1_sg_fwht<E>(sg, lane, v);
#pragma unroll
            for (int k = 0; k < E; ++k) {
                y[blk*M + k*16 + lane] = v[k];
            }
        });
    constexpr int WG2 = 64;
    if (tmp) {
        const int n3 = nt*D;
        stream->parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t) (n3 + WG2 - 1)/WG2*WG2), sycl::range<1>(WG2)),
            [=](sycl::nd_item<1> it) {
                const int g = (int) it.get_global_id(0);
                if (g >= n3) {
                    return;
                }
                const int t = g / D;
                const int i = g % D;
                const int a = i / M;
                const int b = i % M;
                const float * x = tmp + (int64_t) t*D;
                const uint32_t * H = R == 12 ? mach1_rt_h12 : mach1_rt_h20;
                float acc = 0.0f;
#pragma unroll
                for (int c = 0; c < R; ++c) {
                    const bool neg = OUT ? ((H[a] >> c) & 1u) != 0u : ((H[c] >> a) & 1u) != 0u;
                    const float col = x[c*M + b];
                    acc = neg ? acc - col : acc + col;
                }
                const float o = acc/sc;
                dst[(int64_t) t*D + i] = OUT ? o*side[i] : o;
            });
        return;
    }
    const int n2 = nt*M;
    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) (n2 + WG2 - 1)/WG2*WG2), sycl::range<1>(WG2)),
        [=](sycl::nd_item<1> it) {
            const int g = (int) it.get_global_id(0);
            if (g >= n2) {
                return;
            }
            const int t = g / M;
            const int b = g % M;
            float * y = dst + (int64_t) t*D;
            float col[R];
#pragma unroll
            for (int c = 0; c < R; ++c) {
                col[c] = y[c*M + b];
            }
            const uint32_t * H = R == 12 ? mach1_rt_h12 : mach1_rt_h20;
#pragma unroll
            for (int a = 0; a < R; ++a) {
                float acc = 0.0f;
#pragma unroll
                for (int c = 0; c < R; ++c) {
                    const bool neg = OUT ? ((H[a] >> c) & 1u) != 0u : ((H[c] >> a) & 1u) != 0u;
                    acc = neg ? acc - col[c] : acc + col[c];
                }
                const int   i = a*M + b;
                const float o = acc/sc;
                y[i] = OUT ? o*side[i] : o;
            }
        });
}

// decode: up to this many tokens take the split stage when the blocks are large
#define MACH1_RT_SPLIT_MAX_NT 2

template <bool OUT, int R>
static bool mach1_rt_sg_stage_r(dpct::queue_ptr stream, const float * src, const float * side, float * dst,
                                const int M, const int nt, const float * src2, float * tmp) {
    if constexpr (R > 1) {
        if (nt <= MACH1_RT_SPLIT_MAX_NT) {
            switch (M) {
                case 512:  mach1_rt_sg_stage_split<OUT, R, 32>(stream, src, side, dst, nt, src2, tmp); return true;
                case 1024: mach1_rt_sg_stage_split<OUT, R, 64>(stream, src, side, dst, nt, src2, tmp); return true;
                default:   break;
            }
        }
    }
    switch (M) {
        case 16:  mach1_rt_sg_stage<OUT, R, 1> (stream, src, side, dst, nt, src2); return true;
        case 32:  mach1_rt_sg_stage<OUT, R, 2> (stream, src, side, dst, nt, src2); return true;
        case 64:  mach1_rt_sg_stage<OUT, R, 4> (stream, src, side, dst, nt, src2); return true;
        case 128: mach1_rt_sg_stage<OUT, R, 8> (stream, src, side, dst, nt, src2); return true;
        case 256: mach1_rt_sg_stage<OUT, R, 16>(stream, src, side, dst, nt, src2); return true;
        case 512: mach1_rt_sg_stage<OUT, R, 32>(stream, src, side, dst, nt, src2); return true;
        default:  return false;
    }
}

// the sub-group stage when the shape allows it, else the work-group one; tmp (nt*d floats, or
// null) lets the split stage take its radix pass out of place
template <bool OUT>
static void mach1_rt_stage_any(dpct::queue_ptr stream, const float * src, const float * side, float * dst,
                               const int d, const int nt, const float * src2 = nullptr, float * tmp = nullptr) {
    const int R = mach1_rt_radix(d);
    // a power of two is one block for a single sub-group: past 128 the work-group stage is faster
    const bool done = R == 1  ? d <= 128 && mach1_rt_sg_stage_r<OUT, 1> (stream, src, side, dst, d, nt, src2, tmp) :
                      R == 12 ? mach1_rt_sg_stage_r<OUT, 12>(stream, src, side, dst, d/12, nt, src2, tmp) :
                                mach1_rt_sg_stage_r<OUT, 20>(stream, src, side, dst, d/20, nt, src2, tmp);
    if (!done) {
        mach1_rt_stage<OUT>(stream, src, side, dst, d, nt, src2);
    }
}

// v = hatWr u for one 16-row block and up to TT tokens. A tile is 64 words = 128 byte steps;
// step i is the big-endian u16 window at byte i (wrapping), p = s*(s + 1), and gives the
// weights (+-tlut[2*row], tlut[2*row + 1]) with row = (p >> 6) & 511 at tile row i/8, columns
// (2i) & 15 and that + 1. Work-item r of a 16-group takes tile row r (steps 8r .. 8r + 7, in
// order, as the CPU reference does); the 8 groups split the column tiles.
#define MACH1_RT_WALK_WG (MACH1_RT_NWP*16)
template <int TT>
static void mach1_rt_walk(dpct::queue_ptr stream, const uint16_t * trellis, const sycl::half * tlut,
                          const float * scr_u, float * scr_v, const int n, const int m, const int nt) {
    const int Mb = m/16;
    const int Nb = n/16;
    const int n_tiles = (nt + TT - 1)/TT;
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1>        u_acc(sycl::range<1>(1), cgh);
        sycl::local_accessor<sycl::half2, 1>  tl_acc(sycl::range<1>(512), cgh);
        sycl::local_accessor<float, 1>        part_acc(sycl::range<1>(MACH1_RT_WALK_WG), cgh);
        cgh.parallel_for(
            sycl::nd_range<2>(sycl::range<2>(n_tiles, (size_t) Mb*MACH1_RT_WALK_WG), sycl::range<2>(1, MACH1_RT_WALK_WG)),
            [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
                float        * tg_u = u_acc.get_multi_ptr<sycl::access::decorated::no>().get();
                sycl::half2  * tl   = tl_acc.get_multi_ptr<sycl::access::decorated::no>().get();
                float        * part = part_acc.get_multi_ptr<sycl::access::decorated::no>().get();
                const int tid = (int) it.get_local_id(1);
                const int rb  = (int) it.get_group(1);
                const int t0  = (int) it.get_group(0)*TT;
                const int ntt = sycl::min(TT, nt - t0);
                const int row = tid & 15;
                const int wid = tid >> 4;

                for (int i = tid; i < 512; i += MACH1_RT_WALK_WG) {
                    tl[i] = sycl::half2(tlut[2*i], tlut[2*i + 1]);
                }
                (void) tg_u;
                sycl::group_barrier(it.get_group());

                const int depth = (Nb + MACH1_RT_NWP - 1)/MACH1_RT_NWP;
                const int cb0   = wid*depth;
                const int cbe   = sycl::min(cb0 + depth, Nb);

                // tiles are 128-byte aligned: row r reads words 4r .. 4r + 4 as 32-bit pairs
                const uint32_t * tr = (const uint32_t *)(trellis + (int64_t) rb*Nb*MACH1_RT_WORDS);
                const int q0 = 2*row;
                const int q2 = (2*row + 2) & (MACH1_RT_WORDS/2 - 1);
                const sycl::sub_group sg = it.get_sub_group();

                float acc[TT];
#pragma unroll
                for (int k = 0; k < TT; ++k) {
                    acc[k] = 0.0f;
                }
                for (int cb = cb0; cb < cbe; ++cb) {
                    const uint32_t * tw = tr + cb*(MACH1_RT_WORDS/2);
                    const uint32_t a  = tw[q0];
                    const uint32_t b  = tw[q0 + 1];
                    const uint32_t w4 = tw[q2] & 0xFFFFu;
                    // big-endian word pairs: x0 = w[4r] w[4r+1], x1 = w[4r+2] w[4r+3]
                    const uint32_t x0 = (a << 16) | (a >> 16);
                    const uint32_t x1 = (b << 16) | (b >> 16);
                    uint32_t win[8];
                    win[0] = x0 >> 16;
                    win[1] = (x0 >> 8) & 0xFFFFu;
                    win[2] = x0 & 0xFFFFu;
                    win[3] = ((x0 & 0xFFu) << 8) | (x1 >> 24);
                    win[4] = x1 >> 16;
                    win[5] = (x1 >> 8) & 0xFFFFu;
                    win[6] = x1 & 0xFFFFu;
                    win[7] = ((x1 & 0xFFu) << 8) | (w4 >> 8);
                    float wv[16];
#pragma unroll
                    for (int s = 0; s < 8; ++s) {
                        const uint32_t p = win[s]*(win[s] + 1);
                        const sycl::float2 v = tl[(p >> 6) & 511u].convert<float>();
                        // the sign of the first value is p's bit 15 (exact: a sign-bit flip)
                        wv[2*s + 0] = sycl::bit_cast<float>(sycl::bit_cast<uint32_t>(v.x()) ^ ((p & 0x8000u) << 16));
                        wv[2*s + 1] = v.y();
                    }
                    const int uo = cb*16;
#pragma unroll
                    for (int k = 0; k < TT; ++k) {
                        if (k < ntt) {
                            // the 16 u of the tile are the same for every lane: one 64-byte load
                            // (lane l takes u[l]) and broadcasts
                            const float uv = scr_u[(int64_t)(t0 + k)*n + uo + row];
#pragma unroll
                            for (int s = 0; s < 8; ++s) {
                                acc[k] += wv[2*s]*sycl::select_from_group(sg, uv, 2*s)
                                        + wv[2*s + 1]*sycl::select_from_group(sg, uv, 2*s + 1);
                            }
                        }
                    }
                }
#pragma unroll
                for (int k = 0; k < TT; ++k) {
                    if (k < ntt) {
                        part[tid] = acc[k];
                        sycl::group_barrier(it.get_group());
                        if (tid < 16) {
                            float s2 = 0.0f;
                            for (int w = 0; w < MACH1_RT_NWP; ++w) {
                                s2 += part[w*16 + tid];
                            }
                            scr_v[(int64_t)(t0 + k)*m + rb*16 + tid] = s2;
                        }
                        sycl::group_barrier(it.get_group());
                    }
                }
            });
    });
}

// The single-token walk on the matrix engine. A LUT entry is already the fp16 pair of a step's two
// weights, so a lane's 8 lookups (sign flipped in place) are its slice of the DPAS B operand as
// they come, and the tile's u, split in fp16 hi and lo rows (mach1.hpp), is A: one 2-row DPAS per
// tile in place of 16 fp16 -> fp32 conversions and 16 multiply-adds. Tolerance class, as the batch
// path. A work-group is 8 sub-groups: 8/NR row blocks, each with its column tiles in NR ranges.
// The walk is bound by the wait for a tile's chain (cold trellis words, lookups, DPAS) and by the
// device's rate for the lookups, not by instruction count, so a sub-group takes 2 tiles per step,
// each with its own accumulator, and a step's loads go out before the step ahead of it is worked
// on.
#define MACH1_RT_XMX1_SG 8
#define MACH1_RT_XMX1_U  2
template <int NR>
static void mach1_rt_walk_xmx1(dpct::queue_ptr stream, const uint16_t * trellis, const sycl::half * tlut,
                               const float * scr_u, float * scr_v, const int n, const int m) {
    constexpr int SG  = MACH1_RT_XMX1_SG;
    constexpr int U   = MACH1_RT_XMX1_U;
    constexpr int RBW = SG/NR;
    constexpr int TW  = MACH1_RT_WORDS/2;   // 32-bit words per tile
    const int Mb    = m/16;
    const int Nb    = n/16;
    const int depth = (Nb + NR - 1)/NR;
    const uint32_t * tlg = (const uint32_t *) tlut;
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<uint32_t, 1> tl_acc(sycl::range<1>(512), cgh);
        sycl::local_accessor<float, 1>    part_acc(sycl::range<1>(SG*16), cgh);
        cgh.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t) ((Mb + RBW - 1)/RBW)*SG*16), sycl::range<1>(SG*16)),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                uint32_t * tl   = tl_acc.get_multi_ptr<sycl::access::decorated::no>().get();
                float    * part = part_acc.get_multi_ptr<sycl::access::decorated::no>().get();
                const int tid = (int) it.get_local_id(0);
                // (tlut[2*i], tlut[2*i + 1]) as one word; the loads go out together
                {
                    uint32_t e[512/(SG*16)];
#pragma unroll
                    for (int k = 0; k < 512/(SG*16); ++k) {
                        e[k] = tlg[tid + k*SG*16];
                    }
#pragma unroll
                    for (int k = 0; k < 512/(SG*16); ++k) {
                        tl[tid + k*SG*16] = e[k];
                    }
                }
                sycl::group_barrier(it.get_group());
                const int row = tid & 15;
                const int sgi = tid >> 4;
                const int rb  = (int) it.get_group(0)*RBW + sgi/NR;
                const int rg  = sgi % NR;
                const bool on = rb < Mb;

                // one tile: the row's 8 steps from its words a, b and the wrap word c, times u
                const auto tile = [&](mach1_f2v acc, uint32_t a, uint32_t b, uint32_t c, float u) {
                    // the low 16 bits of win[s] are the big-endian window at byte 8r + s; the bits
                    // above them do not reach the bits of p that are used
                    uint32_t win[8];
                    win[0] = a;
                    win[1] = sycl::rotate(a, 8u);
                    win[2] = a >> 16;
                    win[3] = sycl::rotate((b << 16) | win[2], 8u);
                    win[4] = b;
                    win[5] = sycl::rotate(b, 8u);
                    win[6] = b >> 16;
                    win[7] = sycl::rotate((c << 16) | win[6], 8u);
                    mach1_i8v wb;
#pragma unroll
                    for (int s = 0; s < 8; ++s) {
                        const uint32_t p = win[s]*(win[s] + 1u);
                        // the first value's sign is p's bit 15
                        wb[s] = (int) (tl[(p >> 6) & 511u] ^ (p & 0x8000u));
                    }
                    // the 16 u of the tile are the same for every lane: lane l holds u[l]
                    return mach1_dpas2(acc, mach1_xmx1_split(u), wb);
                };

                mach1_f2v acc[U];
#pragma unroll
                for (int k = 0; k < U; ++k) {
                    acc[k] = 0.0f;
                }
                if (on) {
                    const int cb0 = rg*depth;
                    const int cbe = sycl::min(cb0 + depth, Nb);
                    // tiles are 128-byte aligned: row r reads words 4r .. 4r + 4 as 32-bit pairs
                    const uint32_t * tr = (const uint32_t *)(trellis + ((int64_t) rb*Nb + cb0)*MACH1_RT_WORDS);
                    const uint32_t * ta = tr + 2*row;
                    const uint32_t * tc = tr + ((2*row + 2) & (TW - 1));
                    const float    * up = scr_u + cb0*16 + row;
                    uint32_t a[U], b[U], c[U];
                    float    u[U];
                    const auto load = [&]() {
#pragma unroll
                        for (int k = 0; k < U; ++k) {
                            a[k] = ta[k*TW];
                            b[k] = ta[k*TW + 1];
                            c[k] = tc[k*TW];
                            u[k] = up[k*16];
                        }
                        ta += U*TW;
                        tc += U*TW;
                        up += U*16;
                    };
                    int cb = cb0;
                    if (cb + U <= cbe) {
                        load();
                        for (; cb + 2*U <= cbe; cb += U) {
                            uint32_t a0[U], b0[U], c0[U];
                            float    u0[U];
#pragma unroll
                            for (int k = 0; k < U; ++k) {
                                a0[k] = a[k];
                                b0[k] = b[k];
                                c0[k] = c[k];
                                u0[k] = u[k];
                            }
                            load();
#pragma unroll
                            for (int k = 0; k < U; ++k) {
                                acc[k] = tile(acc[k], a0[k], b0[k], c0[k], u0[k]);
                            }
                        }
#pragma unroll
                        for (int k = 0; k < U; ++k) {
                            acc[k] = tile(acc[k], a[k], b[k], c[k], u[k]);
                        }
                        cb += U;
                    }
                    for (; cb < cbe; ++cb) {
                        acc[0] = tile(acc[0], ta[0], ta[1], tc[0], up[0]);
                        ta += TW;
                        tc += TW;
                        up += 16;
                    }
                }
#pragma unroll
                for (int k = 1; k < U; ++k) {
                    acc[0] += acc[k];
                }
                const float v = mach1_xmx1_join(acc[0]);
                if constexpr (NR == 1) {
                    if (on) {
                        scr_v[rb*16 + row] = v*(1.0f/MACH1_XMX1_US);
                    }
                } else {
                    part[tid] = v;
                    sycl::group_barrier(it.get_group());
                    if (on && rg == 0) {
                        float s2 = 0.0f;
#pragma unroll
                        for (int w = 0; w < NR; ++w) {
                            s2 += part[tid + w*16];
                        }
                        scr_v[rb*16 + row] = s2*(1.0f/MACH1_XMX1_US);
                    }
                }
            });
    });
}

// the single-token walk: on the matrix engine when the device has one, else (or with
// GGML_SYCL_MACH1_RT_XMX1=0) the SIMT walk
static void mach1_rt_walk_one(dpct::queue_ptr stream, const uint16_t * trellis, const sycl::half * tlut,
                              const float * scr_u, float * scr_v, const int n, const int m) {
    static const bool xmx_env = getenv("GGML_SYCL_MACH1_RT_XMX1") == nullptr || atoi(getenv("GGML_SYCL_MACH1_RT_XMX1")) != 0;
    if (!xmx_env || !mach1_caps(stream).xmx) {
        mach1_rt_walk<1>(stream, trellis, tlut, scr_u, scr_v, n, m, 1);
        return;
    }
    switch (mach1_xmx1_ranges(stream, m/16)) {
        case 8:  mach1_rt_walk_xmx1<8>(stream, trellis, tlut, scr_u, scr_v, n, m); break;
        case 4:  mach1_rt_walk_xmx1<4>(stream, trellis, tlut, scr_u, scr_v, n, m); break;
        case 2:  mach1_rt_walk_xmx1<2>(stream, trellis, tlut, scr_u, scr_v, n, m); break;
        default: mach1_rt_walk_xmx1<1>(stream, trellis, tlut, scr_u, scr_v, n, m); break;
    }
}

#if GGML_SYCL_DNNL
// Batches on the matrix engines: hatWr decoded once per call to a dense fp16 [m, n] matrix (exact:
// the LUT is fp16), u split into fp16 high and low parts (u = hi + lo to ~2^-22), and one oneDNN
// GEMM over [hi | lo] with fp32 accumulation; v = C_hi + C_lo.
#define MACH1_RT_GEMM_MIN 16

// one work-group per row block and 16 column tiles: the tiles' words are staged in local memory
// with coalesced loads, then sub-group r decodes tile row r (8 steps of 2 weights per tile, as in
// mach1_rt_walk) into its 512-byte run of the dense row. Lane l takes the half-rows (4 steps) l of
// tiles 0-7 and of tiles 8-15, so each of its two 16-byte stores is part of one contiguous
// 256-byte store of the sub-group: twice the bandwidth of a 32-byte store per lane.
static void mach1_rt_decode_dense(dpct::queue_ptr stream, const uint16_t * trellis, const sycl::half * tlut,
                                  sycl::half * w, const int n, const int m) {
    const int Nb = n/16;
    const int Nc = (Nb + 15)/16;
    const uint32_t * tlg = (const uint32_t *) tlut;
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<uint32_t, 1> tl_acc(sycl::range<1>(512), cgh);
        sycl::local_accessor<uint32_t, 1> tw_acc(sycl::range<1>(16*MACH1_RT_WORDS/2), cgh);
        cgh.parallel_for(
            sycl::nd_range<2>(sycl::range<2>((size_t) m/16, (size_t) Nc*256), sycl::range<2>(1, 256)),
            [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
                uint32_t * tl  = tl_acc.get_multi_ptr<sycl::access::decorated::no>().get();
                uint32_t * stw = tw_acc.get_multi_ptr<sycl::access::decorated::no>().get();
                const int tid = (int) it.get_local_id(1);
                const int rb  = (int) it.get_group(0);
                const int cb0 = (int) it.get_group(1)*16;
                const int nc  = sycl::min(16, Nb - cb0);
                // the tiles of a row block are contiguous
                const uint32_t * src = (const uint32_t *)(trellis + ((int64_t) rb*Nb + cb0)*MACH1_RT_WORDS);
                for (int i = tid; i < 512; i += 256) {
                    tl[i] = tlg[i];
                    if (i < nc*(MACH1_RT_WORDS/2)) {
                        stw[i] = src[i];
                    }
                }
                sycl::group_barrier(it.get_group());
                const int r  = tid >> 4;
                const int l  = tid & 15;
                const int hs = l & 1;
                // steps 4*hs .. 4*hs + 3 of row r of tile t: the big-endian windows at bytes 8r + 4hs ..
                const auto half_row = [&](int t) {
                    const uint32_t * tw = stw + t*(MACH1_RT_WORDS/2);
                    const uint32_t p0 = tw[2*r + hs];
                    const uint32_t p1 = tw[(2*r + hs + 1) & (MACH1_RT_WORDS/2 - 1)];
                    const uint32_t x0 = (p0 << 16) | (p0 >> 16);
                    const uint32_t x1 = (p1 << 16) | (p1 >> 16);
                    uint32_t win[4];
                    win[0] = x0 >> 16;
                    win[1] = (x0 >> 8) & 0xFFFFu;
                    win[2] = x0 & 0xFFFFu;
                    win[3] = ((x0 & 0xFFu) << 8) | (x1 >> 24);
                    sycl::uint4 o;
#pragma unroll
                    for (int s = 0; s < 4; ++s) {
                        const uint32_t p = win[s]*(win[s] + 1);
                        // (tlut[2*row], tlut[2*row + 1]) as one word; the first value's sign is p's bit 15
                        o[s] = tl[(p >> 6) & 511u] ^ (p & 0x8000u);
                    }
                    return o;
                };
                sycl::uint4 * dst = (sycl::uint4 *) (w + ((int64_t) rb*16 + r)*n + cb0*16);
                if ((l >> 1) < nc) {
                    dst[l] = half_row(l >> 1);
                }
                if (8 + (l >> 1) < nc) {
                    dst[16 + l] = half_row(8 + (l >> 1));
                }
            });
    });
}

// c2 = [C_hi | C_lo] (2*nt*m), v = C_hi + C_lo is left to the output stage
static void mach1_rt_gemm(ggml_backend_sycl_context & ctx, dpct::queue_ptr stream, const uint16_t * trellis,
                          const sycl::half * tlut, const float * scr_u, float * c2, const int n, const int m,
                          const int nt) {
    const int64_t nu = (int64_t) nt*n;
    ggml_sycl_pool_alloc<sycl::half> w(ctx.pool(), (size_t) m*n);
    ggml_sycl_pool_alloc<sycl::half> uhl(ctx.pool(), (size_t) 2*nu);
    mach1_rt_decode_dense(stream, trellis, tlut, w.get(), n, m);
    sycl::half * uh = uhl.get();
    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) ((nu + 255)/256)*256), sycl::range<1>(256)),
        [=](sycl::nd_item<1> it) {
            const int64_t k = (int64_t) it.get_global_id(0);
            if (k >= nu) {
                return;
            }
            const float      x  = scr_u[k];
            const sycl::half hi = (sycl::half) x;
            uh[k]      = hi;
            uh[nu + k] = (sycl::half) (x - (float) hi);
        });
    DnnlGemmWrapper::row_gemm(ctx, m, 2*nt, n, w.get(), DnnlGemmWrapper::to_dt<sycl::half>(), uh,
                              DnnlGemmWrapper::to_dt<sycl::half>(), c2, DnnlGemmWrapper::to_dt<float>(), stream);
}
#endif

static bool mach1_rt_dim_ok(int64_t d, int64_t lim) {
    if (d <= 0 || d % 16 != 0 || d > lim) {
        return false;
    }
    while (d % 2 == 0) {
        d /= 2;
    }
    return d == 1 || d == 3 || d == 5;
}

bool ggml_sycl_mach1_rt_supported(const ggml_tensor * op, size_t local_mem) {
    // the u/out stages hold one whole vector in local memory, the single-token walk u
    // next to the 1024-float LUT and the reduction scratch
    const int64_t lim = (int64_t) (local_mem/sizeof(float));
    return op->src[0]->ne[0] == MACH1_RT_WORDS && op->src[3]->type == GGML_TYPE_F16 &&
           mach1_rt_dim_ok(op->src[1]->ne[0], lim - 1024 - MACH1_RT_WALK_WG) && mach1_rt_dim_ok(op->src[2]->ne[0], lim);
}

//
// Decode pairs: two RT products of the same input (the shared expert's gate and up) with the
// input stages in one launch, the two walks, and both output stages plus the SWIGLU in one
// launch. Each product's arithmetic is that of mach1_rt_sg_stage and mach1_rt_walk.
//

// the stage transform of the row whose elements blk*M + k*16 + lane the work-item holds in v (side
// applied for the input stage): on return v[k] holds output tid + k*WG, scaled by 1/sc, before the
// side of the output stage
template <bool OUT, int R, int E>
static inline void mach1_rt_sg_transform(const sycl::nd_item<2> & it, float * sh, float (&v)[E], const float sc) {
    constexpr int M  = 16*E;
    constexpr int WG = 16*R;
    const sycl::sub_group sg = it.get_sub_group();
    const int tid  = (int) it.get_local_id(1);
    const int lane = tid & 15;
    const int blk  = tid >> 4;
    mach1_sg_fwht<E>(sg, lane, v);
#pragma unroll
    for (int k = 0; k < E; ++k) {
        sh[blk*M + k*16 + lane] = v[k];
    }
    sycl::group_barrier(it.get_group());
    // entry (a, c) of H (OUT) or of H^T, as bit a of a word indexed by c
    const uint32_t * H = OUT ? (R == 12 ? mach1_rt_h12c.data() : mach1_rt_h20c.data())
                             : (R == 12 ? mach1_rt_h12 : mach1_rt_h20);
#pragma unroll
    for (int k = 0; k < E; ++k) {
        const int i = tid + k*WG;
        const int a = i / M;
        const int b = i % M;
        float acc = 0.0f;
#pragma unroll
        for (int c = 0; c < R; ++c) {
            const bool neg = ((H[c] >> a) & 1u) != 0u;
            const float col = sh[c*M + b];
            acc = neg ? acc - col : acc + col;
        }
        v[k] = acc/sc;
    }
}

// u for both products: work-groups nt .. 2*nt - 1 take side2, rows of dst after nt
template <int R, int E>
static void mach1_rt_pair_in(dpct::queue_ptr stream, const float * x, const float * side, const float * side2, float * dst,
                             const int nt) {
    constexpr int M  = 16*E;
    constexpr int D  = R*M;
    constexpr int WG = 16*R;
    const float sc = sqrtf((float) D);
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> sh_acc(sycl::range<1>(D), cgh);
        cgh.parallel_for(
            sycl::nd_range<2>(sycl::range<2>((size_t) 2*nt, WG), sycl::range<2>(1, WG)),
            [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
                float * sh = sh_acc.get_multi_ptr<sycl::access::decorated::no>().get();
                const int  g    = (int) it.get_group(0);
                const bool sec  = g >= nt;
                const int  t    = g - (sec ? nt : 0);
                const int  tid  = (int) it.get_local_id(1);
                const int  lane = tid & 15;
                const int  blk  = tid >> 4;
                const float * s  = x + (int64_t) t*D;
                const float * sd = sec ? side2 : side;
                float v[E];
#pragma unroll
                for (int k = 0; k < E; ++k) {
                    const int i = blk*M + k*16 + lane;
                    v[k] = sd[i]*s[i];
                }
                mach1_rt_sg_transform<false, R, E>(it, sh, v, sc);
                float * y = dst + (int64_t) g*D;
#pragma unroll
                for (int k = 0; k < E; ++k) {
                    y[tid + k*WG] = v[k];
                }
            });
    });
}

// y = silu(sv_g .* H v_g/sc) * (sv_u .* H v_u/sc); SWIGLU as ggml's
template <int R, int E>
static void mach1_rt_pair_out_swiglu(dpct::queue_ptr stream, const float * vg, const float * vu, const float * side,
                                     const float * side2, float * dst, const int nt) {
    constexpr int M  = 16*E;
    constexpr int D  = R*M;
    constexpr int WG = 16*R;
    const float sc = sqrtf((float) D);
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> sh_acc(sycl::range<1>(D), cgh);
        cgh.parallel_for(
            sycl::nd_range<2>(sycl::range<2>((size_t) nt, WG), sycl::range<2>(1, WG)),
            [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
                float * sh = sh_acc.get_multi_ptr<sycl::access::decorated::no>().get();
                const int t    = (int) it.get_group(0);
                const int tid  = (int) it.get_local_id(1);
                const int lane = tid & 15;
                const int blk  = tid >> 4;
                const float * sg_ = vg + (int64_t) t*D;
                const float * su_ = vu + (int64_t) t*D;
                float a[E], b[E];
#pragma unroll
                for (int k = 0; k < E; ++k) {
                    a[k] = sg_[blk*M + k*16 + lane];
                    b[k] = su_[blk*M + k*16 + lane];
                }
                mach1_rt_sg_transform<true, R, E>(it, sh, a, sc);
                sycl::group_barrier(it.get_group());   // the radix reads of sh are done
                mach1_rt_sg_transform<true, R, E>(it, sh, b, sc);
                float * y = dst + (int64_t) t*D;
#pragma unroll
                for (int k = 0; k < E; ++k) {
                    const int   i = tid + k*WG;
                    const float g = a[k]*side[i];
                    const float u = b[k]*side2[i];
                    y[i] = g/(1.0f + sycl::exp(-g))*u;
                }
            });
    });
}

template <bool IN, int R>
static bool mach1_rt_pair_stage_r(dpct::queue_ptr q, int M, const float * s1, const float * s2, const float * sd1,
                                  const float * sd2, float * dst, int nt) {
#define MACH1_RT_PAIR_CASE(E) \
    if (M == 16*(E)) { \
        if constexpr (IN) { mach1_rt_pair_in<R, E>(q, s1, sd1, sd2, dst, nt); } \
        else              { mach1_rt_pair_out_swiglu<R, E>(q, s1, s2, sd1, sd2, dst, nt); } \
        return true; \
    }
    MACH1_RT_PAIR_CASE(1) MACH1_RT_PAIR_CASE(2) MACH1_RT_PAIR_CASE(4) MACH1_RT_PAIR_CASE(8)
    MACH1_RT_PAIR_CASE(16) MACH1_RT_PAIR_CASE(32)
#undef MACH1_RT_PAIR_CASE
    return false;
}

static bool mach1_rt_pair_shape_ok(int d) {
    const int R = mach1_rt_radix(d);
    const int M = R > 1 ? d/R : 0;
    return (R == 12 || R == 20) && (M == 16 || M == 32 || M == 64 || M == 128 || M == 256 || M == 512);
}

bool ggml_sycl_mach1_rt_pair_swiglu(ggml_backend_sycl_context & ctx, const ggml_tensor * g, const ggml_tensor * u,
                                    ggml_tensor * out) {
    static const bool on = getenv("GGML_SYCL_MACH1_RT_PAIR") == nullptr || atoi(getenv("GGML_SYCL_MACH1_RT_PAIR")) != 0;
    const ggml_tensor * x = g->src[4];
    const int n  = (int) g->src[1]->ne[0];
    const int m  = (int) g->src[2]->ne[0];
    const int nt = (int)(x->ne[1]*x->ne[2]*x->ne[3]);
    if (!on || u->src[4] != x || u->src[3] != g->src[3] || nt > MACH1_RT_SPLIT_MAX_NT ||
        (int) u->src[1]->ne[0] != n || (int) u->src[2]->ne[0] != m || !mach1_rt_pair_shape_ok(n) || !mach1_rt_pair_shape_ok(m) ||
        g->src[0]->ne[0] != MACH1_RT_WORDS || u->src[0]->ne[0] != MACH1_RT_WORDS || !ggml_is_contiguous(x) ||
        out->type != GGML_TYPE_F32 || !ggml_is_contiguous(out) || ggml_nelements(out) != (int64_t) m*nt) {
        return false;
    }
    dpct::queue_ptr stream = ctx.stream();
    ggml_sycl_pool_alloc<float> scr(ctx.pool(), (size_t) 2*nt*n + (size_t) 2*nt*m);
    float * scr_u = scr.get();
    float * scr_v = scr.get() + (size_t) 2*nt*n;
    const int Rn = mach1_rt_radix(n);
    const int Rm = mach1_rt_radix(m);
    (Rn == 12 ? mach1_rt_pair_stage_r<true, 12> : mach1_rt_pair_stage_r<true, 20>)(
        stream, n/Rn, (const float *) x->data, nullptr, (const float *) g->src[1]->data, (const float *) u->src[1]->data,
        scr_u, nt);
    for (int h = 0; h < 2; ++h) {
        const ggml_tensor * w = h == 0 ? g : u;
        if (nt == 1) {
            mach1_rt_walk_one(stream, (const uint16_t *) w->src[0]->data, (const sycl::half *) w->src[3]->data,
                              scr_u + (size_t) h*nt*n, scr_v + (size_t) h*nt*m, n, m);
        } else {
            mach1_rt_walk<MACH1_RT_TT>(stream, (const uint16_t *) w->src[0]->data, (const sycl::half *) w->src[3]->data,
                                       scr_u + (size_t) h*nt*n, scr_v + (size_t) h*nt*m, n, m, nt);
        }
    }
    (Rm == 12 ? mach1_rt_pair_stage_r<false, 12> : mach1_rt_pair_stage_r<false, 20>)(
        stream, m/Rm, scr_v, scr_v + (size_t) nt*m, (const float *) g->src[2]->data, (const float *) u->src[2]->data,
        (float *) out->data, nt);
    return true;
}

void ggml_sycl_mach1_rt_mm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/5);

    const ggml_tensor * trellis = dst->src[0];
    const ggml_tensor * su      = dst->src[1];
    const ggml_tensor * sv      = dst->src[2];
    const ggml_tensor * tlut    = dst->src[3];
    const ggml_tensor * x       = dst->src[4];

    GGML_ASSERT(trellis->type == GGML_TYPE_I16);
    GGML_ASSERT(su->type   == GGML_TYPE_F32);
    GGML_ASSERT(sv->type   == GGML_TYPE_F32);
    GGML_ASSERT(tlut->type == GGML_TYPE_F16);
    GGML_ASSERT(x->type    == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(trellis) && ggml_is_contiguous(su) && ggml_is_contiguous(sv));
    GGML_ASSERT(ggml_is_contiguous(tlut) && ggml_is_contiguous(x) && ggml_is_contiguous(dst));
    GGML_ASSERT(trellis->ne[0] == MACH1_RT_WORDS);   // K = 4

    const int n  = (int) su->ne[0];
    const int m  = (int) sv->ne[0];
    const int nt = (int)(x->ne[1]*x->ne[2]*x->ne[3]);

    dpct::queue_ptr stream = ctx.stream();

#if GGML_SYCL_DNNL
    const bool gemm = nt >= MACH1_RT_GEMM_MIN && g_ggml_sycl_enable_dnn;
#else
    const bool gemm = false;
#endif
    // the GEMM leaves v as two halves (2*nt*m) for the output stage to add; at decode the split
    // stages take their radix pass out of place, through scr_t
    const size_t nt_tmp = nt <= MACH1_RT_SPLIT_MAX_NT ? (size_t) nt*std::max(n, m) : 0;
    ggml_sycl_pool_alloc<float> scr(ctx.pool(), (size_t) nt*n + (size_t) (gemm ? 2 : 1)*nt*m + nt_tmp);
    float * scr_u = scr.get();
    float * scr_v = scr.get() + (size_t) nt*n;
    float * scr_t = nt_tmp ? scr_v + (size_t) (gemm ? 2 : 1)*nt*m : nullptr;

    mach1_rt_stage_any<false>(stream, (const float *) x->data, (const float *) su->data, scr_u, n, nt, nullptr, scr_t);
#if GGML_SYCL_DNNL
    if (gemm) {
        mach1_rt_gemm(ctx, stream, (const uint16_t *) trellis->data, (const sycl::half *) tlut->data, scr_u, scr_v, n, m, nt);
        mach1_rt_stage_any<true>(stream, scr_v, (const float *) sv->data, (float *) dst->data, m, nt, scr_v + (size_t) nt*m);
        return;
    }
#endif
    if (nt == 1) {
        mach1_rt_walk_one(stream, (const uint16_t *) trellis->data, (const sycl::half *) tlut->data, scr_u, scr_v, n, m);
    } else {
        mach1_rt_walk<MACH1_RT_TT>(stream, (const uint16_t *) trellis->data, (const sycl::half *) tlut->data, scr_u, scr_v, n, m, nt);
    }
    mach1_rt_stage_any<true>(stream, scr_v, (const float *) sv->data, (float *) dst->data, m, nt, nullptr, scr_t);
}
