// GGML_OP_MACH1_D4_MM: port of the CUDA D4 routed-expert matmul (ggml-cuda/mach1-d4.cu).
// y[:, s, t] = diag(sv_e) H_m P_e H_n^T diag(su_e) x[:, s, t] with e = ids[s, t], P_e the
// trellis-coded int4 tiles of expert e. Three passes, as in CUDA's unfused path:
//   ustage: u = H_n (su_e * x)        (one work-group per (slot, token) pair)
//   walk:   p = P_e u                 (one work-group per 16-row block and pair, or per
//                                      block and expert when grouped)
//   redout: y = sv_e * H_m p          (one work-group per pair)
// The fp32 op order follows CUDA's: same butterfly span order, same 8-term dot per lane,
// the lane-pair sum, then the sequential sum over the 8 column ranges.

#include "mach1.hpp"

#include <array>
#include <cstring>

#define MACH1_D4_WG        256
#define MACH1_D4_NWP       8
#define MACH1_D4_TT        8
#define MACH1_D4_GROUP_MIN 4

static constexpr int8_t mach1_d4_h12[144] = {
     1, -1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
    -1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,
     1,  1,  1, -1,  1,  1, -1, -1, -1, -1,  1,  1,
     1, -1, -1, -1,  1, -1, -1,  1, -1,  1,  1, -1,
     1,  1,  1,  1,  1, -1,  1,  1, -1, -1, -1, -1,
     1, -1,  1, -1, -1, -1,  1, -1, -1,  1, -1,  1,
     1,  1, -1, -1,  1,  1,  1, -1,  1,  1, -1, -1,
     1, -1, -1,  1,  1, -1, -1, -1,  1, -1, -1,  1,
     1,  1, -1, -1, -1, -1,  1,  1,  1, -1,  1,  1,
     1, -1, -1,  1, -1,  1,  1, -1, -1, -1,  1, -1,
     1,  1,  1,  1, -1, -1, -1, -1,  1,  1,  1, -1,
     1, -1,  1, -1, -1,  1, -1,  1,  1, -1, -1, -1
};
static constexpr int8_t mach1_d4_h20[400] = {
     1, -1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
    -1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,
     1,  1,  1, -1,  1,  1,  1,  1,  1,  1, -1, -1, -1, -1,  1,  1, -1, -1, -1, -1,
     1, -1, -1, -1,  1, -1,  1, -1,  1, -1, -1,  1, -1,  1,  1, -1, -1,  1, -1,  1,
     1,  1,  1,  1,  1, -1,  1,  1, -1, -1,  1,  1, -1, -1, -1, -1,  1,  1, -1, -1,
     1, -1,  1, -1, -1, -1,  1, -1, -1,  1,  1, -1, -1,  1, -1,  1,  1, -1, -1,  1,
     1,  1,  1,  1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1, -1, -1, -1, -1,  1,  1,
     1, -1,  1, -1,  1, -1, -1, -1, -1,  1, -1,  1,  1, -1, -1,  1, -1,  1,  1, -1,
     1,  1,  1,  1, -1, -1, -1, -1,  1, -1,  1,  1,  1,  1,  1,  1, -1, -1, -1, -1,
     1, -1,  1, -1, -1,  1, -1,  1, -1, -1,  1, -1,  1, -1,  1, -1, -1,  1, -1,  1,
     1,  1, -1, -1,  1,  1, -1, -1,  1,  1,  1, -1,  1,  1, -1, -1,  1,  1, -1, -1,
     1, -1, -1,  1,  1, -1, -1,  1,  1, -1, -1, -1,  1, -1, -1,  1,  1, -1, -1,  1,
     1,  1, -1, -1, -1, -1,  1,  1,  1,  1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1,
     1, -1, -1,  1, -1,  1,  1, -1,  1, -1,  1, -1, -1, -1, -1,  1, -1,  1,  1, -1,
     1,  1,  1,  1, -1, -1, -1, -1,  1,  1, -1, -1, -1, -1,  1, -1,  1,  1,  1,  1,
     1, -1,  1, -1, -1,  1, -1,  1,  1, -1, -1,  1, -1,  1, -1, -1,  1, -1,  1, -1,
     1,  1, -1, -1,  1,  1, -1, -1, -1, -1,  1,  1, -1, -1,  1,  1,  1, -1,  1,  1,
     1, -1, -1,  1,  1, -1, -1,  1, -1,  1,  1, -1, -1,  1,  1, -1, -1, -1,  1, -1,
     1,  1, -1, -1, -1, -1,  1,  1, -1, -1, -1, -1,  1,  1,  1,  1,  1,  1,  1, -1,
     1, -1, -1,  1, -1,  1,  1, -1, -1,  1, -1,  1,  1, -1,  1, -1,  1, -1, -1, -1
};

struct mach1_d4_hash {
    int32_t v[15];
};

// the Hadamard transform of one length-d vector held in local memory: butterflies over the
// power-of-two factor M, the 1/sqrt(M) scale, then the 12- or 20-point Paley factor.
// scm = sqrtf(M) and rs = sqrtf(d/M) come from the host, so they are correctly rounded.
struct mach1_d4_fwht_shape {
    int   d;
    int   M;
    int   lm;   // log2(M)
    int   R;    // 1, 12 or 20
    float scm;
    float rs;
};

// column c of the r-point Hadamard: bit a set = entry (a, c) is -1, so a radix sum over c with
// a work-item's own row a indexes the table by the (unrolled, constant) c only
template <int R>
static constexpr std::array<uint32_t, R> mach1_d4_cols(const int8_t (&H)[R*R]) {
    std::array<uint32_t, R> t{};
    for (int c = 0; c < R; ++c) {
        for (int a = 0; a < R; ++a) {
            t[c] |= (H[a*R + c] < 0 ? 1u : 0u) << a;
        }
    }
    return t;
}
static constexpr std::array<uint32_t, 12> mach1_d4_h12c = mach1_d4_cols<12>(mach1_d4_h12);
static constexpr std::array<uint32_t, 20> mach1_d4_h20c = mach1_d4_cols<20>(mach1_d4_h20);

static mach1_d4_fwht_shape mach1_d4_shape(int d) {
    mach1_d4_fwht_shape s;
    s.d = d;
    s.R = 1;
    if ((d & (d - 1)) != 0) {
        const int m12 = d/12;
        s.R = (d % 12 == 0 && (m12 & (m12 - 1)) == 0) ? 12 : 20;
    }
    s.M  = d/s.R;
    s.lm = 0;
    while ((1 << s.lm) < s.M) {
        s.lm++;
    }
    s.scm = sqrtf((float) s.M);
    s.rs  = sqrtf((float) s.R);
    return s;
}

template <int R>
static inline float mach1_d4_radix_one(const float * sh, const mach1_d4_fwht_shape & s, const int i) {
    const int8_t * H = R == 12 ? mach1_d4_h12 : mach1_d4_h20;
    const int a = i >> s.lm;
    const int b = i & (s.M - 1);
    float acc = 0.0f;
#pragma unroll
    for (int c = 0; c < R; ++c) {
        acc += (float) H[a*R + c]*sh[c*s.M + b];
    }
    return acc/s.rs;
}

// all work-items of the group call this; on return sh holds H_d v (radix applied when
// RADIX, else only the butterflies and the 1/sqrt(M) scale)
template <bool RADIX = true>
static inline void mach1_d4_fwht(float * sh, const mach1_d4_fwht_shape & s, const sycl::nd_item<2> & it) {
    constexpr int KMAX = 8192/MACH1_D4_WG;
    const int tid = (int) it.get_local_id(1);
    const int wg  = (int) it.get_local_range(1);
    const int d   = s.d;
    for (int span = 1, ls = 0; span < s.M; span <<= 1, ++ls) {
        for (int b = tid; b < d/2; b += wg) {
            const int base = ((b >> ls) << (ls + 1)) + (b & (span - 1));
            const float a0 = sh[base];
            const float a1 = sh[base + span];
            sh[base]        = a0 + a1;
            sh[base + span] = a0 - a1;
        }
        sycl::group_barrier(it.get_group());
    }
    for (int i = tid; i < d; i += wg) {
        sh[i] = sh[i]/s.scm;
    }
    sycl::group_barrier(it.get_group());
    if (!RADIX || s.R == 1) {
        return;
    }
    float o[KMAX];
#pragma unroll
    for (int k = 0; k < KMAX; ++k) {
        const int i = tid + k*wg;
        if (i < d) {
            o[k] = s.R == 12 ? mach1_d4_radix_one<12>(sh, s, i) : mach1_d4_radix_one<20>(sh, s, i);
        }
    }
    sycl::group_barrier(it.get_group());
#pragma unroll
    for (int k = 0; k < KMAX; ++k) {
        const int i = tid + k*wg;
        if (i < d) {
            sh[i] = o[k];
        }
    }
    sycl::group_barrier(it.get_group());
}

static inline int mach1_d4_id(const int32_t * ids, int p, int n_used, int nb0, int nb1) {
    return ids[(p % n_used)*nb0 + (p / n_used)*nb1];
}

// The u of the XMX walk (see there): u = hi + lo in fp16, lo the rounding error of hi, under a
// per-pair power-of-two scale, as two row-major planes (hi, then lo at ua + lo_off) with the pairs
// in expert-sorted order. The input stage writes it directly when ua is set. (lo is not scaled up:
// where it falls below the fp16 normal range it is of no weight in a product with the row's
// larger elements, and the two DPAS of a tile then chain without a multiply in between.)
struct mach1_d4_split {
    sycl::half    * ua;
    float         * inv_s;   // 1/scale per sorted slot
    const int32_t * pos;     // sorted slot of each pair
    int64_t         lo_off;
};

static int mach1_ceil_log2(float x) {
    int e = 0;
    const float f = sycl::frexp(x, &e);   // x = f * 2^e, f in [0.5, 1)
    return f == 0.5f ? e - 1 : e;
}

// the per-pair scale for a vector with max |u| <= bound: max |u*s| <= 2^13
static inline float mach1_d4_split_scale(float bound) {
    return bound > 0.0f && bound < 3.0e38f ? sycl::ldexp(1.0f, 13 - mach1_ceil_log2(bound)) : 1.0f;
}

// the hi/lo pair of x = u*s
static inline uint32_t mach1_d4_split_hl(float x, sycl::half & lo) {
    const sycl::half hi = (sycl::half) x;
    lo = (sycl::half) (x - (float) hi);
    return sycl::bit_cast<uint16_t>(hi);
}

// pair p's row of the split u; work-item tid holds u[tid + k*wg] = v(k) for k < K, all items of
// the group call this
template <int K, typename V>
static inline void mach1_d4_split_row(const sycl::nd_item<2> & it, const mach1_d4_split & sp, const V & v, int p,
                                      int tid, int wg, int d) {
    float mx = 0.0f;
#pragma unroll
    for (int k = 0; k < K; ++k) {
        if (tid + k*wg < d) {
            mx = sycl::fmax(mx, sycl::fabs(v(k)));
        }
    }
    mx = sycl::reduce_over_group(it.get_group(), mx, sycl::maximum<float>());
    const float  s  = mach1_d4_split_scale(mx);
    const int    k0 = sp.pos[p];
    sycl::half * uh = sp.ua + (int64_t) k0*d;
#pragma unroll
    for (int k = 0; k < K; ++k) {
        const int i = tid + k*wg;
        if (i < d) {
            const float      x  = v(k)*s;
            const sycl::half hi = (sycl::half) x;
            uh[i]           = hi;
            uh[sp.lo_off + i] = (sycl::half) (x - (float) hi);
        }
    }
    if (tid == 0) {
        sp.inv_s[k0] = 1.0f/s;
    }
}

// u = H_n (su_e * x) for one pair per work-group (into the split u when sp.ua is set)
static void mach1_d4_ustage(dpct::queue_ptr stream, const float * x, const sycl::half * su, const int32_t * ids,
                            float * scr_u, const mach1_d4_fwht_shape s, const int n_pairs, const int n_used,
                            const int xne1, const int nb0, const int nb1, const mach1_d4_split sp) {
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> sh_acc(sycl::range<1>(s.d), cgh);
        cgh.parallel_for(
            sycl::nd_range<2>(sycl::range<2>(n_pairs, MACH1_D4_WG), sycl::range<2>(1, MACH1_D4_WG)),
            [=](sycl::nd_item<2> it) {
                float * sh = sh_acc.get_multi_ptr<sycl::access::decorated::no>().get();
                const int p = (int) it.get_group(0);
                const int e = mach1_d4_id(ids, p, n_used, nb0, nb1);
                const int t = p / n_used;
                const int d = s.d;
                const float      * xc = x  + (int64_t)(xne1 == 1 ? t : p)*d;
                const sycl::half * se = su + (int64_t) e*d;
                for (int i = (int) it.get_local_id(1); i < d; i += MACH1_D4_WG) {
                    sh[i] = (float) se[i]*xc[i];
                }
                sycl::group_barrier(it.get_group());
                mach1_d4_fwht(sh, s, it);
                const int tid = (int) it.get_local_id(1);
                if (sp.ua) {
                    mach1_d4_split_row<8192/MACH1_D4_WG>(it, sp, [&](int k) { return sh[tid + k*MACH1_D4_WG]; },
                                                         p, tid, MACH1_D4_WG, d);
                    return;
                }
                float * u = scr_u + (int64_t) p*d;
                for (int i = tid; i < d; i += MACH1_D4_WG) {
                    u[i] = sh[i];
                }
            });
    });
}

// map[0..E] = per-expert pair offsets, map[E + 1 + k] = the pairs grouped by expert, and with
// pos (the XMX walk) pos[p] = the sorted slot of pair p. With units, the walk's work list: unit u
// is the expert units[2*u] and its pairs from units[2*u + 1] on, at most `unit` of them (expert
// -1: no work), n_units entries.
// One work-group counts the pairs per expert, turns the counts into offsets and scatters the
// pairs. With E <= WG the offsets are a prefix sum in two levels (within the sub-groups, then
// over the sub-groups' totals) and every work-item writes its expert's units; a single work-item
// walking the experts, which is what larger E fall back to, took longer than everything else here.
static void mach1_d4_map(dpct::queue_ptr stream, const int32_t * ids, int32_t * map, int32_t * pos, const int E,
                         const int n_used, const int P, const int nb0, const int nb1, int32_t * units = nullptr,
                         const int unit = 1, const int n_units = 0) {
    constexpr int WG  = 1024;
    constexpr int NSG = WG/16;
    const bool par = E <= WG;
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<int, 1> cnt_acc(sycl::range<1>(E), cgh);
        sycl::local_accessor<int, 1> tot_acc(sycl::range<1>(2*NSG), cgh);
        cgh.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(WG), sycl::range<1>(WG)),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                int * cnt = cnt_acc.get_multi_ptr<sycl::access::decorated::no>().get();
                int * tot = tot_acc.get_multi_ptr<sycl::access::decorated::no>().get();
                const int tid = (int) it.get_local_id(0);
                for (int e = tid; e < E; e += WG) {
                    cnt[e] = 0;
                }
                sycl::group_barrier(it.get_group());
                for (int p = tid; p < P; p += WG) {
                    sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::work_group,
                                     sycl::access::address_space::local_space> a(cnt[mach1_d4_id(ids, p, n_used, nb0, nb1)]);
                    a.fetch_add(1);
                }
                sycl::group_barrier(it.get_group());
                if (par) {
                    // work-item tid is expert tid
                    const auto sg   = it.get_sub_group();
                    const int  lane = tid & 15;
                    const int  sgi  = tid >> 4;
                    const int  c    = tid < E ? cnt[tid] : 0;
                    const int  nu   = units ? (c + unit - 1)/unit : 0;
                    const int  pc   = sycl::exclusive_scan_over_group(sg, c,  sycl::plus<int>());
                    const int  pu   = sycl::exclusive_scan_over_group(sg, nu, sycl::plus<int>());
                    const int  tc   = sycl::reduce_over_group(sg, c,  sycl::plus<int>());
                    const int  tu   = sycl::reduce_over_group(sg, nu, sycl::plus<int>());
                    if (lane == 0) {
                        tot[sgi]       = tc;
                        tot[NSG + sgi] = tu;
                    }
                    sycl::group_barrier(it.get_group());
                    // the pairs and units of the sub-groups before this one, and all units
                    int bc = 0, bu = 0, au = 0;
#pragma unroll
                    for (int k = 0; k < NSG/16; ++k) {
                        const int i  = k*16 + lane;
                        const int vc = tot[i];
                        const int vu = tot[NSG + i];
                        bc += i < sgi ? vc : 0;
                        bu += i < sgi ? vu : 0;
                        au += vu;
                    }
                    bc = sycl::reduce_over_group(sg, bc, sycl::plus<int>());
                    bu = sycl::reduce_over_group(sg, bu, sycl::plus<int>());
                    au = sycl::reduce_over_group(sg, au, sycl::plus<int>());
                    if (tid < E) {
                        map[tid] = bc + pc;
                        cnt[tid] = bc + pc;
                    }
                    if (tid == 0) {
                        map[E] = P;
                    }
                    if (units) {
                        int j = bu + pu;
                        for (int k0 = 0; k0 < c; k0 += unit, ++j) {
                            units[2*j]     = tid;
                            units[2*j + 1] = k0;
                        }
                        for (int u = au + tid; u < n_units; u += WG) {
                            units[2*u]     = -1;
                            units[2*u + 1] = 0;
                        }
                    }
                } else if (tid == 0) {
                    int acc = 0;
                    int nu  = 0;
                    for (int e = 0; e < E; ++e) {
                        const int c = cnt[e];
                        map[e] = acc;
                        cnt[e] = acc;
                        acc += c;
                        if (units) {
                            for (int k0 = 0; k0 < c; k0 += unit) {
                                units[2*nu]     = e;
                                units[2*nu + 1] = k0;
                                ++nu;
                            }
                        }
                    }
                    map[E] = acc;
                    if (units) {
                        for (; nu < n_units; ++nu) {
                            units[2*nu]     = -1;
                            units[2*nu + 1] = 0;
                        }
                    }
                }
                sycl::group_barrier(it.get_group());
                // the order of the pairs within an expert does not change any result
                for (int p = tid; p < P; p += WG) {
                    const int e = mach1_d4_id(ids, p, n_used, nb0, nb1);
                    sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::work_group,
                                     sycl::access::address_space::local_space> a(cnt[e]);
                    const int k = a.fetch_add(1);
                    map[E + 1 + k] = p;
                    if (pos) {
                        pos[p] = k;
                    }
                }
            });
    });
}

// p = P_e u for one 16-row block: 8 groups of 16 work-items, each group over a contiguous range
// of the 16-column blocks, work-item r of a group on tile row r. A row is the 4 trellis states
// 4r .. 4r + 3 (4 weights each); the fp32 order is CUDA's: the 8-term dots of columns 0-7 and
// 8-15 accumulate separately over the range, their sum is the range's row partial, and the 8
// partials are summed in range order.
// NWP sub-groups per row block: about 20 column tiles each does best (fewer sub-groups mean less
// of the partial-sum reduction and its barriers), see mach1_d4_walk_nwp
template <bool GROUPED, bool HASHED, int NWP = MACH1_D4_NWP>
static void mach1_d4_walk(dpct::queue_ptr stream, const uint16_t * trellis, const int32_t * offs, const float * gw,
                          const uint16_t * zt, const float * units, const int32_t * ids, const int32_t * map,
                          const float * scr_u, float * scr_p, const int n, const int Mb, const int Nb,
                          const int n_used, const int n_expert, const int nb0, const int nb1, const int n_rows,
                          const mach1_d4_hash hash) {
    constexpr int MACH1_D4_WALK_WG = NWP*16;
    // the column tiles split into RNG ranges whatever NWP is (CUDA's order: the range partials are
    // summed in range order); a sub-group takes RPS consecutive ranges
    constexpr int RNG = MACH1_D4_NWP;
    constexpr int RPS = RNG/NWP;
    static_assert(RNG % NWP == 0 && (!GROUPED || NWP == RNG), "fewer sub-groups only for the ungrouped walk");
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> part_acc(sycl::range<1>(RNG*16), cgh);
        // lev2[b] = (level of nibble b & 15, level of nibble b >> 4): two weights per lookup
        sycl::local_accessor<sycl::half2, 1> lev_acc(sycl::range<1>(256), cgh);
        // the row block's trellis words (Nb tiles of at most 32), staged with coalesced loads
        sycl::local_accessor<uint32_t, 1> tw_acc(sycl::range<1>((size_t) Nb*16), cgh);
        cgh.parallel_for(
            sycl::nd_range<2>(sycl::range<2>(n_rows, (size_t) Mb*MACH1_D4_WALK_WG), sycl::range<2>(1, MACH1_D4_WALK_WG)),
            [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
                float        * part = part_acc.get_multi_ptr<sycl::access::decorated::no>().get();
                sycl::half2  * lev2 = lev_acc.get_multi_ptr<sycl::access::decorated::no>().get();
                uint32_t     * stw  = tw_acc.get_multi_ptr<sycl::access::decorated::no>().get();
                const int tid = (int) it.get_local_id(1);
                const int rb  = (int) it.get_group(1);
                const int row = tid & 15;
                const int wid = tid >> 4;
                const int m   = Mb*16;
                const sycl::sub_group sg = it.get_sub_group();

                int e, np, pl0;
                if constexpr (GROUPED) {
                    e   = (int) it.get_group(0);
                    pl0 = map[e];
                    np  = map[e + 1] - pl0;
                    if (np == 0) {
                        return;
                    }
                } else {
                    pl0 = (int) it.get_group(0);
                    e   = mach1_d4_id(ids, pl0, n_used, nb0, nb1);
                    np  = 1;
                }

                const int      K4    = offs[2*e + 1];
                const int      words = 4*K4;
                const int      r     = K4 - 4;
                const float    un    = units[r];
                const uint32_t mult  = HASHED ? (uint32_t) hash.v[3*r] : 0u;
                const uint16_t * zr  = zt + (int64_t) r*65536;
                const uint16_t * tr  = trellis + (int64_t) offs[2*e] + (int64_t) rb*Nb*words;
                const float    * gwe = gw + (int64_t) e*(Mb + Nb);

                // hashed codebooks: the 16 signed 4-bit levels of hash word pair r (32-bit ops only:
                // 64-bit integer shifts are emulated on Intel GPUs); tabled ones: nibble - 8
                for (int b = tid; b < 256; b += MACH1_D4_WALK_WG) {
                    float l[2];
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const int k = (b >> (4*h)) & 15;
                        if (HASHED) {
                            const uint32_t half = (uint32_t) hash.v[3*r + 1 + (k >> 3)];
                            l[h] = (float)(((int32_t)(half << (28 - 4*(k & 7)))) >> 28);
                        } else {
                            l[h] = (float)(k - 8);
                        }
                    }
                    // small integers: exact in fp16, half the local memory traffic of float2
                    lev2[b] = sycl::half2((sycl::half) l[0], (sycl::half) l[1]);
                }
                {
                    // words = 4*K4 is even, so the slice is whole 32-bit pairs, 4-byte aligned
                    // whenever the expert offset is even
                    const int nw2 = Nb*words/2;
                    if ((offs[2*e] & 1) == 0) {
                        const uint32_t * src = (const uint32_t *) tr;
                        for (int i = tid; i < nw2; i += MACH1_D4_WALK_WG) {
                            stw[i] = src[i];
                        }
                    } else {
                        uint16_t * dst16 = (uint16_t *) stw;
                        for (int i = tid; i < 2*nw2; i += MACH1_D4_WALK_WG) {
                            dst16[i] = tr[i];
                        }
                    }
                }
                sycl::group_barrier(it.get_group());
                const uint16_t * stw16 = (const uint16_t *) stw;

                // state 4*row + s starts at bit b0 + s*K4 of the tile; the four windows lie in the
                // 64 bits of words wi .. wi + 3 (wrapping), at offsets o + s*K4 <= 39
                const int b0  = 4*row*K4;
                const int wi  = b0 >> 4;
                const int o   = b0 & 15;
                const int wi1 = wi + 1 < words ? wi + 1 : wi + 1 - words;
                const int wi2 = wi + 2 < words ? wi + 2 : wi + 2 - words;
                const int wi3 = wi + 3 < words ? wi + 3 : wi + 3 - words;

                const int depth = (Nb + RNG - 1)/RNG;

                // the 4 levels of state st, as two lookups of two
                const auto levels = [&](uint32_t st, float * w) {
                    const uint32_t h = HASHED ? (st*mult) >> 16 : (uint32_t) zr[st];
                    const sycl::float2 a = lev2[h & 0xFFu].convert<float>();
                    const sycl::float2 b = lev2[(h >> 8) & 0xFFu].convert<float>();
                    w[0] = a.x(); w[1] = a.y(); w[2] = b.x(); w[3] = b.y();
                };

                constexpr int TT = GROUPED ? MACH1_D4_TT : 1;
                for (int k0 = 0; k0 < np; k0 += TT) {
                    const int ntt = sycl::min(TT, np - k0);
                    int pk[TT];
#pragma unroll
                    for (int k = 0; k < TT; ++k) {
                        pk[k] = GROUPED ? (k < ntt ? map[n_expert + 1 + pl0 + k0 + k] : 0) : pl0;
                    }
                    float acc0[TT], acc1[TT];
                    // the 16 weights of this row in tile cb, and the tile's wave scale
                    const auto decode = [&](int cb, float * ws) {
                        const uint16_t * tw = stw16 + cb*words;
                        const uint32_t w0 = tw[wi], w1 = tw[wi1], w2 = tw[wi2], w3 = tw[wi3];
                        const uint32_t x01 = (w0 << 16) | w1;
                        const uint32_t x12 = (w1 << 16) | w2;
                        const uint32_t x23 = (w2 << 16) | w3;
#pragma unroll
                        for (int s = 0; s < 4; ++s) {
                            const int b = o + s*K4;
                            const uint32_t win = b <= 16 ? x01 >> (16 - b) : b <= 32 ? x12 >> (32 - b) : x23 >> (48 - b);
                            levels(win & 0xFFFFu, ws + 4*s);
                        }
                        const int wv = rb + cb <= Nb - 1 ? Mb + Nb - 1 - (rb + cb) : Mb + Nb - 2 - (rb + cb);
                        return un*gwe[wv];
                    };
                    const auto apply = [&](int cb, const float * ws, float sc) {
                        const int uo = cb*16;
#pragma unroll
                        for (int k = 0; k < TT; ++k) {
                            if (k < ntt) {
                                // u straight from global (L1): staging it in local memory costs
                                // occupancy. The tile's 16 u are the same for every lane: one 64-byte
                                // load (lane l takes u[l]) and broadcasts
                                const float uv = scr_u[(int64_t) pk[k]*n + uo + row];
                                const auto  up = [&](int j) { return sycl::select_from_group(sg, uv, j); };
                                acc0[k] += sc*(ws[0]*up(0) + ws[1]*up(1) + ws[2]*up(2) + ws[3]*up(3)
                                             + ws[4]*up(4) + ws[5]*up(5) + ws[6]*up(6) + ws[7]*up(7));
                                acc1[k] += sc*(ws[8]*up(8) + ws[9]*up(9) + ws[10]*up(10) + ws[11]*up(11)
                                             + ws[12]*up(12) + ws[13]*up(13) + ws[14]*up(14) + ws[15]*up(15));
                            }
                        }
                    };
                    for (int j = 0; j < RPS; ++j) {
                        const int rg  = wid*RPS + j;
                        const int cb0 = rg*depth;
                        const int cbe = sycl::min(cb0 + depth, Nb);
#pragma unroll
                        for (int k = 0; k < TT; ++k) {
                            acc0[k] = 0.0f;
                            acc1[k] = 0.0f;
                        }
                        // two tiles per step: both decodes are independent, the accumulation stays
                        // in tile order
                        int cb = cb0;
                        for (; cb + 1 < cbe; cb += 2) {
                            float wa[16], wb[16];
                            const float sa = decode(cb, wa);
                            const float sb = decode(cb + 1, wb);
                            apply(cb, wa, sa);
                            apply(cb + 1, wb, sb);
                        }
                        if (cb < cbe) {
                            float wa[16];
                            const float sa = decode(cb, wa);
                            apply(cb, wa, sa);
                        }
                        if constexpr (!GROUPED) {
                            part[rg*16 + row] = acc0[0] + acc1[0];
                        }
                    }
                    if constexpr (!GROUPED) {
                        sycl::group_barrier(it.get_group());
                        if (tid < 16) {
                            float s2 = 0.0f;
                            for (int w = 0; w < RNG; ++w) {
                                s2 += part[w*16 + tid];
                            }
                            scr_p[(int64_t) pk[0]*m + rb*16 + tid] = s2;
                        }
                    } else {
#pragma unroll
                        for (int k = 0; k < TT; ++k) {
                            if (k < ntt) {
                                part[tid] = acc0[k] + acc1[k];
                                sycl::group_barrier(it.get_group());
                                if (tid < 16) {
                                    float s2 = 0.0f;
                                    for (int w = 0; w < NWP; ++w) {
                                        s2 += part[w*16 + tid];
                                    }
                                    scr_p[(int64_t) pk[k]*m + rb*16 + tid] = s2;
                                }
                                sycl::group_barrier(it.get_group());
                            }
                        }
                    }
                }
            });
    });
}

//
// the walk with the sub-group count for Nb column tiles
template <bool GROUPED, bool HASHED>
static void mach1_d4_walk_any(dpct::queue_ptr stream, const uint16_t * trellis, const int32_t * offs, const float * gw,
                              const uint16_t * zt, const float * units, const int32_t * ids, const int32_t * map,
                              const float * scr_u, float * scr_p, const int n, const int Mb, const int Nb,
                              const int n_used, const int n_expert, const int nb0, const int nb1, const int n_rows,
                              const mach1_d4_hash hash) {
    if (GROUPED || Nb >= 128) {
        mach1_d4_walk<GROUPED, HASHED, 8>(stream, trellis, offs, gw, zt, units, ids, map, scr_u, scr_p, n, Mb, Nb, n_used,
                                          n_expert, nb0, nb1, n_rows, hash);
    } else if (Nb >= 64) {
        mach1_d4_walk<false, HASHED, 4>(stream, trellis, offs, gw, zt, units, ids, map, scr_u, scr_p, n, Mb, Nb, n_used,
                                          n_expert, nb0, nb1, n_rows, hash);
    } else {
        mach1_d4_walk<false, HASHED, 2>(stream, trellis, offs, gw, zt, units, ids, map, scr_u, scr_p, n, Mb, Nb, n_used,
                                          n_expert, nb0, nb1, n_rows, hash);
    }
}

// Batches on the matrix engines (grouped walk, MACH1_D4_XMX_MIN+ tokens). Per expert e and 16-row
// block: C[pair][row] = sum over tiles of sc_tile * U[pair][cols] W_tile^T[cols][row], W the exact
// integer levels in fp16 and U an fp16 hi/lo pair (u = hi + lo, to ~2^-22) under a per-pair
// power-of-two scale that keeps it in fp16 range. Per tile, as CUDA's tensor-core walk:
//   t = U_l W + U_h W,   C += sc*t (fp32),   p = C / s_pair.
// Lane r of a sub-group decodes tile row r, which is exactly its slice of the DPAS B operand, so
// the tile goes from the decode straight into the matrix engine without local memory. The input
// stage writes u in expert-sorted order (see mach1_d4_split), and A is one 2D block read of 8
// sorted rows: rows past an expert's own reach only rows of C that are not stored.
//
// A sub-group takes two column tiles per step (one block read brings the A operands of both), and
// its tile loop has no branches: the pair groups of the unit are a template parameter.
//
// The experts' loads are far from even (in a 512-token batch of the model a few experts get nearly
// every token and half get none), and a sub-group that took all of an expert's pairs, 64 per
// pass over its tiles, would be a chain many times longer than the others while the device
// idles. So the work is cut into units of one expert and at most 64 of its pairs (mach1_d4_map),
// each a work-group of its own: every sub-group makes one pass.
#define MACH1_D4_XMX_MIN 16
#define MACH1_D4_XMX_M   8      // pairs per DPAS (rows of A and C)
#define MACH1_D4_XMX_CH  8      // pair groups per unit
#define MACH1_D4_XMX_LA  1      // pair groups whose A operands are read ahead
#define MACH1_D4_XMX_SG  8      // sub-groups (16-row blocks of one expert) per work-group
#define MACH1_D4_XMX_UNIT (MACH1_D4_XMX_M*MACH1_D4_XMX_CH)   // pairs per unit

struct mach1_d4_xmx_args {
    const uint16_t * tr;      // the row block's tiles
    const uint32_t * lev2h;
    const uint16_t * zr;
    const float    * gwe;
    const int32_t  * map;
    mach1_d4_split   sp;
    float          * scr_p;
    uint32_t         mult;
    float            un;
    int row, rb, Mb, Nb, np, pl0, n, P, n_expert;   // np pairs from sorted slot pl0 on
};

// one sub-group's 16-row block over all column tiles for the np <= M*NCH pairs of a unit, for K4 =
// 4 .. 8 bits per state and NCH pair groups. The group count is a template parameter so that the
// tile loop is straight-line code: a thread is bound by its chain of loads waited for, and with
// the A operands of LA groups read ahead (the first ones before the tile is decoded) the block
// reads of a tile are in flight together instead of one group's after the other's DPAS.
template <bool HASHED, int K4, int NCH>
static inline void mach1_d4_xmx_rb(const mach1_d4_xmx_args & a) {
    constexpr int M     = MACH1_D4_XMX_M;
    constexpr int words = 4*K4;
    constexpr int LA    = MACH1_D4_XMX_LA < NCH ? MACH1_D4_XMX_LA : NCH;
    const int row = a.row;
    const int Mb  = a.Mb;
    const int Nb  = a.Nb;
    const int np  = a.np;
    const int pl0 = a.pl0;
    const int m   = Mb*16;

    // state 4*row + s starts at bit b0 + s*K4 of the tile; with the 64 bits of words wi .. wi + 3
    // (wrapping) shifted left by o, its window is at the constant offset s*K4
    const int b0  = 4*row*K4;
    const int wi  = b0 >> 4;
    const int o   = b0 & 15;
    const int wi1 = wi + 1 < words ? wi + 1 : wi + 1 - words;
    const int wi2 = wi + 2 < words ? wi + 2 : wi + 2 - words;
    const int wi3 = wi + 3 < words ? wi + 3 : wi + 3 - words;

    const uint16_t * p0 = a.tr + wi;
    const uint16_t * p1 = a.tr + wi1;
    const uint16_t * p2 = a.tr + wi2;
    const uint16_t * p3 = a.tr + wi3;

    const sycl::half * uh = a.sp.ua;
    const sycl::half * ul = a.sp.ua + a.sp.lo_off;

    mach1_f8v acc[NCH];
#pragma unroll
    for (int c = 0; c < NCH; ++c) {
        acc[c] = 0.0f;
    }
    // two tiles per step, their loads and lookups in flight together; an odd Nb ends on the tiles
    // Nb - 2 and Nb - 1 with the scale of the first, which the step before took, at zero
    const int steps = (Nb + 1)/2;
    for (int i = 0; i < steps; ++i) {
        const int cb = 2*i <= Nb - 2 ? 2*i : Nb - 2;
        mach1_s16v al[NCH], ah[NCH];
#pragma unroll
        for (int c = 0; c < LA; ++c) {
            al[c] = mach1_load_a2(ul, 2*a.n, a.P, cb*16, pl0 + c*M);
            ah[c] = mach1_load_a2(uh, 2*a.n, a.P, cb*16, pl0 + c*M);
        }
        // the lane's four words through pointers of its own: one 64-bit add per load
        const int tw = cb*words;
        uint32_t x01[2], x23[2];
#pragma unroll
        for (int u = 0; u < 2; ++u) {
            x01[u] = ((uint32_t) p0[tw + u*words] << 16) | p1[tw + u*words];
            x23[u] = ((uint32_t) p2[tw + u*words] << 16) | p3[tw + u*words];
        }
        mach1_i8v b[2];
        float     sc[2];
#pragma unroll
        for (int u = 0; u < 2; ++u) {
            // o < 16; the split shift keeps o = 0 defined
            const uint32_t hi = (x01[u] << o) | ((x23[u] >> 1) >> (31 - o));
            const uint32_t lo = x23[u] << o;
#pragma unroll
            for (int s = 0; s < 4; ++s) {
                const int c = s*K4;
                const uint32_t win = (c <= 16 ? hi >> (16 - c) : (hi << (c - 16)) | (lo >> (48 - c))) & 0xFFFFu;
                const uint32_t h = HASHED ? (win*a.mult) >> 16 : (uint32_t) a.zr[win];
                b[u][2*s + 0] = (int) a.lev2h[h & 0xFFu];
                b[u][2*s + 1] = (int) a.lev2h[(h >> 8) & 0xFFu];
            }
            const int d  = a.rb + cb + u;
            const int wv = d <= Nb - 1 ? Mb + Nb - 1 - d : Mb + Nb - 2 - d;
            sc[u] = a.un*a.gwe[wv];
        }
        if (cb != 2*i) {
            sc[0] = 0.0f;
        }
#pragma unroll
        for (int c = 0; c < NCH; ++c) {
            if (c + LA < NCH) {
                al[c + LA] = mach1_load_a2(ul, 2*a.n, a.P, cb*16, pl0 + (c + LA)*M);
                ah[c + LA] = mach1_load_a2(uh, 2*a.n, a.P, cb*16, pl0 + (c + LA)*M);
            }
#pragma unroll
            for (int u = 0; u < 2; ++u) {
                const mach1_s8v l = u == 0 ? __builtin_shufflevector(al[c], al[c], 0, 1, 2, 3, 4, 5, 6, 7)
                                           : __builtin_shufflevector(al[c], al[c], 8, 9, 10, 11, 12, 13, 14, 15);
                const mach1_s8v h = u == 0 ? __builtin_shufflevector(ah[c], ah[c], 0, 1, 2, 3, 4, 5, 6, 7)
                                           : __builtin_shufflevector(ah[c], ah[c], 8, 9, 10, 11, 12, 13, 14, 15);
                mach1_f8v t = 0.0f;
                t = mach1_dpas(t, l, b[u]);
                t = mach1_dpas(t, h, b[u]);
                acc[c] += sc[u]*t;
            }
        }
    }
#pragma unroll
    for (int c = 0; c < NCH; ++c) {
#pragma unroll
        for (int q = 0; q < M; ++q) {
            const int k = c*M + q;
            if (k < np) {
                a.scr_p[(int64_t) a.map[a.n_expert + 1 + pl0 + k]*m + a.rb*16 + row] = acc[c][q]*a.sp.inv_s[pl0 + k];
            }
        }
    }
}

// the group count of a unit, rounded up to one of the instantiated loops: a group past the unit's
// own reads pairs of the next experts (or zeros past the end) into rows of C that are not stored
template <bool HASHED, int K4>
static inline void mach1_d4_xmx_rb_k(const mach1_d4_xmx_args & a) {
    switch ((a.np + MACH1_D4_XMX_M - 1)/MACH1_D4_XMX_M) {
        case 1:  mach1_d4_xmx_rb<HASHED, K4, 1>(a); break;
        case 2:  mach1_d4_xmx_rb<HASHED, K4, 2>(a); break;
        case 3:  mach1_d4_xmx_rb<HASHED, K4, 3>(a); break;
        case 4:  mach1_d4_xmx_rb<HASHED, K4, 4>(a); break;
        case 5:
        case 6:  mach1_d4_xmx_rb<HASHED, K4, 6>(a); break;
        default: mach1_d4_xmx_rb<HASHED, K4, 8>(a); break;
    }
}

template <bool HASHED>
static void mach1_d4_walk_xmx(dpct::queue_ptr stream, const uint16_t * trellis, const int32_t * offs, const float * gw,
                              const uint16_t * zt, const float * units, const int32_t * map, const int32_t * work,
                              const int n_work, const mach1_d4_split sp, float * scr_p, const int n, const int Mb,
                              const int Nb, const int n_expert, const int P, const mach1_d4_hash hash) {
    constexpr int SG = MACH1_D4_XMX_SG;
    stream->submit([&](sycl::handler & cgh) {
        // lev2h[b] = (level of nibble b & 15, level of nibble b >> 4) as a half2: one lookup is one
        // packed pair of B values
        sycl::local_accessor<uint32_t, 1> lev_acc(sycl::range<1>(256), cgh);
        cgh.parallel_for(
            sycl::nd_range<2>(sycl::range<2>(n_work, (size_t) ((Mb + SG - 1)/SG)*SG*16), sycl::range<2>(1, SG*16)),
            [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
                uint32_t * lev2h = lev_acc.get_multi_ptr<sycl::access::decorated::no>().get();
                const int tid = (int) it.get_local_id(1);
                const int row = tid & 15;
                // the same in every lane, and known to the compiler as such: what depends only on the row
                // block (the tile scale, the base of its tiles) is then scalar code
                const int sgi = sycl::group_broadcast(it.get_sub_group(), tid >> 4);
                const int wu  = (int) it.get_group(0);
                const int e   = work[2*wu];
                const int rb  = (int) it.get_group(1)*SG + sgi;
                if (e < 0) {
                    return;
                }
                const int pl0 = map[e] + work[2*wu + 1];
                const int np  = sycl::min(MACH1_D4_XMX_UNIT, map[e + 1] - pl0);

                const int      K4    = offs[2*e + 1];
                const int      words = 4*K4;
                const int      r     = K4 - 4;
                const float    un    = units[r];
                const uint32_t mult  = HASHED ? (uint32_t) hash.v[3*r] : 0u;
                const uint16_t * zr  = zt + (int64_t) r*65536;
                const float    * gwe = gw + (int64_t) e*(Mb + Nb);

                for (int b = tid; b < 256; b += SG*16) {
                    sycl::half l[2];
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const int k = (b >> (4*h)) & 15;
                        if (HASHED) {
                            const uint32_t half = (uint32_t) hash.v[3*r + 1 + (k >> 3)];
                            l[h] = (sycl::half) (float)(((int32_t)(half << (28 - 4*(k & 7)))) >> 28);
                        } else {
                            l[h] = (sycl::half) (float)(k - 8);
                        }
                    }
                    lev2h[b] = (uint32_t) sycl::bit_cast<uint16_t>(l[0]) | ((uint32_t) sycl::bit_cast<uint16_t>(l[1]) << 16);
                }
                sycl::group_barrier(it.get_group());
                if (rb >= Mb) {
                    return;
                }
                const uint16_t * tr = trellis + (int64_t) offs[2*e] + (int64_t) rb*Nb*words;
                const mach1_d4_xmx_args a = { tr, lev2h, zr, gwe, map, sp, scr_p, mult, un, row, rb, Mb, Nb, np, pl0, n, P,
                                              n_expert };
                // K4 is uniform per expert: as a constant, the window offsets of every state are too
                switch (K4) {
                    case 4:  mach1_d4_xmx_rb_k<HASHED, 4>(a); break;
                    case 5:  mach1_d4_xmx_rb_k<HASHED, 5>(a); break;
                    case 6:  mach1_d4_xmx_rb_k<HASHED, 6>(a); break;
                    case 7:  mach1_d4_xmx_rb_k<HASHED, 7>(a); break;
                    default: mach1_d4_xmx_rb_k<HASHED, 8>(a); break;
                }
            });
    });
}

//
// The single-token walk on the matrix engine (decode): the tile decode of the batch walk above, one
// 2-row DPAS per tile with the tile's u split in fp16 hi and lo rows (mach1.hpp) in place of 16
// fp16 -> fp32 conversions and 32 multiplies and adds. Tolerance class, as the batch walk: it
// leaves CUDA's fp32 order, which the SIMT walk keeps. A work-group is 8 sub-groups of one pair:
// 8/NR row blocks, each with its column tiles in NR ranges; a sub-group takes 2 tiles per step,
// each with its own accumulator, so their loads and lookups are in flight together.
//
#define MACH1_D4_XMX1_SG 8
#define MACH1_D4_XMX1_U  2

struct mach1_d4_xmx1_args {
    const uint16_t * tr;      // the row block's tiles
    const uint32_t * lev2h;
    const uint16_t * zr;
    const float    * gwe;
    const float    * u;       // the pair's u
    uint32_t         mult;
    int row, rb, Mb, Nb, cb0, cbe;
};

// the partial of one row over the column tiles cb0 .. cbe - 1, without the unit scale
template <bool HASHED, int K4>
static inline float mach1_d4_xmx1_rb(const mach1_d4_xmx1_args & a) {
    constexpr int U     = MACH1_D4_XMX1_U;
    constexpr int words = 4*K4;
    const int row = a.row;
    const int Mb  = a.Mb;
    const int Nb  = a.Nb;

    // as in mach1_d4_xmx_rb: the row's 64 bits, shifted left by o, hold the window of state s at
    // the constant offset s*K4
    const int b0  = 4*row*K4;
    const int wi  = b0 >> 4;
    const int o   = b0 & 15;
    const int wi1 = wi + 1 < words ? wi + 1 : wi + 1 - words;
    const int wi2 = wi + 2 < words ? wi + 2 : wi + 2 - words;
    const int wi3 = wi + 3 < words ? wi + 3 : wi + 3 - words;

    const auto tile = [&](mach1_f2v acc, uint32_t w0, uint32_t w1, uint32_t w2, uint32_t w3, float u, float sc) {
        const uint32_t x01 = (w0 << 16) | w1;
        const uint32_t x23 = (w2 << 16) | w3;
        // o < 16; the split shift keeps o = 0 defined
        const uint32_t hi = (x01 << o) | ((x23 >> 1) >> (31 - o));
        const uint32_t lo = x23 << o;
        mach1_i8v b;
#pragma unroll
        for (int s = 0; s < 4; ++s) {
            const int c = s*K4;
            const uint32_t win = (c <= 16 ? hi >> (16 - c) : (hi << (c - 16)) | (lo >> (48 - c))) & 0xFFFFu;
            const uint32_t h = HASHED ? (win*a.mult) >> 16 : (uint32_t) a.zr[win];
            b[2*s + 0] = (int) a.lev2h[h & 0xFFu];
            b[2*s + 1] = (int) a.lev2h[(h >> 8) & 0xFFu];
        }
        // the 16 u of the tile are the same for every lane: lane l holds u[l]. The tile scale goes
        // on the product, not on u, so the fp16 range of the split does not depend on it
        const mach1_f2v t = mach1_dpas2(mach1_f2v(0.0f), mach1_xmx1_split(u), b);
        return acc + t*sc;
    };

    mach1_f2v acc[U];
#pragma unroll
    for (int k = 0; k < U; ++k) {
        acc[k] = 0.0f;
    }
    const uint16_t * tw = a.tr + a.cb0*words;
    const float    * up = a.u + a.cb0*16 + row;
    // the wave scale of tile cb: gwe[g0 - cb], one less from cb = Nb - rb on
    const float    * gp = a.gwe + (Mb + Nb - 1 - a.rb - a.cb0);
    const int        cx = Nb - a.rb;
    int cb = a.cb0;
    for (; cb + U <= a.cbe; cb += U) {
        uint32_t w0[U], w1[U], w2[U], w3[U];
        float    u[U], sc[U];
#pragma unroll
        for (int k = 0; k < U; ++k) {
            w0[k] = tw[k*words + wi];
            w1[k] = tw[k*words + wi1];
            w2[k] = tw[k*words + wi2];
            w3[k] = tw[k*words + wi3];
            u[k]  = up[k*16];
            sc[k] = gp[-k - (cb + k >= cx ? 1 : 0)];
        }
#pragma unroll
        for (int k = 0; k < U; ++k) {
            acc[k] = tile(acc[k], w0[k], w1[k], w2[k], w3[k], u[k], sc[k]);
        }
        tw += U*words;
        up += U*16;
        gp -= U;
    }
    for (; cb < a.cbe; ++cb) {
        acc[0] = tile(acc[0], tw[wi], tw[wi1], tw[wi2], tw[wi3], up[0], gp[cb >= cx ? -1 : 0]);
        tw += words;
        up += 16;
        gp -= 1;
    }
#pragma unroll
    for (int k = 1; k < U; ++k) {
        acc[0] += acc[k];
    }
    return mach1_xmx1_join(acc[0]);
}

template <bool HASHED, int NR>
static void mach1_d4_walk_xmx1(dpct::queue_ptr stream, const uint16_t * trellis, const int32_t * offs, const float * gw,
                               const uint16_t * zt, const float * units, const int32_t * ids, const float * scr_u,
                               float * scr_p, const int n, const int Mb, const int Nb, const int n_used, const int nb0,
                               const int nb1, const int n_pairs, const mach1_d4_hash hash) {
    constexpr int SG  = MACH1_D4_XMX1_SG;
    constexpr int RBW = SG/NR;
    const int depth = (Nb + NR - 1)/NR;
    stream->submit([&](sycl::handler & cgh) {
        // lev2h[b] = (level of nibble b & 15, level of nibble b >> 4) as a half2
        sycl::local_accessor<uint32_t, 1> lev_acc(sycl::range<1>(256), cgh);
        sycl::local_accessor<float, 1>    part_acc(sycl::range<1>(SG*16), cgh);
        cgh.parallel_for(
            sycl::nd_range<2>(sycl::range<2>(n_pairs, (size_t) ((Mb + RBW - 1)/RBW)*SG*16), sycl::range<2>(1, SG*16)),
            [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
                uint32_t * lev2h = lev_acc.get_multi_ptr<sycl::access::decorated::no>().get();
                float    * part  = part_acc.get_multi_ptr<sycl::access::decorated::no>().get();
                const int tid = (int) it.get_local_id(1);
                const int row = tid & 15;
                const int sgi = tid >> 4;
                const int p   = (int) it.get_group(0);
                const int rb  = (int) it.get_group(1)*RBW + sgi/NR;
                const int rg  = sgi % NR;
                const int e   = mach1_d4_id(ids, p, n_used, nb0, nb1);
                const int m   = Mb*16;

                const int      K4    = offs[2*e + 1];
                const int      words = 4*K4;
                const int      r     = K4 - 4;
                const float    un    = units[r];
                const uint32_t mult  = HASHED ? (uint32_t) hash.v[3*r] : 0u;
                const uint16_t * zr  = zt + (int64_t) r*65536;
                const float    * gwe = gw + (int64_t) e*(Mb + Nb);

                for (int b = tid; b < 256; b += SG*16) {
                    sycl::half l[2];
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const int k = (b >> (4*h)) & 15;
                        if (HASHED) {
                            const uint32_t half = (uint32_t) hash.v[3*r + 1 + (k >> 3)];
                            l[h] = (sycl::half) (float)(((int32_t)(half << (28 - 4*(k & 7)))) >> 28);
                        } else {
                            l[h] = (sycl::half) (float)(k - 8);
                        }
                    }
                    lev2h[b] = (uint32_t) sycl::bit_cast<uint16_t>(l[0]) | ((uint32_t) sycl::bit_cast<uint16_t>(l[1]) << 16);
                }
                sycl::group_barrier(it.get_group());
                const bool on = rb < Mb;
                float v = 0.0f;
                if (on) {
                    const uint16_t * tr = trellis + (int64_t) offs[2*e] + (int64_t) rb*Nb*words;
                    const int cb0 = rg*depth;
                    const mach1_d4_xmx1_args a = { tr, lev2h, zr, gwe, scr_u + (int64_t) p*n, mult, row, rb, Mb, Nb,
                                                   cb0, sycl::min(cb0 + depth, Nb) };
                    // K4 is uniform per expert: as a constant, the window offsets of every state are too
                    switch (K4) {
                        case 4:  v = mach1_d4_xmx1_rb<HASHED, 4>(a); break;
                        case 5:  v = mach1_d4_xmx1_rb<HASHED, 5>(a); break;
                        case 6:  v = mach1_d4_xmx1_rb<HASHED, 6>(a); break;
                        case 7:  v = mach1_d4_xmx1_rb<HASHED, 7>(a); break;
                        default: v = mach1_d4_xmx1_rb<HASHED, 8>(a); break;
                    }
                }
                if constexpr (NR == 1) {
                    if (on) {
                        scr_p[(int64_t) p*m + rb*16 + row] = v*(un*(1.0f/MACH1_XMX1_US));
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
                        scr_p[(int64_t) p*m + rb*16 + row] = s2*(un*(1.0f/MACH1_XMX1_US));
                    }
                }
            });
    });
}

// the single-token walk for a few pairs: on the matrix engine when the device has one, else (or
// with GGML_SYCL_MACH1_D4_XMX1=0) the SIMT walk in CUDA's fp32 order
template <bool HASHED>
static void mach1_d4_walk_one(dpct::queue_ptr stream, const uint16_t * trellis, const int32_t * offs, const float * gw,
                              const uint16_t * zt, const float * units, const int32_t * ids, const float * scr_u,
                              float * scr_p, const int n, const int Mb, const int Nb, const int n_used,
                              const int n_expert, const int nb0, const int nb1, const int n_pairs,
                              const mach1_d4_hash hash) {
    static const bool xmx_env = getenv("GGML_SYCL_MACH1_D4_XMX1") == nullptr || atoi(getenv("GGML_SYCL_MACH1_D4_XMX1")) != 0;
    if (!xmx_env || !mach1_caps(stream).xmx) {
        mach1_d4_walk_any<false, HASHED>(stream, trellis, offs, gw, zt, units, ids, nullptr, scr_u, scr_p, n, Mb, Nb,
                                         n_used, n_expert, nb0, nb1, n_pairs, hash);
        return;
    }
#define MACH1_D4_XMX1_CASE(NR) \
    mach1_d4_walk_xmx1<HASHED, NR>(stream, trellis, offs, gw, zt, units, ids, scr_u, scr_p, n, Mb, Nb, n_used, nb0, nb1, \
                                   n_pairs, hash)
    switch (mach1_xmx1_ranges(stream, n_pairs*Mb)) {
        case 8:  MACH1_D4_XMX1_CASE(8); break;
        case 4:  MACH1_D4_XMX1_CASE(4); break;
        case 2:  MACH1_D4_XMX1_CASE(2); break;
        default: MACH1_D4_XMX1_CASE(1); break;
    }
#undef MACH1_D4_XMX1_CASE
}

// y = sv_e * H_m p for one pair per work-group
static void mach1_d4_redout(dpct::queue_ptr stream, const float * scr_p, const sycl::half * sv, const int32_t * ids,
                            float * dst, const mach1_d4_fwht_shape s, const int n_pairs, const int n_used,
                            const int nb0, const int nb1) {
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> sh_acc(sycl::range<1>(s.d), cgh);
        cgh.parallel_for(
            sycl::nd_range<2>(sycl::range<2>(n_pairs, MACH1_D4_WG), sycl::range<2>(1, MACH1_D4_WG)),
            [=](sycl::nd_item<2> it) {
                float * sh = sh_acc.get_multi_ptr<sycl::access::decorated::no>().get();
                const int p = (int) it.get_group(0);
                const int e = mach1_d4_id(ids, p, n_used, nb0, nb1);
                const int d = s.d;
                const float * v = scr_p + (int64_t) p*d;
                for (int i = (int) it.get_local_id(1); i < d; i += MACH1_D4_WG) {
                    sh[i] = v[i];
                }
                sycl::group_barrier(it.get_group());
                mach1_d4_fwht(sh, s, it);
                const sycl::half * se = sv + (int64_t) e*d;
                float * y = dst + (int64_t) p*d;
                for (int i = (int) it.get_local_id(1); i < d; i += MACH1_D4_WG) {
                    y[i] = sh[i]*(float) se[i];
                }
            });
    });
}

// The stages for d = R*M (R = 12 or 20, M = 16*E): sub-group a holds block a, lane l its elements
// l*E .. l*E + E - 1. The butterflies run in registers (spans below E) and across lanes (spans
// E .. M/2), in CUDA's span order, so the result is bit-identical to mach1_d4_fwht; only the
// radix factor goes through local memory.
//   OUT = false: dst = H (side_e .* src)   OUT = true: dst = side_e .* H src
// Into the split u (OUT = false, sp.ua set), the scale comes from ||u|| = ||side_e .* src|| (H is
// orthonormal), summed per sub-group before the barrier the radix needs anyway, so no reduction
// over u; it only bounds max |u| more loosely, and a power-of-two scale changes no hi/lo value
// unless one underflows fp16.
template <bool OUT, int R, int E>
static void mach1_d4_sg_stage(dpct::queue_ptr stream, const float * src, const sycl::half * side, const int32_t * ids,
                              float * dst, const mach1_d4_fwht_shape s, const int n_pairs, const int n_used,
                              const int xne1, const int nb0, const int nb1, const mach1_d4_split sp) {
    constexpr int M  = 16*E;
    constexpr int WG = 16*R;
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> sh_acc(sycl::range<1>(R*M), cgh);
        sycl::local_accessor<float, 1> n2_acc(sycl::range<1>(R), cgh);
        cgh.parallel_for(
            sycl::nd_range<2>(sycl::range<2>(n_pairs, WG), sycl::range<2>(1, WG)),
            [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
                float * sh = sh_acc.get_multi_ptr<sycl::access::decorated::no>().get();
                float * n2 = n2_acc.get_multi_ptr<sycl::access::decorated::no>().get();
                const bool split = !OUT && sp.ua != nullptr;
                const sycl::sub_group sg = it.get_sub_group();
                const int p    = (int) it.get_group(0);
                const int tid  = (int) it.get_local_id(1);
                const int lane = tid & 15;
                const int blk  = tid >> 4;
                const int e    = mach1_d4_id(ids, p, n_used, nb0, nb1);
                const int t    = p / n_used;
                const sycl::half * se = side + (int64_t) e*(R*M);
                const float      * xc = src + (int64_t)(OUT ? p : (xne1 == 1 ? t : p))*(R*M);

                // the work-item holds the elements k*16 + lane of its block (see mach1_sg_fwht)
                float v[E];
#pragma unroll
                for (int k = 0; k < E; ++k) {
                    const int i = blk*M + k*16 + lane;
                    v[k] = OUT ? xc[i] : (float) se[i]*xc[i];
                }
                if (split) {
                    float q = 0.0f;
#pragma unroll
                    for (int k = 0; k < E; ++k) {
                        q += v[k]*v[k];
                    }
                    q = sycl::reduce_over_group(sg, q, sycl::plus<float>());
                    if (lane == 0) {
                        n2[blk] = q;
                    }
                }
                mach1_sg_fwht<E>(sg, lane, v);
#pragma unroll
                for (int k = 0; k < E; ++k) {
                    sh[blk*M + k*16 + lane] = v[k]/s.scm;
                }
                sycl::group_barrier(it.get_group());

                // +-col is exact, so acc += (+-1)*col and the select below give the same sums
                const uint32_t * HC = R == 12 ? mach1_d4_h12c.data() : mach1_d4_h20c.data();
                if constexpr (!OUT && E % 2 == 0) {
                    if (split) {
                        float nn = 0.0f;
                        for (int c = 0; c < R; ++c) {
                            nn += n2[c];
                        }
                        const float  sc = mach1_d4_split_scale(sycl::sqrt(nn));
                        const int    k0 = sp.pos[p];
                        uint32_t   * uh = (uint32_t *) (sp.ua + (int64_t) k0*(R*M));
                        uint32_t   * ul = (uint32_t *) (sp.ua + sp.lo_off + (int64_t) k0*(R*M));
                        // two adjacent outputs per work-item, one 32-bit store each for hi and lo
#pragma unroll
                        for (int k = 0; k < E/2; ++k) {
                            const int i = 2*(tid + k*WG);
                            const int a = i / M;
                            const int b = i % M;
                            float acc0 = 0.0f, acc1 = 0.0f;
#pragma unroll
                            for (int c = 0; c < R; ++c) {
                                const sycl::float2 col = *(const sycl::float2 *) (sh + c*M + b);
                                const bool neg = ((HC[c] >> a) & 1u) != 0u;
                                acc0 = neg ? acc0 - col.x() : acc0 + col.x();
                                acc1 = neg ? acc1 - col.y() : acc1 + col.y();
                            }
                            sycl::half l0, l1;
                            const uint32_t h0 = mach1_d4_split_hl(acc0/s.rs*sc, l0);
                            const uint32_t h1 = mach1_d4_split_hl(acc1/s.rs*sc, l1);
                            uh[i/2] = h0 | (h1 << 16);
                            ul[i/2] = (uint32_t) sycl::bit_cast<uint16_t>(l0) | ((uint32_t) sycl::bit_cast<uint16_t>(l1) << 16);
                        }
                        if (tid == 0) {
                            sp.inv_s[k0] = 1.0f/sc;
                        }
                        return;
                    }
                }
#pragma unroll
                for (int k = 0; k < E; ++k) {
                    const int i = tid + k*WG;
                    const int a = i / M;
                    const int b = i % M;
                    float acc = 0.0f;
#pragma unroll
                    for (int c = 0; c < R; ++c) {
                        const float col = sh[c*M + b];
                        acc = ((HC[c] >> a) & 1u) != 0u ? acc - col : acc + col;
                    }
                    v[k] = acc/s.rs;
                }
                if (split) {
                    mach1_d4_split_row<E>(it, sp, [&](int k) { return v[k]; }, p, tid, WG, R*M);
                    return;
                }
                float * y = dst + (int64_t) p*(R*M);
#pragma unroll
                for (int k = 0; k < E; ++k) {
                    const int i = tid + k*WG;
                    y[i] = OUT ? v[k]*(float) se[i] : v[k];
                }
            });
    });
}

template <bool OUT, int R>
static bool mach1_d4_sg_stage_r(dpct::queue_ptr stream, const float * src, const sycl::half * side, const int32_t * ids,
                                float * dst, const mach1_d4_fwht_shape & s, int n_pairs, int n_used, int xne1, int nb0, int nb1,
                                const mach1_d4_split & sp) {
    switch (s.M) {
        case 16:  mach1_d4_sg_stage<OUT, R, 1> (stream, src, side, ids, dst, s, n_pairs, n_used, xne1, nb0, nb1, sp); return true;
        case 32:  mach1_d4_sg_stage<OUT, R, 2> (stream, src, side, ids, dst, s, n_pairs, n_used, xne1, nb0, nb1, sp); return true;
        case 64:  mach1_d4_sg_stage<OUT, R, 4> (stream, src, side, ids, dst, s, n_pairs, n_used, xne1, nb0, nb1, sp); return true;
        case 128: mach1_d4_sg_stage<OUT, R, 8> (stream, src, side, ids, dst, s, n_pairs, n_used, xne1, nb0, nb1, sp); return true;
        case 256: mach1_d4_sg_stage<OUT, R, 16>(stream, src, side, ids, dst, s, n_pairs, n_used, xne1, nb0, nb1, sp); return true;
        default:  return false;
    }
}

// the sub-group stage when the shape allows it (false: use the work-group one)
template <bool OUT>
static bool mach1_d4_sg_stage_try(dpct::queue_ptr stream, const float * src, const sycl::half * side, const int32_t * ids,
                                  float * dst, const mach1_d4_fwht_shape & s, int n_pairs, int n_used, int xne1, int nb0, int nb1,
                                  const mach1_d4_split & sp = {}) {
    if (s.R == 12) {
        return mach1_d4_sg_stage_r<OUT, 12>(stream, src, side, ids, dst, s, n_pairs, n_used, xne1, nb0, nb1, sp);
    }
    if (s.R == 20) {
        return mach1_d4_sg_stage_r<OUT, 20>(stream, src, side, ids, dst, s, n_pairs, n_used, xne1, nb0, nb1, sp);
    }
    return false;
}

//
// Decode pairs: the gate and up matrices of the same experts and input in one launch per pass
// (input stages, walk, output stages), and the output pass ends in the SWIGLU of the two. Each
// matrix's arithmetic is that of mach1_d4_sg_stage and the ungrouped walk.
//

// the transform of the row whose elements blk*M + k*16 + lane the work-item holds in v (side
// already applied): on return v[k] holds output tid + k*WG before the side of the output stage
template <int R, int E>
static inline void mach1_d4_sg_transform(const sycl::nd_item<2> & it, float * sh, float (&v)[E],
                                         const mach1_d4_fwht_shape & s) {
    constexpr int M  = 16*E;
    constexpr int WG = 16*R;
    const sycl::sub_group sg = it.get_sub_group();
    const int tid  = (int) it.get_local_id(1);
    const int lane = tid & 15;
    const int blk  = tid >> 4;
    mach1_sg_fwht<E>(sg, lane, v);
#pragma unroll
    for (int k = 0; k < E; ++k) {
        sh[blk*M + k*16 + lane] = v[k]/s.scm;
    }
    sycl::group_barrier(it.get_group());
    const uint32_t * HC = R == 12 ? mach1_d4_h12c.data() : mach1_d4_h20c.data();
#pragma unroll
    for (int k = 0; k < E; ++k) {
        const int i = tid + k*WG;
        const int a = i / M;
        const int b = i % M;
        float acc = 0.0f;
#pragma unroll
        for (int c = 0; c < R; ++c) {
            const float col = sh[c*M + b];
            acc = ((HC[c] >> a) & 1u) != 0u ? acc - col : acc + col;
        }
        v[k] = acc/s.rs;
    }
}

// u for both matrices: work-groups n_pairs .. 2*n_pairs - 1 take side2, rows of dst after n_pairs
template <int R, int E>
static void mach1_d4_pair_in(dpct::queue_ptr stream, const float * x, const sycl::half * side, const sycl::half * side2,
                             const int32_t * ids, float * dst, const mach1_d4_fwht_shape s, const int n_pairs,
                             const int n_used, const int xne1, const int nb0, const int nb1) {
    constexpr int M  = 16*E;
    constexpr int WG = 16*R;
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> sh_acc(sycl::range<1>(R*M), cgh);
        cgh.parallel_for(
            sycl::nd_range<2>(sycl::range<2>((size_t) 2*n_pairs, WG), sycl::range<2>(1, WG)),
            [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
                float * sh = sh_acc.get_multi_ptr<sycl::access::decorated::no>().get();
                const int pa   = (int) it.get_group(0);
                const bool sec = pa >= n_pairs;
                const int p    = pa - (sec ? n_pairs : 0);
                const int tid  = (int) it.get_local_id(1);
                const int lane = tid & 15;
                const int blk  = tid >> 4;
                const int e    = mach1_d4_id(ids, p, n_used, nb0, nb1);
                const int t    = p / n_used;
                const sycl::half * se = (sec ? side2 : side) + (int64_t) e*(R*M);
                const float      * xc = x + (int64_t)(xne1 == 1 ? t : p)*(R*M);
                float v[E];
#pragma unroll
                for (int k = 0; k < E; ++k) {
                    const int i = blk*M + k*16 + lane;
                    v[k] = (float) se[i]*xc[i];
                }
                mach1_d4_sg_transform<R, E>(it, sh, v, s);
                float * y = dst + (int64_t) pa*(R*M);
#pragma unroll
                for (int k = 0; k < E; ++k) {
                    y[tid + k*WG] = v[k];
                }
            });
    });
}

// y = silu(sv_g .* H p_g) * (sv_u .* H p_u), p_u the rows of src after n_pairs; SWIGLU as ggml's
template <int R, int E>
static void mach1_d4_pair_out_swiglu(dpct::queue_ptr stream, const float * src, const sycl::half * side,
                                     const sycl::half * side2, const int32_t * ids, float * dst,
                                     const mach1_d4_fwht_shape s, const int n_pairs, const int n_used, const int nb0,
                                     const int nb1) {
    constexpr int M  = 16*E;
    constexpr int WG = 16*R;
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> sh_acc(sycl::range<1>(R*M), cgh);
        cgh.parallel_for(
            sycl::nd_range<2>(sycl::range<2>((size_t) n_pairs, WG), sycl::range<2>(1, WG)),
            [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
                float * sh = sh_acc.get_multi_ptr<sycl::access::decorated::no>().get();
                const int p    = (int) it.get_group(0);
                const int tid  = (int) it.get_local_id(1);
                const int lane = tid & 15;
                const int blk  = tid >> 4;
                const int e    = mach1_d4_id(ids, p, n_used, nb0, nb1);
                const sycl::half * sg_ = side  + (int64_t) e*(R*M);
                const sycl::half * su_ = side2 + (int64_t) e*(R*M);
                const float * xg = src + (int64_t) p*(R*M);
                const float * xu = src + (int64_t) (n_pairs + p)*(R*M);
                float vg[E], vu[E];
#pragma unroll
                for (int k = 0; k < E; ++k) {
                    vg[k] = xg[blk*M + k*16 + lane];
                    vu[k] = xu[blk*M + k*16 + lane];
                }
                mach1_d4_sg_transform<R, E>(it, sh, vg, s);
                sycl::group_barrier(it.get_group());   // the radix reads of sh are done
                mach1_d4_sg_transform<R, E>(it, sh, vu, s);
                float * y = dst + (int64_t) p*(R*M);
#pragma unroll
                for (int k = 0; k < E; ++k) {
                    const int   i = tid + k*WG;
                    const float g = vg[k]*(float) sg_[i];
                    const float u = vu[k]*(float) su_[i];
                    y[i] = g/(1.0f + sycl::exp(-g))*u;
                }
            });
    });
}

template <int R>
static bool mach1_d4_pair_in_r(dpct::queue_ptr q, const float * x, const sycl::half * s1, const sycl::half * s2,
                               const int32_t * ids, float * dst, const mach1_d4_fwht_shape & s, int np, int nu,
                               int xne1, int nb0, int nb1) {
    switch (s.M) {
        case 16:  mach1_d4_pair_in<R, 1> (q, x, s1, s2, ids, dst, s, np, nu, xne1, nb0, nb1); return true;
        case 32:  mach1_d4_pair_in<R, 2> (q, x, s1, s2, ids, dst, s, np, nu, xne1, nb0, nb1); return true;
        case 64:  mach1_d4_pair_in<R, 4> (q, x, s1, s2, ids, dst, s, np, nu, xne1, nb0, nb1); return true;
        case 128: mach1_d4_pair_in<R, 8> (q, x, s1, s2, ids, dst, s, np, nu, xne1, nb0, nb1); return true;
        case 256: mach1_d4_pair_in<R, 16>(q, x, s1, s2, ids, dst, s, np, nu, xne1, nb0, nb1); return true;
        default:  return false;
    }
}

template <int R>
static bool mach1_d4_pair_out_r(dpct::queue_ptr q, const float * src, const sycl::half * s1, const sycl::half * s2,
                                const int32_t * ids, float * dst, const mach1_d4_fwht_shape & s, int np, int nu,
                                int nb0, int nb1) {
    switch (s.M) {
        case 16:  mach1_d4_pair_out_swiglu<R, 1> (q, src, s1, s2, ids, dst, s, np, nu, nb0, nb1); return true;
        case 32:  mach1_d4_pair_out_swiglu<R, 2> (q, src, s1, s2, ids, dst, s, np, nu, nb0, nb1); return true;
        case 64:  mach1_d4_pair_out_swiglu<R, 4> (q, src, s1, s2, ids, dst, s, np, nu, nb0, nb1); return true;
        case 128: mach1_d4_pair_out_swiglu<R, 8> (q, src, s1, s2, ids, dst, s, np, nu, nb0, nb1); return true;
        case 256: mach1_d4_pair_out_swiglu<R, 16>(q, src, s1, s2, ids, dst, s, np, nu, nb0, nb1); return true;
        default:  return false;
    }
}

static bool mach1_d4_sg_shape_ok(const mach1_d4_fwht_shape & s) {
    return (s.R == 12 || s.R == 20) && (s.M == 16 || s.M == 32 || s.M == 64 || s.M == 128 || s.M == 256);
}

bool ggml_sycl_mach1_d4_pair_swiglu(ggml_backend_sycl_context & ctx, const ggml_tensor * g, const ggml_tensor * u,
                                    ggml_tensor * out) {
    static const bool on = getenv("GGML_SYCL_MACH1_D4_PAIR") == nullptr || atoi(getenv("GGML_SYCL_MACH1_D4_PAIR")) != 0;
    const ggml_tensor * ids = g->src[7];
    const ggml_tensor * x   = g->src[8];
    const int n      = (int) g->src[2]->ne[0];
    const int m      = (int) g->src[3]->ne[0];
    const int n_tok  = (int) ids->ne[1];
    const bool hg = ggml_get_op_params_i32(g, 15) == 1;
    const bool hu = ggml_get_op_params_i32(u, 15) == 1;
    const mach1_d4_fwht_shape sn = mach1_d4_shape(n);
    const mach1_d4_fwht_shape sm = mach1_d4_shape(m);
    if (!on || u->src[7] != ids || u->src[8] != x || n_tok >= MACH1_D4_GROUP_MIN || hg != hu ||
        (int) u->src[2]->ne[0] != n || (int) u->src[3]->ne[0] != m || u->src[1]->ne[1] != g->src[1]->ne[1] ||
        !mach1_d4_sg_shape_ok(sn) || !mach1_d4_sg_shape_ok(sm) || out->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(out) || ggml_nelements(out) != ggml_nelements(g)) {
        return false;
    }
    const int n_used  = (int) ids->ne[0];
    const int n_pairs = n_used*n_tok;
    const int xne1    = (int) x->ne[1];
    const int nb0     = (int)(ids->nb[0]/sizeof(int32_t));
    const int nb1     = (int)(ids->nb[1]/sizeof(int32_t));
    const int32_t * idsd = (const int32_t *) ids->data;
    dpct::queue_ptr stream = ctx.stream();

    mach1_d4_hash hash_g, hash_u;
    memset(&hash_g, 0, sizeof(hash_g));
    memset(&hash_u, 0, sizeof(hash_u));
    if (hg) {
        for (int i = 0; i < 15; ++i) {
            hash_g.v[i] = ggml_get_op_params_i32(g, i);
            hash_u.v[i] = ggml_get_op_params_i32(u, i);
        }
    }

    // declared in allocation order (the VMM pool frees in stack order)
    ggml_sycl_pool_alloc<float> p_buf(ctx.pool(), (size_t) 2*n_pairs*m);
    ggml_sycl_pool_alloc<float> u_buf(ctx.pool(), (size_t) 2*n_pairs*n);

    const sycl::half * su_g = (const sycl::half *) g->src[2]->data;
    const sycl::half * su_u = (const sycl::half *) u->src[2]->data;
    const sycl::half * sv_g = (const sycl::half *) g->src[3]->data;
    const sycl::half * sv_u = (const sycl::half *) u->src[3]->data;

    (sn.R == 12 ? mach1_d4_pair_in_r<12> : mach1_d4_pair_in_r<20>)(
        stream, (const float *) x->data, su_g, su_u, idsd, u_buf.get(), sn, n_pairs, n_used, xne1, nb0, nb1);

    // the two walks as separate launches (one paired launch ran slower than both together)
    for (int h = 0; h < 2; ++h) {
        const ggml_tensor * w = h == 0 ? g : u;
        (hg ? mach1_d4_walk_one<true> : mach1_d4_walk_one<false>)(
            stream, (const uint16_t *) w->src[0]->data, (const int32_t *) w->src[1]->data, (const float *) w->src[4]->data,
            (const uint16_t *) w->src[5]->data, (const float *) w->src[6]->data, idsd,
            u_buf.get() + (size_t) h*n_pairs*n, p_buf.get() + (size_t) h*n_pairs*m,
            n, m/16, n/16, n_used, (int) w->src[1]->ne[1], nb0, nb1, n_pairs, h == 0 ? hash_g : hash_u);
    }

    (sm.R == 12 ? mach1_d4_pair_out_r<12> : mach1_d4_pair_out_r<20>)(
        stream, p_buf.get(), sv_g, sv_u, idsd, (float *) out->data, sm, n_pairs, n_used, nb0, nb1);
    return true;
}

bool ggml_sycl_mach1_d4_supported(const ggml_tensor * op) {
    const int64_t n = op->src[2]->ne[0];
    const int64_t m = op->src[3]->ne[0];
    const int64_t E = op->src[1]->ne[1];
    auto dim_ok = [](int64_t d) {
        if (d <= 0 || d % 16 != 0) {
            return false;
        }
        while (d % 2 == 0) {
            d /= 2;
        }
        return d == 1 || d == 3 || d == 5;
    };
    return dim_ok(n) && dim_ok(m) && n <= 8192 && m <= 8192 && E <= 8192 &&
           ggml_is_contiguous(op->src[8]) && ggml_is_contiguous(op);
}

void ggml_sycl_mach1_d4_mm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/9);

    const ggml_tensor * trellis = dst->src[0];
    const ggml_tensor * offs    = dst->src[1];
    const ggml_tensor * su      = dst->src[2];
    const ggml_tensor * sv      = dst->src[3];
    const ggml_tensor * gw      = dst->src[4];
    const ggml_tensor * zt      = dst->src[5];
    const ggml_tensor * units   = dst->src[6];
    const ggml_tensor * ids     = dst->src[7];
    const ggml_tensor * x       = dst->src[8];

    const int n        = (int) su->ne[0];
    const int m        = (int) sv->ne[0];
    const int n_expert = (int) offs->ne[1];
    const int n_used   = (int) ids->ne[0];
    const int n_tok    = (int) ids->ne[1];
    const int n_pairs  = n_used*n_tok;
    const int xne1     = (int) x->ne[1];
    const int nb0      = (int)(ids->nb[0]/sizeof(int32_t));
    const int nb1      = (int)(ids->nb[1]/sizeof(int32_t));

    dpct::queue_ptr stream = ctx.stream();

    // declared in allocation order (the VMM pool frees in stack order)
    ggml_sycl_pool_alloc<float>      p_buf  (ctx.pool(), (size_t) n_pairs*m);
    ggml_sycl_pool_alloc<float>      u_buf  (ctx.pool());
    ggml_sycl_pool_alloc<int32_t>    map_buf(ctx.pool());
    ggml_sycl_pool_alloc<int32_t>    pos_buf(ctx.pool());
    ggml_sycl_pool_alloc<sycl::half> ua_buf (ctx.pool());
    ggml_sycl_pool_alloc<float>      is_buf (ctx.pool());
    ggml_sycl_pool_alloc<int32_t>    work_buf(ctx.pool());

    mach1_d4_hash hash;
    memset(&hash, 0, sizeof(hash));
    if (ggml_get_op_params_i32(dst, 15) == 1) {
        for (int i = 0; i < 15; ++i) {
            hash.v[i] = ggml_get_op_params_i32(dst, i);
        }
    }

    const bool grouped = n_tok >= MACH1_D4_GROUP_MIN;
    const int32_t * idsd = (const int32_t *) ids->data;

    static const bool xmx_env = getenv("GGML_SYCL_MACH1_D4_XMX") == nullptr || atoi(getenv("GGML_SYCL_MACH1_D4_XMX")) != 0;
    const bool xmx = grouped && n_tok >= MACH1_D4_XMX_MIN && n >= 32 && xmx_env &&
                     stream->get_device().has(sycl::aspect::ext_intel_matrix);

    // the XMX walk takes u split and expert-sorted, written by the input stage, and a work list of
    // at most one unit per expert plus one per MACH1_D4_XMX_UNIT pairs
    mach1_d4_split sp = {};
    const int n_work = n_expert + (n_pairs + MACH1_D4_XMX_UNIT - 1)/MACH1_D4_XMX_UNIT;
    if (xmx) {
        // the lo plane starts on a 64-byte boundary
        const int64_t lo_off = (int64_t)((n_pairs + 31) & ~31)*n;
        map_buf.alloc((size_t) n_expert + 1 + n_pairs);
        pos_buf.alloc((size_t) n_pairs);
        ua_buf.alloc((size_t) (lo_off + (int64_t) n_pairs*n));
        is_buf.alloc((size_t) n_pairs);
        work_buf.alloc((size_t) 2*n_work);
        mach1_d4_map(stream, idsd, map_buf.get(), pos_buf.get(), n_expert, n_used, n_pairs, nb0, nb1, work_buf.get(),
                     MACH1_D4_XMX_UNIT, n_work);
        sp = { ua_buf.get(), is_buf.get(), pos_buf.get(), lo_off };
    } else {
        u_buf.alloc((size_t) n_pairs*n);
    }
    if (!mach1_d4_sg_stage_try<false>(stream, (const float *) x->data, (const sycl::half *) su->data, idsd, u_buf.get(),
                                      mach1_d4_shape(n), n_pairs, n_used, xne1, nb0, nb1, sp)) {
        mach1_d4_ustage(stream, (const float *) x->data, (const sycl::half *) su->data, idsd, u_buf.get(),
                        mach1_d4_shape(n), n_pairs, n_used, xne1, nb0, nb1, sp);
    }
    if (xmx) {
        (hash.v[0] != 0 ? mach1_d4_walk_xmx<true> : mach1_d4_walk_xmx<false>)(
            stream, (const uint16_t *) trellis->data, (const int32_t *) offs->data, (const float *) gw->data,
            (const uint16_t *) zt->data, (const float *) units->data, map_buf.get(), work_buf.get(), n_work, sp,
            p_buf.get(), n, m/16, n/16, n_expert, n_pairs, hash);
    } else if (grouped) {
        map_buf.alloc((size_t) n_expert + 1 + n_pairs);
        mach1_d4_map(stream, idsd, map_buf.get(), nullptr, n_expert, n_used, n_pairs, nb0, nb1);
        (hash.v[0] != 0 ? mach1_d4_walk_any<true, true> : mach1_d4_walk_any<true, false>)(
                            stream, (const uint16_t *) trellis->data, (const int32_t *) offs->data,
                            (const float *) gw->data, (const uint16_t *) zt->data, (const float *) units->data,
                            idsd, map_buf.get(), u_buf.get(), p_buf.get(), n, m/16, n/16, n_used, n_expert,
                            nb0, nb1, n_expert, hash);
    } else {
        (hash.v[0] != 0 ? mach1_d4_walk_one<true> : mach1_d4_walk_one<false>)(
                             stream, (const uint16_t *) trellis->data, (const int32_t *) offs->data,
                             (const float *) gw->data, (const uint16_t *) zt->data, (const float *) units->data,
                             idsd, u_buf.get(), p_buf.get(), n, m/16, n/16, n_used, n_expert,
                             nb0, nb1, n_pairs, hash);
    }
    if (!mach1_d4_sg_stage_try<true>(stream, p_buf.get(), (const sycl::half *) sv->data, idsd, (float *) dst->data,
                                     mach1_d4_shape(m), n_pairs, n_used, 1, nb0, nb1)) {
        mach1_d4_redout(stream, p_buf.get(), (const sycl::half *) sv->data, idsd, (float *) dst->data,
                        mach1_d4_shape(m), n_pairs, n_used, nb0, nb1);
    }
}
