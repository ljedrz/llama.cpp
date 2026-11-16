#pragma once

#include "common.hpp"

// DPAS on plain registers (IGC's builtin behind cl_intel_subgroup_matrix_multiply_accumulate;
// sub-group 16 only), for kernels whose lanes already hold their operand slices:
//   a[r] = A[r][lane] (8 x 16 fp16), b[j] = (B[2j][lane], B[2j + 1][lane]) (16 x 16 fp16, packed
//   in pairs), acc[r] = C[r][lane] (8 x 16 fp32); returns acc + A B.
// mach1_load_a: the A operand for 8 rows from y and 16 fp16 columns from x of a row-major
// matrix of h rows, w bytes per row (>= 64) and a base aligned to 64 bytes, by one 2D block
// read; rows past h read as zero.
// The __builtin_IB_ names matter: ggml-sycl links with device image dependencies, and the
// runtime would look for any other undefined symbol in another image.
typedef short mach1_s8v __attribute__((ext_vector_type(8)));
typedef short mach1_s16v __attribute__((ext_vector_type(16)));
typedef int   mach1_i8v __attribute__((ext_vector_type(8)));
typedef float mach1_f8v __attribute__((ext_vector_type(8)));
typedef int   mach1_i2v __attribute__((ext_vector_type(2)));
typedef short mach1_s2v __attribute__((ext_vector_type(2)));
typedef float mach1_f2v __attribute__((ext_vector_type(2)));

// One token on the matrix engine (the decode walks): u*MACH1_XMX1_US = hi + lo/MACH1_XMX1_LO in
// fp16, the two as the rows of a 2-row DPAS. The fixed scale keeps |u| up to 2^22 in fp16 range;
// an element is kept to 22 bits, or to 2^-30 absolute below 2^-7. (Mach-2-Additive-Medium at
// decode: max |u| 6.5, rms 0.5 - 0.6, 2 - 4% of the elements below 2^-7.)
#define MACH1_XMX1_US 0.015625f
#define MACH1_XMX1_LO 2048.0f

#ifdef __SYCL_DEVICE_ONLY__
extern "C" SYCL_EXTERNAL mach1_f8v __builtin_IB_sub_group16_fdpas_f_f_hf_hf_8_8(mach1_f8v acc, mach1_s8v a, mach1_i8v b);
extern "C" SYCL_EXTERNAL mach1_f2v __builtin_IB_sub_group16_fdpas_f_f_hf_hf_8_2(mach1_f2v acc, mach1_s2v a, mach1_i8v b);
// the 2-row form: a[0] = hi, a[1] = lo of this lane's u
static inline mach1_f2v mach1_dpas2(mach1_f2v acc, mach1_s2v a, mach1_i8v b) {
    return __builtin_IB_sub_group16_fdpas_f_f_hf_hf_8_2(acc, a, b);
}
extern "C" SYCL_EXTERNAL mach1_s8v __builtin_IB_subgroup_block_read_flat_u16_m8k16v1(long base, int w_m1, int h_m1, int pitch_m1,
                                                                                      mach1_i2v coord);
static inline mach1_f8v mach1_dpas(mach1_f8v acc, mach1_s8v a, mach1_i8v b) {
    return __builtin_IB_sub_group16_fdpas_f_f_hf_hf_8_8(acc, a, b);
}
// two blocks side by side in one message: [0..7] the rows of columns x .. x + 15, [8..15] of
// x + 16 .. x + 31
extern "C" SYCL_EXTERNAL mach1_s16v __builtin_IB_subgroup_block_read_flat_u16_m8k16v2(long base, int w_m1, int h_m1, int pitch_m1,
                                                                                       mach1_i2v coord);
static inline mach1_s16v mach1_load_a2(const void * base, int w, int h, int x, int y) {
    return __builtin_IB_subgroup_block_read_flat_u16_m8k16v2((long) base, w - 1, h - 1, w - 1, mach1_i2v{x, y});
}
static inline mach1_s8v mach1_load_a(const void * base, int w, int h, int x, int y) {
    return __builtin_IB_subgroup_block_read_flat_u16_m8k16v1((long) base, w - 1, h - 1, w - 1, mach1_i2v{x, y});
}
#else
static inline mach1_f8v mach1_dpas(mach1_f8v acc, mach1_s8v, mach1_i8v) {
    return acc;
}
static inline mach1_f2v mach1_dpas2(mach1_f2v acc, mach1_s2v, mach1_i8v) {
    return acc;
}
static inline mach1_s8v mach1_load_a(const void *, int, int, int, int) {
    return mach1_s8v{};
}
static inline mach1_s16v mach1_load_a2(const void *, int, int, int, int) {
    return mach1_s16v{};
}
#endif

// the hi/lo rows of one u for mach1_dpas2
static inline mach1_s2v mach1_xmx1_split(float u) {
    const float      x  = u*MACH1_XMX1_US;
    const sycl::half hi = (sycl::half) x;
    const sycl::half lo = (sycl::half) ((x - (float) hi)*MACH1_XMX1_LO);
    return mach1_s2v{ sycl::bit_cast<short>(hi), sycl::bit_cast<short>(lo) };
}
// the value of an accumulated pair of rows
static inline float mach1_xmx1_join(mach1_f2v acc) {
    return acc[0] + acc[1]*(1.0f/MACH1_XMX1_LO);
}

// The 2^k stage of a block of M = 16*E elements held by one sub-group in registers: work-item
// `lane` has its elements k*16 + lane in v[k], so that a load or store of v[k] is contiguous over
// the sub-group (with the elements lane*E + k every one of them lands on 16 cache lines). The
// butterflies run in the order of the element index, as the scalar transforms take them: spans
// 1 .. 8 across the lanes, then 16 .. M/2 within the work-item.
template <int E>
static inline void mach1_sg_fwht(const sycl::sub_group & sg, const int lane, float (&v)[E]) {
#pragma unroll
    for (int lm = 1; lm < 16; lm <<= 1) {
        const bool upper = (lane & lm) != 0;
#pragma unroll
        for (int k = 0; k < E; ++k) {
            const float o = sycl::permute_group_by_xor(sg, v[k], lm);
            v[k] = upper ? o - v[k] : v[k] + o;
        }
    }
#pragma unroll
    for (int span = 1; span < E; span <<= 1) {
#pragma unroll
        for (int k = 0; k < E; ++k) {
            if ((k & span) == 0) {
                const float a0 = v[k];
                const float a1 = v[k + span];
                v[k]        = a0 + a1;
                v[k + span] = a0 - a1;
            }
        }
    }
}

// what the single-token walks ask of the device, looked up once per queue
struct mach1_dev_caps {
    bool xmx;          // matrix engine (DPAS)
    int  hw_threads;   // EUs x threads per EU
};
static inline const mach1_dev_caps & mach1_caps(dpct::queue_ptr stream) {
    static thread_local dpct::queue_ptr last = nullptr;
    static thread_local mach1_dev_caps  caps = {};
    if (stream != last) {
        const sycl::device d = stream->get_device();
        caps.xmx        = d.has(sycl::aspect::ext_intel_matrix);
        caps.hw_threads = 2048;
        if (d.has(sycl::aspect::ext_intel_gpu_eu_count) && d.has(sycl::aspect::ext_intel_gpu_hw_threads_per_eu)) {
            caps.hw_threads = (int) (d.get_info<sycl::ext::intel::info::device::gpu_eu_count>()*
                                     d.get_info<sycl::ext::intel::info::device::gpu_hw_threads_per_eu>());
        }
        last = stream;
    }
    return caps;
}
// The single-token walks are bound by the latency of a sub-group's chain of tiles, so they want
// many short chains, but a launch with more sub-groups than hardware threads runs in waves: the
// column ranges per row block (8, 4, 2 or 1) that keep `blocks` row blocks in one wave.
static inline int mach1_xmx1_ranges(dpct::queue_ptr stream, int blocks) {
    const int t = mach1_caps(stream).hw_threads;
    return 8*blocks <= t ? 8 : 4*blocks <= t ? 4 : 2*blocks <= t ? 2 : 1;
}

// Mach-1 codec ops (ports of ggml-cuda/mach1*.cu, plain kernels without fusion)
void ggml_sycl_mach1_d4_mm(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
bool ggml_sycl_mach1_d4_supported(const ggml_tensor * op);
// gate and up of the same experts and input, then SWIGLU(gate, up), at decode; false: not applicable
bool ggml_sycl_mach1_d4_pair_swiglu(ggml_backend_sycl_context & ctx, const ggml_tensor * g, const ggml_tensor * u,
                                    ggml_tensor * out);
void ggml_sycl_mach1_rt_mm(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
bool ggml_sycl_mach1_rt_supported(const ggml_tensor * op, size_t local_mem);
// two RT products of the same input, then SWIGLU of the two, at decode; false: not applicable
bool ggml_sycl_mach1_rt_pair_swiglu(ggml_backend_sycl_context & ctx, const ggml_tensor * g, const ggml_tensor * u,
                                    ggml_tensor * out);
void ggml_sycl_mach1_head_mm(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
void ggml_sycl_mach1_embed_gather(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

// fusions of the Mach-1 payload graph (expert pairs and weighted sum, hc activation tails); returns the
// number of nodes after i that the fused kernel also computed
int ggml_sycl_mach1_fuse(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int i);
