#include "mmvf.hpp"

static constexpr int MMVF_MAX_COLS = 8;
static constexpr int MMVF_NSG      = 8; // sub-groups (dst rows) per work-group

// the columns of a work unit and its batch (b2, i13): the src1 columns, or with GQA the src1 channels that share
// one src0 channel (then b2 is that src0 channel)
struct mmvf_args {
    int64_t ne00, ne01;
    int64_t ncols;
    int64_t nb2d, ne13;    // extents of b2 and i13
    int64_t x_div, r3;     // src0 channel = b2 / x_div, i13 / r3
    size_t  nb01, nb02, nb03;
    size_t  sy_col, sy_b2, nb13;
    size_t  sd_col, sd_b2, nb3;
};

template <typename T> static __dpct_inline__ float mmvf_load(const T * x, int64_t k) {
    return (float) x[k];
}

// VEC: 4 values per load, needs ne00 % 4 == 0 and aligned rows
// ROW_WG: the whole work-group takes one row, for few rows over long ones; else a sub-group per row
template <typename T, int NCOLS, bool VEC, bool ROW_WG>
static void mul_mat_vec_f_kernel(const char * __restrict__ src0, const char * __restrict__ src1, char * __restrict__ dst,
                                 const mmvf_args a, const sycl::nd_item<3> & item) {
    const auto    sg     = item.get_sub_group();
    const int     lane   = ROW_WG ? (int) item.get_local_linear_id() : (int) sg.get_local_linear_id();
    constexpr int stride = ROW_WG ? MMVF_NSG * WARP_SIZE : WARP_SIZE;
    const int64_t row    = ROW_WG ? (int64_t) item.get_group(1) : (int64_t) item.get_group(1) * MMVF_NSG + sg.get_group_linear_id();
    if (!ROW_WG && row >= a.ne01) {
        return;
    }
    const int64_t b2  = item.get_group(0) % a.nb2d;
    const int64_t i13 = item.get_group(0) / a.nb2d;

    const T * x = (const T *) (src0 + (b2 / a.x_div) * a.nb02 + (i13 / a.r3) * a.nb03 + row * a.nb01);
    const float * y[NCOLS];
#pragma unroll
    for (int j = 0; j < NCOLS; ++j) {
        y[j] = (const float *) (src1 + j * a.sy_col + b2 * a.sy_b2 + i13 * a.nb13);
    }

    float acc[NCOLS] = {};
    if constexpr (VEC) {
        for (int64_t k = 4 * lane; k < a.ne00; k += 4 * stride) {
            float w[4];
            if constexpr (std::is_same_v<T, float>) {
                const sycl::float4 v = *(const sycl::float4 *) (x + k);
                w[0] = v.x(); w[1] = v.y(); w[2] = v.z(); w[3] = v.w();
            } else {
                const sycl::vec<uint16_t, 4> v = *(const sycl::vec<uint16_t, 4> *) (x + k);
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    w[i] = (float) sycl::bit_cast<T>((uint16_t) v[i]);
                }
            }
#pragma unroll
            for (int j = 0; j < NCOLS; ++j) {
                const sycl::float4 u = *(const sycl::float4 *) (y[j] + k);
                acc[j] += w[0] * u.x() + w[1] * u.y() + w[2] * u.z() + w[3] * u.w();
            }
        }
    } else {
        for (int64_t k = lane; k < a.ne00; k += stride) {
            const float w = mmvf_load(x, k);
#pragma unroll
            for (int j = 0; j < NCOLS; ++j) {
                acc[j] += w * y[j][k];
            }
        }
    }

#pragma unroll
    for (int j = 0; j < NCOLS; ++j) {
        float sum;
        if constexpr (ROW_WG) {
            sum = sycl::reduce_over_group(item.get_group(), acc[j], sycl::plus<float>());
        } else {
            sum = sycl::reduce_over_group(sg, acc[j], sycl::plus<float>());
        }
        if (lane == 0) {
            *(float *) (dst + row * sizeof(float) + j * a.sd_col + b2 * a.sd_b2 + i13 * a.nb3) = sum;
        }
    }
}

template <typename T, int NCOLS, bool ROW_WG>
static void launch_mul_mat_vec_f(const char * src0, const char * src1, char * dst, const mmvf_args & a, bool vec,
                                 dpct::queue_ptr stream) {
    const sycl::range<3> block_nums(a.nb2d * a.ne13, ROW_WG ? a.ne01 : (a.ne01 + MMVF_NSG - 1) / MMVF_NSG, 1);
    const sycl::range<3> block_dims(1, 1, MMVF_NSG * WARP_SIZE);
    if (vec) {
        stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                             [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                 mul_mat_vec_f_kernel<T, NCOLS, true, ROW_WG>(src0, src1, dst, a, item);
                             });
    } else {
        stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                             [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                 mul_mat_vec_f_kernel<T, NCOLS, false, ROW_WG>(src0, src1, dst, a, item);
                             });
    }
}

template <typename T, int NCOLS>
static void launch_mul_mat_vec_f(const char * src0, const char * src1, char * dst, const mmvf_args & a, bool vec,
                                 dpct::queue_ptr stream) {
    // a sub-group per row leaves most of the device idle when there are few rows
    if (a.ne01 * a.nb2d * a.ne13 < 2048 && a.ne00 >= 512) {
        launch_mul_mat_vec_f<T, NCOLS, true>(src0, src1, dst, a, vec, stream);
    } else {
        launch_mul_mat_vec_f<T, NCOLS, false>(src0, src1, dst, a, vec, stream);
    }
}

template <typename T>
static void mul_mat_vec_f_switch(const char * src0, const char * src1, char * dst, const mmvf_args & a, bool vec,
                                 dpct::queue_ptr stream) {
    switch (a.ncols) {
        case 1: launch_mul_mat_vec_f<T, 1>(src0, src1, dst, a, vec, stream); break;
        case 2: launch_mul_mat_vec_f<T, 2>(src0, src1, dst, a, vec, stream); break;
        case 3: launch_mul_mat_vec_f<T, 3>(src0, src1, dst, a, vec, stream); break;
        case 4: launch_mul_mat_vec_f<T, 4>(src0, src1, dst, a, vec, stream); break;
        case 5: launch_mul_mat_vec_f<T, 5>(src0, src1, dst, a, vec, stream); break;
        case 6: launch_mul_mat_vec_f<T, 6>(src0, src1, dst, a, vec, stream); break;
        case 7: launch_mul_mat_vec_f<T, 7>(src0, src1, dst, a, vec, stream); break;
        case 8: launch_mul_mat_vec_f<T, 8>(src0, src1, dst, a, vec, stream); break;
        default: GGML_ABORT("unsupported column count %d", (int) a.ncols);
    }
}

bool ggml_sycl_mul_mat_vec_f(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                             ggml_tensor * dst) {
    if (!g_ggml_sycl_mmvf || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    const size_t ts = ggml_type_size(src0->type);
    if (src0->nb[0] != ts || src1->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float) ||
        src1->ne[2] % src0->ne[2] != 0 || src1->ne[3] % src0->ne[3] != 0) {
        return false;
    }
    const int64_t r2 = src1->ne[2] / src0->ne[2];
    const int64_t r3 = src1->ne[3] / src0->ne[3];

    // one token of grouped-query attention: the src1 channels that share a src0 channel are the columns, so a
    // row of the KV cache is read once for all of them
    const bool gqa = src1->ne[1] == 1 && r2 > 1 && r2 <= MMVF_MAX_COLS;

    // F16 only for the KV cache views; contiguous F16 weights keep their path
    if (src0->type == GGML_TYPE_F16) {
        if (ggml_is_contiguous(src0) && !gqa) {
            return false;
        }
    } else if (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_BF16) {
        return false;
    }
    if (!gqa) {
        if (src1->ne[1] < 1 || src1->ne[1] > MMVF_MAX_COLS) {
            return false;
        }
        // over several columns the library GEMM reads a large matrix fewer times
        if (src1->ne[1] > 1 && ggml_nbytes(src0) > 4*1024*1024) {
            return false;
        }
    }

    const mmvf_args a = gqa ?
        mmvf_args {
            src0->ne[0], src0->ne[1], r2, src0->ne[2], src1->ne[3], 1, r3,
            src0->nb[1], src0->nb[2], src0->nb[3],
            src1->nb[2], r2 * src1->nb[2], src1->nb[3],
            dst->nb[2],  r2 * dst->nb[2],  dst->nb[3],
        } :
        mmvf_args {
            src0->ne[0], src0->ne[1], src1->ne[1], src1->ne[2], src1->ne[3], r2, r3,
            src0->nb[1], src0->nb[2], src0->nb[3],
            src1->nb[1], src1->nb[2], src1->nb[3],
            dst->nb[1],  dst->nb[2],  dst->nb[3],
        };

    // the vector loads read 4 values of a row at once
    const size_t vb  = 4 * ts;
    const bool   vec = a.ne00 % 4 == 0 && (uintptr_t) src0->data % vb == 0 && (uintptr_t) src1->data % 16 == 0 &&
                       a.nb01 % vb == 0 && a.nb02 % vb == 0 && a.nb03 % vb == 0 &&
                       a.sy_col % 16 == 0 && a.sy_b2 % 16 == 0 && a.nb13 % 16 == 0;

    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    const char * x = (const char *) src0->data;
    const char * y = (const char *) src1->data;
    char *       d = (char *) dst->data;
    switch (src0->type) {
        case GGML_TYPE_F32:  mul_mat_vec_f_switch<float>(x, y, d, a, vec, stream); break;
        case GGML_TYPE_F16:  mul_mat_vec_f_switch<sycl::half>(x, y, d, a, vec, stream); break;
        default:             mul_mat_vec_f_switch<sycl::ext::oneapi::bfloat16>(x, y, d, a, vec, stream); break;
    }
    return true;
}
