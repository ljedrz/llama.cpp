//
// MIT license
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include "pool.hpp"
#include <float.h>

template <typename Ti, typename To>
static void pool2d_nchw_kernel(
        const int ih, const int iw, const int oh, const int ow,
        const int kh, const int kw, const int sh, const int sw,
        const int ph, const int pw, const int parallel_elements,
        const Ti* src, To* dst, const enum ggml_op_pool op,
        const sycl::nd_item<3> &item_ct1) {
        int idx = item_ct1.get_local_id(2) +
                  item_ct1.get_group(2) * item_ct1.get_local_range(2);
        if (idx >= parallel_elements) {
            return;
        }

        const int I_HW = ih * iw;
        const int O_HW = oh * ow;
        const int nc = idx / O_HW;
        const int cur_oh = idx % O_HW / ow;
        const int cur_ow = idx % O_HW % ow;
        const Ti* i_ptr = src + nc * I_HW;
        To* o_ptr = dst + nc * O_HW;
        const int start_h = cur_oh * sh - ph;
        const int bh = sycl::max(0, start_h);
        const int eh = sycl::min(ih, start_h + kh);
        const int start_w = cur_ow * sw - pw;
        const int bw = sycl::max(0, start_w);
        const int ew = sycl::min(iw, start_w + kw);

        To res = 0;

        switch (op) {
            case GGML_OP_POOL_AVG: res = 0; break;
            case GGML_OP_POOL_MAX: res = -FLT_MAX; break;
            default:
                res      = (To) sycl::nan(uint32_t(0));
                break;
        }

        for (int i = bh; i < eh; i += 1) {
            for (int j = bw; j < ew; j += 1) {
                Ti cur = i_ptr[i * iw + j];
                switch (op) {
                    case GGML_OP_POOL_AVG: res += (cur / (kh * kw)); break;
                    case GGML_OP_POOL_MAX: res = sycl::max(res, (To)cur); break;
                    default:
                        res = (To) sycl::nan(uint32_t(0));
                        break;
                }
            }
        }
        o_ptr[cur_oh * ow + cur_ow] = res;
}

template <typename Ti, typename To>
static void pool1d_ncw_kernel(
        const int iw, const int ow,
        const int k, const int s,
        const int p, const int parallel_elements,
        const Ti * src, To * dst, const enum ggml_op_pool op,
        const sycl::nd_item<3> & item_ct1) {
    int idx = item_ct1.get_local_id(2) +
              item_ct1.get_group(2) * item_ct1.get_local_range(2);
    if (idx >= parallel_elements) {
        return;
    }

    const int nc     = idx / ow;
    const int cur_ow = idx % ow;
    const Ti * i_ptr = src + nc * iw;
    To *       o_ptr = dst + nc * ow;
    const int  start = cur_ow * s - p;
    const int  b     = sycl::max(0, start);
    const int  e     = sycl::min(iw, start + k);

    To res = 0;
    switch (op) {
        case GGML_OP_POOL_AVG: res = 0;        break;
        case GGML_OP_POOL_MAX: res = -FLT_MAX; break;
        default:
            res = (To) sycl::nan(uint32_t(0));
            break;
    }

    for (int j = b; j < e; j += 1) {
        Ti cur = i_ptr[j];
        switch (op) {
            case GGML_OP_POOL_AVG: res += cur; break;
            case GGML_OP_POOL_MAX: res = sycl::max(res, (To) cur); break;
            default:
                res = (To) sycl::nan(uint32_t(0));
                break;
        }
    }

    const int count = e - b;
    if (op == GGML_OP_POOL_AVG) {
        res = (count > 0) ? (res / count) : (To) 0;
    }
    o_ptr[cur_ow] = res;
}

void ggml_sycl_op_pool2d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    const float * src0_dd = static_cast<const float *>(dst->src[0]->data);
    float *       dst_dd  = static_cast<float *>(dst->data);

    const int32_t * opts = (const int32_t *)dst->op_params;
    enum ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    const int k0 = opts[1];
    const int k1 = opts[2];
    const int s0 = opts[3];
    const int s1 = opts[4];
    const int p0 = opts[5];
    const int p1 = opts[6];

    const int64_t IH = dst->src[0]->ne[1];
    const int64_t IW = dst->src[0]->ne[0];

    const int64_t N = dst->ne[3];
    const int64_t OC = dst->ne[2];
    const int64_t OH = dst->ne[1];
    const int64_t OW = dst->ne[0];

    const int parallel_elements = N * OC * OH * OW;
    const int num_blocks = (parallel_elements + SYCL_POOL2D_BLOCK_SIZE - 1) / SYCL_POOL2D_BLOCK_SIZE;
    sycl::range<3> block_nums(1, 1, num_blocks);
    main_stream->parallel_for(
        sycl::nd_range<3>(block_nums *
                              sycl::range<3>(1, 1, SYCL_IM2COL_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_IM2COL_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
            pool2d_nchw_kernel(IH, IW, OH, OW, k1, k0, s1, s0, p1, p0,
                               parallel_elements, src0_dd, dst_dd, op,
                               item_ct1);
        });
}

// one work-group per output, for few outputs with long windows (a max over a whole row)
template <typename Ti, typename To>
static void pool1d_ncw_wg_kernel(
        const int iw, const int ow,
        const int k, const int s,
        const int p,
        const Ti * src, To * dst, const enum ggml_op_pool op,
        const sycl::nd_item<1> & item) {
    const int  idx    = item.get_group(0);
    const int  nc     = idx / ow;
    const int  cur_ow = idx % ow;
    const Ti * i_ptr  = src + (size_t) nc * iw;
    const int  start  = cur_ow * s - p;
    const int  b      = sycl::max(0, start);
    const int  e      = sycl::min(iw, start + k);

    To res = op == GGML_OP_POOL_MAX ? (To) -FLT_MAX : (To) 0;
    for (int j = b + (int) item.get_local_id(0); j < e; j += (int) item.get_local_range(0)) {
        const To cur = i_ptr[j];
        res = op == GGML_OP_POOL_MAX ? sycl::max(res, cur) : res + cur;
    }
    const auto grp = item.get_group();
    res = op == GGML_OP_POOL_MAX ? sycl::reduce_over_group(grp, res, sycl::maximum<To>()) :
                                   sycl::reduce_over_group(grp, res, sycl::plus<To>());
    if (item.get_local_id(0) == 0) {
        if (op == GGML_OP_POOL_AVG) {
            res = (e - b > 0) ? res / (e - b) : (To) 0;
        }
        dst[(size_t) nc * ow + cur_ow] = res;
    }
}

void ggml_sycl_op_pool1d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    const float * src0_dd = static_cast<const float *>(dst->src[0]->data);
    float *       dst_dd  = static_cast<float *>(dst->data);

    const int32_t * opts = (const int32_t *)dst->op_params;
    enum ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    const int k0 = opts[1];
    const int s0 = opts[2];
    const int p0 = opts[3];

    const int64_t IW = dst->src[0]->ne[0];
    const int64_t OW = dst->ne[0];
    const int64_t NC = dst->ne[3] * dst->ne[2] * dst->ne[1];

    const int parallel_elements = NC * OW;

    // a work-item per output leaves the device idle when there are few outputs, each a long serial loop
    if ((op == GGML_OP_POOL_MAX || op == GGML_OP_POOL_AVG) && k0 >= 4096 && parallel_elements <= 4096) {
        const size_t wg = std::min<size_t>(1024, ggml_sycl_info().max_work_group_sizes[ctx.device]);
        main_stream->parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t) parallel_elements * wg), sycl::range<1>(wg)),
            [=](sycl::nd_item<1> item) {
                pool1d_ncw_wg_kernel(IW, OW, k0, s0, p0, src0_dd, dst_dd, op, item);
            });
        return;
    }

    const int num_blocks = (parallel_elements + SYCL_POOL1D_BLOCK_SIZE - 1) / SYCL_POOL1D_BLOCK_SIZE;
    sycl::range<3> block_nums(1, 1, num_blocks);
    main_stream->parallel_for(
        sycl::nd_range<3>(block_nums *
                              sycl::range<3>(1, 1, SYCL_POOL1D_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_POOL1D_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
            pool1d_ncw_kernel(IW, OW, k0, s0, p0,
                              parallel_elements, src0_dd, dst_dd, op,
                              item_ct1);
        });
}
