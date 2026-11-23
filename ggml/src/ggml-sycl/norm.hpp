//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef GGML_SYCL_NORM_HPP
#define GGML_SYCL_NORM_HPP

#include "common.hpp"

void ggml_sycl_op_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst);

void ggml_sycl_op_rms_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst);

void ggml_sycl_op_rms_norm_fused(ggml_backend_sycl_context& ctx, ggml_tensor* dst, ggml_tensor* mul);

void ggml_sycl_op_rms_norm_scale_fused(ggml_backend_sycl_context& ctx, ggml_tensor* dst, ggml_tensor* scale_tensor);

void ggml_sycl_op_rms_norm_fused_add(ggml_backend_sycl_context& ctx, ggml_tensor* dst, ggml_tensor* mul_tensor, ggml_tensor* add_tensor);

bool ggml_sycl_op_add_rms_norm_fused(ggml_backend_sycl_context & ctx,
                                     ggml_tensor *               add,
                                     ggml_tensor *               rms,
                                     ggml_tensor *               mul);

void ggml_sycl_op_rms_norm_back(ggml_backend_sycl_context& ctx, ggml_tensor* dst);

void ggml_sycl_op_group_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst);

void ggml_sycl_op_l2_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst);
// SCALE(L2_NORM(x)) or SCALE(L2_NORM(MUL(x, w))) with w [ncols, nrows or 1] broadcast over the other dims, mul optional
void ggml_sycl_op_l2_norm_fused(ggml_backend_sycl_context & ctx, ggml_tensor * mul, ggml_tensor * l2, ggml_tensor * scale);

#define GGML_SYCL_L2_BATCH_MAX 8
void ggml_sycl_l2_norm_batch(ggml_backend_sycl_context & ctx, ggml_tensor ** nodes, int count);

// An L2 (x/max(|x|, eps), times out_scale, after an optional weight) or RMS norm of rows of ncols, written straight into
// the rows of a SET_ROWS destination (KV cache): token t's nrows normed rows go to row idx[t], element r*ncols + c.
struct ggml_sycl_norm_set_rows_args {
    const float * x;
    size_t        x_nb1, x_nb2;      // bytes: row, token (columns contiguous)
    const float * w;                 // optional [ncols, w_nrows] multiplied before the norm (row r % w_nrows)
    int64_t       w_nrows;
    int64_t       ncols, nrows, ntok;
    bool          rms;
    float         eps;
    float         out_scale;
    char *        dst;
    size_t        dst_nb1;
    bool          dst_f16;
    const void *  idx;
    bool          idx_i64;
};

void ggml_sycl_norm_set_rows(ggml_backend_sycl_context & ctx, const ggml_sycl_norm_set_rows_args & a);

#endif // GGML_SYCL_NORM_HPP
