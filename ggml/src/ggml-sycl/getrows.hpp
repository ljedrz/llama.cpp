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

#ifndef GGML_SYCL_GETROWS_HPP
#define GGML_SYCL_GETROWS_HPP

#include "common.hpp"

void ggml_sycl_op_get_rows(ggml_backend_sycl_context & ctx, ggml_tensor *dst);
void ggml_sycl_op_get_rows_back(ggml_backend_sycl_context & ctx, ggml_tensor *dst);

// dst[k, q, h] = table[h, idx[k, q], q] + mask[k, q]: a gathered per-head bias added to a KQ mask, written per head.
// Strides in bytes. The mask is F32 or F16, dst is F32 or F16 with contiguous rows.
struct ggml_sycl_rel_bias_mask_args {
    const char * table;
    const char * idx;
    const char * mask;
    char *       dst;
    int64_t      n_head, n_kv, n_tok;
    size_t       tb0, tb1, tb2;
    size_t       ib0, ib1;
    size_t       mb0, mb1;
    size_t       db1, db2;
    bool         mask_f16;
    bool         dst_f16;
};

void ggml_sycl_rel_bias_mask(const ggml_sycl_rel_bias_mask_args & a, queue_ptr stream);

#endif // GGML_SYCL_GETROWS_HPP
