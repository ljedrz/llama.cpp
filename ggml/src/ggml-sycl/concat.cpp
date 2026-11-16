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

#include "ggml.h"

#include "concat.hpp"

static inline size_t elem_size(ggml_type t) {
    return ggml_type_size(t) / ggml_blck_size(t);
}

template <typename T>
static void concat_T_dim0(const T *x, const T *y, T *dst,
                            const int ne0, const int ne00,
                            const sycl::nd_item<3> &item_ct1) {
  int nidx = item_ct1.get_local_id(2) +
             item_ct1.get_group(2) * item_ct1.get_local_range(2);
  if (nidx >= ne0) {
    return;
  }
  // operation
  int offset_dst = nidx + item_ct1.get_group(1) * ne0 +
                   item_ct1.get_group(0) * ne0 * item_ct1.get_group_range(1);
  if (nidx < ne00) { // src0
    int offset_src = nidx + item_ct1.get_group(1) * ne00 +
                     item_ct1.get_group(0) * ne00 * item_ct1.get_group_range(1);
    dst[offset_dst] = x[offset_src];
  } else {
    int offset_src =
        nidx - ne00 + item_ct1.get_group(1) * (ne0 - ne00) +
        item_ct1.get_group(0) * (ne0 - ne00) * item_ct1.get_group_range(1);
    dst[offset_dst] = y[offset_src];
  }
}

template <typename T>
static void concat_T_dim1(const T *x, const T *y, T *dst,
                            const int ne0, const int ne01,
                            const sycl::nd_item<3> &item_ct1) {
  int nidx = item_ct1.get_local_id(2) +
             item_ct1.get_group(2) * item_ct1.get_local_range(2);
  if (nidx >= ne0) {
    return;
  }
  // operation
  int offset_dst = nidx + item_ct1.get_group(1) * ne0 +
                   item_ct1.get_group(0) * ne0 * item_ct1.get_group_range(1);
  if (item_ct1.get_group(1) < (size_t) ne01) { // src0
    int offset_src =
        nidx + item_ct1.get_group(1) * ne0 + item_ct1.get_group(0) * ne0 * ne01;
    dst[offset_dst] = x[offset_src];
  } else {
    int offset_src =
        nidx + (item_ct1.get_group(1) - ne01) * ne0 +
        item_ct1.get_group(0) * ne0 * (item_ct1.get_group_range(1) - ne01);
    dst[offset_dst] = y[offset_src];
  }
}

template <typename T>
static void concat_T_dim2(const T *x, const T *y, T *dst,
                            const int ne0, const int ne02,
                            const sycl::nd_item<3> &item_ct1) {
  int nidx = item_ct1.get_local_id(2) +
             item_ct1.get_group(2) * item_ct1.get_local_range(2);
  if (nidx >= ne0) {
    return;
  }
  // operation
  int offset_dst = nidx + item_ct1.get_group(1) * ne0 +
                   item_ct1.get_group(0) * ne0 * item_ct1.get_group_range(1);
  if (item_ct1.get_group(0) < (size_t) ne02) { // src0
    int offset_src = nidx + item_ct1.get_group(1) * ne0 +
                     item_ct1.get_group(0) * ne0 * item_ct1.get_group_range(1);
    dst[offset_dst] = x[offset_src];
  } else {
    int offset_src =
        nidx + item_ct1.get_group(1) * ne0 +
        (item_ct1.get_group(0) - ne02) * ne0 * item_ct1.get_group_range(1);
    dst[offset_dst] = y[offset_src];
  }
}

template <typename T>
static void concat_T_sycl(const T *x, const T *y, T *dst,
                            int ne00, int ne01, int ne02, int ne0, int ne1,
                            int ne2, int dim, queue_ptr stream) {
  int num_blocks = (ne0 + SYCL_CONCAT_BLOCK_SIZE - 1) / SYCL_CONCAT_BLOCK_SIZE;
  sycl::range<3> gridDim(ne2, ne1, num_blocks);
  switch (dim) {
  case 0:
      stream->parallel_for(sycl::nd_range<3>(gridDim * sycl::range<3>(1, 1, SYCL_CONCAT_BLOCK_SIZE),
                                          sycl::range<3>(1, 1, SYCL_CONCAT_BLOCK_SIZE)),
                        [=](sycl::nd_item<3> item_ct1) { concat_T_dim0<T>(x, y, dst, ne0, ne00, item_ct1); });
      break;
  case 1:
      stream->parallel_for(sycl::nd_range<3>(gridDim * sycl::range<3>(1, 1, SYCL_CONCAT_BLOCK_SIZE),
                                          sycl::range<3>(1, 1, SYCL_CONCAT_BLOCK_SIZE)),
                        [=](sycl::nd_item<3> item_ct1) { concat_T_dim1<T>(x, y, dst, ne0, ne01, item_ct1); });
      break;
  // dim >=2 will be dispatched to the default path
  default:
      stream->parallel_for(sycl::nd_range<3>(gridDim * sycl::range<3>(1, 1, SYCL_CONCAT_BLOCK_SIZE),
                                          sycl::range<3>(1, 1, SYCL_CONCAT_BLOCK_SIZE)),
                        [=](sycl::nd_item<3> item_ct1) { concat_T_dim2<T>(x, y, dst, ne0, ne02, item_ct1); });
      break;
  }
}

// non-contiguous kernel: one work-item per destination element
template<typename T>
static void concat_T_sycl_non_cont(
    queue_ptr stream, const char *src0, const char *src1, char *dst,
    int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03, uint64_t nb00,
    uint64_t nb01, uint64_t nb02, uint64_t nb03, int64_t /*ne10*/,
    int64_t /*ne11*/, int64_t /*ne12*/, int64_t /*ne13*/, uint64_t nb10,
    uint64_t nb11, uint64_t nb12, uint64_t nb13, int64_t ne0, int64_t ne1,
    int64_t ne2, int64_t ne3, uint64_t nb0, uint64_t nb1, uint64_t nb2,
    uint64_t nb3, int32_t dim) {
  const int64_t total = ne0*ne1*ne2*ne3;
  const int64_t o0 = dim == 0 ? ne00 : 0;
  const int64_t o1 = dim == 1 ? ne01 : 0;
  const int64_t o2 = dim == 2 ? ne02 : 0;
  const int64_t o3 = dim == 3 ? ne03 : 0;
  // 32-bit index math when every offset fits (64-bit division is emulated on Intel GPUs)
  const uint64_t lim = (uint64_t) INT32_MAX;
  const bool fits32 = (uint64_t) total <= lim &&
      (uint64_t) ne0*nb0 + ne1*nb1 + ne2*nb2 + ne3*nb3 <= lim &&
      (uint64_t) ne00*nb00 + ne01*nb01 + ne02*nb02 + ne03*nb03 <= lim &&
      (uint64_t) ne0*nb10 + ne1*nb11 + ne2*nb12 + ne3*nb13 <= lim;
  if (fits32) {
    const uint32_t e0 = (uint32_t) ne0, e1 = (uint32_t) ne1, e2 = (uint32_t) ne2;
    const uint32_t a0 = (uint32_t) ne00, a1 = (uint32_t) ne01, a2 = (uint32_t) ne02, a3 = (uint32_t) ne03;
    const uint32_t p0 = (uint32_t) o0, p1 = (uint32_t) o1, p2 = (uint32_t) o2, p3 = (uint32_t) o3;
    const uint32_t b00 = (uint32_t) nb00, b01 = (uint32_t) nb01, b02 = (uint32_t) nb02, b03 = (uint32_t) nb03;
    const uint32_t b10 = (uint32_t) nb10, b11 = (uint32_t) nb11, b12 = (uint32_t) nb12, b13 = (uint32_t) nb13;
    const uint32_t c0 = (uint32_t) nb0, c1 = (uint32_t) nb1, c2 = (uint32_t) nb2, c3 = (uint32_t) nb3;
    const uint32_t n = (uint32_t) total;
    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) ceil_div(total, (int64_t) SYCL_CONCAT_BLOCK_SIZE)*SYCL_CONCAT_BLOCK_SIZE),
                          sycl::range<1>(SYCL_CONCAT_BLOCK_SIZE)),
        [=](sycl::nd_item<1> item_ct1) {
            const uint32_t gid = (uint32_t) item_ct1.get_global_id(0);
            if (gid >= n) {
                return;
            }
            const uint32_t i0 = gid % e0;
            const uint32_t r1 = gid / e0;
            const uint32_t i1 = r1 % e1;
            const uint32_t r2 = r1 / e1;
            const uint32_t i2 = r2 % e2;
            const uint32_t i3 = r2 / e2;
            const T * x;
            if (i0 < a0 && i1 < a1 && i2 < a2 && i3 < a3) {
                x = (const T *) (src0 + (i3*b03 + i2*b02 + i1*b01 + i0*b00));
            } else {
                x = (const T *) (src1 + ((i3 - p3)*b13 + (i2 - p2)*b12 + (i1 - p1)*b11 + (i0 - p0)*b10));
            }
            *(T *)(dst + (i3*c3 + i2*c2 + i1*c1 + i0*c0)) = *x;
        });
    return;
  }
  stream->parallel_for(
      sycl::nd_range<1>(sycl::range<1>((size_t) ceil_div(total, (int64_t) SYCL_CONCAT_BLOCK_SIZE)*SYCL_CONCAT_BLOCK_SIZE),
                        sycl::range<1>(SYCL_CONCAT_BLOCK_SIZE)),
      [=](sycl::nd_item<1> item_ct1) {
          const int64_t gid = item_ct1.get_global_id(0);
          if (gid >= total) {
              return;
          }
          const int64_t i0 = gid % ne0;
          const int64_t i1 = (gid / ne0) % ne1;
          const int64_t i2 = (gid / (ne0*ne1)) % ne2;
          const int64_t i3 = gid / (ne0*ne1*ne2);

          const T * x;
          if (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
              x = (const T *) (src0 + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
          } else {
              x = (const T *) (src1 + (i3 - o3)*nb13 + (i2 - o2)*nb12 + (i1 - o1)*nb11 + (i0 - o0)*nb10);
          }
          *(T *)(dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0) = *x;
      });
}

template <typename T>
void concat_impl_sycl(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    const ggml_tensor *  src0   = dst->src[0];
    const ggml_tensor *  src1   = dst->src[1];
    queue_ptr            stream = ctx.stream();

    const int32_t dim = ((int32_t *) dst->op_params)[0];

    // the contiguous kernels run one work-group per row of dst: short rows (e.g. a [4, 10240] conv window) take the flat kernel
    const bool short_rows = dst->ne[0] < SYCL_CONCAT_BLOCK_SIZE/4;
    if (ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && !(dim != 3 && short_rows)) {
        const T * src0_d = (const T *) src0->data;
        const T * src1_d = (const T *) src1->data;
        T * dst_d = (T *) dst->data;
        size_t type_size = elem_size(dst->type);
        if (dim != 3) {
            for (int i3 = 0; i3 < dst->ne[3]; i3++) {
                concat_T_sycl<T>(src0_d + i3 * (src0->nb[3] / type_size), src1_d + i3 * (src1->nb[3] / type_size),
                                dst_d + i3 * (dst->nb[3] / type_size), src0->ne[0], src0->ne[1], src0->ne[2], dst->ne[0],
                                dst->ne[1], dst->ne[2], dim, stream);
            }
        } else {
            const size_t size0 = ggml_nbytes(src0);
            const size_t size1 = ggml_nbytes(src1);

            SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(dst_d, src0_d, size0)));
            SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(dst_d + size0 / type_size, src1_d, size1)));
        }
    } else {
        concat_T_sycl_non_cont<T>(stream, (const char *) src0->data, (const char *) src1->data, (char *) dst->data,
                                 src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], src0->nb[0], src0->nb[1],
                                 src0->nb[2], src0->nb[3], src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
                                 src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3], dst->ne[0], dst->ne[1], dst->ne[2],
                                 dst->ne[3], dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3], dim);
    }
}

template <typename block_t, int qk, ggml_type qtype>
static void concat_impl_q_sycl(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    const ggml_tensor * src0   = dst->src[0];
    const ggml_tensor * src1   = dst->src[1];
    queue_ptr           stream = ctx.stream();

    const int32_t dim = ((int32_t *) dst->op_params)[0];

    GGML_ASSERT(src0->type == qtype);
    GGML_ASSERT(src1->type == qtype);
    GGML_ASSERT(dst->type == qtype);
    GGML_ASSERT(src0->ne[0] % qk == 0);
    GGML_ASSERT(src1->ne[0] % qk == 0);
    GGML_ASSERT(dst->ne[0] % qk == 0);

    const int ne00_blk = src0->ne[0] / qk;
    const int ne0_blk  = dst->ne[0] / qk;

    if (ggml_is_contiguous(src0) && ggml_is_contiguous(src1)) {
        const block_t * src0_d = (const block_t *) src0->data;
        const block_t * src1_d = (const block_t *) src1->data;
        block_t * dst_d        = (block_t *) dst->data;
        const size_t type_size = sizeof(block_t);

        if (dim != 3) {
            for (int i3 = 0; i3 < dst->ne[3]; i3++) {
                concat_T_sycl<block_t>(
                    src0_d + i3 * (src0->nb[3] / type_size),
                    src1_d + i3 * (src1->nb[3] / type_size),
                    dst_d + i3 * (dst->nb[3] / type_size),
                    ne00_blk, src0->ne[1], src0->ne[2], ne0_blk,
                    dst->ne[1], dst->ne[2], dim, stream);
            }
        } else {
            const size_t size0 = ggml_nbytes(src0);
            const size_t size1 = ggml_nbytes(src1);

            SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(dst_d, src0_d, size0)));
            SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy((char *) dst_d + size0, src1_d, size1)));
        }
    } else {
        concat_T_sycl_non_cont<block_t>(
            stream, (const char *) src0->data, (const char *) src1->data,
            (char *) dst->data,
            ne00_blk, src0->ne[1], src0->ne[2], src0->ne[3],
            src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            src1->ne[0] / qk, src1->ne[1], src1->ne[2], src1->ne[3],
            src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
            ne0_blk, dst->ne[1], dst->ne[2], dst->ne[3],
            dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3], dim);
    }
}

void ggml_sycl_op_concat(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {

    switch (dst->type) {
    case GGML_TYPE_F32:
        concat_impl_sycl<float>(ctx, dst);
        break;
    case GGML_TYPE_F16:
        concat_impl_sycl<sycl::half>(ctx, dst);
        break;
#ifdef GGML_SYCL_HAS_BF16
    case GGML_TYPE_BF16:
        concat_impl_sycl<sycl::ext::oneapi::bfloat16>(ctx, dst);
        break;
#endif
    case GGML_TYPE_I32:
        concat_impl_sycl<int32_t>(ctx, dst);
        break;
    case GGML_TYPE_I16:
        concat_impl_sycl<int16_t>(ctx, dst);
        break;
    case GGML_TYPE_I64:
        concat_impl_sycl<int64_t>(ctx, dst);
        break;
    case GGML_TYPE_I8:
        concat_impl_sycl<int8_t>(ctx, dst);
        break;
    case GGML_TYPE_Q4_0:
        concat_impl_q_sycl<block_q4_0, QK4_0, GGML_TYPE_Q4_0>(ctx, dst);
        break;
    case GGML_TYPE_Q4_1:
        concat_impl_q_sycl<block_q4_1, QK4_1, GGML_TYPE_Q4_1>(ctx, dst);
        break;
    case GGML_TYPE_Q5_0:
        concat_impl_q_sycl<block_q5_0, QK5_0, GGML_TYPE_Q5_0>(ctx, dst);
        break;
    case GGML_TYPE_Q5_1:
        concat_impl_q_sycl<block_q5_1, QK5_1, GGML_TYPE_Q5_1>(ctx, dst);
        break;
    case GGML_TYPE_Q8_0:
        concat_impl_q_sycl<block_q8_0, QK8_0, GGML_TYPE_Q8_0>(ctx, dst);
        break;
    default:
        fprintf(stderr, "%s: unsupported types: dst: %s\n", __func__, ggml_type_name(dst->type));
        GGML_ASSERT(false);
    }
}
