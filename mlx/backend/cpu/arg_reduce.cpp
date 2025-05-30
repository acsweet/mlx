// Copyright © 2023 Apple Inc.

#include <cassert>

#include "mlx/backend/common/utils.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/primitives.h"

namespace mlx::core {

namespace {

template <typename InT, typename OpT>
void arg_reduce(const array& in, array& out, const OpT& op, int axis) {
  auto axis_size = in.shape()[axis];
  auto axis_stride = in.strides()[axis];
  Strides strides = remove_index(in.strides(), axis);
  Shape shape = remove_index(in.shape(), axis);
  auto in_ptr = in.data<InT>();
  auto out_ptr = out.data<uint32_t>();

  for (uint32_t i = 0; i < out.size(); ++i) {
    auto loc = elem_to_loc(i, shape, strides);
    auto local_in_ptr = in_ptr + loc;
    uint32_t ind_v = 0;
    InT v = (*local_in_ptr);
    for (uint32_t j = 0; j < axis_size; ++j, local_in_ptr += axis_stride) {
      op(j, (*local_in_ptr), &ind_v, &v);
    }
    out_ptr[i] = ind_v;
  }
}

template <typename InT>
void arg_reduce_dispatch(
    const array& in,
    array& out,
    ArgReduce::ReduceType rtype,
    int axis) {
  switch (rtype) {
    case ArgReduce::ArgMin: {
      auto op = [](auto ind_x, auto x, auto ind_y, auto y) {
        if (x < (*y)) {
          (*y) = x;
          (*ind_y) = ind_x;
        }
      };
      arg_reduce<InT>(in, out, op, axis);
      break;
    }
    case ArgReduce::ArgMax: {
      auto op = [](auto ind_x, auto x, auto ind_y, auto y) {
        if (x > (*y)) {
          (*y) = x;
          (*ind_y) = ind_x;
        }
      };
      arg_reduce<InT>(in, out, op, axis);
      break;
    }
  }
}

} // namespace

void ArgReduce::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  auto& in = inputs[0];
  out.set_data(allocator::malloc(out.nbytes()));
  auto& encoder = cpu::get_command_encoder(stream());
  encoder.set_input_array(in);
  encoder.set_output_array(out);
  encoder.dispatch([in = array::unsafe_weak_copy(in),
                    out = array::unsafe_weak_copy(out),
                    reduce_type_ = reduce_type_,
                    axis_ = axis_]() mutable {
    switch (in.dtype()) {
      case bool_:
        arg_reduce_dispatch<bool>(in, out, reduce_type_, axis_);
        break;
      case uint8:
        arg_reduce_dispatch<uint8_t>(in, out, reduce_type_, axis_);
        break;
      case uint16:
        arg_reduce_dispatch<uint16_t>(in, out, reduce_type_, axis_);
        break;
      case uint32:
        arg_reduce_dispatch<uint32_t>(in, out, reduce_type_, axis_);
        break;
      case uint64:
        arg_reduce_dispatch<uint64_t>(in, out, reduce_type_, axis_);
        break;
      case int8:
        arg_reduce_dispatch<int8_t>(in, out, reduce_type_, axis_);
        break;
      case int16:
        arg_reduce_dispatch<int16_t>(in, out, reduce_type_, axis_);
        break;
      case int32:
        arg_reduce_dispatch<int32_t>(in, out, reduce_type_, axis_);
        break;
      case int64:
        arg_reduce_dispatch<int64_t>(in, out, reduce_type_, axis_);
        break;
      case float16:
        arg_reduce_dispatch<float16_t>(in, out, reduce_type_, axis_);
        break;
      case float32:
        arg_reduce_dispatch<float>(in, out, reduce_type_, axis_);
        break;
      case bfloat16:
        arg_reduce_dispatch<bfloat16_t>(in, out, reduce_type_, axis_);
        break;
      case float64:
        arg_reduce_dispatch<double>(in, out, reduce_type_, axis_);
        break;
      case complex64:
        arg_reduce_dispatch<complex64_t>(in, out, reduce_type_, axis_);
        break;
    }
  });
}

} // namespace mlx::core
