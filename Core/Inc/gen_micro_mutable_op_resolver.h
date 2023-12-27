/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Generated based on TinyFallNet_6axis_qat_dynR.tflite.

#pragma once

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

constexpr int kNumberOperators = 12;

inline tflite::MicroMutableOpResolver<kNumberOperators> get_resolver()
{
  tflite::MicroMutableOpResolver<kNumberOperators> micro_op_resolver;

  micro_op_resolver.AddAdd();
  micro_op_resolver.AddAveragePool2D();
  micro_op_resolver.AddConv2D();
  micro_op_resolver.AddDequantize();
  micro_op_resolver.AddFullyConnected();
  micro_op_resolver.AddMaxPool2D();
  micro_op_resolver.AddPack();
  micro_op_resolver.AddQuantize();
  micro_op_resolver.AddReshape();
  micro_op_resolver.AddShape();
  micro_op_resolver.AddSoftmax();
  micro_op_resolver.AddStridedSlice();

  return micro_op_resolver;
}
