/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */
 /**
  * based on:
  * Copyright (c) 2016-present, Facebook, Inc.
  *
  * Licensed under the Apache License, Version 2.0 (the "License");
  * you may not use this file except in compliance with the License.
  * You may obtain a copy of the License at
  *
  *     http://www.apache.org/licenses/LICENSE-2.0
  *
  * Unless required by applicable law or agreed to in writing, software
  * distributed under the License is distributed on an "AS IS" BASIS,
  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  * See the License for the specific language governing permissions and
  * limitations under the License.
  */

#include "caffe2/video/affine_nd_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(AffineNd,
                      AffineNdOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(AffineNdGradient,
                      AffineNdGradientOp<float, CPUContext>);

// Input: X, scale, bias; Output: Y
OPERATOR_SCHEMA(AffineNd)
    .NumInputs(3)
    .NumOutputs(1)
    .AllowInplace({{0, 0}});
// Input: scale, dY; Output: dX
OPERATOR_SCHEMA(AffineNdGradient)
    .NumInputs(2)
    .NumOutputs(1)
    .AllowInplace({{1, 0}});

class GetAffineNdGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "AffineNdGradient", "",
        vector<string>{I(1), GO(0)},
        vector<string>{GI(0)});
  }
};

REGISTER_GRADIENT(AffineNd, GetAffineNdGradient);

} // namespace caffe2
