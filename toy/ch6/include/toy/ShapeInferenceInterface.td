//===- ShapeInferenceInterface.td - Shape Inference Interface -*- tablegen -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the operations of the Shape Inference Op Interface.
// 定义形状推理的操作接口的operations 
//
//===----------------------------------------------------------------------===//

#ifndef SHAPE_INFERENCE_INTERFACE
#define SHAPE_INFERENCE_INTERFACE

#include "mlir/IR/OpBase.td"

def ShapeInferenceOpInterface : OpInterface<"ShapeInference"> {

    let description = [{
        Interface t access a registered method to infer the return types for an operation that can
        be  used during type Inference
    }];

    let methods = [
        InterfaceMethod<"Infer and set the output shape for the current operatioon.",
                    "void","inferShapes">
    ];
}

#endif