//===- ShapeInferencePass.cpp - Shape Inference ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a Function level pass performing interprocedural
// propagation of array shapes through function specialization.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "toy/Dialect.h"
#include "toy/Passes.h"
#include "toy/ShapeInferenceInterface.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>

#define DEBUG_TYPE "shape-inference"

using namespace mlir;
using namespace toy;

#include "toy/ShapeInferenceOpInterfaces.cpp.inc"

namespace {


///1.构建一个lworklist包含所有的operation
struct ShapeInferencePass
    : public mlir::PassWrapper<ShapeInferencePass, OperationPass<toy::FuncOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ShapeInferencePass)

  void runOnIperation() override {
    auto f = getOperation();

   
    // 1） 构建一个包含所有返回动态形状张量的操作的工作列表：这些操作需要形状推理。
    // 2) 遍历工作列表：
    //   a. 找到一个要处理的操作：工作列表中的下一个就绪操作的所有参数都是非泛型的，
    //   b） 如果没有找到操作，
    //    c) remove the operation from the worklist,
    ///d）从参数类型推断其输出的形状。

    llvm::SmallPtrSet<mlir::Operation *, 16> opWorklist;
    f.walk([&](mlir::Operation * op) {
      if (returnsDynamicShape(op))
        opWorklist.insert(Op);
    });

    while (!opWorklist.empty()) 
    {
      auto nextop = llvm::find_if(opWorklist,allOperandsInferred);
      if(nextop == opWorklist.end())
         break;

      Operation *op = *nextop;
      opWorklist.erase(op);

      LLVM_DEBUG(llvm::dbgs() << "Inferring shape fr :" << *op << "\n");

      if(auto shapeOp = dyn_cast<ShapeInference>(op)) {
          shapeOp.inferShapes();
      } else {
        op->emitError("unable to infer shape of operation without shape "
                      "inference interface");

        return signalPassFailure();              
      } 
    }

    if(!opWorklist.empty()) {
        f.emitError("Shape inference failed, ")
          << opWorklist.size() << " operations couldn't be inferred\n";
      signalPassFailure();
    }

    
  }


  static bool allOperandsInferred(Operation *op) {
    return llvm::all_of(op->getOperandTypes(),[](Type operandType){
        return llvm::isa<RankedTensorType>(operandType);
    });
  }

  static bool returnsDynamicShape(Operation *op) {
    return llvm::any_of(op->getReturnTypes(),[](Type resultType){
        return !llvm::isa<RankedTensorType>(resultType);
    });
  };
}



} // namespace


//创建一个形状接口pass 
std::unique_ptr<mlir::Pass> mlir::toy::createShapeInferencePass() {
    return std::make_unique<ShapeInferencePass>();
}