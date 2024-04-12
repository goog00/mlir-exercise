//====- LowerToLLVM.cpp - Lowering from Toy+Affine+Std to LLVM ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements full lowering of Toy operations to LLVM MLIR dialect.
// 'toy.print' is lowered to a loop nest that calls `printf` on each element of
// the input array. The file also sets up the ToyToLLVMLoweringPass. This pass
// lowers the combination of Arithmetic + Affine + SCF + Func dialects to the
// LLVM one:
//
//                         Affine --
//                                  |
//                                  v
//                       Arithmetic + Func --> LLVM (Dialect)
//                                  ^
//                                  |
//     'toy.print' --> Loop (SCF) --
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/TypeID.h"
#include "toy/Dialect.h"
#include "toy/Passes.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Casting.h"
#include <memory>
#include <utility>


using namespace mlir;

//===----------------------------------------------------------------------===//
// ToyToLLVM RewritePatterns
//===----------------------------------------------------------------------===//
namespace {

class PrintOpLowering : public ConversionPattern {
public:
    explicit PrintOpLowering(MLIRContext *context)
         : ConversionPattern(toy::PrintOp::getOperationName(),1,context){}

    LogicalResult
    matchAndRewrite(Operation *op,ArrayRef<Value> operands,ConversionPatternRewriter &rewriter) const override {
        auto *context = rewriter.getContext();
        auto memRefType = llvm::cast<MemRefType>((*op->operand_type_begin()));
        auto memRefShape = memRefType.getShape();
        auto loc = op->getLoc();

        ModuleOp parentModule = op->getParenTofType<ModuleOp>();


        


    }    
};

}


//===----------------------------------------------------------------------===//
// ToyToLLVMLoweringPass
//===----------------------------------------------------------------------===//


namespace {

struct ToyToLLVMLoweringPass 
        : public PassWrapper<ToyToLLVMLoweringPass,OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ToyToLLVMLoweringPass)

    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<LLVM::LLVMDialect,scf::SCFDialect>();
    }

    void runOnOperation() final;

};

}//namespace

void ToyToLLVMLoweringPass::runOnOperation() {

    LLVMConversionTarget target(getContext());

    target.addLegalOp<ModuleOp>();


    LLVMTypeConverter typeConverter(&getContext());

    RewritePatternSet patterns(&getContext());
    populateAffineToStdConversionPatterns(patterns);
    populateSCFToControlFlowConversionPatterns(patterns);

    mlir::arith::populateArithToLLVMConversionPatterns(typeConverter,patterns);
    populateFinalizeMemRefToLLVMConversionPatterns(typeConverter,patterns);
    cf::populateControlFlowToLLVMConversionPatterns(typeConverter,patterns);
    populateFuncToLLVMConversionPatterns(typeConverter,patterns);



    patterns.add<PrintOpLowering>(&getContext());

    auto module = getOperation();



    
}