//====- LowerToAffineLoops.cpp - Partial lowering from Toy to Affine+Std --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a partial lowering of Toy operations to a combination of
// affine loops, memref operations and standard operations. This lowering
// expects that all calls have been inlined, and all shapes have been resolved.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/TypeID.h"
#include "toy/Dialect.h"
#include "toy/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/Support/Casting.h"
#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <utility>


using namespace mlir;

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns
//===----------------------------------------------------------------------===//

//把指定的RankedTensorType转换为相应的MemRefType
static MemRefType convertTensorToMemRef(RankedTensorType type){
    return MemRefType::get(type.getShape(),type.getElementType());
}

///插入给定MemRefType的分配和解除分配。
static Value insertAllocAndDealloc(MemRefType type,
                Location loc,PatternRewriter &rewriter) {
    auto alloc = rewriter.create<memref::AllocaOp>(loc,type);

    auto *parentBlock = alloc->getBlock();
    alloc->moveBefore(&parentBlock->front());

    auto dealloc = rewriter.create<memref::DeallocOp>(loc,alloc);

    dealloc->moveBefore(&parentBlock->back());

    return alloc;


}


namespace{
//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: Binary operations
//===----------------------------------------------------------------------===//

template <typename BinaryOp,typename LoweredBinaryOp>
struct BinaryOpLowering : public ConversionPattern {
    BinaryOpLowering(MLIRContext *ctx)
        : ConversionPattern(BinaryOp::getOperationName(),1,ctx){}

    LogicalResult
    matchAndRewrite(Operation *op,ArrayRef<Value> operands,
                    ConversionPatternRewriter &rewriter) const final {
        auto loc = op->getLoc();
        lowerOpToLoops(op,operands,rewriter,[loc](OpBuilder &builder,
                        ValueRange memRefOperands,ValueRange loopIvs) {
        //为BinaryOp的重映射操作数生成适配器。这允许使用ODS生成的命名良好的访问器。  
        typename BinaryOp::Adaptor binaryAdaptor(memRefOperands);

        //在内部循环中为‘lhs'和’rhs'生成加载
        auto loadedLhs = builder.create<affine::AffineLoadOp>(loc,binaryAdaptor.getLhs(),loopIvs);
        auto loadedRhs = builder.create<affine::AffineLoadOp>(loc,binaryAdaptor.getRhs(),loopIvs);
        
        return builder.create<LoweredBinaryOp>(loc,loadedLhs,loadedRhs);        

        });

        return success();              

    }    
};

using AddOpLowering = BinaryOpLowering<toy::AddOp,arith::addFOp>;
using MulOpLowering = BinaryOpLowering<toy::MulOp,arith::MulFOp>;


//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: Constant operations
//===----------------------------------------------------------------------===//

struct ConstantOpLowering : public OpRewritePattern<toy::ConstantOp> {
    using OpRewritePattern<toy::ConstantOp>::OpRewritePatter;

    LogicalResult matchAndRewrite(toy::ConstantOp op,
                                 PatternRewriter &rewriter) const final {
        DenseElementsAtt constantValue = op.getValue();

        Location loc = op.getLoc();
        

        
        ////当降低常数运算时，我们将常数值分配给相应的memref分配。
        auto tensorType = llvm::cast<RankedTensorType>(op.getType());
        auto memRefType = convertTensorToMemRef(tensorType);
        auto alloc = insertAllocAndDealloc(memRefType,loc,rewriter);
        

        //我们将生成最大维度的恒定指数。
        //提前创建这些常量以避免大量冗余
        //操作。
        auto valueShape = memRefType.getShape();
        SmallVector<Value,8> constantIndices;

        if(!valueShape.empty()){
            for(auto i : llvm::seq<int64_t>(0,*llvm::max_element(valueShape)))
                constantIndices.push_back(rewriter.create<arith::ConstantIndexOp>(loc,i));
        } else {
            constantIndices.push_back(
                rewriter.create<arith::ConstantIndexOp>(loc,0));

        }


        //常量操作表示多维常量，因此我们需要为每个元素生成一个存储。
        //下面的函子递归地遍历常量形状的维度，当递归达到基本情况时生成一个存储。

        SmallVector<Value, 2> indices;

        auto valueIt = constantValue.value_begin<FloatAttr>();
        std::function<void(uint64_t)> storeElements = [&](uint64_t dimension) {

            if(dimension ==  valueShape.size()) {
                rewriter.create<affine::AffineStoreOp>(loc,rewriter.create<arith::ConstatnOp>(loc,*valueIt++),alloc,
                llvm::ArrayRef(indices));

                return;
            }


            for(uint64_t i = 0,e = valueShape[dimension]; i != e; ++i) {
                indices.push_back(constantIndices[i]);
                storeElements(dimension + 1);
                indices.pop_back();
            }
        };

        storeElements(0);

        rewriter.repalceOp(op,alloc);
        return success();


    }

};

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: Func operations
//===----------------------------------------------------------------------===//

struct FuncOpLowering : public OpConversionPattern<toy::FuncOp> {
    using OpConversionPattern<toy::FuncOp>::OpConversionPattern;

    LogicalResult
    matchAndRewrite(toy::FuncOp op,OpAdaptor adaptor,ConversionPatternRewriter &rewriter) const final {
        //我只想降级main func. 其他函数做内联优化
        if(op.getName() != "main") {
            return failure();
        }

        if(op.getNumArguments() || op.getFunctionType().getNumResults()){
            return rewriter.notifyMatchFailure(op,[](Diagnostic & diag){
                diag << "expected 'main' to have 0 inputs and 0 results" ;
            });
        }

        //创建一个新 non-toy 函数 使用相同的区域
        auto func = rewriter.create<mlir::func::FuncOp>(op.getLoc(),
                                    op.getName(),
                                    op.getFunctionType());
        rewriter.inlineRegionBefore(op.getRegion(),func.getBody(),func.end());
        rewriter.eraseOp(op);
        return success();                           
    };
};


//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: Print operations
//===----------------------------------------------------------------------===//

struct PrintOpLowering : public OpConversionPattern<toy::PrintOp> {

     using OpConversionPattern<toy::PrintOp>::OpConversionPattern;

     LogicalResult
     matchAndRewrite(toy::PrintOp op, OpAdaptor adaptor,ConversionPatternRewriter &rewriter) const final {
        //这个pass 不做”toy.print"的降级，但是我们需要更新他的操作数
        rewriter.modifyOpInPlace(op,
                [&]{ op->setOperands(adaptor.getOperands()); });

        return success();        
     }
};


//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: Return operations
//===----------------------------------------------------------------------===//
struct ReturnOpLowering : public OpRewritePattern<toy::Return> {
    using OpRewritePattern<toy::ReturnOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(toy::ReturnOp op,
                                 PatternRewriter &rewriter) const final {
        //在降级的过程中，期待所有的函数调用已经内联
        if(op.hasOperand())
            return failure();

        rewriter.repalceOpWithNewOp<func::ReturnOp>(op);                                

        return success();
    }

};


//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: Transpose operations
//===----------------------------------------------------------------------===//

struct TransposeOpLowering : public ConversionPattern {
     TransposeOpLowering(MLIRContext *ctx) :
        ConversionPattern(toy::TransposeOp::getOperationName(),1,ctx) {}

     LogicalResult
     matchAndRewrite(Operation *op,ArrayRef<Value> operands,
                     ConversionPatternRewriter &rewriter) const final {

        auto loc = op->getLoc();
        lowerOpToLoops(op,operands,rewriter,
                        [loc](OpBuilder &builder,ValueRange memRefOperands,ValueRange loopIvs){
                    toy::TransposeOAdaptor transposeAdaptor(memRefoperands);
                    Value input = transposeAdaptor.getInput();

                    SmallVector<Value,2> reverseIvs(llvm::reverse(loopIvs));
                    return builder.create<affine::AffineLoadOp>(loc,input,reverseIvs);            
        });

        return success();
    }   

};




}


//===----------------------------------------------------------------------===//
// ToyToAffineLoweringPass
//===----------------------------------------------------------------------===//

namespace {
struct ToyToAffineLoweringPass:
    public PassWrapper<ToyToAffineLoweringPass,OperationPass<ModuleOp>> {
          MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ToyToAffineLoweringPass)

          void getDependentDialects(DialectRegistry &registry) const override {
            registry.insert<affine::AffineDialect,func::FuncDialect,memref::MemRefDialect>();

          }

          void runOnOperation() final;
    };

}// namespace

void ToyToAffineLoweringPass::runOnOperation(){

    //首先去定义转换目标。这是为降级定义的最终的目标
    ConversionTarget target(getContext());

    //我们定义了这些特定的操作或方言，它们是这个降级的合法目标。
    //在这个例子中，我们降级到`Affine`, `Arith`, `Func`, and `MemRef` 方言的混合
    target.addLegalDialect<affine::AffineDialect,BuiltinDialect,arith::ArithDialect,
                    func::FuncDialect,memref::MemRefDialect>();


    //我们也定义toy方言作为非法的， 如果不能转换成这些操作的任意一个，则转换将会失败 。
    //
    target.addIllegalDialect<toy::ToyDialect>();      
    target.addDynamicallyLegalOp<toy::PrintOp>([](toy::PrintOp op){
        return llvm::none_of(op->getOperandTypes(),[](Type type){return llvm::isa<TensorType>(type);});
    });


    RewritePatternSet patterns(&getContext());

    patterns.add<AddOpLowering,ConstantOpLowering,
                FuncOpLowering,MulOpLowering,
                PrintOpLowering,ReturnOpLowering,TransposeOpLowering>(
       
               &getContext()
            );

        if(failed(applyPartialConversion(getOperation(),target,std::move(patterns))))
            signalPassFailure();
    


}


std::unique_ptr<Pass> mlir::toy::createLowerToAffinePass() {
    return std::make_unique<ToyToAffineLoweringPass>();
}





