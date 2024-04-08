//===----------------------------------------------------------------------===//
//
// This file implements a set of simple combiners for optimizing operations in
// the Toy dialect.
//
//===----------------------------------------------------------------------===//


#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "toy/Dialect.h"


using namespace mlir;
using namespace toy;

namespace {
    #include "ToyCombine.inc";
}


//使用c++重写模式实现TransposeOp。用来优化transpose(transpose(x)) -> x
//

struct SimplifyRedundantTranspose : public mlir::OpRewritePattern<TransposeOp> {

    //我注册这个模式去匹配IR中每一个toy.transpose
    //构造函数
    SimplifyRedundantTranspose(mlir::MLIRContext *context)
       : OpRewritePattern<TransposeOp>(context,1){}

    mlir::LogicalResult  
    matchAndRewrite(TransposeOp op,mlir::PatternRewriter &rewriter) const override {

        mlir::Value transposeInput = op.getOperand();
        TransposeOp transposeInputOp = transposeInput.getDefiningOp<TransposeOp>();

        if(!transposeInputOp){
            return failure();
        }

        rewriter.replaceOp(op,{transposeInputOp.getOperand()});

        return success();

    }
}



void TransposeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context){
    results.add<SimplifyRedundantTranspose>(context);
}


void ReshapeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.add<ReshapeReshapeOptPattern, RedundantReshapeOptPattern,
              FoldConstantReshapeOptPattern>(context);
}