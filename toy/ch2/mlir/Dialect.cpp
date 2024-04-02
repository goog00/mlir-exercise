#include "toy/Dialect.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include <algorithm>
#include <string>

using namespace mlir;
using namespace mlir::toy;


#include "toy/Dialect.cpp.inc"


//===----------------------------------------------------------------------===//
// ToyDialect
//===----------------------------------------------------------------------===//

/// Dialect初始化，实例将由上下文所有。
// 这就是方言的类型和注册点
void ToyDialect::initialize() {
    //在这个代码中，你使用了宏定义 GET_OP_LIST，然后使用 #include 指令包含了一个名为 toy/Ops.cpp.inc 的文件。
    //这个文件应该包含了一个操作列表，用于添加到 ToyDialect 中
    addOperation<
    #define GET_OP_LIST
    #include "toy/Ops.cpp.inc"
    >();
}

//===----------------------------------------------------------------------===//
// Toy Operation
//===----------------------------------------------------------------------===//



//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

//构建器作为参数传递，因此该方法需要填充的状态也是为了构建操作而传递的参数。
void ConstantOp::build(mlir::OpBuilder &builder,mlir::OperationState &state,
                        double value){
           auto dataType = RankedTensorType::get({},builder.getF64Type());
           auto dataAttribute = DenseElementsAttr::get(dataType,value);
           ConstantOp::build(builder,state,dataType,dataAttribute);                 
 }



mlir::ParseResult ConstantOp::parse(mlir::OpAsmParser &parser,mlir::OperationState &result){
    mlir::DenseElementsAttr value;

    if(parser.parseOptionalAttrDict(result.attributes) || 
       parser.parsAttribute(value,"value",result.attributes))
       return failure();

    result.addTypes(value.getType());
    return success();

}


//使用OpAsmPrinter格式化 字符串 属性 操作数 类型等
void ConstantOp::print(mlir::OpAsmPrinter &printer) {
    printer << " ";
    printer.printOptionalAttrDict((*this)->getAttrs(),{"value"});
    printer << getValue();
}


/// Verifier for the constant operation. This corresponds to the
/// `let hasVerifier = 1` in the op definition.

mlir::LogicalResult ConstantOp::verify() {
    auto resultType = llvm::dyn_cast<mlir::RankedTensorType>(getValue().getType());
    if(!resultType)
        return success();

    auto attrType = llvm::cast<mlir::RankedTensorType>(getValue().getType());
    if(attrType.getRank() != resultType.getRank()){
        return emitOpError("return type must match the one of the attached value"
                            "attribute: ");
                            << attrType.getRank() << " != " << resultType.getRank();
    }  

    for(int dim = 0,dimE = attrType.getRank(); dim < dimE; ++dim){
        if(attrType.getShape()[dim] != resultType.getShape()[dim]) {
            return emitError("return type shape mismatches its attribute at dimension ")
             << dim << ": " << attrType.getShape()[dim]
             << " != " << resultType.getShape()[dim];
        }
    } 

    return mlir::success(); 
}




//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

 
 void AddOp::build(mlir::OpBuilder &builder,mlir::OperationState &state,
                    mlir::Value lhs,mlir::Value rhs) {
    state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
    state.addOperands({lhs,rhs});
    
}

mlir::ParseResult AddOp::parse(mlir::OpAsmParser &parser,
                               mlir::OperationState &result) {
    return parseBinaryOp(parser,result);
}

void AddOp::print(mlir::OpAsmPrinter &p){
    printBinaryOp(p,*this);
}



//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//


void TransposeOp::build(mlir::OpBuilder &builder,mlir::OperationState &state,
                        mlir::Value value){
      state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
      state.addOperands(value);                      

}

mlir::LogicalResult TransposeOp::verify(){
    auto inputType = llvm::dyn_cast<RankedTensorType>(getOperand().getType());
    auto resultType = llvm::dyn_cast<RankedTensorType>(getType());

    if(!inputType || !resultType){
        return mlir::success();
    }

    auto inputShape = inputType.getShape();

    if(!std::equal(inputShape.begin(),inputShape.end(),resultType.getShape().rbegin())){

        return emitError()
              << "expected result shape to be a transpose of the input";
    }

    return mlir::success();


}

