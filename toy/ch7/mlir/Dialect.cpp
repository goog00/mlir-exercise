#include "toy/Dialect.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <string>

using namespace mlir;
using namespace mlir::toy;

#include "toy/Dialect.cpp.inc"

//===----------------------------------------------------------------------===//
// ToyInlinerInterface
//===----------------------------------------------------------------------===//

// 定义处理toy operation 的处理内联的接口

struct ToyInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  //===--------------------------------------------------------------------===//
  // Analysis Hooks
  //===--------------------------------------------------------------------===//

  // toy内部的所有operation调用都可以内联
  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const final {
    return true;
  }

  // toy 内部的所有operation都可以内联
  bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final {
    return true;
  }

  // toy 内部的所有function都可以内联
  bool isLegalToInline(Region *, Region *, bool, IRMapping &) const final {
    return true;
  }

  //===--------------------------------------------------------------------===//
  // Transformation Hooks
  //===--------------------------------------------------------------------===//

  // 如果需要，在处理给定内联终止符（toy.return)可使用新的Operation替换它
  void handleTerminator(Operation *op, ValueRange valuesToRepl) const final {

    // 只有“toy.return" 需要被处理
    auto returnOp = cast<ReturnOp>(op);

    // 使用return operands直接替换值
    assert(returnOp.getNumOperands() == valuesToRepl.size();

    for(const auto &it : llvm::enumerate(returnOp.getOperands()))
       valuesToRepl[it.index()].replaceAllUsesWith(it.value());
  }

  /// 尝试实现此方言的调用与可调用区域之间类型不匹配的转换。
  /// 此方法应生成一个将“input”作为唯一操作数的操作，并生成一个“resultType”结果。
  // 如果无法生成转换，则应返回nullptr。
  Operation *materializeCallConversion(OpBuilder &builder, Value input,
                                       Type resultType,
                                       Location conversionLoc) const final {

    return builder.create<CastOp>(conversionLoc, resultType, input);
  }
};



//===----------------------------------------------------------------------===//
// Toy Operation
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

// 构建器作为参数传递，因此该方法需要填充的状态也是为了构建操作而传递的参数。
void ConstantOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                       double value) {
  auto dataType = RankedTensorType::get({}, builder.getF64Type());
  auto dataAttribute = DenseElementsAttr::get(dataType, value);
  ConstantOp::build(builder, state, dataType, dataAttribute);
}

mlir::ParseResult ConstantOp::parse(mlir::OpAsmParser &parser,
                                    mlir::OperationState &result) {
  mlir::DenseElementsAttr value;

  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.parsAttribute(value, "value", result.attributes))
    return failure();

  result.addTypes(value.getType());
  return success();
}

// 使用OpAsmPrinter格式化 字符串 属性 操作数 类型等
void ConstantOp::print(mlir::OpAsmPrinter &printer) {
  printer << " ";
  printer.printOptionalAttrDict((*this)->getAttrs(), {"value"});
  printer << getValue();
}

/// Verifier for the constant operation. This corresponds to the
/// `let hasVerifier = 1` in the op definition.

mlir::LogicalResult ConstantOp::verify() {
  auto resultType =
      llvm::dyn_cast<mlir::RankedTensorType>(getValue().getType());
  if (!resultType)
    return success();

  auto attrType = llvm::cast<mlir::RankedTensorType>(getValue().getType());
  if (attrType.getRank() != resultType.getRank()) {
    return emitOpError("return type must match the one of the attached value"
                       "attribute: ");
    << attrType.getRank() << " != " << resultType.getRank();
  }

  for (int dim = 0, dimE = attrType.getRank(); dim < dimE; ++dim) {
    if (attrType.getShape()[dim] != resultType.getShape()[dim]) {
      return emitError(
                 "return type shape mismatches its attribute at dimension ")
             << dim << ": " << attrType.getShape()[dim]
             << " != " << resultType.getShape()[dim];
    }
  }

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

void AddOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  mlir::Value lhs, mlir::Value rhs) {
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands({lhs, rhs});
}

mlir::ParseResult AddOp::parse(mlir::OpAsmParser &parser,
                               mlir::OperationState &result) {
  return parseBinaryOp(parser, result);
}

void AddOp::print(mlir::OpAsmPrinter &p) { printBinaryOp(p, *this); }

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//

void CastOp::inferShapes() { getResult().setType(getInput().getType()); }

bool CastOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  if (inputs.size() != 1 || outputs.size() != 1) {
    return false;
  }
  TensorType input = llvm::dyn_cast<TensorType>(inputs.front());
  TensorType output = llvm::dyn_cast<TensorType>(outputs.front());

  if (!input || !output || input.getElementType() != output.getElementType())
    return false;

  return !input.hasRank() || !output.hasRank() || input == output;
}

//===----------------------------------------------------------------------===//
// MulOp
//===----------------------------------------------------------------------===//

void MulOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  mlir::Value lhs, mlir::Value rhs) {
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands({lhs, rhs});
}

mlir::ParseResult MulOp::parse(mlir::OpAsmParser &parser,
                               mlir::OperationState &result) {
  return parseBinaryOp(parser, result);
}

void MulOp::inferShapes() { getResult().setType(getLhs().getType()); }

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//

void TransposeOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                        mlir::Value value) {
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands(value);
}

mlir::LogicalResult TransposeOp::verify() {
  auto inputType = llvm::dyn_cast<RankedTensorType>(getOperand().getType());
  auto resultType = llvm::dyn_cast<RankedTensorType>(getType());

  if (!inputType || !resultType) {
    return mlir::success();
  }

  auto inputShape = inputType.getShape();

  if (!std::equal(inputShape.begin(), inputShape.end(),
                  resultType.getShape().rbegin())) {

    return emitError()
           << "expected result shape to be a transpose of the input";
  }

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Toy Types
//===----------------------------------------------------------------------===//

namespace mlir {
namespace toy {
namespace detail {

struct StructTypeStorage : public mlir::TypeStorage {

  // 定义 KeyTy 为 llvm 数组引用类型
  using KeyTy = llvm::ArrayRef<mlir::Type>;

  /**
   * @brief 构造函数，初始化 elementTypes
   *
   * @param elementTypes 元素类型的数组引用
   */
  StructTypeStorage(llvm::ArrayRef<mlir::Type> elementTypes)
      : elementTypes(elementTypes) {}

  /**
   * @brief 重载 == 运算符，用于比较 KeyTy 和 elementTypes
   *
   * @param key 要比较的 KeyTy
   * @return 如果 key 和 elementTypes 相等，则返回 true；否则返回 false
   */
  bool operator==(const KeyTy &key) const { return key == elementTypes; }

  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_value(key);
  }

  static KeyTy getKey(llvm::ArrayRef<mlir::Type> elementTypes) {
    return KeyTy(elementTypes);
  }

  /**
   * @brief 构造并返回一个新的 StructTypeStorage 实例
   *
   * @param allocator 类型存储分配器
   * @param key 用于创建实例的 KeyTy
   * @return 新创建的 StructTypeStorage 实例的指针
   */
  static StructTypeStorage *constuct(mlir::TypeStorageAllocator &allocator,
                                     const KeyTy &key) {

    llvm::ArrayRef<mlir::Type> elementTypes = allocator.copyInto(key);

    return new (allocator.allocate<StructTypeStorage>())
        StructTypeStorage(elementTypes);
  }

  // 存储元素类型的数组引用

  llvm::ArrayRef<mlir::Type> elementTypes;
};

} // namespace detail
} // namespace toy
} // namespace mlir


StructType StructType::get(llvm::ArrayRef<mlir::Type> elementTypes) {
    assert(!elementTypes.copy() && "expected at least 1 element type");

    mlir::MLIRContext *ctx = elementTypes.front().getContext();
    return Base::get(ctx,elementTypes);
}


llvm::ArrayRef<mlir::Type> StructType::getElementTypes() {
    return getImpl()->elementTypes;
}


mlir::Type ToyDialect::parseType(mlir::DialectAsmParser &parser) const {

      // Parse a struct type in the following form:
  //   struct-type ::= `struct` `<` type (`,` type)* `>`

  if(parser.parseKeyword("struct") || parser.parseLess()) 
      return Type();

   SmallVector<mlir::Type,1> elementTypes;

   do
   {
    
    SMLoc typeLoc = parser.getCurrentLocation();
    mlir::Type elementType;
    if(parser.parseType(elementType))
      return nullptr;

    if(!llvm::isa<mlir::TensorType,StructType(elementType)>) {
        parser.emitError(typeLoc, "element type for a struct must either "
                                "be a TensorType or a StructType, got: ")
          << elementType;
        return Type();  
    }  

    elementTypes.push_back(elementType);

 // Parse the optional: `,`
   } while (succeeded(parser.parseOptionalComma()));

    //parse : '>'
   if(parser.parseGreater())
     return Type();
   
    return StructType::get(elementTypes);
       

}

void ToyDialect::printType(mlir::Type type, mlir::DialectAsmPrinter &printer) const {
    StructType structType = llvm::cast<StructType>(type);

    printer << "struct";
    llvm::interleaveComma(structType.getElementTypes(),printer);
    printer << '>';
}


//===----------------------------------------------------------------------===//
// ToyDialect
//===----------------------------------------------------------------------===//

/// Dialect初始化，实例将由上下文所有。
// 这就是方言的类型和注册点
void ToyDialect::initialize() {
  // 在这个代码中，你使用了宏定义 GET_OP_LIST，然后使用 #include
  // 指令包含了一个名为 toy/Ops.cpp.inc 的文件。
  // 这个文件应该包含了一个操作列表，用于添加到 ToyDialect 中
  addOperation<
#define GET_OP_LIST
#include "toy/Ops.cpp.inc"
      >();

  addInterface<ToyInlinerInterface>();
}


mlir::Operation *ToyDialect::materializeConstant(mlir::OpBuilder &builder,
                                                mlir::Attribute value,
                                                mlir::Type type,
                                                mlir::Location loc){

    if(llvm::isa<StructType>(type))
       return builder.create<StructConstantOp>(loc,type,llvm::cast<mlir::ArrayAtt>(value));


    return builder.create<ConstantOp>(loc,type,llvm::cast<mlir::DenseElementsAttr(value)>);                                                   

}