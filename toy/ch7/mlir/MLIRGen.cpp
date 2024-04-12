////该文件实现了针对Toy语言的模块AST的MLIR的简单IR生成。

#include "toy/MLIRGen.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "toy/AST.h"
#include "toy/Dialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "toy/Lexer.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include <cassert>
#include <cstdint>
#include <functional>
#include <numeric>
#include <optional>
#include <vector>

using namespace mlir::toy;
using namespace toy;

using llvm::ArrayRef;
using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;
using llvm::ScopedHashTableScope;
using llvm::SmallVector;
using llvm::StringRef;
using llvm::Twine;

namespace {

// toy ast 到 简单MLIR的实现

// 发射toy 语言的操作，保留语言的语义 ，允许执行基于高层语义信息的分析和转换

class MLIRGenImpl {

private:

  //一个“module”对应一个包含一系列函数的 Toy 源文件。
  mlir::ModuleOp theModule;

  //Builder 是一个辅助类，用于在函数内部创建 IR。
  // Builder 是一个具有状态的类，特别是它保持一个“插入点”：这是下一个操作将被引入的地方。
  mlir::OpBuilder builder;


  //符号表将变量名映射到当前作用域中的值。进入函数会创建一个新的作用域，并将函数参数添加到映射中。
  //当函数的处理终止时，作用域将被销毁，并且在该作用域中创建的映射也将被删除。
  llvm::ScopedHashTable<StringRef, mlir::Value> symbolTable;


  /// Helper conversion for a Toy AST location to an MLIR location.

  mlir::Location loc(const Location &loc) {
    return mlir::FileLineColLoc::get(builder.getStringAttr(*loc.file), loc.line,
                                     loc.col);
  }

  mlir::LogicalResult declare(llvm::StringRef var, mlir::Value value) {
    if (symbolTable.count(var)) {
      return mlir::failure();
    }

    symbolTable.insert(var, value);
    return mlir::success();
  }

  mlir::toy::FuncOp mlirGen(PrototypeAST &proto) {
    auto location = loc(proto.loc());

    //`llvm::SmallVector<mlir::Type, 4> argTypes(proto.getArgs().size(),
    // getType(VarType{}));` 这是一个 C++ 代码中的表达式，不是一个方法。

    // 这个表达式创建了一个 `llvm::SmallVector`
    // 对象，它是一个动态数组，可以存储最多 4 个 `mlir::Type`
    // 类型的元素。`proto.getArgs().size()` 计算了 `proto` 对象的
    // `args`成员的大小，
    // 并将结果作为第一个参数传递给
    // `SmallVector`的构造函数。`getType(VarType{})` 是一个函数调用， 它返回一个
    // `mlir::Type`对象，并将其作为第二个参数传递给 `SmallVector` 的构造函数。

    // 因此，这个表达式的作用是创建一个包含 `proto.getArgs().size()` 个
    // `mlir::Type` 对象的 `llvm::SmallVector` 对象。
    llvm::SmallVector<mlir::Type, 4> argTypes(proto.getArgs().size(),
                                              getType(VarType{}));

    auto funcType = builder.getFunctionType(argTypes, std::nullopt);
    
    return builder.create<mlir::toy::FuncOp>(location, proto.getName(),
                                             funcType);
  }

  // 生成一个新函数并添加到MLIR module 中
  mlir::toy::FunOp mlirGen(FunctionAST &funcAST) {

    //
    ScopedHashTableScope<llvm::StringRef, mlir::Value> varScope(symbolTable);

    builder.setInsertionPointToEnd(theModule.getBody());
    mlir::toy::FuncOp function = mlirGen(*funcAST.getProto());

    if (!function) {
      return nullptr;
    }

    // 开始创建函数的 body
    mlir::Block &entryBlock = function.front();
    auto protoArgs = funcAST.getProto()->getArgs();

    for (const auto nameValue :
         llvm::zip(protoArgs, entryBlock.getArguments())) {
      if (failed(declare(std::get<0>(nameValue)->getName(),
                         std::get<1>(nameValue))))

        return nullptr;
    }

    // 将构建器中的插入点设置为函数体的开头，它将在整个代码生成过程中用于创建此函数中的操作。

    builder.setInsertionPointToStart(&entryBlock);

    // 退出函数体
    if (mlir::failed(mlirGen(*funcAST.getBody()))) {
      function.erase();
      return nullptr;
    }

    ReturnOp  returnOp;
    if(!entryBlock.empty()){
      returnOp = dyn_cast<ReturnOp>(entryBlock.back());
    }

    if(!returnOp){
      builder.create<ReturnOp>(loc(funcAST.getProto()->loc()));
    }
     else if(returnOp.hasOperand()){
      function.setType(builder.getFunctionType(function.getFunctionType().getInputs(),getType(VarType{})));
     }

     if(funcAST.getProto()->getName() != "main")
       function.setPrivate();

     return function;  

  }

  /// 生成打印表达式
  mlir::LogicalResult mlirGen(PrintExprAST &call) {
    auto arg = mlirGen(*call.getArg());

    if (!arg) {
      return mlir::failure();
    }

    builder.create<PringOp>(loc(call.loc()), arg);
    return mlir::success();
  }

  // 为单个数据生成常量
  mlir::Value mlirGen(NumberExprAST &num) {

    return builder.create<ConstantOp>(loc(num.loc()), num.getValue());
  }

  /// Dispatch codegen for the right expression subclass using RTTI.
  mlir::Value mlirGen(ExprAST &expr) {
    switch (expr.getKind()) {
    case toy::ExprAST::Expr_BinOp:
      return mlirGen(cast<BinaryExprAST>(expr));
    case toy::ExprAST::Expr_Var:
      return mlirGen(cast<VariableExprAST>(expr));
    case toy::ExprAST::Expr_Literal:
      return mlirGen(cast<LiteralExprAST>(expr));
    case toy::ExprAST::Expr_Call:
      return mlirGen(cast<CallExprAST>(expr));
    case toy::ExprAST::Expr_Num:
      return mlirGen(cast<NumberExprAST>(expr));

    default:
      emitError(loc(expr.loc()))
          << "MLIR codegen encountered an unhandled expr kind '"
          << Twine(expr.getKind()) << "'";

      return nullptr;
    }
  }

  ///发出一个文字/常量数组。它将作为附加到“toy.constant”操作的属性中的扁平数据数组发出。
  //有关更多详细信息，请参阅[Attributes]（LangRef.md#Attributes）文档。
  //在 MLIR 中，属性是一种机制，用于在不允许使用变量的地方指定常量数据[...]。
  //它们由一个名称和一个具体的属性值组成。预期的属性集、它们的结构以及它们的解释都取决于它们所附加的上下文。

  // Example, the source level statement:
  ///   var a<2, 3> = [[1, 2, 3], [4, 5, 6]];
  /// will be converted to:
  ///   %0 = "toy.constant"() {value: dense<tensor<2x3xf64>,
  ///     [[1.000000e+00, 2.000000e+00, 3.000000e+00],
  ///      [4.000000e+00, 5.000000e+00, 6.000000e+00]]>} : () -> tensor<2x3xf64>
  ///

  mlir::Value mlirGen(LiteralExprAST &lit){
    auto type = getType(lit.getDims());

    std::vector<double> data;
    data.reserve(std::accumulate(lit.getDims().begin(),lit.getDims().end(),1,
                                std::multiplies<int>()));
    collectData(lit,data);

    mlir::Type elementType = builder.getF64Type();
    auto dataType = mlir::RankedTensorType::get(lit.getDims(),elementType);



    auto dataAttribute = mlir::DenseElementsAttr::get(dataType,llvm::ArrayRef(data));

    return builder.create<ConstantOp>(loc(lit.loc()),type,dataAttribute);


  }


 /// example with this array:
  ///  [[1, 2], [3, 4]]
  /// we will generate:
  ///  [ 1, 2, 3, 4 ]
  void collectData(ExprAST &expr,std::vector<double> &data){
    if(auto *lit = dyn_cast<LiteralExprAST>(&expr)){
        for(auto &value : lit->getValues())
           collectData(*value,data);

        return;   
    }
  }


  //发出一个调用表达式。它为内置的“transpose”函数发出特定的操作。其他标识符被假定为用户定义的函数。
  mlir::Value mlirGen(CallExprAST &call) {
    llvm::StringRef callee = call.getCallee();

    auto location = loc(call.loc());

    SmallVector<mlir::Value,4> operands;

    for(auto &expr : call.getArgs()){
        auto arg = mlirGen(*expr);
        if(!arg)
          return nullptr;

        operands.push_back(arg);  
    }

     //内置调用 transpose 需要特殊处理
    if(callee == "transpose") {
        if(call.getArgs().size() != 1){
             emitError(location, "MLIR codegen encountered an error: toy.transpose "
                            "does not accept multiple arguments");
         return nullptr;
        }

        return builder.create<TransposeOp>(location,operands[0]);
    }

    //其他用户自定义函数
    return builder.create<GenericCallOp>(location,callee,operands);

  }
   


  //这是表达式总一个变量的引用。这个变量已被声明，在符号表中也有值。 否则提示错误 
  mlir::Value mlirGen(VariableExprAST &expr) {
    if(auto variable = symbolTable.lookup(expr.getName()))
       return variable;

    emitError(loc(expr.loc()),"error:unknown variable ' ")
            << expr.getName() << " '" ;

     return nullptr;         
  }

  // 生成return operation
  mlir::LogicalResult mlirGen(ReturnExprAST &ret) {
    auto location = loc(ret.loc());

    mlir::Value expr = nullptr;

    if (ret.getExpr().has_value()) {
      if (!(expr = mlirGen(**ret.getExpr())))
        return mlir::failure();
    }

    builder.create<ReturnOp>(location,
                             expr ? ArrayRef(expr) : ArrayRef<mlir::Value>());
    return mlir::success();
  }
  /// 在处理变量声明时，我们将对构成初始值设定项的表达式进行代码生成，
  // 并在返回值之前将其记录在符号表中。未来的表达式将能够通过符号表查找引用此变量。

  mlir::Value mlirGen(VarDeclExprAST &vardecl) {
    auto *init = vardecl.getInitVal();
    if (!init) {
        emitError(loc(vardecl.loc())),"missing initializer in variable declaration");
        return nullptr;
    }

    mlir::Value value = mlirGen(*init);
    if (!value) {
      return nullptr;
    }
    // We have the initializer value, but in case the variable was declared
    // with specific shape, we emit a "reshape" operation. It will get
    // optimized out later as needed.
    if (!vardecl.getType().shape.empty()) {
      value = builder.create<ReshapeOp>(loc(vardecl.loc()),
                                        getType(vardecl.getType()), value);
    }

    // Register the value in the symbol table.
    if (failed(declare(vardecl.getName(), value)))
      return nullptr;
    return value;
  }

  // 生成表达式
  mlir::LogicalResult mlirGen(ExprASTList &blockAST) {
    ScopedHashTableScope<StringRef, mlir::Value> varScope(symbolTable);

    for (auto &expr : blockAST) {

      if (auto *vardecl = dyn_cast<VarDeclExprAST>(expr.get())) {
        if (!mlirGen(*vardecl)) {
          return mlir::failure();
        }
        continue;
      }

      if (auto *ret = dyn_cast<ReturnExprAST>(expr.get()))
        return mlirGen(*ret);

      if (auto *print = dyn_cast<PrintExprAST>(expr.get()))
        if (mlir::failed(mlirGen(*print)))
          return mlir::success();

        continue;

        // Generic expression dispatch codegen.
      if (!mlirGen(*expr))
        return mlir::failure();   
    }

    return mlir::success();

  }

  // Build a tensor type from a list of shape dimensions.
  mlir::Type getType(ArrayRef<int64_t> shape) {
    if (shape.empty())
      return mlir::UnrankedTensorType::get(builder.getF64Type());

    return mlir::RankedTensorType::get(shape, builder.getF64Type());
  }

  // 从toy ast 变量类型构建mlir type
  mlir::Type getType(const VarType &type) { return getType(type.shape); }

public:
  //// 构造函数接受一个 MLIRContext 引用作为参数，并将其存储在 builder
  /// 成员变量中。
  MLIRGenImpl(mlir::MLIRContext &context) : builder(&context) {}

  // toy module 的AST 转换成MLIR的 module operation
  mlir::ModuleOp mlirGen(ModuleAST &moduleAST) {

    theModule = mlir::ModuleOp::create(builder.getUnknownLoc());

    for (FunctionAST &f : moduleAST)
      mlirGen(f);

    if (failed(mlir::verify(theModule))) {
      theModule.emitError("module verification error");
      return nullptr;
    }

    return theModule;
  }


};


} // namespace

namespace toy{

    // The public API for codegen.
    mlir::OwningOpRef<mlir::ModuleOp> mlirGen(mlir::MLIRContext &context,ModuleAST &ModuleAST){
       
        return MLIRGenImpl(context).mlirGen(ModuleAST);

    }
}// namespace toy

