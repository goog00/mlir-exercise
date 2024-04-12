#ifndef MLIR_TUTORIAL_TOY_DIALECT_H_
#define MLIR_TUTORIAL_TOY_DIALECT_H_

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "toy/Dialect.h.inc"

/// Include the auto-generated header file containing the declarations of the
/// toy operations.
#define GET_OP_CLASSES
#include "toy/Ops.h.inc"


namespace mlir {
namespace toy{

class StructType : public mlir::Type::TypeBase<StructType,mlir::Type,detail::StructTypeStorage>{


public:
  using Base::Base;


  static StructType get(llvm::ArrayRef<mlir::Type> elementTypes);

  llvm::ArrayRef<mlir::Type> getElementTypes();

  size_t getNumElementTypes() {
    return getElementTypes().size();
  }

  static constexpr StringLiteral name = "toy.struct";

};
}    
}

#endif