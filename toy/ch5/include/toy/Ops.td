//===----------------------------------------------------------------------===//
//
// Defines the operations of the Toy dialect.
//
//===----------------------------------------------------------------------===//


#ifndef TOY_OPS
#define TOY_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"


//定义toy dialect
def Toy_Dialect : Dialect {
    let name = "toy";
    let cppNamespace = "::mlir::toy";
}

//toy 方言的操作基础类。这个操作继承基础Op类 提供：
//1.操作的父方言 
//2.操作的助记符 或者没有方言前缀的名字
//3.操作的特征列表
class Toy_Op<string mnemonic,list<Trait> traits = []>:
Op<Toy_Dialect,mnemonic,traits>;


//===----------------------------------------------------------------------===//
// Toy Operations
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

//我们定义一个从基础类“Toy_Op"继承来的toy Operation
//这里我提供助记符和操作的特征
//常量运算被标记为Pure作为一个纯操作，如果操作死亡可以删除该操作
def ConstantOp : Toy_Op<"constant",[Pure]> {
  let summary = "constant";
  let description = [{
    常量操作把文本转为SSA值，随附数据作为属性添加到操作 例如：

    ```mlir 
        %0 = toy.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]>
                        : tensor<2x3xf64>
    ```                     
  }];

  //常量运算将属性作为唯一的输入
  let arguments = (ins F64ElementsAttr:$value);

  //常量运算返回单个TensorType值
  let results = (outs F64Tensor);

  //表明运算有自定义的解析和打印方法
  let hasCustomAssemblyFormat = 1;

  //为常量运算添加自定义构建方法
  //这些方法填充MLIR 用来创建的运算的‘state' 例如：builder.create<ConstantOp>(..)
  let builders = [
    //使用给定的常量 tensor 值构建一个常量
    OpBuilder<(ins "DenseElementsAttr":$value),[{
      build($_builder,$_state,value.getType(),value);
        }]>,
    //使用给定的浮点数据构建一个常量
    OpBuilder<(ins "double":$value)>
  ];


  let hasVerifier = 1;

}

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//

def CastOp : Toy_Op<"cast",[
        DeclareOpInterfaceMethods<CastOpInterface>,
        DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
        Pure,
        SameOperandsAndResultShape
]> {
    let summary = "shape cast operation";
    let description = [{
    The "cast" operation converts a tensor from one type to an equivalent type
    without changing any data elements. The source and destination types must
    both be tensor types with the same element type. If both are ranked, then
    shape is required to match. The operation is invalid if converting to a
    mismatching constant dimension.
  }];

  let arguments = (ins F64Tensor:$input);
  let results = (outs F64Tensor:$output);

  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($output)";

}


//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//


def FunOp : Toy_Op<"func",[
    FunctionOpInterface,IsolatedFromAbove
]>{
 
 let summary = " user defined function operation";
 let description = [{
    "toy.fun" 操作表示用户定义的函数。
    这些是包含玩具计算的可调用SSA区域操作。

 ```mlir
    toy.func @main() {
      %0 = toy.constant dense<5.500000e+00> : tensor<f64>
      %1 = toy.reshape(%0 : tensor<f64>) to tensor<2x2xf64>
      toy.print %1 : tensor<2x2xf64>
      toy.return
    }
    ```
 }];
 
 let arguments = (ins 
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
    );

    let regions = (region AnyRegioon:$body);

    let builders = [OpBuilder<(ins 
    "StringRef":$name,"FunctionType":$type,
    CArg<"ArrayRef<NamedAttribute>","{}">:$attrs)
    >];

    let extraClassDeclaration = [{

    //===------------------------------------------------------------------===//
    // FunctionOpInterface Methods
    //===------------------------------------------------------------------===//
     //返回这个函数的参数类型

     ArrayRef<Type> getArgumentTypes(){return getFunctionType().getInputs();}
     //返回这个函数的结果类型
     ArrayRef<Type> getResultTypes(){return getFunctionType().getResults();}

     Region *getCallableRegion(){return &getBody();}


    }];

    let hasCustomAssemblyFormat = 1;
    let skipDefaultBuilders = 1;


}


//===----------------------------------------------------------------------===//
// GenericCallOp  泛型调用
//===----------------------------------------------------------------------===//

def GenericCallOp : Toy_Op<"generic_call",[DeclareOpInterfaceMethods<CallOpInterface>]> {
    let summary = "generic call operation";

    let description = [{
        //泛型调用表示对用户定义的函数的调用，该函数需要针对其参数的形状进行专门化。被调用方名称通过属性作为符号引用附加。参数列表必须与被调用者所需的参数匹配。
         ```mlir
         %4 = toy.generic_call @my_func(%1, %3)
           : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
        ```

    }];

    //泛型调用操作将符号引用属性作为被调用者，并为调用输入。
    let arguments = (ins FlatSymbolRefAttr:$callee,Variadic<F64Tensor>:$inputs);

    let results = (outs F64Tensor);

    let assemblyFormat = [{
           $callee `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
    }];

    let builders = [
      OpBuilder<(ins "StringRef":$callee, "ArrayRef<Value>":$arguments)>
    ];

}


//===----------------------------------------------------------------------===//
// MulOp
//===----------------------------------------------------------------------===//


def MulOp : Toy_Op<"mul"，[DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
    let summary = "element-wise multiplication operation";

    let description = [{
         The "mul" operation performs element-wise multiplication between two
    tensors. The shapes of the tensor operands are expected to match.
    }];
    
    let arguments = (ins F64Tensor:$lhs,F64Tensor:$rhs);

    let results = (outs F64Tensor);

    let hasCustomAssemblyFormat = 1;

    let builders = [
        OpBuilder<ins "Value":$lhs,"Value":$rhs>
    ];

}


//===----------------------------------------------------------------------===//
// PrintOp
//===----------------------------------------------------------------------===//

def PrintOp : Toy_Op<"print"> {
    let summary = "print operation" ;

    let description = [{
        print 是内嵌的操作，用来打印指定的输入Tensor，不返回结果

    }];

    let arguments = (ins F64Tensor:$input);

    let assemblyFormat = "$input attr-dict `:` type($input)";

}


//===----------------------------------------------------------------------===//
// ReshapeOp
//===----------------------------------------------------------------------===//



def ReshapeOp : Toy_Op<"reshape"> {
  let summary = "tensor reshape operation";
  let description = [{
    Reshape operation is transforming its input tensor into a new tensor with
    the same number of elements but different shapes. For example:

    ```mlir
       %0 = toy.reshape (%arg1 : tensor<10xf64>) to tensor<5x2xf64>
    ```
  }];

  let arguments = (ins F64Tensor:$input);

  // We expect that the reshape operation returns a statically shaped tensor.
  let results = (outs StaticShapeTensorOf<[F64]>);

  let assemblyFormat = [{
    `(` $input `:` type($input) `)` attr-dict `to` type(results)
  }];
}


//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

def ReturnOp : Toy_Op<"return", [Pure, HasParent<"FuncOp">,
                                 Terminator]> {
  let summary = "return operation";
  let description = [{
    The "return" operation represents a return operation within a function.
    The operation takes an optional tensor operand and produces no results.
    The operand type must match the signature of the function that contains
    the operation. For example:

    ```mlir
      toy.func @foo() -> tensor<2xf64> {
        ...
        toy.return %0 : tensor<2xf64>
      }
    ```
  }];

  // The return operation takes an optional input operand to return. This
  // value must match the return type of the enclosing function.
  let arguments = (ins Variadic<F64Tensor>:$input);

  // The return operation only emits the input in the format if it is present.
  let assemblyFormat = "($input^ `:` type($input))? attr-dict ";

  // Allow building a ReturnOp with no return operand.
  let builders = [
    OpBuilder<(ins), [{ build($_builder, $_state, std::nullopt); }]>
  ];

  // Provide extra utility definitions on the c++ operation class definition.
  let extraClassDeclaration = [{
    bool hasOperand() { return getNumOperands() != 0; }
  }];

  // Invoke a static verify method to verify this return operation.
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//


def TransposeOp : Toy_Op<"transpose",[NoSideEffect]>{
    let summary = "transpose operation";

    let arguments = (ins F64Tensor:$input);

    let results = (outs F64Tensor);

    let assemblyFormat = [{
         `(` $input `:` type($input) `)` attr-dict `to` type(results)
    }];

    let builders = [
        OpBuilder<(ins "Value":$input)>
    ];

    // Invoke a static verify method to verify this transpose operation.
    let hasVerifier = 1;
}

#endif //TOY_OPS