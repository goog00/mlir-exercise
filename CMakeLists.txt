cmake_minimum_required(VERSION 3.20)
project(mlir-exercise VERSION 0.0.0) # 创建一个名为 mlir-toy 的项目，并设置版本号为 0.0.0


#防止cmake 生成一堆warning
cmake_policy(SET CMP0116 NEW)



set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # 生成 compile_commands.json 文件，便于代码高亮
set(CMAKE_CXX_STANDARD 17) # 设置 C++ 的语言标准为 C++17
set(CMAKE_CXX_STANDARDREQUIRED YES) # 要求所有源文件都必须符合 C++17 标准

find_package(MLIR REQUIRED CONFIG) # 查找 MLIR 库并将其添加到项目中
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}") # 添加 MLIR 的 CMake 模块路径到搜索路径中
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}") # 添加 LLVM 的 CMake 模块路径到搜索路径中

include(TableGen) # 包含 TableGen 模块
include(AddLLVM) # 包含 AddLLVM 模块
include(AddMLIR) # 包含 AddMLIR 模块
include(HandleLLVMOptions) # 包含 HandleLLVMOptions 模块



include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS}) # 添加 LLVM 和 MLIR 的头文件路径到搜索路径中
 

# add_executable(mlir-toy main.cpp) # 添加一个名为 mlir-toy 的可执行文件，它的源文件是 main.cpp


add_executable(ex3-dialect ex3-dialect/tools/toy-opt/toy-opt.cpp)
target_link_libraries(
  ex3-dialect
  MLIRIR
  MLIRParser
  MLIRFuncDialect
  MLIRArithDialect
)
 

# add_subdirectory(ex3-dialect)