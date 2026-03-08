# ==============================================================================
# 基础配置：CMake版本、项目名称、模块路径
# ==============================================================================
# 声明CMake所需的最低和最高版本范围（兼容3.10到3.18）
cmake_minimum_required(VERSION 3.10...3.18)

# 定义项目名称为 "basalt"
project(basalt VERSION 0.0.1)

# 定义自定义变量（构建时间）
string(TIMESTAMP BUILD_TIME "%Y-%m-%d %H:%M:%S" UTC)

# 获取Git提交哈希（可选）
execute_process(
    COMMAND git rev-parse --short HEAD
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_COMMIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# 创建版本文件
# - CMAKE_CURRENT_SOURCE_DIR 当前处理的 CMakeLists.txt 文件所在的源码目录
# - CMAKE_CURRENT_BINARY_DIR 当前处理的 CMakeLists.txt 文件对应的编译输出目录
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/VERSION.in"  # 输入模板
  "${CMAKE_CURRENT_BINARY_DIR}/VERSION"     # 输出文件
  @ONLY                                     # 仅替换@xxx@，如果不加 @ONLY：CMake 会同时替换 @变量名@ 和 ${变量名} 两种格式的占位符；
)

# 将项目内的cmake_modules目录添加到CMake的模块搜索路径，用于寻找自定义的Find脚本
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules/" ${CMAKE_MODULE_PATH})

# ==============================================================================
# Debian打包相关配置（生成.deb安装包）
# ==============================================================================
# 将项目名转换为小写，通常用于包名或文件名
string(TOLOWER "${PROJECT_NAME}" PROJECT_NAME_LOWERCASE)

# 查找Debian/Ubuntu系统的dpkg程序，用于获取系统架构
find_program(DPKG_PROGRAM dpkg DOC "dpkg program of Debian-based systems")
if(DPKG_PROGRAM)
  # 获取系统架构（如amd64、arm64）
  execute_process(
    COMMAND ${DPKG_PROGRAM} --print-architecture        # 执行命令获取系统架构：dpkg --print-architecture
    OUTPUT_VARIABLE CPACK_DEBIAN_PACKAGE_ARCHITECTURE   # 将命令输出存储在变量 CPACK_DEBIAN_PACKAGE_ARCHITECTURE 中
    OUTPUT_STRIP_TRAILING_WHITESPACE                    # 去除输出末尾的换行符和空白字符
  )
endif(DPKG_PROGRAM)

# 查找lsb_release程序（获取系统版本）
find_program(LSB_RELEASE_PROGRAM lsb_release DOC "lsb_release program of Debian-based systems")
if(LSB_RELEASE_PROGRAM)
  # 获取系统版本号（如22.04、20.04）
  execute_process(
    COMMAND ${LSB_RELEASE_PROGRAM} -rs    # 相当于在终端运行：lsb_release -rs （-r: 显示发行版本号，-s: 简洁输出（只输出版本号，不输出标签））
    OUTPUT_VARIABLE LSB_RELEASE_ID_SHORT  # 将命令输出存储在变量 LSB_RELEASE_ID_SHORT 中
    OUTPUT_STRIP_TRAILING_WHITESPACE      # 去除输出末尾的换行符和空白字符
  )

  # 根据不同Ubuntu版本设置Debian包依赖（不同版本库版本不同）
  if(${LSB_RELEASE_ID_SHORT} EQUAL "22.04")
    set(DEBIAN_DEPENDS "libtbb12, liblz4-1, libbz2-1.0, libfmt8, libboost-filesystem1.74.0, libboost-date-time1.74.0, libboost-program-options1.74.0, libboost-regex1.74.0, libopencv-dev, libglew2.2, libjpeg8, libpng16-16, librealsense2, librealsense2-dkms, librealsense2-gl, librealsense2-utils")

  elseif(${LSB_RELEASE_ID_SHORT} EQUAL "20.04")
    set(DEBIAN_DEPENDS "libtbb2, liblz4-1, libbz2-1.0, libboost-filesystem1.71.0, libboost-date-time1.71.0, libboost-program-options1.71.0, libboost-regex1.71.0, libopencv-dev, libglew2.1, libjpeg8, libpng16-16, librealsense2, librealsense2-dkms, librealsense2-gl, librealsense2-utils")

  elseif(${LSB_RELEASE_ID_SHORT} EQUAL "18.04")
    set(DEBIAN_DEPENDS "libtbb2, liblz4-1, libbz2-1.0, libboost-filesystem1.65.1, libboost-date-time1.65.1, libboost-program-options1.65.1, libboost-regex1.65.1, libopencv-dev, libglew2.0, libjpeg8, libpng16-16, librealsense2, librealsense2-dkms, librealsense2-gl, librealsense2-utils")

  elseif(${LSB_RELEASE_ID_SHORT} EQUAL "16.04")
    set(DEBIAN_DEPENDS "libtbb2, liblz4-1, libbz2-1.0, libboost-filesystem1.58.0, libboost-date-time1.58.0, libboost-program-options1.58.0, libboost-regex1.58.0, libopencv-dev, libglew1.13, libjpeg8, libpng12-0, libstdc++6, librealsense2, librealsense2-dkms, librealsense2-gl, librealsense2-utils")
  endif()

endif(LSB_RELEASE_PROGRAM)

# 生成时间戳用于版本号（格式：年月日时分）
string(TIMESTAMP PROJECT_VERSION_REVISION "%Y%m%d%H%M")

# 设置 CPack 打包配置（用于生成 Debian 包）
set(CPACK_GENERATOR "DEB")                                                    # 生成Debian包
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Vladyslav Usenko <vlad.usenko@tum.de>")  # 维护者信息
set(CPACK_PACKAGE_VERSION_MAJOR "0")                                          # 主版本号
set(CPACK_PACKAGE_VERSION_MINOR "1")                                          # 次版本号
set(CPACK_PACKAGE_VERSION_PATCH "0-${PROJECT_VERSION_REVISION}~${LSB_RELEASE_ID_SHORT}")    # 补丁版本号：包含时间戳和系统版本
set(CPACK_DEBIAN_PACKAGE_DEPENDS ${DEBIAN_DEPENDS})                           # 包依赖

# 生成最终包文件名：项目名_版本_架构.deb
set(CPACK_PACKAGE_FILE_NAME "${PROJECT_NAME_LOWERCASE}_${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH}_${CPACK_DEBIAN_PACKAGE_ARCHITECTURE}")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")        # 许可证文件
include(CPack)

# ==============================================================================
# 编译优化：CCache（编译缓存，加速重复构建）
# ==============================================================================
# Configure CCache if available
if (NOT CMAKE_C_COMPILER_LAUNCHER AND NOT CMAKE_CXX_COMPILER_LAUNCHER)
  find_program(CCACHE_PROGRAM ccache)
  if(CCACHE_PROGRAM)
    message(STATUS "Found ccache: ${CCACHE_PROGRAM}")
    # 设置CCache作为C/C++编译器启动器
    set(CMAKE_C_COMPILER_LAUNCHER   ${CCACHE_PROGRAM})
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
  else()
    message(STATUS "Didn't find ccache")
  endif()
else()
  message(STATUS "Compiler launcher already set. Not configuring ccache.")
  message(STATUS "CMAKE_C_COMPILER_LAUNCHER: ${CMAKE_C_COMPILER_LAUNCHER}")
  message(STATUS "CMAKE_CXX_COMPILER_LAUNCHER: ${CMAKE_CXX_COMPILER_LAUNCHER}")
endif()

# ==============================================================================
# 编译类型与架构优化
# ==============================================================================
# 如果没有指定构建类型，默认为 Release
if( NOT CMAKE_BUILD_TYPE )
  set( CMAKE_BUILD_TYPE Release)
endif()

# 默认CPU架构优化为native（使用当前CPU的所有特性）
if(NOT CXX_MARCH)
  set(CXX_MARCH native)
endif()

# Apple M1/M2 (arm64) 禁用-march（该架构不支持此参数）
IF(NOT APPLE OR NOT CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
  set(BASALT_MARCH_FLAGS "-march=${CXX_MARCH}")
ELSE()
  message(STATUS "Running on Apple ${CMAKE_SYSTEM_PROCESSOR}. Disabled -march flag.")
ENDIF()

# ==============================================================================
# C++标准与编译选项
# ==============================================================================
# 设置 C++ 标准为 17
set(CMAKE_CXX_STANDARD 17)                  # 指定C++17标准
set(CMAKE_CXX_STANDARD_REQUIRED ON)         # 强制要求C++17（不降级）
set(CMAKE_CXX_EXTENSIONS OFF)               # 禁用编译器特有扩展（如GNU扩展）
set(CMAKE_POSITION_INDEPENDENT_CODE ON)     # 生成位置无关代码（PIC，用于动态库）


# Flags used for CHECK_CXX_SOURCE_COMPILES
# 用于CHECK_CXX_SOURCE_COMPILES的临时标志（禁用警告转错误）
set(CMAKE_REQUIRED_FLAGS "-Wno-error")  # -Wno-error：虽然仍有警告，但不会导致编译失败，检查能正常进行


# save flags passed by user
# 保存用户传入的编译标志（避免被覆盖）
set(BASALT_PASSED_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

# 设置不同构建类型的编译标志
set(CMAKE_CXX_FLAGS_DEBUG  "-O0 -g -DEIGEN_INITIALIZE_MATRICES_BY_NAN")           # cmake default: "-g"                 Debug：无优化+调试信息+Eigen矩阵初始化为NaN（方便调试）
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -DEIGEN_INITIALIZE_MATRICES_BY_NAN")   # cmake default: "-O2 -g -DNDEBUG"    带调试信息的Release：O3优化+调试信息
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")                                       # cmake default: "-O3 -DNDEBUG"       Release：O3优化+禁用调试
set(CMAKE_CXX_FLAGS_CIDEBUG  "-O0 -DEIGEN_INITIALIZE_MATRICES_BY_NAN")            # CI version with no debug symbols    CI Debug：无优化+NaN初始化（无调试符号）
set(CMAKE_CXX_FLAGS_CIRELWITHDEBINFO "-O3 -DEIGEN_INITIALIZE_MATRICES_BY_NAN")    # CI version with no debug symbols    CI Release：O3优化+NaN初始化

# base set of compile flags
# 基础编译标志
# 解释：
# -Wall: 开启所有基础警告
# -Wextra: 开启额外警告
# -Werror: 将所有警告视为错误（强制修复）
# -Wno-error=unused-parameter: 未使用参数不视为错误（兼容部分代码）
# -ftemplate-backtrace-limit=0: 禁用模板回溯限制（方便调试模板错误）
set(BASALT_CXX_FLAGS "-Wall -Wextra -Werror -Wno-error=unused-parameter -ftemplate-backtrace-limit=0")


# ==============================================================================
# 编译器特有标志（Clang/GCC/AppleClang）
# ==============================================================================
# Clang: 由 Apple​ 主导开发，是 LLVM​ 项目的前端部分。
# GCC: GNU 项目的核心组件，是自由软件基金会的代表作品，是许多开源系统（如Linux）的标准编译器。
# clang-specific compile flags
# clang 特定的编译标志
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
  # Clang/AppleClang特有标志
  # 解释：
  # -Wno-exceptions: 禁用异常相关警告
  # -fcolor-diagnostics: 彩色编译输出（方便阅读）
  # -frelaxed-template-template-args: 放宽模板参数检查
  # -Wno-error=deprecated-declarations: 废弃声明不视为错误
  set(BASALT_CXX_FLAGS "${BASALT_CXX_FLAGS} -Wno-exceptions -fcolor-diagnostics -frelaxed-template-template-args -Wno-error=deprecated-declarations")

  if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 10)
      # These are disabled to avoid lot's of warnings in Eigen code with clang 10
      # Clang 10+ 禁用Eigen相关警告
      set(BASALT_CXX_FLAGS "${BASALT_CXX_FLAGS} -Wno-misleading-indentation -Wno-error=deprecated-copy")
  endif()

  if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 15)
      # These are disabled to avoid lot's of warnings in Eigen code with clang 15
      # Clang 15+ 禁用未使用但设置的变量警告
      set(BASALT_CXX_FLAGS "${BASALT_CXX_FLAGS} -Wno-error=unused-but-set-variable")
  endif()

  #   - Added TBB_USE_GLIBCXX_VERSION macro to specify the version of GNU
  #     libstdc++ when it cannot be properly recognized, e.g. when used
  #     with Clang on Linux* OS. Adopted from https://github.com/wjakob/tbb
  # Linux+Clang：设置TBB的GLIBCXX版本（解决TBB编译问题）
  if(NOT TBB_USE_GLIBCXX_VERSION AND UNIX AND NOT APPLE)
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
      string(REPLACE "." "0" TBB_USE_GLIBCXX_VERSION ${CMAKE_CXX_COMPILER_VERSION})
    endif()
    add_definitions(-DTBB_USE_GLIBCXX_VERSION=${TBB_USE_GLIBCXX_VERSION})
  endif()
else()  # GUN
  # GCC特有标志
  # 解释：
  # -Wno-error=maybe-uninitialized: 可能未初始化变量不视为错误
  # -Wno-error=implicit-fallthrough: switch隐式贯穿不视为错误
  set(BASALT_CXX_FLAGS "${BASALT_CXX_FLAGS} -Wno-error=maybe-uninitialized -Wno-error=implicit-fallthrough")

  # GCC < 10 禁用未初始化变量警告
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 10.0)
      set(BASALT_CXX_FLAGS "${BASALT_CXX_FLAGS} -Wno-error=uninitialized")
  endif()

  # GCC 9+ 禁用废弃拷贝警告（Eigen相关）
  if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 9.0)
      # These are disabled to avoid lot's of warnings in Eigen code with gcc-9
      set(BASALT_CXX_FLAGS "${BASALT_CXX_FLAGS} -Wno-error=deprecated-copy")
  endif()

  # GCC 10+ 禁用括号/数组越界警告（Pangolin相关）
  if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 10.0)
    # These are disabled due to warnings in pangolin
    #set(BASALT_CXX_FLAGS "${BASALT_CXX_FLAGS} -Wno-error=parentheses")
    set(BASALT_CXX_FLAGS "${BASALT_CXX_FLAGS} -Wno-parentheses -Wno-error=array-bounds")
  endif()
endif()


# ==============================================================================
# 系统特有配置（macOS/Linux）
# ==============================================================================
# Set platform / compiler specific compile flags and checks
# 平台特定的设置
if(APPLE)
  # Need to investigate how to reliably detect and use OpenMP on macOS...
#  set(USE_OPENMP_DEFAULT OFF)

  # Among others, setting CMAKE_FIND_FRAMEWORK to LAST fixed issues
  # with installed Mono that contains old headers (libpng, ...).
  # See: https://github.com/openMVG/openMVG/issues/1349#issuecomment-401492811
  # macOS：设置框架查找优先级（解决Mono库冲突）
  set(CMAKE_FIND_FRAMEWORK LAST)

  # macOS < 10.14 (Mojave)：使用brew安装的LLVM libc++
  if(CMAKE_SYSTEM_VERSION VERSION_LESS 19.0.0)
      # use brewed llvm's libc++
      # 使用 brew 安装的 llvm 的 libc++
      include_directories("/usr/local/opt/llvm/include/c++/v1")
      link_directories("/usr/local/opt/llvm/lib")
      add_compile_options("-nostdinc++")
      #set(STD_CXX_FS c++fs)

      # Workaround for cmake not to filter the manually added standard include path
      # See: https://gitlab.kitware.com/cmake/cmake/issues/19227#note_669894
      # 移除标准包含路径，避免 CMake 过滤手动添加的路径
      list(REMOVE_ITEM CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES "/usr/local/opt/llvm/include/c++/v1")
  endif()

  # 编译器类型提示
  if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    message(STATUS "Detected macOS with non-Apple clang")

  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    message(STATUS "Detected macOS with Apple clang")
    # Apple clang on macOS < 10.14 Mojave is too old
    if(CMAKE_SYSTEM_VERSION VERSION_LESS 18.0.0)
      message(WARNING "Detected Darwin version ${CMAKE_SYSTEM_VERSION}, which is earlier than macos 10.14 Mojave. Apple clang is too old and not supported. Use clang from homebrew.")
    endif()

  else()
    message(WARNING "Detected macOS with unsupported compiler ${CMAKE_CXX_COMPILER_ID}")
  endif()

elseif(UNIX)
#  set(USE_OPENMP_DEFAULT ON)

  # assume libstdc++
  # Linux：设置stdc++fs（C++17文件系统库）
  set(STD_CXX_FS stdc++fs)

  # 编译器类型提示
  if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    message(STATUS "Detected Linux with clang.")

    # Note: Whatever the issue was, this seems to not be an issue any more, at least with more recent clang and Ubuntu versions (tested clang-12 and Ubuntu 18.04)
    #message(WARNING "Clang on Linux is currently not fully supported. You'll likely need to get a recent version of TBB.")
    #set(LINUX_CLANG 1)

  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    message(STATUS "Detected Linux with gcc.")

  else()
    message(WARNING "Detected Linux with unsupported compiler ${CMAKE_CXX_COMPILER_ID}")
  endif()

else()
  # 非Linux/macOS：警告不支持
  message(WARNING "Only Linux and macOS are currently supported")
endif()


# OpenMP option and compile flags
#
# Note: OpenMP and TBB don't mix well, so we disable Eigen's parallelization.
# It's trying to paralellize matrix products during SC, which we run in a parallel_reduce using TBB.
# Turns out using OpenMP can slow down the computby factor 10-100x! So for now we discable it completely.
# One place where Eigen's parallelization could still have been useful is the CG solver in the mapper.
# We could in the future investiagte other implementations (paralellized with TBB) or selectively enabling
# Eigen's parallelization just for CG, setting number of threads to 1 everywhere else.
# Another way to ensure Eigen doesn't use OpenMP regardless of how it was built is setting the environment
# variable OMP_NUM_THREADS=1 beofre running the application.
#
# See: https://eigen.tuxfamily.org/dox/TopicMultiThreading.html
#
# If we enable BLAS / LAPACK either directly or via thirdparty libs like ceres,
# make sure to disable OpenMP for the linked BLAS library. In particular on Ubuntu it seems OpenBLAS is often installed,
# and it can have similar issues in multithreaded applications if it's own parallelization with OpenMP is enabled.
# You can set the environment varaibles OPENBLAS_NUM_THREADS=1 or OMP_NUM_THREADS=1. This is also mentioned in the ceres
# installation documentation.
#
# See also: https://github.com/xianyi/OpenBLAS/wiki/faq#multi-threaded
#
# Set EIGEN_DONT_PARALLELIZE to be sure it doesn't use openmp,
# just in case some dependency enables openmp without us noticing.
# ==============================================================================
# OpenMP禁用（核心优化：避免与TBB冲突）
# ==============================================================================
# 关键说明：
# - OpenMP与TBB（Intel线程库）混用会导致性能下降10-100倍
# - 强制禁用Eigen的并行化（即使依赖库启用了OpenMP）
set(BASALT_CXX_FLAGS "${BASALT_CXX_FLAGS} -DEIGEN_DONT_PARALLELIZE")

#option(USE_OPENMP "Use OpenMP (e.g. for parallel computation in Eigen)" ${USE_OPENMP_DEFAULT})
#if(USE_OPENMP)
#  message(STATUS "OpenMP Enabled")
#  set(BASALT_CXX_FLAGS "${BASALT_CXX_FLAGS} -fopenmp")
#else()
#  message(STATUS "OpenMP Disabled")
#endif()

# Make specialization for double / float optional.
# Disabling them can be useful for better compile times during development.
# ==============================================================================
# 模板实例化配置（可选双精度/单精度）
# ==============================================================================
# 选项：是否实例化double类型模板（默认开启）
option(BASALT_INSTANTIATIONS_DOUBLE "Instatiate templates for Scalar=double." ON)
# 选项：是否实例化float类型模板（默认开启）
option(BASALT_INSTANTIATIONS_FLOAT "Instatiate templates for Scalar=float." ON)

# 根据选项添加编译定义
if(BASALT_INSTANTIATIONS_DOUBLE)
  list(APPEND BASALT_COMPILE_DEFINITIONS BASALT_INSTANTIATIONS_DOUBLE)
endif()
if(BASALT_INSTANTIATIONS_FLOAT)
  list(APPEND BASALT_COMPILE_DEFINITIONS BASALT_INSTANTIATIONS_FLOAT)
endif()


# setup combined compiler flags
# ==============================================================================
# 合并编译标志
# ==============================================================================
set(CMAKE_CXX_FLAGS "${BASALT_CXX_FLAGS} ${BASALT_MARCH_FLAGS} ${BASALT_PASSED_CXX_FLAGS}")


# ==============================================================================
# 依赖库查找
# ==============================================================================
# Eigen库路径配置：如果用户没有定义EIGEN_ROOT环境变量或缓存变量，则设置默认路径指向thirdparty中的Eigen库
if(NOT EIGEN_ROOT)
  set(EIGEN_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/basalt-headers/thirdparty/eigen")
endif()

# 1.查找 Eigen3 库（精确版本3.4.0）
set(EIGEN_INCLUDE_DIR_HINTS ${EIGEN_ROOT})
find_package(Eigen3 3.4.0 EXACT REQUIRED MODULE)
message(STATUS "Found Eigen headers in: ${EIGEN3_INCLUDE_DIR}")
# 检查Eigen路径是否符合预期（非必须，仅警告）
if(NOT EIGEN3_INCLUDE_DIR MATCHES "^${EIGEN_ROOT}")
  message(WARNING "Found Eigen headers are outside of specified EIGEN_ROOT '${EIGEN_ROOT}'")
endif()

# 2.查找 TBB 库（Intel线程库）
find_package(TBB REQUIRED)
message(STATUS "Found TBB ${TBB_VERSION_MAJOR}.${TBB_VERSION_MINOR} (interface version ${TBB_INTERFACE_VERSION}) headers in: ${TBB_INCLUDE_DIRS}")
if (TBB_INTERFACE_VERSION LESS 11004)
  # enable global_control header for earlier TBB versions (Ubuntu 16.04, 18.04)
  # TBB < 11004：启用全局控制头文件
  add_definitions(-DTBB_PREVIEW_GLOBAL_CONTROL)
endif()

# NOTE: not specifying version, since 2, 3 or 4 is fine
# 3.查找 OpenCV（核心、图像处理、标定、高GUI）
find_package(OpenCV REQUIRED COMPONENTS core imgproc calib3d highgui)
message(STATUS "Found OpenCV ${OpenCV_VERSION} headers in: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "Found OpenCV_LIBS: ${OpenCV_LIBS}")

# Add our own custom scoped opencv target since none is provided by OpenCV itself
# 创建自定义OpenCV目标（命名空间隔离）
add_library(basalt::opencv INTERFACE IMPORTED)
set_property(TARGET basalt::opencv PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${OpenCV_INCLUDE_DIRS})
set_property(TARGET basalt::opencv PROPERTY INTERFACE_LINK_LIBRARIES ${OpenCV_LIBS})

# 4.查找 fmt 库（格式化库）
find_package(fmt REQUIRED)
message(STATUS "Found {fmt} ${fmt_VERSION} in: ${fmt_DIR}")


# ==============================================================================
# 第三方库构建
# ==============================================================================
# 定义Pangolin目标列表（后续链接使用）
set(BASALT_PANGO_TARGETS pango_display pango_image pango_windowing pango_plot pango_video)
# 构建第三方库（如Pangolin、apriltag、opengv等）
add_subdirectory(thirdparty)

# ==============================================================================
# 自定义第三方目标（命名空间隔离）
# ==============================================================================
# custom scoped cli11 target
# CLI11（命令行解析）
add_library(basalt::cli11 INTERFACE IMPORTED)
set_property(TARGET basalt::cli11 PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/CLI11/include)

# custom scoped magic_enum target
# magic_enum（枚举反射）
add_library(basalt::magic_enum INTERFACE IMPORTED)
set_property(TARGET basalt::magic_enum PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/magic_enum/include)


# ==============================================================================
# 核心库：basalt共享库
# ==============================================================================
# main library
# 创建主库（共享库）
add_library(basalt SHARED)

# List all header and source files with target_sources.
# This ensures, e.g., that QtCreator >= 6.0 properly recognize these headers to belong to the project.
# To support cmake < 3.13, use absolute paths (see: https://crascit.com/2016/01/31/enhanced-source-file-handling-with-target_sources/)
# 添加源文件（PUBLIC：头文件对外暴露，PRIVATE：源文件仅内部使用）
target_sources(basalt
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/calibration/aprilgrid.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/calibration/calibration_helper.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/calibration/cam_calib.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/calibration/cam_imu_calib.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/calibration/vignette.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/device/rs_t265.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/hash_bow/hash_bow.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/io/dataset_io.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/io/dataset_io_euroc.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/io/dataset_io_kitti.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/io/dataset_io_rosbag.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/io/dataset_io_uzh.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/io/marg_data_io.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/linearization/block_diagonal.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/linearization/imu_block.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/linearization/landmark_block.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/linearization/landmark_block_abs_dynamic.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/linearization/linearization_abs_qr.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/linearization/linearization_abs_sc.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/linearization/linearization_base.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/linearization/linearization_rel_sc.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/optical_flow/frame_to_frame_optical_flow.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/optical_flow/multiscale_frame_to_frame_optical_flow.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/optical_flow/optical_flow.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/optical_flow/patch.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/optical_flow/patch_optical_flow.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/optical_flow/patterns.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/optimization/accumulator.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/optimization/linearize.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/optimization/poses_linearize.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/optimization/poses_optimize.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/optimization/spline_linearize.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/optimization/spline_optimize.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/utils/ba_utils.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/utils/cast_utils.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/utils/common_types.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/utils/filesystem.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/utils/format.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/utils/imu_types.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/utils/keypoints.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/utils/nfr.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/utils/sim_utils.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/utils/system_utils.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/utils/test_utils.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/utils/time_utils.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/utils/tracks.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/utils/union_find.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/utils/vio_config.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/utils/vis_utils.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/vi_estimator/ba_base.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/vi_estimator/landmark_database.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/vi_estimator/marg_helper.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/vi_estimator/nfr_mapper.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/vi_estimator/sc_ba_base.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/vi_estimator/sqrt_ba_base.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/vi_estimator/sqrt_keypoint_vio.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/vi_estimator/sqrt_keypoint_vo.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/vi_estimator/vio_estimator.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/api/basalt_vio.h
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/calibration/aprilgrid.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/calibration/calibraiton_helper.cpp  # 注意：原代码拼写错误calibraiton -> calibration
    ${CMAKE_CURRENT_SOURCE_DIR}/src/calibration/vignette.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/io/dataset_io.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/io/marg_data_io.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/linearization/landmark_block.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/linearization/linearization_abs_qr.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/linearization/linearization_abs_sc.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/linearization/linearization_base.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/linearization/linearization_rel_sc.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/optical_flow/optical_flow.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/keypoints.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/system_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/time_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/vio_config.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/vi_estimator/ba_base.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/vi_estimator/landmark_database.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/vi_estimator/marg_helper.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/vi_estimator/nfr_mapper.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/vi_estimator/sc_ba_base.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/vi_estimator/sqrt_ba_base.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/vi_estimator/sqrt_keypoint_vio.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/vi_estimator/sqrt_keypoint_vo.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/vi_estimator/vio_estimator.cpp
)

# 链接依赖库
target_link_libraries(basalt
  PUBLIC ${STD_CXX_FS} basalt::opencv basalt::basalt-headers TBB::tbb
  PRIVATE basalt::magic_enum rosbag apriltag opengv nlohmann::json fmt::fmt)
# 头文件路径（对外暴露include目录）
target_include_directories(basalt PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
# 编译定义（模板实例化等）
target_compile_definitions(basalt PUBLIC ${BASALT_COMPILE_DEFINITIONS})
#target_compile_definitions(basalt PUBLIC BASALT_DISABLE_ASSERTS)

# ==============================================================================
# 可执行程序构建
# ==============================================================================
# 创建可执行文件 basalt_calibrate
add_executable(basalt_calibrate src/calibrate.cpp src/calibration/cam_calib.cpp)
target_link_libraries(basalt_calibrate basalt ${BASALT_PANGO_TARGETS} basalt::cli11)

# 创建可执行文件 basalt_calibrate_imu
add_executable(basalt_calibrate_imu src/calibrate_imu.cpp src/calibration/cam_imu_calib.cpp)
target_link_libraries(basalt_calibrate_imu basalt ${BASALT_PANGO_TARGETS} basalt::cli11)

# 创建可执行文件 basalt_vio_sim
add_executable(basalt_vio_sim src/vio_sim.cpp)
target_link_libraries(basalt_vio_sim basalt ${BASALT_PANGO_TARGETS} basalt::cli11)

# 创建可执行文件 basalt_mapper_sim
add_executable(basalt_mapper_sim src/mapper_sim.cpp)
target_link_libraries(basalt_mapper_sim basalt ${BASALT_PANGO_TARGETS} basalt::cli11)

# mapper sim native doesn't use template free interface
# 如果启用双精度实例化，创建 naive 版本的映射器模拟
if(BASALT_INSTANTIATIONS_DOUBLE)
  add_executable(basalt_mapper_sim_naive src/mapper_sim_naive.cpp)
  target_link_libraries(basalt_mapper_sim_naive basalt ${BASALT_PANGO_TARGETS} basalt::cli11)
endif()

# 创建可执行文件 basalt_mapper
add_executable(basalt_mapper src/mapper.cpp)
target_link_libraries(basalt_mapper basalt ${BASALT_PANGO_TARGETS} basalt::cli11)

# 创建可执行文件 basalt_opt_flow
add_executable(basalt_opt_flow src/opt_flow.cpp)
target_link_libraries(basalt_opt_flow basalt ${BASALT_PANGO_TARGETS} basalt::cli11)

# 创建可执行文件 basalt_vio
add_executable(basalt_vio src/vio.cpp)
target_link_libraries(basalt_vio basalt ${BASALT_PANGO_TARGETS} basalt::cli11 fmt::fmt)

# 查找 Threads 库
find_package(Threads REQUIRED)

# sudo apt-get install nlohmann-json3-dev
# 查找 nlohmann_json 库
find_package(nlohmann_json REQUIRED)

# 创建可执行文件 basalt_vio_test
add_executable(basalt_vio_test test/src/api/test_basalt_vio.cpp src/api/basalt_vio.cpp)
target_include_directories(basalt_vio_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(basalt_vio_test basalt ${BASALT_PANGO_TARGETS} basalt::cli11 fmt::fmt Threads::Threads nlohmann_json::nlohmann_json)

# 创建可执行文件 basalt_time_alignment
add_executable(basalt_time_alignment src/time_alignment.cpp)
target_link_libraries(basalt_time_alignment basalt ${BASALT_PANGO_TARGETS} basalt::cli11)

# 创建可执行文件 basalt_kitti_eval
add_executable(basalt_kitti_eval src/kitti_eval.cpp)
target_link_libraries(basalt_kitti_eval basalt::basalt-headers basalt::cli11)

# 查找 realsense2 库
find_package(realsense2 QUIET)
if(realsense2_FOUND)
  add_executable(basalt_rs_t265_record src/rs_t265_record.cpp src/device/rs_t265.cpp)
  target_link_libraries(basalt_rs_t265_record basalt realsense2::realsense2 ${BASALT_PANGO_TARGETS} basalt::cli11)

  add_executable(basalt_rs_t265_vio src/rs_t265_vio.cpp src/device/rs_t265.cpp)
  target_link_libraries(basalt_rs_t265_vio basalt realsense2::realsense2 ${BASALT_PANGO_TARGETS} basalt::cli11)
endif()

# ==============================================================================
# 安装配置
# ==============================================================================
# 安装可执行程序和库
install(TARGETS basalt_calibrate basalt_calibrate_imu basalt_vio_sim basalt_mapper_sim basalt_mapper basalt_opt_flow basalt_vio basalt_kitti_eval basalt_time_alignment basalt
  EXPORT BasaltTargets
  RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
  ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)

# 安装原生Mapper仿真工具（仅double类型启用）
if(BASALT_INSTANTIATIONS_DOUBLE)
  install(TARGETS basalt_mapper_sim_naive
    EXPORT BasaltTargets
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
    ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
endif()

# 安装RealSense工具（仅找到realsense2时）
if(realsense2_FOUND)
    install(TARGETS basalt_rs_t265_record basalt_rs_t265_vio
      EXPORT BasaltTargets
      RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
      LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
      ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
endif()

# 安装 Python 脚本
file(GLOB SCRIPTS_TO_INSTALL "${CMAKE_CURRENT_SOURCE_DIR}/scripts/basalt_*.py")
install(PROGRAMS ${SCRIPTS_TO_INSTALL} DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)

# 安装配置文件
file(GLOB CONFIG_FILES "${CMAKE_CURRENT_SOURCE_DIR}/data/*.json")
install(FILES ${CONFIG_FILES}
  DESTINATION ${CMAKE_INSTALL_PREFIX}/etc/basalt)

# ==============================================================================
# 测试构建（临时禁用install避免冲突）
# ==============================================================================
# Replace install() to do-nothing macro.
# 临时替换install宏为空（避免测试代码被安装）
macro(install)
endmacro()

# Include subproject (or any other CMake code) with "disabled" install().
# 启用测试
enable_testing()
# 构建basalt-headers测试
add_subdirectory(thirdparty/basalt-headers/test)
# 构建项目自身测试
add_subdirectory(test)

# Restore original install() behavior.
# 恢复原始install宏
macro(install)
  _install(${ARGN})
endmacro()
