/**
BSD 3-Clause License

This file is part of the Basalt project.
https://gitlab.com/VladyslavUsenko/basalt.git

Copyright (c) 2019, Vladyslav Usenko and Nikolaus Demmel.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#pragma once

#include <memory>

#include <Eigen/Geometry>

#include <basalt/utils/vio_config.h>

#include <basalt/io/dataset_io.h>
#include <basalt/calibration/calibration.hpp>
#include <basalt/camera/stereographic_param.hpp>
#include <basalt/utils/sophus_utils.hpp>

#include <tbb/concurrent_queue.h>

namespace basalt {

// 类型别名：特征点ID，用无符号整数（size_t）表示，增强代码可读性
using KeypointId = size_t;

/**
 * @brief 光流计算的输入数据结构体
 * 
 * @note 存储双目图像的时间戳和图像数据，作为光流算法的输入
 */
struct OpticalFlowInput {
  // 智能指针类型别名：简化 shared_ptr 声明，符合C++工程实践
  using Ptr = std::shared_ptr<OpticalFlowInput>;

  int64_t t_ns;                     // 时间戳，单位：纳秒（ns），保证高精度时间对齐
  std::vector<ImageData> img_data;  // 双目图像数据
};

/**
 * @brief 视觉前端输出的光流结果结构体
 * 
 * @note 存储光流跟踪后的特征点信息，是光流算法的核心输出
 */
struct OpticalFlowResult {
  using Ptr = std::shared_ptr<OpticalFlowResult>;

  int64_t t_ns;   // 时间戳（纳秒）

  /**
   * @brief 双目相机特征点观测结果
   *  
   * observations.size(): 表示相机的个数（默认是2，即双目相机）
   * Eigen::aligned_map（Eigen兼容的map，保证内存对齐，避免SIMD指令报错）
   *   Key: KeypointId：表示特征点ID
   *   Value: Eigen::AffineCompact2f：存放的特征点的像素位置（带方向）  [ 1  0  x ] translation().x()
   *                                                              [ 0  1  y ] translation().y()
   *                                                              
   */
  std::vector<Eigen::aligned_map<KeypointId, Eigen::AffineCompact2f>> observations;

  /**
   * @brief 特征点金字塔层级信息
   * 
   * pyramid_levels.size(): 表示相机的个数（默认是2，即双目相机）
   * map:
   *   Key: KeypointId：表示特征点ID
   *   Value: 该特征点跟踪时使用的图像金字塔层级
   * 
   * @note 图像金字塔：低层（小数值）对应高分辨率，高层对应低分辨率，用于提升光流跟踪的鲁棒性
   */
  std::vector<std::map<KeypointId, size_t>> pyramid_levels;

  // 指向对应输入图像数据的智能指针，便于回溯原始图像
  OpticalFlowInput::Ptr input_images;
};

/**
 * @brief 光流算法基类
 * 
 * @note 采用基类+工厂模式设计：统一接口，便于扩展不同的光流实现（如LK光流、金字塔光流等）
 */
class OpticalFlowBase {
 public:
  using Ptr = std::shared_ptr<OpticalFlowBase>;

  /**
   * 输入并发队列：存储待处理的光流输入数据
   * - tbb::concurrent_bounded_queue：TBB的有界并发队列，线程安全，支持多生产者-单消费者/多消费者模型
   * - 用途：视觉前端的图像读取线程（生产者）往队列塞数据，光流计算线程（消费者）从队列取数据
   */
  tbb::concurrent_bounded_queue<OpticalFlowInput::Ptr> input_queue;
  
  
  /**
   * 输出并发队列指针：存储光流计算完成的结果
   * - 指针类型：允许外部绑定不同的输出队列（如绑定到VIO主线程的队列）
   * - 用途：光流线程计算完成后，将结果写入该队列，供后续模块（如后端优化）消费
   */
  tbb::concurrent_bounded_queue<OpticalFlowResult::Ptr>* output_queue = nullptr;

  /**
   * 特征点补丁（patch）坐标矩阵
   * - Eigen::MatrixXf：动态大小的浮点矩阵
   * - 用途：预存储光流跟踪时使用的特征点邻域补丁坐标（如8x8/16x16像素块），避免重复计算，提升效率
   */
  Eigen::MatrixXf patch_coord;
};

/**
 * @brief 光流算法工厂类
 * @note 工厂模式：封装光流算法的创建逻辑，用户只需传入配置和标定参数，无需关心具体实现类
 */
class OpticalFlowFactory {
 public:

  /**
   * @brief 创建光流算法实例的静态工厂方法
   * 
   * @param config VIO配置结构体：包含光流相关参数（如跟踪窗口大小、金字塔层数、最大跟踪数等）
   * @param cam 相机标定结构体：包含相机内参、畸变参数、多相机外参等，光流需要基于标定参数矫正像素坐标
   * 
   * @return 光流算法基类指针：指向具体的光流实现类（如LKOpticalFlow），符合面向接口编程
   */
  static OpticalFlowBase::Ptr getOpticalFlow(const VioConfig& config,
                                             const Calibration<double>& cam);
};
}  // namespace basalt
