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

#include <string>

namespace basalt {

enum class LinearizationType { ABS_QR, ABS_SC, REL_SC };

struct VioConfig {
  /**
   * @brief 构造函数，初始化所有VIO配置参数的默认值
   * 
   */
  VioConfig();
  void load(const std::string& filename);
  void save(const std::string& filename);

  // 光流类型
  // 默认值："frame_to_frame"
  std::string optical_flow_type;

  // 光流特征检测网格大小（像素），将图像分成50x50网格，每个网格提取特征，避免特征扎堆
  // 默认值：50
  int optical_flow_detection_grid_size;

  // 光流恢复的最大距离平方（归一化坐标），即右目返投到左目的特征像素与原特征像素值的差异，超过该值的特征视为跟踪失败
  // 默认值：0.09f
  float optical_flow_max_recovered_dist2;

  // 光流匹配的特征模板模式
  // 默认值：51
  int optical_flow_pattern;

  // 光流迭代优化的最大次数
  // 默认值：5
  int optical_flow_max_iterations;

  // 光流金字塔层数
  // 默认值：3（兼顾跟踪精度和速度，多层金字塔适应大位移）
  int optical_flow_levels;

  // 光流极线误差阈值（归一化坐标），用于剔除极线约束下的异常匹配
  // 默认值：0.005
  float optical_flow_epipolar_error;

  // 光流跳过帧数，每几帧计算一次光流
  // 默认值：1（不跳帧）
  int optical_flow_skip_frames;

  // ===================== VIO核心参数 =====================

  // VIO线性化类型
  // 默认值：ABS_QR（绝对位姿+QR分解）
  LinearizationType vio_linearization_type;

  // 是否使用平方根边缘化，启用可提升数值稳定性
  // 默认值：true
  bool vio_sqrt_marg;

  // VIO状态缓存的最大数量
  // 默认值：3
  int vio_max_states;

  // 滑动窗口中关键帧最大数量
  // 默认值：7
  int vio_max_kfs;

  // 新增关键帧的最小间隔帧数
  // 默认值：5
  int vio_min_frames_after_kf;

  // 新增关键帧的特征阈值，如果左目相机中已关联3D点的特征点占比小于阈值，则创建新的关键帧
  // 默认值：0.7
  float vio_new_kf_keypoints_thresh;

  // 调试模式开关
  // 默认值：false
  bool vio_debug;

  // 扩展日志开关，输出更详细的VIO运行日志
  // 默认值：false
  bool vio_extended_logging;

  //  double vio_outlier_threshold;
  //  int vio_filter_iteration;

  // VIO优化最大迭代次数
  // 默认值：7
  int vio_max_iterations;

  // 观测值标准差（像素）：用于计算观测值的权重
  // 默认值：0.5像素（经验值）
  double vio_obs_std_dev;

  // 观测值Huber鲁棒阈值，用于处理异常值（超过阈值的残差为异常值，会被鲁棒加权）
  // 默认值：1.0像素
  double vio_obs_huber_thresh;

  // 三角化最小深度阈值，小于阈值的特征不进行三角化，避免数值不稳定
  // 默认值：0.05米
  double vio_min_triangulation_dist;

  // 是否强制实时性：false（不强制，优先精度）/ true（强制，超时则停止优化）
  // 默认值：false
  bool vio_enforce_realtime;

  // VIO：是否使用Levenberg-Marquardt（LM）优化
  // 默认值：false（默认用Gauss-Newton，速度更快）
  bool vio_use_lm;

  // VIO：LM优化初始lambda值
  // 默认值：1e-4（控制步长，初始值小，优先快速收敛）
  double vio_lm_lambda_initial;

  // VIO：LM优化最小lambda值
  // 默认值：1e-6（lambda越小，越接近Gauss-Newton）
  double vio_lm_lambda_min;

  // VIO：LM优化最大lambda值
  // 默认值：1e2（lambda越大，步长越小，越稳定）
  double vio_lm_lambda_max;

  // 是否缩放雅可比矩阵
  // 默认值：true（启用，提升数值稳定性）
  bool vio_scale_jacobian;

  // VIO初始化：位姿权重
  // 默认值：1e8（极大值，强制位姿固定）
  double vio_init_pose_weight;

  // VIO初始化：加速度计偏置权重
  // 默认值：1e1
  double vio_init_ba_weight;

  // VIO初始化：陀螺仪偏置权重
  // 默认值：1e2
  double vio_init_bg_weight;

  // 是否边缘化丢失的路标点
  // 默认值：true（启用，清理无效路标点，减少计算量）
  bool vio_marg_lost_landmarks;

  // 关键帧边缘化的特征比例阈值
  // 默认值：0.1（关键帧保留10%的特征，其余边缘化）
  double vio_kf_marg_feature_ratio;

  // ===================== 地图构建器（Mapper）参数 =====================

  // 地图构建观测值标准差
  // 默认值：0.25像素（比VIO更严格，提升地图精度）
  double mapper_obs_std_dev;

  // 地图构建观测值Huber阈值
  // 默认值：1.5（处理地图优化中的异常值）
  double mapper_obs_huber_thresh;

  // 地图特征检测点数，每帧提取的特征点个数
  // 默认值：800
  int mapper_detection_num_points;

  // 匹配的最大帧数
  // 默认值：30（最多匹配30帧内的特征，避免无效匹配）
  double mapper_num_frames_to_match;

  // 帧匹配阈值（归一化误差），低于该值视为匹配成功
  // 默认值：0.04
  double mapper_frames_to_match_threshold;

  // 最小匹配数
  // 默认值：20（至少20个匹配对才认为帧匹配有效）
  double mapper_min_matches;

  // RANSAC阈值（归一化坐标），用于剔除外点，提升匹配鲁棒性
  // 默认值：5e-5
  double mapper_ransac_threshold;

  // 最小跟踪长度
  // 默认值：5（特征至少跟踪5帧才纳入地图，避免短期噪声特征）
  double mapper_min_track_length;

  // 最大汉明距离，用于特征描述子匹配，距离越小匹配越准
  // 默认值：70
  double mapper_max_hamming_distance;

  // 次优匹配测试比例：1.2（最优匹配/次优匹配 < 1.2 才视为有效匹配）
  double mapper_second_best_test_ratio;

  // BOW（词袋）特征位数：16位，用于快速帧匹配
  int mapper_bow_num_bits;

  // 地图三角化最小深度：0.07米（比VIO更严格，提升地图点精度）
  double mapper_min_triangulation_dist;

  // 是否禁用因子权重：false（启用权重，平衡不同观测值的影响）
  bool mapper_no_factor_weights;

  // 是否使用因子图优化：true（启用，地图构建的核心优化方式）
  bool mapper_use_factors;

  // 建图：是否使用Levenberg-Marquardt（LM）优化
  // 默认值：false（默认用Gauss-Newton，速度更快）
  bool mapper_use_lm;

  // 建图：LM优化初始lambda值
  // 默认值：1e-32（控制步长，初始值小，优先快速收敛）
  double mapper_lm_lambda_min;

  // 建图：LM优化最大lambda值
  // 默认值：1e2（lambda越大，步长越小，越稳定）
  double mapper_lm_lambda_max;
};

}  // namespace basalt
