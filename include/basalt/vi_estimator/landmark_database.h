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

#include <basalt/utils/imu_types.h>
#include <basalt/utils/eigen_utils.hpp>

namespace basalt {
 
/**
 * @brief 单次特征点观测记录，表示"在某一帧图像中，观测到了某个路标点，其2D坐标为pos"。
 * 
 * @note 用法：用于前端光流跟踪结果传递给后端时，描述每一个特征点的观测信息
 *       注意：观测发生在哪一帧（TimeCamId）并不存储在此结构体中，
 *            而是由调用上下文提供（如 addObservation(tcid_target, o) 中的 tcid_target）
 * 
 * @tparam Scalar_ 模板参数，数值类型（如float/double），用于控制精度
 */
template <class Scalar_>
struct KeypointObservation {
  using Scalar = Scalar_;

  // 路标点的全局唯一ID（KeypointId），用于在路标点数据库 kpts 中索引。
  // 同一个路标点在不同帧中被观测到时，kpt_id 保持一致
  int kpt_id;

  // 该路标点在当前观测帧中的2D归一化坐标
  //   - 归一化坐标是经过相机内参去畸变后的坐标，位于归一化平面(z=1)上
  //   - 与像素坐标的区别：像素坐标依赖内参(fx,fy,cx,cy)，归一化坐标与内参无关
  Eigen::Matrix<Scalar, 2, 1> pos;

  // Eigen内存对齐宏，确保包含Eigen固定大小向量的结构体满足SIMD对齐要求
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
 
// keypoint position defined relative to some frame
/**
 * @brief 路标点（3D地图点）的参数化表示
 * 
 * @note 采用"逆深度参数化"：在宿主帧(host frame)的相机坐标系下，
 *       用观测方向(direction) + 逆深度(inv_dist) 来表示路标点的3D位置。
 *       相比直接存储3D坐标(x,y,z)，逆深度参数化的优点：
 *          1) 可以自然地表示无穷远点（inv_dist → 0）
 *          2) 对远距离路标点的数值稳定性更好
 *          3) 逆深度的不确定性近似高斯分布，适合线性化优化
 * 
 * @tparam Scalar_ 模板参数，数值类型（如float/double）
 */
template <class Scalar_>
struct Keypoint {
  using Scalar = Scalar_;
  using Vec2 = Eigen::Matrix<Scalar, 2, 1>;

  // ObsMap: 该路标点的观测列表，key=观测帧的TimeCamId, value=2D归一化坐标
  using ObsMap = Eigen::aligned_map<TimeCamId, Vec2>;
  using MapIter = typename ObsMap::iterator;

  // ===== 路标点的3D位置参数（逆深度参数化）=====
  // direction: 路标点在宿主帧相机坐标系下的观测方向（2D bearing向量）
  //   从宿主帧相机光心出发，指向路标点的单位方向在某种参数化下的2D表示
  // 3D position parameters
  Vec2 direction;
  // inv_dist: 路标点的逆深度（1/depth），depth是路标点到宿主帧相机光心的距离
  //   路标点在宿主帧相机坐标系下的3D坐标可由 direction 和 inv_dist 恢复
  Scalar inv_dist;

  // ===== 观测信息 =====
  // host_kf_id: 宿主关键帧的ID（时间戳+相机ID），即路标点首次被观测到的帧
  //   路标点的3D参数(direction, inv_dist)定义在该帧的相机坐标系下
  // Observations
  TimeCamId host_kf_id;   // 关键帧ID和相机ID

  // obs: 所有观测到该路标点的帧及其对应的2D归一化坐标
  //   key = TimeCamId (观测帧), value = Vec2 (该路标点在观测帧中的2D坐标)
  //   注意：obs中通常也包含宿主帧自身的观测
  ObsMap obs;

  // backup/restore: 用于LM优化中的回退机制
  // LM算法尝试一步更新后，如果代价函数没有下降，需要撤销更新回到上一步的状态
  // backup() 在更新前保存当前参数，restore() 在更新失败时恢复参数
  inline void backup() {
    backup_direction = direction;
    backup_inv_dist = inv_dist;
  }

  inline void restore() {
    direction = backup_direction;
    inv_dist = backup_inv_dist;
  }

  // Eigen内存对齐宏，确保SIMD指令（如SSE/AVX）可以正确访问成员变量
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 private:
  // 备份存储，仅供 backup()/restore() 使用
  Vec2 backup_direction;
  Scalar backup_inv_dist;
};

template <class Scalar_>
class LandmarkDatabase {
 public:
  using Scalar = Scalar_;

  // Non-const
  void addLandmark(KeypointId lm_id, const Keypoint<Scalar>& pos);

  void removeFrame(const FrameId& frame);

  void removeKeyframes(const std::set<FrameId>& kfs_to_marg,
                       const std::set<FrameId>& poses_to_marg,
                       const std::set<FrameId>& states_to_marg_all);

  // 将该观测关联添加到路标点数据库
  void addObservation(const TimeCamId& tcid_target,
                      const KeypointObservation<Scalar>& o);

  Keypoint<Scalar>& getLandmark(KeypointId lm_id);

  // Const
  // 获取
  const Keypoint<Scalar>& getLandmark(KeypointId lm_id) const;

  std::vector<TimeCamId> getHostKfs() const;

  std::vector<const Keypoint<Scalar>*> getLandmarksForHost(
      const TimeCamId& tcid) const;

  const std::unordered_map<TimeCamId,
                           std::map<TimeCamId, std::set<KeypointId>>>&
  getObservations() const;

  const Eigen::aligned_unordered_map<KeypointId, Keypoint<Scalar>>&
  getLandmarks() const;

  bool landmarkExists(int lm_id) const;

  size_t numLandmarks() const;

  int numObservations() const;

  int numObservations(KeypointId lm_id) const;

  void removeLandmark(KeypointId lm_id);

  void removeObservations(KeypointId lm_id, const std::set<TimeCamId>& obs);

  inline void backup() {
    for (auto& kv : kpts) kv.second.backup();
  }

  inline void restore() {
    for (auto& kv : kpts) kv.second.restore();
  }

 private:
  using MapIter =
      typename Eigen::aligned_unordered_map<KeypointId,
                                            Keypoint<Scalar>>::iterator;
  MapIter removeLandmarkHelper(MapIter it);
  typename Keypoint<Scalar>::MapIter removeLandmarkObservationHelper(
      MapIter it, typename Keypoint<Scalar>::MapIter it2);

  /**
   * @brief 路标点数据库的核心存储
   *        存储了滑动窗口中所有活跃路标点的完整信息——知道它在哪被首次看到、3D位置是多少、以及它在哪些帧中又被观测到。
   * 
   *  key: KeypointId（即 int），路标点的全局唯一ID
   *  value: Keypoint<Scalar> 结构体，包含：
   *           - direction + inv_dist：路标点在宿主帧下的逆深度参数化3D位置
   *           - host_kf_id：宿主帧ID（路标点的3D参数定义在该帧坐标系下）
   *           - obs：所有观测到该路标点的帧及其2D归一化坐标
   */
  Eigen::aligned_unordered_map<KeypointId, Keypoint<Scalar>> kpts;

  /**
   * @brief 帧间共视关系索引表，记录：“host帧上首次观测到的路标点，在target帧中又被观测到了哪些？”
   *        例如：observations[tcid_h][tcid_t] 返回的是：所有以 tcid_h 为宿主帧、同时在 tcid_t 中也被观测到的路标点ID集合。
   * 
   * @note  这是一个三层嵌套的映射结构：
   *          - 外层 key：TimeCamId (host帧)，表示路标点的宿主帧（路标点首次被观测到的帧+相机）
   *          - 中层 key：TimeCamId (target帧)，观测帧（观测到该路标点的其他帧+相机）
   *          - 内层 value： std::set<KeypointId> ，在这对 (host, target) 关系中，被共同涉及的路标点ID集合
   */
  std::unordered_map<TimeCamId, std::map<TimeCamId, std::set<KeypointId>>>
      observations;

  static constexpr int min_num_obs = 2;
};

}  // namespace basalt
