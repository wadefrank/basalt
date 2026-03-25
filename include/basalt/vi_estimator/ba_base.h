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

#include <basalt/vi_estimator/landmark_database.h>

namespace basalt {

template <class Scalar_>
class BundleAdjustmentBase {
 public:
  using Scalar = Scalar_;
  using Vec2 = Eigen::Matrix<Scalar, 2, 1>;
  using Vec3 = Eigen::Matrix<Scalar, 3, 1>;
  using Vec4 = Eigen::Matrix<Scalar, 4, 1>;
  using Vec6 = Eigen::Matrix<Scalar, 6, 1>;
  using VecX = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

  using Mat4 = Eigen::Matrix<Scalar, 4, 4>;
  using Mat6 = Eigen::Matrix<Scalar, 6, 6>;
  using MatX = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

  using SE3 = Sophus::SE3<Scalar>;

  void computeError(Scalar& error,
                    std::map<int, std::vector<std::pair<TimeCamId, Scalar>>>*
                        outliers = nullptr,
                    Scalar outlier_threshold = 0) const;

  void filterOutliers(Scalar outlier_threshold, int min_num_obs);

  void optimize_single_frame_pose(
      PoseStateWithLin<Scalar>& state_t,
      const std::vector<std::vector<int>>& connected_obs) const;

  template <class Scalar2>
  void get_current_points(
      Eigen::aligned_vector<Eigen::Matrix<Scalar2, 3, 1>>& points,
      std::vector<int>& ids) const;

  void computeDelta(const AbsOrderMap& marg_order, VecX& delta) const;

  void linearizeMargPrior(const MargLinData<Scalar>& mld,
                          const AbsOrderMap& aom, MatX& abs_H, VecX& abs_b,
                          Scalar& marg_prior_error) const;

  void computeMargPriorError(const MargLinData<Scalar>& mld,
                             Scalar& marg_prior_error) const;

  Scalar computeMargPriorModelCostChange(const MargLinData<Scalar>& mld,
                                         const VecX& marg_scaling,
                                         const VecX& marg_pose_inc) const;

  // TODO: Old version for squared H and b. Remove when no longer needed.
  Scalar computeModelCostChange(const MatX& H, const VecX& b,
                                const VecX& inc) const;

  template <class Scalar2>
  void computeProjections(
      std::vector<Eigen::aligned_vector<Eigen::Matrix<Scalar2, 4, 1>>>& data,
      FrameId last_state_t_ns) const;

  /// Triangulates the point and returns homogenous representation. First 3
  /// components - unit-length direction vector. Last component inverse
  /// distance.

  /** @brief 对点进行三角化并返回齐次坐标表示（直接线性三角化，Direct Linear Triangulation）
   *  
   *  返回值说明：
   *     - 前3个分量：单位长度的方向向量（归一化的三维空间点方向）
   *     - 最后1个分量：逆距离（inverse distance）
   * 
   * @tparam Derived 输入向量的Eigen派生类型（自动推导）
   * 
   * @param f0 第一个相机坐标系下的归一化平面坐标（3维向量，单位长度方向）
   * @param f1 第二个相机坐标系下的归一化平面坐标（3维向量，单位长度方向）
   * @param T_0_1 从相机1到相机0的位姿变换（SE3：旋转+平移）
   * 
   * @return 4维齐次向量，前3维为单位方向向量，最后1维为逆距离
   */
  template <class Derived>
  static Eigen::Matrix<typename Derived::Scalar, 4, 1> triangulate(
      const Eigen::MatrixBase<Derived>& f0,
      const Eigen::MatrixBase<Derived>& f1,
      const Sophus::SE3<typename Derived::Scalar>& T_0_1) {
    
    // 静态断言：确保输入的Derived类型是3维向量（编译期检查）
    EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived, 3);

    // suffix "2" to avoid name clash with class-wide typedefs
    // 定义局部类型别名，避免与类级别的typedef命名冲突（后缀2区分）
    // Scalar_2：输入向量的数值类型（如float/double）
    using Scalar_2 = typename Derived::Scalar;

    // Vec4_2：4维列向量，元素类型为Scalar_2
    using Vec4_2 = Eigen::Matrix<Scalar_2, 4, 1>;

    // 定义两个相机的投影矩阵（3x4）：P1对应相机0，P2对应相机1
    Eigen::Matrix<Scalar_2, 3, 4> P1, P2;

    // 相机0的投影矩阵设为单位矩阵（假设相机0为参考坐标系，内参已归一化）
    // 归一化平面投影：P = [I | 0]，即 X_c = R*X_w + t，这里R=I, t=0
    P1.setIdentity();

    // 相机1的投影矩阵：由位姿T_0_1（相机1→相机0）的逆变换得到
    // T_0_1.inverse() = T_1_0（相机0→相机1），matrix3x4()返回SE3的3x4投影矩阵 [R | t]
    // 因此P2 = T_1_0的投影矩阵，即相机1的投影矩阵（将世界点投影到相机1的归一化平面）
    P2 = T_0_1.inverse().matrix3x4();

    // 构建4x4的线性方程组矩阵A：Ax=0，x为齐次空间点
    // 每个相机的归一化坐标对应2个约束方程，两个相机共4个方程
    Eigen::Matrix<Scalar_2, 4, 4> A(4, 4);

    // 共线约束：相机光心→空间点的射线，与相机光心→归一化平面点的射线重合。
    // 两个向量共线 -> 叉乘 = 0
    //   1. f0 = [u0, v0, 1]^T（归一化坐标，最后一维为1），满足 f0 × (P1 * X) = 0
    //      叉乘展开后得到: u0*P1(2,:) - P1(0,:) = 0
    //                    v0*P1(2,:) - P1(1,:) = 0


    // 相机0的第一个约束方程：f0[0]*Z0 - X0 = 0 （归一化平面：x=X/Z, y=Y/Z → xZ-X=0）
    // P1.row(2)是投影矩阵第3行（对应Z分量），P1.row(0)是第1行（对应X分量）
    // 推导：
    // 
    A.row(0) = f0[0] * P1.row(2) - f0[2] * P1.row(0);
    
    // 相机0的第二个约束方程：f0[1]*Z0 - Y0 = 0
    A.row(1) = f0[1] * P1.row(2) - f0[2] * P1.row(1);

    // 相机1的第一个约束方程：f1[0]*Z1 - X1 = 0
    A.row(2) = f1[0] * P2.row(2) - f1[2] * P2.row(0);

    // 相机1的第二个约束方程：f1[1]*Z1 - Y1 = 0
    A.row(3) = f1[1] * P2.row(2) - f1[2] * P2.row(1);


    // 对矩阵A进行SVD分解（奇异值分解），求解Ax=0的最小二乘解
    // Eigen::ComputeFullV：计算完整的右奇异矩阵V
    // SVD分解：A = U*Σ*V^T，Ax=0的解是V中对应最小奇异值的列向量（最后一列）
    Eigen::JacobiSVD<Eigen::Matrix<Scalar_2, 4, 4>> mySVD(A,
                                                          Eigen::ComputeFullV);

    // 提取V矩阵的第4列（索引3）作为齐次空间点的解（最小奇异值对应的列）
    Vec4_2 worldPoint = mySVD.matrixV().col(3);
    
    // 归一化：将前3个分量（三维空间点）归一化为单位长度的方向向量
    // head<3>()：取前3个元素，norm()计算其L2范数，整体除以范数得到单位向量
    worldPoint /= worldPoint.template head<3>().norm();

    // Enforce same direction of bearing vector and initial point
    // 方向一致性检查：确保三角化得到的方向向量与相机0的观测方向f0同向
    // 如果点积小于0，说明方向相反，取反修正
    if (f0.dot(worldPoint.template head<3>()) < 0) worldPoint *= -1;

    // 返回齐次表示：前3维=单位方向向量，最后1维=逆距离（由归一化过程自然得到）
    return worldPoint;
  }

  inline void backup() {
    for (auto& kv : frame_states) kv.second.backup();
    for (auto& kv : frame_poses) kv.second.backup();
    lmdb.backup();
  }

  inline void restore() {
    for (auto& kv : frame_states) kv.second.restore();
    for (auto& kv : frame_poses) kv.second.restore();
    lmdb.restore();
  }
  
  // protected:
  /**
   * @brief 根据时间戳获取帧的位姿状态（含线性化点信息）
   *
   * 系统中帧的状态存储在两个不同的容器中：
   *   - frame_poses: 仅包含位姿 (T_w_i)，用于已被边缘化的帧（如mapper中的关键帧）
   *   - frame_states: 包含完整状态 (位姿 + 速度 + IMU偏置)，用于滑动窗口中的活跃帧
   *
   * 查找策略：优先在 frame_poses 中查找，找不到再去 frame_states 中查找。
   * 返回类型统一为 PoseStateWithLin（仅含位姿部分），如果来源是 frame_states，
   * 则通过转换构造函数从 PoseVelBiasStateWithLin 中提取位姿相关字段
   * （T_w_i、delta的前6维、linearized标志）。
   *
   * @param t_ns 帧的时间戳（纳秒），作为帧的唯一标识
   * @return 该帧的位姿状态，包含线性化点位姿、当前位姿和增量delta
   */
  PoseStateWithLin<Scalar> getPoseStateWithLin(int64_t t_ns) const {
    // 首先在 frame_poses 中查找（边缘化后仅保留位姿的帧）
    auto it = frame_poses.find(t_ns);
    if (it != frame_poses.end()) return it->second;

    // frame_poses 中没有，则在 frame_states 中查找（滑动窗口中的活跃帧）
    auto it2 = frame_states.find(t_ns);
    if (it2 == frame_states.end()) {
      // 两个容器中都找不到，说明该时间戳对应的帧不存在，属于严重错误，直接终止
      std::cerr << "Could not find pose " << t_ns << std::endl;
      std::abort();
    }

    // 从 PoseVelBiasStateWithLin 构造 PoseStateWithLin：
    // 提取 linearized 标志、delta 的前6维（位姿增量）、线性化点位姿，
    // 并通过 incPose(delta) 恢复出当前位姿 T_w_i_current
    return PoseStateWithLin<Scalar>(it2->second);
  }

  // IMU预积分状态（位姿 + 速度 + IMU 偏置）
  Eigen::aligned_map<int64_t, PoseVelBiasStateWithLin<Scalar>> frame_states;
  Eigen::aligned_map<int64_t, PoseStateWithLin<Scalar>> frame_poses;

  // Point management
  LandmarkDatabase<Scalar> lmdb;  // 路标点数据集，管理3D路标点

  Scalar obs_std_dev;
  Scalar huber_thresh;

  basalt::Calibration<Scalar> calib;
};
}  // namespace basalt
