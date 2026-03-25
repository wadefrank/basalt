/**
BSD 3-Clause License

This file is part of the Basalt project.
https://gitlab.com/VladyslavUsenko/basalt.git

Copyright (c) 2021, Vladyslav Usenko and Nikolaus Demmel.
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

#include <basalt/linearization/linearization_abs_qr.hpp>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

#include <basalt/utils/ba_utils.h>
#include <basalt/linearization/imu_block.hpp>
#include <basalt/utils/cast_utils.hpp>

namespace basalt {

template <typename Scalar, int POSE_SIZE>
LinearizationAbsQR<Scalar, POSE_SIZE>::LinearizationAbsQR(
    BundleAdjustmentBase<Scalar>* estimator, const AbsOrderMap& aom,
    const Options& options, const MargLinData<Scalar>* marg_lin_data,
    const ImuLinData<Scalar>* imu_lin_data,
    const std::set<FrameId>* used_frames,
    const std::unordered_set<KeypointId>* lost_landmarks,
    int64_t last_state_to_marg)
    : options_(options),
      estimator(estimator),
      lmdb_(estimator->lmdb),
      frame_poses(estimator->frame_poses),
      calib(estimator->calib),
      aom(aom),
      used_frames(used_frames),
      marg_lin_data(marg_lin_data),
      imu_lin_data(imu_lin_data),
      pose_damping_diagonal(0),
      pose_damping_diagonal_sqrt(0) {
  UNUSED(last_state_to_marg);

  BASALT_ASSERT_STREAM(
      options.lb_options.huber_parameter == estimator->huber_thresh,
      "Huber threshold should be set to the same value");

  BASALT_ASSERT_STREAM(options.lb_options.obs_std_dev == estimator->obs_std_dev,
                       "obs_std_dev should be set to the same value");

  // Allocate memory for relative pose linearization
  for (const auto& [tcid_h, target_map] : lmdb_.getObservations()) {
    // if (used_frames && used_frames->count(tcid_h.frame_id) == 0) continue;
    const size_t host_idx = host_to_idx_.size();
    host_to_idx_.try_emplace(tcid_h, host_idx);
    host_to_landmark_block.try_emplace(tcid_h);

    // assumption: every host frame has at least target frame with
    // observations
    // NOTE: in case a host frame loses all of its landmarks due
    // to outlier removal or marginalization of other frames, it becomes quite
    // useless and is expected to be removed before optimization.
    BASALT_ASSERT(!target_map.empty());

    for (const auto& [tcid_t, obs] : target_map) {
      // assumption: every target frame has at least one observation
      BASALT_ASSERT(!obs.empty());

      std::pair<TimeCamId, TimeCamId> key(tcid_h, tcid_t);
      relative_pose_lin.emplace(key, RelPoseLin<Scalar>());
    }
  }

  // Populate lookup for relative poses grouped by host-frame
  for (const auto& [tcid_h, target_map] : lmdb_.getObservations()) {
    // if (used_frames && used_frames->count(tcid_h.frame_id) == 0) continue;
    relative_pose_per_host.emplace_back();

    for (const auto& [tcid_t, _] : target_map) {
      std::pair<TimeCamId, TimeCamId> key(tcid_h, tcid_t);
      auto it = relative_pose_lin.find(key);

      BASALT_ASSERT(it != relative_pose_lin.end());

      relative_pose_per_host.back().emplace_back(it);
    }
  }

  num_cameras = frame_poses.size();

  landmark_ids.clear();
  for (const auto& [k, v] : lmdb_.getLandmarks()) {
    if (used_frames || lost_landmarks) {
      if (used_frames && used_frames->count(v.host_kf_id.frame_id)) {
        landmark_ids.emplace_back(k);
      } else if (lost_landmarks && lost_landmarks->count(k)) {
        landmark_ids.emplace_back(k);
      }
    } else {
      landmark_ids.emplace_back(k);
    }
  }
  size_t num_landmakrs = landmark_ids.size();

  // std::cout << "num_landmakrs " << num_landmakrs << std::endl;

  landmark_blocks.resize(num_landmakrs);

  {
    auto body = [&](const tbb::blocked_range<size_t>& range) {
      for (size_t r = range.begin(); r != range.end(); ++r) {
        KeypointId lm_id = landmark_ids[r];
        auto& lb = landmark_blocks[r];
        auto& landmark = lmdb_.getLandmark(lm_id);

        lb = LandmarkBlock<Scalar>::template createLandmarkBlock<POSE_SIZE>();

        lb->allocateLandmark(landmark, relative_pose_lin, calib, aom,
                             options.lb_options);
      }
    };

    tbb::blocked_range<size_t> range(0, num_landmakrs);
    tbb::parallel_for(range, body);
  }

  landmark_block_idx.reserve(num_landmakrs);

  num_rows_Q2r = 0;
  for (size_t i = 0; i < num_landmakrs; i++) {
    landmark_block_idx.emplace_back(num_rows_Q2r);
    const auto& lb = landmark_blocks[i];
    num_rows_Q2r += lb->numQ2rows();

    host_to_landmark_block.at(lb->getHostKf()).emplace_back(lb.get());
  }

  if (imu_lin_data) {
    for (const auto& kv : imu_lin_data->imu_meas) {
      imu_blocks.emplace_back(
          new ImuBlock<Scalar>(kv.second, imu_lin_data, aom));
    }
  }

  //    std::cout << "num_rows_Q2r " << num_rows_Q2r << " num_poses " <<
  //    num_cameras
  //              << std::endl;
}

template <typename Scalar, int POSE_SIZE>
LinearizationAbsQR<Scalar, POSE_SIZE>::~LinearizationAbsQR() = default;

template <typename Scalar_, int POSE_SIZE_>
void LinearizationAbsQR<Scalar_, POSE_SIZE_>::log_problem_stats(
    ExecutionStats& stats) const {
  UNUSED(stats);
}

/**
 * @brief 对整个BA问题进行线性化
 * 
 * @note 核心步骤: 
 *        1) 线性化相对位姿
 *        2) 线性化路标点
 *        3) 线性化IMU
 *        4) 计算边缘化先验误差
 * 
 * @tparam Scalar 
 * @tparam POSE_SIZE 
 * @param numerically_valid   输出参数，标记线性化过程中是否存在数值问题（如NaN）
 * @return Scalar 总的误差/代价值 (所有残差的加权平方和)
 */
template <typename Scalar, int POSE_SIZE>
Scalar LinearizationAbsQR<Scalar, POSE_SIZE>::linearizeProblem(
    bool* numerically_valid) {
  // 重置阻尼和缩放参数（可能在上一次迭代中被设置过）
  // 在LM算法中，每次线性化前需要重置阻尼项，后续会根据需要重新设置
  // reset damping and scaling (might be set from previous iteration)
  pose_damping_diagonal = 0;
  pose_damping_diagonal_sqrt = 0;
  marg_scaling = VecX();

  // ===================== 第一步：线性化相对位姿 =====================
  // 遍历所有观测关系，计算每对(host帧, target帧)之间的相对位姿及其雅可比矩阵
  // lmdb_ 是路标点数据库，getObservations() 返回所有帧对帧的观测关系
  //   - tcid_h: host帧的 TimeCamId（时间戳+相机ID）
  //   - target_map: 该host帧所观测到的所有target帧的映射
  // Linearize relative poses
  for (const auto& [tcid_h, target_map] : lmdb_.getObservations()) {
    // if (used_frames && used_frames->count(tcid_h.frame_id) == 0) continue;

    //   std::unordered_map<TimeCamId, std::map<TimeCamId, std::set<KeypointId>>> observations;
    // observations[tcid_h][tcid_t] 返回的是：所有以 tcid_h 为宿主帧、同时在 tcid_t 中也被观测到的路标点ID集合。
    for (const auto& [tcid_t, _] : target_map) {
      // tcid_t: target帧的 TimeCamId
      // key: (host, target) 帧对，用于索引相对位姿线性化结构体
      std::pair<TimeCamId, TimeCamId> key(tcid_h, tcid_t);
      RelPoseLin<Scalar>& rpl = relative_pose_lin.at(key);

      // 如果host帧和target帧不是同一帧，则需要计算相对位姿
      if (tcid_h != tcid_t) {
        // 获取host帧和target帧的位姿状态（包含线性化点信息）
        const PoseStateWithLin<Scalar>& state_h =
            estimator->getPoseStateWithLin(tcid_h.frame_id);
        const PoseStateWithLin<Scalar>& state_t =
            estimator->getPoseStateWithLin(tcid_t.frame_id);

        // 在线性化点处计算相对位姿 T_t_h 及其对host和target位姿的雅可比矩阵
        // T_t_h = T_t_c_t * T_c_t_w * T_w_c_h * T_c_h_h
        //       = T_i_c[t]^{-1} * T_w_i[t]^{-1} * T_w_i[h] * T_i_c[h]
        // d_rel_d_h: 相对位姿对host帧位姿增量的雅可比 (6x6)
        // d_rel_d_t: 相对位姿对target帧位姿增量的雅可比 (6x6)
        // getPoseLin() 返回线性化点处的位姿（FEJ - First Estimate Jacobian）
        // compute relative pose & Jacobians at linearization point
        Sophus::SE3<Scalar> T_t_h_sophus =
            computeRelPose(state_h.getPoseLin(), calib.T_i_c[tcid_h.cam_id],
                           state_t.getPoseLin(), calib.T_i_c[tcid_t.cam_id],
                           &rpl.d_rel_d_h, &rpl.d_rel_d_t);

        // FEJ（First Estimate Jacobian）策略：
        // 雅可比始终在线性化点处计算（保证零空间的一致性），
        // 但相对位姿的值需要用当前最新的状态估计来计算，以获得准确的残差。
        // 如果任一帧已经被线性化过（即当前估计 != 线性化点），
        // 则用当前状态重新计算相对位姿的值（不重新计算雅可比）
        // if either state is already linearized, then the current state
        // estimate is different from the linearization point, so recompute
        // the value (not Jacobian) again based on the current state.
        if (state_h.isLinearized() || state_t.isLinearized()) {
          T_t_h_sophus =
              computeRelPose(state_h.getPose(), calib.T_i_c[tcid_h.cam_id],
                             state_t.getPose(), calib.T_i_c[tcid_t.cam_id]);
        }

        // 将Sophus的SE3对象转为4x4齐次变换矩阵存储
        rpl.T_t_h = T_t_h_sophus.matrix();
      } else {
        // host帧和target帧是同一帧（同一时刻的不同相机，如双目的左右目）
        // 相对位姿为单位阵，雅可比为零（因为同一帧的位姿变化不影响帧内相对关系，
        // 帧内的相对关系由固定的外参 T_i_c 决定）
        rpl.T_t_h.setIdentity();
        rpl.d_rel_d_h.setZero();
        rpl.d_rel_d_t.setZero();
      }
    }
  }

  // ===================== 第二步：线性化路标点 =====================
  // 使用TBB并行归约来线性化所有路标点，同时累加误差值
  // Linearize landmarks
  size_t num_landmarks = landmark_blocks.size();

  // body lambda: TBB parallel_reduce 的工作函数
  // 对每个路标点块执行线性化，累加误差并检查数值有效性
  // error_valid.first: 累积的重投影误差
  // error_valid.second: 是否所有路标点线性化都数值有效（无NaN/Inf）
  auto body = [&](const tbb::blocked_range<size_t>& range,
                  std::pair<Scalar, bool> error_valid) {
    for (size_t r = range.begin(); r != range.end(); ++r) {
      // linearizeLandmark(): 对单个路标点进行线性化
      // 计算该路标点所有观测的重投影残差、雅可比矩阵，返回该路标点的误差贡献
      error_valid.first += landmark_blocks[r]->linearizeLandmark();
      // 检查是否有数值失败（如路标点深度为负、投影到图像外等）
      error_valid.second =
          error_valid.second && !landmark_blocks[r]->isNumericalFailure();
    }
    return error_valid;
  };

  // parallel_reduce 的初始值：误差为0，数值有效性为true
  std::pair<Scalar, bool> initial_value = {0.0, true};
  // join lambda: 合并两个子任务的结果（误差求和，有效性取逻辑与）
  auto join = [](auto p1, auto p2) {
    p1.first += p2.first;
    p1.second = p1.second && p2.second;
    return p1;
  };

  // 使用TBB并行归约，将所有路标点的线性化工作分配到多个线程
  // TBB会自动将 [0, num_landmarks) 范围划分为多个子块并行执行
  tbb::blocked_range<size_t> range(0, num_landmarks);
  auto reduction_res = tbb::parallel_reduce(range, initial_value, body, join);

  // 将数值有效性结果通过输出参数返回给调用者
  if (numerically_valid) *numerically_valid = reduction_res.second;

  // ===================== 第三步：线性化IMU预积分因子 =====================
  // 如果存在IMU线性化数据（VIO模式下），则对每个IMU预积分块进行线性化
  // 并将IMU残差的误差累加到总误差中
  if (imu_lin_data) {
    for (auto& imu_block : imu_blocks) {
      reduction_res.first += imu_block->linearizeImu(estimator->frame_states);
    }
  }

  // ===================== 第四步：计算边缘化先验误差 =====================
  // 如果存在边缘化先验数据，计算边缘化先验的误差贡献
  // 边缘化先验来自于之前被边缘化掉的帧和路标点，以先验因子的形式
  // 约束当前滑动窗口中的状态，防止信息丢失
  if (marg_lin_data) {
    Scalar marg_prior_error;
    estimator->computeMargPriorError(*marg_lin_data, marg_prior_error);
    reduction_res.first += marg_prior_error;
  }

  // 返回总误差 = 视觉重投影误差 + IMU预积分误差 + 边缘化先验误差
  return reduction_res.first;
}

template <typename Scalar, int POSE_SIZE>
void LinearizationAbsQR<Scalar, POSE_SIZE>::performQR() {
  auto body = [&](const tbb::blocked_range<size_t>& range) {
    for (size_t r = range.begin(); r != range.end(); ++r) {
      landmark_blocks[r]->performQR();
    }
  };

  tbb::blocked_range<size_t> range(0, landmark_block_idx.size());
  tbb::parallel_for(range, body);
}

template <typename Scalar, int POSE_SIZE>
void LinearizationAbsQR<Scalar, POSE_SIZE>::setPoseDamping(
    const Scalar lambda) {
  BASALT_ASSERT(lambda >= 0);

  pose_damping_diagonal = lambda;
  pose_damping_diagonal_sqrt = std::sqrt(lambda);
}

template <typename Scalar, int POSE_SIZE>
Scalar LinearizationAbsQR<Scalar, POSE_SIZE>::backSubstitute(
    const VecX& pose_inc) {
  BASALT_ASSERT(pose_inc.size() == signed_cast(aom.total_size));

  auto body = [&](const tbb::blocked_range<size_t>& range, Scalar l_diff) {
    for (size_t r = range.begin(); r != range.end(); ++r) {
      landmark_blocks[r]->backSubstitute(pose_inc, l_diff);
    }
    return l_diff;
  };

  tbb::blocked_range<size_t> range(0, landmark_block_idx.size());
  Scalar l_diff =
      tbb::parallel_reduce(range, Scalar(0), body, std::plus<Scalar>());

  if (imu_lin_data) {
    for (auto& imu_block : imu_blocks) {
      imu_block->backSubstitute(pose_inc, l_diff);
    }
  }

  if (marg_lin_data) {
    size_t marg_size = marg_lin_data->H.cols();
    VecX pose_inc_marg = pose_inc.head(marg_size);

    l_diff += estimator->computeMargPriorModelCostChange(
        *marg_lin_data, marg_scaling, pose_inc_marg);
  }

  return l_diff;
}

template <typename Scalar, int POSE_SIZE>
typename LinearizationAbsQR<Scalar, POSE_SIZE>::VecX
LinearizationAbsQR<Scalar, POSE_SIZE>::getJp_diag2() const {
  // TODO: group relative by host frame

  struct Reductor {
    Reductor(size_t num_rows,
             const std::vector<LandmarkBlockPtr>& landmark_blocks)
        : num_rows_(num_rows), landmark_blocks_(landmark_blocks) {
      res_.setZero(num_rows);
    }

    void operator()(const tbb::blocked_range<size_t>& range) {
      for (size_t r = range.begin(); r != range.end(); ++r) {
        const auto& lb = landmark_blocks_[r];
        lb->addJp_diag2(res_);
      }
    }

    Reductor(Reductor& a, tbb::split)
        : num_rows_(a.num_rows_), landmark_blocks_(a.landmark_blocks_) {
      res_.setZero(num_rows_);
    };

    inline void join(const Reductor& b) { res_ += b.res_; }

    size_t num_rows_;
    const std::vector<LandmarkBlockPtr>& landmark_blocks_;
    VecX res_;
  };

  Reductor r(aom.total_size, landmark_blocks);

  tbb::blocked_range<size_t> range(0, landmark_block_idx.size());
  tbb::parallel_reduce(range, r);
  // r(range);

  if (imu_lin_data) {
    for (auto& imu_block : imu_blocks) {
      imu_block->addJp_diag2(r.res_);
    }
  }

  // TODO: We don't include pose damping here, b/c we use this to compute
  // jacobian scale. Make sure this is clear in the the usage, possibly rename
  // to reflect this, or add assert such that it fails when pose damping is
  // set.

  // Note: ignore damping here

  // Add marginalization prior
  // Asumes marginalization part is in the head. Check for this is located
  // outside
  if (marg_lin_data) {
    size_t marg_size = marg_lin_data->H.cols();
    if (marg_scaling.rows() > 0) {
      r.res_.head(marg_size) += (marg_lin_data->H * marg_scaling.asDiagonal())
                                    .colwise()
                                    .squaredNorm();
    } else {
      r.res_.head(marg_size) += marg_lin_data->H.colwise().squaredNorm();
    }
  }

  return r.res_;
}

template <typename Scalar, int POSE_SIZE>
void LinearizationAbsQR<Scalar, POSE_SIZE>::scaleJl_cols() {
  auto body = [&](const tbb::blocked_range<size_t>& range) {
    for (size_t r = range.begin(); r != range.end(); ++r) {
      landmark_blocks[r]->scaleJl_cols();
    }
  };

  tbb::blocked_range<size_t> range(0, landmark_block_idx.size());
  tbb::parallel_for(range, body);
}

template <typename Scalar, int POSE_SIZE>
void LinearizationAbsQR<Scalar, POSE_SIZE>::scaleJp_cols(
    const VecX& jacobian_scaling) {
  //    auto body = [&](const tbb::blocked_range<size_t>& range) {
  //      for (size_t r = range.begin(); r != range.end(); ++r) {
  //        landmark_blocks[r]->scaleJp_cols(jacobian_scaling);
  //      }
  //    };

  //    tbb::blocked_range<size_t> range(0, landmark_block_idx.size());
  //    tbb::parallel_for(range, body);

  if (true) {
    // In case of absolute poses, we scale Jp in the LMB.

    auto body = [&](const tbb::blocked_range<size_t>& range) {
      for (size_t r = range.begin(); r != range.end(); ++r) {
        landmark_blocks[r]->scaleJp_cols(jacobian_scaling);
      }
    };

    tbb::blocked_range<size_t> range(0, landmark_block_idx.size());
    tbb::parallel_for(range, body);
  } else {
    // In case LMB use relative poses we cannot directly scale the relative pose
    // Jacobians. We have
    //
    //     Jp * diag(S) = Jp_rel * J_rel_to_abs * diag(S)
    //
    // so instead we scale the rel-to-abs jacobians.
    //
    // Note that since we do perform operations like J^T * J on the relative
    // pose Jacobians, we should maybe consider additional scaling like
    //
    //     (Jp_rel * diag(S_rel)) * (diag(S_rel)^-1 * J_rel_to_abs * diag(S)),
    //
    // but this might be only relevant if we do something more sensitive like
    // also include camera intrinsics in the optimization.

    for (auto& [k, v] : relative_pose_lin) {
      size_t h_idx = aom.abs_order_map.at(k.first.frame_id).first;
      size_t t_idx = aom.abs_order_map.at(k.second.frame_id).first;

      v.d_rel_d_h *=
          jacobian_scaling.template segment<POSE_SIZE>(h_idx).asDiagonal();

      v.d_rel_d_t *=
          jacobian_scaling.template segment<POSE_SIZE>(t_idx).asDiagonal();
    }
  }

  if (imu_lin_data) {
    for (auto& imu_block : imu_blocks) {
      imu_block->scaleJp_cols(jacobian_scaling);
    }
  }

  // Add marginalization scaling
  if (marg_lin_data) {
    // We are only supposed to apply the scaling once.
    BASALT_ASSERT(marg_scaling.size() == 0);

    size_t marg_size = marg_lin_data->H.cols();
    marg_scaling = jacobian_scaling.head(marg_size);
  }
}

template <typename Scalar, int POSE_SIZE>
void LinearizationAbsQR<Scalar, POSE_SIZE>::setLandmarkDamping(Scalar lambda) {
  auto body = [&](const tbb::blocked_range<size_t>& range) {
    for (size_t r = range.begin(); r != range.end(); ++r) {
      landmark_blocks[r]->setLandmarkDamping(lambda);
    }
  };

  tbb::blocked_range<size_t> range(0, landmark_block_idx.size());
  tbb::parallel_for(range, body);
}

template <typename Scalar, int POSE_SIZE>
void LinearizationAbsQR<Scalar, POSE_SIZE>::get_dense_Q2Jp_Q2r(
    MatX& Q2Jp, VecX& Q2r) const {
  size_t total_size = num_rows_Q2r;
  size_t poses_size = aom.total_size;

  size_t lm_start_idx = 0;

  // Space for IMU data if present
  size_t imu_start_idx = total_size;
  if (imu_lin_data) {
    total_size += imu_lin_data->imu_meas.size() * POSE_VEL_BIAS_SIZE;
  }

  // Space for damping if present
  size_t damping_start_idx = total_size;
  if (hasPoseDamping()) {
    total_size += poses_size;
  }

  // Space for marg-prior if present
  size_t marg_start_idx = total_size;
  if (marg_lin_data) total_size += marg_lin_data->H.rows();

  Q2Jp.setZero(total_size, poses_size);
  Q2r.setZero(total_size);

  auto body = [&](const tbb::blocked_range<size_t>& range) {
    for (size_t r = range.begin(); r != range.end(); ++r) {
      const auto& lb = landmark_blocks[r];
      lb->get_dense_Q2Jp_Q2r(Q2Jp, Q2r, lm_start_idx + landmark_block_idx[r]);
    }
  };

  // go over all host frames
  tbb::blocked_range<size_t> range(0, landmark_block_idx.size());
  tbb::parallel_for(range, body);
  // body(range);

  if (imu_lin_data) {
    size_t start_idx = imu_start_idx;
    for (const auto& imu_block : imu_blocks) {
      imu_block->add_dense_Q2Jp_Q2r(Q2Jp, Q2r, start_idx);
      start_idx += POSE_VEL_BIAS_SIZE;
    }
  }

  // Add damping
  get_dense_Q2Jp_Q2r_pose_damping(Q2Jp, damping_start_idx);

  // Add marginalization
  get_dense_Q2Jp_Q2r_marg_prior(Q2Jp, Q2r, marg_start_idx);
}

/**
 * @brief 构建稠密的Hessian矩阵H和残差向量b（用于LM算法求解线性系统 H*dx = b），针对 VIO 优化中最耗时的 “构建稠密 H/b” 步骤做了并行优化
 * 
 *  模块合并顺序：
 *   - 先并行累加视觉残差的 H/b（占比最大）
 *   - 再加入IMU 残差的 H/b（运动约束）
 *   - 接着加入LM 阻尼项（数值稳定）
 *   - 最后加入边缘化先验的 H/b（窗口约束）
 * 
 * @tparam Scalar     数值类型（float/double）
 * @tparam POSE_SIZE  位姿状态维度（默认值: 6）
 * @param H           Hessian矩阵
 * @param b           残差向量b
 */
template <typename Scalar, int POSE_SIZE>
void LinearizationAbsQR<Scalar, POSE_SIZE>::get_dense_H_b(MatX& H,
                                                          VecX& b) const {
  
  // -------------------------- 并行归约器定义 --------------------------
  // 嵌套结构体Reductor：用于TBB并行计算，合并所有特征点块的H和b
  // 核心作用：将多个特征点块的H/b累加为全局稠密H/b，支持多线程并行
  struct Reductor {
    /**
     * @brief 构造函数：初始化归约器
     *        
     * @param opt_size            优化变量总维度（所有相机/IMU状态的维度和）
     * @param landmark_blocks     所有特征点块的指针列表（每个块对应一个特征点的残差/雅可比）
     */
    Reductor(size_t opt_size,
             const std::vector<LandmarkBlockPtr>& landmark_blocks)
        : opt_size_(opt_size), landmark_blocks_(landmark_blocks) {
      H_.setZero(opt_size_, opt_size_);     // 初始化H为零矩阵（opt_size x opt_size）
      b_.setZero(opt_size_);                // 初始化b为零向量（opt_size x 1）
    }

    /**
     * @brief 重载()运算符：单线程处理一个特征点块范围，累加H/b
     * 
     * @param range TBB分配的并行处理范围（[begin, end)）
     */
    void operator()(const tbb::blocked_range<size_t>& range) {
      for (size_t r = range.begin(); r != range.end(); ++r) {
        auto& lb = landmark_blocks_[r];   // 获取第r个特征点块

        // 累加该特征点块的H/b到当前线程的局部H_/b_（视觉残差对应的H/b）
        lb->add_dense_H_b(H_, b_);        
      }
    }

    /**
     * @brief 分裂构造函数：TBB并行归约时，分裂当前归约器为两个（用于多线程）
     * 
     * @param a 原归约器，tbb::split: TBB分裂标记（无实际值）
     */
    Reductor(Reductor& a, tbb::split)
        : opt_size_(a.opt_size_), landmark_blocks_(a.landmark_blocks_) {
      H_.setZero(opt_size_, opt_size_);   // 新线程初始化局部H为零
      b_.setZero(opt_size_);              // 新线程初始化局部b为零
    };

    // 合并函数：将另一个归约器的H/b合并到当前归约器（TBB并行归约的最后一步）
    inline void join(Reductor& b) {
      H_ += b.H_;   // 累加其他线程的H矩阵
      b_ += b.b_;   // 累加其他线程的b向量
    }

    // 成员变量
    size_t opt_size_;                                       // 优化变量总维度
    const std::vector<LandmarkBlockPtr>& landmark_blocks_;  // 特征点块列表（只读）

    MatX H_;                                                // 当前线程的局部Hessian矩阵（稠密）
    VecX b_;                                                // 当前线程的局部残差向量（稠密）
  };

  // -------------------------- 并行构建视觉残差的H/b --------------------------

  // 1. 获取优化变量总维度（所有相机/IMU状态的维度和，由AbsOrderMap统计）
  size_t opt_size = aom.total_size;

  // 2. 初始化归约器：传入优化变量维度和特征点块列表
  Reductor r(opt_size, landmark_blocks);

  // go over all host frames
  // 3. 定义TBB并行范围：遍历所有特征点块索引（0 ~ landmark_block_idx.size()-1）
  tbb::blocked_range<size_t> range(0, landmark_block_idx.size());

  // 4. TBB并行归约：多线程累加所有特征点块的H/b到归约器r中
  // 核心：每个线程处理一部分特征点块，最后自动调用join合并所有线程的H/b
  tbb::parallel_reduce(range, r);

  // -------------------------- 合并其他模块的H/b --------------------------

  // Add imu
  // 1. 加入IMU残差对应的H/b（IMU预积分残差的Hessian和残差向量）
  add_dense_H_b_imu(r.H_, r.b_);

  // Add damping
  // 2. 加入位姿阻尼项（LM算法的核心：H += lambda*I，增强数值稳定性）
  add_dense_H_b_pose_damping(r.H_);

  // Add marginalization
  // 3. 加入边缘化先验残差对应的H/b（旧帧边缘化后得到的先验约束）
  add_dense_H_b_marg_prior(r.H_, r.b_);

  // -------------------------- 输出结果 --------------------------
  // 将归约器中的H/b移动到输出参数（std::move避免拷贝，提升效率）
  H = std::move(r.H_);
  b = std::move(r.b_);
}

template <typename Scalar, int POSE_SIZE>
void LinearizationAbsQR<Scalar, POSE_SIZE>::get_dense_Q2Jp_Q2r_pose_damping(
    MatX& Q2Jp, size_t start_idx) const {
  size_t poses_size = num_cameras * POSE_SIZE;
  if (hasPoseDamping()) {
    Q2Jp.template block(start_idx, 0, poses_size, poses_size)
        .diagonal()
        .array() = pose_damping_diagonal_sqrt;
  }
}

template <typename Scalar, int POSE_SIZE>
void LinearizationAbsQR<Scalar, POSE_SIZE>::get_dense_Q2Jp_Q2r_marg_prior(
    MatX& Q2Jp, VecX& Q2r, size_t start_idx) const {
  if (!marg_lin_data) return;

  BASALT_ASSERT(marg_lin_data->is_sqrt);

  size_t marg_rows = marg_lin_data->H.rows();
  size_t marg_cols = marg_lin_data->H.cols();

  VecX delta;
  estimator->computeDelta(marg_lin_data->order, delta);

  if (marg_scaling.rows() > 0) {
    Q2Jp.template block(start_idx, 0, marg_rows, marg_cols) =
        marg_lin_data->H * marg_scaling.asDiagonal();
  } else {
    Q2Jp.template block(start_idx, 0, marg_rows, marg_cols) = marg_lin_data->H;
  }

  Q2r.template segment(start_idx, marg_rows) =
      marg_lin_data->H * delta + marg_lin_data->b;
}

template <typename Scalar, int POSE_SIZE>
void LinearizationAbsQR<Scalar, POSE_SIZE>::add_dense_H_b_pose_damping(
    MatX& H) const {
  if (hasPoseDamping()) {
    H.diagonal().array() += pose_damping_diagonal;
  }
}

template <typename Scalar, int POSE_SIZE>
void LinearizationAbsQR<Scalar, POSE_SIZE>::add_dense_H_b_marg_prior(
    MatX& H, VecX& b) const {
  if (!marg_lin_data) return;

  // Scaling not supported ATM
  BASALT_ASSERT(marg_scaling.rows() == 0);

  Scalar marg_prior_error;
  estimator->linearizeMargPrior(*marg_lin_data, aom, H, b, marg_prior_error);

  //  size_t marg_size = marg_lin_data->H.cols();

  //  VecX delta;
  //  estimator->computeDelta(marg_lin_data->order, delta);

  //  if (marg_scaling.rows() > 0) {
  //    H.topLeftCorner(marg_size, marg_size) +=
  //        marg_scaling.asDiagonal() * marg_lin_data->H.transpose() *
  //        marg_lin_data->H * marg_scaling.asDiagonal();

  //    b.head(marg_size) += marg_scaling.asDiagonal() *
  //                         marg_lin_data->H.transpose() *
  //                         (marg_lin_data->H * delta + marg_lin_data->b);

  //  } else {
  //    H.topLeftCorner(marg_size, marg_size) +=
  //        marg_lin_data->H.transpose() * marg_lin_data->H;

  //    b.head(marg_size) += marg_lin_data->H.transpose() *
  //                         (marg_lin_data->H * delta + marg_lin_data->b);
  //  }
}

template <typename Scalar, int POSE_SIZE>
void LinearizationAbsQR<Scalar, POSE_SIZE>::add_dense_H_b_imu(
    DenseAccumulator<Scalar>& accum) const {
  if (!imu_lin_data) return;

  for (const auto& imu_block : imu_blocks) {
    imu_block->add_dense_H_b(accum);
  }
}

template <typename Scalar, int POSE_SIZE>
void LinearizationAbsQR<Scalar, POSE_SIZE>::add_dense_H_b_imu(MatX& H,
                                                              VecX& b) const {
  if (!imu_lin_data) return;

  // workaround: create an accumulator here, to avoid implementing the
  // add_dense_H_b(H, b) overload in ImuBlock
  DenseAccumulator<Scalar> accum;
  accum.reset(b.size());

  for (const auto& imu_block : imu_blocks) {
    imu_block->add_dense_H_b(accum);
  }

  H += accum.getH();
  b += accum.getB();
}

// //////////////////////////////////////////////////////////////////
// instatiate templates

#ifdef BASALT_INSTANTIATIONS_DOUBLE
// Scalar=double, POSE_SIZE=6
template class LinearizationAbsQR<double, 6>;
#endif

#ifdef BASALT_INSTANTIATIONS_FLOAT
// Scalar=float, POSE_SIZE=6
template class LinearizationAbsQR<float, 6>;
#endif

}  // namespace basalt