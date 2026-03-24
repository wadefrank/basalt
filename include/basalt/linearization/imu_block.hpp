#pragma once

#include <basalt/imu/preintegration.h>
#include <basalt/optimization/accumulator.h>
#include <basalt/utils/imu_types.h>

namespace basalt {

template <class Scalar_>
class ImuBlock {
 public:
  using Scalar = Scalar_;

  using Vec3 = Eigen::Matrix<Scalar, 3, 1>;
  using VecX = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

  using MatX = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

  ImuBlock(const IntegratedImuMeasurement<Scalar>* meas,
           const ImuLinData<Scalar>* imu_lin_data, const AbsOrderMap& aom)
      : imu_meas(meas), imu_lin_data(imu_lin_data), aom(aom) {
    Jp.resize(POSE_VEL_BIAS_SIZE, 2 * POSE_VEL_BIAS_SIZE);
    r.resize(POSE_VEL_BIAS_SIZE);
  }

  /**
   * @brief IMU残差线性化函数。
   *        计算两个相邻关键帧之间的IMU预积分残差，并构建线性化后的雅可比矩阵Jp和残差向量r
   * 
   * @note 整体结构：Jp 是 15×30 的矩阵，r 是 15×1 的向量
   *        - 前9行：IMU预积分残差（旋转3 + 速度3 + 位移3）关于起始帧和终止帧状态的雅可比
   *        - 第9~11行：陀螺仪bias随机游走约束
   *        - 第12~14行：加速度计bias随机游走约束
   *        - 前15列对应起始帧状态（位姿6 + 速度3 + 陀螺仪bias3 + 加速度计bias3）
   *        - 后15列对应终止帧状态（同上）
   * 
   * @param frame_states    状态
   * 
   * @return Scalar 加权残差的总代价（IMU预积分误差 + 陀螺仪bias误差 + 加速度计bias误差）
   */
  Scalar linearizeImu(
      const Eigen::aligned_map<int64_t, PoseVelBiasStateWithLin<Scalar>>&
          frame_states) {
    
    // 将雅可比矩阵和残差向量清零，准备重新计算
    Jp.setZero();
    r.setZero();

    // 获取IMU预积分的起始时间戳和终止时间戳（纳秒）
    const int64_t start_t = imu_meas->get_start_t_ns();
    const int64_t end_t = imu_meas->get_start_t_ns() + imu_meas->get_dt_ns();

    // Jp矩阵中起始帧和终止帧状态变量的列偏移
    // 起始帧占据列 [0, 15)，终止帧占据列 [15, 30)
    // 每个帧的状态维度为 POSE_VEL_BIAS_SIZE = 15
    const size_t start_idx = 0;
    const size_t end_idx = POSE_VEL_BIAS_SIZE;

    // 从滑动窗口中取出起始帧和终止帧的状态（包含位姿、速度、bias）
    PoseVelBiasStateWithLin<Scalar> start_state = frame_states.at(start_t);
    PoseVelBiasStateWithLin<Scalar> end_state = frame_states.at(end_t);

    // 声明IMU残差对各状态的雅可比矩阵：
    //   d_res_d_start: 9×9, 残差对起始帧位姿+速度（9维）的雅可比
    //   d_res_d_end:   9×9, 残差对终止帧位姿+速度（9维）的雅可比
    //   d_res_d_bg:    9×3, 残差对陀螺仪bias的雅可比
    //   d_res_d_ba:    9×3, 残差对加速度计bias的雅可比
    typename IntegratedImuMeasurement<Scalar>::MatNN d_res_d_start, d_res_d_end;
    typename IntegratedImuMeasurement<Scalar>::MatN3 d_res_d_bg, d_res_d_ba;

    // 在线性化点（getStateLin）处计算IMU预积分残差及其雅可比
    // 使用线性化点计算雅可比是 FEJ（First Estimates Jacobian）策略的核心：
    //   雅可比始终在首次线性化的状态点处求值，以保证系统的一致性/可观性
    typename PoseVelState<Scalar>::VecN res = imu_meas->residual(
        start_state.getStateLin(), imu_lin_data->g, end_state.getStateLin(),
        start_state.getStateLin().bias_gyro,
        start_state.getStateLin().bias_accel, &d_res_d_start, &d_res_d_end,
        &d_res_d_bg, &d_res_d_ba);

    // FEJ策略的第二部分：如果状态已经被边缘化（isLinearized），
    // 则残差值需要在当前最新状态估计（getState）处重新计算，
    // 但雅可比仍然保持在线性化点处的值不变。
    // 这样做的目的是：雅可比固定保证一致性，残差更新保证优化方向正确。
    if (start_state.isLinearized() || end_state.isLinearized()) {
      res = imu_meas->residual(
          start_state.getState(), imu_lin_data->g, end_state.getState(),
          start_state.getState().bias_gyro, start_state.getState().bias_accel);
    }

    // ==================== 第一部分：IMU预积分残差 ====================
    // 计算加权IMU误差: e = 0.5 * ||Σ^{-1/2} * r||^2
    // 其中 get_sqrt_cov_inv() 返回预积分协方差的逆平方根 Σ^{-1/2}（信息矩阵的平方根）
    // 乘以 Σ^{-1/2} 即对残差进行"白化"（whitening），将马氏距离转化为欧氏距离
    Scalar imu_error =
        Scalar(0.5) * (imu_meas->get_sqrt_cov_inv() * res).squaredNorm();

    // 构建白化后的雅可比矩阵 Jp = Σ^{-1/2} * J
    // 白化后 H = Jp^T * Jp = J^T * Σ^{-1} * J，无需再显式处理信息矩阵

    // 残差对起始帧位姿+速度（9维）的白化雅可比，填入Jp的 [0:9, 0:9] 块
    Jp.template block<9, 9>(0, start_idx) =
        imu_meas->get_sqrt_cov_inv() * d_res_d_start;
    // 残差对终止帧位姿+速度（9维）的白化雅可比，填入Jp的 [0:9, 15:24] 块
    Jp.template block<9, 9>(0, end_idx) =
        imu_meas->get_sqrt_cov_inv() * d_res_d_end;

    // 残差对起始帧陀螺仪bias（3维）的白化雅可比，填入Jp的 [0:9, 9:12] 块
    Jp.template block<9, 3>(0, start_idx + 9) =
        imu_meas->get_sqrt_cov_inv() * d_res_d_bg;
    // 残差对起始帧加速度计bias（3维）的白化雅可比，填入Jp的 [0:9, 12:15] 块
    Jp.template block<9, 3>(0, start_idx + 12) =
        imu_meas->get_sqrt_cov_inv() * d_res_d_ba;

    // 白化后的IMU残差向量，填入r的前9维
    r.template segment<9>(0) = imu_meas->get_sqrt_cov_inv() * res;

    // ==================== 第二部分：Bias随机游走约束 ====================
    // IMU的bias建模为随机游走过程: b_{k+1} = b_k + w_k, w_k ~ N(0, Q*dt)
    // 因此相邻帧bias之差的先验约束为: (b_start - b_end) ~ N(0, Q*dt)
    // 对应的权重（信息矩阵的平方根）与 1/sqrt(dt) 成正比：
    //   时间间隔越长，bias漂移越大，约束越弱

    // 将时间间隔从纳秒转换为秒
    Scalar dt = imu_meas->get_dt_ns() * Scalar(1e-9);

    // --- 陀螺仪bias随机游走 ---
    // 计算陀螺仪bias的白化权重: σ_bg^{-1/2} / sqrt(dt)
    Vec3 gyro_bias_weight_dt =
        imu_lin_data->gyro_bias_weight_sqrt / std::sqrt(dt);
    // 陀螺仪bias残差: r_bg = b_gyro_start - b_gyro_end
    Vec3 res_bg =
        start_state.getState().bias_gyro - end_state.getState().bias_gyro;

    // 陀螺仪bias残差对起始帧gyro bias的雅可比 = +W（对角权重矩阵）
    // 填入Jp的 [9:12, 9:12] 块
    Jp.template block<3, 3>(9, start_idx + 9) =
        gyro_bias_weight_dt.asDiagonal();
    // 陀螺仪bias残差对终止帧gyro bias的雅可比 = -W
    // 填入Jp的 [9:12, 24:27] 块
    Jp.template block<3, 3>(9, end_idx + 9) =
        (-gyro_bias_weight_dt).asDiagonal();

    // 白化后的陀螺仪bias残差，累加到r的第9~11维
    r.template segment<3>(9) += gyro_bias_weight_dt.asDiagonal() * res_bg;

    // 陀螺仪bias的加权误差
    Scalar bg_error =
        Scalar(0.5) * (gyro_bias_weight_dt.asDiagonal() * res_bg).squaredNorm();

    // --- 加速度计bias随机游走 ---
    // 计算加速度计bias的白化权重: σ_ba^{-1/2} / sqrt(dt)
    Vec3 accel_bias_weight_dt =
        imu_lin_data->accel_bias_weight_sqrt / std::sqrt(dt);
    // 加速度计bias残差: r_ba = b_accel_start - b_accel_end
    Vec3 res_ba =
        start_state.getState().bias_accel - end_state.getState().bias_accel;

    // 加速度计bias残差对起始帧accel bias的雅可比 = +W
    // 填入Jp的 [12:15, 12:15] 块
    Jp.template block<3, 3>(12, start_idx + 12) =
        accel_bias_weight_dt.asDiagonal();
    // 加速度计bias残差对终止帧accel bias的雅可比 = -W
    // 填入Jp的 [12:15, 27:30] 块
    Jp.template block<3, 3>(12, end_idx + 12) =
        (-accel_bias_weight_dt).asDiagonal();

    // 白化后的加速度计bias残差，累加到r的第12~14维
    r.template segment<3>(12) += accel_bias_weight_dt.asDiagonal() * res_ba;

    // 加速度计bias的加权误差
    Scalar ba_error =
        Scalar(0.5) *
        (accel_bias_weight_dt.asDiagonal() * res_ba).squaredNorm();

    // 返回三部分误差之和：
    //   imu_error: IMU预积分残差的马氏距离（旋转+速度+位移）
    //   bg_error:  陀螺仪bias随机游走约束的代价
    //   ba_error:  加速度计bias随机游走约束的代价
    return imu_error + bg_error + ba_error;
  }

  void add_dense_Q2Jp_Q2r(MatX& Q2Jp, VecX& Q2r, size_t row_start_idx) const {
    int64_t start_t = imu_meas->get_start_t_ns();
    int64_t end_t = imu_meas->get_start_t_ns() + imu_meas->get_dt_ns();

    const size_t start_idx = aom.abs_order_map.at(start_t).first;
    const size_t end_idx = aom.abs_order_map.at(end_t).first;

    Q2Jp.template block<POSE_VEL_BIAS_SIZE, POSE_VEL_BIAS_SIZE>(row_start_idx,
                                                                start_idx) +=
        Jp.template topLeftCorner<POSE_VEL_BIAS_SIZE, POSE_VEL_BIAS_SIZE>();

    Q2Jp.template block<POSE_VEL_BIAS_SIZE, POSE_VEL_BIAS_SIZE>(row_start_idx,
                                                                end_idx) +=
        Jp.template topRightCorner<POSE_VEL_BIAS_SIZE, POSE_VEL_BIAS_SIZE>();

    Q2r.template segment<POSE_VEL_BIAS_SIZE>(row_start_idx) += r;
  }

  void add_dense_H_b(DenseAccumulator<Scalar>& accum) const {
    int64_t start_t = imu_meas->get_start_t_ns();
    int64_t end_t = imu_meas->get_start_t_ns() + imu_meas->get_dt_ns();

    const size_t start_idx = aom.abs_order_map.at(start_t).first;
    const size_t end_idx = aom.abs_order_map.at(end_t).first;

    const MatX H = Jp.transpose() * Jp;
    const VecX b = Jp.transpose() * r;

    accum.template addH<POSE_VEL_BIAS_SIZE, POSE_VEL_BIAS_SIZE>(
        start_idx, start_idx,
        H.template block<POSE_VEL_BIAS_SIZE, POSE_VEL_BIAS_SIZE>(0, 0));

    accum.template addH<POSE_VEL_BIAS_SIZE, POSE_VEL_BIAS_SIZE>(
        end_idx, start_idx,
        H.template block<POSE_VEL_BIAS_SIZE, POSE_VEL_BIAS_SIZE>(
            POSE_VEL_BIAS_SIZE, 0));

    accum.template addH<POSE_VEL_BIAS_SIZE, POSE_VEL_BIAS_SIZE>(
        start_idx, end_idx,
        H.template block<POSE_VEL_BIAS_SIZE, POSE_VEL_BIAS_SIZE>(
            0, POSE_VEL_BIAS_SIZE));

    accum.template addH<POSE_VEL_BIAS_SIZE, POSE_VEL_BIAS_SIZE>(
        end_idx, end_idx,
        H.template block<POSE_VEL_BIAS_SIZE, POSE_VEL_BIAS_SIZE>(
            POSE_VEL_BIAS_SIZE, POSE_VEL_BIAS_SIZE));

    accum.template addB<POSE_VEL_BIAS_SIZE>(
        start_idx, b.template segment<POSE_VEL_BIAS_SIZE>(0));
    accum.template addB<POSE_VEL_BIAS_SIZE>(
        end_idx, b.template segment<POSE_VEL_BIAS_SIZE>(POSE_VEL_BIAS_SIZE));
  }

  void scaleJp_cols(const VecX& jacobian_scaling) {
    int64_t start_t = imu_meas->get_start_t_ns();
    int64_t end_t = imu_meas->get_start_t_ns() + imu_meas->get_dt_ns();

    const size_t start_idx = aom.abs_order_map.at(start_t).first;
    const size_t end_idx = aom.abs_order_map.at(end_t).first;

    Jp.template topLeftCorner<POSE_VEL_BIAS_SIZE, POSE_VEL_BIAS_SIZE>() *=
        jacobian_scaling.template segment<POSE_VEL_BIAS_SIZE>(start_idx)
            .asDiagonal();

    Jp.template topRightCorner<POSE_VEL_BIAS_SIZE, POSE_VEL_BIAS_SIZE>() *=
        jacobian_scaling.template segment<POSE_VEL_BIAS_SIZE>(end_idx)
            .asDiagonal();
  }

  void addJp_diag2(VecX& res) const {
    int64_t start_t = imu_meas->get_start_t_ns();
    int64_t end_t = imu_meas->get_start_t_ns() + imu_meas->get_dt_ns();

    const size_t start_idx = aom.abs_order_map.at(start_t).first;
    const size_t end_idx = aom.abs_order_map.at(end_t).first;

    res.template segment<POSE_VEL_BIAS_SIZE>(start_idx) +=
        Jp.template topLeftCorner<POSE_VEL_BIAS_SIZE, POSE_VEL_BIAS_SIZE>()
            .colwise()
            .squaredNorm();

    res.template segment<POSE_VEL_BIAS_SIZE>(end_idx) +=
        Jp.template topRightCorner<POSE_VEL_BIAS_SIZE, POSE_VEL_BIAS_SIZE>()
            .colwise()
            .squaredNorm();
  }

  void backSubstitute(const VecX& pose_inc, Scalar& l_diff) {
    int64_t start_t = imu_meas->get_start_t_ns();
    int64_t end_t = imu_meas->get_start_t_ns() + imu_meas->get_dt_ns();

    const size_t start_idx = aom.abs_order_map.at(start_t).first;
    const size_t end_idx = aom.abs_order_map.at(end_t).first;

    VecX pose_inc_reduced(2 * POSE_VEL_BIAS_SIZE);
    pose_inc_reduced.template head<POSE_VEL_BIAS_SIZE>() =
        pose_inc.template segment<POSE_VEL_BIAS_SIZE>(start_idx);
    pose_inc_reduced.template tail<POSE_VEL_BIAS_SIZE>() =
        pose_inc.template segment<POSE_VEL_BIAS_SIZE>(end_idx);

    // We want to compute the model cost change. The model function is
    //
    //     L(inc) = F(x) + incT JT r + 0.5 incT JT J inc
    //
    // and thus the expect decrease in cost for the computed increment is
    //
    //     l_diff = L(0) - L(inc)
    //            = - incT JT r - 0.5 incT JT J inc.
    //            = - incT JT (r + 0.5 J inc)
    //            = - (J inc)T (r + 0.5 (J inc))

    VecX Jinc = Jp * pose_inc_reduced;
    l_diff -= Jinc.transpose() * (Scalar(0.5) * Jinc + r);
  }

 protected:
  std::array<FrameId, 2> frame_ids;
  MatX Jp;
  VecX r;

  const IntegratedImuMeasurement<Scalar>* imu_meas;
  const ImuLinData<Scalar>* imu_lin_data;
  const AbsOrderMap& aom;
};  // namespace basalt

}  // namespace basalt
