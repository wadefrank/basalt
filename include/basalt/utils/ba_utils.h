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

/**
 * @brief 计算从host相机坐标系到target相机坐标系的相对位姿，并可选地计算其对
 *        host和target IMU位姿的雅可比矩阵
 *
 * 相对位姿变换链：
 *   T_t_c_h_c = T_i_c_t^{-1} * T_w_i_t^{-1} * T_w_i_h * T_i_c_h
 *             = T_c_t_i_t * T_i_t_w * T_w_i_h * T_i_h_c_h
 *
 * 其中：
 *   T_w_i_h: host帧 IMU 在世界坐标系下的位姿
 *   T_i_c_h: host帧相机相对于IMU的外参（固定标定值）
 *   T_w_i_t: target帧 IMU 在世界坐标系下的位姿
 *   T_i_c_t: target帧相机相对于IMU的外参（固定标定值）
 *
 * Basalt 位姿扰动模型 (incPose):
 *   对 T_w_i = (R_w_i, p_w_i)，增量 δ = [δp; δφ] (6维) 定义为：
 *     - 平移：p_w_i → p_w_i + δp      （世界坐标系下的加法扰动）
 *     - 旋转：R_w_i → Exp(δφ) * R_w_i  （世界坐标系下的左乘扰动）
 *
 * 雅可比定义（对应 test_vio.cpp 的数值验证）：
 *   J * δ = Log(T_rel(δ) * T_rel_nominal^{-1})
 *   即：δ 映射到 T_rel 上的 **左扰动**
 *
 * @param T_w_i_h  host帧 IMU 位姿（世界系）
 * @param T_i_c_h  host帧 IMU→相机 外参
 * @param T_w_i_t  target帧 IMU 位姿（世界系）
 * @param T_i_c_t  target帧 IMU→相机 外参
 * @param d_rel_d_h  [输出] 相对位姿对 host IMU 位姿扰动的雅可比 (6×6)，可为nullptr
 * @param d_rel_d_t  [输出] 相对位姿对 target IMU 位姿扰动的雅可比 (6×6)，可为nullptr
 * @return T_t_c_h_c: target相机系到host相机系的相对变换
 */
template <class Scalar>
Sophus::SE3<Scalar> computeRelPose(
    const Sophus::SE3<Scalar>& T_w_i_h, const Sophus::SE3<Scalar>& T_i_c_h,
    const Sophus::SE3<Scalar>& T_w_i_t, const Sophus::SE3<Scalar>& T_i_c_t,
    Sophus::Matrix6<Scalar>* d_rel_d_h = nullptr,
    Sophus::Matrix6<Scalar>* d_rel_d_t = nullptr) {
  // tmp2 = T_i_c_t^{-1} = T_c_t_i_t（target相机系到target IMU系的变换）
  Sophus::SE3<Scalar> tmp2 = (T_i_c_t).inverse();

  // ---- 计算 T_t_i_h_i = T_w_i_t^{-1} * T_w_i_h ----
  // 这里显式分解旋转和平移分别计算，而不是直接用 SE3 的 inverse() * 运算。
  // 原因：Basalt 的扰动模型对旋转和平移分别定义（平移加法 + 旋转左乘），
  // 这种分解方式与后面雅可比推导中的扰动传播一一对应，便于理解和推导。
  Sophus::SE3<Scalar> T_t_i_h_i;
  // 旋转部分：R_i_t_i_h = R_w_i_t^T * R_w_i_h
  T_t_i_h_i.so3() = T_w_i_t.so3().inverse() * T_w_i_h.so3();
  // 平移部分：t_i_t_i_h = R_w_i_t^T * (p_w_i_h - p_w_i_t)
  T_t_i_h_i.translation() =
      T_w_i_t.so3().inverse() * (T_w_i_h.translation() - T_w_i_t.translation());

  // tmp = T_c_t_i_t * T_i_t_i_h = T_c_t_i_h（target相机系到host IMU系）
  Sophus::SE3<Scalar> tmp = tmp2 * T_t_i_h_i;
  // res = T_c_t_i_h * T_i_h_c_h = T_c_t_c_h（最终的相对位姿：target相机系到host相机系）
  Sophus::SE3<Scalar> res = tmp * T_i_c_h;

  // ===================== d_rel_d_h: 相对位姿对 host 位姿的雅可比 =====================
  //
  // 推导思路：
  //   1) 对 T_w_i_h 施加扰动 δ_h = [δp; δφ]：
  //        R_w_i_h → Exp(δφ) * R_w_i_h,  p_w_i_h → p_w_i_h + δp
  //
  //   2) T_t_i_h_i 变为 T_t_i_h_i * Exp(RR * δ_h)（右扰动）
  //      其中 RR = blkdiag(R_i_h_w, R_i_h_w) 将世界坐标系扰动转换到 host IMU 体坐标系
  //      （因为 SE3 的右扰动 Exp(ξ) 中 ξ 定义在体坐标系）
  //
  //   3) 从 T_t_i_h_i 传播到 T_rel：
  //        tmp(δ) = tmp2 * T_t_i_h_i * Exp(RR*δ) = tmp * Exp(RR*δ)
  //        res(δ) = tmp * Exp(RR*δ) * T_i_c_h
  //
  //   4) 表示为 T_rel 的左扰动：
  //        res(δ) * res^{-1} = tmp * Exp(RR*δ) * tmp^{-1} = Exp(Ad(tmp) * RR * δ)
  //
  //   5) 因此：d_rel_d_h = Ad(tmp) * RR
  //
  if (d_rel_d_h) {
    // R = R_w_i_h^{-1} = R_i_h_w
    Sophus::Matrix3<Scalar> R = T_w_i_h.so3().inverse().matrix();

    // RR = blkdiag(R_i_h_w, R_i_h_w)
    // 作用：将 δ = [δp; δφ]（世界系）转换为 host IMU 体坐标系下的 se3 扰动
    Sophus::Matrix6<Scalar> RR;
    RR.setZero();
    RR.template topLeftCorner<3, 3>() = R;
    RR.template bottomRightCorner<3, 3>() = R;

    // d_rel_d_h = Ad(T_c_t_i_h) * RR
    // Ad(tmp) 将 tmp（T_c_t_i_h）的右扰动传播为 res 的左扰动
    *d_rel_d_h = tmp.Adj() * RR;
  }

  // ===================== d_rel_d_t: 相对位姿对 target 位姿的雅可比 =====================
  //
  // 推导思路：
  //   1) 对 T_w_i_t 施加扰动 δ_t = [δp; δφ]：
  //        R_w_i_t → Exp(δφ) * R_w_i_t,  p_w_i_t → p_w_i_t + δp
  //
  //   2) T_w_i_t^{-1} 变化导致 T_t_i_h_i 产生左扰动 Exp(-RR * δ_t) * T_t_i_h_i
  //      其中 RR = blkdiag(R_i_t_w, R_i_t_w)，负号因为扰动作用在"分母"（逆变换）上
  //
  //   3) 传播到 res：
  //        tmp(δ) = tmp2 * Exp(-RR*δ) * T_t_i_h_i = Exp(-Ad(tmp2)*RR*δ) * tmp
  //        res(δ) = Exp(-Ad(tmp2)*RR*δ) * tmp * T_i_c_h = Exp(-Ad(tmp2)*RR*δ) * res
  //
  //   4) 因此：d_rel_d_t = -Ad(tmp2) * RR
  //
  if (d_rel_d_t) {
    // R = R_w_i_t^{-1} = R_i_t_w
    Sophus::Matrix3<Scalar> R = T_w_i_t.so3().inverse().matrix();

    // RR = blkdiag(R_i_t_w, R_i_t_w)
    // 作用：将 δ = [δp; δφ]（世界系）转换为 target IMU 体坐标系下的 se3 扰动
    Sophus::Matrix6<Scalar> RR;
    RR.setZero();
    RR.template topLeftCorner<3, 3>() = R;
    RR.template bottomRightCorner<3, 3>() = R;

    // d_rel_d_t = -Ad(T_c_t_i_t) * RR
    // 负号来源：target位姿出现在逆变换中，扰动方向取反
    *d_rel_d_t = -tmp2.Adj() * RR;
  }

  return res;
}


/**
 * @brief 内联函数：线性化特征点投影模型，计算投影残差及雅可比矩阵（BA核心函数）
 * @tparam Scalar 标量类型（float/double），保证与相机模型标量类型一致
 * @tparam CamT 相机模型类型（如针孔相机、鱼眼相机等，需实现project接口）
 * @param kpt_obs 特征点的观测坐标（图像平面2D坐标，用于计算残差）
 * @param kpt_pos 特征点（路标点）的参数化表示（立体投影坐标+逆深度）
 * @param T_t_h 4x4齐次变换矩阵：从参考帧h到目标帧t的位姿变换 (T_t_h * P_h = P_t)
 * @param cam 相机内参模型（多态，支持不同相机投影模型）
 * @param[out] res 输出投影残差 = 投影预测值 - 观测值（2D向量）
 * @param[out] d_res_d_xi 可选输出：残差对位姿的雅可比矩阵 (2xPOSE_SIZE)，POSE_SIZE通常为6（3平移+3旋转）
 * @param[out] d_res_d_p 可选输出：残差对3D点坐标的雅可比矩阵 (2x3)
 * @param[out] proj 可选输出：4维投影结果 [x, y, 归一化逆深度, 预留]
 * @return bool 投影是否有效（true=有效，false=无效/超出图像范围/数值异常）
 * 
 * 核心流程：
 * 1. 立体投影逆变换：将特征点的方向向量转换为3D齐次坐标（带逆深度）
 * 2. 位姿变换：将参考帧的3D点转换到目标帧坐标系
 * 3. 相机投影：将目标帧3D点投影到图像平面，得到预测坐标
 * 4. 残差计算：预测坐标 - 观测坐标
 * 5. 雅可比计算：（可选）计算残差对位姿/3D点的雅可比矩阵（用于BA优化）
 */
template <class Scalar, class CamT>
inline bool linearizePoint(
    const Eigen::Matrix<Scalar, 2, 1>& kpt_obs, const Keypoint<Scalar>& kpt_pos,
    const Eigen::Matrix<Scalar, 4, 4>& T_t_h, const CamT& cam,
    Eigen::Matrix<Scalar, 2, 1>& res,
    Eigen::Matrix<Scalar, 2, POSE_SIZE>* d_res_d_xi = nullptr,
    Eigen::Matrix<Scalar, 2, 3>* d_res_d_p = nullptr,
    Eigen::Matrix<Scalar, 4, 1>* proj = nullptr) {

  // 静态断言：确保相机模型的标量类型与函数模板标量类型一致，避免精度不匹配
  static_assert(std::is_same_v<typename CamT::Scalar, Scalar>);

  // Todo implement without jacobians
  // Todo: 待优化：实现无雅可比的轻量版本（仅计算投影坐标，不计算导数）
  // 临时变量：立体投影逆变换的雅可比矩阵（4x2），存储unproject的导数
  Eigen::Matrix<Scalar, 4, 2> Jup;

  // 参考帧下的3D齐次点坐标（前3维为方向，第4维为逆深度）
  Eigen::Matrix<Scalar, 4, 1> p_h_3d;

  // 步骤1：立体投影逆变换（unproject）
  // 将特征点的2D方向向量（kpt_pos.direction）转换为3D齐次坐标
  // Jup：输出unproject操作的雅可比矩阵（4x2），用于后续残差对方向的导数计算
  p_h_3d = StereographicParam<Scalar>::unproject(kpt_pos.direction, &Jup);

  // 第4维赋值为逆深度（inv_dist），完整表示参考帧下的3D点（逆深度参数化）
  p_h_3d[3] = kpt_pos.inv_dist;

  // 步骤2：位姿变换
  // 将参考帧h的3D点转换到目标帧t的坐标系下（T_t_h为h->t的变换矩阵）
  const Eigen::Matrix<Scalar, 4, 1> p_t_3d = T_t_h * p_h_3d;

  // 临时变量：相机投影的雅可比矩阵（2x4），存储project的导数
  Eigen::Matrix<Scalar, 2, 4> Jp;

  // 步骤3：相机投影
  // 将目标帧的3D齐次点投影到图像平面，得到预测坐标（存入res）
  // Jp：输出project操作的雅可比矩阵（2x4），res初始为投影预测值
  // valid：标记投影是否有效（如是否在图像范围内、无数值异常）
  bool valid = cam.project(p_t_3d, res, &Jp);

  // 额外检查：投影结果是否为有限值（排除NaN/Inf）
  valid &= res.array().isFinite().all();

  // 投影无效处理：返回false，跳过该特征点的优化
  if (!valid) {
    //      std::cerr << " Invalid projection! kpt_pos.dir "
    //                << kpt_pos.dir.transpose() << " kpt_pos.id " <<
    //                kpt_pos.id
    //                << " idx " << kpt_obs.kpt_id << std::endl;

    //      std::cerr << "T_t_h\n" << T_t_h << std::endl;
    //      std::cerr << "p_h_3d\n" << p_h_3d.transpose() << std::endl;
    //      std::cerr << "p_t_3d\n" << p_t_3d.transpose() << std::endl;

    return false;
  }

  // 可选：填充投影结果（proj参数非空时）
  if (proj) {
    // proj前2维：图像平面的投影预测坐标
    proj->template head<2>() = res;
    // 对逆深度进行归一化，主要用于可视化，作用：
    // 1.消除 SLAM 的尺度不确定性，让不同帧 / 场景的特征点可统一可视化
    // 2.压缩数值范围，避免颜色映射失真，清晰区分近 / 中 / 远特征点
    // 3.提升数值稳定性，过滤极端值，避免可视化 / 优化异常
    (*proj)[2] = p_t_3d[3] / p_t_3d.template head<3>().norm();
  }


  // 步骤4：计算残差 = 投影预测值 - 观测值（BA优化的核心残差）
  res -= kpt_obs;

  // 步骤5.1：计算残差对位姿的雅可比矩阵（d_res_d_xi非空时）
  if (d_res_d_xi) {
    // 临时变量：3D点对位姿的雅可比矩阵（4xPOSE_SIZE，POSE_SIZE=6）
    Eigen::Matrix<Scalar, 4, POSE_SIZE> d_point_d_xi;

    // 前3行前3列：平移部分的导数（单位矩阵 * 逆深度）
    d_point_d_xi.template topLeftCorner<3, 3>() =
        Eigen::Matrix<Scalar, 3, 3>::Identity() * kpt_pos.inv_dist;

    // 前3行后3列：旋转部分的导数（-反对称矩阵 * 3D点前3维）
    // Sophus::SO3::hat：将3D向量转换为3x3反对称矩阵，用于旋转导数计算
    d_point_d_xi.template topRightCorner<3, 3>() =
        -Sophus::SO3<Scalar>::hat(p_t_3d.template head<3>());

    // 第4行置零：逆深度对位姿无导数
    d_point_d_xi.row(3).setZero();

    // 链式法则：残差对位姿的雅可比 = 投影雅可比 * 点对位姿的雅可比
    *d_res_d_xi = Jp * d_point_d_xi;
  }

  // 步骤5.2：计算残差对3D点的雅可比矩阵（d_res_d_p非空时）
  if (d_res_d_p) {
    // 临时变量：3D点对特征点参数的雅可比矩阵（4x3）
    Eigen::Matrix<Scalar, 4, 3> Jpp;
    Jpp.setZero();

    // 前3行前2列：方向向量的导数（位姿变换矩阵前3行 * unproject雅可比）
    Jpp.template block<3, 2>(0, 0) = T_t_h.template topLeftCorner<3, 4>() * Jup;

    // 第3列：逆深度的导数（位姿变换矩阵的第3列，平移分量）
    Jpp.col(2) = T_t_h.col(3);

    // 链式法则：残差对3D点的雅可比 = 投影雅可比 * 点对参数的雅可比
    *d_res_d_p = Jp * Jpp;
  }

  // 投影有效，返回true
  return true;
}

}  // namespace basalt
