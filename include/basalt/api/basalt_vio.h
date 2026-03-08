#pragma once

// 基础库
#include <algorithm>            // 算法库
#include <chrono>               // 时间库
#include <condition_variable>   // 线程同步
#include <iostream>             // 输入输出
#include <thread>               // 多线程
#include <shared_mutex>         // 共享锁

#include <fmt/format.h>         // 字符串格式化
#include <sophus/se3.hpp>       // 李群SE3操作

// 并行计算
#include <tbb/concurrent_unordered_map.h> // 线程安全哈希表
#include <tbb/global_control.h>           // TBB全局控制

// 可视化库 (Pangolin)
// #include <pangolin/display/default_font.h>
// #include <pangolin/display/image_view.h>    // 图像显示
// #include <pangolin/gl/gldraw.h>             // OpenGL绘制
// #include <pangolin/image/image.h>           // 图像处理
// #include <pangolin/image/image_io.h>
// #include <pangolin/image/typed_image.h>
// #include <pangolin/pangolin.h>              // 主库

// 命令行解析
#include <CLI/CLI.hpp>

#include <basalt/io/dataset_io.h>
#include <basalt/io/marg_data_io.h>
#include <basalt/spline/se3_spline.h>
#include <basalt/vi_estimator/vio_estimator.h>
#include <basalt/calibration/calibration.hpp>

#include <basalt/serialization/headers_serialization.h>

#include <basalt/utils/system_utils.h>
#include <basalt/utils/vis_utils.h>
#include <basalt/utils/format.hpp>
#include <basalt/utils/time_utils.hpp>

#include <opencv2/highgui/highgui.hpp>

namespace basalt {

// IMU数据
struct ImuData {
    int64_t timestamp;  		///< IMU时间戳（单位：纳秒）
    Eigen::Vector3d accel;    	///< 加速度计测量，线加速度（单位：m/s^2）
    Eigen::Vector3d gyro;  		///< 陀螺仪测量，角速度（单位：rad/s）
};

// 图像数据
struct ImageData {
    int64_t timestamp;  		///< 相机时间戳（单位：纳秒）
    cv::Mat image;				///< 图像（type: CV_8UC1）
    int width;                  ///< 图像宽度
    int height;                 ///< 图像高度
    double exposure;			///< (可选)图像曝光时间
};

// 双目图像数据
struct StereoImageData {
    int64_t timestamp;  		///< 相机时间戳（单位：纳秒）
    ImageData leftImage;		///< 左目图像（type: CV_8UC1）
    ImageData rightImage;		///< 右目图像（type: CV_8UC1）
};

// 位姿数据
struct PoseWithCovariance {
    int64_t timestamp;					///< 位姿时间戳（单位：纳秒）
    Eigen::Vector3d position;           ///< 位置 (世界坐标系，单位：米)
    Eigen::Quaterniond orientation;     ///< 姿态四元数 (世界坐标系，w,x,y,z顺序)
    Eigen::Matrix<double, 6, 6> cov;   	///< 6x6 协方差矩阵 (位置+姿态)
    double confidence;                  ///< 置信度 [0, 1]
};

// 算法运行状态
enum class VioStatus {
    UNINITIALIZED,  ///< 未初始化：算法尚未启动
    INITIALIZING,   ///< 初始化中：正在处理初始帧和IMU数据
    RUNNING,       	///< 正常运行：位姿跟踪稳定
    LOST,          	///< 跟踪丢失：暂时丢失特征点，尝试恢复
    FAILED          ///< 算法失败：无法恢复，需重置
};


/**
 * @brief 视觉VIO接口
 */
class VIO {
public:

    VIO();
    ~VIO();

    // 禁止拷贝
    VIO(const VIO&) = delete;
    VIO& operator=(const VIO&) = delete;

    // -------------------------- 配置与参数相关接口 --------------------------
    /**
     * @brief 加载算法配置（yaml格式）
     * @param configFilePath 算法配置文件路径
     * @return 是否加载成功
     */
    bool loadConfig(const std::string& configFilePath);

    /**
     * @brief 打印配置参数
     */
    void printConfig();

    /**
     * @brief 加载标定参数（yaml格式）
     * @param calibFilePath 标定参数文件路径
     * @return 是否加载成功
     */
    bool loadCalib(const std::string& calibFilePath);

    /**
     * @brief 打印标定参数
     */
    void printCalib();


    // -------------------------- 传感器数据输入接口 --------------------------
    /**
     * @brief 推送IMU数据到VIO算法
     * @param imuData IMU数据结构体
     */
    void pushIMU(const ImuData& imuData);

    /**
     * @brief 推送双目图像数据到VIO算法
     * @param stereoImageData 双目图像数据结构体
     */
    void pushStereoImage(const StereoImageData& stereoImageData);


    // -------------------------- 回调函数接口 --------------------------
    /**
     * @brief 注册位姿回调函数（异步输出位姿）
     * @param callback 回调函数，参数为PoseWithCovariance
     */
    void registerPoseCallback(const std::function<void(const PoseWithCovariance&)>& callback);

    /**
     * @brief 注销位姿回调函数
     */
    void unregisterPoseCallback();

    /**
     * @brief 注册状态回调函数（定期或者状态变化时触发）
     * @param callback 回调函数，参数为VioStatus
     */
    void registerStatusCallback(const std::function<void(VioStatus)>& callback);

    /**
     * @brief 注销状态回调函数
     */
    void unregisterStatusCallback();

    // -------------------------- 算法控制接口 --------------------------
    /**
     * @brief 启动VIO算法
     * @note 启动前需确保配置和标定参数已加载
     *
     * @return 是否启动成功
     */
    bool start();

    /**
     * @brief 重置VIO算法（清空状态，重新初始化）
     * @note 重置过程中会暂停数据处理
     *
     * @return 是否重置成功
     */
    bool reset();

    /**
     * @brief 停止VIO算法
     * @note 停止后不再处理传感器数据，需重新start
     *
     *
     * @return 是否关闭成功
     */
    bool shutdown();


    // -------------------------- 状态相关接口 --------------------------
    /**
     * @brief 获取当前VIO算法状态
     * @return 当前状态
     */
    VioStatus getCurrentStatus() const;

    /**
     * @brief 更新当前VIO算法状态
     *
     * @param new_status
     */
    void updateStatus(const VioStatus& new_status);

    // EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
    void startPoseThread();
    void stopPoseThread();



    basalt::VioConfig vio_config_;                 // VIO配置参数
    basalt::Calibration<double> calib_;            // 相机-IMU标定参数

    basalt::OpticalFlowBase::Ptr opt_flow_ptr_;    // 光流前端
    basalt::VioEstimatorBase::Ptr vio_;            // VIO估计器

    bool is_load_vio_config_;
    bool is_load_calib_;

    mutable std::shared_mutex status_mutex_;  // 读写锁（读取频率高，提高性能）
    VioStatus vio_status_;

    // 线程安全队列
    tbb::concurrent_bounded_queue<basalt::PoseVelBiasState<double>::Ptr> out_state_queue_;   // VIO状态队列

    std::thread pose_thread_;               // 位姿处理线程
    bool is_pose_thread_running_;   // 线程运行标志

    std::function<void(const PoseWithCovariance&)> pose_callback_;      // 保存回调函数
    std::shared_mutex pose_callback_mutex_;                             // 保护回调函数的读写

    std::function<void(const VioStatus&)> status_callback_;             // 保存回调函数
    std::shared_mutex status_callback_mutex_;                           // 保护回调函数的读写
};

} // namespace basalt