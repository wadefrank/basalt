#include "qxwz_vio.h"

namespace qxwz {

// 构造函数
VIO::VIO()
{
    // 显式初始化，调用默认构造函数
    vio_config_ = basalt::VioConfig();
    calib_ = basalt::Calibration<double>();

    opt_flow_ptr_ = nullptr;
    vio_ = nullptr;

    is_load_vio_config_ = false;
    is_load_calib_ = false;

    is_pose_thread_running_ = false; // 初始化线程运行标志

    vio_status_ = VioStatus::UNINITIALIZED;
}

// -------------------------- 析构函数 --------------------------
VIO::~VIO() {
    try {
        // 确保线程安全退出
        shutdown();
    } catch (const std::exception& e) {
        std::cerr << "Exception in VIO destructor: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "Unknown exception in VIO destructor" << std::endl;
    }
}

// -------------------------- 配置与参数相关接口 --------------------------
bool VIO::loadConfig(const std::string& configFilePath) {
    if (configFilePath.empty()) return false;

    try {
        // 核心操作：加载配置文件
        vio_config_.load(configFilePath);

        // 打印成功日志，便于调试和确认
        std::cout << "Successfully loaded VIO config file: " << configFilePath << std::endl;

    } catch (const std::exception& e) {
        // 捕获标准库异常（如文件不存在、格式错误、IO错误等）
        std::cerr << "Error: failed to load VIO config - " << e.what() << std::endl;
        return false;
    } catch (...) {
        // 捕获所有未预期的非标准异常，作为兜底
        std::cerr << "Error: unknown exception occurred while loading config file: "
                  << configFilePath << std::endl;
        return false;
    }


    is_load_vio_config_ = true;

    return true;
}

void VIO::printConfig()
{
    vio_config_.print();
}

bool VIO::loadCalib(const std::string& calibFilePath)
{
    // 1. 以二进制模式打开标定文件
    std::ifstream is(calibFilePath, std::ios::binary); // 变量名修正为is（input stream）更符合语义

    // 2. 检查文件是否成功打开
    if (!is.is_open()) {
        std::cerr << "Error: could not open calibration file - " << calibFilePath << std::endl;
        return false;
    }

    try {
        // 3. 创建cereal的JSON输入归档对象
        cereal::JSONInputArchive archive(is);

        // 4. 反序列化：将JSON数据读取到calib对象中（核心风险点）
        archive(calib_);

        // 5. 打印加载成功的日志
        std::cout << "Successfully loaded camera calibration with "
                  << calib_.intrinsics.size() << " cameras" << std::endl;

    } catch (const cereal::Exception& e) {
        // 捕获cereal库抛出的特定异常（JSON解析错误、字段不匹配等）
        std::cerr << "Error: cereal deserialization failed - " << e.what() << std::endl;
        return false;
    } catch (const std::exception& e) {
        // 捕获标准库异常（如内存不足、流操作错误等）
        std::cerr << "Error: standard exception occurred - " << e.what() << std::endl;
        return false;
    } catch (...) {
        // 捕获所有未预期的异常，避免程序崩溃
        std::cerr << "Error: unknown exception occurred while loading calibration" << std::endl;
        return false;
    }

    is_load_calib_ = true;

    return true;
}

void VIO::printCalib()
{
    std::cout << "==================================== Calib 参数 ====================================" << std::endl;

    // 设置输出精度（保留6位小数，适配IMU噪声/偏置等小数值）
    std::cout << std::fixed << std::setprecision(6);

    // -------------------------- 1. 相机数量基本信息 --------------------------
    size_t cam_num = calib_.intrinsics.size();
    std::cout << "=== Camera-IMU Calibration Info ===" << std::endl;
    std::cout << "Number of cameras: " << cam_num << std::endl;

    // -------------------------- 2. 相机-IMU时间偏移 --------------------------
    std::cout << "Camera-IMU time offset (ns): " << calib_.cam_time_offset_ns << std::endl;

    // -------------------------- 3. IMU基本参数 --------------------------
    std::cout << "IMU update rate (Hz): " << calib_.imu_update_rate << std::endl;

    // IMU连续时间噪声标准差
    std::cout << "IMU continuous noise (rad/s for gyro, m/s² for accel):" << std::endl;
    std::cout << "  Gyro noise std:  [" << calib_.gyro_noise_std.x() << ", "
                << calib_.gyro_noise_std.y() << ", " << calib_.gyro_noise_std.z() << "]" << std::endl;
    std::cout << "  Accel noise std: [" << calib_.accel_noise_std.x() << ", "
                << calib_.accel_noise_std.y() << ", " << calib_.accel_noise_std.z() << "]" << std::endl;

    // IMU偏置随机游走标准差
    std::cout << "IMU bias random walk (rad/(s·√Hz) for gyro, m/(s²·√Hz) for accel):" << std::endl;
    std::cout << "  Gyro bias std:  [" << calib_.gyro_bias_std.x() << ", "
                << calib_.gyro_bias_std.y() << ", " << calib_.gyro_bias_std.z() << "]" << std::endl;
    std::cout << "  Accel bias std: [" << calib_.accel_bias_std.x() << ", "
                << calib_.accel_bias_std.y() << ", " << calib_.accel_bias_std.z() << "]" << std::endl;

    // IMU离散时间噪声（通过成员函数计算）
    Eigen::Vector3d gyro_discrete = calib_.dicrete_time_gyro_noise_std();
    Eigen::Vector3d accel_discrete = calib_.dicrete_time_accel_noise_std();
    std::cout << "IMU discrete noise (rad/s for gyro, m/s² for accel):" << std::endl;
    std::cout << "  Gyro discrete std:  [" << gyro_discrete.x() << ", "
                << gyro_discrete.y() << ", " << gyro_discrete.z() << "]" << std::endl;
    std::cout << "  Accel discrete std: [" << accel_discrete.x() << ", "
                << accel_discrete.y() << ", " << accel_discrete.z() << "]" << std::endl;

    // -------------------------- 4. IMU静态偏置（标定值） --------------------------
    std::cout << "IMU static bias (calibrated):" << std::endl;
    std::cout << "  Accel bias: [" << calib_.calib_accel_bias.getParam().x() << ", "
                << calib_.calib_accel_bias.getParam().y() << ", "
                << calib_.calib_accel_bias.getParam().z() << "]" << std::endl;
    std::cout << "  Gyro bias:  [" << calib_.calib_gyro_bias.getParam().x() << ", "
                << calib_.calib_gyro_bias.getParam().y() << ", "
                << calib_.calib_gyro_bias.getParam().z() << "]" << std::endl;

    // -------------------------- 5. 多相机详细参数 --------------------------
    for (size_t i = 0; i < cam_num; ++i) { // 替换size_t为Eigen::Index
        std::cout << "--- Camera " << i << " Info ---" << std::endl;

        // 5.1 相机到IMU位姿 T_i_c
        const Sophus::SE3d& T_ic = calib_.T_i_c[i];
        std::cout << "  T_i_c (camera to IMU):" << std::endl;
        Eigen::Quaterniond q = T_ic.unit_quaternion();
        std::cout << "    Rotation (quaternion): w=" << q.w() << ", x=" << q.x()
                << ", y=" << q.y() << ", z=" << q.z() << std::endl;
        Eigen::Vector3d t = T_ic.translation();
        std::cout << "    Translation (m): [" << t.x() << ", " << t.y() << ", " << t.z() << "]" << std::endl;

        // 5.2 相机分辨率
        const Eigen::Vector2i& res = calib_.resolution[i];
        std::cout << "  Resolution: " << res.x() << "x" << res.y() << std::endl;

        // 5.3 相机内参（修复循环变量类型）
        const basalt::GenericCamera<double>& cam_intrin = calib_.intrinsics[i];
        std::cout << "  Camera intrinsics params: ";
        const auto& intrins_params = cam_intrin.getParam();
        for (Eigen::Index j = 0; j < intrins_params.size(); ++j) {
            std::cout << intrins_params[j] << (j == intrins_params.size()-1 ? "" : ", ");
        }
        std::cout << std::endl;
    }
    std::cout << "============================================================================================" << std::endl;

    // 恢复默认输出格式
    std::cout << std::resetiosflags(std::ios::fixed);
    std::cout << std::endl;
}

// -------------------------- 传感器数据输入接口 --------------------------
void VIO::pushIMU(const ImuData& imuData) {
    // std::cout << "pushIMU" << std::endl;

    // 如果VIO不是运行状态，直接返回
    if (getCurrentStatus() != VioStatus::RUNNING) {
        return;
    }

    basalt::ImuData<double>::Ptr imu_data(new basalt::ImuData<double>);
    imu_data->t_ns = imuData.timestamp;

    imu_data->accel = imuData.accel;
    imu_data->gyro = imuData.gyro;

    vio_->imu_data_queue.push(imu_data);
}

void VIO::pushStereoImage(const StereoImageData& stereoImageData) {

    // std::cout << "pushStereoImage" << std::endl;
    // 如果VIO不是运行状态，提示报错信息，然后return
    if (getCurrentStatus() != VioStatus::RUNNING) {
        std::cerr << "VIO is not running." << std::endl;
        return;
    }

    // 如果左目图像为空，提示报错信息，然后return
    if (stereoImageData.leftImage.image.empty()) {
        std::cerr << "Left image is empty." << std::endl;
        return;
    }

    // 如果右目图像为空，提示报错信息，然后return
    if (stereoImageData.rightImage.image.empty()) {
        std::cerr << "Right image is empty." << std::endl;
        return;
    }

    if (stereoImageData.leftImage.image.type() != CV_8UC1) {
        std::cerr << "Left image type required to be CV_8UC1, but is : " << stereoImageData.leftImage.image.type() << std::endl;
        return;
    }

    if (stereoImageData.rightImage.image.type() != CV_8UC1) {
        std::cerr << "Right image type required to be CV_8UC1, but is : " << stereoImageData.leftImage.image.type() << std::endl;
        return;
    }

    basalt::OpticalFlowInput::Ptr image_data(new basalt::OpticalFlowInput);
    image_data->img_data.resize(2);

    // 获取图像时间戳
    image_data->t_ns = stereoImageData.timestamp;

    // 左目图像
    image_data->img_data[0].img.reset(new basalt::ManagedImage<uint16_t>(stereoImageData.leftImage.image.cols, stereoImageData.leftImage.image.rows));
    image_data->img_data[1].img.reset(new basalt::ManagedImage<uint16_t>(stereoImageData.rightImage.image.cols, stereoImageData.rightImage.image.rows));

    const uint8_t *left_data_in = stereoImageData.leftImage.image.ptr();
    const uint8_t *right_data_in = stereoImageData.rightImage.image.ptr();
    uint16_t *left_data_out = image_data->img_data[0].img->ptr;
    uint16_t *right_data_out = image_data->img_data[1].img->ptr;

    size_t left_full_size = stereoImageData.leftImage.image.cols * stereoImageData.leftImage.image.rows;
    for (size_t i = 0; i < left_full_size; i++) {
        int left_val = left_data_in[i];
        left_val = left_val << 8;
        left_data_out[i] = left_val;
    }

    size_t right_full_size = stereoImageData.rightImage.image.cols * stereoImageData.rightImage.image.rows;
    for (size_t i = 0; i < right_full_size; i++) {
        int right_val = right_data_in[i];
        right_val = right_val << 8;
        right_data_out[i] = right_val;
    }

    opt_flow_ptr_->input_queue.push(image_data);
}

// -------------------------- 回调函数接口 --------------------------
void VIO::registerPoseCallback(const std::function<void(const PoseWithCovariance&)>& callback) {
    // 线程安全地保存回调函数
    std::unique_lock<std::shared_mutex> lock(pose_callback_mutex_);
    pose_callback_ = callback;
}

void VIO::unregisterPoseCallback() {
    // 线程安全地清除回调函数
    std::unique_lock<std::shared_mutex> lock(pose_callback_mutex_);
    pose_callback_ = nullptr;
}

void VIO::registerStatusCallback(const std::function<void(VioStatus)>& callback) {
    // 线程安全地清除回调函数
    std::unique_lock<std::shared_mutex> lock(status_callback_mutex_);
    status_callback_ = callback;
}

void VIO::unregisterStatusCallback() {
    // 线程安全地清除回调函数
    std::unique_lock<std::shared_mutex> lock(status_callback_mutex_);
    status_callback_ = nullptr;
}

// -------------------------- 算法控制接口 --------------------------
bool VIO::start() {
    // 1. 检查配置和标定是否加载
    if (!is_load_vio_config_) {
        // std::cerr <<
        return false;
    }

    if (!is_load_calib_) {
        // std::cerr <<
        return false;
    }


    // 2.初始化光流追踪模块和VIO模块
    opt_flow_ptr_ = basalt::OpticalFlowFactory::getOpticalFlow(vio_config_, calib_);

    bool use_imu = true;          // 是否使用IMU
    bool use_double = false;      // 是否使用双精度浮点数
    vio_ = basalt::VioEstimatorFactory::getVioEstimator(vio_config_, calib_, basalt::constants::g, use_imu, use_double);

    // 3.启动 VIO 主线程
    vio_->initialize(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());


    opt_flow_ptr_->output_queue = &vio_->vision_data_queue; // 光流输出到VIO
    vio_->out_state_queue = &out_state_queue_;              // VIO状态输出队列

    // 4. 启动位姿处理线程
    startPoseThread();

    // 5.更新状态
    updateStatus(VioStatus::RUNNING);

    return true;
}

bool VIO::reset() {
    try
    {
        // 停止位姿线程
        stopPoseThread();

        // 停止VIO主线程
        vio_->maybe_join();
        vio_->drain_input_queues();

        // 重新启动 VIO 主线程
        vio_->initialize(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());

        opt_flow_ptr_->output_queue = &vio_->vision_data_queue; // 光流输出到VIO
        vio_->out_state_queue = &out_state_queue_;              // VIO状态输出队列

        // 启动位姿处理线程
        startPoseThread();

        updateStatus(VioStatus::RUNNING);

        return true;
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        updateStatus(VioStatus::FAILED);
        return false;
    }

    return false;
}

bool VIO::shutdown() {
    try
    {
        // 停止位姿处理线程
        stopPoseThread();

        vio_->maybe_join();
        vio_->drain_input_queues();

        return true;
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        updateStatus(VioStatus::FAILED);
        return false;
    }
}

// -------------------------- 状态相关接口 --------------------------
VioStatus VIO::getCurrentStatus() const {
    std::shared_lock<std::shared_mutex> lock(status_mutex_);
    return vio_status_;
}

void VIO::updateStatus(const VioStatus& new_status) {
    std::unique_lock<std::shared_mutex> lock(status_mutex_);
    vio_status_ = new_status;
}

// -------------------------- 私有辅助函数：位姿线程管理 --------------------------
void VIO::startPoseThread() {
    if (is_pose_thread_running_) {
        return; // 避免重复启动
    }

    is_pose_thread_running_ = true;

    // 启动位姿处理线程
    pose_thread_ = std::thread([this]() {
        basalt::PoseVelBiasState<double>::Ptr data;

        while (is_pose_thread_running_) {
            out_state_queue_.pop(data);
            if (!data.get()) {
                break;
            }

            PoseWithCovariance pose_data;
            pose_data.timestamp =  data->t_ns;

            Sophus::SE3d T_w_i = data->T_w_i;                   // 获取SE3位姿
            pose_data.position = T_w_i.translation();           // SE3的平移部分 → Vector3d
            pose_data.orientation = T_w_i.unit_quaternion();    // SE3的旋转部分 → 四元数
            pose_data.orientation.normalize();                  // 确保四元数是归一化的（可选，但建议做）

            // TODO
            pose_data.cov.setZero();                            // 若有协方差数据，替换为实际值
            pose_data.confidence = 1.0;                         // 若有置信度计算逻辑，替换为实际值

            // 线程安全地调用回调函数
            std::shared_lock<std::shared_mutex> lock(pose_callback_mutex_);
            if (pose_callback_) {
                try {
                    pose_callback_(pose_data); // 调用注册的回调
                } catch (const std::exception& e) {
                    std::cerr << "Error in pose callback: " << e.what() << std::endl;
                } catch (...) {
                    std::cerr << "Unknown error in pose callback" << std::endl;
                }
            }

        }

        std::cout << "Finished pose thread" << std::endl;
    });
}

void VIO::stopPoseThread() {
    if (!is_pose_thread_running_) {
        return;
    }

    is_pose_thread_running_ = false;

    // 发送结束信号（根据队列实现，可能需要push空指针）
    out_state_queue_.push(nullptr);

    // 等待线程退出
    if (pose_thread_.joinable()) {
        pose_thread_.join();
    }
}

} // namespace qxwz