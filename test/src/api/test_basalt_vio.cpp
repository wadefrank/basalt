// sudo apt install nlohmann-json3-dev

#include <iostream>
#include <memory>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>
#include <chrono>
#include <thread>

// 第三方库
#include <nlohmann/json.hpp>  // 解析EuRoC的cam/imu时间戳文件（需安装nlohmann/json）
#include <opencv2/opencv.hpp>

// 自定义VIO头文件
#include "basalt/api/basalt_vio.h"

// 命名空间简化
namespace fs = std::filesystem;
using json = nlohmann::json;
using namespace basalt_vio;

// -------------------------- 全局变量/工具函数 --------------------------
// 位姿保存文件（确保线程安全）
std::ofstream pose_file;
std::mutex pose_file_mutex;

/**
 * @brief 位姿回调函数：将位姿写入文件（TUM格式：timestamp x y z qx qy qz qw）
 * @param pose 带协方差的位姿数据
 */
void poseCallback(const PoseWithCovariance& pose) {
    std::lock_guard<std::mutex> lock(pose_file_mutex);
    if (pose_file.is_open()) {
        // 转换时间戳：纳秒 -> 秒（EuRoC数据集时间戳单位为秒，保留9位小数）
        double timestamp_sec = static_cast<double>(pose.timestamp) / 1e9;

        // TUM格式：timestamp x y z qx qy qz qw
        pose_file << std::fixed << std::setprecision(9)
                  << timestamp_sec << " "
                  << pose.position.x() << " " << pose.position.y() << " " << pose.position.z() << " "
                  << pose.orientation.x() << " " << pose.orientation.y() << " " << pose.orientation.z() << " " << pose.orientation.w() << std::endl;
    }
}

/**
 * @brief 状态回调函数：打印VIO运行状态
 * @param status VIO当前状态
 */
void statusCallback(VioStatus status) {
    std::string status_str;
    switch (status) {
        case VioStatus::UNINITIALIZED: status_str = "UNINITIALIZED"; break;
        case VioStatus::INITIALIZING:  status_str = "INITIALIZING";  break;
        case VioStatus::RUNNING:       status_str = "RUNNING";       break;
        case VioStatus::LOST:          status_str = "LOST";          break;
        case VioStatus::FAILED:        status_str = "FAILED";        break;
        default:                       status_str = "UNKNOWN";       break;
    }
    std::cout << "[VIO Status] " << status_str << std::endl;
}

/**
 * @brief 读取EuRoC数据集的时间戳文件（如cam0/data.csv）
 * @param timestamp_path 时间戳文件路径
 * @return 时间戳列表（ns）
 */
std::vector<int64_t> loadEuRoCTimestamps(const std::string& timestamp_path) {
    std::vector<int64_t> timestamps;
    std::ifstream file(timestamp_path);
    if (!file.is_open()) {
        std::cerr << "Failed to open timestamp file: " << timestamp_path << std::endl;
        return timestamps;
    }

    std::string line;
    // 跳过第一行（表头）
    std::getline(file, line);
    while (std::getline(file, line)) {


        std::stringstream ss(line);
        char tmp;
        int64_t t_ns;
        std::string path;
        ss >> t_ns >> tmp >> path;
        timestamps.push_back(t_ns);
    }
    file.close();
    return timestamps;
}

/**
 * @brief 读取EuRoC IMU数据
 * @param imu_path IMU文件路径（如imu0/data.csv）
 * @return IMU数据列表
 */
std::vector<ImuData> loadEuRoCImuData(const std::string& imu_path) {
    std::vector<ImuData> imu_data_list;
    std::ifstream file(imu_path);
    if (!file.is_open()) {
        std::cerr << "Failed to open IMU file: " << imu_path << std::endl;
        return imu_data_list;
    }

    std::string line;
    // 跳过第一行（表头）
    std::getline(file, line);
    while (std::getline(file, line)) {
        // 格式：timestamp,wx,wy,wz,ax,ay,az
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream iss(line);
        uint64_t t_ns;
        double wx, wy, wz, ax, ay, az;
        iss >> t_ns >> wx >> wy >> wz >> ax >> ay >> az;

        ImuData imu_data;
        imu_data.timestamp = t_ns;          // 时间
        imu_data.gyro = Eigen::Vector3d(wx, wy, wz);                // 角速度 (rad/s)
        imu_data.accel = Eigen::Vector3d(ax, ay, az);               // 线加速度 (m/s²)
        imu_data_list.push_back(imu_data);
    }
    file.close();
    return imu_data_list;
}

int main(int argc, char** argv) {
    // -------------------------- 1. 初始化路径与参数 --------------------------
    // 配置/标定文件路径
    std::string config_file_path = "/home/wade/wade/Code/SLAM/basalt/data/euroc_config.json";
    std::string calib_file_path = "/home/wade/wade/Code/SLAM/basalt/data/euroc_ds_calib.json";

    // EuRoC数据集路径
    std::string euroc_data_path = "/home/wade/wade/Data/machine_hall/MH_05_difficult/MH_05_difficult";
    std::string cam0_ts_path = euroc_data_path + "/mav0/cam0/data.csv";
    std::string cam1_ts_path = euroc_data_path + "/mav0/cam1/data.csv";
    std::string imu_ts_path = euroc_data_path + "/mav0/imu0/data.csv";
    std::string cam0_img_dir = euroc_data_path + "/mav0/cam0/data/";
    std::string cam1_img_dir = euroc_data_path + "/mav0/cam1/data/";

    // 位姿保存路径（确保目录存在）
    std::string result_dir = "/home/wade/wade/Result/basalt_vio/trajectory";
    if (!fs::exists(result_dir)) {
        fs::create_directories(result_dir);
    }
    std::string pose_save_path = result_dir + "/vio_pose_tum.txt";

    // -------------------------- 2. 初始化VIO对象 --------------------------
    std::shared_ptr<VIO> basalt_vio = std::make_shared<VIO>();

    // -------------------------- 3. 加载配置与标定 --------------------------
    std::cout << "Loading VIO config..." << std::endl;
    if (!basalt_vio->loadConfig(config_file_path)) {
        std::cerr << "Failed to load config file!" << std::endl;
        return -1;
    }
    // basalt_vio->printConfig();  // 可选：打印配置

    std::cout << "Loading calibration file..." << std::endl;
    if (!basalt_vio->loadCalib(calib_file_path)) {
        std::cerr << "Failed to load calibration file!" << std::endl;
        return -1;
    }
    // basalt_vio->printCalib();  // 可选：打印标定参数

    // -------------------------- 4. 注册回调函数 --------------------------
    // 打开位姿保存文件
    pose_file.open(pose_save_path, std::ios::out | std::ios::trunc);
    if (!pose_file.is_open()) {
        std::cerr << "Failed to open pose save file: " << pose_save_path << std::endl;
        return -1;
    }
    // 写入TUM格式表头（可选）
    pose_file << "# timestamp x y z qx qy qz qw" << std::endl;

    // 注册位姿回调和状态回调
    basalt_vio->registerPoseCallback(poseCallback);
    basalt_vio->registerStatusCallback(statusCallback);

    // -------------------------- 5. 启动VIO --------------------------
    std::cout << "Starting VIO..." << std::endl;
    bool is_vio_start_succ = basalt_vio->start();
    if (!is_vio_start_succ) {
        std::cerr << "VIO start failed!" << std::endl;
        pose_file.close();
        return -1;
    }
    std::cout << "VIO started successfully!" << std::endl;

    // -------------------------- 6. 加载数据集 --------------------------
    // 加载IMU数据（200Hz）
    std::cout << "Loading IMU data..." << std::endl;
    std::vector<ImuData> imu_data_list = loadEuRoCImuData(imu_ts_path);
    if (imu_data_list.empty()) {
        std::cerr << "No IMU data loaded!" << std::endl;
        basalt_vio->shutdown();
        pose_file.close();
        return -1;
    }

    // 加载相机时间戳（20Hz）
    std::cout << "Loading camera timestamps..." << std::endl;
    std::vector<int64_t> cam0_timestamps = loadEuRoCTimestamps(cam0_ts_path);
    std::vector<int64_t> cam1_timestamps = loadEuRoCTimestamps(cam1_ts_path);
    if (cam0_timestamps.empty() || cam1_timestamps.empty() || cam0_timestamps.size() != cam1_timestamps.size()) {
        std::cerr << "Camera timestamps load failed or mismatch!" << std::endl;
        basalt_vio->shutdown();
        pose_file.close();
        return -1;
    }

    // -------------------------- 7. 推送数据到VIO --------------------------
    std::cout << "Pushing data to VIO (total IMU: " << imu_data_list.size()
              << ", total images: " << cam0_timestamps.size() << ")..." << std::endl;

    // 多线程推送IMU（模拟200Hz实时性）
    std::thread imu_thread([&]() {
        // int64_t prev_ts = imu_data_list[0].timestamp;
        for (const auto& imu_data : imu_data_list) {
            // 模拟实时推送：计算与上一帧的时间差，睡眠对应时长
            // int64_t delta_ts = imu_data.timestamp - prev_ts;
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            // prev_ts = imu_data.timestamp;

            // 推送IMU数据
            basalt_vio->pushIMU(imu_data);
        }
    });

    // 主线程推送双目图像（20Hz）
    for (size_t i = 0; i < cam0_timestamps.size(); ++i) {
        // 构造图像文件名（EuRoC格式：timestamp.png）
        std::string cam0_img_name = std::to_string(cam0_timestamps[i]) + ".png";
        std::string cam1_img_name = std::to_string(cam1_timestamps[i]) + ".png";
        std::string cam0_img_path = cam0_img_dir + cam0_img_name;
        std::string cam1_img_path = cam1_img_dir + cam1_img_name;

        // 读取图像（CV_8UC1）
        cv::Mat left_img = cv::imread(cam0_img_path, cv::IMREAD_GRAYSCALE);
        cv::Mat right_img = cv::imread(cam1_img_path, cv::IMREAD_GRAYSCALE);
        if (left_img.empty() || right_img.empty()) {
            std::cerr << "Failed to read image: " << cam0_img_path << " or " << cam1_img_path << std::endl;
            continue;
        }

        // 构造双目图像数据
        StereoImageData stereo_img_data;
        stereo_img_data.timestamp = cam0_timestamps[i];  // 双目时间戳一致

        // 左目图像
        stereo_img_data.leftImage.timestamp = cam0_timestamps[i];
        stereo_img_data.leftImage.image = left_img;
        stereo_img_data.leftImage.width = left_img.cols;
        stereo_img_data.leftImage.height = left_img.rows;
        stereo_img_data.leftImage.exposure = 0.0;  // EuRoC无曝光时间，设为0

        // 右目图像
        stereo_img_data.rightImage.timestamp = cam1_timestamps[i];
        stereo_img_data.rightImage.image = right_img;
        stereo_img_data.rightImage.width = right_img.cols;
        stereo_img_data.rightImage.height = right_img.rows;
        stereo_img_data.rightImage.exposure = 0.0;

        // 推送双目图像数据
        basalt_vio->pushStereoImage(stereo_img_data);

        // 模拟20Hz实时推送（50ms）
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    // -------------------------- 8. 等待数据处理完成 --------------------------
    // 等待IMU线程结束
    if (imu_thread.joinable()) {
        imu_thread.join();
    }

    // 等待VIO处理剩余数据（额外等待5秒）
    std::cout << "Waiting for VIO to process remaining data..." << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(5));

    // -------------------------- 9. 停止VIO并清理 --------------------------
    std::cout << "Shutting down VIO..." << std::endl;
    basalt_vio->shutdown();

    // 注销回调
    basalt_vio->unregisterPoseCallback();
    basalt_vio->unregisterStatusCallback();

    // 关闭文件
    pose_file.close();
    std::cout << "Pose saved to: " << pose_save_path << std::endl;

    std::cout << "Test finished successfully!" << std::endl;
    return 0;
}