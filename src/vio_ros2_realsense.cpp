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

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <thread>

#include <fmt/format.h>

#include <sophus/se3.hpp>

#include <tbb/concurrent_unordered_map.h>
#include <tbb/global_control.h>

#include <pangolin/display/default_font.h>
#include <pangolin/display/image_view.h>
#include <pangolin/gl/gldraw.h>
#include <pangolin/image/image.h>
#include <pangolin/image/image_io.h>
#include <pangolin/image/typed_image.h>
#include <pangolin/pangolin.h>

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

// ROS2 头文件
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/pose_with_covariance_stamped.hpp"
#include "geometry_msgs/msg/twist_with_covariance_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/compressed_image.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "std_srvs/srv/empty.hpp"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.h"
#include "tf2_ros/transform_broadcaster.h"

// enable the "..."_format(...) string literal
using namespace basalt::literals;

// GUI functions
void draw_image_overlay(pangolin::View& v, size_t cam_id);
void draw_scene(pangolin::View& view);
void load_data(const std::string& calib_path);
bool next_step();
bool prev_step();
void draw_plots();
void alignButton();
void alignDeviceButton();
void saveTrajectoryButton();

// Pangolin variables
constexpr int UI_WIDTH = 200;

using Button = pangolin::Var<std::function<void(void)>>;

pangolin::DataLog imu_data_log, vio_data_log, error_data_log;
pangolin::Plotter* plotter;

pangolin::Var<int> show_frame("ui.show_frame", 0, 0, 15000);

pangolin::Var<bool> show_flow("ui.show_flow", false, false, true);
pangolin::Var<bool> show_obs("ui.show_obs", true, false, true);
pangolin::Var<bool> show_ids("ui.show_ids", false, false, true);

pangolin::Var<bool> show_est_pos("ui.show_est_pos", true, false, true);
pangolin::Var<bool> show_est_vel("ui.show_est_vel", false, false, true);
pangolin::Var<bool> show_est_bg("ui.show_est_bg", false, false, true);
pangolin::Var<bool> show_est_ba("ui.show_est_ba", false, false, true);

pangolin::Var<bool> show_gt("ui.show_gt", false, false, true);

Button next_step_btn("ui.next_step", &next_step);
Button prev_step_btn("ui.prev_step", &prev_step);

pangolin::Var<bool> continue_btn("ui.continue", false, false, true);
pangolin::Var<bool> continue_fast("ui.continue_fast", true, false, true);

Button align_se3_btn("ui.align_se3", &alignButton);

pangolin::Var<bool> euroc_fmt("ui.euroc_fmt", true, false, true);
pangolin::Var<bool> tum_rgbd_fmt("ui.tum_rgbd_fmt", false, false, true);
pangolin::Var<bool> kitti_fmt("ui.kitti_fmt", false, false, true);
pangolin::Var<bool> save_groundtruth("ui.save_groundtruth", false, false, true);
Button save_traj_btn("ui.save_traj", &saveTrajectoryButton);

pangolin::Var<bool> follow("ui.follow", true, false, true);

// pangolin::Var<bool> record("ui.record", false, false, true);

pangolin::OpenGlRenderState camera;

// Visualization variables
std::unordered_map<int64_t, basalt::VioVisualizationData::Ptr> vis_map;
basalt::VioVisualizationData::Ptr curr_vis_data;

tbb::concurrent_bounded_queue<basalt::VioVisualizationData::Ptr> out_vis_queue;
tbb::concurrent_bounded_queue<basalt::PoseVelBiasState<double>::Ptr>
    out_state_queue;

std::vector<int64_t> vio_t_ns;
Eigen::aligned_vector<Eigen::Vector3d> vio_t_w_i;
Eigen::aligned_vector<Sophus::SE3d> vio_T_w_i;

std::vector<int64_t> gt_t_ns;
Eigen::aligned_vector<Eigen::Vector3d> gt_t_w_i;

std::string marg_data_path;
size_t last_frame_processed = 0;

tbb::concurrent_unordered_map<int64_t, int, std::hash<int64_t>> timestamp_to_id;

std::mutex m;
std::condition_variable cond_var;
bool step_by_step = false;
size_t max_frames = 0;

std::atomic<bool> terminate = false;

// VIO variables
basalt::Calibration<double> calib;

basalt::VioDatasetPtr vio_dataset;
basalt::VioConfig vio_config;
basalt::OpticalFlowBase::Ptr opt_flow_ptr;
basalt::VioEstimatorBase::Ptr vio;

// 图像缓存结构
struct StereoImage {
  int64_t t_ns = 0;
  cv::Mat left_img;
  cv::Mat right_img;
  bool left_ready = false;
  bool right_ready = false;

  StereoImage() = default;  // 添加默认构造函数
  StereoImage(int64_t ts) : t_ns(ts) {}
};

// 图像缓存队列
std::map<int64_t, StereoImage> stereo_image_cache;
std::mutex image_cache_mutex;
std::condition_variable image_cache_cv;

// ROS2 相关变量
class BasaltVioNode : public rclcpp::Node {
 public:
  BasaltVioNode() : Node("basalt_vio") {
    // 声明参数
    this->declare_parameter("publish_odometry", true);
    this->declare_parameter("publish_tf", true);
    this->declare_parameter("publish_pose", true);
    this->declare_parameter("odom_frame_id", "odom");
    this->declare_parameter("base_frame_id", "base_link");
    this->declare_parameter("world_frame_id", "world");
    this->declare_parameter("publish_rate", 10.0);
    this->declare_parameter("use_odometry_covariance", true);
    this->declare_parameter("left_image_topic", "/camera/infra1/image_rect_raw");
    this->declare_parameter("right_image_topic", "/camera/infra2/image_rect_raw");
    this->declare_parameter("imu_topic", "/camera/imu");

    // 获取参数
    publish_odometry_ = this->get_parameter("publish_odometry").as_bool();
    publish_tf_ = this->get_parameter("publish_tf").as_bool();
    publish_pose_ = this->get_parameter("publish_pose").as_bool();
    odom_frame_id_ = this->get_parameter("odom_frame_id").as_string();
    base_frame_id_ = this->get_parameter("base_frame_id").as_string();
    world_frame_id_ = this->get_parameter("world_frame_id").as_string();
    publish_rate_ = this->get_parameter("publish_rate").as_double();
    use_odometry_covariance_ =
        this->get_parameter("use_odometry_covariance").as_bool();
    left_image_topic_ = this->get_parameter("left_image_topic").as_string();
    right_image_topic_ = this->get_parameter("right_image_topic").as_string();
    imu_topic_ = this->get_parameter("imu_topic").as_string();

    RCLCPP_INFO(this->get_logger(), "Subscribing to topics:");
    RCLCPP_INFO(this->get_logger(), "  Left image: %s",
                left_image_topic_.c_str());
    RCLCPP_INFO(this->get_logger(), "  Right image: %s",
                right_image_topic_.c_str());
    RCLCPP_INFO(this->get_logger(), "  IMU: %s", imu_topic_.c_str());

    // 创建发布器
    if (publish_odometry_) {
      odom_pub_ =
          this->create_publisher<nav_msgs::msg::Odometry>("odometry", 10);
      RCLCPP_INFO(this->get_logger(),
                  "Created odometry publisher on topic: odometry");
    }

    if (publish_pose_) {
      pose_pub_ =
          this->create_publisher<geometry_msgs::msg::PoseStamped>("pose", 10);
      pose_cov_pub_ =
          this->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(
              "pose_with_covariance", 10);
      RCLCPP_INFO(this->get_logger(), "Created pose publishers");
    }

    // 创建 TF 广播器
    if (publish_tf_) {
      tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);
      RCLCPP_INFO(this->get_logger(), "Created TF broadcaster");
    }

      // 创建订阅器
      realsense_left_image_sub_ =
          this->create_subscription<sensor_msgs::msg::Image>(
              left_image_topic_, 10,
              std::bind(&BasaltVioNode::realsense_left_image_callback, this,
                        std::placeholders::_1));

      realsense_right_image_sub_ =
          this->create_subscription<sensor_msgs::msg::Image>(
              right_image_topic_, 10,
              std::bind(&BasaltVioNode::realsense_right_image_callback, this,
                        std::placeholders::_1));
    
    // 在节点类中
    rclcpp::QoS qos_profile(100); // 设置队列深度
    qos_profile.best_effort(); // 关键：设置为 BEST_EFFORT 策略
    imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
        imu_topic_, qos_profile,
        std::bind(&BasaltVioNode::imu_callback, this, std::placeholders::_1));

    // 创建重置服务
    reset_service_ = this->create_service<std_srvs::srv::Empty>(
        "reset_estimator",
        std::bind(&BasaltVioNode::reset_callback, this, std::placeholders::_1,
                  std::placeholders::_2, std::placeholders::_3));

    RCLCPP_INFO(this->get_logger(), "Basalt VIO ROS2 Node initialized");

    // 创建定时器用于发布
    if (publish_rate_ > 0) {
      publish_timer_ = this->create_wall_timer(
          std::chrono::duration<double>(1.0 / publish_rate_),
          std::bind(&BasaltVioNode::publish_timer_callback, this));
    }
  }

  // realsense 左目图像回调
  void realsense_left_image_callback(
      const sensor_msgs::msg::Image::SharedPtr msg) {
    try {
      
      cv_bridge::CvImagePtr left_cv_ptr = cv_bridge::toCvCopy(msg, msg->encoding);

      if (left_cv_ptr->image.empty()) {
        RCLCPP_ERROR(this->get_logger(),
                     "Failed to decode left compressed image");
        return;
      }

      // 转换时间戳
      int64_t t_ns =
          msg->header.stamp.sec * 1000000000LL + msg->header.stamp.nanosec;

      std::lock_guard<std::mutex> lock(image_cache_mutex);

      // 使用insert或emplace而不是operator[]，避免需要默认构造函数
      auto it = stereo_image_cache.find(t_ns);
      if (it == stereo_image_cache.end()) {
        auto result = stereo_image_cache.emplace(t_ns, StereoImage(t_ns));
        it = result.first;
      }

      it->second.left_img = left_cv_ptr->image.clone();
      it->second.left_ready = true;

      // 检查是否已收到右目图像
      if (it->second.right_ready) {
        // 通知有新图像可用
        image_cache_cv.notify_one();
      }

    } catch (const std::exception& e) {
      RCLCPP_ERROR(this->get_logger(), "Error in left image callback: %s",
                   e.what());
    }
  }

  // realsense 右目图像回调
  void realsense_right_image_callback(
      const sensor_msgs::msg::Image::SharedPtr msg) {
    try {

      cv_bridge::CvImagePtr right_cv_ptr = cv_bridge::toCvCopy(msg, msg->encoding);

      if (right_cv_ptr->image.empty()) {
        RCLCPP_ERROR(this->get_logger(),
                     "Failed to decode right compressed image");
        return;
      }

      // 转换时间戳
      int64_t t_ns =
          msg->header.stamp.sec * 1000000000LL + msg->header.stamp.nanosec;

      std::lock_guard<std::mutex> lock(image_cache_mutex);

      // 使用insert或emplace而不是operator[]，避免需要默认构造函数
      auto it = stereo_image_cache.find(t_ns);
      if (it == stereo_image_cache.end()) {
        auto result = stereo_image_cache.emplace(t_ns, StereoImage(t_ns));
        it = result.first;
      }

      it->second.right_img = right_cv_ptr->image.clone();
      it->second.right_ready = true;

      // 检查是否已收到左目图像
      if (it->second.left_ready) {
        // 通知有新图像可用
        image_cache_cv.notify_one();
      }

    } catch (const std::exception& e) {
      RCLCPP_ERROR(this->get_logger(), "Error in right image callback: %s",
                   e.what());
    }
  }

  // IMU回调
  void imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg) {
    try {

      // 转换时间戳
      int64_t t_ns =
          msg->header.stamp.sec * 1000000000LL + msg->header.stamp.nanosec;

      // 创建IMU数据
      basalt::ImuData<double>::Ptr imu_data(new basalt::ImuData<double>());
      imu_data->t_ns = t_ns;
      imu_data->accel << msg->linear_acceleration.x, msg->linear_acceleration.y,
          msg->linear_acceleration.z;
      imu_data->gyro << msg->angular_velocity.x, msg->angular_velocity.y,
          msg->angular_velocity.z;

      // 将IMU数据放入队列
      std::lock_guard<std::mutex> lock(imu_mutex_);
      imu_queue_.push(imu_data);

    } catch (const std::exception& e) {
      RCLCPP_ERROR(this->get_logger(), "Error in IMU callback: %s", e.what());
    }
  }

  // 发布里程计信息
  void publish_odometry(const Sophus::SE3d& pose,
                        const Eigen::Vector3d& velocity,
                        const Eigen::Vector3d& gyro_bias,
                        const Eigen::Vector3d& accel_bias,
                        int64_t timestamp_ns) {
    (void)gyro_bias;   // 明确标记为未使用，消除警告
    (void)accel_bias;  // 明确标记为未使用，消除警告

    if (!publish_odometry_ && !publish_pose_ && !publish_tf_) {
      return;
    }

    // 转换时间戳
    rclcpp::Time stamp(timestamp_ns / 1000000000LL,
                       timestamp_ns % 1000000000LL);

    // 提取位姿信息
    Eigen::Vector3d position = pose.translation();
    Eigen::Quaterniond quat = pose.unit_quaternion();

    // 发布里程计消息
    if (publish_odometry_ && odom_pub_) {
      auto odom_msg = std::make_unique<nav_msgs::msg::Odometry>();
      odom_msg->header.stamp = stamp;
      odom_msg->header.frame_id = odom_frame_id_;
      odom_msg->child_frame_id = base_frame_id_;

      // 设置位姿
      odom_msg->pose.pose.position.x = position.x();
      odom_msg->pose.pose.position.y = position.y();
      odom_msg->pose.pose.position.z = position.z();
      odom_msg->pose.pose.orientation.x = quat.x();
      odom_msg->pose.pose.orientation.y = quat.y();
      odom_msg->pose.pose.orientation.z = quat.z();
      odom_msg->pose.pose.orientation.w = quat.w();

      // 设置速度
      odom_msg->twist.twist.linear.x = velocity.x();
      odom_msg->twist.twist.linear.y = velocity.y();
      odom_msg->twist.twist.linear.z = velocity.z();

      // 设置协方差（可选的，可以根据估计的不确定性设置）
      if (use_odometry_covariance_) {
        // 位置协方差（假设各向同性）
        for (int i = 0; i < 6; ++i) {
          odom_msg->pose.covariance[i * 7] = 0.01;  // 对角线元素
        }

        // 速度协方差
        for (int i = 0; i < 6; ++i) {
          odom_msg->twist.covariance[i * 7] = 0.1;  // 对角线元素
        }
      }

      odom_pub_->publish(std::move(odom_msg));
    }

    // 发布位姿消息
    if (publish_pose_ && pose_pub_) {
      auto pose_msg = std::make_unique<geometry_msgs::msg::PoseStamped>();
      pose_msg->header.stamp = stamp;
      pose_msg->header.frame_id = world_frame_id_;
      pose_msg->pose.position.x = position.x();
      pose_msg->pose.position.y = position.y();
      pose_msg->pose.position.z = position.z();
      pose_msg->pose.orientation.x = quat.x();
      pose_msg->pose.orientation.y = quat.y();
      pose_msg->pose.orientation.z = quat.z();
      pose_msg->pose.orientation.w = quat.w();
      pose_pub_->publish(std::move(pose_msg));

      // 发布带协方差的位姿
      if (pose_cov_pub_) {
        auto pose_cov_msg =
            std::make_unique<geometry_msgs::msg::PoseWithCovarianceStamped>();
        pose_cov_msg->header.stamp = stamp;
        pose_cov_msg->header.frame_id = world_frame_id_;
        pose_cov_msg->pose.pose.position.x = position.x();
        pose_cov_msg->pose.pose.position.y = position.y();
        pose_cov_msg->pose.pose.position.z = position.z();
        pose_cov_msg->pose.pose.orientation.x = quat.x();
        pose_cov_msg->pose.pose.orientation.y = quat.y();
        pose_cov_msg->pose.pose.orientation.z = quat.z();
        pose_cov_msg->pose.pose.orientation.w = quat.w();

        if (use_odometry_covariance_) {
          for (int i = 0; i < 6; ++i) {
            pose_cov_msg->pose.covariance[i * 7] = 0.01;
          }
        }

        pose_cov_pub_->publish(std::move(pose_cov_msg));
      }
    }

    // 发布 TF
    if (publish_tf_ && tf_broadcaster_) {
      geometry_msgs::msg::TransformStamped transform;
      transform.header.stamp = stamp;
      transform.header.frame_id = world_frame_id_;
      transform.child_frame_id = base_frame_id_;

      transform.transform.translation.x = position.x();
      transform.transform.translation.y = position.y();
      transform.transform.translation.z = position.z();
      transform.transform.rotation.x = quat.x();
      transform.transform.rotation.y = quat.y();
      transform.transform.rotation.z = quat.z();
      transform.transform.rotation.w = quat.w();

      tf_broadcaster_->sendTransform(transform);
    }
  }

  // 获取 IMU 数据
  basalt::ImuData<double>::Ptr get_imu_data() {
    std::lock_guard<std::mutex> lock(imu_mutex_);
    if (imu_queue_.empty()) {
      return nullptr;
    }

    auto imu_data = imu_queue_.front();
    imu_queue_.pop();
    return imu_data;
  }

  // 获取 ROS2 节点指针
  std::shared_ptr<rclcpp::Node> get_node_ptr() { return shared_from_this(); }

 private:
  // 服务回调：重置估计器
  void reset_callback(
      const std::shared_ptr<rmw_request_id_t> request_header,
      const std::shared_ptr<std_srvs::srv::Empty::Request> request,
      const std::shared_ptr<std_srvs::srv::Empty::Response> response) {
    (void)request_header;
    (void)request;
    (void)response;

    RCLCPP_INFO(this->get_logger(), "Reset estimator service called");
    // 这里可以添加重置 VIO 估计器的代码
  }

  // 定时器回调
  void publish_timer_callback() {
    // 如果需要定期发布，可以在这里实现
  }

  // ROS2 发布器
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr
      pose_cov_pub_;

  // ROS2 订阅器
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr
      realsense_left_image_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr
      realsense_right_image_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;

  // TF 广播器
  std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

  // 服务
  rclcpp::Service<std_srvs::srv::Empty>::SharedPtr reset_service_;

  // 定时器
  rclcpp::TimerBase::SharedPtr publish_timer_;

  // 参数
  bool publish_odometry_;
  bool publish_tf_;
  bool publish_pose_;
  std::string odom_frame_id_;
  std::string base_frame_id_;
  std::string world_frame_id_;
  double publish_rate_;
  bool use_odometry_covariance_;
  std::string left_image_topic_;
  std::string right_image_topic_;
  std::string imu_topic_;

  // IMU 队列
  std::queue<basalt::ImuData<double>::Ptr> imu_queue_;
  std::mutex imu_mutex_;
};

// 全局 ROS2 节点指针
std::shared_ptr<BasaltVioNode> ros_node = nullptr;

// Feed functions
void feed_images() {
  std::cout << "Started image feed thread" << std::endl;

  int frame_count = 0;

  while (!vio->finished && !terminate) {
    if (step_by_step) {
      std::unique_lock<std::mutex> lk(m);
      cond_var.wait(lk);
    }

    // 等待立体图像
    StereoImage stereo(0);
    bool got_stereo = false;

    {
      std::unique_lock<std::mutex> lock(image_cache_mutex);

      // 查找已配对的立体图像
      auto it = stereo_image_cache.begin();
      while (it != stereo_image_cache.end()) {
        if (it->second.left_ready && it->second.right_ready) {
          stereo = it->second;
          got_stereo = true;

          // 移除已处理的图像
          it = stereo_image_cache.erase(it);
          break;
        } else {
          ++it;
        }
      }

      // 如果没有已配对的图像，等待
      if (!got_stereo) {
        image_cache_cv.wait_for(lock, std::chrono::milliseconds(10));
        continue;
      }
    }

    if (!got_stereo) {
      continue;
    }

    // 创建图像数据
    basalt::OpticalFlowInput::Ptr data(new basalt::OpticalFlowInput);
    data->t_ns = stereo.t_ns;
    data->img_data.resize(2);  // 双目相机

    // 左目图像
    if (!stereo.left_img.empty()) {
      // 转换为16位灰度图
      cv::Mat left_img_16bit;
      stereo.left_img.convertTo(left_img_16bit, CV_16UC1, 256.0);

      data->img_data[0].img.reset(new basalt::ManagedImage<uint16_t>(
          left_img_16bit.cols, left_img_16bit.rows));

      // 复制数据
      memcpy(data->img_data[0].img->ptr, left_img_16bit.data,
             left_img_16bit.cols * left_img_16bit.rows * sizeof(uint16_t));
    }

    // 右目图像
    if (!stereo.right_img.empty()) {
      // 转换为16位灰度图
      cv::Mat right_img_16bit;
      stereo.right_img.convertTo(right_img_16bit, CV_16UC1, 256.0);

      data->img_data[1].img.reset(new basalt::ManagedImage<uint16_t>(
          right_img_16bit.cols, right_img_16bit.rows));

      // 复制数据
      memcpy(data->img_data[1].img->ptr, right_img_16bit.data,
             right_img_16bit.cols * right_img_16bit.rows * sizeof(uint16_t));
    }

    // 记录时间戳到ID的映射
    timestamp_to_id[data->t_ns] = frame_count;
    frame_count++;

    // 更新显示帧数
    if (show_frame < 15000) {
      show_frame = frame_count;
      show_frame.Meta().gui_changed = true;
    }

    // 将图像数据送入光流模块
    opt_flow_ptr->input_queue.push(data);

    // 限制处理的帧数
    if (max_frames > 0 &&
        static_cast<int>(frame_count) >= static_cast<int>(max_frames)) {
      break;
    }
  }

  // 结束信号
  opt_flow_ptr->input_queue.push(nullptr);
  std::cout << "Finished image feed thread" << std::endl;
}

void feed_imu() {
  std::cout << "Started IMU feed thread" << std::endl;

  while (!vio->finished && !terminate) {
    // 从ROS节点获取IMU数据
    auto imu_data = ros_node->get_imu_data();

    if (imu_data) {
      vio->imu_data_queue.push(imu_data);
    } else {
      // 没有数据，短暂睡眠
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }

  // 结束信号
  vio->imu_data_queue.push(nullptr);
  std::cout << "Finished IMU feed thread" << std::endl;
}

int main(int argc, char** argv) {
  // 初始化 ROS2
  rclcpp::init(argc, argv);

  bool show_gui = true;
  bool print_queue = false;
  std::string cam_calib_path;
  std::string config_path;
  std::string result_path;
  std::string trajectory_fmt;
  bool trajectory_groundtruth = false;
  int num_threads = 0;
  bool use_imu = true;
  bool use_double = false;
  bool use_ros = true;
  std::string node_name = "basalt_vio";

  CLI::App app{"Basalt VIO with ROS2"};

  app.add_option("--show-gui", show_gui, "Show GUI");
  app.add_option("--cam-calib", cam_calib_path, "Camera calibration file.")
      ->required();

  app.add_option("--marg-data", marg_data_path,
                 "Path to folder where marginalization data will be stored.");

  app.add_option("--print-queue", print_queue, "Print queue.");
  app.add_option("--config-path", config_path, "Path to config file.");
  app.add_option("--result-path", result_path,
                 "Path to result file where the system will write RMSE ATE.");
  app.add_option("--num-threads", num_threads, "Number of threads.");
  app.add_option("--step-by-step", step_by_step, "Step by step mode.");
  app.add_option("--save-trajectory", trajectory_fmt,
                 "Save trajectory. Supported formats <tum, euroc, kitti>");
  app.add_option("--save-groundtruth", trajectory_groundtruth,
                 "In addition to trajectory, save also ground truth");
  app.add_option("--use-imu", use_imu, "Use IMU.");
  app.add_option("--use-double", use_double, "Use double not float.");
  app.add_option("--use-ros", use_ros,
                 "Use ROS2 for data input and publishing.");
  app.add_option("--node-name", node_name, "ROS2 node name.");
  app.add_option("--max-frames", max_frames,
                 "Limit number of frames to process (0 means unlimited)");

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError& e) {
    return app.exit(e);
  }

  // 创建 ROS2 节点
  if (use_ros) {
    ros_node = std::make_shared<BasaltVioNode>();
    RCLCPP_INFO(ros_node->get_logger(), "Basalt VIO ROS2 Node started");
  } else {
    std::cerr << "ROS2 must be enabled for this version" << std::endl;
    return 1;
  }

  // 设置线程数
  std::unique_ptr<tbb::global_control> tbb_global_control;
  if (num_threads > 0) {
    tbb_global_control = std::make_unique<tbb::global_control>(
        tbb::global_control::max_allowed_parallelism, num_threads);
  }

  if (!config_path.empty()) {
    vio_config.load(config_path);

    if (vio_config.vio_enforce_realtime) {
      vio_config.vio_enforce_realtime = false;
      std::cout
          << "The option vio_config.vio_enforce_realtime was enabled, "
             "but it should only be used with the live executables (supply "
             "images at a constant framerate). This executable runs on the "
             "datasets and processes images as fast as it can, so the option "
             "will be disabled. "
          << std::endl;
    }
  }

  // 加载相机标定
  load_data(cam_calib_path);

  // 创建光流模块
  opt_flow_ptr = basalt::OpticalFlowFactory::getOpticalFlow(vio_config, calib);

  // 创建VIO估计器
  const int64_t start_t_ns = 0;
  {
    vio = basalt::VioEstimatorFactory::getVioEstimator(
        vio_config, calib, basalt::constants::g, use_imu, use_double);
    vio->initialize(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());

    opt_flow_ptr->output_queue = &vio->vision_data_queue;
    if (show_gui) vio->out_vis_queue = &out_vis_queue;
    vio->out_state_queue = &out_state_queue;
  }

  basalt::MargDataSaver::Ptr marg_data_saver;

  if (!marg_data_path.empty()) {
    marg_data_saver.reset(new basalt::MargDataSaver(marg_data_path));
    vio->out_marg_queue = &marg_data_saver->in_marg_queue;
  }

  vio_data_log.Clear();

  // 启动数据feed线程
  std::thread t1(&feed_images);
  std::thread t2(&feed_imu);

  std::shared_ptr<std::thread> t3;

  if (show_gui)
    t3.reset(new std::thread([&]() {
      basalt::VioVisualizationData::Ptr data;

      while (true) {
        out_vis_queue.pop(data);

        if (data.get()) {
          vis_map[data->t_ns] = data;
          curr_vis_data = data;
        } else {
          break;
        }
      }

      std::cout << "Finished visualization thread" << std::endl;
    }));

  std::thread t4([&]() {
    basalt::PoseVelBiasState<double>::Ptr data;

    while (true) {
      out_state_queue.pop(data);

      if (!data.get()) break;

      int64_t t_ns = data->t_ns;

      Sophus::SE3d T_w_i = data->T_w_i;
      Eigen::Vector3d vel_w_i = data->vel_w_i;
      Eigen::Vector3d bg = data->bias_gyro;
      Eigen::Vector3d ba = data->bias_accel;

      vio_t_ns.emplace_back(data->t_ns);
      vio_t_w_i.emplace_back(T_w_i.translation());
      vio_T_w_i.emplace_back(T_w_i);

      // 发布 ROS2 里程计
      if (use_ros && ros_node) {
        ros_node->publish_odometry(T_w_i, vel_w_i, bg, ba, t_ns);
      }

      if (show_gui) {
        std::vector<float> vals;
        vals.push_back((t_ns - start_t_ns) * 1e-9);

        for (int i = 0; i < 3; i++) vals.push_back(vel_w_i[i]);
        for (int i = 0; i < 3; i++) vals.push_back(T_w_i.translation()[i]);
        for (int i = 0; i < 3; i++) vals.push_back(bg[i]);
        for (int i = 0; i < 3; i++) vals.push_back(ba[i]);

        vio_data_log.Log(vals);
      }
    }

    std::cout << "Finished state thread" << std::endl;
  });

  std::shared_ptr<std::thread> t5;

  auto print_queue_fn = [&]() {
    std::cout << "opt_flow_ptr->input_queue "
              << opt_flow_ptr->input_queue.size()
              << " opt_flow_ptr->output_queue "
              << opt_flow_ptr->output_queue->size() << " out_state_queue "
              << out_state_queue.size() << " imu_data_queue "
              << vio->imu_data_queue.size() << std::endl;
  };

  if (print_queue) {
    t5.reset(new std::thread([&]() {
      while (!terminate) {
        print_queue_fn();
        std::this_thread::sleep_for(std::chrono::seconds(1));
      }
    }));
  }

  auto time_start = std::chrono::high_resolution_clock::now();

  // 记录是否在VIO完成前关闭GUI
  bool aborted = false;

  if (show_gui) {
    pangolin::CreateWindowAndBind("Main", 1800, 1000);

    glEnable(GL_DEPTH_TEST);

    pangolin::View& main_display = pangolin::CreateDisplay().SetBounds(
        0.0, 1.0, pangolin::Attach::Pix(UI_WIDTH), 1.0);

    pangolin::View& img_view_display = pangolin::CreateDisplay()
                                           .SetBounds(0.4, 1.0, 0.0, 0.4)
                                           .SetLayout(pangolin::LayoutEqual);

    pangolin::View& plot_display = pangolin::CreateDisplay().SetBounds(
        0.0, 0.4, pangolin::Attach::Pix(UI_WIDTH), 1.0);

    plotter = new pangolin::Plotter(&imu_data_log, 0.0, 100, -10.0, 10.0, 0.01f,
                                    0.01f);
    plot_display.AddDisplay(*plotter);

    pangolin::CreatePanel("ui").SetBounds(0.0, 1.0, 0.0,
                                          pangolin::Attach::Pix(UI_WIDTH));

    std::vector<std::shared_ptr<pangolin::ImageView>> img_view;
    while (img_view.size() < calib.intrinsics.size()) {
      std::shared_ptr<pangolin::ImageView> iv(new pangolin::ImageView);

      size_t idx = img_view.size();
      img_view.push_back(iv);

      img_view_display.AddDisplay(*iv);
      iv->extern_draw_function =
          std::bind(&draw_image_overlay, std::placeholders::_1, idx);
    }

    Eigen::Vector3d cam_p(-0.5, -3, -5);
    cam_p = vio->getT_w_i_init().so3() * calib.T_i_c[0].so3() * cam_p;

    camera = pangolin::OpenGlRenderState(
        pangolin::ProjectionMatrix(640, 480, 400, 400, 320, 240, 0.001, 10000),
        pangolin::ModelViewLookAt(cam_p[0], cam_p[1], cam_p[2], 0, 0, 0,
                                  pangolin::AxisZ));

    pangolin::View& display3D =
        pangolin::CreateDisplay()
            .SetAspect(-640 / 480.0)
            .SetBounds(0.4, 1.0, 0.4, 1.0)
            .SetHandler(new pangolin::Handler3D(camera));

    display3D.extern_draw_function = draw_scene;

    main_display.AddDisplay(img_view_display);
    main_display.AddDisplay(display3D);

    // ROS2 旋转处理
    auto ros_spin_thread = std::thread([&]() {
      if (use_ros && ros_node) {
        rclcpp::spin(ros_node);
      }
    });

    // 用于显示的图像缓存
    std::vector<basalt::ManagedImage<uint16_t>> display_images(
        calib.intrinsics.size());

    while (!pangolin::ShouldQuit()) {
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

      if (follow) {
        if (curr_vis_data.get() && !curr_vis_data->states.empty()) {
          auto T_w_i = curr_vis_data->states.back();
          T_w_i.so3() = Sophus::SO3d();
          camera.Follow(T_w_i.matrix());
        }
      }

      display3D.Activate(camera);
      glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

      img_view_display.Activate();

      {
        pangolin::GlPixFormat fmt;
        fmt.glformat = GL_LUMINANCE;
        fmt.gltype = GL_UNSIGNED_SHORT;
        fmt.scalable_internal_format = GL_LUMINANCE16;

        if (curr_vis_data.get() && curr_vis_data->opt_flow_res.get() &&
            curr_vis_data->opt_flow_res->input_images.get()) {
          auto& img_data = curr_vis_data->opt_flow_res->input_images->img_data;

          for (size_t cam_id = 0; cam_id < calib.intrinsics.size(); cam_id++) {
            if (img_data[cam_id].img.get())
              img_view[cam_id]->SetImage(
                  img_data[cam_id].img->ptr, img_data[cam_id].img->w,
                  img_data[cam_id].img->h, img_data[cam_id].img->pitch, fmt);
          }
        }

        draw_plots();
      }

      if (show_est_vel.GuiChanged() || show_est_pos.GuiChanged() ||
          show_est_ba.GuiChanged() || show_est_bg.GuiChanged()) {
        draw_plots();
      }

      if (euroc_fmt.GuiChanged()) {
        euroc_fmt = true;
        tum_rgbd_fmt = false;
        kitti_fmt = false;
      }

      if (tum_rgbd_fmt.GuiChanged()) {
        tum_rgbd_fmt = true;
        euroc_fmt = false;
        kitti_fmt = false;
      }

      if (kitti_fmt.GuiChanged()) {
        kitti_fmt = true;
        euroc_fmt = false;
        tum_rgbd_fmt = false;
      }

      //      if (record) {
      //        main_display.RecordOnRender(
      //            "ffmpeg:[fps=50,bps=80000000,unique_filename]///tmp/"
      //            "vio_screencap.avi");
      //        record = false;
      //      }

      pangolin::FinishFrame();

      if (continue_btn) {
        if (!next_step())
          std::this_thread::sleep_for(std::chrono::milliseconds(50));
      } else {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
      }
    }

    // 停止 ROS2
    rclcpp::shutdown();
    if (ros_spin_thread.joinable()) {
      ros_spin_thread.join();
    }

    // 如果GUI关闭但VIO仍在运行，则中止
    if (!vio->finished) {
      std::cout << "GUI closed but odometry still running --> aborting.\n";
      print_queue_fn();
      terminate = true;
      aborted = true;
    }
  } else {
    // 如果没有GUI，单独运行 ROS2
    if (use_ros && ros_node) {
      // rclcpp::spin(ros_node);
      
      // 创建执行器而不是直接使用spin
      rclcpp::executors::SingleThreadedExecutor executor;
      executor.add_node(ros_node);
      
      // 循环检查退出条件
      while (rclcpp::ok() && !terminate) {
        // 非阻塞方式处理ROS2消息，每次处理100ms
        executor.spin_once(std::chrono::milliseconds(100));
      }
      print_queue_fn();
      terminate = true;
      aborted = true;
      saveTrajectoryButton();
      RCLCPP_INFO(ros_node->get_logger(), "ROS2节点正在关闭...");
    }
  }

  // 等待VIO完成处理
  vio->maybe_join();

  // 清空输入队列
  vio->drain_input_queues();

  // 结束输入线程
  terminate = true;
  t1.join();
  t2.join();

  // 结束其他线程
  if (t3) t3->join();
  t4.join();
  if (t5) t5->join();

  // 打印最终队列大小
  if (print_queue) {
    std::cout << "Final queue sizes:" << std::endl;
    print_queue_fn();
  }

  auto time_end = std::chrono::high_resolution_clock::now();
  const double duration_total =
      std::chrono::duration<double>(time_end - time_start).count();

  vio->debug_finalize();
  std::cout << "Total runtime: {:.3f}s\n"_format(duration_total);

  {
    basalt::ExecutionStats stats;
    stats.add("exec_time_s", duration_total);
    stats.add("num_frames", vio_t_w_i.size());

    {
      basalt::MemoryInfo mi;
      if (get_memory_info(mi)) {
        stats.add("resident_memory_peak", mi.resident_memory_peak);
      }
    }

    stats.save_json("stats_vio.json");
  }

  if (!aborted && !trajectory_fmt.empty()) {
    std::cout << "Saving trajectory..." << std::endl;

    if (trajectory_fmt == "kitti") {
      kitti_fmt = true;
      euroc_fmt = false;
      tum_rgbd_fmt = false;
    }
    if (trajectory_fmt == "euroc") {
      euroc_fmt = true;
      kitti_fmt = false;
      tum_rgbd_fmt = false;
    }
    if (trajectory_fmt == "tum") {
      tum_rgbd_fmt = true;
      euroc_fmt = false;
      kitti_fmt = false;
    }

    save_groundtruth = trajectory_groundtruth;

    saveTrajectoryButton();
  }

  if (!aborted && !result_path.empty()) {
    auto exec_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        time_end - time_start);

    std::ofstream os(result_path);
    {
      cereal::JSONOutputArchive ar(os);
      ar(cereal::make_nvp("num_frames", vio_t_w_i.size()));
      ar(cereal::make_nvp("exec_time_ns", exec_time_ns.count()));
    }
    os.close();
  }

  return 0;
}

void draw_image_overlay(pangolin::View& v, size_t cam_id) {
  UNUSED(v);

  if (show_obs) {
    glLineWidth(1.0);
    glColor3f(1.0, 0.0, 0.0);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    if (curr_vis_data.get() && cam_id < curr_vis_data->projections.size()) {
      const auto& points = curr_vis_data->projections[cam_id];

      if (!points.empty()) {
        double min_id = points[0][2], max_id = points[0][2];

        for (const auto& points2 : curr_vis_data->projections)
          for (const auto& p : points2) {
            min_id = std::min(min_id, p[2]);
            max_id = std::max(max_id, p[2]);
          }

        for (const auto& c : points) {
          const float radius = 6.5;

          float r, g, b;
          getcolor(c[2] - min_id, max_id - min_id, b, g, r);
          glColor3f(r, g, b);

          pangolin::glDrawCirclePerimeter(c[0], c[1], radius);

          if (show_ids)
            pangolin::default_font().Text("%d", int(c[3])).Draw(c[0], c[1]);
        }
      }

      glColor3f(1.0, 0.0, 0.0);
      pangolin::default_font()
          .Text("Tracked %d points", points.size())
          .Draw(5, 20);
    }
  }

  if (show_flow) {
    glLineWidth(1.0);
    glColor3f(1.0, 0.0, 0.0);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    if (curr_vis_data.get() && curr_vis_data->opt_flow_res.get()) {
      const Eigen::aligned_map<basalt::KeypointId, Eigen::AffineCompact2f>&
          kp_map = curr_vis_data->opt_flow_res->observations[cam_id];

      for (const auto& kv : kp_map) {
        Eigen::MatrixXf transformed_patch =
            kv.second.linear() * opt_flow_ptr->patch_coord;
        transformed_patch.colwise() += kv.second.translation();

        for (int i = 0; i < transformed_patch.cols(); i++) {
          const Eigen::Vector2f c = transformed_patch.col(i);
          pangolin::glDrawCirclePerimeter(c[0], c[1], 0.5f);
        }

        const Eigen::Vector2f c = kv.second.translation();

        if (show_ids)
          pangolin::default_font()
              .Text("%d", kv.first)
              .Draw(5 + c[0], 5 + c[1]);
      }

      pangolin::default_font()
          .Text("%d opt_flow patches", kp_map.size())
          .Draw(5, 20);
    }
  }
}

void draw_scene(pangolin::View& view) {
  UNUSED(view);
  view.Activate(camera);
  glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

  glPointSize(3);
  glColor3f(1.0, 0.0, 0.0);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  glColor3ubv(cam_color);
  if (!vio_t_w_i.empty()) {
    pangolin::glDrawLineStrip(vio_t_w_i);
  }

  glColor3ubv(gt_color);
  if (show_gt) pangolin::glDrawLineStrip(gt_t_w_i);

  if (curr_vis_data.get()) {
    for (size_t i = 0; i < calib.T_i_c.size(); i++)
      if (!curr_vis_data->states.empty()) {
        render_camera(
            (curr_vis_data->states.back() * calib.T_i_c[i]).matrix(), 2.0f,
            cam_color, 0.1f);
      } else if (!curr_vis_data->frames.empty()) {
        render_camera(
            (curr_vis_data->frames.back() * calib.T_i_c[i]).matrix(), 2.0f,
            cam_color, 0.1f);
      }

    for (const auto& p : curr_vis_data->states)
      for (size_t i = 0; i < calib.T_i_c.size(); i++)
        render_camera((p * calib.T_i_c[i]).matrix(), 2.0f, state_color, 0.1f);

    for (const auto& p : curr_vis_data->frames)
      for (size_t i = 0; i < calib.T_i_c.size(); i++)
        render_camera((p * calib.T_i_c[i]).matrix(), 2.0f, pose_color, 0.1f);

    glColor3ubv(pose_color);
    pangolin::glDrawPoints(curr_vis_data->points);
  }

  pangolin::glDrawAxis(Sophus::SE3d().matrix(), 1.0);
}

void load_data(const std::string& calib_path) {
  std::ifstream os(calib_path, std::ios::binary);

  if (os.is_open()) {
    cereal::JSONInputArchive archive(os);
    archive(calib);
    std::cout << "Loaded camera with " << calib.intrinsics.size() << " cameras"
              << std::endl;

  } else {
    std::cerr << "could not load camera calibration " << calib_path
              << std::endl;
    std::abort();
  }
}

bool next_step() {
  if (show_frame < 1500) {
    show_frame = show_frame + 1;
    show_frame.Meta().gui_changed = true;
    cond_var.notify_one();
    return true;
  } else {
    return false;
  }
}

bool prev_step() {
  if (show_frame > 1) {
    show_frame = show_frame - 1;
    show_frame.Meta().gui_changed = true;
    return true;
  } else {
    return false;
  }
}

void draw_plots() {
  plotter->ClearSeries();
  plotter->ClearMarkers();

  if (show_est_pos) {
    plotter->AddSeries("$0", "$4", pangolin::DrawingModeLine,
                       pangolin::Colour::Red(), "position x", &vio_data_log);
    plotter->AddSeries("$0", "$5", pangolin::DrawingModeLine,
                       pangolin::Colour::Green(), "position y", &vio_data_log);
    plotter->AddSeries("$0", "$6", pangolin::DrawingModeLine,
                       pangolin::Colour::Blue(), "position z", &vio_data_log);
  }

  if (show_est_vel) {
    plotter->AddSeries("$0", "$1", pangolin::DrawingModeLine,
                       pangolin::Colour::Red(), "velocity x", &vio_data_log);
    plotter->AddSeries("$0", "$2", pangolin::DrawingModeLine,
                       pangolin::Colour::Green(), "velocity y", &vio_data_log);
    plotter->AddSeries("$0", "$3", pangolin::DrawingModeLine,
                       pangolin::Colour::Blue(), "velocity z", &vio_data_log);
  }

  if (show_est_bg) {
    plotter->AddSeries("$0", "$7", pangolin::DrawingModeLine,
                       pangolin::Colour::Red(), "gyro bias x", &vio_data_log);
    plotter->AddSeries("$0", "$8", pangolin::DrawingModeLine,
                       pangolin::Colour::Green(), "gyro bias y", &vio_data_log);
    plotter->AddSeries("$0", "$9", pangolin::DrawingModeLine,
                       pangolin::Colour::Blue(), "gyro bias z", &vio_data_log);
  }

  if (show_est_ba) {
    plotter->AddSeries("$0", "$10", pangolin::DrawingModeLine,
                       pangolin::Colour::Red(), "accel bias x", &vio_data_log);
    plotter->AddSeries("$0", "$11", pangolin::DrawingModeLine,
                       pangolin::Colour::Green(), "accel bias y",
                       &vio_data_log);
    plotter->AddSeries("$0", "$12", pangolin::DrawingModeLine,
                       pangolin::Colour::Blue(), "accel bias z", &vio_data_log);
  }

  if (!vio_t_ns.empty() && static_cast<size_t>(show_frame) < vio_t_ns.size()) {
    double t = vio_t_ns[show_frame] * 1e-9;
    plotter->AddMarker(pangolin::Marker::Vertical, t, pangolin::Marker::Equal,
                       pangolin::Colour::White());
  }
}

void alignButton() {
  // 实时模式下，不需要对齐地面真值
  std::cout << "Alignment not available in real-time mode" << std::endl;
}

void saveTrajectoryButton() {
  if (tum_rgbd_fmt) {
    {
      std::ofstream os("trajectory.txt");

      os << "# timestamp tx ty tz qx qy qz qw" << std::endl;

      for (size_t i = 0; i < vio_t_ns.size(); i++) {
        const Sophus::SE3d& pose = vio_T_w_i[i];
        os << std::scientific << std::setprecision(18) << vio_t_ns[i] * 1e-9
           << " " << pose.translation().x() << " " << pose.translation().y()
           << " " << pose.translation().z() << " " << pose.unit_quaternion().x()
           << " " << pose.unit_quaternion().y() << " "
           << pose.unit_quaternion().z() << " " << pose.unit_quaternion().w()
           << std::endl;
      }

      os.close();
    }

    std::cout
        << "Saved trajectory in TUM RGB-D Dataset format in trajectory.txt"
        << std::endl;
  } else if (euroc_fmt) {
    std::ofstream os("trajectory.csv");

    os << "#timestamp [ns],p_RS_R_x [m],p_RS_R_y [m],p_RS_R_z [m],q_RS_w "
          "[],q_RS_x [],q_RS_y [],q_RS_z []"
       << std::endl;

    for (size_t i = 0; i < vio_t_ns.size(); i++) {
      const Sophus::SE3d& pose = vio_T_w_i[i];
      os << std::scientific << std::setprecision(18) << vio_t_ns[i] << ","
         << pose.translation().x() << "," << pose.translation().y() << ","
         << pose.translation().z() << "," << pose.unit_quaternion().w() << ","
         << pose.unit_quaternion().x() << "," << pose.unit_quaternion().y()
         << "," << pose.unit_quaternion().z() << std::endl;
    }

    os.close();

    std::cout << "Saved trajectory in Euroc Dataset format in trajectory.csv"
              << std::endl;
  } else {
    std::ofstream os("trajectory_kitti.txt");

    for (size_t i = 0; i < vio_t_ns.size(); i++) {
      Eigen::Matrix<double, 3, 4> mat = vio_T_w_i[i].matrix3x4();
      os << std::scientific << std::setprecision(12) << mat.row(0) << " "
         << mat.row(1) << " " << mat.row(2) << " " << std::endl;
    }

    os.close();

    std::cout
        << "Saved trajectory in KITTI Dataset format in trajectory_kitti.txt"
        << std::endl;
  }
}
