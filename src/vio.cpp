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

/**
 • Basalt VIO 系统主程序

 • 功能：基于优化的视觉惯性里程计系统，包含双目视觉前端+IMU紧耦合后端优化

 • 特点：支持数据集离线处理、实时显示、轨迹保存和性能分析

 */

// C++基础库
#include <algorithm>            // 算法库
#include <chrono>               // 时间库
#include <condition_variable>   // 线程同步
#include <iostream>             // 输入输出
#include <thread>               // 多线程


#include <fmt/format.h>         // 字符串格式化
#include <sophus/se3.hpp>       // 李群SE3操作

// 并行计算（tbb）
#include <tbb/concurrent_unordered_map.h> // 线程安全哈希表
#include <tbb/global_control.h>           // TBB全局控制

// 可视化库 (pangolin)
#include <pangolin/display/default_font.h>
#include <pangolin/display/image_view.h>    // 图像显示
#include <pangolin/gl/gldraw.h>             // OpenGL绘制
#include <pangolin/image/image.h>           // 图像处理
#include <pangolin/image/image_io.h>
#include <pangolin/image/typed_image.h>
#include <pangolin/pangolin.h>              // 主库

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

pangolin::Var<int> show_frame("ui.show_frame", 0, 0, 1500);

pangolin::Var<bool> show_flow("ui.show_flow", false, false, true);
pangolin::Var<bool> show_obs("ui.show_obs", true, false, true);
pangolin::Var<bool> show_ids("ui.show_ids", false, false, true);

pangolin::Var<bool> show_est_pos("ui.show_est_pos", true, false, true);
pangolin::Var<bool> show_est_vel("ui.show_est_vel", false, false, true);
pangolin::Var<bool> show_est_bg("ui.show_est_bg", false, false, true);
pangolin::Var<bool> show_est_ba("ui.show_est_ba", false, false, true);

pangolin::Var<bool> show_gt("ui.show_gt", true, false, true);

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
std::unordered_map<int64_t, basalt::VioVisualizationData::Ptr> vis_map;                 // 可视化数据映射表

// 线程安全队列
tbb::concurrent_bounded_queue<basalt::VioVisualizationData::Ptr> out_vis_queue;         // 可视化队列
tbb::concurrent_bounded_queue<basalt::PoseVelBiasState<double>::Ptr> out_state_queue;   // 状态队列

// 轨迹数据存储
std::vector<int64_t> vio_t_ns;                      // 时间戳(ns)
Eigen::aligned_vector<Eigen::Vector3d> vio_t_w_i;   // 位置轨迹
Eigen::aligned_vector<Sophus::SE3d> vio_T_w_i;      // 位姿轨迹

std::vector<int64_t> gt_t_ns;
Eigen::aligned_vector<Eigen::Vector3d> gt_t_w_i;

std::string marg_data_path;
size_t last_frame_processed = 0;

// 基于时间戳查找相机frame_id的哈希表
tbb::concurrent_unordered_map<int64_t, int, std::hash<int64_t>> timestamp_to_id;

std::mutex m;
std::condition_variable cv;
bool step_by_step = false;
size_t max_frames = 0;

std::atomic<bool> terminate = false;

// VIO variables
basalt::Calibration<double> calib;            // 相机-IMU标定参数

basalt::VioDatasetPtr vio_dataset;            // 数据集接口
basalt::VioConfig vio_config;                 // VIO配置参数
basalt::OpticalFlowBase::Ptr opt_flow_ptr;    // 光流前端
basalt::VioEstimatorBase::Ptr vio;            // VIO估计器

// Feed functions
// 读取图像数据，喂给光流前端
void feed_images() {
  std::cout << "Started input_data thread " << std::endl;


  for (size_t i = 0; i < vio_dataset->get_image_timestamps().size(); i++) {
    if (vio->finished || terminate || (max_frames > 0 && i >= max_frames)) {
      // stop loop early if we set a limit on number of frames to process
      break;
    }

    // 一步一步调试：只有在UI界面上点击next_step时，才会到下一帧
    if (step_by_step) {
      std::unique_lock<std::mutex> lk(m);
      cv.wait(lk);
    }

    basalt::OpticalFlowInput::Ptr data(new basalt::OpticalFlowInput);

    // 获取图像时间戳
    data->t_ns = vio_dataset->get_image_timestamps()[i];

    // 获取图像cv::Mat数据（双目）
    data->img_data = vio_dataset->get_image_data(data->t_ns);

    timestamp_to_id[data->t_ns] = i;
    
    opt_flow_ptr->input_queue.push(data);
  }

  // Indicate the end of the sequence
  opt_flow_ptr->input_queue.push(nullptr);

  std::cout << "Finished input_data thread " << std::endl;
}

// 读取IMU数据，喂给VIO估计器的IMU数据队列，vio->imu_data_queue
void feed_imu() {
  for (size_t i = 0; i < vio_dataset->get_gyro_data().size(); i++) {
    if (vio->finished || terminate) {
      break;
    }

    basalt::ImuData<double>::Ptr data(new basalt::ImuData<double>);
    data->t_ns = vio_dataset->get_gyro_data()[i].timestamp_ns;

    data->accel = vio_dataset->get_accel_data()[i].data;
    data->gyro = vio_dataset->get_gyro_data()[i].data;

    vio->imu_data_queue.push(data);
  }
  vio->imu_data_queue.push(nullptr);
}

int main(int argc, char** argv) {
  // 定义命令行参数变量
  bool show_gui = true;         // 是否显示GUI界面
  bool print_queue = false;     // 是否打印队列信息
  std::string cam_calib_path;   // 相机标定文件路径
  std::string dataset_path;     // 数据集路径
  std::string dataset_type;     // 数据集类型
  std::string config_path;      // 配置文件路径
  std::string result_path;      // 结果文件路径
  std::string trajectory_fmt;   // 轨迹保存格式
  bool trajectory_groundtruth;  // 是否保存真值轨迹
  int num_threads = 0;          // 线程数量
  bool use_imu = true;          // 是否使用IMU
  bool use_double = false;      // 是否使用双精度浮点数

  // 初始化CLI11命令行参数解析器
  CLI::App app{"App description"};

  // 添加命令行参数选项
  app.add_option("--show-gui", show_gui, "Show GUI");
  app.add_option("--cam-calib", cam_calib_path,
                 "Ground-truth camera calibration used for simulation.")
      ->required();

  app.add_option("--dataset-path", dataset_path, "Path to dataset.")
      ->required();

  app.add_option("--dataset-type", dataset_type, "Dataset type <euroc, bag>.")
      ->required();

  app.add_option("--marg-data", marg_data_path,
                 "Path to folder where marginalization data will be stored.");

  app.add_option("--print-queue", print_queue, "Print queue.");
  app.add_option("--config-path", config_path, "Path to config file.");
  app.add_option("--result-path", result_path,
                 "Path to result file where the system will write RMSE ATE.");
  app.add_option("--num-threads", num_threads, "Number of threads.");
  app.add_option("--step-by-step", step_by_step, "Path to config file.");
  app.add_option("--save-trajectory", trajectory_fmt,
                 "Save trajectory. Supported formats <tum, euroc, kitti>");
  app.add_option("--save-groundtruth", trajectory_groundtruth,
                 "In addition to trajectory, save also ground turth");
  app.add_option("--use-imu", use_imu, "Use IMU.");
  app.add_option("--use-double", use_double, "Use double not float.");
  app.add_option(
      "--max-frames", max_frames,
      "Limit number of frames to process from dataset (0 means unlimited)");

  
  // 解析命令行参数
  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError& e) {
    return app.exit(e);
  }

  // global thread limit is in effect until global_control object is destroyed
  // 设置TBB全局线程控制 
  std::unique_ptr<tbb::global_control> tbb_global_control;
  if (num_threads > 0) {
    tbb_global_control = std::make_unique<tbb::global_control>(
        tbb::global_control::max_allowed_parallelism, num_threads);
  }

  // 加载配置文件（如果提供）
  if (!config_path.empty()) {
    vio_config.load(config_path);
    
    // 调整实时性设置（数据集处理不需要强制实时）
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

  // 加载相机标定数据
  load_data(cam_calib_path);

  // 初始化数据集读取器
  {
    // 读取数据类型
    basalt::DatasetIoInterfacePtr dataset_io =
        basalt::DatasetIoFactory::getDatasetIo(dataset_type);

    // 读取数据集
    // 图像： dataset_io->data->image_timestamps（时间戳）+ dataset_io->data->image_path（文件名）
    // IMU： dataset_io->data->accel_data（加速度计）+ dataset_io->data->gyro_data（陀螺仪）
    dataset_io->read(dataset_path);
    
    // 获取数据集对象
    // vio_dataset = dataset_io->data
    vio_dataset = dataset_io->get_data();
    
    // 设置GUI中显示帧数的范围
    show_frame.Meta().range[1] = vio_dataset->get_image_timestamps().size() - 1;
    show_frame.Meta().gui_changed = true;

    // 创建光流处理器
    opt_flow_ptr =
        basalt::OpticalFlowFactory::getOpticalFlow(vio_config, calib);

    // 提取真值轨迹数据用于后续评估
    for (size_t i = 0; i < vio_dataset->get_gt_pose_data().size(); i++) {
      gt_t_ns.push_back(vio_dataset->get_gt_timestamps()[i]);
      gt_t_w_i.push_back(vio_dataset->get_gt_pose_data()[i].translation());
    }
  }

  // 获取起始时间戳
  const int64_t start_t_ns = vio_dataset->get_image_timestamps().front();

  // 创建VIO估计器
  {
    vio = basalt::VioEstimatorFactory::getVioEstimator(
        vio_config, calib, basalt::constants::g, use_imu, use_double);
    vio->initialize(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());  // 初始化VIO

    // 连接数据处理管道
    opt_flow_ptr->output_queue = &vio->vision_data_queue; // 光流输出到VIO
    if (show_gui) vio->out_vis_queue = &out_vis_queue;    // 可视化输出队列
    vio->out_state_queue = &out_state_queue;              // 状态输出队列
  }

  // 边缘化数据保存器
  basalt::MargDataSaver::Ptr marg_data_saver;

  if (!marg_data_path.empty()) {
    marg_data_saver.reset(new basalt::MargDataSaver(marg_data_path));
    vio->out_marg_queue = &marg_data_saver->in_marg_queue;  // 连接边缘化队列

    // Save gt.
    // 保存真值数据到文件
    {
      std::string p = marg_data_path + "/gt.cereal";
      std::ofstream os(p, std::ios::binary);

      {
        cereal::BinaryOutputArchive archive(os);  // 使用cereal序列化库
        archive(gt_t_ns);                         // 序列化时间戳
        archive(gt_t_w_i);                        // 序列化真值轨迹
      }
      os.close();                                 // 关闭文件
    }
  }

  // 清空VIO数据日志
  vio_data_log.Clear();

  // 启动数据处理线程
  std::thread t1(&feed_images); // 图像馈送线程
  std::thread t2(&feed_imu);    // IMU馈送线程

  // 可视化数据处理线程（仅在GUI模式下启动）
  std::shared_ptr<std::thread> t3;

  if (show_gui)
    t3.reset(new std::thread([&]() {
      basalt::VioVisualizationData::Ptr data;
      
      // 循环处理可视化数据
      while (true) {
        out_vis_queue.pop(data);        // 从队列获取数据

        if (data.get()) {
          vis_map[data->t_ns] = data;   // 存储到可视化映射表
        } else {
          break;                        // 接收到空指针表示结束
        }
      }

      std::cout << "Finished t3" << std::endl;
    }));

  // 状态数据处理线程
  std::thread t4([&]() {
    basalt::PoseVelBiasState<double>::Ptr data;

    while (true) {
      out_state_queue.pop(data);    // 从队列获取状态数据

      if (!data.get()) break;       // 结束信号

      int64_t t_ns = data->t_ns;    // 提取时间戳

      // std::cerr << "t_ns " << t_ns << std::endl;
      // 提取状态信息
      Sophus::SE3d T_w_i = data->T_w_i;           // 位姿
      Eigen::Vector3d vel_w_i = data->vel_w_i;    // 速度
      Eigen::Vector3d bg = data->bias_gyro;       // 陀螺偏置
      Eigen::Vector3d ba = data->bias_accel;      // 加速度计偏置
      
      // 保存轨迹数据
      vio_t_ns.emplace_back(data->t_ns);
      vio_t_w_i.emplace_back(T_w_i.translation());
      vio_T_w_i.emplace_back(T_w_i);

      // GUI模式下记录数据日志
      if (show_gui) {
        std::vector<float> vals;
        vals.push_back((t_ns - start_t_ns) * 1e-9);                             // 相对时间（秒）

        // 添加状态数据到日志
        for (int i = 0; i < 3; i++) vals.push_back(vel_w_i[i]);                 // 速度
        for (int i = 0; i < 3; i++) vals.push_back(T_w_i.translation()[i]);     // 位置    
        for (int i = 0; i < 3; i++) vals.push_back(bg[i]);                      // 陀螺偏置
        for (int i = 0; i < 3; i++) vals.push_back(ba[i]);                      // 加速度偏置

        vio_data_log.Log(vals);                                                 // 记录日志
      }
    }

    std::cout << "Finished t4" << std::endl;
  });

  // 队列监控线程
  std::shared_ptr<std::thread> t5;

  // 定义队列打印函数
  auto print_queue_fn = [&]() {
    std::cout << "opt_flow_ptr->input_queue "
              << opt_flow_ptr->input_queue.size()
              << " opt_flow_ptr->output_queue "
              << opt_flow_ptr->output_queue->size() << " out_state_queue "
              << out_state_queue.size() << " imu_data_queue "
              << vio->imu_data_queue.size() << std::endl;
  };

  // 如果需要打印队列信息，启动监控线程
  if (print_queue) {
    t5.reset(new std::thread([&]() {
      while (!terminate) {
        print_queue_fn();
        std::this_thread::sleep_for(std::chrono::seconds(1));
      }
    }));
  }

  // 记录开始时间用于性能统计
  auto time_start = std::chrono::high_resolution_clock::now();

  // record if we close the GUI before VIO is finished.
  // 标记是否提前终止
  bool aborted = false;

  // GUI模式主循环
  if (show_gui) {
    // 创建Pangolin窗口
    pangolin::CreateWindowAndBind("Main", 1800, 1000);

    // 启用深度测试
    glEnable(GL_DEPTH_TEST);

    // 创建多视图布局
    // 创建主显示区域
    pangolin::View& main_display = pangolin::CreateDisplay().SetBounds(
        0.0, 1.0, pangolin::Attach::Pix(UI_WIDTH), 1.0);
    
    // 创建图像显示区域
    pangolin::View& img_view_display = pangolin::CreateDisplay()
                                           .SetBounds(0.4, 1.0, 0.0, 0.4)
                                           .SetLayout(pangolin::LayoutEqual);
    
    // 创建曲线图显示区域
    pangolin::View& plot_display = pangolin::CreateDisplay().SetBounds(
        0.0, 0.4, pangolin::Attach::Pix(UI_WIDTH), 1.0);

    // 创建曲线绘制器
    plotter = new pangolin::Plotter(&imu_data_log, 0.0, 100, -10.0, 10.0, 0.01f,
                                    0.01f);
    // 添加到显示区域
    plot_display.AddDisplay(*plotter);

    // 创建控制面板
    pangolin::CreatePanel("ui").SetBounds(0.0, 1.0, 0.0,
                                          pangolin::Attach::Pix(UI_WIDTH));

    // 创建图像视图用于显示相机图像
    std::vector<std::shared_ptr<pangolin::ImageView>> img_view;

    while (img_view.size() < calib.intrinsics.size()) {
      std::shared_ptr<pangolin::ImageView> iv(new pangolin::ImageView);

      size_t idx = img_view.size();
      img_view.push_back(iv);

      img_view_display.AddDisplay(*iv); // 添加到图像显示区域

      // 设置图像叠加绘制函数
      iv->extern_draw_function =
          std::bind(&draw_image_overlay, std::placeholders::_1, idx);
    }

    // 设置3D相机初始位置
    Eigen::Vector3d cam_p(-0.5, -3, -5);
    cam_p = vio->getT_w_i_init().so3() * calib.T_i_c[0].so3() * cam_p;

    // 创建OpenGL渲染状态
    camera = pangolin::OpenGlRenderState(
        pangolin::ProjectionMatrix(640, 480, 400, 400, 320, 240, 0.001, 10000),
        pangolin::ModelViewLookAt(cam_p[0], cam_p[1], cam_p[2], 0, 0, 0,
                                  pangolin::AxisZ));
    
    // 创建3D场景显示视图
    pangolin::View& display3D =
        pangolin::CreateDisplay()
            .SetAspect(-640 / 480.0)
            .SetBounds(0.4, 1.0, 0.4, 1.0)
            .SetHandler(new pangolin::Handler3D(camera));
    
    // 设置场景绘制函数
    display3D.extern_draw_function = draw_scene;

    // 组合显示布局
    main_display.AddDisplay(img_view_display);
    main_display.AddDisplay(display3D);

    // GUI主循环
    while (!pangolin::ShouldQuit()) {
      // 清空缓冲区
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

      // 相机跟随模式
      if (follow) {
        size_t frame_id = show_frame;
        int64_t t_ns = vio_dataset->get_image_timestamps()[frame_id];
        auto it = vis_map.find(t_ns);

        if (it != vis_map.end()) {
          Sophus::SE3d T_w_i;
          // 获取当前帧的位姿
          if (!it->second->states.empty()) {
            T_w_i = it->second->states.back();
          } else if (!it->second->frames.empty()) {
            T_w_i = it->second->frames.back();
          }

          // 重置旋转部分
          T_w_i.so3() = Sophus::SO3d();
          
          // 相机跟随位姿
          camera.Follow(T_w_i.matrix());
        }
      }

      // 激活3D视图
      display3D.Activate(camera);

      // 设置清屏颜色为白色
      glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

      // 激活图像视图
      img_view_display.Activate();

      // 处理帧切换事件
      if (show_frame.GuiChanged()) {
        for (size_t cam_id = 0; cam_id < calib.intrinsics.size(); cam_id++) {
          size_t frame_id = static_cast<size_t>(show_frame);
          int64_t timestamp = vio_dataset->get_image_timestamps()[frame_id];

          // 获取对应时间戳的图像数据
          std::vector<basalt::ImageData> img_vec =
              vio_dataset->get_image_data(timestamp);

          // 设置OpenGL像素格式
          pangolin::GlPixFormat fmt;
          fmt.glformat = GL_LUMINANCE;                    // 亮度格式
          fmt.gltype = GL_UNSIGNED_SHORT;                 // 无符号短整型
          fmt.scalable_internal_format = GL_LUMINANCE16;  // 16位亮度
          
          // 更新图像显示
          if (img_vec[cam_id].img.get())
            img_view[cam_id]->SetImage(
                img_vec[cam_id].img->ptr, img_vec[cam_id].img->w,
                img_vec[cam_id].img->h, img_vec[cam_id].img->pitch, fmt);
        }
        
        // 更新曲线图
        draw_plots();
      }
      
      // 处理状态显示选项变化
      if (show_est_vel.GuiChanged() || show_est_pos.GuiChanged() ||
          show_est_ba.GuiChanged() || show_est_bg.GuiChanged()) {
        draw_plots(); // 重新绘制曲线
      }

      // 处理轨迹格式切换（互斥选择）
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

      // 录制功能（已注释）
      //      if (record) {
      //        main_display.RecordOnRender(
      //            "ffmpeg:[fps=50,bps=80000000,unique_filename]///tmp/"
      //            "vio_screencap.avi");
      //        record = false;
      //      }

      // 结束当前帧渲染
      pangolin::FinishFrame();

      // 处理连续运行模式
      if (continue_btn) {
        // 执行单步前进
        if (!next_step())
          std::this_thread::sleep_for(std::chrono::milliseconds(50));
      } else {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
      }

      // 快速连续模式
      if (continue_fast) {
        int64_t t_ns = vio->last_processed_t_ns;  // 获取最后处理的时间戳
        if (timestamp_to_id.count(t_ns)) {
          show_frame = timestamp_to_id[t_ns];     // 更新当前显示帧
          show_frame.Meta().gui_changed = true;
        }

        // VIO处理完成时停止快速模式
        if (vio->finished) {
          continue_fast = false;
        }
      }
    }

    // If GUI closed but VIO not yet finished --> abort input queues, which in
    // turn aborts processing
    // GUI关闭但VIO未完成时的处理
    if (!vio->finished) {
      std::cout << "GUI closed but odometry still running --> aborting.\n";
      
      // 打印队列状态
      print_queue_fn();  // print queue size at time of aborting
      
      // 设置终止标志
      terminate = true;
      
      // 标记为提前终止
      aborted = true;
    }
  }

  // wait first for vio to complete processing
  // 等待VIO处理完成
  vio->maybe_join();

  // input threads will abort when vio is finished, but might be stuck in full
  // push to full queue, so drain queue now
  // 清空输入队列以确保线程能正常退出
  vio->drain_input_queues();

  // join input threads
  // 等待输入线程结束
  t1.join();  // 图像线程
  t2.join();  // IMU线程

  // std::cout << "Data input finished, terminate auxiliary threads.";
  // 设置终止标志并等待其他线程
  terminate = true;

  // join other threads
  // 等待其他线程结束
  if (t3) t3->join();   // 可视化线程
  t4.join();            // 状态处理线程
  if (t5) t5->join();   // 队列监控线程

  // after joining all threads, print final queue sizes.
  // 打印最终队列状态
  if (print_queue) {
    std::cout << "Final queue sizes:" << std::endl;
    print_queue_fn();
  }

  // 计算总运行时间
  auto time_end = std::chrono::high_resolution_clock::now();
  const double duration_total =
      std::chrono::duration<double>(time_end - time_start).count();

  // TODO: remove this unconditional call (here for debugging);
  // 计算轨迹误差
  const double ate_rmse =
      basalt::alignSVD(vio_t_ns, vio_t_w_i, gt_t_ns, gt_t_w_i);
  vio->debug_finalize();
  std::cout << "Total runtime: {:.3f}s\n"_format(duration_total);

  // 保存性能统计信息
  {
    basalt::ExecutionStats stats;
    stats.add("exec_time_s", duration_total);                               // 执行时间
    stats.add("ate_rmse", ate_rmse);                                        // 绝对轨迹误差
    stats.add("ate_num_kfs", vio_t_w_i.size());                             // 关键帧数量
    stats.add("num_frames", vio_dataset->get_image_timestamps().size());    // 总帧数

    // 内存使用统计
    {
      basalt::MemoryInfo mi;
      if (get_memory_info(mi)) {
        stats.add("resident_memory_peak", mi.resident_memory_peak);         // 峰值内存
      }
    }

    stats.save_json("stats_vio.json");                                      // 保存为JSON文件
  }

  // 保存轨迹数据（如果未提前终止且指定了格式）
  if (!aborted && !trajectory_fmt.empty()) {
    std::cout << "Saving trajectory..." << std::endl;

    // 根据格式设置相应的标志
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

    save_groundtruth = trajectory_groundtruth;    // 设置真值保存标志

    saveTrajectoryButton();                       // 调用轨迹保存函数
  }

  // 保存结果文件（如果指定了路径）
  if (!aborted && !result_path.empty()) {
    double error = basalt::alignSVD(vio_t_ns, vio_t_w_i, gt_t_ns, gt_t_w_i);

    // 计算执行时间（纳秒）
    auto exec_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        time_end - time_start);

    // 写入结果文件
    std::ofstream os(result_path);
    {
      cereal::JSONOutputArchive ar(os);                                   // JSON格式输出
      ar(cereal::make_nvp("rms_ate", error));                             // RMSE绝对轨迹误差
      ar(cereal::make_nvp("num_frames",               
                          vio_dataset->get_image_timestamps().size()));   // 帧数
      ar(cereal::make_nvp("exec_time_ns", exec_time_ns.count()));         // 执行时间
    }
    os.close(); // 关闭文件
  }

  return 0; // 程序正常退出
}

// 图像叠加绘制函数 - 在图像上显示特征点和光流信息
void draw_image_overlay(pangolin::View& v, size_t cam_id) {
  UNUSED(v);

  //  size_t frame_id = show_frame;
  //  basalt::TimeCamId tcid =
  //      std::make_pair(vio_dataset->get_image_timestamps()[frame_id],
  //      cam_id);

  size_t frame_id = show_frame;
  auto it = vis_map.find(vio_dataset->get_image_timestamps()[frame_id]);

  if (show_obs) {
    glLineWidth(1.0);
    glColor3f(1.0, 0.0, 0.0);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    if (it != vis_map.end() && cam_id < it->second->projections.size()) {
      const auto& points = it->second->projections[cam_id];

      if (points.size() > 0) {
        double min_id = points[0][2], max_id = points[0][2];

        for (const auto& points2 : it->second->projections)
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

    if (it != vis_map.end()) {
      const Eigen::aligned_map<basalt::KeypointId, Eigen::AffineCompact2f>&
          kp_map = it->second->opt_flow_res->observations[cam_id];

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
          pangolin::default_font().Text("%d", kv.first).Draw(5 + c[0], 5 + c[1]);
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
    size_t end = std::min(vio_t_w_i.size(), size_t(show_frame + 1));
    Eigen::aligned_vector<Eigen::Vector3d> sub_gt(vio_t_w_i.begin(),
                                                  vio_t_w_i.begin() + end);
    pangolin::glDrawLineStrip(sub_gt);
  }

  glColor3ubv(gt_color);
  if (show_gt) pangolin::glDrawLineStrip(gt_t_w_i);

  size_t frame_id = show_frame;
  int64_t t_ns = vio_dataset->get_image_timestamps()[frame_id];
  auto it = vis_map.find(t_ns);

  if (it != vis_map.end()) {
    for (size_t i = 0; i < calib.T_i_c.size(); i++)
      if (!it->second->states.empty()) {
        render_camera((it->second->states.back() * calib.T_i_c[i]).matrix(),
                      2.0f, cam_color, 0.1f);
      } else if (!it->second->frames.empty()) {
        render_camera((it->second->frames.back() * calib.T_i_c[i]).matrix(),
                      2.0f, cam_color, 0.1f);
      }

    for (const auto& p : it->second->states)
      for (size_t i = 0; i < calib.T_i_c.size(); i++)
        render_camera((p * calib.T_i_c[i]).matrix(), 2.0f, state_color, 0.1f);

    for (const auto& p : it->second->frames)
      for (size_t i = 0; i < calib.T_i_c.size(); i++)
        render_camera((p * calib.T_i_c[i]).matrix(), 2.0f, pose_color, 0.1f);

    glColor3ubv(pose_color);
    pangolin::glDrawPoints(it->second->points);
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
  if (show_frame < int(vio_dataset->get_image_timestamps().size()) - 1) {
    show_frame = show_frame + 1;
    show_frame.Meta().gui_changed = true;
    cv.notify_one();
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

  double t = vio_dataset->get_image_timestamps()[show_frame] * 1e-9;
  plotter->AddMarker(pangolin::Marker::Vertical, t, pangolin::Marker::Equal,
                     pangolin::Colour::White());
}

void alignButton() { basalt::alignSVD(vio_t_ns, vio_t_w_i, gt_t_ns, gt_t_w_i); }

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

    if (save_groundtruth) {
      std::ofstream os("groundtruth.txt");

      os << "# timestamp tx ty tz qx qy qz qw" << std::endl;

      for (size_t i = 0; i < gt_t_ns.size(); i++) {
        const Eigen::Vector3d& pos = gt_t_w_i[i];
        os << std::scientific << std::setprecision(18) << gt_t_ns[i] * 1e-9
           << " " << pos.x() << " " << pos.y() << " " << pos.z() << " "
           << "0 0 0 1" << std::endl;
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
