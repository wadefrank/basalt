# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Basalt is a visual-inertial odometry (VIO) and mapping system from TUM (Technical University of Munich). It provides:
- Camera, IMU, and motion capture calibration
- Visual-inertial odometry and mapping (stereo + IMU)
- Visual odometry (stereo, no IMU)
- Simulation environment for testing

The project uses C++17, CMake build system, and relies heavily on Eigen, Sophus (Lie groups), TBB (threading), Pangolin (visualization), and OpenCV.

## Build Commands

```bash
# Install dependencies (auto-detects macOS vs Ubuntu)
./scripts/install_deps.sh

# Build (from project root)
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo
make -j8

# Run tests (from build directory)
ctest
# Or run individual test binaries:
./test_vio
./test_nfr
./test_spline_opt
./test_qr
./test_linearization
./test_patch
```

Tests use Google Test. Test sources are in `test/src/`. The basalt-headers submodule also has its own tests built via `thirdparty/basalt-headers/test/`.

## CMake Options

- `BASALT_INSTANTIATIONS_DOUBLE` (ON/OFF) - template instantiation for double precision
- `BASALT_INSTANTIATIONS_FLOAT` (ON/OFF) - template instantiation for float precision
- Disabling these during development speeds up compile times

## Code Formatting

Uses clang-format (version 11) with Google style, 2-space indent (see `.clang-format`). Format all files:
```bash
./scripts/clang-format-all.sh
```
Python formatting: `./scripts/yapf-all.sh`

## Architecture

### Data Pipeline (Producer-Consumer with TBB queues)

The system is a multi-threaded pipeline connected by `tbb::concurrent_bounded_queue`:

1. **Dataset I/O** (`src/io/`, `include/basalt/io/`) - Reads sensor data from various formats (EuRoC, KITTI, UZH-FPV, rosbag). Factory pattern via `DatasetIoFactory`.

2. **Optical Flow Frontend** (`src/optical_flow/`, `include/basalt/optical_flow/`) - Tracks feature points across stereo frames using patch-based optical flow with image pyramids. Factory pattern via `OpticalFlowFactory`. Outputs `OpticalFlowResult` (per-camera keypoint observations with pyramid levels).

3. **VIO/VO Backend** (`src/vi_estimator/`) - State estimation using sliding-window bundle adjustment:
   - `VioEstimatorBase` / `VioEstimatorFactory` - Abstract base and factory
   - `SqrtKeypointVioEstimator` - Square-root VIO (IMU + vision), the primary estimator
   - `SqrtKeypointVoEstimator` - Square-root VO (vision only, no IMU)
   - `BaBase` / `SqrtBaBase` / `ScBaBase` - Bundle adjustment base classes with Schur complement and square-root marginalization

4. **Linearization** (`src/linearization/`, `include/basalt/linearization/`) - Landmark and pose linearization strategies:
   - `LinearizationAbsSC` - Absolute poses with Schur complement
   - `LinearizationAbsQR` - Absolute poses with QR decomposition
   - `LinearizationRelSC` - Relative poses with Schur complement

5. **Calibration** (`src/calibration/`) - Camera and camera-IMU calibration using AprilGrid targets and B-spline trajectory representation.

### Public API

`include/basalt/api/basalt_vio.h` defines `basalt_vio::VIO` - a self-contained API class that wraps the optical flow frontend and VIO backend. It accepts IMU and stereo image data, and outputs poses via callbacks. This is the integration point for embedding Basalt into external systems.

### Key Executables

| Binary | Purpose |
|--------|---------|
| `basalt_vio` | Main VIO application (stereo + IMU) |
| `basalt_opt_flow` | Standalone optical flow visualization |
| `basalt_calibrate` | Camera calibration |
| `basalt_calibrate_imu` | Camera-IMU calibration |
| `basalt_mapper` | Mapping from VIO marginalization data |
| `basalt_vio_sim` / `basalt_mapper_sim` | Simulation tools |
| `basalt_kitti_eval` | KITTI dataset evaluation |
| `basalt_rs_t265_*` | RealSense T265 recording/VIO (optional, requires librealsense2) |

### Header-Only Library

Core math (camera models, splines, Lie group operations, image types) lives in the separate `thirdparty/basalt-headers/` submodule. Headers are under `thirdparty/basalt-headers/include/basalt/`. Documentation: https://vladyslavusenko.gitlab.io/basalt-headers/

### Config and Calibration Files

JSON config and calibration files are in `data/`:
- `euroc_config.json`, `tumvi_512_config.json`, `kitti_config.json` - VIO/VO runtime configs
- `euroc_ds_calib.json`, `tumvi_512_ds_calib.json` - Camera calibration (Double Sphere model)

### Third-Party Dependencies

All in `thirdparty/` as git submodules: Pangolin (visualization), basalt-headers (core math), OpenGV (geometric vision), apriltag (calibration targets), CLI11 (CLI parsing), nlohmann/json, magic_enum, ros (rosbag reading).

## Important Technical Notes

- OpenMP is intentionally disabled. TBB is used for all parallelism. OpenMP + TBB causes 10-100x slowdowns. `EIGEN_DONT_PARALLELIZE` is always set.
- The codebase uses Eigen extensively. All containers of Eigen types use `Eigen::aligned_map` / `Eigen::aligned_vector` for SIMD alignment.
- Compile flags include `-Werror` (warnings are errors). Check compiler-specific exceptions in CMakeLists.txt if you hit warning-as-error issues.
- The CMake comments are written in Chinese (学习笔记 style) by the repo maintainer.
