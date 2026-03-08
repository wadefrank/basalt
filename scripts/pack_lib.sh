#! /bin/bash

echo "===pack VIO==="
echo "===version 1.0==="

# ===================== 基础路径参数处理 =====================
# 获取脚本所在目录（相对路径转绝对路径）
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
echo "SCRIPT_DIR: $SCRIPT_DIR"

# 获取上一级目录
BASALT_ROOT_DIR=$(dirname "$SCRIPT_DIR")
echo "BASALT_ROOT_DIR: $BASALT_ROOT_DIR"

# ===================== 版本号提取 =====================
VERSION_FILE=${BASALT_ROOT_DIR}/build/VERSION

# 从VERSION文件中提取版本号
version=$(grep -E "^Version: " "$VERSION_FILE" | awk '{print $2}')
echo ${version}
if [ -z "$version" ]; then
    echo "错误: 无法从 VERSION 文件中提取版本号"
    exit 1
fi

# ===================== 目标目录初始化 =====================
# 定义目标打包目录：包含版本号，便于区分不同版本
TARGET_PROGRAM_DIR="${BASALT_ROOT_DIR}/dist/basalt_vio_v${version}/"

# 检查目标文件夹是否存在
if [ -d "$TARGET_PROGRAM_DIR" ]; then
    # 如果目标文件夹存在，则删除它及其内容
    echo "Removing existing dir: $TARGET_PROGRAM_DIR ..."
    rm -rf "$TARGET_PROGRAM_DIR"
fi

# 创建目标文件夹及其子文件夹
echo "Creating dir: $TARGET_PROGRAM_DIR ..."
mkdir -p "$TARGET_PROGRAM_DIR/bin"
mkdir -p "$TARGET_PROGRAM_DIR/config"
mkdir -p "$TARGET_PROGRAM_DIR/lib"
# mkdir -p "$target_progam_dir/model"
# mkdir -p "$target_progam_dir/scripts"
# mkdir -p "$target_progam_dir/service"

# ===================== 可执行程序拷贝 =====================
app_list=("basalt_vio")
# 依赖库
lib_list=()
for app_item in "${app_list[@]}"
do
    # 拼接可执行程序的绝对路径
    app_abs_path="${BASALT_ROOT_DIR}/build/${app_item}"
    
    # 检查可执行程序是否存在
    if ! [ -f "$app_abs_path" ]; then
        echo "app does not exist: $app_abs_path"
        echo "exit"
        exit 1  # 程序不存在，退出脚本
    else
        # echo "${app_abs_path}"
        # 拷贝可执行程序到目标bin目录：
        # -L：跟随符号链接（拷贝实际文件而非链接）
        # -n：不覆盖已存在的文件（避免意外覆盖）
        cp -L -n ${app_abs_path} $TARGET_PROGRAM_DIR/bin
        echo "copy ${app_abs_path} ==> $TARGET_PROGRAM_DIR/bin"
        
        # 获取程序依赖的所有动态库：
        # 1. ldd 列出程序的动态依赖库
        # 2. awk '{print $3}' 提取库文件的绝对路径
        # 3. mapfile -t 将结果存入dependencies数组
        mapfile -t dependencies < <(ldd "${app_abs_path}" | awk '{print $3}')
        for item in "${dependencies[@]}"
        do
            lib_list+=("$item")
        done
    fi
done



# ===================== 依赖库去重 =====================
# 原始依赖库数量
raw_lib_length=${#lib_list[@]}

# 去重处理：
# 1. echo 将数组转为字符串
# 2. tr ' ' '\n' 将空格分隔转为换行分隔
# 3. sort -u 排序并去重
# 4. 重新转为数组
unique_lib=($(echo "${lib_list[@]}" | tr ' ' '\n' | sort -u))

# 去重后的依赖库数量
unique_lib_length=${#unique_lib[@]}

# 打印去重前和去重后的依赖库数量
echo "libs unrepeated ${raw_lib_length} ==> ${unique_lib_length}"

# ===================== 增量依赖库筛选 =====================
# 初始化增量依赖库列表（最终需要拷贝的库）
increment_lib=()

# -------------------- 读取历史忽略库列表 --------------------
# 历史忽略库路径：存储不需要打包的系统库路径
dependence_history_path="${base_path}/deploy/dependence/dependence_ingored/"

lines=()
# 遍历忽略库目录下的所有文件
for file in "$dependence_history_path"/*; do
    # 跳过非文件类型（如目录、链接）
    if [ ! -f "$file" ]; then
        continue
    fi
    
    # 逐行读取文件内容，存入lines数组
    while IFS= read -r line; do
        lines+=("$line")
    done < "$file"
done
# 对历史忽略库列表去重
history_lib=($(echo "${lines[@]}" | tr ' ' '\n' | sort -u))
# history_lib+=("/lib/aarch64-linux-gnu")
# history_lib+=("/usr/local")

# 打印历史忽略库数量
history_lib_length=${#history_lib[@]}
echo "history libs(path) ${history_lib_length}"

# -------------------- 读取强制导入库列表 --------------------
# 强制导入库路径：存储即使在忽略列表中也需要打包的库路径
dependence_import_path="${base_path}/deploy/dependence/dependence_required/"

lines=()
# 遍历目录下的每个文件
for file in "$dependence_import_path"/*; do
    # 跳过非文件类型
    if [ ! -f "$file" ]; then
        continue
    fi
    # 逐行读取文件内容，存入lines数组
    while IFS= read -r line; do
        lines+=("$line")
    done < "$file"
done
# 对强制导入库列表去重
dependence_import_lib=($(echo "${lines[@]}" | tr ' ' '\n' | sort -u))

# 打印强制导入库数量
dependence_import_lib_length=${#dependence_import_lib[@]}
echo "dependence_import_libs ${dependence_import_lib_length}"

# -------------------- 筛选增量依赖库 --------------------
for item in "${unique_lib[@]}"
do
    is_found=0  # 标记是否需要忽略该库（0=不忽略，1=忽略）
    
    # 检查当前库是否在忽略列表中
    for lib in "${history_lib[@]}"; do
        if [[ "$item" == *"$lib"* ]]; then
            is_found=1      # 匹配到忽略路径，标记为忽略
            # echo "found"
        fi
    done

    # 检查当前库是否在强制导入列表中（优先级高于忽略列表）
    for lib in "${dependence_import_lib[@]}"; do
        if [[ "$item" == *"$lib"* ]]; then
            is_found=0      # 匹配到强制导入路径，取消忽略标记
            # echo "found"
        fi
    done

    # 如果不需要忽略，则加入增量库列表
    if [ ${is_found} = 0 ]; then
        increment_lib+=("$item")
    fi
done

# 打印增量库数量
increment_lib_length=${#increment_lib[@]}
echo "increment libs ${increment_lib_length}"


# ===================== 依赖库拷贝 =====================
for item in "${increment_lib[@]}"
do
    # 拷贝依赖库到目标lib目录（-L跟随链接，-n不覆盖）
    cp -L -n ${item} $TARGET_PROGRAM_DIR/lib
    echo "copy ${item} ==> $TARGET_PROGRAM_DIR/lib"
done

# check_seg_model() {
#     # 检查模型文件是否存在
#     model_file="${base_path}/modules/agri_navi_engine/build/model/*.rknn"
#     if [ ! -f "$model_file" ]; then
#         echo "模型文件不存在: $model_file"
#         echo "请检查模型文件是否存在"
#         echo "exit"
#         exit 1
#     fi
# }

#拷贝配置文件
cp -L -n -r ${BASALT_ROOT_DIR}/config/* $TARGET_PROGRAM_DIR/config/

# #拷贝脚本
# cp -L -n -r ${base_path}/deploy/scripts/run_on_host $target_progam_dir/scripts

# #拷贝服务文件
# cp -L -n -r ${base_path}/deploy/service $target_progam_dir/service

# #拷贝模型
# cp -L -n -r ${base_path}/modules/assets/model_hub $target_progam_dir/model_hub


#拷贝版本信息
cp -L -n -r ${BASALT_ROOT_DIR}/build/VERSION $TARGET_PROGRAM_DIR





#打包zip文件

# 检查VERSION文件是否存在
PACK_VERSION_FILE="$TARGET_PROGRAM_DIR/VERSION"
if [ ! -f "$PACK_VERSION_FILE" ]; then
    echo "错误: 未找到 VERSION 文件: $PACK_VERSION_FILE"
    exit 1
fi



# 构建带版本号的ZIP文件名
ZIP_FILE="basalt_vio_v${version}.zip"

echo "正在打包版本: $version"

cd ${BASALT_ROOT_DIR}/dist

zip -r ${ZIP_FILE} basalt_vio_v${version}

# # md5校验，并加密
# PACKAGE_NAME=${versioned_zip_file}
# md5sum ${PACKAGE_NAME} > ${PACKAGE_NAME}.md5sum
# tar czf - ${PACKAGE_NAME} ${PACKAGE_NAME}.md5sum | openssl enc -e -aes-256-cbc -md sha512 -pbkdf2 -iter 100000 -salt -pass pass:15158 -out ${PACKAGE_NAME}.pkg
# rm ${PACKAGE_NAME}.md5sum

# 解压命令
# openssl enc -d -aes-256-cbc -md sha512 -pbkdf2 -iter 100000 -salt -pass pass:15158 -in AgriNaviEngine_v1.0.0.zip.pkg | tar -xzvf -