#!/bin/bash
# 文件名：build_and_run.sh
# 功能：编译算子 → 安装算子 → 运行 AclNNInvocation

set -e  # 只要有命令出错就退出
set -o pipefail

BUILD_SCRIPT="/home/HwHiAiUser/study/all-prj/operator/Softplus/SoftplusCustom/build.sh"
RUN_PACKAGE="./SoftplusCustom/build_out/custom_opp_ubuntu_aarch64.run"
RUN_SCRIPT_DIR="./AclNNInvocation"
RUN_SCRIPT="$RUN_SCRIPT_DIR/run.sh"

# 颜色定义
RED='\033[31m'
GREEN='\033[32m'
YELLOW='\033[33m'
CYAN='\033[36m'
BOLD='\033[1m'
RESET='\033[0m'

# 计时函数
step_start_time=0
start_step() {
    step_start_time=$(date +%s)
    echo -e "${BOLD}${CYAN}\n===== Step $1: $2 =====${RESET}"
}
end_step() {
    local end_time=$(date +%s)
    local elapsed=$((end_time - step_start_time))
    echo -e "${GREEN}✓ Step 完成，用时 ${elapsed} 秒${RESET}"
}

# Step 1: 编译算子工程
start_step 1 "编译算子工程"
if [ ! -f "$BUILD_SCRIPT" ]; then
    echo -e "${RED}[ERROR] 没有找到 $BUILD_SCRIPT${RESET}"
    exit 1
fi
bash "$BUILD_SCRIPT"
end_step

# Step 2: 安装算子包
start_step 2 "安装算子包"
if [ ! -f "$RUN_PACKAGE" ]; then
    echo -e "${RED}[ERROR] 没有找到 $RUN_PACKAGE${RESET}"
    exit 1
fi
chmod +x "$RUN_PACKAGE"
"$RUN_PACKAGE"
end_step

# Step 3: 运行执行脚本
start_step 3 "运行执行脚本"
if [ ! -d "$RUN_SCRIPT_DIR" ]; then
    echo -e "${RED}[ERROR] 找不到 $RUN_SCRIPT_DIR 目录${RESET}"
    exit 1
fi
if [ ! -f "$RUN_SCRIPT" ]; then
    echo -e "${RED}[ERROR] 找不到 $RUN_SCRIPT${RESET}"
    exit 1
fi
chmod +x "$RUN_SCRIPT"
bash "$RUN_SCRIPT"
end_step

echo -e "${BOLD}${GREEN}\n===== 全部步骤完成 =====${RESET}"
