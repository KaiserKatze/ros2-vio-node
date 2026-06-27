#!/usr/bin/env bash

# 设置输出颜色
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0;m' # 无颜色

# 默认查找的对象，如果脚本后面没带参数，默认查 nav_msgs
TARGET_PKG=${1:-"nav_msgs"}
ROS_DISTRO_DIR="/opt/ros/jazzy"

echo -e "${CYAN}================================================================${NC}"
echo -e "  🔍  ROS 2 CMake Target Inspector (Target: ${YELLOW}${TARGET_PKG}${NC})"
echo -e "  📂  Search Directory: ${CYAN}${ROS_DISTRO_DIR}/share/${TARGET_PKG}/cmake/${NC}"
echo -e "${CYAN}================================================================${NC}"

# 1. 检查目录是否存在
if [ ! -d "${ROS_DISTRO_DIR}/share/${TARGET_PKG}/cmake" ]; then
    echo -e "${RED}❌ 错误: 找不到该 ROS 2 包的 CMake 目录。请确认包名是否正确或该包已安装。${NC}"
    exit 1
fi

# 2. 检索文件中所有通过 add_library(... IMPORTED) 导出的 Target 名字
echo -e "\n${YELLOW}[1] 正在匹配该包导出的所有有效 CMake Targets:${NC}"
echo -e "----------------------------------------------------------------"

# 核心过滤逻辑：寻找类似 INTERFACE IMPORTED, UNKNOWN IMPORTED 并抓取前置的 Target 名字
TARGETS_FOUND=$(grep -E -r -h "add_library\(.*IMPORTED" "${ROS_DISTRO_DIR}/share/${TARGET_PKG}/cmake/" 2>/dev/null | \
                sed -E 's/add_library\(([^ ]+) .*/\1/' | sort -u)

if [ -z "$TARGETS_FOUND" ]; then
    # 兼容某些包直接使用带有命名空间的别名或手工导出的情况
    TARGETS_FOUND=$(grep -E -r -h "set_target_properties\(.*PROPERTIES" "${ROS_DISTRO_DIR}/share/${TARGET_PKG}/cmake/" 2>/dev/null | \
                    sed -E 's/set_target_properties\(([^ ]+) .*/\1/' | grep "::" | sort -u)
fi

if [ -z "$TARGETS_FOUND" ]; then
    echo -e "${RED}  未直接匹配到标准的 IMPORTED 目标，尝试输出包内定义的所有 :: 关键词:${NC}"
    grep -r -h "::" "${ROS_DISTRO_DIR}/share/${TARGET_PKG}/cmake/" 2>/dev/null | grep -E "(add_library|set_target_properties|target_link_libraries)" | sort -u
else
    # 打印找到的 Targets
    while read -r line; do
        if [[ $line == *"::"* ]]; then
            echo -e "  ⭐ ${GREEN}${line}${NC}"
        else
            echo -e "  🔹 ${line} ${YELLOW}(注意: 可能需要加上 ${TARGET_PKG}:: 前缀)${NC}"
        fi
    done <<< "$TARGETS_FOUND"
fi

# 3. 深度分析：提取与 C++ 接口相关的 头文件包含路径（INTERFACE_INCLUDE_DIRECTORIES）
echo -e "\n${YELLOW}[2] 正在分析底层的属性与头文件传导链 (Interface Include Dirs):${NC}"
echo -e "----------------------------------------------------------------"

# 抓取包含 INTERFACE_INCLUDE_DIRECTORIES 设定的代码行，帮助研判它到底把路径绑在哪个 Target 身上了
grep -r -H -n "INTERFACE_INCLUDE_DIRECTORIES" "${ROS_DISTRO_DIR}/share/${TARGET_PKG}/cmake/" 2>/dev/null | while read -r line; do
    file_info=$(echo "$line" | cut -d: -f1,2)
    content=$(echo "$line" | cut -d: -f3-)
    echo -e "  📄 ${CYAN}${file_info}${NC}: ${content}"
done

echo -e "${CYAN}----------------------------------------------------------------${NC}"
echo -e "${GREEN}建议:${NC} 请在上方 [1] 中选择带有 ${YELLOW}::${NC} 的标准 C++ 目标填入你的 CMakeLists.txt 中。"
