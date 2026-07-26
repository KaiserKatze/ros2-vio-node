#!/usr/bin/env bash

# CMake Target Inspector —— 检索任意已安装包 (ROS 2 或系统级) 导出的 CMake Targets
# 用法:
#   ./inspect_cmake_targets.sh                       # 默认查 nav_msgs (ROS 2)
#   ./inspect_cmake_targets.sh yaml-cpp              # 非 ROS 包
#   ./inspect_cmake_targets.sh Sophus /opt/mylibs    # 追加自定义搜索根目录
#   ./inspect_cmake_targets.sh Threads               # CMake 内建 Find 模块也能识别
# 严格匹配失败时, 用 Levenshtein 距离给出最接近的 5 个已安装候选包名

# 设置输出颜色
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0;m' # 无颜色

# ---------------------------------------------------------------
# 模糊建议: 收集系统中所有可被本脚本检索到的包名, 按 Levenshtein
# 编辑距离排序, 输出与用户输入最接近的 5 个 (在严格匹配失败时调用)
# ---------------------------------------------------------------
suggest_similar_names() {
    # 候选来源 1: 各搜索根目录下含 .cmake 文件的包目录名
    local config_candidates
    config_candidates=$(for root in "${SEARCH_ROOTS[@]}"; do
        [ -d "$root" ] || continue
        find "$root" -maxdepth 3 -type f -name "*.cmake" 2>/dev/null
    done | awk -F/ '{
        dir = $(NF-1)
        if (dir == "cmake") dir = $(NF-2)   # share/<pkg>/cmake/*.cmake 形态取上一级
        if (dir ~ /^(lib|lib64|share|cmake|Modules)$/) next
        if (dir ~ /^cmake-/ || dir ~ /-linux-/) next
        print dir
    }' | sort -u)

    # 候选来源 2: CMake 内建 Find 模块名 (FindXxx.cmake -> Xxx)
    local module_candidates
    module_candidates=$(find /usr/share/cmake* /usr/local/share/cmake* -maxdepth 2 \
                        -type f -name "Find*.cmake" 2>/dev/null | \
                        sed -E 's|.*/Find(.+)\.cmake|\1|' | sort -u)

    local candidates
    candidates=$(printf '%s\n%s\n' "$config_candidates" "$module_candidates" | \
                 grep -v '^$' | sort -u)
    [ -z "$candidates" ] && return

    echo -e "\n${YELLOW}[?] 按 Levenshtein 编辑距离最接近的 5 个候选:${NC}"
    echo -e "----------------------------------------------------------------"
    awk -v query="$TARGET_PKG" '
    function levenshtein(a, b,    la, lb, i, j, cost, m, d) {
        la = length(a); lb = length(b)
        for (i = 0; i <= la; i++) d[i, 0] = i
        for (j = 0; j <= lb; j++) d[0, j] = j
        for (i = 1; i <= la; i++)
            for (j = 1; j <= lb; j++) {
                cost = (substr(a, i, 1) == substr(b, j, 1)) ? 0 : 1
                m = d[i - 1, j] + 1
                if (d[i, j - 1] + 1 < m) m = d[i, j - 1] + 1
                if (d[i - 1, j - 1] + cost < m) m = d[i - 1, j - 1] + cost
                d[i, j] = m
            }
        return d[la, lb]
    }
    { print levenshtein(tolower(query), tolower($0)) "\t" $0 }
    ' <<< "$candidates" | sort -t$'\t' -n -k1,1 | head -5 | \
    while IFS=$'\t' read -r distance name; do
        echo -e "  ⭐ ${GREEN}${name}${NC}  ${CYAN}(编辑距离 ${distance})${NC}"
    done
}

# 默认查找的对象，如果脚本后面没带参数，默认查 nav_msgs
TARGET_PKG=${1:-"nav_msgs"}
EXTRA_ROOT=${2:-""}
ROS_DISTRO_DIR="/opt/ros/${ROS_DISTRO:-jazzy}"

echo -e "${CYAN}================================================================${NC}"
echo -e "  🔍  CMake Target Inspector (Target: ${YELLOW}${TARGET_PKG}${NC})"
echo -e "${CYAN}================================================================${NC}"

# ---------------------------------------------------------------
# 0. 构建搜索根目录列表: ROS 2 + 系统级 CMake 配置的常见安装位置
# ---------------------------------------------------------------
SEARCH_ROOTS=(
    "${ROS_DISTRO_DIR}/share"          # ROS 2:      share/<pkg>/cmake
    "/usr/local/lib/cmake"             # 源码安装:    lib/cmake/<Pkg>      (如 Sophus)
    "/usr/local/share"                 # 源码安装:    share/<pkg>/cmake
    "/usr/lib/cmake"                   # apt 安装:    lib/cmake/<Pkg>      (如 yaml-cpp)
    "/usr/share"                       # apt 安装:    share/<pkg>/cmake    (如 eigen3)
    "/usr/lib64/cmake"                 # RPM 系发行版
)
# 多架构目录 (如 /usr/lib/x86_64-linux-gnu/cmake, Ceres/OpenCV 常在此)
for multiarch_dir in /usr/lib/*/cmake /usr/local/lib/*/cmake; do
    [ -d "$multiarch_dir" ] && SEARCH_ROOTS+=("$multiarch_dir")
done
[ -n "$EXTRA_ROOT" ] && SEARCH_ROOTS=("$EXTRA_ROOT" "${SEARCH_ROOTS[@]}")

# 目录名大小写不敏感 + 前缀匹配 (opencv 可命中 opencv4), 再验证其中确有 .cmake 文件
RESOLVED_DIRS=()
for root in "${SEARCH_ROOTS[@]}"; do
    [ -d "$root" ] || continue
    while IFS= read -r matched_dir; do
        for candidate in "$matched_dir" "$matched_dir/cmake"; do
            if [ -d "$candidate" ] && compgen -G "$candidate/*.cmake" > /dev/null; then
                RESOLVED_DIRS+=("$candidate")
            fi
        done
    done < <(find "$root" -maxdepth 3 -type d -iname "${TARGET_PKG}*" 2>/dev/null)
done
if [ ${#RESOLVED_DIRS[@]} -gt 0 ]; then
    mapfile -t RESOLVED_DIRS < <(printf '%s\n' "${RESOLVED_DIRS[@]}" | sort -u)
fi

# ---------------------------------------------------------------
# 若找不到 Config 包, 回退检查是否为 CMake 内建 Find 模块 (如 Threads)
# ---------------------------------------------------------------
if [ ${#RESOLVED_DIRS[@]} -eq 0 ]; then
    BUILTIN_MODULES=$(find /usr/share/cmake* /usr/local/share/cmake* -maxdepth 2 \
                      -iname "Find${TARGET_PKG}*.cmake" 2>/dev/null | sort -u)
    if [ -n "$BUILTIN_MODULES" ]; then
        echo -e "\n${YELLOW}[!] 该目标由 CMake 内建 Find 模块提供, 不是独立安装的 Config 包:${NC}"
        while read -r module_file; do
            echo -e "  📄 ${CYAN}${module_file}${NC}"
        done <<< "$BUILTIN_MODULES"
        echo -e "\n${GREEN}说明:${NC} 这类模块随 ${YELLOW}cmake${NC} 软件包一起安装, 无需 (也没有) 单独的 apt 包。"
        echo -e "      例如 Threads::Threads 解析为系统 pthread (glibc 自带), 装好"
        echo -e "      ${YELLOW}cmake${NC} 与 ${YELLOW}build-essential${NC} 后 find_package(Threads) 即可直接使用。"
        exit 0
    fi
    echo -e "${RED}❌ 错误: 在以下位置均未找到 '${TARGET_PKG}' 的 CMake 配置:${NC}"
    printf '  %s\n' "${SEARCH_ROOTS[@]}"
    suggest_similar_names
    echo -e "\n${GREEN}提示:${NC} 若以上候选均不符, 请确认包已安装; 装在自定义前缀时可传第二个参数:"
    echo -e "      $0 ${TARGET_PKG} /your/install/prefix"
    exit 1
fi

# ---------------------------------------------------------------
# 1. 展示命中的配置目录、Config 文件, 并推导 find_package 应使用的名称
# ---------------------------------------------------------------
echo -e "\n${YELLOW}[1] 命中的 CMake 配置目录与 find_package 名称:${NC}"
echo -e "----------------------------------------------------------------"
for dir in "${RESOLVED_DIRS[@]}"; do
    echo -e "  📂 ${CYAN}${dir}${NC}"
done
CONFIG_FILES=$(find "${RESOLVED_DIRS[@]}" -maxdepth 1 -type f \
               \( -iname "*config.cmake" \) 2>/dev/null | grep -ivE "config-?version" | sort -u)
if [ -n "$CONFIG_FILES" ]; then
    while read -r config_file; do
        config_base=$(basename "$config_file")
        find_name=${config_base%Config.cmake}
        find_name=${find_name%-config.cmake}
        echo -e "  📄 ${config_base}  →  ${GREEN}find_package(${find_name} REQUIRED)${NC}"
    done <<< "$CONFIG_FILES"
fi

# ---------------------------------------------------------------
# 2. 检索所有通过 add_library(... IMPORTED) 导出的 Target 名字
# ---------------------------------------------------------------
echo -e "\n${YELLOW}[2] 该包导出的所有有效 CMake Targets:${NC}"
echo -e "----------------------------------------------------------------"

# 核心过滤逻辑：寻找类似 INTERFACE IMPORTED, UNKNOWN IMPORTED 并抓取前置的 Target 名字
TARGETS_FOUND=$(grep -E -r -h "add_library\(.*IMPORTED" "${RESOLVED_DIRS[@]}" 2>/dev/null | \
                sed -E 's/^[[:space:]]*//; s/add_library\(([^ )]+).*/\1/' | sort -u)

if [ -z "$TARGETS_FOUND" ]; then
    # 兼容某些包直接使用带有命名空间的别名或手工导出的情况
    TARGETS_FOUND=$(grep -E -r -h "set_target_properties\(.*PROPERTIES" "${RESOLVED_DIRS[@]}" 2>/dev/null | \
                    sed -E 's/^[[:space:]]*//; s/set_target_properties\(([^ )]+).*/\1/' | grep "::" | sort -u)
fi

if [ -z "$TARGETS_FOUND" ]; then
    echo -e "${RED}  未直接匹配到标准的 IMPORTED 目标，尝试输出包内定义的所有 :: 关键词:${NC}"
    grep -r -h "::" "${RESOLVED_DIRS[@]}" 2>/dev/null | \
        grep -E "(add_library|set_target_properties|target_link_libraries)" | sort -u
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

# ---------------------------------------------------------------
# 3. 深度分析：头文件包含路径传导链 (INTERFACE_INCLUDE_DIRECTORIES)
# ---------------------------------------------------------------
echo -e "\n${YELLOW}[3] 底层属性与头文件传导链 (Interface Include Dirs):${NC}"
echo -e "----------------------------------------------------------------"

# 抓取包含 INTERFACE_INCLUDE_DIRECTORIES 设定的代码行，帮助研判它到底把路径绑在哪个 Target 身上了
grep -r -H -n "INTERFACE_INCLUDE_DIRECTORIES" "${RESOLVED_DIRS[@]}" 2>/dev/null | while read -r line; do
    file_info=$(echo "$line" | cut -d: -f1,2)
    content=$(echo "$line" | cut -d: -f3-)
    echo -e "  📄 ${CYAN}${file_info}${NC}: ${content}"
done

echo -e "${CYAN}----------------------------------------------------------------${NC}"
echo -e "${GREEN}建议:${NC} 请在上方 [2] 中选择带有双冒号 ${YELLOW}::${NC} 的标准 C++ 目标填入你的 CMakeLists.txt 中。"
