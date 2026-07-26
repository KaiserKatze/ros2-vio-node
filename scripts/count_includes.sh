#!/bin/bash

# 检查是否传入了目录参数，如果没有则默认使用当前目录
TARGET_DIR="${1:-.}"

if [ ! -d "$TARGET_DIR" ]; then
    echo "错误: 路径 '$TARGET_DIR' 不存在或不是一个目录！" >&2
    exit 1
fi

echo "============================================================"
echo "正在分析目录: $(cd "$TARGET_DIR" && pwd) ..."
echo "============================================================"

# 使用 find 找到所有相关的 C++ 文件
# 统计文件总数
FILE_COUNT=$(find "$TARGET_DIR" -type f \( -name "*.cpp" -o -name "*.hpp" -o -name "*.h" -o -name "*.cc" -o -name "*.cxx" -o -name "*.c" \) | wc -l)
echo "共扫描了 $FILE_COUNT 个 C++ 源/头文件。"
echo "============================================================"

# 核心统计函数
# 参数 1: 正则表达式（用于区分 <> 和 ""）
# 参数 2: 提取后的包裹符号（用于格式化输出，如 '<' 和 '>'）
# 参数 3: 打印的标题
count_includes() {
    local pattern=$1
    local left_wrapper=$2
    local right_wrapper=$3
    local title=$4

    echo -e "\n$title"
    printf "%-40s | %s\n" "Header File" "Count"
    echo "--------------------------------------------------"

    # 1. find 找出所有文件
    # 2. xargs + grep 提取 #include 行（忽略大小写，允许前后有空格）
    # 3. sed 提取出括号或双引号内的文件名
    # 4. sort + uniq -c 进行去重计数
    # 5. sort -rn 按数量从大到小排序
    # 6. head -n 30 取前 30 名
    find "$TARGET_DIR" -type f \( -name "*.cpp" -o -name "*.hpp" -o -name "*.h" -o -name "*.cc" -o -name "*.cxx" -o -name "*.c" \) -print0 2>/dev/null | \
    xargs -0 grep -E '^[[:space:]]*#[[:space:]]*include[[:space:]]*'"$pattern" 2>/dev/null | \
    sed -E 's/.*include[[:space:]]*'"$pattern"'.*/\1/' | \
    sort | \
    uniq -c | \
    sort -rn | \
    head -n 30 | \
    while read -r count header; do
        if [ -not -z "$header" ]; then
            printf "%-40s | %s\n" "${left_wrapper}${header}${right_wrapper}" "$count"
        fi
    done
}

# 统计系统头文件 <...>
# 正则匹配 <([^>]+)>
count_includes '<([^>]+)>' '<' '>' "[系统/标准库头文件使用频率 (Top 30)]"

# 统计用户头文件 "..."
# 正则匹配 "([^"]+)"
count_includes '"([^"]+)"' '"' '"' "[用户自定义头文件使用频率 (Top 30)]"
