#!/bin/bash
# 用法: ./cpp_linter.sh [目标目录]

target_dir="${1:-.}"
cd "$target_dir" || { echo "错误: 无法进入目录 $target_dir" >&2; exit 1; }

# 加速字符处理
export LC_ALL=C

# 递归查找所有目标文件，并行传入 awk 处理
find . -type f \( -name "*.cpp" -o -name "*.hpp" -o -name "*.ixx" -o -name "*.cppm" \) -print0 |
xargs -0 -P "$(nproc 2>/dev/null || echo 1)" awk '
FNR == 1 {
    # 上一个文件处理完毕，判断是否符合条件并输出
    if (lastfile != "" && has_size && !has_cstddef) print lastfile
    lastfile = FILENAME
    has_size = 0
    has_cstddef = 0
}
{
    # 若未发现 <cstddef>，则继续检查
    if (!has_cstddef) {
        if (index($0, "<cstddef>") > 0) {
            has_cstddef = 1
        } else if (!has_size && index($0, "std::size_t") > 0) {
            has_size = 1
        }
    }
}
END {
    # 处理最后一个文件
    if (lastfile != "" && has_size && !has_cstddef) print lastfile
}
'
