#pragma once

#include <cstddef>

struct TrackingConfig
{
  // 如果在 count_updates_before_deletion_ 次更新中
  // 至少有 count_missing_updates_ 次更新丢失
  // 那么删除轨迹
  std::size_t count_updates_before_deletion_;
  std::size_t count_missing_updates_;
  // 如果在 count_updates_before_creation_ 次更新中
  // 至少有 count_tracked_updates_ 次更新被检测到
  // 那么创建轨迹
  std::size_t count_updates_before_creation_;
  std::size_t count_tracked_updates_;
};
