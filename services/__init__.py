"""
服务层：页面与（后续的）本机任务执行器共用的取数 / 分析流程

  change_detection  两期变化检测：资料收集、运行流水线、读取结果、PDF 报告、任务记录
  classification    地物分类：取数、加载波段、快速分类 / GMM / 主动学习、后处理、导出
  timeseries        多期时间序列趋势
  batch             批量任务中的单个任务

约定:
  - 不依赖 streamlit，也不读写会话状态；页面负责把返回值放进 session_state
  - 进度通过 progress_callback(msg, pct) 回调，pct 为 0–100
  - 可预期的失败（参数无效、无影像、标注不足等）抛 ServiceError，消息可直接展示给用户；
    其他异常照常抛出
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


class ServiceError(RuntimeError):
    """可预期的失败，消息可直接展示给用户。"""


def report(progress_callback, msg, pct):
    """调用进度回调（可为 None）。"""
    if progress_callback:
        progress_callback(msg, pct)
