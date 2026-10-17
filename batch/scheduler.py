"""
遥感解译助手 v3 — 定时任务调度器

Usage:
  python batch/scheduler.py    # 启动定时调度
"""

import os
import sys
import time
import subprocess

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_batch():
    """运行批量处理。"""
    runner = os.path.join(BASE_DIR, "batch", "batch_runner.py")
    subprocess.run([sys.executable, runner])


def start_scheduler():
    """启动定时调度循环。"""
    try:
        import schedule
    except ImportError:
        print("Installing schedule...")
        subprocess.run([sys.executable, "-m", "pip", "install", "schedule"],
                       capture_output=True)
        import schedule

    # 每天凌晨 2:00 运行
    schedule.every().day.at("02:00").do(run_batch)

    print("=" * 50)
    print("  Scheduler started")
    print("  Schedule: daily at 02:00")
    print("  Press Ctrl+C to stop")
    print("=" * 50)

    try:
        while True:
            schedule.run_pending()
            time.sleep(60)
    except KeyboardInterrupt:
        print("\n  Scheduler stopped")


if __name__ == "__main__":
    start_scheduler()
