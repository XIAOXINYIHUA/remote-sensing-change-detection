"""
本机任务执行器：在后台进程中执行耗时任务（检索下载、变化检测、时间序列、批量任务），
页面只提交任务并轮询进度；切换页面、刷新浏览器都不会打断任务。

任务记录（默认 <项目>/data/jobs，环境变量 RSAP_JOBS_DIR 可改），每个任务一个目录:

    job.json     任务规格，创建后不变：kind、target、params、title、meta、runner
    status.json  状态：state、progress、message、started_at、finished_at、error、pid
    result.pkl   成功时的返回值（pickle）
    cancel       取消请求标记
    log.txt      工作进程的输出（engine 日志与异常堆栈）

状态: queued → running → succeeded / failed / cancelled；
      执行器所在进程退出时未结束的任务，下次启动执行器时记为 interrupted。

执行方式: 最多 max_workers 个任务同时运行，每个任务一个独立的 Python 进程（结束即释放内存，
GDAL 缓存等不会在服务进程里累积）。取消时先在进度回调处协作退出，宽限期后仍未退出则终止进程。
执行器所在进程（Streamlit 服务）正常退出时终止工作进程，这些任务下次启动时记为 interrupted。
mode="thread" 在本进程的线程中执行（测试用，或多进程不可用时的退路），取消只能在进度回调处生效。

工作进程用 subprocess 启动，没有用 multiprocessing：spawn 会在子进程里重新执行父进程的 __main__，
而 Streamlit（及其 AppTest）把 __main__ 换成了页面脚本，每个工作进程都会把整个应用跑一遍。
也没有用 concurrent.futures.ProcessPoolExecutor：它不能终止运行中的任务，退出时还会等所有任务跑完。
本模块只依赖标准库，工作进程启动时不额外加载 streamlit / numpy。
"""

import atexit
import importlib
import json
import os
import pickle
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

ACTIVE_STATES = ("queued", "running")
FINAL_STATES = ("succeeded", "failed", "cancelled", "interrupted")

# 任务类型 → 入口 "模块:函数"，签名 fn(params: dict, progress_callback) -> 结果
KINDS = {
    "cd.collect": "services.job_tasks:cd_collect",
    "cd.run": "services.job_tasks:cd_run",
    "cl.download": "services.job_tasks:cl_download",
    "ts.run": "services.job_tasks:ts_run",
    "batch.task": "services.job_tasks:batch_task",
}

CANCEL_GRACE_S = 5.0        # 取消后等待协作退出的时间，超时终止进程
PROGRESS_INTERVAL_S = 0.5   # 进度写盘的最小间隔
KEEP_FINISHED = 200         # 执行器启动时最多保留的已结束任务数

_JOB_ID = re.compile(r"^[\w-]+$")


def default_jobs_dir():
    """任务目录：环境变量 RSAP_JOBS_DIR，默认 <项目>/data/jobs。"""
    if os.environ.get("RSAP_JOBS_DIR"):
        return os.environ["RSAP_JOBS_DIR"]
    from services import BASE_DIR
    return str(BASE_DIR / "data" / "jobs")


class JobCancelled(BaseException):
    """工作进程内检测到取消请求。

    继承 BaseException：流水线里 `except Exception` 的兜底不会把取消吞掉。
    """


@dataclass
class Job:
    """任务快照（读取时的状态）。"""
    id: str
    kind: str
    title: str
    created_at: str
    params: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
    state: str = "queued"
    progress: float = 0.0
    message: str = ""
    started_at: str = ""
    finished_at: str = ""
    error: str = ""
    cancel_requested: bool = False

    @property
    def active(self):
        return self.state in ACTIVE_STATES

    @property
    def elapsed_s(self):
        """已运行（结束后为总共运行）的秒数；尚未开始为 0。"""
        if not self.started_at:
            return 0.0
        end = datetime.fromisoformat(self.finished_at) if self.finished_at else datetime.now()
        return max((end - datetime.fromisoformat(self.started_at)).total_seconds(), 0.0)


# =====================================================================
#  任务目录读写
# =====================================================================

def _now():
    return datetime.now().isoformat(timespec="seconds")


def _write_json(path, data):
    """原子写入（先写临时文件再替换）；Windows 上目标正被读取时稍后重试。"""
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    for _ in range(40):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.05)
    os.replace(tmp, path)


def _read_json(path, default=None):
    for _ in range(40):
        try:
            return json.loads(Path(path).read_text(encoding="utf-8"))
        except FileNotFoundError:
            return default
        except (PermissionError, json.JSONDecodeError):
            time.sleep(0.05)
    return default


def _update_status(job_dir, **fields):
    path = Path(job_dir) / "status.json"
    status = _read_json(path, {}) or {}
    status.update(fields)
    _write_json(path, status)


# =====================================================================
#  工作进程
# =====================================================================

def _resolve(target):
    module, _, name = target.partition(":")
    return getattr(importlib.import_module(module), name)


def _execute(job_dir):
    """执行一个任务，结果与状态写入任务目录（工作进程或线程中调用）。"""
    job_dir = Path(job_dir)
    spec = _read_json(job_dir / "job.json")
    cancel_flag = job_dir / "cancel"
    if cancel_flag.exists():
        _update_status(job_dir, state="cancelled", message="已取消", finished_at=_now())
        return
    _update_status(job_dir, state="running", message="开始执行", started_at=_now(), pid=os.getpid())
    last_write = [0.0]

    def progress(msg, pct):
        if cancel_flag.exists():
            raise JobCancelled()
        now = time.monotonic()
        if now - last_write[0] >= PROGRESS_INTERVAL_S or pct >= 100:
            last_write[0] = now
            _update_status(job_dir, progress=round(min(max(float(pct), 0.0), 100.0), 1),
                           message=str(msg))

    try:
        result = _resolve(spec["target"])(spec["params"], progress)
        tmp = job_dir / ".result.pkl.tmp"
        with open(tmp, "wb") as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, job_dir / "result.pkl")
        _update_status(job_dir, state="succeeded", progress=100.0, message="完成",
                       finished_at=_now())
    except JobCancelled:
        _update_status(job_dir, state="cancelled", message="已取消", finished_at=_now())
    except Exception as e:
        from services import ServiceError
        expected = isinstance(e, (ServiceError, ValueError))     # 无影像、参数无效等
        if not expected:
            traceback.print_exc()
        _update_status(job_dir, state="failed", finished_at=_now(),
                       error=str(e) if expected else f"{type(e).__name__}: {e}")


# 工作进程的启动代码：先恢复父进程的 sys.path（项目根目录、测试目录等），再执行任务
_BOOTSTRAP = ("import json, os, sys; sys.path[:0] = json.loads(os.environ.pop('RSAP_JOB_SYS_PATH')); "
              "from services.jobs import _execute; _execute(sys.argv[1])")


class _ProcessWorker:
    """一个任务的工作进程（接口与 threading.Thread 的 is_alive / join 一致）。

    stdout / stderr（含 engine 日志、异常堆栈、GDAL 的输出）写入任务目录的 log.txt。
    """

    def __init__(self, job_dir):
        env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1",
               "RSAP_JOB_SYS_PATH": json.dumps([os.path.abspath(p) for p in sys.path])}
        self._log = open(Path(job_dir) / "log.txt", "ab")
        try:
            self._proc = subprocess.Popen(
                [sys.executable, "-c", _BOOTSTRAP, str(job_dir)], env=env,
                stdin=subprocess.DEVNULL, stdout=self._log, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except Exception:
            self._log.close()
            raise

    def is_alive(self):
        return self._proc.poll() is None

    @property
    def exitcode(self):
        return self._proc.poll()

    def terminate(self):
        if self.is_alive():
            self._proc.terminate()

    def join(self, timeout=None):
        try:
            self._proc.wait(timeout)
        except subprocess.TimeoutExpired:
            pass

    def close(self):
        self._log.close()


# =====================================================================
#  执行器
# =====================================================================

class JobRunner:
    """任务执行器：提交、调度、取消、查询。一个任务目录在一个进程里只应有一个执行器（见 get_runner）。"""

    def __init__(self, jobs_dir=None, max_workers=2, mode="process"):
        if mode not in ("process", "thread"):
            raise ValueError(f"未知执行方式 {mode}，可选 process / thread")
        self.dir = Path(jobs_dir or default_jobs_dir())
        self.dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max(1, int(max_workers))
        self.mode = mode
        self.id = uuid.uuid4().hex
        self._lock = threading.RLock()
        self._last_id_dt = None           # 保证任务 ID 时间戳严格递增（同毫秒会破坏 list() 顺序）
        self._queue = deque()
        self._running = {}              # job_id -> [worker, 取消请求时刻 | None]
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread = None
        self._recover()
        self.prune()
        atexit.register(self._terminate_workers)

    # ---- 查询 ----

    def _job_dirs(self):
        """任务目录，新的在前（任务 ID 以时间开头）。"""
        return sorted((p for p in self.dir.iterdir() if p.is_dir() and _JOB_ID.match(p.name)),
                      key=lambda p: p.name, reverse=True)

    def get(self, job_id):
        """任务快照；不存在时 None。"""
        if not job_id or not _JOB_ID.match(str(job_id)):
            return None
        job_dir = self.dir / job_id
        spec = _read_json(job_dir / "job.json")
        if not spec:
            return None
        status = _read_json(job_dir / "status.json", {}) or {}
        return Job(
            id=job_id, kind=spec.get("kind", ""), title=spec.get("title", ""),
            created_at=spec.get("created_at", ""),
            params=spec.get("params", {}), meta=spec.get("meta", {}),
            state=status.get("state", "queued"), progress=float(status.get("progress", 0)),
            message=status.get("message", ""), started_at=status.get("started_at", ""),
            finished_at=status.get("finished_at", ""), error=status.get("error", ""),
            cancel_requested=(job_dir / "cancel").exists(),
        )

    def list(self, limit=None):
        """任务列表，新的在前。"""
        dirs = self._job_dirs()
        jobs = (self.get(d.name) for d in (dirs[:limit] if limit else dirs))
        return [job for job in jobs if job is not None]

    def queue_position(self, job_id):
        """排队任务前面还有几个（含自己，从 1 起）；不在队列中为 0。"""
        with self._lock:
            return self._queue.index(job_id) + 1 if job_id in self._queue else 0

    def result(self, job_id):
        """成功任务的返回值；没有结果时 None。"""
        path = self.dir / job_id / "result.pkl"
        if not _JOB_ID.match(str(job_id)) or not path.exists():
            return None
        with open(path, "rb") as f:
            return pickle.load(f)

    def log_tail(self, job_id, lines=40):
        path = self.dir / job_id / "log.txt"
        if not _JOB_ID.match(str(job_id)) or not path.exists():
            return ""
        text = path.read_text(encoding="utf-8", errors="replace")
        return "\n".join(text.splitlines()[-lines:])

    # ---- 提交与管理 ----

    def submit(self, kind, params, *, title="", meta=None):
        """提交任务，返回任务 ID。

        kind 为 KINDS 中的任务类型，或 "模块:函数" 形式的入口；params / meta 须可 JSON 序列化。
        """
        target = KINDS.get(kind, kind)
        if ":" not in target:
            raise ValueError(f"未知任务类型 {kind}")
        # 时间戳严格递增：同一毫秒内连续提交时 +1ms，确保目录名排序 = 提交顺序
        with self._lock:
            now = datetime.now()
            if self._last_id_dt is not None and now <= self._last_id_dt:
                now = self._last_id_dt + timedelta(milliseconds=1)
            self._last_id_dt = now
        job_id = f"{now:%Y%m%d-%H%M%S-%f}"[:-3] + f"-{uuid.uuid4().hex[:4]}"
        spec = {"id": job_id, "kind": kind, "target": target, "title": title or kind,
                "params": params, "meta": meta or {}, "created_at": _now(), "runner": self.id}
        json.dumps(spec, ensure_ascii=False)      # 不可序列化时在建目录之前报错
        job_dir = self.dir / job_id
        job_dir.mkdir(parents=True)
        _write_json(job_dir / "job.json", spec)
        _update_status(job_dir, state="queued", progress=0.0, message="排队中")
        with self._lock:
            self._queue.append(job_id)
        self._ensure_dispatcher()
        self._wake.set()
        return job_id

    def retry(self, job_id):
        """以相同参数重新提交，返回新任务 ID。"""
        spec = _read_json(self.dir / job_id / "job.json")
        if not spec:
            raise ValueError(f"任务 {job_id} 不存在")
        return self.submit(spec["kind"], spec["params"], title=spec.get("title", ""),
                           meta=spec.get("meta"))

    def cancel(self, job_id):
        """取消排队或运行中的任务；任务已结束时返回 False。"""
        job = self.get(job_id)
        if job is None or not job.active:
            return False
        (self.dir / job_id / "cancel").touch()
        with self._lock:
            if job_id in self._queue:
                self._queue.remove(job_id)
                _update_status(self.dir / job_id, state="cancelled", message="已取消",
                               finished_at=_now())
            elif job_id in self._running:
                self._running[job_id][1] = time.monotonic()
        self._wake.set()
        return True

    def delete(self, job_id):
        """删除已结束任务的记录（不影响分析输出与影像缓存）。"""
        job = self.get(job_id)
        if job is None or job.active:
            return False
        shutil.rmtree(self.dir / job_id, ignore_errors=True)
        return True

    def prune(self, keep=KEEP_FINISHED):
        """只保留最近 keep 个已结束任务的记录。"""
        finished = [d for d in self._job_dirs()
                    if (_read_json(d / "status.json", {}) or {}).get("state") in FINAL_STATES]
        for job_dir in finished[keep:]:
            shutil.rmtree(job_dir, ignore_errors=True)

    def _recover(self):
        """上一个执行器留下的未结束任务记为 interrupted（执行器重启后它们不会再运行）。"""
        for job_dir in self._job_dirs():
            status = _read_json(job_dir / "status.json", {}) or {}
            if status.get("state") in ACTIVE_STATES:
                _update_status(job_dir, state="interrupted", finished_at=_now(),
                               error="应用关闭或重启时任务尚未完成，可在任务中心重试")

    # ---- 调度 ----

    def _ensure_dispatcher(self):
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                self._stop.clear()
                self._thread = threading.Thread(target=self._loop, name="rsap-job-dispatcher",
                                                daemon=True)
                self._thread.start()

    def _loop(self):
        while not self._stop.is_set():
            self._wake.wait(0.25)
            self._wake.clear()
            try:
                self._tick()
            except Exception:
                traceback.print_exc()

    def _tick(self):
        with self._lock:
            for job_id, entry in list(self._running.items()):
                worker, cancel_at = entry
                if worker.is_alive():
                    if (cancel_at is not None and self.mode == "process"
                            and time.monotonic() - cancel_at > CANCEL_GRACE_S):
                        worker.terminate()          # 协作退出超时（如正在下载大文件）
                        worker.join(5)
                    continue
                del self._running[job_id]
                self._finalize(job_id, worker)
            while self._queue and len(self._running) < self.max_workers:
                job_id = self._queue.popleft()
                job_dir = self.dir / job_id
                if (job_dir / "cancel").exists():
                    _update_status(job_dir, state="cancelled", message="已取消", finished_at=_now())
                    continue
                try:
                    self._running[job_id] = [self._start(job_dir), None]
                except Exception as e:
                    _update_status(job_dir, state="failed", finished_at=_now(),
                                   error=f"无法启动工作进程: {type(e).__name__}: {e}")

    def _start(self, job_dir):
        if self.mode == "process":
            return _ProcessWorker(job_dir)
        worker = threading.Thread(target=_execute, args=(str(job_dir),), daemon=True,
                                  name=f"rsap-job-{job_dir.name}")
        worker.start()
        return worker

    def _finalize(self, job_id, worker):
        """工作进程已退出：没写下结束状态的（被终止 / 崩溃）在这里补上。"""
        job_dir = self.dir / job_id
        state = (_read_json(job_dir / "status.json", {}) or {}).get("state")
        if state not in FINAL_STATES:
            if (job_dir / "cancel").exists():
                _update_status(job_dir, state="cancelled", message="已取消", finished_at=_now())
            else:
                code = getattr(worker, "exitcode", None)
                _update_status(job_dir, state="failed", finished_at=_now(),
                               error=f"工作进程异常退出（退出码 {code}），详见任务日志")
        if hasattr(worker, "close"):
            worker.close()

    def _terminate_workers(self):
        """服务进程退出时（atexit）：终止工作进程，不改状态，下次启动时由 _recover 记为 interrupted。"""
        with self._lock:
            for worker, _ in self._running.values():
                if self.mode == "process":
                    worker.terminate()

    def shutdown(self, timeout=10):
        """停止调度：排队任务取消，运行中的任务终止（线程方式只能等它在进度回调处退出）。"""
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout)
        with self._lock:
            for job_id in list(self._queue):
                _update_status(self.dir / job_id, state="cancelled", message="已取消",
                               finished_at=_now())
            self._queue.clear()
            for job_id, (worker, _) in list(self._running.items()):
                (self.dir / job_id / "cancel").touch()
                if self.mode == "process":
                    worker.terminate()
                worker.join(timeout)
                if not worker.is_alive():
                    self._finalize(job_id, worker)
            self._running.clear()


# =====================================================================
#  进程内共享的执行器
# =====================================================================

_RUNNERS = {}
_RUNNERS_LOCK = threading.Lock()


def get_runner(jobs_dir=None, max_workers=None, mode=None):
    """本进程中某个任务目录的执行器，重复调用返回同一个。

    max_workers 默认取环境变量 RSAP_JOB_WORKERS，否则 config.JOB_MAX_WORKERS；
    mode 默认取环境变量 RSAP_JOB_MODE，否则 "process"。
    """
    jobs_dir = str(Path(jobs_dir or default_jobs_dir()).resolve())
    mode = mode or os.environ.get("RSAP_JOB_MODE") or "process"
    if max_workers is None:
        import config
        max_workers = int(os.environ.get("RSAP_JOB_WORKERS") or getattr(config, "JOB_MAX_WORKERS", 2))
    with _RUNNERS_LOCK:
        runner = _RUNNERS.get((jobs_dir, mode))
        if runner is None:
            runner = _RUNNERS[(jobs_dir, mode)] = JobRunner(jobs_dir, max_workers, mode)
        runner.max_workers = max(1, int(max_workers))
        return runner


def shutdown_all():
    """停止本进程中的全部执行器（测试收尾用）。"""
    with _RUNNERS_LOCK:
        runners = list(_RUNNERS.values())
        _RUNNERS.clear()
    for runner in runners:
        runner.shutdown()
