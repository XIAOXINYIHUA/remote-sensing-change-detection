FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 创建非 root 用户
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /bin/bash appuser

# 系统依赖：rasterio / fiona 的 wheel 已自带 GDAL，opencv 用 headless 版无需 OpenGL；
# 这里只装中文字体、OpenMP 运行库和健康检查用的 curl。
# 可选镜像源（不指定则用官方源），例如 --build-arg APT_MIRROR=https://mirrors.tuna.tsinghua.edu.cn
ARG APT_MIRROR=""
RUN if [ -n "$APT_MIRROR" ]; then \
        sed -i "s|http://deb.debian.org|$APT_MIRROR|g" /etc/apt/sources.list.d/debian.sources; \
    fi \
    && apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    fonts-wqy-zenhei \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 配置 Matplotlib 中文字体
ENV MATPLOTLIBRC=/app/.matplotlib
RUN mkdir -p /app/.matplotlib && \
    echo "font.sans-serif: WenQuanYi Zen Hei, DejaVu Sans" > /app/.matplotlib/matplotlibrc && \
    echo "axes.unicode_minus: False" >> /app/.matplotlib/matplotlibrc

# 先装依赖（利用层缓存），再复制源码（.dockerignore 排除数据/输出/测试）。
# 可选镜像源（不指定则用 PyPI），例如 --build-arg PIP_MIRROR=https://pypi.tuna.tsinghua.edu.cn/simple
COPY requirements.txt .
ARG PIP_MIRROR=""
RUN if [ -n "$PIP_MIRROR" ]; then export PIP_INDEX_URL="$PIP_MIRROR"; fi \
    && pip install -r requirements.txt

COPY --chown=appuser:appuser . .

RUN mkdir -p data output reports tmp .cache && chown -R appuser:appuser data output reports tmp .cache

# 项目状态文件放进 data（compose 挂载的卷）：/app 归 root 所有写不进，且重建容器会丢
ENV RSAP_STATE_DIR=/app/data
# matplotlib 字体缓存与 fontconfig 缓存：HOME=/app 不可写，否则每个进程启动都重建字体缓存
ENV XDG_CACHE_HOME=/app/.cache \
    MPLCONFIGDIR=/app/.cache/matplotlib

USER appuser

EXPOSE 8501

# 资源限制
ENV GDAL_CACHEMAX=512 \
    NUMEXPR_MAX_THREADS=4

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false", \
     "--server.maxUploadSize=200"]
