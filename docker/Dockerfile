FROM python:3.11-slim

WORKDIR /app

# 创建非 root 用户
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /bin/bash appuser

# 安装系统依赖 (GDAL + 中文字体 + 工具)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgdal-dev \
    gdal-bin \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    fonts-wqy-zenhei \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 配置 Matplotlib 中文字体路径
ENV MATPLOTLIBRC=/app/.matplotlib
RUN mkdir -p /app/.matplotlib && \
    echo "font.sans-serif: WenQuanYi Zen Hei, DejaVu Sans" > /app/.matplotlib/matplotlibrc && \
    echo "axes.unicode_minus: False" >> /app/.matplotlib/matplotlibrc

# 安装 Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目源码（排除不需要的文件）
COPY --chown=appuser:appuser . .

# 数据目录
RUN mkdir -p data output reports tmp && chown -R appuser:appuser data output reports tmp

# 切换到非 root 用户
USER appuser

# 暴露 Streamlit 默认端口
EXPOSE 8501

# 资源限制
ENV GDAL_CACHEMAX=512
ENV NUMEXPR_MAX_THREADS=4

# 健康检查
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# 启动 Streamlit
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false", \
     "--server.maxUploadSize=200"]
