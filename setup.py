"""遥感解译助手 v3 — pip 包安装配置"""
from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="remote-sensing-tool",
    version="3.0.0",
    description="遥感解译助手 — 多源遥感植被/水体/火烧变化检测与采样点生成",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Remote Sensing Tool Team",
    python_requires=">=3.10",
    packages=find_packages(include=["engine", "engine.*", "batch", "batch.*"]),
    install_requires=[
        "numpy>=2.0",
        "rasterio>=1.3",
        "opencv-python>=4.8",
        "matplotlib>=3.7",
        "shapely>=2.0",
        "pyproj>=3.5",
        "pandas>=2.0",
        "scipy>=1.10",
        "scikit-learn>=1.2",
        "pystac-client>=0.7",
        "planetary-computer>=1.0",
        "requests>=2.28",
        "streamlit>=1.28",
        "folium>=0.15",
        "streamlit-folium>=0.20",
        "reportlab>=4.0",
        "fiona>=1.9",
        "tqdm>=4.64",
    ],
    extras_require={
        "ml": ["torch>=2.0", "torchvision>=0.15"],
        "dev": ["pytest>=7.0", "black>=23.0", "flake8>=6.0"],
    },
    entry_points={
        "console_scripts": [
            "rs-tool=main:main",
            "rs-batch=batch.batch_runner:batch_main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: GIS",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
