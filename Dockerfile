# -----------------------------------------------------------------
# 阶段 1: 基础环境 (适配你的 CUDA 12.1)
# -----------------------------------------------------------------
# (基于你的 torch 2.5.1+cu121)
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# -----------------------------------------------------------------
# 阶段 2: 配置环境和安装 Python 3.11
# -----------------------------------------------------------------

# 2. 设置环境变量
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1


# 3. 安装 Python 3.11, 编译工具, 并为 3.11 安装 pip
#    (最终修正版：修复了 ln 命令的语法错误)
RUN apt-get update && apt-get install -y --no-install-recommends \
    # 1. 安装 Python 3.11 和 3.11-dev
    python3.11 \
    python3.11-dev \
    \
    # 2. 安装编译工具 (gcc, git) 和下载工具 (curl)
    git \
    curl \
    build-essential \
    \
    # 3. 安装系统依赖 (OpenCV, Matplotlib)
    libgl1-mesa-glx \
    libglib2.0-0 \
    && \
    # 4. 为 3.11 安装专属 pip
    curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py \
    && \
    python3.11 get-pip.py \
    && \
    rm get-pip.py \
    && \
    # ----------------------------------------------------
    # 5. [!!] 建立软链接 (已修复) [!!]
    # ----------------------------------------------------
    ln -sf /usr/bin/python3.11 /usr/bin/python && \
    # [!!] 我之前在这里漏掉了 '&&' [!!]
    ln -sf /usr/local/bin/pip /usr/bin/pip \
    && \
    # ----------------------------------------------------
    \
    # 6. 清理
    rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------
# 阶段 3: 安装 Python 依赖 (来自你的 requirements.txt)
# -----------------------------------------------------------------

# 4. 设置工作目录
WORKDIR /app

# 5. 复制依赖文件
COPY requirements.txt .

# 6. 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu121

# -----------------------------------------------------------------
# 阶段 4: 复制项目代码并运行
# -----------------------------------------------------------------

# 7. 复制你所有的项目代码
# (这会把你 *修改后* 的 data_config.py 复制进来)
COPY . .

# 8. 定义默认启动命令
# (假设你的 main_cd_new.py 也不需要参数，会自己跑)
CMD ["python", "main_cd_new.py"]