# -----------------------------------------------------------------
# 1. 基础镜像
# -----------------------------------------------------------------
# 【修复】使用你找到的、兼容 Torch 2.6 的真实 MCR 路径
# 这与你的 flash_attn-2.7.4.post1+cu12torch2.6...whl 完美匹配
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch260

# -----------------------------------------------------------------
# 2. 翻译你的 `environment.setup:` 块
# -----------------------------------------------------------------
# 【翻译】export DEBIAN_FRONTEND=noninteractive (修复 apt 弹窗)
ENV DEBIAN_FRONTEND=noninteractive

# 【翻译】sudo apt update && sudo apt-get install -y libglib2.0-0
# (在 Dockerfile 中，RUN 默认就是 root，所以不需要 sudo)
RUN apt-get update -y && \
    apt-get install -y libglib2.0-0 curl git && \
    # 清理 apt 缓存
    rm -rf /var/lib/apt/lists/*

# 【翻译】curl -LsSf https://astral.sh/uv/install.sh | sh
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# 【翻译】source $$HOME/.local/bin/env
# (在 Dockerfile 中, $HOME 是 /root, 我们用 ENV 把它永久添加到 PATH)
ENV PATH="/root/.local/bin:$PATH"

# -----------------------------------------------------------------
# 3. 拷贝你的所有代码 (使用 .dockerignore)
# -----------------------------------------------------------------
# 我们把所有代码（$CONFIG_DIR）复制到镜像的 /app 目录
WORKDIR /app
COPY . .

# -----------------------------------------------------------------
# 4. 翻译你的 `jobs.command:` 块 (安装部分)
# -----------------------------------------------------------------

# 【翻译】安装 Miniconda
RUN echo "--- 1. Installing Miniconda ---" && \
    curl -L -O "https://github.com/conda-forge/miniforge/releases/download/25.9.1-0/Miniforge3-25.9.1-0-Linux-x86_64.sh" && \
    # 【修复】安装到 /opt/miniforge3，而不是一个临时的 ./miniforge3
    bash Miniforge3-25.9.1-0-Linux-x86_64.sh -b -p /opt/miniforge3 && \
    rm Miniforge3-25.9.1-0-Linux-x86_64.sh && \
    # 【翻译】./miniforge3/bin/conda init bash
    /opt/miniforge3/bin/conda init bash

# 【翻译】设置 SandboxFusion 服务器 (环境 1)
# (我们用 /bin/bash -c "..." 来确保 source 和 conda activate 在同一个 RUN 步骤中生效)
RUN echo "--- 2. Setting up SandboxFusion Server Environment ---" && \
    . /opt/miniforge3/etc/profile.d/conda.sh && \
    mamba create -n sandbox-runtime -y python=3.11 && \
    conda activate sandbox-runtime && \
    which python && python --version && \
    curl -sSL https://install.python-poetry.org | python - && \
    # 【翻译】安装 SandboxFusion 依赖 (在 /app/SandboxFusion)
    cd /app/SandboxFusion && \
    poetry install && \
    # 【翻译】pip install -r ./requirements.txt (这是你脚本里的命令)
    pip install -r ./requirements.txt && \
    # 【翻译】下载 NLTK 数据
    python -c "import nltk; nltk.download('punkt')" && \
    python -c "import nltk; nltk.download('stopwords')" && \
    # 【翻译】运行测试 (在构建时)
    mkdir -p docs/build && \
    make test-case CASE=test_python_assert && \
    cd /app && \
    # 【翻译】conda activate base && conda deactivate
    conda activate base && \
    conda deactivate && \
    # 清理缓存
    mamba clean -afy

# 【翻译】设置你的第二个环境 (verl)
# (这同样在你混乱的脚本中，它创建了 .venv 而不是 conda env)
RUN echo "--- 3. Setting up Conda Env: verl-env ---" && \
    # 再次激活 conda 的 shell 功能
    . /opt/miniforge3/etc/profile.d/conda.sh && \
    # 1. 创建第二个 conda 环境
    mamba create -n verl-env -y python=3.11 && \
    # 2. 激活这个新环境
    conda activate verl-env && \
    # 3. 使用 uv (它在基础 PATH 中) 来安装包
    #    uv 会自动识别已激活的 conda 环境，并将包安装进去
    which python && python --version && \
    uv pip install -e .[vllm] && \
    uv pip install --no-cache-dir flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl && \
    # 4. 清理并停用
    conda deactivate && \
    mamba clean -afy
# -----------------------------------------------------------------
# 5. 最终设置
# -----------------------------------------------------------------
# 设置默认的工作目录
WORKDIR /app
# 设置默认的 shell，确保 .bashrc (被 conda init 修改过) 会被加载
CMD ["/bin/bash"]