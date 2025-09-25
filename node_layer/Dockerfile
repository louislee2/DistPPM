FROM python:3.9-slim

# 安装依赖工具
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc graphviz libgraphviz-dev pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /application

# 复制依赖清单
COPY node_layer/requirements.txt ./requirements.txt

# 全阿里源安装：
# 1. 普通包用阿里 PyPI 源；2. PyTorch 用阿里 Anaconda 源（稳定版 2.4.0，适配 CUDA 12.1）
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ && \
    # 临时移除 requirements.txt 中的 torch 相关依赖（单独处理）
    cp requirements.txt requirements.tmp && \
    sed -i '/torch/d' requirements.tmp && \
    # 安装普通依赖（阿里 PyPI 源）
    pip install --no-cache-dir -r requirements.tmp && \
    rm requirements.tmp && \
    # 安装 PyTorch + CUDA 依赖（阿里 Anaconda 源，延长超时到 120 秒）
    pip install torch==2.4.0 \
        --extra-index-url https://mirrors.aliyun.com/anaconda/cloud/pytorch/ \
        --timeout 120 && \
    # 清理临时文件
    rm requirements.txt

# 复制项目代码
COPY . /application/

# 确保包结构完整
RUN touch /application/__init__.py && \
    touch /application/node_layer/__init__.py && \
    touch /application/coordination_layer/__init__.py && \
    touch /application/federated_layer/__init__.py

# 设置 Python 路径
ENV PYTHONPATH=/application

# 暴露端口
EXPOSE 80

# 启动命令
CMD ["python", "-u", "/application/node_layer/predictive_activity_node.py"]
