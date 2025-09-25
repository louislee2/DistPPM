FROM python:3.9-slim

# 安装编译依赖和网络工具
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc netcat-openbsd && rm -rf /var/lib/apt/lists/*

WORKDIR /application

# 复制依赖清单
COPY federated_layer/requirements.txt ./requirements.txt

# 全阿里源安装：普通包 + PyTorch
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ && \
    # 临时移除 torch 依赖
    cp requirements.txt requirements.tmp && \
    sed -i '/torch/d' requirements.tmp && \
    # 安装普通依赖
    pip install --no-cache-dir -r requirements.tmp && \
    rm requirements.tmp && \
    # 阿里 Anaconda 源安装 PyTorch（稳定版 2.4.0）
    pip install torch==2.4.0 \
        --extra-index-url https://mirrors.aliyun.com/anaconda/cloud/pytorch/ \
        --timeout 120 && \
    # 清理
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
CMD ["python", "-u", "/application/federated_layer/federated_coordinator.py"]
