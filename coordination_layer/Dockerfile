FROM python:3.9-slim

# 安装网络工具和编译依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    netcat-openbsd gcc && rm -rf /var/lib/apt/lists/*

WORKDIR /application

# 复制依赖清单
COPY coordination_layer/requirements.txt ./requirements.txt

# 阿里 PyPI 源安装依赖
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ && \
    pip install --no-cache-dir -r requirements.txt && \
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
CMD ["python", "-u", "/application/coordination_layer/prediction_coordinator.py"]
