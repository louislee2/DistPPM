#!/bin/bash
# 终极修复脚本 - 全阿里源版（解决 PyTorch 安装问题）

set -e

echo "🚀 开始 DistPPM 终极修复..."

# 1. 清理所有资源
echo "🧹 清理所有 Docker 资源..."
docker ps -q --filter "label=distppm" | xargs -r docker stop
docker ps -aq --filter "label=distppm" | xargs -r docker rm
docker images -q distppm-* | xargs -r docker rmi -f
docker network ls -q --filter "name=distppm" | xargs -r docker network rm

# 2. 确保包结构
echo "📦 设置包结构..."
touch __init__.py
touch node_layer/__init__.py
touch coordination_layer/__init__.py
touch federated_layer/__init__.py

# 3. 修复构建脚本
echo "🔧 修复构建脚本..."
cat > build_docker_images.sh << 'EOF'
#!/bin/bash
set -e

ACTIVITY_NODE_IMAGE="distppm-activity-node"
COORDINATOR_IMAGE="distppm-coordinator"
FED_COORDINATOR_IMAGE="distppm-fed-coordinator"

echo "🏗️  构建 DistPPM Docker 镜像..."

if [ ! -f "main.py" ]; then
    echo "❌ 请在项目根目录运行"
    exit 1
fi

touch __init__.py
touch node_layer/__init__.py
touch coordination_layer/__init__.py
touch federated_layer/__init__.py

echo "🔧 构建预测性活动节点镜像..."
docker build -f node_layer/Dockerfile -t $ACTIVITY_NODE_IMAGE .

echo "🔧 构建预测协调器镜像..."
docker build -f coordination_layer/Dockerfile -t $COORDINATOR_IMAGE .

echo "🔧 构建联邦学习协调器镜像..."
docker build -f federated_layer/Dockerfile -t $FED_COORDINATOR_IMAGE .

echo "✅ 构建完成:"
docker images | grep distppm
EOF

chmod +x build_docker_images.sh

# 4. 修复 node_layer Dockerfile（全阿里源：PyPI + Anaconda）
echo "🐳 修复 node_layer Dockerfile..."
cat > node_layer/Dockerfile << 'EOF'
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
EOF

# 5. 修复 coordination_layer Dockerfile（无需 PyTorch，保持阿里 PyPI 源）
echo "🐳 修复 coordination_layer Dockerfile..."
cat > coordination_layer/Dockerfile << 'EOF'
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
EOF

# 6. 修复 federated_layer Dockerfile（同 node_layer，全阿里源）
echo "🐳 修复 federated_layer Dockerfile..."
cat > federated_layer/Dockerfile << 'EOF'
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
EOF

# 7. 创建测试数据（保持不变）
if [ ! -d "data" ] || [ -z "$(ls -A data 2>/dev/null)" ]; then
    echo "📊 创建测试数据..."
    mkdir -p data
    cat > data/sample_log.csv << 'EOF'
case:concept:name,concept:name,time:timestamp
Case_1,Activity_A,2024-01-01T10:00:00
Case_1,Activity_B,2024-01-01T10:30:00
Case_1,Activity_C,2024-01-01T11:00:00
Case_2,Activity_A,2024-01-01T10:15:00
Case_2,Activity_D,2024-01-01T10:45:00
Case_3,Activity_B,2024-01-01T10:20:00
Case_3,Activity_C,2024-01-01T10:50:00
EOF
fi

# 8. 构建镜像
echo "🏗️  开始构建镜像（全阿里源）..."
./build_docker_images.sh

# 9. 验证
echo "✅ 验证镜像构建..."
if docker images | grep -q distppm-activity-node; then
    echo "✅ 全阿里源镜像构建成功！"
    echo "现在可以启动系统："
    echo "python main.py"
else
    echo "❌ 镜像构建失败"
    exit 1
fi