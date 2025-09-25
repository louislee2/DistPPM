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
