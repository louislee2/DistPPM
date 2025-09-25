#!/bin/bash
# DistPPM 系统管理脚本（无 docker-compose 版本）

set -e

# 检查 Python 是否可用
if ! command -v python &> /dev/null; then
    echo "错误: Python 未安装或不可用"
    exit 1
fi

# 检查并创建目录
[ -d "./event_logs" ] || { echo "创建 event_logs 目录..."; mkdir -p ./event_logs; }
[ -d "./outputs" ] || { echo "创建 outputs 目录..."; mkdir -p ./outputs/predictions; }

case "$1" in
    start)
        echo "启动 DistPPM 系统..."
        python main.py
        ;;

    build)
        echo "构建 Docker 镜像..."
        ./build_docker_images.sh
        ;;

    *)
        echo "用法: $0 {start|build}"
        exit 1
        ;;
esac
