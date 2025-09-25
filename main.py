# Copyright 2024 Kiel University
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import threading
import time
import traceback
import requests

import docker
import numpy as np
from dotenv import load_dotenv

from streaming_event_processor import StreamingEventProcessor
from coordination_layer.prediction_coordinator_client import PredictionCoordinatorClient

load_dotenv()

# 基础配置
BASE_SERVER_PORT = int(os.getenv('BASE_SERVER_PORT', 5000))
DOCKER_LABEL = str(os.getenv('DOCKER_LABEL', 'distppm'))
OUTPUTS_PATH = str(os.getenv('OUTPUTS_PATH', './outputs'))
FILE_PATH = str(os.getenv('FILE_PATH', './event_logs'))

# 预测相关配置
PREDICTION_MODEL_TYPE = str(os.getenv('PREDICTION_MODEL_TYPE', 'gru'))
PREDICTION_WINDOW_SIZE = int(os.getenv('PREDICTION_WINDOW_SIZE', 100))

# 联邦学习配置
FEDERATED_ROUNDS = int(os.getenv('FEDERATED_ROUNDS', 50))
PRIVACY_EPSILON = float(os.getenv('PRIVACY_EPSILON', 1.0))

# 镜像配置
PREDICTIVE_ACTIVITY_NODE_IMAGE = str(os.getenv('PREDICTIVE_ACTIVITY_NODE_IMAGE', 'distppm-activity-node'))
PREDICTION_COORDINATOR_IMAGE = str(os.getenv('PREDICTION_COORDINATOR_IMAGE', 'distppm-coordinator'))
FED_COORDINATOR_IMAGE = str(os.getenv('FED_COORDINATOR_IMAGE', 'distppm-fed-coordinator'))


# --- 辅助函数 ---

def attach_logs(container):
    """附加容器日志到控制台"""
    def _print(name, stream):
        for line in stream:
            print(f"[{name}] {line.decode('utf8').strip()}")

    t = threading.Thread(
        target=_print,
        args=(container.name, container.attach(logs=True, stream=True))
    )
    t.daemon = True
    t.start()


def remove_resources():
    """清理现有容器和网络资源"""
    try:
        # 移除所有相关容器
        containers = client.containers.list(
            filters={"label": DOCKER_LABEL}, all=True
        )
        for container in containers:
            container.remove(force=True)
        print("已清理现有容器")
    except Exception as exc:
        print(f"清理容器失败: {exc}")

    try:
        # 移除专用网络
        nets = client.networks.list(names=[f"{DOCKER_LABEL}_net"])
        for net in nets:
            net.remove()
        print("已清理现有网络")
    except Exception as exc:
        print(f"清理网络失败: {exc}")


def get_server_name_list_str():
    """生成节点名称列表字符串"""
    server_str = ""
    # 预测性活动节点
    for i in range(NUM_ACTIVITY_NODES):
        server_str += f"{DOCKER_LABEL}_activity_node_{i},"
    # 协调层节点
    server_str += f"{DOCKER_LABEL}_prediction_coordinator,"
    # 联邦学习协调器
    server_str += f"{DOCKER_LABEL}_fed_coordinator"
    return server_str


# --- 主程序 ---

if __name__ == "__main__":
    client = docker.from_env()

    # 初始化流式事件处理器
    stream_processor = StreamingEventProcessor(
        window_size=PREDICTION_WINDOW_SIZE,
        file_path=FILE_PATH
    )
    # 获取活动数量（用于确定预测性活动节点数量）
    NUM_ACTIVITY_NODES = stream_processor.get_activity_count()
    # 总节点数：活动节点 + 协调器 + 联邦协调器
    TOTAL_NODES = NUM_ACTIVITY_NODES + 2

    # 清理历史资源
    remove_resources()

    # 创建专用网络
    network = client.networks.create(
        f"{DOCKER_LABEL}_net",
        driver="bridge",
        attachable=True
    )
    print(f"已创建网络: {network.name}")

    # 部署节点容器
    for node_id in range(TOTAL_NODES):
        if node_id < NUM_ACTIVITY_NODES:
            # 部署预测性活动节点
            node_type = "activity_node"
            server_name = f"{DOCKER_LABEL}_{node_type}_{node_id}"
            image = PREDICTIVE_ACTIVITY_NODE_IMAGE
            activity_name = stream_processor.get_activity_name(node_id)
            
            # 环境变量配置
            env_vars = {
                "SERVER_NAME_LIST": get_server_name_list_str(),
                "SERVER_ID": node_id,
                "ACTIVITY_NAME": activity_name,
                "MODEL_TYPE": PREDICTION_MODEL_TYPE,
                "WINDOW_SIZE": PREDICTION_WINDOW_SIZE,
                "FED_ROUNDS": FEDERATED_ROUNDS
            }

        elif node_id == NUM_ACTIVITY_NODES:
            # 部署预测协调器（协调层）
            node_type = "prediction_coordinator"
            server_name = f"{DOCKER_LABEL}_{node_type}"
            image = PREDICTION_COORDINATOR_IMAGE
            
            env_vars = {
                "SERVER_NAME_LIST": get_server_name_list_str(),
                "COORDINATION_STRATEGY": os.getenv('COORDINATION_STRATEGY', 'weighted_voting'),
                "MFP_CACHE_SIZE": os.getenv('MFP_CACHE_SIZE', '1000'),
                "PREDICTION_TIMEOUT": os.getenv('PREDICTION_TIMEOUT', '1.0')
            }

        else:
            # 部署联邦学习协调器（联邦层）
            node_type = "fed_coordinator"
            server_name = f"{DOCKER_LABEL}_{node_type}"
            image = FED_COORDINATOR_IMAGE
            
            env_vars = {
                "SERVER_NAME_LIST": get_server_name_list_str(),
                "FED_ROUNDS": FEDERATED_ROUNDS,
                "PRIVACY_EPSILON": PRIVACY_EPSILON,
                "PRIVACY_DELTA": os.getenv('PRIVACY_DELTA', '1e-5'),
                "MODEL_COMPRESSION": os.getenv('MODEL_COMPRESSION', 'True')
            }

        # 启动容器
        container = client.containers.run(
            image=image,
            detach=True,
            labels={DOCKER_LABEL: node_type},
            name=server_name,
            ports={'80': ('127.0.0.1', BASE_SERVER_PORT + node_id)},
            network=f"{DOCKER_LABEL}_net",
            volumes={
                OUTPUTS_PATH: {
                    'bind': '/application/outputs',
                    'mode': 'rw'
                }
            },
            environment=env_vars
        )
        attach_logs(container)
        print(f"已启动 {node_type} 容器: {server_name} (端口: {BASE_SERVER_PORT + node_id})")

    # 系统启动完成，开始处理流式事件
    print("系统启动完成，开始处理流式事件...")
    try:
        # 等待所有节点初始化完成
        time.sleep(30)
        
        # 初始化预测协调器客户端
        pred_coordinator = PredictionCoordinatorClient(
            coord_ip=f"127.0.0.1:{BASE_SERVER_PORT + NUM_ACTIVITY_NODES}"
        )
        
        # 处理流式事件并触发预测
        while True:
            # 从流中获取下一个事件
            event = stream_processor.get_next_event()
            if not event:
                time.sleep(0.1)
                continue
            
            # 触发活动节点事件处理
            activity_id = stream_processor.get_activity_id(event['concept:name'])
            activity_port = BASE_SERVER_PORT + activity_id
            try:
                response = requests.post(
                    f"http://127.0.0.1:{activity_port}/trigger_event",
                    json=event,
                    timeout=2.0
                )
                if not response.ok:
                    print(f"活动节点 {activity_id} 处理事件失败: {response.text}")
            except Exception as e:
                print(f"发送事件到活动节点 {activity_id} 失败: {str(e)}")
            
            # 触发预测（每N个事件或特定活动后）
            if stream_processor.should_trigger_prediction(event):
                case_id = event['case:concept:name']
                try:
                    prediction = pred_coordinator.coordinate_prediction(
                        case_id=case_id,
                        last_event=event
                    )
                    if prediction:
                        print(f"案例 {case_id} 预测结果: {prediction}")
                        stream_processor.log_prediction(case_id, prediction)
                except Exception as e:
                    print(f"协调预测失败: {str(e)}")
            
            # 定期触发联邦学习轮次
            if stream_processor.should_trigger_federated_round():
                fed_coordinator_port = BASE_SERVER_PORT + TOTAL_NODES - 1
                try:
                    response = requests.post(
                        f"http://127.0.0.1:{fed_coordinator_port}/start_round",
                        timeout=30.0
                    )
                    if response.ok:
                        print(f"联邦学习轮次完成: {response.json()}")
                    else:
                        print(f"联邦学习轮次失败: {response.text}")
                except Exception as e:
                    print(f"触发联邦学习轮次失败: {str(e)}")

    except KeyboardInterrupt:
        print("\n用户中断，开始清理资源...")
    finally:
        remove_resources()
        print("系统已关闭")
