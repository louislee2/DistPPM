# Copyright 2024 Kiel University
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from flask import Flask, request, jsonify
import os
import time
import requests
import threading
import numpy as np
import torch
from collections import defaultdict

app = Flask(__name__)


class PrivacyBudget:
    """隐私预算管理，增强容错机制"""

    def __init__(self, epsilon=1.0, delta=1e-5):
        self.initial_epsilon = epsilon
        self.epsilon = epsilon
        self.delta = delta
        self.rounds_used = 0
        self.lock = threading.Lock()  # 新增：线程安全锁

    def calculate_noise_scale(self):
        """计算噪声尺度，增加边界检查"""
        with self.lock:
            if self.epsilon <= 0 or self.delta <= 0:
                return 0.0
            try:
                return np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
            except ZeroDivisionError:
                return 0.0
            except Exception as e:
                print(f"计算噪声尺度出错: {str(e)}")
                return 0.0

    def consume_budget(self, fraction=0.1):
        """消耗部分隐私预算，确保线程安全"""
        with self.lock:
            if fraction <= 0 or fraction > 1:
                print(f"无效的预算消耗比例: {fraction}，使用默认值0.1")
                fraction = 0.1

            reduction = self.initial_epsilon * fraction
            self.epsilon = max(0, self.epsilon - reduction)
            self.rounds_used += 1
            print(f"隐私预算更新: 剩余epsilon={self.epsilon:.4f}, 已使用轮次={self.rounds_used}")

    def reset(self):
        """重置隐私预算"""
        with self.lock:
            self.epsilon = self.initial_epsilon
            self.rounds_used = 0
            print("隐私预算已重置")


class FederatedCoordinator:
    """联邦学习协调器，管理全局模型和训练轮次"""

    def __init__(self):
        # 从环境变量获取配置
        self.server_name_list = os.getenv('SERVER_NAME_LIST', '').split(',')
        self.fed_rounds = int(os.getenv('FED_ROUNDS', 10))
        self.privacy_epsilon = float(os.getenv('PRIVACY_EPSILON', 1.0))
        self.privacy_delta = float(os.getenv('PRIVACY_DELTA', 1e-5))
        self.model_compression = os.getenv('MODEL_COMPRESSION', 'True').lower() == 'true'
        self.node_timeout = int(os.getenv('NODE_TIMEOUT', 30))  # 新增：节点超时配置
        self.max_retries = int(os.getenv('MAX_RETRIES', 3))  # 新增：最大重试次数

        # 初始化组件
        self.global_model = None
        self.model_type = None
        self.training_round = 0
        self.privacy_budget = PrivacyBudget(
            epsilon=self.privacy_epsilon,
            delta=self.privacy_delta
        )
        self.participating_nodes = set()
        self.node_status = {}  # 状态: ready, training, unhealthy, offline
        self.lock = threading.Lock()

        # 性能指标
        self.federated_metrics = {
            'rounds_completed': 0,
            'average_accuracy': 0.0,
            'total_communication': 0,
            'successful_aggregations': 0,
            'failed_aggregations': 0  # 新增：聚合失败计数
        }

        print(f"联邦协调器初始化，服务器列表: {self.server_name_list}")

        # 启动节点发现
        threading.Thread(target=self.discover_nodes, daemon=True).start()

    def discover_nodes(self):
        """发现可用的节点，增强健康检查逻辑"""
        print("开始节点发现...")
        activity_node_count = 0
        health_check_interval = 10  # 检查间隔（秒）

        while True:
            discovered_count = 0
            for i, server_name in enumerate(self.server_name_list):
                # 只检查活动节点
                if 'activity_node' not in server_name:
                    continue

                try:
                    # 检查节点健康状态，增加超时配置
                    response = requests.get(
                        f"http://{server_name}:80/health",
                        timeout=2.0
                    )
                    if response.ok:
                        with self.lock:
                            self.participating_nodes.add(i)
                            self.node_status[i] = 'ready'
                        discovered_count += 1
                        # 只在状态变化时打印日志
                        if i not in [n for n in self.participating_nodes if self.node_status.get(n) == 'ready']:
                            print(f"发现活动节点: {server_name} (ID: {i})")
                    else:
                        with self.lock:
                            self.node_status[i] = 'unhealthy'
                            print(f"节点 {server_name} 健康检查失败，状态码: {response.status_code}")
                except requests.exceptions.Timeout:
                    with self.lock:
                        self.node_status[i] = 'unhealthy'
                    print(f"节点 {server_name} 健康检查超时")
                except Exception as e:
                    with self.lock:
                        self.node_status[i] = 'offline'
                    print(f"节点 {server_name} 连接失败: {str(e)}")

            if discovered_count != activity_node_count:
                print(f"活动节点状态更新: {discovered_count} 个节点就绪")
                activity_node_count = discovered_count

            # 按配置间隔检查
            time.sleep(health_check_interval)

    def select_nodes_for_training(self, fraction=0.8):
        """选择参与训练的节点，优化选择策略"""
        with self.lock:
            # 筛选状态为ready的节点
            ready_nodes = [n for n in self.participating_nodes
                           if self.node_status.get(n, 'offline') == 'ready']

            print(f"可用节点: {ready_nodes}")

            # 处理空节点列表情况
            if not ready_nodes:
                print("没有可用的节点参与训练，返回空列表")
                return []

            # 选择一定比例的节点，至少选择1个（确保有节点参与）
            num_nodes = max(1, int(len(ready_nodes) * fraction))
            num_nodes = min(num_nodes, len(ready_nodes))  # 避免超过实际可用节点数

            # 优先选择最近成功参与训练的节点（简化实现）
            if len(ready_nodes) <= num_nodes:
                selected_nodes = ready_nodes.copy()
            else:
                # 随机选择但确保可重现性
                np.random.seed(int(time.time()) % 1000)  # 简单的种子策略
                selected_nodes = np.random.choice(ready_nodes, num_nodes, replace=False).tolist()

            # 更新选中节点的状态为training
            for node_id in selected_nodes:
                self.node_status[node_id] = 'training'

            print(f"选择了 {len(selected_nodes)} 个节点参与训练: {selected_nodes}")
            return selected_nodes

    def add_differential_privacy_noise(self, model_update):
        """添加差分隐私噪声，增强参数校验"""
        if self.privacy_epsilon <= 0:
            return model_update

        noise_scale = self.privacy_budget.calculate_noise_scale()
        if noise_scale <= 0:
            print("噪声尺度无效，跳过添加噪声")
            return model_update

        # 为模型参数添加噪声，增加类型检查
        try:
            for param_name, param_value in model_update.items():
                if isinstance(param_value, list):
                    param_array = np.array(param_value)
                    # 确保参数是数值型
                    if not np.issubdtype(param_array.dtype, np.number):
                        print(f"参数 {param_name} 不是数值类型，跳过添加噪声")
                        continue
                    noise = np.random.normal(0, noise_scale, param_array.shape)
                    noisy_param = (param_array + noise).tolist()
                    model_update[param_name] = noisy_param
            return model_update
        except Exception as e:
            print(f"添加差分隐私噪声失败: {str(e)}")
            return model_update

    def aggregate_model_updates(self, local_updates):
        """聚合本地模型更新，增强容错和日志"""
        if not local_updates or len(local_updates) == 0:
            print("没有模型更新可以聚合")
            with self.lock:
                self.federated_metrics['failed_aggregations'] += 1
            return None

        print(f"开始聚合 {len(local_updates)} 个本地模型更新")

        try:
            # 如果是第一轮且没有全局模型，使用第一个更新作为初始模型
            if self.global_model is None:
                first_update = next(iter(local_updates.values()))
                if 'parameters' in first_update:
                    self.global_model = first_update['parameters']
                    self.model_type = self.global_model.get('model_type', 'gru')
                    print(f"初始化全局模型，类型: {self.model_type}")
                    with self.lock:
                        self.federated_metrics['successful_aggregations'] += 1
                    return self.global_model

            # 加权平均（按样本数量）
            total_samples = sum(update.get('sample_count', 1) for update in local_updates.values())
            if total_samples == 0:
                total_samples = len(local_updates)
                print("所有节点样本数为0，使用节点数量加权")

            print(f"总样本数: {total_samples}")

            # 初始化聚合参数
            aggregated_params = {}
            param_names = None

            # 检查所有更新的参数名称是否一致
            for node_id, update in local_updates.items():
                if 'parameters' not in update:
                    continue
                current_params = set(update['parameters'].keys())
                if param_names is None:
                    param_names = current_params
                else:
                    if current_params != param_names:
                        print(f"节点 {node_id} 参数名称不匹配，跳过该节点更新")
                        del local_updates[node_id]

            # 重新计算总样本数（移除不匹配的节点后）
            total_samples = sum(update.get('sample_count', 1) for update in local_updates.values())
            if total_samples == 0:
                total_samples = len(local_updates)

            # 收集所有参数
            for node_id, update in local_updates.items():
                if 'parameters' not in update:
                    continue

                params = update['parameters']
                sample_count = update.get('sample_count', 1)
                weight = sample_count / total_samples

                print(f"节点 {node_id}: 样本数={sample_count}, 权重={weight:.3f}")

                # 聚合参数
                for param_name, param_value in params.items():
                    # 跳过元数据
                    if param_name in ['model_type', 'activity_to_idx', 'n',
                                      'hidden_size', 'num_layers', 'sequence_length',
                                      'd_model', 'nhead']:
                        aggregated_params[param_name] = param_value
                        continue

                    # 初始化参数数组
                    if param_name not in aggregated_params:
                        try:
                            if isinstance(param_value, list):
                                aggregated_params[param_name] = np.zeros_like(np.array(param_value), dtype=np.float64)
                            else:
                                aggregated_params[param_name] = 0.0
                        except Exception as e:
                            print(f"初始化参数 {param_name} 失败: {str(e)}")
                            aggregated_params[param_name] = 0.0

                    # 加权累加
                    try:
                        if isinstance(param_value, list):
                            aggregated_params[param_name] += np.array(param_value, dtype=np.float64) * weight
                        else:
                            aggregated_params[param_name] += float(param_value) * weight
                    except Exception as e:
                        print(f"聚合参数 {param_name} 失败: {str(e)}")

            # 转换回列表格式
            for param_name, param_value in aggregated_params.items():
                if isinstance(param_value, np.ndarray):
                    aggregated_params[param_name] = param_value.tolist()

            # 应用差分隐私
            if self.privacy_epsilon > 0:
                aggregated_params = self.add_differential_privacy_noise(aggregated_params)

            # 更新全局模型
            self.global_model = aggregated_params
            print("模型聚合完成")

            with self.lock:
                self.federated_metrics['successful_aggregations'] += 1

            return self.global_model

        except Exception as e:
            print(f"模型聚合过程出错: {str(e)}")
            with self.lock:
                self.federated_metrics['failed_aggregations'] += 1
            return None

    def distribute_global_model(self, node_ids):
        """向节点分发全局模型，增强重试机制"""
        if not self.global_model:
            print("没有全局模型可以分发")
            return False

        success_count = 0
        retry_nodes = []

        # 第一轮分发
        for node_id in node_ids:
            if self._send_model_to_node(node_id):
                success_count += 1
            else:
                retry_nodes.append(node_id)
                print(f"节点 {node_id} 模型分发失败，加入重试列表")

        # 重试失败的节点
        if retry_nodes and self.max_retries > 1:
            print(f"开始重试分发模型到 {len(retry_nodes)} 个节点")
            for attempt in range(1, self.max_retries):
                current_retry = []
                for node_id in retry_nodes:
                    if self._send_model_to_node(node_id):
                        success_count += 1
                    else:
                        current_retry.append(node_id)
                        time.sleep(2)  # 重试间隔
                retry_nodes = current_retry
                if not retry_nodes:
                    break
            if retry_nodes:
                print(f"最终有 {len(retry_nodes)} 个节点分发失败: {retry_nodes}")

        print(f"模型分发完成，成功 {success_count}/{len(node_ids)} 个节点")
        return success_count > 0

    def _send_model_to_node(self, node_id):
        """向单个节点发送模型，作为分发逻辑的辅助函数"""
        try:
            if 0 <= node_id < len(self.server_name_list):
                node_name = self.server_name_list[node_id]
                url = f"http://{node_name}:80/update_model"

                payload = {'parameters': self.global_model}

                if self.model_compression:
                    compressed_payload = self.compress_model(payload)
                else:
                    compressed_payload = payload

                # 发送请求，使用配置的超时时间
                response = requests.post(
                    url,
                    json=compressed_payload,
                    timeout=self.node_timeout
                )

                if response.ok:
                    with self.lock:
                        self.node_status[node_id] = 'ready'
                    print(f"成功向节点 {node_id} 分发模型")
                    return True
                else:
                    print(f"向节点 {node_id} 分发模型失败: {response.text}")
                    with self.lock:
                        self.node_status[node_id] = 'unhealthy'
                    return False
            else:
                print(f"无效的节点ID: {node_id}")
                return False
        except Exception as e:
            print(f"向节点 {node_id} 分发模型时出错: {str(e)}")
            with self.lock:
                self.node_status[node_id] = 'offline'
            return False

    def compress_model(self, model_data):
        """压缩模型数据，增强压缩逻辑"""
        try:
            compressed = {}
            for k, v in model_data.items():
                if isinstance(v, list):
                    # 量化为float16并处理空列表
                    if len(v) == 0:
                        compressed[k] = v
                    else:
                        # 检查是否为数值列表
                        try:
                            arr = np.array(v)
                            if np.issubdtype(arr.dtype, np.number):
                                compressed[k] = arr.astype(np.float16).tolist()
                            else:
                                compressed[k] = v  # 非数值列表不压缩
                        except Exception as e:
                            print(f"压缩参数 {k} 失败: {str(e)}")
                            compressed[k] = v
                else:
                    compressed[k] = v
            return compressed
        except Exception as e:
            print(f"模型压缩失败，返回原始数据: {str(e)}")
            return model_data

    def start_federated_round(self):
        """启动一轮联邦学习，增强流程控制和错误处理"""
        if self.training_round >= self.fed_rounds:
            print("已完成所有联邦学习轮次")
            return {
                'status': 'completed',
                'round': self.training_round,
                'total_rounds': self.fed_rounds
            }

        try:
            # 1. 选择参与节点
            selected_nodes = self.select_nodes_for_training()
            if not selected_nodes:
                print("无法选择参与节点，跳过本轮训练")
                return {
                    'status': 'error',
                    'message': '没有可用节点参与训练',
                    'round': self.training_round
                }

            # 2. 等待节点完成本地训练并收集更新
            local_updates = {}
            retry_count = 0
            collection_timeout = self.node_timeout  # 收集超时时间
            start_time = time.time()

            while retry_count < self.max_retries and len(local_updates) < len(selected_nodes):
                # 检查是否超时
                if time.time() - start_time > collection_timeout:
                    print(f"收集模型更新超时（{collection_timeout}秒）")
                    break

                for node_id in selected_nodes:
                    if node_id in local_updates:
                        continue  # 已收集的节点跳过

                    try:
                        node_name = self.server_name_list[node_id]
                        url = f"http://{node_name}:80/get_model_parameters"
                        response = requests.get(url, timeout=10.0)

                        if response.ok:
                            update_data = response.json()
                            if 'parameters' in update_data:
                                local_updates[node_id] = update_data
                                print(f"已收集节点 {node_id} 的模型更新")
                            else:
                                print(f"节点 {node_id} 返回数据不包含模型参数")
                    except requests.exceptions.Timeout:
                        print(f"收集节点 {node_id} 超时")
                    except Exception as e:
                        print(f"收集节点 {node_id} 的模型更新失败: {str(e)}")

                if len(local_updates) < len(selected_nodes):
                    retry_count += 1
                    print(f"未收集到所有节点的更新，重试 {retry_count}/{self.max_retries}")
                    time.sleep(5)  # 重试间隔

            if not local_updates:
                print("未收集到任何模型更新，本轮训练失败")
                # 重置节点状态
                with self.lock:
                    for node_id in selected_nodes:
                        self.node_status[node_id] = 'unhealthy'
                return {
                    'status': 'error',
                    'message': '未收集到模型更新',
                    'round': self.training_round
                }

            print(f"成功收集 {len(local_updates)}/{len(selected_nodes)} 个节点的模型更新")

            # 3. 聚合模型更新
            aggregated_model = self.aggregate_model_updates(local_updates)
            if not aggregated_model:
                print("模型聚合失败")
                # 重置节点状态
                with self.lock:
                    for node_id in selected_nodes:
                        self.node_status[node_id] = 'ready'
                return {
                    'status': 'error',
                    'message': '模型聚合失败',
                    'round': self.training_round
                }

            # 4. 分发更新后的全局模型
            distribution_success = self.distribute_global_model(selected_nodes)

            # 5. 更新轮次和指标
            self.training_round += 1
            self.federated_metrics['rounds_completed'] = self.training_round

            print(f"第 {self.training_round}/{self.fed_rounds} 轮联邦学习完成")
            return {
                'status': 'success',
                'round': self.training_round,
                'total_rounds': self.fed_rounds,
                'nodes_updated': len(selected_nodes),
                'successful_nodes': len(local_updates),
                'privacy_remaining': self.privacy_budget.epsilon
            }

        except Exception as e:
            print(f"联邦学习轮次执行出错: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'round': self.training_round
            }


# 初始化联邦协调器
coordinator = FederatedCoordinator()


# API路由
@app.route('/start_round', methods=['POST'])
def start_round():
    result = coordinator.start_federated_round()
    return jsonify(result)


@app.route('/get_status', methods=['GET'])
def get_status():
    with coordinator.lock:
        return jsonify({
            'round': coordinator.training_round,
            'total_rounds': coordinator.fed_rounds,
            'node_status': coordinator.node_status,
            'privacy_budget': {
                'remaining_epsilon': coordinator.privacy_budget.epsilon,
                'rounds_used': coordinator.privacy_budget.rounds_used
            },
            'metrics': coordinator.federated_metrics
        })


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'service': 'federated_coordinator',
        'round': coordinator.training_round,
        'nodes': len(coordinator.participating_nodes),
        'uptime': time.time() - coordinator.start_time if hasattr(coordinator, 'start_time') else 0
    })


if __name__ == "__main__":
    # 记录启动时间用于健康检查
    coordinator.start_time = time.time()
    app.run(host='0.0.0.0', port=80, debug=False)