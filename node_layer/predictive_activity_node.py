# Copyright 2024 Kiel University
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from flask import Flask, request, jsonify
import os
import sys
import time
import json
import numpy as np
import threading
from collections import defaultdict, deque
import torch  # 新增：引入PyTorch用于模型管理

# 添加项目根目录到Python路径
sys.path.append('/application')

# 绝对导入
from node_layer.local_prediction_engine import LocalPredictionEngine
from coordination_layer.prediction_coordinator_client import PredictionCoordinatorClient

app = Flask(__name__)


class TraceBuffer:
    """轨迹缓冲区，存储案例的事件序列（增强版）"""

    def __init__(self, window_size=1000, event_ttl=3600):  # 新增：事件存活时间（秒）
        self.window_size = window_size
        self.event_ttl = event_ttl  # 事件过期时间
        self.traces = {}  # case_id -> list of events
        self.access_time = {}  # case_id -> timestamp
        self.lock = threading.Lock()
        # 启动定期清理线程
        self.cleanup_thread = threading.Thread(target=self._periodic_cleanup, daemon=True)
        self.cleanup_thread.start()

    def _periodic_cleanup(self):
        """定期清理过期事件"""
        while True:
            time.sleep(60)  # 每分钟清理一次
            with self.lock:
                current_time = time.time()
                # 清理过期案例
                expired_cases = [
                    case_id for case_id, ts in self.access_time.items()
                    if current_time - ts > self.event_ttl
                ]
                for case_id in expired_cases:
                    del self.traces[case_id]
                    del self.access_time[case_id]
                if expired_cases:
                    print(f"清理了 {len(expired_cases)} 个过期案例轨迹")

    def add_event(self, case_id, event):
        """添加事件到轨迹"""
        with self.lock:
            if case_id not in self.traces:
                self.traces[case_id] = []

            # 添加事件并维护窗口大小
            self.traces[case_id].append(event)
            self.access_time[case_id] = time.time()

            # 如果超出容量，移除最久未访问的案例
            if len(self.traces) > self.window_size:
                oldest_case = min(self.access_time, key=self.access_time.get)
                del self.traces[oldest_case]
                del self.access_time[oldest_case]

    def get_trace(self, case_id):
        """获取案例的轨迹"""
        with self.lock:
            if case_id in self.traces:
                self.access_time[case_id] = time.time()  # 更新访问时间
                return self.traces[case_id]
            return []

    def get_recent_traces(self, limit=100):
        """获取最近的轨迹用于训练"""
        with self.lock:
            # 按最近访问时间排序
            sorted_cases = sorted(
                self.access_time.items(),
                key=lambda x: x[1],
                reverse=True
            )[:limit]

            return [self.traces[case_id] for case_id, _ in sorted_cases]

    def clear_old_traces(self, max_keep=50):
        """手动清理旧轨迹（新增方法）"""
        with self.lock:
            if len(self.traces) > max_keep:
                sorted_cases = sorted(
                    self.access_time.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[max_keep:]
                for case_id, _ in sorted_cases:
                    del self.traces[case_id]
                    del self.access_time[case_id]
                return f"已清理 {len(sorted_cases)} 个旧轨迹"
        return "无需清理"


class PredictiveActivityNode:
    """预测性活动节点，处理特定类型的活动并提供预测功能"""

    def __init__(self):
        # 从环境变量获取配置
        self.server_id = int(os.getenv('SERVER_ID', 0))
        self.activity_name = os.getenv('ACTIVITY_NAME', 'unknown')
        self.server_name_list = os.getenv('SERVER_NAME_LIST', '').split(',')
        self.model_type = os.getenv('MODEL_TYPE', 'gru')
        self.window_size = int(os.getenv('WINDOW_SIZE', 100))
        self.fed_rounds = int(os.getenv('FED_ROUNDS', 50))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 新增：设备配置

        # 初始化组件
        self.trace_buffer = TraceBuffer(
            window_size=self.window_size,
            event_ttl=int(os.getenv('EVENT_TTL', 3600))  # 可通过环境变量配置
        )
        self.prediction_engine = LocalPredictionEngine(default_model=self.model_type)
        self.coordinator_client = self._init_coordinator_client()
        self.local_training_data = []
        self.event_counter = 0
        self.model_version = 0
        self.training_lock = threading.Lock()  # 新增：训练锁防止并发冲突

        # 联邦学习相关
        self.fed_client = None
        self.init_federated_client()
        self.fed_round_in_progress = False  # 新增：联邦轮次状态标记

        # 性能指标（增强）
        self.performance_metrics = {
            'predictions_made': 0,
            'correct_predictions': 0,
            'average_latency': 0.0,
            'training_rounds': 0,  # 新增
            'model_updates': 0,  # 新增
            'memory_usage': 0  # 新增
        }

    def _init_coordinator_client(self):
        """初始化协调器客户端"""
        # 找到协调器的位置（在服务器列表中的位置）
        coord_index = None
        for i, name in enumerate(self.server_name_list):
            if 'prediction_coordinator' in name:
                coord_index = i
                break

        if coord_index is not None and coord_index < len(self.server_name_list):
            coord_name = self.server_name_list[coord_index]
            return PredictionCoordinatorClient(f"{coord_name}:80")
        return None

    def init_federated_client(self):
        """初始化联邦学习客户端（增强版）"""
        try:
            from federated_layer.federated_client import FederatedClient

            # 找到联邦协调器的位置
            fed_coord_index = None
            for i, name in enumerate(self.server_name_list):
                if 'fed_coordinator' in name:
                    fed_coord_index = i
                    break

            if fed_coord_index is not None:
                fed_coord_name = self.server_name_list[fed_coord_index]
                self.fed_client = FederatedClient(
                    node_id=self.server_id,
                    coordinator_url=f"http://{fed_coord_name}:80"
                )
                print(f"联邦学习客户端初始化成功，连接到: {fed_coord_name}")
        except Exception as e:
            print(f"初始化联邦学习客户端失败: {str(e)}")
            self.fed_client = None

    def process_event(self, event):
        """处理事件并更新本地数据结构"""
        start_time = time.time()

        case_id = event.get('case:concept:name')
        if not case_id:
            return {'status': 'error', 'message': '缺少案例ID'}

        # 数据清洗：确保事件包含必要字段
        cleaned_event = {
            'case:concept:name': case_id,
            'concept:name': event.get('concept:name', 'unknown_activity'),
            'time:timestamp': event.get('time:timestamp', str(time.time())),
            **{k: v for k, v in event.items() if k.startswith('org:') or k.startswith('case:')}
        }

        # 将事件添加到轨迹缓冲区
        self.trace_buffer.add_event(case_id, cleaned_event)
        self.event_counter += 1

        # 注册事件到协调器
        if self.coordinator_client:
            self.coordinator_client.register_case_event(
                case_id=case_id,
                node_id=self.server_id,
                event=cleaned_event
            )

        # 定期更新训练数据
        if self.event_counter % 10 == 0:
            self._update_training_data()

            # 定期自动训练
            if self.event_counter % 30 == 0 and len(self.local_training_data) >= 5:
                threading.Thread(target=self._background_training, daemon=True).start()

        latency = time.time() - start_time
        return {
            'status': 'success',
            'message': f"活动节点 {self.server_id} 已处理事件",
            'latency': latency
        }

    def _background_training(self):
        """后台训练任务"""
        try:
            result = self.local_training(epochs=5)
            if result.get('status') == 'success':
                print(f"节点 {self.server_id} 后台训练完成，模型版本: {self.model_version}")
        except Exception as e:
            print(f"后台训练失败: {str(e)}")

    def _update_training_data(self):
        """更新本地训练数据并自动训练"""
        recent_traces = self.trace_buffer.get_recent_traces(limit=50)
        self.local_training_data = recent_traces

        # 自动训练条件：有足够的轨迹且模型未训练
        if len(recent_traces) >= 5 and self.model_version == 0:
            try:
                result = self.local_training(epochs=3)
                if result.get('status') == 'success':
                    print(f"节点 {self.server_id} 自动训练完成，模型版本: {self.model_version}")
            except Exception as e:
                print(f"自动训练失败: {str(e)}")

        print(f"节点 {self.server_id} 已更新训练数据，共 {len(recent_traces)} 个轨迹")

    def _frequency_based_prediction(self, trace):
        """基于频率的简单预测（当模型未训练时使用）"""
        if not self.local_training_data:
            return {'activity': 'unknown', 'confidence': 0.0}

        # 收集所有活动及其频率
        activity_counts = {}
        for tr in self.local_training_data:
            for event in tr:
                act = event.get('concept:name', 'unknown')
                activity_counts[act] = activity_counts.get(act, 0) + 1

        if not activity_counts:
            return {'activity': 'unknown', 'confidence': 0.0}

        # 返回最频繁的活动
        most_frequent = max(activity_counts, key=activity_counts.get)
        total_count = sum(activity_counts.values())
        confidence = activity_counts[most_frequent] / total_count

        return {
            'activity': most_frequent,
            'confidence': min(confidence, 0.8)  # 限制最大置信度
        }

    def predict_next_activity(self, case_id, last_event=None):
        """预测案例的下一个活动（修复版）"""
        start_time = time.time()

        # 获取案例轨迹
        trace = self.trace_buffer.get_trace(case_id)
        if not trace and last_event:
            trace = [last_event]

        if not trace:
            return {
                'next_activity': 'unknown',
                'confidence': 0.0,
                'node_id': self.server_id,
                'model_version': self.model_version
            }

        # 使用本地预测引擎进行预测
        try:
            prediction = self.prediction_engine.predict_next_activity(trace)

            # 如果模型未训练或置信度为0，尝试基于历史频率的简单预测
            if prediction['confidence'] == 0.0 and self.local_training_data:
                prediction = self._frequency_based_prediction(trace)

        except Exception as e:
            print(f"预测失败: {str(e)}")
            prediction = self._frequency_based_prediction(trace)

        # 记录性能指标
        self.performance_metrics['predictions_made'] += 1
        latency = time.time() - start_time
        self.performance_metrics['average_latency'] = (
                                                              self.performance_metrics['average_latency'] *
                                                              (self.performance_metrics[
                                                                   'predictions_made'] - 1) + latency
                                                      ) / self.performance_metrics['predictions_made']

        return {
            'next_activity': prediction['activity'],
            'confidence': prediction['confidence'],
            'remaining_time': prediction.get('remaining_time'),
            'node_id': self.server_id,
            'model_version': self.model_version
        }

    def update_local_model(self, model_parameters):
        """更新本地模型（线程安全版）"""
        with self.training_lock:
            try:
                self.prediction_engine.update_model_parameters(model_parameters)
                self.model_version += 1
                self.performance_metrics['model_updates'] += 1  # 记录模型更新次数
                return {'status': 'success', 'new_version': self.model_version}
            except Exception as e:
                return {'status': 'error', 'message': str(e)}

    def get_local_model_parameters(self):
        """获取本地模型参数"""
        try:
            params = self.prediction_engine.get_model_parameters()
            if not params:
                return {
                    'parameters': None,
                    'node_id': self.server_id,
                    'sample_count': len(self.local_training_data),
                    'model_type': self.model_type,
                    'status': 'no_model'
                }
            # 模型参数压缩（新增）
            compressed_params = self._compress_parameters(params)
            return {
                'parameters': compressed_params,
                'node_id': self.server_id,
                'sample_count': len(self.local_training_data),
                'model_type': self.model_type
            }
        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    def _compress_parameters(self, params):
        """压缩模型参数以减少传输量（新增）"""
        if not params:
            return params

        compressed = {}
        for k, v in params.items():
            if isinstance(v, list) and len(v) > 0:
                # 量化为较低精度的浮点数
                try:
                    compressed[k] = [round(float(x), 4) for x in v]
                except:
                    compressed[k] = v
            elif isinstance(v, np.ndarray):
                # 转换为列表并量化
                compressed[k] = np.round(v, 4).tolist()
            else:
                compressed[k] = v
        return compressed

    def local_training(self, epochs=3):
        """执行本地训练（线程安全版）"""
        with self.training_lock:
            if not self.local_training_data:
                return {'status': 'error', 'message': '没有训练数据'}

            print(f"节点 {self.server_id} 开始训练，数据量: {len(self.local_training_data)}")

            try:
                # 训练模型
                history = self.prediction_engine.train(
                    self.local_training_data,
                    epochs=epochs
                )

                # 检查训练是否成功
                if history.get('status') == 'error':
                    return history

                self.model_version += 1
                self.performance_metrics['training_rounds'] += 1

                print(f"节点 {self.server_id} 训练完成，模型版本: {self.model_version}")

                return {
                    'status': 'success',
                    'training_history': history,
                    'model_version': self.model_version
                }
            except Exception as e:
                print(f"节点 {self.server_id} 训练失败: {str(e)}")
                return {'status': 'error', 'message': str(e)}

    def get_performance_metrics(self):
        """获取节点性能指标（增强版）"""
        # 计算内存使用量（近似）
        self.performance_metrics['memory_usage'] = len(self.trace_buffer.traces) * 0.5  # 估算值
        return self.performance_metrics


# 初始化活动节点
node = PredictiveActivityNode()


# API路由（新增联邦学习相关接口）
@app.route('/trigger_event', methods=['POST'])
def trigger_event():
    event = request.json
    if not event:
        return jsonify({'error': '没有提供事件数据'}), 400

    result = node.process_event(event)
    return jsonify(result)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if not data or 'case_id' not in data:
        return jsonify({'error': '缺少案例ID'}), 400

    prediction = node.predict_next_activity(
        data['case_id'],
        data.get('last_event')
    )
    return jsonify(prediction)


@app.route('/update_model', methods=['POST'])
def update_model():
    data = request.json
    if not data or 'parameters' not in data:
        return jsonify({'error': '缺少模型参数'}), 400

    result = node.update_local_model(data['parameters'])
    return jsonify(result)


@app.route('/get_model_parameters', methods=['GET'])
def get_model_parameters():
    result = node.get_local_model_parameters()
    return jsonify(result)


@app.route('/local_training', methods=['POST'])
def local_training():
    data = request.json or {}
    epochs = int(data.get('epochs', 3))
    result = node.local_training(epochs=epochs)
    return jsonify(result)


@app.route('/get_performance', methods=['GET'])
def get_performance():
    metrics = node.get_performance_metrics()
    return jsonify(metrics)


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'service': 'predictive_activity_node',
        'activity_name': node.activity_name,
        'node_id': node.server_id,
        'model_version': node.model_version,
        'fed_client_connected': bool(node.fed_client),
        'device': str(node.device)  # 新增：设备信息
    })


@app.route('/debug_status', methods=['GET'])
def debug_status():
    """调试状态接口"""
    try:
        model_trained = node.prediction_engine.models[node.prediction_engine.active_model].trained
    except:
        model_trained = False

    return jsonify({
        'node_id': node.server_id,
        'activity_name': node.activity_name,
        'model_version': node.model_version,
        'training_data_count': len(node.local_training_data),
        'trace_buffer_size': len(node.trace_buffer.traces),
        'event_counter': node.event_counter,
        'model_trained': model_trained,
        'active_model': node.prediction_engine.active_model,
        'performance_metrics': node.performance_metrics
    })


@app.route('/force_training', methods=['POST'])
def force_training():
    """强制训练接口"""
    data = request.json or {}
    epochs = int(data.get('epochs', 5))

    if len(node.local_training_data) == 0:
        return jsonify({
            'status': 'error',
            'message': '没有训练数据，请先处理一些事件'
        })

    result = node.local_training(epochs=epochs)
    return jsonify(result)


if __name__ == "__main__":
    # 配置生产环境参数
    app.run(
        host='0.0.0.0',
        port=80,
        debug=False,
        threaded=True,
        processes=1
    )