from flask import Flask, request, jsonify
import os
import time
import requests
from collections import defaultdict, deque
import numpy as np
import threading
from streaming_event_processor import StreamingEventProcessor



app = Flask(__name__)


class PredictionCache:
    """预测结果缓存"""

    def __init__(self, max_size=1000):
        self.cache = {}
        self.max_size = max_size
        self.access_order = deque()

    def get(self, case_id):
        if case_id in self.cache:
            # 更新访问顺序
            self.access_order.remove(case_id)
            self.access_order.append(case_id)
            return self.cache[case_id]
        return None

    def put(self, case_id, prediction):
        if case_id in self.cache:
            self.access_order.remove(case_id)
        elif len(self.cache) >= self.max_size:
            # LRU淘汰策略
            oldest = self.access_order.popleft()
            del self.cache[oldest]

        self.cache[case_id] = {
            'prediction': prediction,
            'timestamp': time.time()
        }
        self.access_order.append(case_id)


class DistributedCaseRegistry:
    """分布式案例注册表，跟踪案例在各节点的分布"""

    def __init__(self):
        self.case_distribution = defaultdict(set)  # case_id -> {node_ids}
        self.node_cases = defaultdict(set)  # node_id -> {case_ids}
        self.case_last_event = {}  # case_id -> last_event
        self.lock = threading.Lock()

    def register_case_event(self, case_id, node_id, event):
        """注册案例事件，更新案例分布"""
        with self.lock:
            self.case_distribution[case_id].add(node_id)
            self.node_cases[node_id].add(case_id)
            self.case_last_event[case_id] = event

    def get_participating_nodes(self, case_id):
        """获取参与处理特定案例的节点"""
        return self.case_distribution.get(case_id, set())

    def get_case_last_event(self, case_id):
        """获取案例的最后一个事件"""
        return self.case_last_event.get(case_id)


class PredictionCoordinator:
    """预测协调器，负责协调多节点预测"""

    def __init__(self):
        # 从环境变量获取配置
        server_list_str = os.getenv('SERVER_NAME_LIST', '')
        self.server_name_list = server_list_str.split(',') if server_list_str else []
        self.coordination_strategy = os.getenv('COORDINATION_STRATEGY', 'weighted_voting')
        self.mfp_cache_size = int(os.getenv('MFP_CACHE_SIZE', 1000))
        self.prediction_timeout = float(os.getenv('PREDICTION_TIMEOUT', 1.0))

        # 初始化组件
        self.case_registry = DistributedCaseRegistry()
        self.prediction_cache = PredictionCache(max_size=self.mfp_cache_size)
        self.mfp_cache = defaultdict(dict)  # Most Frequent Predictors缓存
        self.node_performance = defaultdict(lambda: {'accuracy': 0.8, 'count': 10})  # 初始默认值

        print(f"预测协调器初始化完成，服务器列表: {self.server_name_list}")

    def update_node_performance(self, node_id, correct):
        """更新节点性能指标"""
        perf = self.node_performance[node_id]
        total = perf['count']
        new_accuracy = (perf['accuracy'] * total + (1 if correct else 0)) / (total + 1)
        self.node_performance[node_id] = {
            'accuracy': new_accuracy,
            'count': total + 1
        }

    def calculate_node_weights(self, case_id, node_ids):
        """计算节点权重"""
        if not node_ids:
            return {}

        # 基础权重基于节点性能
        base_weights = {
            node_id: self.node_performance[node_id]['accuracy']
            for node_id in node_ids
        }

        # 归一化权重
        total = sum(base_weights.values())
        if total == 0:
            return {node_id: 1 / len(node_ids) for node_id in node_ids}

        return {node_id: w / total for node_id, w in base_weights.items()}

    def request_prediction_from_node(self, node_id, case_id):
        """向节点请求预测"""
        try:
            # 获取节点地址
            if node_id < len(self.server_name_list):
                node_name = self.server_name_list[node_id]
                url = f"http://{node_name}:80/predict"

                # 获取案例最后一个事件
                last_event = self.case_registry.get_case_last_event(case_id)
                if not last_event:
                    return None

                # 发送预测请求
                response = requests.post(
                    url,
                    json={
                        'case_id': case_id,
                        'last_event': last_event
                    },
                    timeout=self.prediction_timeout
                )

                if response.ok:
                    return response.json()
                else:
                    print(f"节点 {node_id} 预测请求失败: {response.text}")
                    return None
            return None
        except Exception as e:
            print(f"向节点 {node_id} 请求预测失败: {str(e)}")
            return None

    def aggregate_predictions(self, predictions, case_id):
        """聚合多个预测结果"""
        if not predictions:
            return None

        # 计算节点权重
        weights = self.calculate_node_weights(case_id, predictions.keys())

        # 下一活动预测聚合
        activity_votes = defaultdict(float)
        for node_id, pred in predictions.items():
            if pred.get('status') == 'error':
                continue
            next_activity = pred['next_activity']
            activity_votes[next_activity] += weights[node_id] * pred['confidence']

        if not activity_votes:
            return None

        best_activity = max(activity_votes, key=activity_votes.get)
        total_votes = sum(activity_votes.values())
        confidence = activity_votes[best_activity] / total_votes if total_votes > 0 else 0

        # 剩余时间预测聚合
        remaining_times = []
        weights_list = []
        for node_id, pred in predictions.items():
            if pred.get('status') == 'error':
                continue
            if 'remaining_time' in pred and pred['remaining_time'] is not None:
                remaining_times.append(pred['remaining_time'])
                weights_list.append(weights[node_id])

        avg_remaining_time = None
        if remaining_times:
            avg_remaining_time = np.average(remaining_times, weights=weights_list)

        result = {
            'next_activity': best_activity,
            'confidence': confidence,
            'participating_nodes': list(predictions.keys())
        }

        if avg_remaining_time is not None:
            result['remaining_time'] = avg_remaining_time

        return result

    def coordinate_prediction(self, case_id, last_event=None):
        """协调多节点预测"""
        # 先检查缓存
        cached_pred = self.prediction_cache.get(case_id)
        if cached_pred:
            # 检查缓存是否过期（5分钟）
            if time.time() - cached_pred['timestamp'] < 300:
                return cached_pred['prediction']

        # 获取参与节点
        participating_nodes = self.case_registry.get_participating_nodes(case_id)
        if not participating_nodes:
            # 如果没有参与节点且提供了last_event，尝试确定相关节点
            if last_event and 'concept:name' in last_event:
                # 简化实现：将活动映射到节点ID
                activity_name = last_event['concept:name']
                # 这里需要一个活动到节点的映射机制
                # 暂时使用hash方式分配
                activity_node = hash(activity_name) % len([n for n in self.server_name_list if 'activity_node' in n])
                participating_nodes = {activity_node}

        if len(participating_nodes) == 1:
            # 单节点案例，直接返回本地预测
            node_id = next(iter(participating_nodes))
            prediction = self.request_prediction_from_node(node_id, case_id)
        else:
            # 多节点案例，需要协调
            predictions = {}
            for node_id in participating_nodes:
                pred = self.request_prediction_from_node(node_id, case_id)
                if pred:
                    predictions[node_id] = pred

            prediction = self.aggregate_predictions(predictions, case_id)

        # 缓存预测结果
        if prediction:
            self.prediction_cache.put(case_id, prediction)

        return prediction


# 初始化协调器
coordinator = PredictionCoordinator()


# API路由
@app.route('/register_case_event', methods=['POST'])
def register_case_event():
    data = request.json
    if not data or 'case_id' not in data or 'node_id' not in data or 'event' not in data:
        return jsonify({'error': '缺少必要参数'}), 400

    coordinator.case_registry.register_case_event(
        data['case_id'],
        data['node_id'],
        data['event']
    )
    return jsonify({'status': 'success'})


@app.route('/coordinate_prediction', methods=['POST'])
def coordinate_prediction():
    data = request.json
    if not data or 'case_id' not in data:
        return jsonify({'error': '缺少案例ID'}), 400

    prediction = coordinator.coordinate_prediction(
        data['case_id'],
        data.get('last_event')
    )

    if prediction:
        return jsonify(prediction)
    return jsonify({'error': '无法获取预测结果'}), 500


@app.route('/update_node_performance', methods=['POST'])
def update_node_performance():
    data = request.json
    if not data or 'node_id' not in data or 'correct' not in data:
        return jsonify({'error': '缺少必要参数'}), 400

    coordinator.update_node_performance(
        data['node_id'],
        data['correct']
    )
    return jsonify({'status': 'success'})


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'service': 'prediction_coordinator'})


if __name__ == "__main__":
    print("启动预测协调器")
    app.run(host='0.0.0.0', port=80, debug=False)