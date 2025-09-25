# Copyright 2024 Kiel University
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
import json
import csv
import random
from collections import defaultdict, deque
import requests


class StreamingEventProcessor:
    """流式事件处理器，负责读取和分发事件日志，新增案例完成检测和性能反馈功能"""

    def __init__(self, window_size=100, file_path="./event_logs"):
        self.window_size = window_size
        self.file_path = file_path
        self.event_buffer = deque(maxlen=window_size)
        self.activity_list = self._discover_activities()
        self.activity_id_mapping = {act: i for i, act in enumerate(self.activity_list)}
        self.case_events = defaultdict(list)  # 存储每个案例的事件
        self.case_completed = set()  # 标记已完成的案例
        self.event_counter = 0
        self.fed_round_counter = 0
        self.fed_round_interval = 100  # 每100个事件触发一次联邦学习检查
        # 记录预测结果用于后续验证
        self.prediction_records = defaultdict(list)  # case_id -> list of predictions

    def _discover_activities(self):
        """从日志文件中发现所有活动"""
        activities = set()

        # 检查目录中的所有CSV文件
        for filename in os.listdir(self.file_path):
            if filename.endswith(".csv"):
                file_path = os.path.join(self.file_path, filename)
                try:
                    with open(file_path, 'r', newline='', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        if 'concept:name' in reader.fieldnames:
                            for row in reader:
                                activities.add(row['concept:name'])
                except Exception as e:
                    print(f"读取文件 {filename} 时出错: {str(e)}")

        # 也检查XES文件（简化处理）
        for filename in os.listdir(self.file_path):
            if filename.endswith(".xes"):
                print(f"注意: XES文件 {filename} 处理未实现，仅使用CSV文件")

        return list(activities)

    def get_activity_count(self):
        """返回活动数量"""
        return len(self.activity_list)

    def get_activity_name(self, activity_id):
        """根据ID获取活动名称"""
        if 0 <= activity_id < len(self.activity_list):
            return self.activity_list[activity_id]
        return "unknown"

    def get_activity_id(self, activity_name):
        """根据活动名称获取ID"""
        return self.activity_id_mapping.get(activity_name, -1)

    def get_next_event(self):
        """获取下一个事件（模拟流式处理）"""
        # 在实际应用中，这应该连接到实时数据源
        time.sleep(0.1)  # 模拟事件间隔

        # 随机选择一个文件
        csv_files = [f for f in os.listdir(self.file_path) if f.endswith(".csv")]
        if not csv_files:
            return None

        # 随机选择一个文件和一行
        filename = random.choice(csv_files)
        file_path = os.path.join(self.file_path, filename)

        try:
            with open(file_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if not rows:
                    return None

                # 随机选择一行，但倾向于选择较新的案例
                row = random.choices(rows, weights=[i + 1 for i in range(len(rows))])[0]

                # 确保必要的字段存在
                required_fields = ['case:concept:name', 'concept:name', 'time:timestamp']
                for field in required_fields:
                    if field not in row:
                        row[field] = f"default_{field}_{random.randint(1, 1000)}"

                # 添加事件ID和完成标记（假设包含"lifecycle:transition"字段标识完成）
                row['event_id'] = f"event_{self.event_counter}"
                row['case:completed'] = row.get('lifecycle:transition') == 'complete'
                self.event_counter += 1

                # 保存到案例事件列表
                case_id = row['case:concept:name']
                self.case_events[case_id].append(row)

                # 检查案例是否已完成
                if row['case:completed']:
                    self.case_completed.add(case_id)
                    self._validate_case_predictions(case_id)  # 验证该案例的所有预测

                # 添加到事件缓冲区
                self.event_buffer.append(row)

                return row
        except Exception as e:
            print(f"获取事件时出错: {str(e)}")
            return None

    def _validate_case_predictions(self, case_id):
        """验证已完成案例的预测结果并反馈给节点"""
        if case_id not in self.prediction_records or case_id not in self.case_events:
            return

        # 获取案例的实际活动序列
        actual_events = self.case_events[case_id]
        actual_activities = [event['concept:name'] for event in actual_events]

        # 验证每个预测
        for pred in self.prediction_records[case_id]:
            pred_index = pred['event_index']  # 预测时的事件位置
            # 确保实际活动序列长度足够（预测下一个活动的实际值）
            if pred_index + 1 < len(actual_activities):
                actual_next = actual_activities[pred_index + 1]
                is_correct = (pred['predicted_activity'] == actual_next)

                # 反馈给对应节点
                self._send_feedback(
                    node_id=pred['node_id'],
                    case_id=case_id,
                    correct=is_correct
                )

        # 清除已验证的记录
        del self.prediction_records[case_id]

    def _send_feedback(self, node_id, case_id, correct):
        """向活动节点发送预测准确性反馈"""
        # 构建节点地址（假设节点端口为BASE_SERVER_PORT + node_id）
        # 实际应用中应从配置获取基础端口
        base_port = int(os.getenv('BASE_SERVER_PORT', 5000))
        node_port = base_port + node_id
        url = f"http://127.0.0.1:{node_port}/feedback_prediction"

        try:
            response = requests.post(
                url,
                json={
                    'case_id': case_id,
                    'correct': correct
                },
                timeout=2.0
            )
            if not response.ok:
                print(f"向节点 {node_id} 发送反馈失败: {response.text}")
        except Exception as e:
            print(f"发送反馈到节点 {node_id} 失败: {str(e)}")

    def should_trigger_prediction(self, event):
        """判断是否应该触发预测"""
        case_id = event['case:concept:name']
        case_events = self.case_events.get(case_id, [])

        # 策略1: 每个案例每N个事件触发一次预测
        if len(case_events) % 3 == 0:  # 每3个事件
            return True

        # 策略2: 特定活动后触发预测
        critical_activities = self._get_critical_activities()
        if event['concept:name'] in critical_activities:
            return True

        return False

    def _get_critical_activities(self):
        """获取需要触发预测的关键活动"""
        # 在实际应用中，这应该基于领域知识配置
        if len(self.activity_list) <= 5:
            return self.activity_list
        return self.activity_list[:5]  # 返回前5个活动作为关键活动

    def should_trigger_federated_round(self):
        """判断是否应该触发联邦学习轮次"""
        self.fed_round_counter += 1
        return self.fed_round_counter >= self.fed_round_interval

    def trigger_activity_node(self, node_url, event):
        """触发活动节点处理事件"""
        try:
            response = requests.post(
                node_url,
                json=event,
                timeout=2.0
            )
            return response.ok
        except Exception as e:
            print(f"触发活动节点失败: {str(e)}")
            return False

    def trigger_federated_round(self, coordinator_url):
        """触发联邦学习轮次"""
        try:
            response = requests.post(
                coordinator_url,
                timeout=30.0
            )
            if response.ok:
                self.fed_round_counter = 0  # 重置计数器
                return response.json()
            return None
        except Exception as e:
            print(f"触发联邦学习轮次失败: {str(e)}")
            return None

    def log_prediction(self, case_id, prediction):
        """记录预测结果（新增：关联事件索引用于后续验证）"""
        # 记录预测时的事件位置（用于验证）
        event_index = len(self.case_events.get(case_id, [])) - 1
        self.prediction_records[case_id].append({
            'event_index': event_index,
            'predicted_activity': prediction.get('next_activity'),
            'node_id': prediction.get('node_id'),
            'timestamp': time.time()
        })

        # 保存到文件
        prediction_log_path = os.path.join("outputs", "predictions")
        os.makedirs(prediction_log_path, exist_ok=True)

        log_file = os.path.join(prediction_log_path, f"predictions_{case_id}.jsonl")
        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                json.dump({
                    'timestamp': time.time(),
                    'case_id': case_id,
                    'prediction': prediction,
                    'event_index': event_index
                }, f)
                f.write('\n')
            return True
        except Exception as e:
            print(f"记录预测结果失败: {str(e)}")
            return False