# Copyright 2024 Kiel University
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import requests
import time

class PredictionCoordinatorClient:
    """预测协调器客户端，用于与协调器通信"""
    def __init__(self, coord_ip):
        self.coord_url = f"http://{coord_ip}"
        self.timeout = 5.0  # 5秒超时
        
    def register_case_event(self, case_id, node_id, event):
        """注册案例事件到协调器"""
        try:
            response = requests.post(
                f"{self.coord_url}/register_case_event",
                json={
                    'case_id': case_id,
                    'node_id': node_id,
                    'event': event
                },
                timeout=self.timeout
            )
            return response.ok
        except Exception as e:
            print(f"注册案例事件失败: {str(e)}")
            return False
    
    def coordinate_prediction(self, case_id, last_event=None):
        """通过协调器获取预测结果"""
        try:
            payload = {'case_id': case_id}
            if last_event:
                payload['last_event'] = last_event
                
            response = requests.post(
                f"{self.coord_url}/coordinate_prediction",
                json=payload,
                timeout=self.timeout
            )
            
            if response.ok:
                return response.json()
            return None
        except Exception as e:
            print(f"协调预测失败: {str(e)}")
            return None
    
    def update_node_performance(self, node_id, correct):
        """更新节点性能指标"""
        try:
            response = requests.post(
                f"{self.coord_url}/update_node_performance",
                json={
                    'node_id': node_id,
                    'correct': correct
                },
                timeout=self.timeout
            )
            return response.ok
        except Exception as e:
            print(f"更新节点性能失败: {str(e)}")
            return False
    
    def check_health(self):
        """检查协调器健康状态"""
        try:
            response = requests.get(
                f"{self.coord_url}/health",
                timeout=self.timeout
            )
            return response.ok
        except:
            return False
