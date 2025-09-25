# Copyright 2024 Kiel University
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import requests
import time
import threading

class FederatedClient:
    """联邦学习客户端，运行在活动节点上"""
    def __init__(self, node_id, coordinator_url):
        self.node_id = node_id
        self.coordinator_url = coordinator_url
        self.running = True
        self.training_in_progress = False
        self.model_version = 0
        
        # 启动状态检查线程
        self.status_thread = threading.Thread(target=self.check_coordinator_status, daemon=True)
        self.status_thread.start()
    
    def check_coordinator_status(self):
        """定期检查协调器状态并响应训练请求"""
        while self.running:
            try:
                # 检查协调器是否需要本节点参与训练
                response = requests.get(
                    f"{self.coordinator_url}/get_status",
                    timeout=5.0
                )
                
                if response.ok:
                    status = response.json()
                    node_status = status.get('node_status', {})
                    
                    # 如果协调器标记本节点为训练中，则开始本地训练
                    if node_status.get(str(self.node_id)) == 'training' and not self.training_in_progress:
                        threading.Thread(target=self.perform_local_training, daemon=True).start()
            except Exception as e:
                # 协调器可能暂时不可用，稍后重试
                pass
                
            # 每10秒检查一次
            time.sleep(10)
    
    def perform_local_training(self):
        """执行本地训练并提交更新"""
        self.training_in_progress = True
        try:
            # 从本地节点获取模型参数（这里是对自身API的调用）
            # 实际实现中，这应该直接调用本地模型的方法
            from node_layer.predictive_activity_node import node
            
            # 执行本地训练
            print(f"节点 {self.node_id} 开始本地训练")
            training_result = node.local_training(epochs=3)
            
            if training_result.get('status') == 'error':
                print(f"节点 {self.node_id} 本地训练失败: {training_result.get('message')}")
                return
            
            # 获取更新后的模型参数
            model_params = node.get_local_model_parameters()
            
            # 提交更新到协调器（实际中这应该是协调器主动收集）
            # 这里仅作为演示
            print(f"节点 {self.node_id} 提交模型更新")
            
            # 更新本地模型版本
            self.model_version += 1
            
            return model_params
            
        except Exception as e:
            print(f"节点 {self.node_id} 训练过程出错: {str(e)}")
        finally:
            self.training_in_progress = False
    
    def stop(self):
        """停止联邦学习客户端"""
        self.running = False
        if self.status_thread.is_alive():
            self.status_thread.join()
