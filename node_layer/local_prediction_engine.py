# Copyright 2024 Kiel University
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import time
import json
from collections import defaultdict, Counter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class NGramModel:
    """N-gram模型，用于序列预测"""
    def __init__(self, n=3):
        self.n = n
        self.transitions = defaultdict(Counter)
        self.activities = set()
        self.activity_to_idx = {}
        self.idx_to_activity = {}
        self.trained = False
        
    def _get_ngrams(self, sequence):
        """从序列中提取n-gram"""
        ngrams = []
        for i in range(len(sequence) - self.n + 1):
            ngram = tuple(sequence[i:i+self.n])
            ngrams.append(ngram)
        return ngrams
        
    def train(self, traces, epochs=1):
        """训练模型"""
        # 收集所有活动
        for trace in traces:
            activities = [event['concept:name'] for event in trace]
            for act in activities:
                self.activities.add(act)
        
        # 创建活动映射
        self.activity_to_idx = {act: i for i, act in enumerate(self.activities)}
        self.idx_to_activity = {i: act for act, i in self.activity_to_idx.items()}
        
        # 学习转移概率
        for trace in traces:
            activities = [event['concept:name'] for event in trace]
            if len(activities) < self.n:
                continue
                
            ngrams = self._get_ngrams(activities)
            for ngram in ngrams:
                prefix = ngram[:-1]
                next_act = ngram[-1]
                self.transitions[prefix][next_act] += 1
        
        self.trained = True
        return {'loss': 0.0, 'accuracy': 0.0}  # 简化实现
        
    def predict_next_activity(self, trace):
        """预测下一个活动"""
        if not self.trained:
            return {'activity': 'unknown', 'confidence': 0.0}
            
        # 提取活动序列
        activities = [event['concept:name'] for event in trace]
        if len(activities) < self.n - 1:
            # 如果序列太短，使用所有可用活动
            prefix = tuple(activities)
        else:
            # 使用最后n-1个活动作为前缀
            prefix = tuple(activities[-(self.n-1):])
        
        # 找到最可能的下一个活动
        if prefix in self.transitions:
            transitions = self.transitions[prefix]
            total = sum(transitions.values())
            if total > 0:
                next_act = max(transitions, key=transitions.get)
                confidence = transitions[next_act] / total
                return {
                    'activity': next_act,
                    'confidence': confidence
                }
        
        # 如果没有找到匹配的前缀，返回最常见的活动
        all_acts = [act for trace in self.transitions for act in self.transitions[trace]]
        if all_acts:
            most_common = Counter(all_acts).most_common(1)[0]
            return {
                'activity': most_common[0],
                'confidence': most_common[1] / len(all_acts)
            }
            
        return {'activity': 'unknown', 'confidence': 0.0}
    
    def get_model_parameters(self):
        """获取模型参数"""
        return {
            'n': self.n,
            'transitions': {str(k): dict(v) for k, v in self.transitions.items()},
            'activities': list(self.activities),
            'activity_to_idx': self.activity_to_idx,
            'model_type': 'ngram'
        }
    
    def update_model_parameters(self, params):
        """更新模型参数"""
        if 'n' in params:
            self.n = params['n']
        if 'transitions' in params:
            self.transitions = defaultdict(Counter)
            for k, v in params['transitions'].items():
                # 将字符串键转换回元组
                key_tuple = tuple(k.strip('()').split(', '))
                if len(key_tuple) == 1 and key_tuple[0] == '':  # 处理空元组
                    key_tuple = ()
                self.transitions[key_tuple] = Counter(v)
        if 'activities' in params:
            self.activities = set(params['activities'])
        if 'activity_to_idx' in params:
            self.activity_to_idx = params['activity_to_idx']
            self.idx_to_activity = {i: act for act, i in self.activity_to_idx.items()}
        self.trained = True

class CompactGRU(nn.Module):
    """紧凑的GRU模型，适用于边缘设备"""
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=10):
        super(CompactGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(
            hidden_size, 
            hidden_size, 
            num_layers,
            batch_first=True,
            dropout=0.2
        )
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x, hidden=None):
        if hidden is None:
            hidden = self.init_hidden(x.size(0))
            
        x = self.embedding(x)
        out, hidden = self.gru(x, hidden)
        out = self.fc(out[:, -1, :])  # 取最后一个时间步
        out = self.softmax(out)
        return out, hidden
        
    def init_hidden(self, batch_size):
        """初始化隐藏状态"""
        device = next(self.parameters()).device
        return torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)

class TraceDataset(Dataset):
    """轨迹数据集，用于训练神经网络模型"""
    def __init__(self, traces, activity_to_idx, sequence_length=5):
        self.sequence_length = sequence_length
        self.activity_to_idx = activity_to_idx
        self.data = []
        self.labels = []
        
        # 处理每个轨迹
        for trace in traces:
            # 提取活动序列
            activities = [event['concept:name'] for event in trace]
            # 过滤未知活动
            activities = [a for a in activities if a in self.activity_to_idx]
            
            # 创建序列和标签
            for i in range(len(activities) - self.sequence_length):
                seq = activities[i:i+self.sequence_length]
                label = activities[i+self.sequence_length]
                
                # 转换为索引
                seq_idx = [self.activity_to_idx[a] for a in seq]
                label_idx = self.activity_to_idx[label]
                
                self.data.append(seq_idx)
                self.labels.append(label_idx)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.data[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )

class GRUModel:
    """基于GRU的预测模型"""
    def __init__(self, hidden_size=64, num_layers=2, sequence_length=5):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.model = None
        self.activity_to_idx = {}
        self.idx_to_activity = {}
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = None
        self.trained = False
        
        # 设备配置（CPU/GPU）
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def train(self, traces, epochs=10, batch_size=32, learning_rate=0.001):
        """训练模型"""
        # 收集所有活动
        activities = set()
        for trace in traces:
            for event in trace:
                activities.add(event['concept:name'])
        
        # 创建活动映射
        self.activity_to_idx = {act: i for i, act in enumerate(activities)}
        self.idx_to_activity = {i: act for act, i in self.activity_to_idx.items()}
        num_activities = len(activities)
        
        if num_activities == 0:
            raise ValueError("没有足够的活动数据用于训练")
        
        # 创建数据集和数据加载器
        dataset = TraceDataset(
            traces, 
            self.activity_to_idx,
            sequence_length=self.sequence_length
        )
        
        if len(dataset) == 0:
            raise ValueError("没有足够的序列数据用于训练")
            
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True
        )
        
        # 初始化模型
        if self.model is None:
            self.model = CompactGRU(
                input_size=num_activities,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                output_size=num_activities
            ).to(self.device)
        
        # 初始化优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # 训练循环
        self.model.train()
        history = {'loss': [], 'accuracy': []}
        
        for epoch in range(epochs):
            total_loss = 0.0
            correct_predictions = 0
            total_predictions = 0
            
            for sequences, labels in dataloader:
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)
                
                # 前向传播
                outputs, _ = self.model(sequences)
                loss = self.loss_fn(outputs, labels)
                
                # 反向传播和优化
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # 计算指标
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_predictions += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
            
            # 计算平均损失和准确率
            avg_loss = total_loss / len(dataloader)
            accuracy = correct_predictions / total_predictions
            
            history['loss'].append(avg_loss)
            history['accuracy'].append(accuracy)
            
            # 打印 epoch 信息
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')
        
        self.trained = True
        return history
    
    def predict_next_activity(self, trace):
        """预测下一个活动"""
        if not self.trained or not self.model:
            return {'activity': 'unknown', 'confidence': 0.0}
            
        # 提取活动序列
        activities = [event['concept:name'] for event in trace]
        # 过滤未知活动
        activities = [a for a in activities if a in self.activity_to_idx]
        
        if len(activities) < self.sequence_length:
            # 如果序列太短，用填充值补齐
            padding = [next(iter(self.activity_to_idx.keys()))] * (self.sequence_length - len(activities))
            input_seq = padding + activities
        else:
            # 使用最后N个活动
            input_seq = activities[-self.sequence_length:]
        
        # 转换为索引并添加批次维度
        input_idx = [self.activity_to_idx[a] for a in input_seq]
        input_tensor = torch.tensor([input_idx], dtype=torch.long).to(self.device)
        
        # 预测
        self.model.eval()
        with torch.no_grad():
            outputs, _ = self.model(input_tensor)
            probabilities = outputs[0].cpu().numpy()
            
        # 获取最可能的活动
        max_idx = np.argmax(probabilities)
        next_activity = self.idx_to_activity.get(max_idx, 'unknown')
        confidence = float(probabilities[max_idx])
        
        return {
            'activity': next_activity,
            'confidence': confidence
        }
    
    def get_model_parameters(self):
        """获取模型参数"""
        if not self.model:
            return None
            
        # 提取模型参数
        params = {
            'state_dict': {k: v.cpu().numpy().tolist() for k, v in self.model.state_dict().items()},
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'sequence_length': self.sequence_length,
            'activity_to_idx': self.activity_to_idx,
            'model_type': 'gru'
        }
        
        return params
    
    def update_model_parameters(self, params):
        """更新模型参数"""
        if 'state_dict' not in params or 'activity_to_idx' not in params:
            raise ValueError("缺少必要的模型参数")
            
        # 更新活动映射
        self.activity_to_idx = params['activity_to_idx']
        self.idx_to_activity = {i: act for act, i in self.activity_to_idx.items()}
        num_activities = len(self.activity_to_idx)
        
        # 更新模型配置
        if 'hidden_size' in params:
            self.hidden_size = params['hidden_size']
        if 'num_layers' in params:
            self.num_layers = params['num_layers']
        if 'sequence_length' in params:
            self.sequence_length = params['sequence_length']
        
        # 初始化或更新模型
        if not self.model:
            self.model = CompactGRU(
                input_size=num_activities,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                output_size=num_activities
            ).to(self.device)
        
        # 加载状态字典
        state_dict = {k: torch.tensor(v).to(self.device) for k, v in params['state_dict'].items()}
        self.model.load_state_dict(state_dict)
        
        self.trained = True

class LightweightTransformer(nn.Module):
    """轻量级Transformer模型，用于序列预测"""
    def __init__(self, input_size, d_model=128, nhead=4, num_layers=2, output_size=10):
        super(LightweightTransformer, self).__init__()
        self.d_model = d_model
        
        self.embedding = nn.Embedding(input_size, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 100, d_model))  # 位置编码
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        self.fc = nn.Linear(d_model, output_size)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # 嵌入和位置编码
        x = self.embedding(x) * np.sqrt(self.d_model)
        if x.size(1) <= 100:  # 限制最大长度
            x += self.pos_encoder[:, :x.size(1), :]
        else:
            x += self.pos_encoder
        
        # Transformer编码
        out = self.transformer_encoder(x)
        
        # 取最后一个时间步
        out = self.fc(out[:, -1, :])
        out = self.softmax(out)
        return out

class TransformerModel:
    """基于Transformer的预测模型"""
    def __init__(self, d_model=128, nhead=4, num_layers=2, sequence_length=5):
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.model = None
        self.activity_to_idx = {}
        self.idx_to_activity = {}
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = None
        self.trained = False
        
        # 设备配置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def train(self, traces, epochs=10, batch_size=32, learning_rate=0.001):
        """训练模型"""
        # 收集所有活动
        activities = set()
        for trace in traces:
            for event in trace:
                activities.add(event['concept:name'])
        
        # 创建活动映射
        self.activity_to_idx = {act: i for i, act in enumerate(activities)}
        self.idx_to_activity = {i: act for act, i in self.activity_to_idx.items()}
        num_activities = len(activities)
        
        if num_activities == 0:
            raise ValueError("没有足够的活动数据用于训练")
        
        # 创建数据集和数据加载器
        dataset = TraceDataset(
            traces, 
            self.activity_to_idx,
            sequence_length=self.sequence_length
        )
        
        if len(dataset) == 0:
            raise ValueError("没有足够的序列数据用于训练")
            
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True
        )
        
        # 初始化模型
        if self.model is None:
            self.model = LightweightTransformer(
                input_size=num_activities,
                d_model=self.d_model,
                nhead=self.nhead,
                num_layers=self.num_layers,
                output_size=num_activities
            ).to(self.device)
        
        # 初始化优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # 训练循环
        self.model.train()
        history = {'loss': [], 'accuracy': []}
        
        for epoch in range(epochs):
            total_loss = 0.0
            correct_predictions = 0
            total_predictions = 0
            
            for sequences, labels in dataloader:
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)
                
                # 前向传播
                outputs = self.model(sequences)
                loss = self.loss_fn(outputs, labels)
                
                # 反向传播和优化
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # 计算指标
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_predictions += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
            
            # 计算平均损失和准确率
            avg_loss = total_loss / len(dataloader)
            accuracy = correct_predictions / total_predictions
            
            history['loss'].append(avg_loss)
            history['accuracy'].append(accuracy)
        
        self.trained = True
        return history
    
    def predict_next_activity(self, trace):
        """预测下一个活动"""
        if not self.trained or not self.model:
            return {'activity': 'unknown', 'confidence': 0.0}
            
        # 提取活动序列
        activities = [event['concept:name'] for event in trace]
        # 过滤未知活动
        activities = [a for a in activities if a in self.activity_to_idx]
        
        if len(activities) < self.sequence_length:
            # 如果序列太短，用填充值补齐
            padding = [next(iter(self.activity_to_idx.keys()))] * (self.sequence_length - len(activities))
            input_seq = padding + activities
        else:
            # 使用最后N个活动
            input_seq = activities[-self.sequence_length:]
        
        # 转换为索引并添加批次维度
        input_idx = [self.activity_to_idx[a] for a in input_seq]
        input_tensor = torch.tensor([input_idx], dtype=torch.long).to(self.device)
        
        # 预测
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = outputs[0].cpu().numpy()
        
        # 获取最可能的活动
        max_idx = np.argmax(probabilities)
        next_activity = self.idx_to_activity.get(max_idx, 'unknown')
        confidence = float(probabilities[max_idx])
        
        return {
            'activity': next_activity,
            'confidence': confidence
        }
    
    def get_model_parameters(self):
        """获取模型参数"""
        if not self.model:
            return None
            
        # 提取模型参数
        params = {
            'state_dict': {k: v.cpu().numpy().tolist() for k, v in self.model.state_dict().items()},
            'd_model': self.d_model,
            'nhead': self.nhead,
            'num_layers': self.num_layers,
            'sequence_length': self.sequence_length,
            'activity_to_idx': self.activity_to_idx,
            'model_type': 'transformer'
        }
        
        return params
    
    def update_model_parameters(self, params):
        """更新模型参数"""
        if 'state_dict' not in params or 'activity_to_idx' not in params:
            raise ValueError("缺少必要的模型参数")
            
        # 更新活动映射
        self.activity_to_idx = params['activity_to_idx']
        self.idx_to_activity = {i: act for act, i in self.activity_to_idx.items()}
        num_activities = len(self.activity_to_idx)
        
        # 更新模型配置
        if 'd_model' in params:
            self.d_model = params['d_model']
        if 'nhead' in params:
            self.nhead = params['nhead']
        if 'num_layers' in params:
            self.num_layers = params['num_layers']
        if 'sequence_length' in params:
            self.sequence_length = params['sequence_length']
        
        # 初始化或更新模型
        if not self.model:
            self.model = LightweightTransformer(
                input_size=num_activities,
                d_model=self.d_model,
                nhead=self.nhead,
                num_layers=self.num_layers,
                output_size=num_activities
            ).to(self.device)
        
        # 加载状态字典
        state_dict = {k: torch.tensor(v).to(self.device) for k, v in params['state_dict'].items()}
        self.model.load_state_dict(state_dict)
        
        self.trained = True

class LocalPredictionEngine:
    """本地预测引擎，集成多种预测模型"""
    def __init__(self, default_model='gru'):
        self.models = {
            'ngram': NGramModel(n=3),
            'gru': GRUModel(hidden_size=64, num_layers=2),
            'transformer': TransformerModel(d_model=128, nhead=4)
        }
        self.active_model = default_model
        self.training_history = {}
        
    def set_active_model(self, model_type):
        """设置活跃模型"""
        if model_type in self.models:
            self.active_model = model_type
            return True
        return False
    
    def train(self, traces, epochs=5):
        """训练活跃模型"""
        if self.active_model not in self.models:
            return {'status': 'error', 'message': '未知模型类型'}
            
        model = self.models[self.active_model]
        try:
            history = model.train(traces, epochs=epochs)
            self.training_history[self.active_model] = history
            return history
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def predict_next_activity(self, trace):
        """预测下一个活动"""
        if self.active_model not in self.models:
            return {'activity': 'unknown', 'confidence': 0.0}
            
        return self.models[self.active_model].predict_next_activity(trace)
    
    def get_model_parameters(self):
        """获取当前模型参数"""
        if self.active_model not in self.models:
            return None
        return self.models[self.active_model].get_model_parameters()
    
    def update_model_parameters(self, params):
        """更新模型参数"""
        if not params or 'model_type' not in params:
            raise ValueError("模型参数不完整")
            
        model_type = params['model_type']
        if model_type not in self.models:
            raise ValueError(f"未知模型类型: {model_type}")
            
        # 更新指定模型
        self.models[model_type].update_model_parameters(params)
        
        # 如果更新的是当前活跃模型，切换到它
        self.active_model = model_type
