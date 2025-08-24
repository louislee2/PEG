# peg_pipeline/prediction.py - 扩展预测模块以计算Accuracy和Top-k Accuracy

import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from collections import defaultdict


class ProcessDataset(Dataset):
    """数据集类，用于训练下一活动预测模型"""

    def __init__(self, prefixes, activity_to_id, pad_id=0, max_len=100):
        self.prefixes = prefixes
        self.activity_to_id = activity_to_id
        self.pad_id = pad_id
        self.max_len = max_len

        # 准备训练样本：每个前缀的输入是前n-1个活动，目标是第n个活动
        self.samples = []
        for prefix in prefixes:
            if len(prefix) >= 2:  # 至少需要2个活动
                for i in range(1, len(prefix)):
                    input_seq = prefix[:i]
                    target = prefix[i]
                    self.samples.append((input_seq, target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_seq, target = self.samples[idx]

        # 填充到固定长度
        padded_input = [self.pad_id] * self.max_len
        seq_len = min(len(input_seq), self.max_len)
        padded_input[:seq_len] = input_seq[:seq_len]

        return {
            'input_ids': torch.tensor(padded_input, dtype=torch.long),
            'target': torch.tensor(target, dtype=torch.long),
            'length': torch.tensor(seq_len, dtype=torch.long)
        }


def train_prediction_model(prefixes, activity_to_id, id_to_activity, cfg, out_dir=None):
    """训练下一活动预测模型"""
    print("开始训练预测模型...")

    # 创建数据集
    dataset = ProcessDataset(prefixes, activity_to_id,
                             pad_id=cfg.get("pad_id", 0),
                             max_len=cfg.get("max_seq_len", 100))

    print(f"训练样本数: {len(dataset)}")

    if len(dataset) == 0:
        print("错误: 没有有效的训练样本")
        from .model_utils import build_user_model, ModelWrapper
        user_model, pad_id = build_user_model(activity_to_id, cfg)
        return ModelWrapper(user_model, activity_to_id, id_to_activity, pad_id)

    # 分割训练/验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    if val_size == 0:
        val_size = 1
        train_size -= 1

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # 数据加载器
    batch_size = min(cfg.get("batch_size", 32), len(train_dataset))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 创建模型
    from .model_utils import build_user_model, ModelWrapper
    model, pad_id = build_user_model(activity_to_id, cfg)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and cfg.get("device", "cpu").lower() in ["cuda", "gpu"] else "cpu")
    model.to(device)
    print(f"使用设备: {device}")

    # 优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=cfg.get("learning_rate", 0.001),
                                 weight_decay=cfg.get("weight_decay", 0.01))
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg.get("scheduler_step_size", 5),
        gamma=cfg.get("scheduler_gamma", 0.5)
    )

    # 训练循环
    model.train()
    num_epochs = cfg.get("num_epochs", 10)
    best_val_acc = 0.0
    patience_counter = 0
    patience = cfg.get("early_stopping_patience", 3)

    training_history = {
        "train_loss": [],
        "train_acc": [],
        "val_acc": [],
        "learning_rate": []
    }

    for epoch in range(num_epochs):
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0

        # 训练循环
        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            input_ids = batch['input_ids'].to(device)
            targets = batch['target'].to(device)
            lengths = batch['length'].to(device)

            optimizer.zero_grad()

            # 前向传播
            attention_mask = (input_ids != pad_id).long()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs['logits']

            # 获取最后一个有效位置的logits
            batch_size = logits.size(0)
            last_logits = []
            for i in range(batch_size):
                last_pos = max(0, lengths[i] - 1)
                last_logits.append(logits[i, last_pos, :])
            last_logits = torch.stack(last_logits)

            # 计算损失
            loss = criterion(last_logits, targets)

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
            optimizer.step()

            # 统计
            total_loss += loss.item()
            predictions = torch.argmax(last_logits, dim=-1)
            correct_predictions += (predictions == targets).sum().item()
            total_predictions += targets.size(0)

        # 验证
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                targets = batch['target'].to(device)
                lengths = batch['length'].to(device)
                attention_mask = (input_ids != pad_id).long()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs['logits']
                batch_size = logits.size(0)
                last_logits = []
                for i in range(batch_size):
                    last_pos = max(0, lengths[i] - 1)
                    last_logits.append(logits[i, last_pos, :])
                last_logits = torch.stack(last_logits)
                predictions = torch.argmax(last_logits, dim=-1)
                val_correct += (predictions == targets).sum().item()
                val_total += targets.size(0)

        train_acc = correct_predictions / total_predictions
        val_acc = val_correct / val_total
        training_history["train_loss"].append(total_loss / len(train_loader))
        training_history["train_acc"].append(train_acc)
        training_history["val_acc"].append(val_acc)
        training_history["learning_rate"].append(optimizer.param_groups[0]['lr'])

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {total_loss / len(train_loader):.4f}, "
              f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        # 早停和最佳模型保存
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            if out_dir and cfg.get("save_best_model", True):
                os.makedirs(out_dir, exist_ok=True)
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'config': cfg,
                    'val_acc': val_acc
                }, os.path.join(out_dir, "best_model.pth"))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"早停在Epoch {epoch + 1}")
                break

        scheduler.step()

    print(f"训练完成，最佳验证准确率: {best_val_acc:.4f}")
    model.load_state_dict(torch.load(os.path.join(out_dir, "best_model.pth"))['model_state_dict'])
    return ModelWrapper(model, activity_to_id, id_to_activity, pad_id, device)


def predict_next_activities(model_wrapper, test_prefixes, true_next_activities, top_k=5):
    """预测下一个活动并返回准确率统计"""
    predictions = []

    print(f"开始预测 {len(test_prefixes)} 个前缀的下一活动...")

    for i, (prefix, true_next) in enumerate(tqdm(zip(test_prefixes, true_next_activities), desc="预测中")):
        try:
            # 获取预测概率
            probs = model_wrapper.predict_proba([prefix])[0]

            # 获取Top-k预测
            top_indices = np.argsort(probs)[-top_k:][::-1]
            top_probs = probs[top_indices]

            # 转换为活动名称
            top_activities = []
            for idx, prob in zip(top_indices, top_probs):
                activity = model_wrapper.id_to_activity.get(idx, f"Unknown_{idx}")
                if activity and prob > 0.001:  # 过滤掉概率极低的预测
                    top_activities.append((activity, float(prob)))

            # 确保至少有一个预测
            if not top_activities and len(top_indices) > 0:
                idx = top_indices[0]
                activity = model_wrapper.id_to_activity.get(idx, f"Unknown_{idx}")
                prob = float(top_probs[0])
                top_activities.append((activity, prob))

            predictions.append({
                'prefix_id': i,
                'prefix': [model_wrapper.id_to_activity.get(aid, str(aid)) for aid in prefix],
                'predictions': top_activities[:top_k],  # 限制数量
                'true_next': true_next,  # 真实下一活动
                'confidence': float(top_activities[0][1]) if top_activities else 0.0
            })

        except Exception as e:
            print(f"预测第 {i} 个前缀时出错: {e}")
            predictions.append({
                'prefix_id': i,
                'prefix': [model_wrapper.id_to_activity.get(aid, str(aid)) for aid in prefix],
                'predictions': [],
                'true_next': true_next,
                'confidence': 0.0,
                'error': str(e)
            })

    return predictions


def evaluate_with_accuracy(predictions):
    """计算Accuracy和Top-k Accuracy"""
    exact_matches = 0
    top_k_matches = {1: 0, 3: 0, 5: 0}  # 支持Top-1, Top-3, Top-5
    total = len(predictions)

    for pred in predictions:
        if not pred.get('predictions') or not pred.get('true_next'):
            continue
        pred_acts = [p[0] for p in pred['predictions']]
        true_act = pred['true_next']

        # Exact Match (Accuracy)
        if pred_acts and pred_acts[0] == true_act:
            exact_matches += 1

        # Top-k Accuracy
        for k in top_k_matches:
            if true_act in pred_acts[:k]:
                top_k_matches[k] += 1

    accuracy = exact_matches / total if total > 0 else 0.0
    top_k_accuracy = {k: v / total for k, v in top_k_matches.items()} if total > 0 else {k: 0.0 for k in top_k_matches}

    return {
        'total_predictions': total,
        'accuracy': accuracy,
        'top_k_accuracy': top_k_accuracy
    }


def load_trained_model(model_path, activity_to_id, id_to_activity, device=None):
    """加载训练好的模型"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(model_path, map_location=device)
    cfg = checkpoint.get('config', {})

    from .model_utils import build_user_model, ModelWrapper
    model, pad_id = build_user_model(activity_to_id, cfg, device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"成功加载训练模型，验证准确率: {checkpoint.get('val_acc', 'Unknown'):.4f}")

    return ModelWrapper(model, activity_to_id, id_to_activity, pad_id, device)


def simple_predict(model_wrapper, prefix_activities, top_k=3):
    """简单的预测接口，接受活动名称列表"""
    prefix_ids = [model_wrapper.activity_to_id.get(a, 0) for a in prefix_activities]

    if not prefix_ids or all(pid == 0 for pid in prefix_ids):
        return []

    try:
        probs = model_wrapper.predict_proba([prefix_ids])[0]
        top_indices = np.argsort(probs)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            activity = model_wrapper.id_to_activity.get(idx, f"Unknown_{idx}")
            prob = float(probs[idx])
            if prob > 0.001:
                results.append((activity, prob))

        return results

    except Exception as e:
        print(f"预测失败: {e}")
        return []