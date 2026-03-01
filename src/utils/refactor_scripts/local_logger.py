"""
Local Training Logger (替代WandB).

Recent Updates:
    - [2025-12-10] Updated: topk_metrics.png now shows Lift + Top-1 Confidence + Hit Accuracy
    - [2025-12-10] Removed: topk_consistency (deprecated metric)
    - [2025-01-29] Initial: 本地训练日志记录器，替代WandB
    - [2025-01-29] Feature: JSON记录、可视化图表、文件组织

Key Features:
    - JSON逐epoch记录训练指标
    - 自动生成训练曲线可视化
    - 文件组织管理（results/{experiment_name}_{timestamp}/）
    - 类WandB API（便于迁移）
    - 无需网络连接，无需登录

Usage:
    >>> from rice_detection.utils.local_logger import LocalLogger
    >>>
    >>> # 初始化
    >>> logger = LocalLogger(
    ...     experiment_name="topk_asymmetric_mil",
    ...     config=config_dict
    ... )
    >>>
    >>> # 记录指标
    >>> logger.log({
    ...     'epoch': 1,
    ...     'train_loss': 0.5,
    ...     'val_loss': 0.6,
    ...     'neg_recall': 0.7
    ... })
    >>>
    >>> # 生成可视化（训练结束时）
    >>> logger.plot_training_curves()
    >>>
    >>> # 关闭
    >>> logger.finish()
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import shutil

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def _to_serializable(obj: Any):
    """将 ndarray / tensor / numpy types 等转换成 JSON 可序列化类型"""
    # Handle numpy scalar types (float32, int64, etc.)
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    # Handle numpy arrays
    if isinstance(obj, np.ndarray):
        if obj.ndim == 0:
            return obj.item()
        return obj.tolist()
    # Handle sets
    if isinstance(obj, (set,)):
        return list(obj)
    # Handle dicts recursively
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    # Handle lists recursively
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(item) for item in obj]
    return obj

class LocalLogger:
    """
    本地训练日志记录器（Local Training Logger）.

    提供类似WandB的API，但所有数据保存在本地，无需网络和登录。

    File Structure:
        results/{experiment_name}_{timestamp}/
            ├── config.yaml              # 训练配置副本
            ├── training_log.json        # 逐epoch训练日志
            ├── metrics_summary.json     # 最终指标汇总
            ├── training_curves/         # 训练曲线图
            ├── heatmaps/               # Heatmap可视化
            ├── evaluation/             # 最终评估结果
            └── checkpoints/            # 模型checkpoint

    Args:
        experiment_name: 实验名称
        config: 训练配置字典
        root_dir: 结果保存根目录（默认: results/）
    """

    def __init__(
        self,
        experiment_name: str,
        config: Dict,
        root_dir: str = "results"
    ):
        self.experiment_name = experiment_name
        self.config = config
        self.root_dir = Path(root_dir)

        # 创建时间戳标识的实验目录
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_name = f"{experiment_name}_{timestamp}"
        self.run_dir = self.root_dir / self.run_name

        # 创建子目录
        self.training_curves_dir = self.run_dir / "training_curves"
        self.heatmaps_dir = self.run_dir / "heatmaps"
        self.evaluation_dir = self.run_dir / "evaluation"
        self.checkpoints_dir = self.run_dir / "checkpoints"

        self._create_directories()

        # 保存配置副本
        self._save_config()

        # 初始化JSON记录
        self.log_path = self.run_dir / "training_log.json"

        # 内存中保存所有指标（用于最后生成图表和保存JSON）
        self.metrics_history = []

        print(f"[LOCAL-LOGGER] Initialized: {self.run_name}")
        print(f"[LOCAL-LOGGER] Results directory: {self.run_dir}")

    def _create_directories(self):
        """创建所有必要的子目录."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.training_curves_dir.mkdir(exist_ok=True)
        self.heatmaps_dir.mkdir(exist_ok=True)
        self.evaluation_dir.mkdir(exist_ok=True)
        self.checkpoints_dir.mkdir(exist_ok=True)

    def _save_config(self):
        """保存配置文件副本."""
        import yaml
        config_path = self.run_dir / "config.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
        print(f"[LOCAL-LOGGER] Config saved to: {config_path}")

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """
        记录指标（类似wandb.log）.

        Args:
            metrics: 指标字典（key: metric_name, value: metric_value）
            step: 当前step/epoch（可选，如果metrics中有'epoch'则自动使用）

        Example:
            >>> logger.log({'train_loss': 0.5, 'val_loss': 0.6, 'epoch': 1})
        """
        # 如果metrics中有step/epoch，优先使用
        if step is None:
            step = metrics.get('step', metrics.get('epoch', len(self.metrics_history)))

        # 确保step在metrics中
        if 'step' not in metrics and 'epoch' not in metrics:
            metrics['epoch'] = step

        # 添加时间戳
        metrics['timestamp'] = datetime.now().isoformat()

        # 保存到内存历史
        self.metrics_history.append(metrics)

        # 实时写入JSON（每次log都保存）
        self._save_training_log()

    def _save_training_log(self):
        """保存训练日志到JSON文件."""
        # Convert all numpy types to JSON-serializable Python types
        serializable_metrics = _to_serializable(self.metrics_history)
        with open(self.log_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_metrics, f, indent=2, ensure_ascii=False)


    def plot_training_curves(self, save: bool = True):
        """
        生成训练曲线可视化.

        生成的图表：
            1. loss_curves.png: Total Loss + Positive Loss + Negative Loss
            2. negative_metrics.png: Negative Recall + Negative Confidence
            3. topk_metrics.png: Top-K Lift + Top-1 Confidence + Hit Accuracy
            4. train_val_gap.png: Train Loss vs Val Loss

        Args:
            save: 是否保存图片（默认True）
        """
        if len(self.metrics_history) == 0:
            print("[LOCAL-LOGGER] No metrics to plot")
            return

        # 提取数据
        epochs = [m.get('epoch', i) for i, m in enumerate(self.metrics_history)]

        # ========== 1. Loss曲线 ==========
        self._plot_loss_curves(epochs, save=save)

        # ========== 2. Negative指标曲线 ==========
        self._plot_negative_metrics(epochs, save=save)

        # ========== 3. Top-K指标曲线 ==========
        self._plot_topk_metrics(epochs, save=save)

        # ========== 4. Train-Val Gap ==========
        self._plot_train_val_gap(epochs, save=save)

        print(f"[LOCAL-LOGGER] Training curves saved to: {self.training_curves_dir}")

    def _plot_loss_curves(self, epochs: List, save: bool = True):
        """绘制Loss曲线（Total + Positive + Negative）."""
        fig, ax = plt.subplots(figsize=(12, 6))

        # 提取loss数据
        train_loss = [m.get('train_loss', np.nan) for m in self.metrics_history]
        val_loss = [m.get('val_loss', np.nan) for m in self.metrics_history]

        # 绘制基本loss
        ax.plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)
        ax.plot(epochs, val_loss, 'r-', label='Val Loss', linewidth=2)

        # 如果有pos_loss和neg_loss，也绘制
        if any('pos_loss' in m for m in self.metrics_history):
            pos_loss = [m.get('pos_loss', np.nan) for m in self.metrics_history]
            neg_loss = [m.get('neg_loss', np.nan) for m in self.metrics_history]
            ax.plot(epochs, pos_loss, 'g--', label='Positive Loss', linewidth=1.5, alpha=0.7)
            ax.plot(epochs, neg_loss, 'orange', linestyle='--', label='Negative Loss', linewidth=1.5, alpha=0.7)

        # 标注warm-up阶段
        warmup_epochs = self.config.get('asymmetric_mil', {}).get('warmup_epochs', 5)
        if max(epochs) > warmup_epochs:
            ax.axvline(x=warmup_epochs, color='gray', linestyle=':', linewidth=2, label='Warm-up End')
            ax.axvspan(0, warmup_epochs, alpha=0.1, color='blue', label='Warm-up Phase')

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training Loss Curves', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        if save:
            save_path = self.training_curves_dir / "loss_curves.png"
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def _plot_negative_metrics(self, epochs: List, save: bool = True):
        """绘制Negative指标曲线（Recall + Confidence）."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Negative Recall
        neg_recall = [m.get('neg_recall', np.nan) for m in self.metrics_history]
        ax1.plot(epochs, neg_recall, 'g-', linewidth=2, marker='o', markersize=4)
        ax1.axhline(y=0.7, color='r', linestyle='--', label='Target: 0.7', linewidth=1.5)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Negative Recall', fontsize=12)
        ax1.set_title('Background Recognition Ability\n(Negative Recall)', fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1.0])

        # Negative Confidence
        neg_conf = [m.get('neg_confidence', np.nan) for m in self.metrics_history]
        ax2.plot(epochs, neg_conf, 'orange', linewidth=2, marker='o', markersize=4)
        ax2.axhline(y=0.3, color='r', linestyle='--', label='Target: < 0.3', linewidth=1.5)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Negative Confidence', fontsize=12)
        ax2.set_title('Background Suppression\n(Max Disease Confidence on Negatives)', fontsize=13, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1.0])

        if save:
            save_path = self.training_curves_dir / "negative_metrics.png"
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def _plot_topk_metrics(self, epochs: List, save: bool = True):
        """绘制Top-K指标曲线（Lift + Top-1 Confidence + Hit Accuracy）."""
        # 检查是否有Top-K数据
        if not any('topk_lift' in m for m in self.metrics_history):
            return

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

        # Left: Top-K Lift (保持不变)
        topk_lift = [m.get('topk_lift', np.nan) for m in self.metrics_history]
        ax1.plot(epochs, topk_lift, 'b-', linewidth=2, marker='s', markersize=4)
        ax1.axhline(y=0.2, color='r', linestyle='--', label='Target: > 0.2', linewidth=1.5)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Top-K Lift', fontsize=12)
        ax1.set_title('Top-K Selection Quality\n(Lift over Random)', fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Middle: Avg Top-1 Confidence (RENAMED: from top1_confidence)
        avg_top1_conf = [m.get('avg_top1_conf', np.nan) for m in self.metrics_history]
        ax2.plot(epochs, avg_top1_conf, 'green', linewidth=2, marker='o', markersize=4)
        ax2.axhline(y=0.5, color='r', linestyle='--', label='Reference: 0.5', linewidth=1.5)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Avg Top-1 Conf', fontsize=12)  # RENAMED: from Top-1 Confidence
        ax2.set_title('Top-1 Tile Confidence\n(Aligned with CE Loss)', fontsize=13, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1.0])

        # Right: Hit Accuracy (NEW: corrected calculation)
        hit_acc = [m.get('hit_acc', np.nan) for m in self.metrics_history]
        ax3.plot(epochs, hit_acc, 'purple', linewidth=2, marker='D', markersize=4)
        ax3.axhline(y=0.5, color='orange', linestyle='--', label='Reference: 0.5', linewidth=1.5)
        ax3.set_xlabel('Epoch', fontsize=12)
        ax3.set_ylabel('Hit Accuracy', fontsize=12)
        ax3.set_title('Top-1 Classification Accuracy\n(Includes ALL Tiles)', fontsize=13, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim([0, 1.0])

        if save:
            save_path = self.training_curves_dir / "topk_metrics.png"
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def _plot_train_val_gap(self, epochs: List, save: bool = True):
        """绘制Train-Val Gap（过拟合检测）."""
        fig, ax = plt.subplots(figsize=(10, 6))

        train_acc = [m.get('train_acc', np.nan) for m in self.metrics_history]
        val_acc = [m.get('val_acc', np.nan) for m in self.metrics_history]

        ax.plot(epochs, train_acc, 'b-', label='Train Acc', linewidth=2)
        ax.plot(epochs, val_acc, 'r-', label='Val Acc', linewidth=2)

        # 计算gap
        gap = [t - v for t, v in zip(train_acc, val_acc)]
        ax2 = ax.twinx()
        ax2.plot(epochs, gap, 'g--', label='Train-Val Gap', linewidth=2, alpha=0.7)
        ax2.axhline(y=0.1, color='orange', linestyle=':', label='Caution: 10%', linewidth=1.5)
        ax2.axhline(y=0.2, color='red', linestyle=':', label='Warning: 20%', linewidth=1.5)
        ax2.set_ylabel('Accuracy Gap', fontsize=12)

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Overfitting Detection (Train-Val Gap)', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        if save:
            save_path = self.training_curves_dir / "train_val_gap.png"
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


    def save_metrics_summary(self, final_metrics: Dict[str, Any]):
        """
        保存最终指标汇总.

        Args:
            final_metrics: 最终评估指标字典
        """
        summary_path = self.run_dir / "metrics_summary.json"

        # Use the global _to_serializable function
        safe_metrics = _to_serializable(final_metrics)

        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(safe_metrics, f, indent=2, ensure_ascii=False)

        print(f"[LOCAL-LOGGER] Metrics summary saved to: {summary_path}")

    def finish(self):
        """关闭logger（类似wandb.finish）."""
        # 保存最终训练日志
        self._save_training_log()
        print(f"[LOCAL-LOGGER] Training log saved to: {self.log_path}")

        # 生成训练曲线
        print("[LOCAL-LOGGER] Generating training curves...")
        self.plot_training_curves(save=True)

        print(f"[LOCAL-LOGGER] Training session finished")
        print(f"[LOCAL-LOGGER] All results saved to: {self.run_dir}")

    def get_run_dir(self) -> Path:
        """获取当前运行的结果目录."""
        return self.run_dir

    def get_checkpoints_dir(self) -> Path:
        """获取checkpoint保存目录."""
        return self.checkpoints_dir

    def get_heatmaps_dir(self) -> Path:
        """获取heatmap保存目录."""
        return self.heatmaps_dir

    def get_evaluation_dir(self) -> Path:
        """获取evaluation保存目录."""
        return self.evaluation_dir


# ========== 便捷函数 ==========

def init_logger(experiment_name: str, config: Dict, root_dir: str = "results") -> LocalLogger:
    """
    初始化logger（类似wandb.init）.

    Args:
        experiment_name: 实验名称
        config: 配置字典
        root_dir: 结果根目录

    Returns:
        LocalLogger实例
    """
    return LocalLogger(experiment_name, config, root_dir)


__all__ = ['LocalLogger', 'init_logger']
