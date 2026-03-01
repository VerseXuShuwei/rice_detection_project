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
    >>> from src.utils.local_logger import LocalLogger
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
        outputs/logs/{experiment_name}_{timestamp}/
            ├── config.yaml              # 训练配置副本
            ├── training_log.json        # 逐epoch训练日志
            ├── metrics_summary.json     # 最终指标汇总
            ├── training_curves/         # 训练曲线图
            ├── heatmaps/               # Heatmap可视化
            └── evaluation/             # 最终评估结果

        outputs/checkpoints/{experiment_name}_{timestamp}/
            ├── epoch_001.pth            # Epoch checkpoint
            ├── epoch_002.pth
            ├── best_model.pth           # Best model checkpoint
            └── hard_mining_state_epoch_001.pkl  # Hard mining state

    Args:
        experiment_name: 实验名称
        config: 训练配置字典
        logs_root: Logs保存根目录（默认: outputs/logs）
        checkpoints_root: Checkpoints保存根目录（默认: outputs/checkpoints）
    """

    def __init__(
        self,
        experiment_name: str,
        config: Dict,
        logs_root: str = "outputs/logs",
        checkpoints_root: str = "outputs/checkpoints"
    ):
        self.experiment_name = experiment_name
        self.config = config
        self.logs_root = Path(logs_root)
        self.checkpoints_root = Path(checkpoints_root)

        # 创建时间戳标识的实验目录
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_name = f"{experiment_name}_{timestamp}"

        # Logs directory (metrics, curves, heatmaps, evaluation)
        self.run_dir = self.logs_root / self.run_name

        # Checkpoints directory (separate from logs)
        self.checkpoints_dir = self.checkpoints_root / self.run_name

        # Create subdirectories under logs
        self.training_curves_dir = self.run_dir / "training_curves"
        self.heatmaps_dir = self.run_dir / "heatmaps"
        self.evaluation_dir = self.run_dir / "evaluation"

        self._create_directories()

        # 保存配置副本
        self._save_config()

        # 初始化JSON记录
        self.log_path = self.run_dir / "training_log.json"

        # 内存中保存所有指标（用于最后生成图表和保存JSON）
        self.metrics_history = []

        print(f"[LOCAL-LOGGER] Initialized: {self.run_name}")
        print(f"[LOCAL-LOGGER] Logs directory: {self.run_dir}")
        print(f"[LOCAL-LOGGER] Checkpoints directory: {self.checkpoints_dir}")

    def _create_directories(self):
        """创建所有必要的子目录."""
        # Create logs subdirectories
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.training_curves_dir.mkdir(exist_ok=True)
        self.heatmaps_dir.mkdir(exist_ok=True)
        self.evaluation_dir.mkdir(exist_ok=True)

        # Create checkpoints directory (separate tree)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

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


    def _merge_metrics_by_epoch(self) -> List[Dict]:
        """
        Merge metrics records by epoch (handle duplicate epoch entries).

        When multiple log() calls happen per epoch (e.g., basic metrics + topk_metrics),
        merge them into a single record per epoch to prevent line breaks in plots.

        Returns:
            List of merged metrics, one per epoch
        """
        from collections import defaultdict

        # Group by epoch
        epoch_metrics = defaultdict(dict)
        for m in self.metrics_history:
            epoch = m.get('epoch', None)
            if epoch is None:
                continue
            # Merge: later values override earlier ones (except timestamp)
            for k, v in m.items():
                if k != 'timestamp' or k not in epoch_metrics[epoch]:
                    epoch_metrics[epoch][k] = v

        # Sort by epoch and return as list
        sorted_epochs = sorted(epoch_metrics.keys())
        return [epoch_metrics[e] for e in sorted_epochs]

    def plot_training_curves(self, save: bool = True):
        """
        Generate comprehensive training curve visualizations.

        Plots generated (7 total):
            1. loss_curves.png: Train/Val/Aligned + Pos/Neg/Inter Loss
            2. accuracy_overview.png: Train/Val/Pos/Neg ACC + FG Ratio
            3. negative_metrics.png: Negative Recall + Confidence
            4. topk_metrics.png: Lift + Top-1 Conf + Hit ACC
            5. pos_neg_diagnostics.png: P/N Ratio + Tier Distribution
            6. confidence_analysis.png: Target vs Class0 Confidence
            7. lr_and_weight.png: Learning Rate + Dynamic Weight

        Args:
            save: Whether to save PNG files (default True)
        """
        if len(self.metrics_history) == 0:
            print("[LOCAL-LOGGER] No metrics to plot")
            return

        merged_metrics = self._merge_metrics_by_epoch()
        if len(merged_metrics) == 0:
            print("[LOCAL-LOGGER] No valid epoch metrics to plot")
            return

        epochs = [m.get('epoch', i) for i, m in enumerate(merged_metrics)]
        warmup_epochs = self.config.get('asymmetric_mil', {}).get('warmup_epochs', 6)

        self._plot_loss_curves(epochs, merged_metrics, warmup_epochs, save)
        self._plot_accuracy_overview(epochs, merged_metrics, warmup_epochs, save)
        self._plot_negative_metrics(epochs, merged_metrics, save)
        self._plot_topk_metrics(epochs, merged_metrics, save)
        self._plot_pos_neg_diagnostics(epochs, merged_metrics, warmup_epochs, save)
        self._plot_confidence_analysis(epochs, merged_metrics, warmup_epochs, save)
        self._plot_lr_and_weight(epochs, merged_metrics, warmup_epochs, save)

        print(f"[LOCAL-LOGGER] Training curves saved to: {self.training_curves_dir}")

    # ========== Helper ==========

    @staticmethod
    def _extract(metrics: List[Dict], key: str) -> List[float]:
        """Extract a metric series, filling missing values with NaN."""
        return [m.get(key, np.nan) for m in metrics]

    @staticmethod
    def _has_any(metrics: List[Dict], key: str) -> bool:
        """Check if any epoch contains the given key."""
        return any(key in m for m in metrics)

    @staticmethod
    def _add_warmup_shading(ax, warmup_epochs: int, max_epoch: int):
        """Add warmup phase shading and vertical line to an axis."""
        if max_epoch > warmup_epochs:
            ax.axvline(x=warmup_epochs, color='gray', linestyle=':', linewidth=1.5, alpha=0.6)
            ax.axvspan(0, warmup_epochs, alpha=0.07, color='blue')

    def _save_or_show(self, fig, filename: str, save: bool):
        """Save figure to training_curves_dir or show interactively."""
        if save:
            plt.tight_layout()
            fig.savefig(self.training_curves_dir / filename, dpi=150, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()

    # ========== Plot 1: Loss Curves ==========

    def _plot_loss_curves(self, epochs: List, metrics: List[Dict],
                          warmup_epochs: int, save: bool = True):
        """Train/Val/Aligned Loss + Pos/Neg/Inter Loss decomposition."""
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(epochs, self._extract(metrics, 'train_loss'), 'b-',
                label='Train Loss', linewidth=2)
        ax.plot(epochs, self._extract(metrics, 'val_loss'), 'r--',
                label='Val Loss (CE)', linewidth=1.5, alpha=0.6)

        val_aligned = self._extract(metrics, 'val_loss_aligned')
        if any(not np.isnan(v) if isinstance(v, float) else v is not None for v in val_aligned):
            ax.plot(epochs, val_aligned, 'r-', label='Val Loss (Aligned)', linewidth=2)

        if self._has_any(metrics, 'pos_loss'):
            ax.plot(epochs, self._extract(metrics, 'pos_loss'), 'g--',
                    label='Pos Loss', linewidth=1.5, alpha=0.7)
            ax.plot(epochs, self._extract(metrics, 'neg_loss'), color='orange',
                    linestyle='--', label='Neg Loss', linewidth=1.5, alpha=0.7)

        if self._has_any(metrics, 'inter_loss'):
            ax.plot(epochs, self._extract(metrics, 'inter_loss'), 'm:',
                    label='Inter-Bag Loss', linewidth=1.5, alpha=0.6)

        self._add_warmup_shading(ax, warmup_epochs, max(epochs))
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Loss Curves', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        self._save_or_show(fig, 'loss_curves.png', save)

    # ========== Plot 2: Accuracy Overview ==========

    def _plot_accuracy_overview(self, epochs: List, metrics: List[Dict],
                                warmup_epochs: int, save: bool = True):
        """Train/Val ACC + Pos/Neg ACC + FG Ratio (all on 0-1 scale)."""
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(epochs, self._extract(metrics, 'train_accuracy'), 'b-',
                label='Train ACC (overall)', linewidth=2)
        ax.plot(epochs, self._extract(metrics, 'val_accuracy'), 'r-',
                label='Val ACC (overall)', linewidth=2)

        if self._has_any(metrics, 'train_pos_accuracy'):
            ax.plot(epochs, self._extract(metrics, 'train_pos_accuracy'), 'g--',
                    label='Pos ACC (train)', linewidth=1.5, marker='o', markersize=3)
        if self._has_any(metrics, 'train_neg_accuracy'):
            ax.plot(epochs, self._extract(metrics, 'train_neg_accuracy'), color='orange',
                    linestyle='--', label='Neg ACC (train)', linewidth=1.5,
                    marker='s', markersize=3)

        ax.plot(epochs, self._extract(metrics, 'foreground_ratio'), 'purple',
                linestyle=':', label='FG Ratio (val)', linewidth=1.5, alpha=0.7)

        self._add_warmup_shading(ax, warmup_epochs, max(epochs))
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy / Ratio')
        ax.set_title('Accuracy Overview', fontsize=14, fontweight='bold')
        ax.set_ylim([-0.05, 1.05])
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        self._save_or_show(fig, 'accuracy_overview.png', save)

    # ========== Plot 3: Negative Metrics ==========

    def _plot_negative_metrics(self, epochs: List, metrics: List[Dict], save: bool = True):
        """Negative Recall + Negative Confidence (validation)."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(epochs, self._extract(metrics, 'negative_recall'), 'g-',
                 linewidth=2, marker='o', markersize=4)
        ax1.axhline(y=0.85, color='r', linestyle='--', label='P0 Target: 0.85', linewidth=1.5)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Negative Recall')
        ax1.set_title('Background Recognition\n(Negative Recall)', fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1.05])

        ax2.plot(epochs, self._extract(metrics, 'neg_disease_hallucination'), 'orange',
                 linewidth=2, marker='o', markersize=4)
        ax2.axhline(y=0.3, color='r', linestyle='--', label='Target: < 0.3', linewidth=1.5)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Neg Disease Hallucination')
        ax2.set_title('Background Suppression\n(Avg Max Disease Prob on Negatives)',
                       fontsize=13, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1.0])

        self._save_or_show(fig, 'negative_metrics.png', save)

    # ========== Plot 4: TopK Metrics ==========

    def _plot_topk_metrics(self, epochs: List, metrics: List[Dict], save: bool = True):
        """Top-K Lift + Top-1 Confidence + Hit Accuracy."""
        if not self._has_any(metrics, 'topk_lift'):
            return

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

        ax1.plot(epochs, self._extract(metrics, 'topk_lift'), 'b-',
                 linewidth=2, marker='s', markersize=4)
        ax1.axhline(y=0.15, color='r', linestyle='--', label='P0 Target: 0.15', linewidth=1.5)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Top-K Lift')
        ax1.set_title('Top-K Selection Quality\n(Lift over Random)', fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(epochs, self._extract(metrics, 'avg_top1_conf'), 'green',
                 linewidth=2, marker='o', markersize=4)
        ax2.axhline(y=0.5, color='r', linestyle='--', label='P0 Target: 0.5', linewidth=1.5)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Avg Top-1 Conf')
        ax2.set_title('Top-1 Tile Confidence', fontsize=13, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1.0])

        ax3.plot(epochs, self._extract(metrics, 'hit_acc'), 'purple',
                 linewidth=2, marker='D', markersize=4)
        ax3.axhline(y=0.5, color='orange', linestyle='--', label='Reference: 0.5', linewidth=1.5)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Hit Accuracy')
        ax3.set_title('Top-1 Classification Accuracy', fontsize=13, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim([0, 1.0])

        self._save_or_show(fig, 'topk_metrics.png', save)

    # ========== Plot 5: Pos/Neg Diagnostics ==========

    def _plot_pos_neg_diagnostics(self, epochs: List, metrics: List[Dict],
                                   warmup_epochs: int, save: bool = True):
        """P/N Loss Ratio + Tier Distribution (stable phase focus)."""
        has_pn = self._has_any(metrics, 'pos_neg_loss_ratio')
        has_tier = self._has_any(metrics, 'tier1_count')
        if not has_pn and not has_tier:
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: P/N Loss Ratio
        ax1 = axes[0]
        if has_pn:
            pn_ratio = self._extract(metrics, 'pos_neg_loss_ratio')
            ax1.plot(epochs, pn_ratio, 'b-', linewidth=2, marker='o', markersize=4)
            ax1.axhline(y=0.5, color='r', linestyle='--',
                        label='Collapse Warning: < 0.5', linewidth=1.5)
            ax1.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
        self._add_warmup_shading(ax1, warmup_epochs, max(epochs))
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Pos/Neg Loss Ratio')
        ax1.set_title('P/N Loss Balance\n(< 0.5 = Over-Suppression)', fontsize=13, fontweight='bold')
        if ax1.get_legend_handles_labels()[1]:
            ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Right: Tier Distribution (stacked area)
        ax2 = axes[1]
        if has_tier:
            t1 = self._extract(metrics, 'tier1_count')
            t2 = self._extract(metrics, 'tier2_count')
            t3 = self._extract(metrics, 'tier3_count')
            # Convert to ratios for stacked plot
            t1_r, t2_r, t3_r = [], [], []
            for a, b, c in zip(t1, t2, t3):
                total = (a or 0) + (b or 0) + (c or 0)
                if total > 0 and not (np.isnan(a) or np.isnan(b) or np.isnan(c)):
                    t1_r.append(a / total)
                    t2_r.append(b / total)
                    t3_r.append(c / total)
                else:
                    t1_r.append(np.nan)
                    t2_r.append(np.nan)
                    t3_r.append(np.nan)

            ax2.plot(epochs, t1_r, 'g-', label='Tier1 (Qualified)', linewidth=2)
            ax2.plot(epochs, t2_r, color='orange', linestyle='-',
                     label='Tier2 (Marginal)', linewidth=2)
            ax2.plot(epochs, t3_r, 'r-', label='Tier3 (Wrong)', linewidth=2)
            ax2.axhline(y=0.3, color='r', linestyle=':', alpha=0.5,
                        label='T1 Collapse: < 30%')
        self._add_warmup_shading(ax2, warmup_epochs, max(epochs))
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Tier Ratio')
        ax2.set_title('Three-Tier Distribution\n(Stable Phase)', fontsize=13, fontweight='bold')
        ax2.set_ylim([-0.05, 1.05])
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        self._save_or_show(fig, 'pos_neg_diagnostics.png', save)

    # ========== Plot 6: Confidence Analysis ==========

    def _plot_confidence_analysis(self, epochs: List, metrics: List[Dict],
                                   warmup_epochs: int, save: bool = True):
        """Pos Target Conf vs Pos Class0 Conf + Neg Class0 Conf."""
        if not self._has_any(metrics, 'train_pos_target_conf'):
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Left: Positive tile confidence breakdown
        ax1.plot(epochs, self._extract(metrics, 'train_pos_target_conf'), 'g-',
                 label='Pos -> Target Class', linewidth=2, marker='o', markersize=3)
        ax1.plot(epochs, self._extract(metrics, 'train_pos_class0_conf'), 'r-',
                 label='Pos -> Class0 (BG)', linewidth=2, marker='x', markersize=4)
        if self._has_any(metrics, 'train_pos_class0_ratio'):
            ax1_twin = ax1.twinx()
            ax1_twin.plot(epochs, self._extract(metrics, 'train_pos_class0_ratio'),
                          'r:', label='Pos misclass as BG', linewidth=1.5, alpha=0.6)
            ax1_twin.set_ylabel('Misclass Ratio', fontsize=10, color='red')
            ax1_twin.set_ylim([-0.05, 1.05])
            ax1_twin.legend(loc='center right', fontsize=8)
        self._add_warmup_shading(ax1, warmup_epochs, max(epochs))
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Confidence')
        ax1.set_title('Positive Tile Confidence\n(Target vs Background)',
                       fontsize=13, fontweight='bold')
        ax1.set_ylim([0, 1.0])
        ax1.legend(loc='upper left', fontsize=9)
        ax1.grid(True, alpha=0.3)

        # Right: Negative tile confidence
        ax2.plot(epochs, self._extract(metrics, 'train_neg_class0_conf'), 'green',
                 label='Neg -> Class0 (correct)', linewidth=2, marker='o', markersize=3)
        ax2.plot(epochs, self._extract(metrics, 'neg_disease_hallucination'), 'orange',
                 label='Neg max disease prob (val)', linewidth=2, marker='s', markersize=3)
        self._add_warmup_shading(ax2, warmup_epochs, max(epochs))
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Confidence')
        ax2.set_title('Negative Tile Confidence\n(BG Confidence vs Disease Leak)',
                       fontsize=13, fontweight='bold')
        ax2.set_ylim([0, 1.0])
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        self._save_or_show(fig, 'confidence_analysis.png', save)

    # ========== Plot 7: LR and Dynamic Weight ==========

    def _plot_lr_and_weight(self, epochs: List, metrics: List[Dict],
                             warmup_epochs: int, save: bool = True):
        """Learning Rate curve + Dynamic Weight evolution."""
        fig, ax1 = plt.subplots(figsize=(10, 5))

        lr = self._extract(metrics, 'learning_rate')
        ax1.plot(epochs, lr, 'b-', linewidth=2, label='Learning Rate')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Learning Rate', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        if self._has_any(metrics, 'dynamic_weight'):
            ax2 = ax1.twinx()
            dw = self._extract(metrics, 'dynamic_weight')
            ax2.plot(epochs, dw, 'r-', linewidth=2, label='Dynamic Weight')
            ax2.set_ylabel('Dynamic Weight', color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            ax2.legend(loc='center right')

        self._add_warmup_shading(ax1, warmup_epochs, max(epochs))
        ax1.set_title('Learning Rate & Dynamic Weight', fontsize=14, fontweight='bold')
        ax1.legend(loc='center left')
        ax1.grid(True, alpha=0.3)
        self._save_or_show(fig, 'lr_and_weight.png', save)


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

def init_logger(
    experiment_name: str,
    config: Dict,
    logs_root: str = "outputs/logs",
    checkpoints_root: str = "outputs/checkpoints"
) -> LocalLogger:
    """
    初始化logger（类似wandb.init）.

    Args:
        experiment_name: 实验名称
        config: 配置字典
        logs_root: Logs保存根目录（默认: outputs/logs）
        checkpoints_root: Checkpoints保存根目录（默认: outputs/checkpoints）

    Returns:
        LocalLogger实例
    """
    return LocalLogger(experiment_name, config, logs_root, checkpoints_root)


__all__ = ['LocalLogger', 'init_logger']
