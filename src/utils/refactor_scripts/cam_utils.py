#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CAM (Class Activation Mapping) 工具库
支持EigenCAM、GradCAM++和Transformer Attribution的实现
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import Dict, List, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


class ModelLayerDetector:
    """自动检测模型架构并推荐CAM目标层"""
    
    @staticmethod
    def detect_architecture(model) -> str:
        """
        检测模型类型
        
        Args:
            model: PyTorch模型
            
        Returns:
            架构类型: 'efficientnet', 'mobilenet', 'resnet', 'unknown'
        """
        model_name = model.__class__.__name__.lower()
        
        # 检测EfficientNet
        if 'efficientnet' in model_name:
            return 'efficientnet'
        # 检测MobileNet
        elif 'mobilenet' in model_name:
            return 'mobilenet'
        # 检测ResNet
        elif 'resnet' in model_name:
            return 'resnet'
        
        # 通过层名称检测
        layer_names = [name for name, _ in model.named_modules()]
        if any('efficientnet' in name.lower() or 'features.8' in name for name in layer_names):
            return 'efficientnet'
        elif any('mobilenet' in name.lower() or 'features.18' in name for name in layer_names):
            return 'mobilenet'
        elif any('layer4' in name for name in layer_names):
            return 'resnet'
        
        logger.warning(f"无法自动识别模型架构: {model_name}")
        return 'unknown'
    
    @staticmethod
    def get_recommended_layers(model) -> Dict[str, str]:
        """
        返回推荐的CAM目标层
        
        Args:
            model: PyTorch模型
            
        Returns:
            推荐层名称字典: {'cam_layer': 'layer_name'}
        """
        arch = ModelLayerDetector.detect_architecture(model)
        
        layer_map = {
            'efficientnet': 'features.8',  # EfficientNet-B0最后卷积层
            'mobilenet': 'features.18',    # MobileNetV2最后卷积层
            'resnet': 'layer4',            # ResNet最后残差块
            'unknown': 'features.8'        # 默认尝试EfficientNet格式
        }
        
        recommended_layer = layer_map.get(arch, 'features.8')
        logger.info(f"检测到模型架构: {arch}, 推荐CAM层: {recommended_layer}")
        
        return {'cam_layer': recommended_layer}


class EigenCAM:
    """
    EigenCAM实现 - 基于特征图主成分分析的无梯度CAM方法
    特别适合实时应用和单病害分类任务
    """
    
    def __init__(self, model, target_layer_name: Optional[str] = None):
        """
        初始化EigenCAM
        
        Args:
            model: 训练好的模型
            target_layer_name: 目标层名称（如果为None则自动检测）
        """
        self.model = model
        
        # 自动检测目标层
        if target_layer_name is None:
            detector = ModelLayerDetector()
            recommended = detector.get_recommended_layers(model)
            target_layer_name = recommended['cam_layer']
            logger.info(f"[EigenCAM] 自动检测目标层: {target_layer_name}")
        
        self.target_layer_name = target_layer_name
        self.activations = None
        self.hooks = []
        
        # 注册hook获取中间特征
        self._register_hooks()
        
    def _register_hooks(self):
        """注册前向传播hooks（带层查找和详细错误提示）"""
        def hook_fn(module, input, output):
            self.activations = output.detach()
        
        # 找到目标层并注册hook
        layer_found = False
        for name, module in self.model.named_modules():
            if name == self.target_layer_name:
                hook = module.register_forward_hook(hook_fn)
                self.hooks.append(hook)
                layer_found = True
                break
        
        if not layer_found:
            available_layers = [name for name, _ in self.model.named_modules() if len(name) > 0]
            raise ValueError(
                f"❌ 未找到目标层 '{self.target_layer_name}'。\n"
                f"前10个可用层: {available_layers[:10]}\n"
                f"建议：检查模型架构或手动指定target_layer_name"
            )
                
    def generate_cam(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """
        生成EigenCAM热力图
        
        Args:
            input_tensor: 输入图像张量 [1, 3, H, W]
            target_class: 目标类别（如果为None，使用预测类别）
            
        Returns:
            cam: CAM热力图 [H, W]
        """
        self.model.eval()
        
        with torch.no_grad():
            # 前向传播
            outputs = self.model(input_tensor)
            
            if target_class is None:
                target_class = outputs.argmax(dim=1).item()
                
            # 获取激活特征 [1, C, H, W]
            activations = self.activations
            
            if activations is None:
                raise ValueError(f"未找到目标层 {self.target_layer_name} 的激活值")
                
            # 转换维度 [C, H*W]
            batch_size, channels, height, width = activations.shape
            activations_reshaped = activations.view(channels, height * width)
            
            # 主成分分析 (PCA)
            # 计算协方差矩阵
            activations_centered = activations_reshaped - activations_reshaped.mean(dim=1, keepdim=True)
            
            # SVD分解获取主成分
            U, S, V = torch.svd(activations_centered)
            
            # 使用第一主成分作为权重
            principal_component = U[:, 0]  # [C]
            
            # 计算加权激活图
            weighted_activations = (principal_component.unsqueeze(-1) * activations_reshaped).sum(dim=0)
            
            # 重塑为空间维度
            cam = weighted_activations.view(height, width)
            
            # 标准化到 [0, 1]
            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-8)
            
            return cam.cpu().numpy()
            
    def cleanup(self):
        """清理hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


class GradCAMPlusPlus:
    """
    GradCAM++实现 - 改进的梯度加权类激活映射
    特别适合多病害检测任务，能同时高亮多个目标区域
    """
    
    def __init__(self, model, target_layer_name: Optional[str] = None):
        """
        初始化GradCAM++
        
        Args:
            model: 训练好的模型
            target_layer_name: 目标层名称（如果为None则自动检测）
        """
        self.model = model
        
        # 自动检测目标层
        if target_layer_name is None:
            detector = ModelLayerDetector()
            recommended = detector.get_recommended_layers(model)
            target_layer_name = recommended['cam_layer']
            logger.info(f"[GradCAM++] 自动检测目标层: {target_layer_name}")
        
        self.target_layer_name = target_layer_name
        self.activations = None
        self.gradients = None
        self.hooks = []
        
        self._register_hooks()
        
    def _register_hooks(self):
        """注册前向和反向传播hooks（带层查找和详细错误提示）"""
        def forward_hook(module, input, output):
            self.activations = output
            
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        # 注册hooks
        layer_found = False
        for name, module in self.model.named_modules():
            if name == self.target_layer_name:
                hook1 = module.register_forward_hook(forward_hook)
                hook2 = module.register_full_backward_hook(backward_hook)
                self.hooks.extend([hook1, hook2])
                layer_found = True
                break
        
        if not layer_found:
            available_layers = [name for name, _ in self.model.named_modules() if len(name) > 0]
            raise ValueError(
                f"❌ 未找到目标层 '{self.target_layer_name}'。\n"
                f"前10个可用层: {available_layers[:10]}\n"
                f"建议：检查模型架构或手动指定target_layer_name"
            )
                
    def generate_cam(self, input_tensor: torch.Tensor, target_classes: List[int]) -> Dict[int, np.ndarray]:
        """
        生成GradCAM++热力图（支持多个目标类别）
        
        Args:
            input_tensor: 输入图像张量 [1, 3, H, W]
            target_classes: 目标类别列表
            
        Returns:
            cams: 每个类别对应的CAM热力图字典 {class_id: cam_array}
        """
        self.model.eval()
        input_tensor.requires_grad_(True)
        
        # 前向传播
        outputs = self.model(input_tensor)
        
        cams = {}
        
        for target_class in target_classes:
            # 清零梯度
            self.model.zero_grad()
            
            # 反向传播
            class_score = outputs[0, target_class]
            class_score.backward(retain_graph=True)
            
            # 获取梯度和激活
            gradients = self.gradients  # [1, C, H, W]
            activations = self.activations  # [1, C, H, W]
            
            # GradCAM++ 权重计算
            alpha_num = gradients.pow(2)
            alpha_denom = 2.0 * gradients.pow(2) + \
                         activations.mul(gradients.pow(3)).view(gradients.size(0), gradients.size(1), -1).sum(dim=2, keepdim=True).view(gradients.size(0), gradients.size(1), 1, 1)
            
            alpha = alpha_num.div(alpha_denom + 1e-7)
            weights = (alpha * F.relu(gradients)).view(gradients.size(0), gradients.size(1), -1).sum(dim=2)
            
            # 生成CAM
            cam = (weights.view(*weights.shape, 1, 1) * activations).sum(dim=1, keepdim=True)
            cam = F.relu(cam)
            
            # 标准化
            cam = cam.squeeze()
            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-8)
            
            cams[target_class] = cam.detach().cpu().numpy()
            
        return cams
        
    def cleanup(self):
        """清理hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


class TransformerAttribution:
    """
    Transformer注意力可视化
    展示ViT组件的注意力权重分布
    """
    
    def __init__(self, model):
        """
        初始化Transformer Attribution
        
        Args:
            model: 包含ViT组件的混合模型
        """
        self.model = model
        self.attention_weights = []
        self.hooks = []
        
        self._register_attention_hooks()
        
    def _register_attention_hooks(self):
        """注册注意力权重获取hooks"""
        def attention_hook(module, input, output):
            # 获取注意力权重
            if hasattr(module, 'attention_weights') and module.attention_weights is not None:
                self.attention_weights.append(module.attention_weights.clone())
                
        # 为每个MultiHeadSelfAttention模块注册hook
        for name, module in self.model.named_modules():
            if 'MultiHeadSelfAttention' in str(type(module)):
                hook = module.register_forward_hook(attention_hook)
                self.hooks.append(hook)
                
    def generate_attention_map(self, input_tensor: torch.Tensor, layer_idx: int = -1) -> np.ndarray:
        """
        生成注意力热力图
        
        Args:
            input_tensor: 输入图像张量
            layer_idx: 使用的层索引（-1表示最后一层）
            
        Returns:
            attention_map: 注意力热力图
        """
        self.model.eval()
        self.attention_weights = []
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            
            if not self.attention_weights:
                # 如果没有获取到注意力权重，使用简化方法
                return self._generate_simple_attention_map(input_tensor)
                
            # 使用指定层的注意力权重
            attention = self.attention_weights[layer_idx]  # [batch, heads, seq_len, seq_len]
            
            # 平均所有注意力头
            attention = attention.mean(dim=1)  # [batch, seq_len, seq_len]
            
            # 获取CLS token对所有patch的注意力
            cls_attention = attention[0, 0, 1:]  # 忽略CLS token自身，获取对196个patch的注意力
            
            # 重塑为空间维度（14x14的patch网格）
            patch_size = 14  # 固定为14x14，因为模型使用AdaptiveAvgPool2d((14, 14))
            if len(cls_attention) == patch_size * patch_size:
                attention_map = cls_attention.view(patch_size, patch_size)
            else:
                # 如果长度不匹配，进行调整
                attention_map = cls_attention[:patch_size * patch_size].view(patch_size, patch_size)
            
            # 标准化到[0, 1]
            attention_map = attention_map - attention_map.min()
            attention_map = attention_map / (attention_map.max() + 1e-8)
            
            return attention_map.cpu().numpy()
            
    def _generate_simple_attention_map(self, input_tensor: torch.Tensor) -> np.ndarray:
        """
        简化的注意力图生成（当无法获取真实注意力权重时）
        基于最终特征的空间重要性
        """
        # 这里可以实现一个基于特征激活的简化注意力图
        # 暂时返回均匀分布
        return np.ones((14, 14)) * 0.5
        
    def cleanup(self):
        """清理hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


class CAMVisualizer:
    """
    CAM可视化工具类
    负责将CAM结果转换为可视化图像
    """
    
    @staticmethod
    def apply_colormap(cam: np.ndarray, colormap: str = 'jet') -> np.ndarray:
        """
        为CAM应用颜色映射
        
        Args:
            cam: CAM热力图 [H, W]
            colormap: 颜色映射名称
            
        Returns:
            colored_cam: 彩色CAM图像 [H, W, 3]
        """
        # 标准化到 [0, 255]
        cam_normalized = (cam * 255).astype(np.uint8)
        
        # 应用颜色映射
        colormap_fn = cm.get_cmap(colormap)
        colored_cam = colormap_fn(cam_normalized)[:, :, :3]  # 去除alpha通道
        
        return (colored_cam * 255).astype(np.uint8)
        
    @staticmethod
    def overlay_cam_on_image(original_image: Union[np.ndarray, Image.Image], 
                           cam: np.ndarray, 
                           alpha: float = 0.4,
                           colormap: str = 'jet') -> np.ndarray:
        """
        将CAM叠加到原始图像上
        
        Args:
            original_image: 原始图像
            cam: CAM热力图
            alpha: 叠加透明度
            colormap: 颜色映射
            
        Returns:
            overlay_image: 叠加后的图像
        """
        # 确保图像是numpy数组
        if isinstance(original_image, Image.Image):
            original_image = np.array(original_image)
            
        # 调整CAM大小到图像大小
        h, w = original_image.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))
        
        # 应用颜色映射
        colored_cam = CAMVisualizer.apply_colormap(cam_resized, colormap)
        
        # 叠加
        overlay = cv2.addWeighted(original_image, 1-alpha, colored_cam, alpha, 0)
        
        return overlay
        
    @staticmethod
    def create_cam_subplot(original_image: Union[np.ndarray, Image.Image],
                          cams: Dict[str, np.ndarray],
                          class_names: List[str],
                          save_path: Optional[str] = None) -> plt.Figure:
        """
        创建CAM对比图
        
        Args:
            original_image: 原始图像
            cams: CAM结果字典 {'method_name': cam_array}
            class_names: 类别名称列表
            save_path: 保存路径（可选）
            
        Returns:
            fig: matplotlib图形对象
        """
        n_cams = len(cams) + 1  # +1 for original image
        
        fig, axes = plt.subplots(1, n_cams, figsize=(4*n_cams, 4))
        if n_cams == 1:
            axes = [axes]
            
        # 显示原始图像
        axes[0].imshow(original_image)
        axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # 显示各种CAM
        for idx, (method_name, cam) in enumerate(cams.items(), 1):
            overlay = CAMVisualizer.overlay_cam_on_image(original_image, cam)
            axes[idx].imshow(overlay)
            axes[idx].set_title(f'{method_name} CAM', fontsize=12, fontweight='bold')
            axes[idx].axis('off')
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig


# 便捷函数
def quick_eigencam(model, input_tensor: torch.Tensor, 
                  target_layer: str = 'cnn_features.5') -> np.ndarray:
    """
    快速生成EigenCAM的便捷函数
    
    Args:
        model: 模型
        input_tensor: 输入张量
        target_layer: 目标层名称
        
    Returns:
        cam: CAM热力图
    """
    eigen_cam = EigenCAM(model, target_layer)
    try:
        cam = eigen_cam.generate_cam(input_tensor)
        return cam
    finally:
        eigen_cam.cleanup()


def quick_gradcam_plus(model, input_tensor: torch.Tensor,
                      target_classes: List[int],
                      target_layer: str = 'cnn_features.5') -> Dict[int, np.ndarray]:
    """
    快速生成GradCAM++的便捷函数
    
    Args:
        model: 模型
        input_tensor: 输入张量
        target_classes: 目标类别列表
        target_layer: 目标层名称
        
    Returns:
        cams: 各类别的CAM结果
    """
    gradcam_plus = GradCAMPlusPlus(model, target_layer)
    try:
        cams = gradcam_plus.generate_cam(input_tensor, target_classes)
        return cams
    finally:
        gradcam_plus.cleanup()


def quick_transformer_attribution(model, input_tensor: torch.Tensor,
                                 layer_idx: int = -1) -> np.ndarray:
    """
    快速生成Transformer注意力图的便捷函数
    
    Args:
        model: 包含ViT组件的模型
        input_tensor: 输入张量
        layer_idx: 使用的Transformer层索引
        
    Returns:
        attention_map: 注意力热力图
    """
    transformer_attr = TransformerAttribution(model)
    try:
        attention_map = transformer_attr.generate_attention_map(input_tensor, layer_idx)
        return attention_map
    finally:
        transformer_attr.cleanup()


if __name__ == "__main__":
    # 测试代码示例
    print("CAM工具库加载完成！")
    print("支持的方法：")
    print("- EigenCAM: 快速无梯度CAM，适合单病害分类")
    print("- GradCAM++: 改进梯度CAM，适合多病害检测")  
    print("- TransformerAttention: ViT注意力可视化")
    print("- CAMVisualizer: 可视化工具")