"""
抽象评估指标基类：BaseMetric
目的： 统一所有评估指标（如 Accuracy, IoU, mAP, F1）的计算和状态管理。

内容：

抽象方法： update(self, preds, targets) (更新累计值)、compute(self) (计算最终指标)、reset(self) (重置状态)。

用途： 使得在训练/验证循环中，指标的累积和报告变得标准化。

避免过度抽象： 不需要为每一种指标（如 IoU, Dice Loss）都建立一个基类，一个 BaseMetric 足以封装指标计算流程。

"""