import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights

class WebFG400Model(nn.Module):
    def __init__(self, num_classes=400, pretrained=True):
        """
        WebFG-400 分类模型
        
        Args:
            num_classes: 类别数量，WebFG-400 有 400 个类别
            pretrained: 是否使用预训练权重
        """
        super(WebFG400Model, self).__init__()
        
        # 使用预训练的 ResNet-50
        if pretrained:
            self.backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        else:
            self.backbone = models.resnet50(weights=None)
        
        # 获取特征维度
        in_features = self.backbone.fc.in_features
        
        # 替换最后的全连接层
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, num_classes)
        )
        
        # 初始化新的层
        self._initialize_weights(self.backbone.fc)
    
    def _initialize_weights(self, module):
        """初始化权重"""
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.backbone(x)

def create_model(num_classes=400, pretrained=True):
    """创建模型实例"""
    model = WebFG400Model(num_classes=num_classes, pretrained=pretrained)
    return model

def count_parameters(model):
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # 测试模型
    model = create_model(num_classes=400)
    print(f"模型参数量: {count_parameters(model):,}")
    
    # 测试前向传播
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    print(f"输出形状: {output.shape}")