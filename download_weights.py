import os
import torch
from torch.hub import load_state_dict_from_url
import timm

# 创建预训练权重目录
os.makedirs('pretrained', exist_ok=True)

# 下载 ResNet 预训练权重
print('下载 ResNet 预训练权重...')
resnet_model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

for model_name, url in resnet_model_urls.items():
    save_path = os.path.join('pretrained', url.split('/')[-1])
    if not os.path.exists(save_path):
        print(f'下载 {model_name} 权重到 {save_path}')
        state_dict = load_state_dict_from_url(url, progress=True)
        torch.save(state_dict, save_path)
    else:
        print(f'{model_name} 权重已存在：{save_path}')

# 下载 ConvNeXt 预训练权重
print('\n下载 ConvNeXt 预训练权重...')
convnext_models = ['convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large']

for model_name in convnext_models:
    save_path = os.path.join('pretrained', f'{model_name}.pth')
    if not os.path.exists(save_path):
        print(f'下载 {model_name} 权重到 {save_path}')
        model = timm.create_model(model_name, pretrained=True)
        torch.save(model.state_dict(), save_path)
    else:
        print(f'{model_name} 权重已存在：{save_path}')

print('\n所有预训练权重已下载完成！')