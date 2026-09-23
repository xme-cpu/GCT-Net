# GCT-Net
GCT-Net: Gated Bidirectional Feature Pyramid and Cooperative Difference Enhancement Network for Remote Sensing Image Change Detection

This repository is the official implementation of **GCT-Net**, a novel multi-module collaborative network for high-performance remote sensing image change detection.

## Abstract
Remote sensing image change detection plays an essential role in urban monitoring, disaster assessment, and ecological analysis. Existing methods still suffer from insufficient multi-scale feature fusion, limited bi-temporal interaction capability, and ambiguous change boundaries in complex scenarios. 

To address these issues, this paper proposes a gated and cooperative feature enhancement network (**GCT-Net**), which consists of three core designs:
1. **Gated Bidirectional Feature Pyramid (G-BiFPN)** for adaptive multi-scale feature aggregation.
2. **Cooperative Difference Enhancement Module (CDEM)** for enhancing change-sensitive difference features.
3. **Transformer-based bi-temporal interaction modeling** for capturing long-range temporal dependencies.

Extensive experiments on three public remote sensing change detection datasets demonstrate that GCT-Net achieves superior performance compared with state-of-the-art methods.

## Datasets
All experiments are conducted on publicly available remote sensing change detection datasets:
- **LEVIR-CD**
- **WHU-CD**

You can download the datasets from their official public releases.

## Requirements
- Python 3.8+
- PyTorch 1.10+
- torchvision
- opencv-python
- numpy
- scipy

You can install dependencies via:
```bash
pip install -r requirements.txt
