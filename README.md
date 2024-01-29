# MMTU-Net
 A semantic segmentation method that employs a U-shaped network, integrating multi-level and multi-scale feature fusion, while also incorporating Transformer for enhanced performance.

# Model Description

Although convolutional neural networks (CNNs) have been the primary method in the field of medical image semantic segmentation, they still face challenges such as limited information propagation, emphasis on local information, and significant information loss.

Hence, we propose the Multiscale and Multilevel U-shaped Network with Transformer (MMTU-Net). This network aims to address the following three requirements:
1. Altering the weights between different channels of the image to allow the model to focus on more critical parts.
2. Enriching the semantic information of the image to some extent, thereby reducing information loss.
3. Expanding the model's receptive field to enable more comprehensive information extraction.

To fulfill these requirements, we have designed three modules: Multilevel Fusion Transformer (MFT), Dual Channel Attention mechanism (DCA), and Double-layer Multi-Scale Feature Fusion (DMFF).

# Model Structure

![image](https://github.com/YF-W/MMTU-Net/assets/66008255/9de7e4a3-e45a-48ec-8779-bb402b1c9a4e)

# Environment：
IDE: Pycharm 2020.1 Professional ED.

Language: Python 3.9.7

Framework: Pytorch 1.11.0
