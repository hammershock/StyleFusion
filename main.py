import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import os

from torchvision.models import VGG19_Weights
from tqdm import tqdm


def load_image(image_path, size=(450, 300), normalization=True):
    image = Image.open(image_path).convert('RGB')
    if size is not None:
        image = image.resize(size)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) if normalization else transforms.Lambda(lambda x: x)
    ])
    image = transform(image).unsqueeze(0)
    return image


def denormalize(tensor):
    """反归一化张量以将其转换回图像格式"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).to(tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).to(tensor.device)
    tensor = tensor * std + mean
    tensor = tensor.clamp(0, 1)
    return tensor.to(tensor.device)


class VGG(nn.Module):
    def __init__(self, content_layers, style_layers):
        super(VGG, self).__init__()
        self.model = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        self.content_layers = content_layers.keys()
        self.style_layers = style_layers.keys()

        # 冻结模型的所有参数
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        """
        对vgg19网络的包装，前向传播时保留了内容层和风格层的中间输出
        :param x:
        :return: 内容层和风格层的特征图
        """
        content_features = {}
        style_features = {}

        for name, layer in self.model._modules.items():
            x = layer(x)
            if name in self.content_layers:
                content_features[name] = x
            if name in self.style_layers:
                style_features[name] = x

        return content_features, style_features


def gram_matrix(feature):
    """
    计算特征图的格莱姆矩阵
    :param feature:
    :return:
    """
    b, c, h, w = feature.size()
    feature = feature.view(c, h * w)  # 沿着h，w维度拉平
    G = torch.matmul(feature, feature.T)  # matmul, T方法适用于二维即以上张量
    # G = torch.mm(feature, feature.t())  # mm, t()仅仅适用于二维矩阵的运算
    return G


def calculate_content_loss(original_feat, generated_feat) -> torch.Tensor:
    """计算内容损失，即生成特征图与标准特征图的规范化误差平方和"""
    b, c, h, w = original_feat.shape
    x = 2. * c * h * w  # 规范化系数
    return torch.sum((generated_feat - original_feat)**2) / x


def calculate_style_loss(style_feat, generated_feat) -> torch.Tensor:
    """计算风格损失，即生成特征图与标准特征图的格拉姆矩阵的规范化误差平方和"""
    b, c, h, w = style_feat.shape
    G = gram_matrix(generated_feat)
    A = gram_matrix(style_feat)
    x = 4. * ((h * w) ** 2) * (c ** 2)  # 规范化系数
    return torch.sum((G - A)**2) / x


def save_image(tensor, filename, denormalization=True):
    """
    保存图像到OUTPUT_DIR
    :param tensor:
    :param filename:
    :param denormalization: 是否使用反归一化
    :return:
    """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    if denormalization:
        tensor = denormalize(tensor)
    image = transforms.ToPILImage()(tensor.squeeze().cpu().detach())
    image.save(f"{OUTPUT_DIR}/{filename}")

# vgg-19模型结构：
# Sequential(
    # (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    # (1): ReLU(inplace=True)
    # (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    # (3): ReLU(inplace=True)
    # (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    # (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    # (6): ReLU(inplace=True)
    # (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    # (8): ReLU(inplace=True)
    # (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    # (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    # (11): ReLU(inplace=True)
    # (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    # (13): ReLU(inplace=True)
    # (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    # (15): ReLU(inplace=True)
    # (16): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    # (17): ReLU(inplace=True)
    # (18): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    # (19): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    # (20): ReLU(inplace=True)
    # (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    # (22): ReLU(inplace=True)
    # (23): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    # (24): ReLU(inplace=True)
    # (25): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    # (26): ReLU(inplace=True)
    # (27): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    # (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    # (29): ReLU(inplace=True)
    # (30): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    # (31): ReLU(inplace=True)
    # (32): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    # (33): ReLU(inplace=True)
    # (34): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    # (35): ReLU(inplace=True)
    # (36): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    # )


if __name__ == "__main__":
    # ----------------路径参数----------------
    CONTENT_IMAGE_PATH = './data/content.jpg'
    STYLE_IMAGE_PATH = './data/style.jpg'
    OUTPUT_DIR = './output'
    # ----------------风格迁移核心参数----------------
    image_size = (450, 300)
    # 内容特征层及loss加权系数
    content_layers = {'5': 0.5, '10': 0.5}  # 使用较浅层作为内容特征，保证生成图片内容结构相似性
    # 风格特征层及loss加权系数
    style_layers = {'0': 0.2, '5': 0.2, '10': 0.2, '19': 0.2, '28': 0.2}  # 使用不同深度的风格特征，生成风格更加层次丰富
    content_weight = 1
    style_weight = 100
    # ----------------训练参数----------------
    EPOCHS = 20
    STEPS_PER_EPOCH = 100
    learning_rate = 0.03
    # ----------------开始训练----------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    content_img = load_image(CONTENT_IMAGE_PATH, size=image_size).to(device)  # 读取内容图像
    style_img = load_image(STYLE_IMAGE_PATH, size=image_size).to(device)  # 读取风格图像

    generated_img = torch.randn_like(content_img, requires_grad=True).to(device)  # 随机初始化目标图像
    save_image(generated_img, 'noise.jpg')  # 保存初始噪声图，供查看

    model = VGG(content_layers, style_layers).to(device).eval()  # 实例化模型和优化器
    optimizer = optim.Adam([generated_img], lr=learning_rate)  # 直接对图像本身进行优化

    content_features, _ = model(content_img)  # 计算内容图的内容特征
    _, style_features = model(style_img)  # 计算风格图的风格特征

    for epoch in range(EPOCHS):
        p_bar = tqdm(range(STEPS_PER_EPOCH), desc=f'epoch {epoch}')
        for step in p_bar:
            generated_content, generated_style = model(generated_img)  # 计算生成图片的不同层次的内容特征和风格特征
            # 不同层次的内容和风格特征损失加权求和
            content_loss = sum(content_weight * content_layers[name] * calculate_content_loss(content_features[name], gen_content) for name, gen_content in generated_content.items())
            style_loss = sum(style_weight * style_layers[name] * calculate_style_loss(style_features[name], gen_style) for name, gen_style in generated_style.items())

            total_loss = style_loss + content_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(generated_img, max_norm=1)  # 梯度裁剪
            optimizer.step()
            p_bar.set_postfix(style_loss=style_loss.item(), content_loss=content_loss.item())

        # 保存生成图像
        save_image(generated_img, f'generated_{epoch+1}.jpg')
