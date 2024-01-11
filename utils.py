import os

import torch
from torchvision import transforms
from PIL import Image


def make_transform(size: tuple, normalize=True):
    """
    将PIL图像处理为可以直接作为模型输入的张量
    :param size: 模型输入的图像尺寸
    :param normalize: 是否进行规范化（vgg的输入需要规范化）
    :return:
    """
    transform_lst = [transforms.Resize(size),  # 将图像大小调整为 450x300
        transforms.ToTensor()]  # 将 PIL 图像转换为 Tensor]

    if normalize:
        transform_lst.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    transform = transforms.Compose(transform_lst)
    return transform


def load_image(image_path, transform):
    """
    加载图像
    :param image_path: 图像路径
    :param transform: 应用图像变换
    :return:
    """
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    return image


def save_model(model: torch.nn.Module, save_path):
    """
    保存pytorch模型文件
    :param model:
    :param save_path:
    :return:
    """
    save_dir, filename = os.path.split(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(model.state_dict(), os.path.join(save_dir, filename))


def denormalize(tensor):
    """反归一化张量以将其转换回图像格式"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).to(tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).to(tensor.device)
    tensor = tensor * std + mean
    tensor = tensor.clamp(0, 1)
    return tensor.to(tensor.device)


def gram_matrix(feature):
    """
    计算特征图的格莱姆矩阵，作为风格特征表示
    :param feature:输入特征图
    :return: 格莱姆矩阵
    """
    b, c, h, w = feature.size()
    feature = feature.view(b, c, h * w)  # 沿着h，w维度拉平
    G = torch.bmm(feature, feature.transpose(1, 2))
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


def save_image(tensor, output_dir, filename, denormalization=True):
    """
    保存图像到OUTPUT_DIR
    :param tensor: [0-1]区间的图像张量，形状为(1, 3, h, w)或(3, h, w)
    :param output_dir: 输出路径
    :param filename: 文件名
    :param denormalization: 是否使用反规范化
    :return:
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if denormalization:
        tensor = denormalize(tensor)
    image = transforms.ToPILImage()(tensor[0].cpu().detach())  # 只保存第一张
    image.save(os.path.join(output_dir, filename))
