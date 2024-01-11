"""
风格迁移算法 - StyleFusion
作者: hammershock
描述: 该脚本用于实现非实时风格迁移，通过深度学习模型融合内容图像和风格图像。
      本实现依赖于PyTorch框架，并使用预训练的VGG模型作为风格和内容特征的提取器。

使用方法:
    运行该脚本时，可以通过命令行参数指定内容图像路径、风格图像路径、输出目录等。
    示例: python stylefusion.py --content_image ./data/city.jpg --style_image ./data/udnie.jpg --output_dir ./output/iterative_style_transfer
"""

import argparse
import torch
import torch.optim as optim
from tqdm import tqdm

from models import VGG
from utils import make_transform, load_image, save_image, calculate_style_loss, calculate_content_loss


def parse_args():
    parser = argparse.ArgumentParser(description='风格迁移算法 - StyleFusion')
    parser.add_argument('--content_image', type=str, default='./data/city.jpg', help='内容图像路径')
    parser.add_argument('--style_image', type=str, default='./data/udnie.jpg', help='风格图像路径')
    parser.add_argument('--output_dir', type=str, default='./output/iterative_style_transfer', help='输出目录')
    parser.add_argument('--image_size', type=int, nargs=2, default=[300, 450], help='图像尺寸 (高度, 宽度)')
    parser.add_argument('--content_weight', type=float, default=1, help='内容权重')
    parser.add_argument('--style_weight', type=float, default=15, help='风格权重')
    parser.add_argument('--epochs', type=int, default=20, help='训练周期数')
    parser.add_argument('--steps_per_epoch', type=int, default=100, help='每个周期的步数')
    parser.add_argument('--learning_rate', type=float, default=0.03, help='学习率')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # 内容特征层及loss加权系数
    content_layers = {'5': 0.5, '10': 0.5}
    # 风格特征层及loss加权系数
    style_layers = {'0': 0.2, '5': 0.2, '10': 0.2, '19': 0.2, '28': 0.2}

    # ----------------训练即推理过程----------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = make_transform(args.image_size, normalize=True)

    content_img = load_image(args.content_image, transform).to(device)  # 读取内容图像
    style_img = load_image(args.style_image, transform).to(device)  # 读取风格图像

    generated_img = torch.randn_like(content_img, requires_grad=True).to(device)  # 随机初始化目标图像
    save_image(generated_img, args.output_dir, 'noise.jpg')  # 保存初始噪声图，供查看

    vgg_model = VGG(content_layers, style_layers).to(device).eval()  # 实例化模型和优化器
    # print(vgg_model.model)  # 打印vgg模型结构
    optimizer = optim.Adam([generated_img], lr=args.learning_rate)  # 直接对图像本身进行优化

    content_features, _ = vgg_model(content_img)  # 计算内容图的内容特征
    _, style_features = vgg_model(style_img)  # 计算风格图的风格特征

    for epoch in range(args.epochs):
        p_bar = tqdm(range(args.steps_per_epoch), desc=f'epoch {epoch}')
        for step in p_bar:
            generated_content, generated_style = vgg_model(generated_img)  # 计算生成图片的不同层次的内容特征和风格特征
            # 不同层次的内容和风格特征损失加权求和
            content_loss = sum(args.content_weight * content_layers[name] * calculate_content_loss(content_features[name], gen_content) for name, gen_content in generated_content.items())
            style_loss = sum(args.style_weight * style_layers[name] * calculate_style_loss(style_features[name], gen_style) for name, gen_style in generated_style.items())

            total_loss = style_loss + content_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(generated_img, max_norm=1)  # 梯度裁剪
            optimizer.step()
            p_bar.set_postfix(style_loss=style_loss.item(), content_loss=content_loss.item())

        # 保存生成图像
        save_image(generated_img, args.output_dir, f'generated_{epoch + 1}.jpg')

