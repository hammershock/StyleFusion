"""
实时风格迁移算法 - Fast Style Transfer
作者: [您的名字或者用户名]
描述: 这个脚本实现了实时风格迁移，支持训练模型、处理一批图像以及处理视频文件。它基于PyTorch实现，并使用VGG网络提取风格特征。

使用方法:
    - 训练模式: python fast_style_transfer.py --mode train --style_image ./data/udnie.jpg --content_dataset data/train2017 --model_save_path ./models/new_model.pth --epochs 10
    - 图像处理模式: python fast_style_transfer.py --mode image --input_images_dir ./data/train2014/default_class --output_images_dir ./output/images_generated --model_path ./models/udnie.pth
    - 视频处理模式: python fast_style_transfer.py --mode video --video_input data/maigua.mp4 --video_output output/videos/maigua_udnie.mp4 --model_path ./models/udnie.pth
"""

import argparse
import os
from datetime import datetime, timedelta
from typing import List, Iterable

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
import cv2
import numpy as np

from models import VGG, TransNet
from datasets import COCODataset
from utils import load_image, save_image, make_transform, save_model, calculate_style_loss, calculate_content_loss, \
    denormalize


def train(model,
          vgg,
          lr, epochs, batch_size, style_weight, content_weight, style_layers, content_layers,
          device, transform,
          image_style,
          content_dataset_root,
          save_path, output_dir,
          log_dir='./runs/fast_style_transfer',
          save_interval=timedelta(seconds=120)):
    """

    :param model: 内容生成模型，输出与输出图像尺寸相同
    :param vgg: 特征提取网络，以vgg19为例
    :param lr:
    :param epochs:
    :param batch_size:
    :param style_weight: 风格损失权重
    :param content_weight: 内容损失权重
    :param style_layers: 选取的风格层及其权重
    :param content_layers: 选取的内容层及其权重
    :param device: 计算设备
    :param transform: 图像变换
    :param image_style: 风格图片
    :param content_dataset_root: 内容图片文件夹路径
    :param save_path: 模型保存路径
    :param output_dir: 中间结果输出路径
    :param log_dir: tensorboard日志的保存路径
    :param save_interval: 保存时间间隔
    :return:
    """
    writer = SummaryWriter(log_dir)  # autodl平台tensorboard默认日志路径
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # 对目标网络进行优化
    dataset = COCODataset(content_dataset_root, transform=transform)
    dataloader = DataLoader(dataset, batch_size, shuffle=True)
    _, style_features = vgg(image_style)
    p_bar = tqdm(range(epochs))
    last_save_time = datetime.now() - save_interval

    for epoch in p_bar:
        running_content_loss, running_style_loss = 0.0, 0.0
        for i, content_img in enumerate(dataloader):
            content_img = content_img.to(device)
            image_generated = model(content_img)  # 只使用内容图像进行风格迁移
            generated_content, generated_style = vgg(image_generated)
            style_loss = sum(
                style_weight * style_layers[name] * calculate_style_loss(style_features[name], gen_style) for
                name, gen_style in generated_style.items())

            content_features, _ = vgg(content_img)  # 计算内容图的内容特征
            content_loss = sum(
                content_weight * content_layers[name] * calculate_content_loss(content_features[name], gen_content) for
                name, gen_content in generated_content.items())

            total_loss = style_loss + content_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)  # 梯度裁剪
            optimizer.step()

            running_content_loss += content_loss.item()
            running_style_loss += style_loss.item()

            p_bar.set_postfix(progress=f'{(i + 1) / len(dataloader) * 100:.3f}%',
                              style_loss=f"{style_loss.item():.3f}",
                              content_loss=f"{content_loss.item():.3f}",
                              last_save_time=last_save_time)

            writer.add_scalar('Loss/content', running_content_loss / (i + 1), epoch * len(dataloader) + i)
            writer.add_scalar('Loss/style', running_style_loss / (i + 1), epoch * len(dataloader) + i)

            if datetime.now() - last_save_time > save_interval:
                last_save_time = datetime.now()
                writer.add_images('image_generated', denormalize(image_generated), epoch * len(dataloader) + i)
                save_model(model, save_path)  # 'fast_style_transfer.pth'
                save_image(torch.cat((image_generated, content_img), 3), output_dir, f'{epoch}_{i}.jpg')

    writer.close()


def process_images(images: Iterable[Image.Image], transform, model, device) -> List[Image.Image]:
    images = torch.stack([transform(image) for image in images]).to(device)
    model.to(device)
    batch_generated = model(images)
    batch_generated = denormalize(batch_generated).detach().cpu()
    batch_generated = [transforms.ToPILImage()(image) for image in batch_generated]
    return batch_generated


def process_video(video_path, output_path, transform, model, device, batch_size=4):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    output_dir, filename = os.path.split(output_path)

    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 获取视频属性
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 定义视频编码器和创建 VideoWriter 对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # 初始化 tqdm 进度条
    pbar = tqdm(total=total_frames, desc="Processing Video")

    # 读取视频并批量处理
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        if len(frames) == batch_size:
            batch_generated = process_images(frames, transform, model, device)
            for gen_frame in batch_generated:
                gen = cv2.cvtColor(np.array(gen_frame), cv2.COLOR_RGB2BGR)
                cv2.imshow("output", gen)
                gen = cv2.resize(gen, (frame_width, frame_height))
                out.write(gen)
                if cv2.waitKey(1) & 0xFF == ord('s'):
                    break

            frames.clear()
            pbar.update(batch_size)

    if frames:
        batch_generated = process_images(frames, transform, model, device)
        for gen_frame in batch_generated:
            out.write(cv2.cvtColor(np.array(gen_frame), cv2.COLOR_RGB2BGR))
        pbar.update(len(frames))
    print(f'video successfully saved to: {output_path}')
    pbar.close()
    cap.release()
    out.release()


def parse_args():
    parser = argparse.ArgumentParser(description='实时风格迁移算法 - Fast Style Transfer')
    parser.add_argument('--mode', type=str, choices=['train', 'image', 'video'], default='video',
                        help='运行模式: train, image, video')
    parser.add_argument('--image_size', type=int, nargs=2, default=[300, 450], help='图像尺寸 (高度, 宽度)')
    parser.add_argument('--style_image', type=str, default='./data/udnie.jpg', help='风格图像路径 (仅训练模式)')
    # data/train2017/default_class下包含了若干个图片，由于ImageFolder的格式需要，我们需要用类别文件夹包含图片，尽管类别标签没有使用到
    parser.add_argument('--content_dataset', type=str, default='data/train2014', help='内容图像数据集路径 (仅训练模式)')
    parser.add_argument('--content_weight', type=float, default=1., help='内容权重(仅训练模式)')
    parser.add_argument('--style_weight', type=float, default=15., help='风格权重(仅训练模式)')
    parser.add_argument('--model_save_path', type=str, default=f'./models/{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.pth', help='模型保存路径 (训练模式)')

    parser.add_argument('--pretrained_model_path', type=str, help='预训练模型加载路径(仅训练模式)')
    parser.add_argument('--epochs', type=int, default=20, help='训练周期数(训练模式)')
    parser.add_argument('--save_interval', type=int, default=120, help='保存时间间隔(秒)(训练模式)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率(训练模式)')
    parser.add_argument('--output_dir', type=str, default='./output/realtime_transfer', help='输出目录 (训练模式)')
    parser.add_argument('--log_dir', type=str, default='./runs/fast_style_transfer', help='tensorboard日志路径 (训练模式)')

    parser.add_argument('--model_path', type=str, default='./models/udnie.pth', help='模型加载路径(图像和视频模式)')
    parser.add_argument('--input_images_dir', type=str, help='输入图像根目录(仅图像模式)')
    parser.add_argument('--output_images_dir', type=str, default='./output/fast_style_transfer/image_generated.jpg',
                        help='输出图像根目录 (仅图像模式)')

    parser.add_argument('--video_input', type=str, default='data/maigua.mp4', help='输入视频路径 (仅视频模式)')
    parser.add_argument('--video_output', type=str, default='output/videos/maigua.mp4',
                        help='输出视频路径 (仅视频模式)')

    parser.add_argument('--batch_size', type=int, default=4, help='批次大小')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    # print(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # ----------------路径参数----------------

    # 内容特征层及loss加权系数
    content_layers = {'5': 0.5, '10': 0.5}  # 使用vgg的较浅层特征作为内容特征，保证生成图片内容结构相似性
    # 风格特征层及loss加权系数
    style_layers = {'0': 0.2, '5': 0.2, '10': 0.2, '19': 0.2, '28': 0.2}  # 使用vgg不同深度的风格特征，生成风格更加层次丰富

    transform = make_transform(size=args.image_size, normalize=True)  # 图像变换
    image_style = load_image(args.style_image, transform=transform).to(device)  # 风格图像
    vgg = VGG(content_layers, style_layers).to(device)  # 特征提取网络，只用来提取特征，不进行训练
    model = TransNet(input_size=args.image_size).to(device)  # 内容生成网络，用于生成风格图片，进行训练

    if args.mode != 'train' and getattr(args, 'model_path'):
        if not os.path.exists(args.model_path):
            raise FileNotFoundError(f'{args.model_path}不存在！')
        model.load_state_dict(torch.load(args.model_path))
    elif args.mode == 'train' and getattr(args, 'pretrained_model_path') and os.path.exists(args.pretrained_model_path):
        if not os.path.exists(args.pretrained_model_path):
            raise FileNotFoundError(f'{args.pretrained_model_path}不存在！')
        model.load_state_dict(torch.load(args.pretrained_model_path))

    if args.mode == 'train':
        # 训练模式
        # 使用大规模内容图像数据训练快速图像风格迁移网络，比如COCO2017数据集
        train(model, vgg, args.learning_rate, args.epochs, args.batch_size, args.style_weight, args.content_weight,
              style_layers,
              content_layers, device, transform, image_style, args.content_dataset, args.model_save_path,
              args.output_dir,
              log_dir=args.log_dir,
              save_interval=timedelta(seconds=args.save_interval))

    elif args.mode == 'image':
        # 使用训练好的风格迁移模型演示批量处理图片
        if not os.path.exists(args.output_images_dir):
            os.makedirs(args.output_images_dir)
        for filename in tqdm(os.listdir(args.input_images_dir), desc='Processing Images'):
            try:
                filepath = os.path.join(args.input_images_dir, filename)

                images_generated = process_images([Image.open(filepath)], transform, model, device)
                images_generated[0].save(os.path.join(args.output_images_dir, filename))
            except Exception as e:
                pass

    elif args.mode == 'video':
        # 视频处理模式
        process_video(args.video_input, args.video_output, transform, model, device,
                      batch_size=args.batch_size)

    else:
        raise ValueError("未知的运行模式")

