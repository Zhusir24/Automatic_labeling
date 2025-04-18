# AutoLabel: 基于YOLO的自动标注工具

**AutoLabel** 是一款基于YOLOE模型的自动标注工具，能够快速生成YOLO格式的目标检测标注文件。

## ✨ 功能特性
- 🚀 支持多种预训练模型
- 🔍 可调节置信度阈值过滤检测结果
- 📁 支持批量处理文件夹中的图片
- 📝 输出YOLO格式标注文件

## 🛠 安装指南
```bash
pip install -r requirements.txt
```

## 🚀 基础使用
```commandline
python main.py \
    --prompts "公交车,行人,汽车" \
    --images_folder_path ./images
```