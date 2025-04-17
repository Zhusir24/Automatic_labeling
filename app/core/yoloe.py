import os
from collections import defaultdict

from loguru import logger
from ultralytics import YOLOE

from app.helper import config


class Yoloe():
    def init_model(self, model_name:str, names:list):
        self.model = YOLOE(os.path.join(config.models_path, model_name))
        self.model.set_classes(names, self.model.get_text_pe(names))
        logger.success(f'{model_name}初始化成功.')

    def predict_image(self, images_path: list, conf: float, output_dir: str):
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        # 统计所有出现的类别
        class_counter = defaultdict(int)

        # 先处理所有图片，统计类别
        all_results = []
        for image_path in images_path:
            results = self.model.predict(image_path, conf=conf)
            all_results.append((image_path, results))

            # 统计当前图片的类别
            for box in results[0].boxes:
                cls_id = int(box.cls.item())
                class_counter[cls_id] += 1

        # 生成classes.txt文件
        classes_file = os.path.join(output_dir, 'classes.txt')
        with open(classes_file, 'w') as f:
            # 按照类别ID排序写入
            for cls_id in sorted(class_counter.keys()):
                class_name = results[0].names[cls_id]
                f.write(f"{class_name}\n")

        # 读取classes.txt中的类别顺序，确保一致性
        with open(classes_file, 'r') as f:
            class_list = [line.strip() for line in f.readlines()]

        # 创建类别到索引的映射（从0开始）
        class_to_idx = {class_name: idx for idx, class_name in enumerate(class_list)}

        # 为每张图片生成标注文件
        for image_path, results in all_results:
            # 获取图片文件名（不带扩展名）
            filename = os.path.splitext(os.path.basename(image_path))[0]
            annotation_file = os.path.join(output_dir, f"{filename}.txt")

            with open(annotation_file, 'w') as f:
                for box in results[0].boxes:
                    # 获取原始类别ID和名称
                    original_cls_id = int(box.cls.item())
                    original_class_name = results[0].names[original_cls_id]

                    # 获取在classes.txt中的新索引
                    new_cls_id = class_to_idx[original_class_name]

                    # 获取归一化的坐标 (xywhn)
                    xywhn = box.xywhn[0].tolist()
                    x_center, y_center, width, height = xywhn

                    # 写入YOLO格式的标注行
                    f.write(f"{new_cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

            logger.info(f'已生成标注文件: {annotation_file}')

        logger.success(f'所有标注文件已生成到目录: {output_dir}')