"""
YOLO模型核心模块 - 修复了异常处理和空指针问题
"""
import os
from collections import defaultdict
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path

from loguru import logger
from ultralytics import YOLOE

from ..helper import config
from ..helper.exceptions import (
    ModelInitializationError,
    ModelPredictionError,
    FileOperationError,
    ImageNotFoundError
)
from ..helper.validators import Validator


class Yoloe:
    """YOLO模型封装类"""
    
    def __init__(self):
        self.model: Optional[YOLOE] = None
        self.model_name: Optional[str] = None
        self.class_names: Optional[List[str]] = None
        self.is_initialized: bool = False
    
    def init_model(self, model_name: str, names: List[str]) -> bool:
        """
        初始化YOLO模型
        
        Args:
            model_name: 模型文件名
            names: 类别名称列表
            
        Returns:
            bool: 初始化是否成功
            
        Raises:
            ModelInitializationError: 模型初始化失败
        """
        try:
            # 验证输入参数
            validated_names = Validator.validate_prompts(names)
            validated_model_name = Validator.validate_model_name(model_name, config.valid_models)
            
            # 检查模型文件是否存在
            model_path = Path(config.models_path) / validated_model_name
            if not model_path.exists():
                raise ModelInitializationError(f"模型文件不存在: {model_path}")
            
            # 初始化模型
            logger.info(f"正在加载模型: {model_path}")
            self.model = YOLOE(str(model_path))
            
            # 设置类别
            logger.info(f"设置模型类别: {validated_names}")
            self.model.set_classes(validated_names, self.model.get_text_pe(validated_names))
            
            # 记录初始化信息
            self.model_name = validated_model_name
            self.class_names = validated_names
            self.is_initialized = True
            
            logger.success(f'{validated_model_name} 初始化成功，类别数量: {len(validated_names)}')
            return True
            
        except Exception as e:
            self.is_initialized = False
            error_msg = f"模型初始化失败 - 模型: {model_name}, 错误: {e}"
            logger.error(error_msg)
            raise ModelInitializationError(error_msg)
    
    def _validate_model_ready(self) -> None:
        """验证模型是否已准备就绪"""
        if not self.is_initialized or self.model is None:
            raise ModelPredictionError("模型未初始化，请先调用 init_model 方法")

    def predict_image(self, images_path: List[str], conf: float, output_dir: str) -> Dict[str, Any]:
        """
        对图片进行预测并生成标注文件
        
        Args:
            images_path: 图片路径列表
            conf: 置信度阈值
            output_dir: 输出目录
            
        Returns:
            Dict[str, Any]: 预测结果统计信息
            
        Raises:
            ModelPredictionError: 预测过程中发生错误
            FileOperationError: 文件操作失败
        """
        try:
            # 验证模型状态
            self._validate_model_ready()
            
            # 验证输入参数
            validated_conf = Validator.validate_confidence(conf)
            output_path = Path(output_dir)
            
            # 确保输出目录存在
            try:
                output_path.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                raise FileOperationError(f"创建输出目录失败: {output_path}, 错误: {e}")
            
            if not images_path:
                raise ModelPredictionError("图片路径列表不能为空")
            
            # 验证所有图片文件存在
            validated_images = []
            for img_path in images_path:
                try:
                    validated_path = Validator.validate_image_file(img_path, config.default_image_extensions)
                    validated_images.append(str(validated_path))
                except Exception as e:
                    logger.warning(f"跳过无效图片: {img_path}, 原因: {e}")
            
            if not validated_images:
                raise ModelPredictionError("没有找到有效的图片文件")
            
            logger.info(f"开始预测 {len(validated_images)} 张图片，置信度阈值: {validated_conf}")
            
            # 统计所有出现的类别和预测结果
            class_counter = defaultdict(int)
            all_results = []
            successful_predictions = 0
            failed_predictions = 0
            
            # 先处理所有图片，统计类别
            for image_path in validated_images:
                try:
                    results = self.model.predict(image_path, conf=validated_conf)
                    
                    # 验证预测结果
                    if not results or len(results) == 0:
                        logger.warning(f"图片预测无结果: {image_path}")
                        failed_predictions += 1
                        continue
                    
                    result = results[0]
                    if not hasattr(result, 'boxes') or result.boxes is None:
                        logger.warning(f"图片未检测到目标: {image_path}")
                        all_results.append((image_path, result, []))  # 添加空的检测结果
                        successful_predictions += 1
                        continue
                    
                    # 统计当前图片的类别
                    detections = []
                    for box in result.boxes:
                        if box.cls is not None:
                            cls_id = int(box.cls.item())
                            class_counter[cls_id] += 1
                            detections.append(box)
                    
                    all_results.append((image_path, result, detections))
                    successful_predictions += 1
                    
                except Exception as e:
                    logger.error(f"预测图片失败: {image_path}, 错误: {e}")
                    failed_predictions += 1
                    continue
            
            if not class_counter:
                logger.warning("所有图片都未检测到目标，生成空的标注文件")
            
            # 生成类别映射（使用模型的names而不是预测结果的names）
            class_to_idx = self._generate_class_mapping(class_counter, output_path)
            
            # 为每张图片生成标注文件
            annotation_files_created = 0
            for image_path, result, detections in all_results:
                try:
                    filename = Path(image_path).stem
                    annotation_file = output_path / f"{filename}.txt"
                    
                    self._write_annotation_file(annotation_file, detections, result, class_to_idx)
                    annotation_files_created += 1
                    logger.debug(f'已生成标注文件: {annotation_file}')
                    
                except Exception as e:
                    logger.error(f"生成标注文件失败: {image_path}, 错误: {e}")
                    continue
            
            # 生成预测统计信息
            stats = {
                'total_images': len(validated_images),
                'successful_predictions': successful_predictions,
                'failed_predictions': failed_predictions,
                'annotation_files_created': annotation_files_created,
                'classes_detected': len(class_counter),
                'total_detections': sum(class_counter.values()),
                'class_distribution': dict(class_counter)
            }
            
            logger.success(
                f'预测完成! 成功: {successful_predictions}, 失败: {failed_predictions}, '
                f'生成标注文件: {annotation_files_created}, 检测到 {len(class_counter)} 个类别'
            )
            logger.success(f'所有标注文件已生成到目录: {output_path}')
            
            return stats
            
        except (ModelPredictionError, FileOperationError):
            # 重新抛出已知异常
            raise
        except Exception as e:
            error_msg = f"预测过程中发生未知错误: {e}"
            logger.error(error_msg)
            raise ModelPredictionError(error_msg)
    
    def _generate_class_mapping(self, class_counter: defaultdict, output_path: Path) -> Dict[str, int]:
        """生成类别映射和classes.txt文件"""
        try:
            classes_file = output_path / 'classes.txt'
            class_to_idx = {}
            
            if class_counter:
                # 有检测到的类别，使用检测到的类别生成映射
                with open(classes_file, 'w', encoding='utf-8') as f:
                    for idx, cls_id in enumerate(sorted(class_counter.keys())):
                        # 使用模型的names属性获取类别名称
                        if self.model and hasattr(self.model.model, 'names'):
                            model_names = self.model.model.names
                            if isinstance(model_names, dict):
                                class_name = model_names.get(cls_id, f"class_{cls_id}")
                            elif isinstance(model_names, (list, tuple)) and cls_id < len(model_names):
                                class_name = model_names[cls_id]
                            else:
                                class_name = f"class_{cls_id}"
                        else:
                            class_name = self.class_names[cls_id] if cls_id < len(self.class_names) else f"class_{cls_id}"
                        
                        f.write(f"{class_name}\n")
                        class_to_idx[class_name] = idx
            else:
                # 没有检测到任何类别，使用配置的类别名称
                with open(classes_file, 'w', encoding='utf-8') as f:
                    for idx, class_name in enumerate(self.class_names or []):
                        f.write(f"{class_name}\n")
                        class_to_idx[class_name] = idx
            
            logger.info(f"生成类别文件: {classes_file}, 包含 {len(class_to_idx)} 个类别")
            return class_to_idx
            
        except Exception as e:
            raise FileOperationError(f"生成类别映射文件失败: {e}")
    
    def _write_annotation_file(self, annotation_file: Path, detections: List, result: Any, class_to_idx: Dict[str, int]) -> None:
        """写入单个图片的标注文件"""
        try:
            with open(annotation_file, 'w', encoding='utf-8') as f:
                for box in detections:
                    if box.cls is None or box.xywhn is None:
                        continue
                    
                    # 获取类别信息
                    original_cls_id = int(box.cls.item())
                    
                    # 获取类别名称
                    if hasattr(result, 'names') and original_cls_id in result.names:
                        original_class_name = result.names[original_cls_id]
                    elif self.model and hasattr(self.model.model, 'names'):
                        model_names = self.model.model.names
                        if isinstance(model_names, dict):
                            original_class_name = model_names.get(original_cls_id, f"class_{original_cls_id}")
                        elif isinstance(model_names, (list, tuple)) and original_cls_id < len(model_names):
                            original_class_name = model_names[original_cls_id]
                        else:
                            original_class_name = f"class_{original_cls_id}"
                    else:
                        original_class_name = self.class_names[original_cls_id] if original_cls_id < len(self.class_names) else f"class_{original_cls_id}"
                    
                    # 获取在classes.txt中的新索引
                    new_cls_id = class_to_idx.get(original_class_name, 0)
                    
                    # 获取归一化的坐标 (xywhn)
                    xywhn = box.xywhn[0].tolist()
                    x_center, y_center, width, height = xywhn
                    
                    # 写入YOLO格式的标注行
                    f.write(f"{new_cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        except Exception as e:
            raise FileOperationError(f"写入标注文件失败: {annotation_file}, 错误: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'is_initialized': self.is_initialized,
            'model_name': self.model_name,
            'class_names': self.class_names,
            'num_classes': len(self.class_names) if self.class_names else 0
        }