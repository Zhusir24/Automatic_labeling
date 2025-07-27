"""
自动标注程序主入口 - 修复了全局变量和参数处理问题
"""
import argparse
import sys
from typing import Optional

from loguru import logger

from app.core.yoloe import Yoloe
from app.helper import config, helper
from app.helper.exceptions import (
    AutoLabelingError,
    ModelInitializationError,
    ModelPredictionError,
    InvalidParameterError,
    ConfigError
)


def create_argument_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description='自动图片标注程序',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python main.py --prompts "bus,car,person" --conf 0.7
  python main.py --prompts "dog,cat" --images_folder_path "./custom_images" --output_folder "./results"
        """
    )
    
    parser.add_argument(
        '--model_name', 
        type=str, 
        default=config.default_model_name, 
        help=f'指定模型名称 (默认: {config.default_model_name})',
        choices=config.valid_models
    )
    
    parser.add_argument(
        '--prompts', 
        type=str, 
        help='检测目标的提示词，用逗号分隔，例如: "bus,car,person"', 
        required=True
    )
    
    parser.add_argument(
        '--conf', 
        type=float, 
        help=f'置信度阈值 (0.0-1.0，默认: {config.default_conf})', 
        required=False, 
        default=config.default_conf
    )
    
    parser.add_argument(
        '--images_folder_path', 
        type=str, 
        help=f'图片文件夹路径 (默认: {config.images_folder_path})', 
        required=False, 
        default=str(config.images_folder_path)
    )
    
    parser.add_argument(
        '--output_folder', 
        type=str, 
        help=f'输出文件夹路径 (默认: {config.outputs_path})', 
        required=False, 
        default=str(config.outputs_path)
    )
    
    # 保留annotation_format参数以保持向后兼容，但标记为未实现
    parser.add_argument(
        '--annotation_format', 
        type=str, 
        help='标注格式 (当前仅支持YOLO格式)', 
        required=False, 
        default=config.default_annotation_format,
        choices=['Yolo']
    )
    
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    """验证命令行参数"""
    try:
        # 验证置信度
        from app.helper.validators import Validator
        Validator.validate_confidence(args.conf)
        
        # 验证模型名称
        Validator.validate_model_name(args.model_name, config.valid_models)
        
        # 验证提示词
        Validator.validate_prompts(args.prompts)
        
        # 验证路径（在后续处理中再验证具体存在性）
        if not args.images_folder_path.strip():
            raise InvalidParameterError("图片文件夹路径不能为空")
            
        if not args.output_folder.strip():
            raise InvalidParameterError("输出文件夹路径不能为空")
            
    except Exception as e:
        logger.error(f"参数验证失败: {e}")
        raise InvalidParameterError(f"参数验证失败: {e}")


def run_automatic_labeling(args: argparse.Namespace) -> Optional[dict]:
    """运行自动标注流程"""
    yoloe = None
    try:
        # 解析提示词
        logger.info("正在解析提示词...")
        prompts = helper.string_to_list(input_str=args.prompts)
        
        # 扫描图片文件
        logger.info(f"正在扫描图片文件: {args.images_folder_path}")
        images_path = helper.scan_image_files(folder_path=args.images_folder_path)
        
        # 创建并初始化模型
        logger.info("正在初始化YOLO模型...")
        yoloe = Yoloe()
        success = yoloe.init_model(model_name=args.model_name, names=prompts)
        
        if not success:
            raise ModelInitializationError("模型初始化返回失败状态")
        
        # 记录配置信息
        logger.info(f'配置信息:')
        logger.info(f'  模型: {args.model_name}')
        logger.info(f'  提示词: {prompts}')
        logger.info(f'  图片路径: {args.images_folder_path}')
        logger.info(f'  置信度: {args.conf}')
        logger.info(f'  标注格式: {args.annotation_format}')
        logger.info(f'  输出路径: {args.output_folder}')
        logger.info(f'  找到图片数量: {len(images_path)}')
        
        # 执行预测
        logger.info("开始图片预测和标注生成...")
        stats = yoloe.predict_image(
            images_path=images_path, 
            conf=args.conf, 
            output_dir=args.output_folder
        )
        
        # 输出统计信息
        logger.success("=" * 50)
        logger.success("自动标注完成!")
        logger.success(f"总图片数: {stats['total_images']}")
        logger.success(f"成功预测: {stats['successful_predictions']}")
        logger.success(f"预测失败: {stats['failed_predictions']}")
        logger.success(f"生成标注文件: {stats['annotation_files_created']}")
        logger.success(f"检测到类别数: {stats['classes_detected']}")
        logger.success(f"总检测数量: {stats['total_detections']}")
        if stats['class_distribution']:
            logger.success("类别分布:")
            for class_id, count in stats['class_distribution'].items():
                logger.success(f"  类别 {class_id}: {count} 个检测")
        logger.success("=" * 50)
        
        return stats
        
    except KeyboardInterrupt:
        logger.warning("用户中断程序执行")
        return None
    except AutoLabelingError as e:
        logger.error(f"自动标注过程发生错误: {e}")
        raise
    except Exception as e:
        logger.error(f"程序执行过程中发生未知错误: {e}")
        raise AutoLabelingError(f"程序执行失败: {e}")


def main() -> int:
    """主函数"""
    try:
        # 设置日志格式
        logger.remove()
        logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level="INFO"
        )
        
        logger.info("自动标注程序启动")
        
        # 解析命令行参数
        parser = create_argument_parser()
        args = parser.parse_args()
        
        # 验证参数
        validate_arguments(args)
        
        # 运行自动标注
        stats = run_automatic_labeling(args)
        
        if stats is not None:
            logger.info("程序执行成功完成")
            return 0
        else:
            logger.warning("程序被用户中断")
            return 1
            
    except ConfigError as e:
        logger.error(f"配置错误: {e}")
        logger.error("请检查配置文件和模型文件是否正确")
        return 2
    except InvalidParameterError as e:
        logger.error(f"参数错误: {e}")
        logger.error("请使用 --help 查看正确的参数格式")
        return 3
    except ModelInitializationError as e:
        logger.error(f"模型初始化失败: {e}")
        logger.error("请检查模型文件是否存在且有效")
        return 4
    except ModelPredictionError as e:
        logger.error(f"模型预测失败: {e}")
        logger.error("请检查图片文件和模型配置")
        return 5
    except AutoLabelingError as e:
        logger.error(f"自动标注失败: {e}")
        return 6
    except Exception as e:
        logger.error(f"未知错误: {e}")
        logger.error("请联系开发者或查看详细日志")
        return 99


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)