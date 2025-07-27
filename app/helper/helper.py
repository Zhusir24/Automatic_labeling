"""
辅助函数模块 - 修复了错误处理策略和文档问题
"""
import os
from typing import List
from pathlib import Path

from loguru import logger
from tqdm import tqdm

from . import config
from .exceptions import (
    InvalidParameterError, 
    ImageNotFoundError, 
    FileOperationError
)
from .validators import Validator


def string_to_list(input_str: str) -> List[str]:
    """
    将输入字符串转换为列表
    
    Args:
        input_str (str): 输入字符串，例如 "bus,taxi,person"
        
    Returns:
        List[str]: 转换后的字符串列表
        
    Raises:
        InvalidParameterError: 输入参数无效
        
    Examples:
        >>> string_to_list("bus,taxi,person")
        ['bus', 'taxi', 'person']
        >>> string_to_list("  dog  ,  cat  ")
        ['dog', 'cat']
    """
    try:
        return Validator.validate_prompts(input_str)
    except Exception as e:
        logger.error(f"提示词转换失败: {e}")
        logger.info("示例输入格式: bus,taxi,person")
        raise InvalidParameterError(f"提示词转换失败: {e}")


def scan_image_files(folder_path: str) -> List[str]:
    """
    扫描文件夹中的图片文件并返回绝对路径列表
    
    Args:
        folder_path: 要扫描的文件夹路径
        
    Returns:
        List[str]: 图片文件的绝对路径列表
        
    Raises:
        ImageNotFoundError: 文件夹不存在、为空或权限不足
        FileOperationError: 文件系统操作失败
        
    Examples:
        >>> scan_image_files("./images")
        ['/path/to/image1.jpg', '/path/to/image2.png']
    """
    try:
        # 验证目录路径
        folder_path_obj = Validator.validate_directory_path(folder_path)
        
        # 获取图片扩展名配置
        image_extensions = config.default_image_extensions
        
        # 收集所有文件
        all_files = []
        try:
            for root, _, files in os.walk(folder_path_obj):
                for file in files:
                    all_files.append(Path(root) / file)
        except PermissionError as e:
            raise FileOperationError(f"访问目录权限不足: {folder_path_obj}, 错误: {e}")
        except OSError as e:
            raise FileOperationError(f"扫描目录时发生错误: {folder_path_obj}, 错误: {e}")
        
        if not all_files:
            raise ImageNotFoundError(f"目录中没有文件: {folder_path_obj}")
        
        # 筛选图片文件（带进度条）
        image_files = []
        for file_path in tqdm(all_files, desc="扫描图片", unit="file"):
            ext = file_path.suffix.lower()
            if ext in image_extensions:
                # 不需要再次调用abspath，因为os.walk已经返回绝对路径
                image_files.append(str(file_path))
        
        if not image_files:
            supported_formats = sorted(image_extensions)
            raise ImageNotFoundError(
                f"目录中没有支持的图片文件: {folder_path_obj}\n"
                f"支持的格式: {supported_formats}\n"
                f"找到的文件数: {len(all_files)}"
            )
        
        logger.info(f'总共扫描到{len(image_files)}张合法图片')
        return image_files
        
    except (ImageNotFoundError, FileOperationError):
        # 重新抛出已知异常
        raise
    except Exception as e:
        logger.error(f"扫描图片文件时发生未知错误: {e}")
        raise FileOperationError(f"扫描图片文件失败: {e}")