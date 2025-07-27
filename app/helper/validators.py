"""
输入验证模块
"""
import os
from typing import List, Union, Optional
from pathlib import Path

from .exceptions import (
    InvalidPathError, 
    InvalidParameterError, 
    ImageNotFoundError,
    ImageFormatError
)


class Validator:
    """输入验证器类"""
    
    @staticmethod
    def validate_confidence(conf: float) -> float:
        """验证置信度参数
        
        Args:
            conf: 置信度值
            
        Returns:
            float: 验证后的置信度
            
        Raises:
            InvalidParameterError: 置信度不在有效范围内
        """
        if not isinstance(conf, (int, float)):
            raise InvalidParameterError(f"置信度必须是数字类型，当前类型: {type(conf)}")
        
        if not 0.0 <= conf <= 1.0:
            raise InvalidParameterError(f"置信度必须在0.0-1.0范围内，当前值: {conf}")
        
        return float(conf)
    
    @staticmethod
    def validate_file_path(file_path: Union[str, Path]) -> Path:
        """验证文件路径
        
        Args:
            file_path: 文件路径
            
        Returns:
            Path: 验证后的路径对象
            
        Raises:
            InvalidPathError: 路径无效
        """
        if not file_path:
            raise InvalidPathError("文件路径不能为空")
        
        path = Path(file_path)
        
        if not path.exists():
            raise InvalidPathError(f"文件不存在: {path}")
        
        if not path.is_file():
            raise InvalidPathError(f"路径不是文件: {path}")
        
        return path
    
    @staticmethod
    def validate_directory_path(dir_path: Union[str, Path]) -> Path:
        """验证目录路径
        
        Args:
            dir_path: 目录路径
            
        Returns:
            Path: 验证后的路径对象
            
        Raises:
            InvalidPathError: 路径无效
        """
        if not dir_path:
            raise InvalidPathError("目录路径不能为空")
        
        path = Path(dir_path)
        
        if not path.exists():
            raise InvalidPathError(f"目录不存在: {path}")
        
        if not path.is_dir():
            raise InvalidPathError(f"路径不是目录: {path}")
        
        return path
    
    @staticmethod
    def validate_image_extensions(extensions: Union[str, List[str]]) -> set:
        """验证图片扩展名
        
        Args:
            extensions: 扩展名字符串或列表
            
        Returns:
            set: 验证后的扩展名集合
            
        Raises:
            InvalidParameterError: 扩展名格式无效
        """
        if isinstance(extensions, str):
            ext_list = extensions.split()
        elif isinstance(extensions, (list, tuple, set)):
            ext_list = list(extensions)
        else:
            raise InvalidParameterError(f"扩展名必须是字符串或列表类型，当前类型: {type(extensions)}")
        
        validated_extensions = set()
        for ext in ext_list:
            ext = ext.strip().lower()
            if not ext:
                continue
            if not ext.startswith('.'):
                ext = '.' + ext
            validated_extensions.add(ext)
        
        if not validated_extensions:
            raise InvalidParameterError("至少需要指定一个有效的图片扩展名")
        
        return validated_extensions
    
    @staticmethod
    def validate_image_file(file_path: Union[str, Path], allowed_extensions: set) -> Path:
        """验证图片文件
        
        Args:
            file_path: 图片文件路径
            allowed_extensions: 允许的扩展名集合
            
        Returns:
            Path: 验证后的路径对象
            
        Raises:
            ImageNotFoundError: 图片文件不存在
            ImageFormatError: 图片格式不支持
        """
        path = Path(file_path)
        
        if not path.exists():
            raise ImageNotFoundError(f"图片文件不存在: {path}")
        
        if not path.is_file():
            raise ImageNotFoundError(f"路径不是文件: {path}")
        
        ext = path.suffix.lower()
        if ext not in allowed_extensions:
            raise ImageFormatError(
                f"不支持的图片格式: {ext}，支持的格式: {sorted(allowed_extensions)}"
            )
        
        return path
    
    @staticmethod
    def validate_prompts(prompts: Union[str, List[str]]) -> List[str]:
        """验证提示词
        
        Args:
            prompts: 提示词字符串或列表
            
        Returns:
            List[str]: 验证后的提示词列表
            
        Raises:
            InvalidParameterError: 提示词格式无效
        """
        if isinstance(prompts, str):
            if not prompts.strip():
                raise InvalidParameterError("提示词不能为空")
            # 按逗号分割并清理空白
            prompt_list = [item.strip() for item in prompts.split(",") if item.strip()]
        elif isinstance(prompts, (list, tuple)):
            prompt_list = [str(item).strip() for item in prompts if str(item).strip()]
        else:
            raise InvalidParameterError(f"提示词必须是字符串或列表类型，当前类型: {type(prompts)}")
        
        if not prompt_list:
            raise InvalidParameterError("至少需要提供一个有效的提示词")
        
        # 验证每个提示词不为空且不包含特殊字符
        for prompt in prompt_list:
            if not prompt or len(prompt.strip()) == 0:
                raise InvalidParameterError("提示词不能为空字符串")
            if any(char in prompt for char in ['/', '\\', ':', '*', '?', '"', '<', '>', '|']):
                raise InvalidParameterError(f"提示词包含非法字符: {prompt}")
        
        return prompt_list
    
    @staticmethod
    def validate_model_name(model_name: str, valid_models: List[str]) -> str:
        """验证模型名称
        
        Args:
            model_name: 模型名称
            valid_models: 有效模型列表
            
        Returns:
            str: 验证后的模型名称
            
        Raises:
            InvalidParameterError: 模型名称无效
        """
        if not isinstance(model_name, str):
            raise InvalidParameterError(f"模型名称必须是字符串类型，当前类型: {type(model_name)}")
        
        if not model_name.strip():
            raise InvalidParameterError("模型名称不能为空")
        
        if model_name not in valid_models:
            raise InvalidParameterError(
                f"无效的模型名称: {model_name}，有效模型: {valid_models}"
            )
        
        return model_name 