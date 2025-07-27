"""
配置管理模块 - 修复了路径问题和异常处理
"""
import configparser
import os
from pathlib import Path
from typing import Set, List

from .exceptions import ConfigFileNotFoundError, ConfigParseError
from .validators import Validator

# 使用__file__获取项目根目录，而不是依赖当前工作目录
PROJECT_ROOT = Path(__file__).parent.parent.parent
APP_PATH = PROJECT_ROOT / 'app'

# 定义所有路径
conf_path = APP_PATH / 'conf'
core_path = APP_PATH / 'core'
images_folder_path = APP_PATH / 'images'
models_path = APP_PATH / 'models'
outputs_path = PROJECT_ROOT / 'outputs'

# 修复：直接使用conf_path而不是重复join
config_file_path = conf_path / 'config.ini'


class Config:
    """配置管理类"""
    
    def __init__(self):
        self._config = configparser.ConfigParser()
        self._load_config()
    
    def _load_config(self) -> None:
        """加载配置文件，添加异常处理"""
        if not config_file_path.exists():
            raise ConfigFileNotFoundError(f"配置文件不存在: {config_file_path}")
        
        try:
            self._config.read(config_file_path, encoding='utf-8')
            
            # 验证必要的section存在
            required_sections = ['Default', 'Models']
            for section in required_sections:
                if not self._config.has_section(section):
                    raise ConfigParseError(f"配置文件缺少必要的section: {section}")
            
        except configparser.Error as e:
            raise ConfigParseError(f"配置文件解析错误: {e}")
        except Exception as e:
            raise ConfigParseError(f"读取配置文件时发生错误: {e}")
    
    @property
    def default_conf(self) -> float:
        """获取默认置信度，添加验证"""
        try:
            conf = self._config.getfloat('Default', 'conf')
            return Validator.validate_confidence(conf)
        except (configparser.NoOptionError, ValueError) as e:
            raise ConfigParseError(f"无效的置信度配置: {e}")
    
    @property
    def default_model_name(self) -> str:
        """获取默认模型名称"""
        try:
            return self._config.get('Default', 'model_name')
        except configparser.NoOptionError as e:
            raise ConfigParseError(f"缺少默认模型名称配置: {e}")
    
    @property
    def default_annotation_format(self) -> str:
        """获取默认标注格式"""
        try:
            return self._config.get('Default', 'annotation_format')
        except configparser.NoOptionError as e:
            raise ConfigParseError(f"缺少默认标注格式配置: {e}")
    
    @property
    def default_image_extensions(self) -> Set[str]:
        """获取默认图片扩展名，添加验证"""
        try:
            extensions_str = self._config.get('Default', 'image_extensions', fallback='.png .jpg .jpeg')
            return Validator.validate_image_extensions(extensions_str)
        except Exception as e:
            raise ConfigParseError(f"图片扩展名配置错误: {e}")
    
    @property
    def valid_models(self) -> List[str]:
        """获取有效模型列表"""
        try:
            models_str = self._config.get('Models', 'valid_models')
            models = models_str.split()
            if not models:
                raise ConfigParseError("有效模型列表不能为空")
            return models
        except configparser.NoOptionError as e:
            raise ConfigParseError(f"缺少有效模型列表配置: {e}")


# 创建全局配置实例
try:
    _config_instance = Config()
    
    # 导出配置属性（保持向后兼容）
    default_conf = _config_instance.default_conf
    default_model_name = _config_instance.default_model_name
    default_annotation_format = _config_instance.default_annotation_format
    default_image_extensions = _config_instance.default_image_extensions
    valid_models = _config_instance.valid_models
    
except Exception as e:
    # 如果配置加载失败，提供默认值但记录错误
    import warnings
    warnings.warn(f"配置加载失败，使用默认配置: {e}")
    
    default_conf = 0.5
    default_model_name = "yoloe-11l-seg.pt"
    default_annotation_format = "Yolo"
    default_image_extensions = {'.png', '.jpg', '.jpeg'}
    valid_models = ["yoloe-11l-seg.pt", "yoloe-11m-seg.pt", "yoloe-11s-seg.pt"]