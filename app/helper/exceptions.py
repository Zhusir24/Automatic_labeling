"""
自定义异常类定义
"""


class AutoLabelingError(Exception):
    """自动标注系统基础异常类"""
    pass


class ConfigError(AutoLabelingError):
    """配置相关异常"""
    pass


class ModelError(AutoLabelingError):
    """模型相关异常"""
    pass


class ImageError(AutoLabelingError):
    """图片处理相关异常"""
    pass


class ValidationError(AutoLabelingError):
    """输入验证异常"""
    pass


class FileOperationError(AutoLabelingError):
    """文件操作异常"""
    pass


class ModelInitializationError(ModelError):
    """模型初始化失败异常"""
    pass


class ModelPredictionError(ModelError):
    """模型预测失败异常"""
    pass


class ImageNotFoundError(ImageError):
    """图片文件未找到异常"""
    pass


class ImageFormatError(ImageError):
    """不支持的图片格式异常"""
    pass


class ConfigFileNotFoundError(ConfigError):
    """配置文件未找到异常"""
    pass


class ConfigParseError(ConfigError):
    """配置文件解析错误异常"""
    pass


class InvalidPathError(ValidationError):
    """无效路径异常"""
    pass


class InvalidParameterError(ValidationError):
    """无效参数异常"""
    pass 