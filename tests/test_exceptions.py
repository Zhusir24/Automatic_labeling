"""
测试自定义异常类
"""
import pytest
from app.helper.exceptions import (
    AutoLabelingError,
    ConfigError,
    ModelError,
    ImageError,
    ValidationError,
    FileOperationError,
    ModelInitializationError,
    ModelPredictionError,
    ImageNotFoundError,
    ImageFormatError,
    ConfigFileNotFoundError,
    ConfigParseError,
    InvalidPathError,
    InvalidParameterError
)


class TestExceptionHierarchy:
    """测试异常继承层次"""
    
    def test_base_exception(self):
        """测试基础异常类"""
        exc = AutoLabelingError("测试错误")
        assert str(exc) == "测试错误"
        assert isinstance(exc, Exception)
    
    def test_config_error_inheritance(self):
        """测试配置错误继承"""
        exc = ConfigError("配置错误")
        assert isinstance(exc, AutoLabelingError)
        assert isinstance(exc, Exception)
    
    def test_model_error_inheritance(self):
        """测试模型错误继承"""
        exc = ModelError("模型错误")
        assert isinstance(exc, AutoLabelingError)
        assert isinstance(exc, Exception)
    
    def test_image_error_inheritance(self):
        """测试图片错误继承"""
        exc = ImageError("图片错误")
        assert isinstance(exc, AutoLabelingError)
        assert isinstance(exc, Exception)
    
    def test_validation_error_inheritance(self):
        """测试验证错误继承"""
        exc = ValidationError("验证错误")
        assert isinstance(exc, AutoLabelingError)
        assert isinstance(exc, Exception)
    
    def test_file_operation_error_inheritance(self):
        """测试文件操作错误继承"""
        exc = FileOperationError("文件操作错误")
        assert isinstance(exc, AutoLabelingError)
        assert isinstance(exc, Exception)


class TestSpecificExceptions:
    """测试具体异常类"""
    
    def test_model_initialization_error(self):
        """测试模型初始化错误"""
        exc = ModelInitializationError("初始化失败")
        assert isinstance(exc, ModelError)
        assert isinstance(exc, AutoLabelingError)
        assert str(exc) == "初始化失败"
    
    def test_model_prediction_error(self):
        """测试模型预测错误"""
        exc = ModelPredictionError("预测失败")
        assert isinstance(exc, ModelError)
        assert isinstance(exc, AutoLabelingError)
        assert str(exc) == "预测失败"
    
    def test_image_not_found_error(self):
        """测试图片未找到错误"""
        exc = ImageNotFoundError("图片不存在")
        assert isinstance(exc, ImageError)
        assert isinstance(exc, AutoLabelingError)
        assert str(exc) == "图片不存在"
    
    def test_image_format_error(self):
        """测试图片格式错误"""
        exc = ImageFormatError("格式不支持")
        assert isinstance(exc, ImageError)
        assert isinstance(exc, AutoLabelingError)
        assert str(exc) == "格式不支持"
    
    def test_config_file_not_found_error(self):
        """测试配置文件未找到错误"""
        exc = ConfigFileNotFoundError("配置文件不存在")
        assert isinstance(exc, ConfigError)
        assert isinstance(exc, AutoLabelingError)
        assert str(exc) == "配置文件不存在"
    
    def test_config_parse_error(self):
        """测试配置解析错误"""
        exc = ConfigParseError("解析失败")
        assert isinstance(exc, ConfigError)
        assert isinstance(exc, AutoLabelingError)
        assert str(exc) == "解析失败"
    
    def test_invalid_path_error(self):
        """测试无效路径错误"""
        exc = InvalidPathError("路径无效")
        assert isinstance(exc, ValidationError)
        assert isinstance(exc, AutoLabelingError)
        assert str(exc) == "路径无效"
    
    def test_invalid_parameter_error(self):
        """测试无效参数错误"""
        exc = InvalidParameterError("参数无效")
        assert isinstance(exc, ValidationError)
        assert isinstance(exc, AutoLabelingError)
        assert str(exc) == "参数无效"


class TestExceptionUsage:
    """测试异常使用场景"""
    
    def test_exception_with_none_message(self):
        """测试空消息异常"""
        exc = AutoLabelingError(None)
        assert str(exc) == "None"
    
    def test_exception_with_empty_message(self):
        """测试空字符串消息"""
        exc = AutoLabelingError("")
        assert str(exc) == ""
    
    def test_exception_chaining(self):
        """测试异常链"""
        try:
            try:
                raise ValueError("原始错误")
            except ValueError as e:
                raise ModelError("模型错误") from e
        except ModelError as exc:
            assert str(exc) == "模型错误"
            assert isinstance(exc.__cause__, ValueError)
            assert str(exc.__cause__) == "原始错误"
    
    def test_exception_with_multiple_args(self):
        """测试多参数异常"""
        exc = AutoLabelingError("错误", "详细信息", 123)
        # 异常的字符串表示会包含所有参数
        assert "错误" in str(exc)
    
    def test_exception_equality(self):
        """测试异常比较"""
        exc1 = AutoLabelingError("测试")
        exc2 = AutoLabelingError("测试")
        # 异常实例不相等，即使消息相同
        assert exc1 is not exc2
        assert str(exc1) == str(exc2) 