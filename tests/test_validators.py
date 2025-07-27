"""
测试输入验证器
"""
import pytest
from pathlib import Path
from app.helper.validators import Validator
from app.helper.exceptions import (
    InvalidParameterError,
    InvalidPathError,
    ImageNotFoundError,
    ImageFormatError
)


class TestValidateConfidence:
    """测试置信度验证"""
    
    def test_valid_confidence_float(self):
        """测试有效的浮点数置信度"""
        assert Validator.validate_confidence(0.5) == 0.5
        assert Validator.validate_confidence(0.0) == 0.0
        assert Validator.validate_confidence(1.0) == 1.0
        assert Validator.validate_confidence(0.123) == 0.123
    
    def test_valid_confidence_int(self):
        """测试有效的整数置信度"""
        assert Validator.validate_confidence(0) == 0.0
        assert Validator.validate_confidence(1) == 1.0
    
    def test_invalid_confidence_type(self):
        """测试无效类型的置信度"""
        with pytest.raises(InvalidParameterError, match="置信度必须是数字类型"):
            Validator.validate_confidence("0.5")
        
        with pytest.raises(InvalidParameterError, match="置信度必须是数字类型"):
            Validator.validate_confidence(None)
        
        with pytest.raises(InvalidParameterError, match="置信度必须是数字类型"):
            Validator.validate_confidence([0.5])
    
    def test_invalid_confidence_range(self):
        """测试超出范围的置信度"""
        with pytest.raises(InvalidParameterError, match="置信度必须在0.0-1.0范围内"):
            Validator.validate_confidence(-0.1)
        
        with pytest.raises(InvalidParameterError, match="置信度必须在0.0-1.0范围内"):
            Validator.validate_confidence(1.1)
        
        with pytest.raises(InvalidParameterError, match="置信度必须在0.0-1.0范围内"):
            Validator.validate_confidence(100)


class TestValidateFilePath:
    """测试文件路径验证"""
    
    def test_valid_file_path(self, temp_dir):
        """测试有效文件路径"""
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")
        
        result = Validator.validate_file_path(str(test_file))
        assert result == test_file
        assert isinstance(result, Path)
    
    def test_valid_file_path_object(self, temp_dir):
        """测试传入Path对象"""
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")
        
        result = Validator.validate_file_path(test_file)
        assert result == test_file
    
    def test_empty_file_path(self):
        """测试空文件路径"""
        with pytest.raises(InvalidPathError, match="文件路径不能为空"):
            Validator.validate_file_path("")
        
        with pytest.raises(InvalidPathError, match="文件路径不能为空"):
            Validator.validate_file_path(None)
    
    def test_nonexistent_file(self):
        """测试不存在的文件"""
        with pytest.raises(InvalidPathError, match="文件不存在"):
            Validator.validate_file_path("/nonexistent/file.txt")
    
    def test_directory_instead_of_file(self, temp_dir):
        """测试传入目录而非文件"""
        with pytest.raises(InvalidPathError, match="路径不是文件"):
            Validator.validate_file_path(str(temp_dir))


class TestValidateDirectoryPath:
    """测试目录路径验证"""
    
    def test_valid_directory_path(self, temp_dir):
        """测试有效目录路径"""
        result = Validator.validate_directory_path(str(temp_dir))
        assert result == temp_dir
        assert isinstance(result, Path)
    
    def test_valid_directory_path_object(self, temp_dir):
        """测试传入Path对象"""
        result = Validator.validate_directory_path(temp_dir)
        assert result == temp_dir
    
    def test_empty_directory_path(self):
        """测试空目录路径"""
        with pytest.raises(InvalidPathError, match="目录路径不能为空"):
            Validator.validate_directory_path("")
    
    def test_nonexistent_directory(self):
        """测试不存在的目录"""
        with pytest.raises(InvalidPathError, match="目录不存在"):
            Validator.validate_directory_path("/nonexistent/directory")
    
    def test_file_instead_of_directory(self, temp_dir):
        """测试传入文件而非目录"""
        test_file = temp_dir / "test.txt"
        test_file.write_text("test")
        
        with pytest.raises(InvalidPathError, match="路径不是目录"):
            Validator.validate_directory_path(str(test_file))


class TestValidateImageExtensions:
    """测试图片扩展名验证"""
    
    def test_string_extensions(self):
        """测试字符串格式的扩展名"""
        result = Validator.validate_image_extensions(".png .jpg .jpeg")
        assert result == {'.png', '.jpg', '.jpeg'}
    
    def test_list_extensions(self):
        """测试列表格式的扩展名"""
        result = Validator.validate_image_extensions(['.png', '.jpg', '.jpeg'])
        assert result == {'.png', '.jpg', '.jpeg'}
    
    def test_extensions_without_dot(self):
        """测试没有点的扩展名"""
        result = Validator.validate_image_extensions("png jpg jpeg")
        assert result == {'.png', '.jpg', '.jpeg'}
    
    def test_mixed_case_extensions(self):
        """测试大小写混合的扩展名"""
        result = Validator.validate_image_extensions(".PNG .Jpg .JPEG")
        assert result == {'.png', '.jpg', '.jpeg'}
    
    def test_extensions_with_spaces(self):
        """测试有空格的扩展名"""
        result = Validator.validate_image_extensions("  .png   .jpg  ")
        assert result == {'.png', '.jpg'}
    
    def test_empty_extensions(self):
        """测试空扩展名"""
        with pytest.raises(InvalidParameterError, match="至少需要指定一个有效的图片扩展名"):
            Validator.validate_image_extensions("")
        
        with pytest.raises(InvalidParameterError, match="至少需要指定一个有效的图片扩展名"):
            Validator.validate_image_extensions([])
    
    def test_invalid_extensions_type(self):
        """测试无效类型的扩展名"""
        with pytest.raises(InvalidParameterError, match="扩展名必须是字符串或列表类型"):
            Validator.validate_image_extensions(123)


class TestValidateImageFile:
    """测试图片文件验证"""
    
    def test_valid_image_file(self, temp_dir):
        """测试有效图片文件"""
        image_file = temp_dir / "test.jpg"
        image_file.write_text("fake image content")
        
        allowed_extensions = {'.jpg', '.png', '.jpeg'}
        result = Validator.validate_image_file(str(image_file), allowed_extensions)
        assert result == image_file
    
    def test_nonexistent_image_file(self):
        """测试不存在的图片文件"""
        allowed_extensions = {'.jpg', '.png'}
        with pytest.raises(ImageNotFoundError, match="图片文件不存在"):
            Validator.validate_image_file("/nonexistent/image.jpg", allowed_extensions)
    
    def test_directory_instead_of_image(self, temp_dir):
        """测试传入目录而非图片"""
        allowed_extensions = {'.jpg', '.png'}
        with pytest.raises(ImageNotFoundError, match="路径不是文件"):
            Validator.validate_image_file(str(temp_dir), allowed_extensions)
    
    def test_unsupported_image_format(self, temp_dir):
        """测试不支持的图片格式"""
        image_file = temp_dir / "test.bmp"
        image_file.write_text("fake image content")
        
        allowed_extensions = {'.jpg', '.png', '.jpeg'}
        with pytest.raises(ImageFormatError, match="不支持的图片格式"):
            Validator.validate_image_file(str(image_file), allowed_extensions)


class TestValidatePrompts:
    """测试提示词验证"""
    
    def test_string_prompts(self):
        """测试字符串格式的提示词"""
        result = Validator.validate_prompts("person,car,bus")
        assert result == ['person', 'car', 'bus']
    
    def test_string_prompts_with_spaces(self):
        """测试有空格的字符串提示词"""
        result = Validator.validate_prompts("  person  ,  car  ,  bus  ")
        assert result == ['person', 'car', 'bus']
    
    def test_list_prompts(self):
        """测试列表格式的提示词"""
        result = Validator.validate_prompts(['person', 'car', 'bus'])
        assert result == ['person', 'car', 'bus']
    
    def test_tuple_prompts(self):
        """测试元组格式的提示词"""
        result = Validator.validate_prompts(('person', 'car', 'bus'))
        assert result == ['person', 'car', 'bus']
    
    def test_empty_prompts_string(self):
        """测试空字符串提示词"""
        with pytest.raises(InvalidParameterError, match="提示词不能为空"):
            Validator.validate_prompts("")
        
        with pytest.raises(InvalidParameterError, match="提示词不能为空"):
            Validator.validate_prompts("   ")
    
    def test_empty_prompts_list(self):
        """测试空列表提示词"""
        with pytest.raises(InvalidParameterError, match="至少需要提供一个有效的提示词"):
            Validator.validate_prompts([])
        
        with pytest.raises(InvalidParameterError, match="至少需要提供一个有效的提示词"):
            Validator.validate_prompts(['', '  ', ''])
    
    def test_invalid_prompts_type(self):
        """测试无效类型的提示词"""
        with pytest.raises(InvalidParameterError, match="提示词必须是字符串或列表类型"):
            Validator.validate_prompts(123)
    
    def test_prompts_with_illegal_characters(self):
        """测试包含非法字符的提示词"""
        illegal_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
        for char in illegal_chars:
            with pytest.raises(InvalidParameterError, match="提示词包含非法字符"):
                Validator.validate_prompts(f"person{char}")


class TestValidateModelName:
    """测试模型名称验证"""
    
    def test_valid_model_name(self):
        """测试有效模型名称"""
        valid_models = ['model1.pt', 'model2.pt']
        result = Validator.validate_model_name('model1.pt', valid_models)
        assert result == 'model1.pt'
    
    def test_invalid_model_name_type(self):
        """测试无效类型的模型名称"""
        valid_models = ['model1.pt']
        with pytest.raises(InvalidParameterError, match="模型名称必须是字符串类型"):
            Validator.validate_model_name(123, valid_models)
    
    def test_empty_model_name(self):
        """测试空模型名称"""
        valid_models = ['model1.pt']
        with pytest.raises(InvalidParameterError, match="模型名称不能为空"):
            Validator.validate_model_name("", valid_models)
        
        with pytest.raises(InvalidParameterError, match="模型名称不能为空"):
            Validator.validate_model_name("   ", valid_models)
    
    def test_invalid_model_name(self):
        """测试无效的模型名称"""
        valid_models = ['model1.pt', 'model2.pt']
        with pytest.raises(InvalidParameterError, match="无效的模型名称"):
            Validator.validate_model_name('invalid_model.pt', valid_models) 