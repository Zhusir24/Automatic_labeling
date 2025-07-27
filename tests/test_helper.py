"""
测试helper模块功能
"""
import pytest
from pathlib import Path
from unittest.mock import patch, Mock
from app.helper.helper import string_to_list, scan_image_files
from app.helper.exceptions import (
    InvalidParameterError,
    ImageNotFoundError,
    FileOperationError
)


class TestStringToList:
    """测试字符串转列表功能"""
    
    def test_valid_comma_separated_string(self):
        """测试有效的逗号分隔字符串"""
        result = string_to_list("person,car,bus")
        assert result == ['person', 'car', 'bus']
    
    def test_string_with_spaces(self):
        """测试包含空格的字符串"""
        result = string_to_list("  person  ,  car  ,  bus  ")
        assert result == ['person', 'car', 'bus']
    
    def test_single_item_string(self):
        """测试单个项目的字符串"""
        result = string_to_list("person")
        assert result == ['person']
    
    def test_empty_string(self):
        """测试空字符串"""
        with pytest.raises(InvalidParameterError, match="提示词转换失败"):
            string_to_list("")
    
    def test_whitespace_only_string(self):
        """测试只有空白的字符串"""
        with pytest.raises(InvalidParameterError, match="提示词转换失败"):
            string_to_list("   ")
    
    def test_string_with_empty_items(self):
        """测试包含空项目的字符串"""
        result = string_to_list("person,,car,  ,bus")
        assert result == ['person', 'car', 'bus']
    
    def test_invalid_input_type(self):
        """测试无效输入类型"""
        with pytest.raises(InvalidParameterError, match="提示词转换失败"):
            string_to_list(123)
        
        with pytest.raises(InvalidParameterError, match="提示词转换失败"):
            string_to_list(None)
        
        with pytest.raises(InvalidParameterError, match="提示词转换失败"):
            string_to_list(['list', 'input'])
    
    def test_special_characters(self):
        """测试特殊字符"""
        with pytest.raises(InvalidParameterError, match="提示词转换失败"):
            string_to_list("person/car")
        
        with pytest.raises(InvalidParameterError, match="提示词转换失败"):
            string_to_list("person\\car")


class TestScanImageFiles:
    """测试图片文件扫描功能"""
    
    def test_scan_valid_image_directory(self, sample_images_dir):
        """测试扫描有效图片目录"""
        result = scan_image_files(str(sample_images_dir))
        
        # 应该找到3个图片文件（.jpg, .png, .jpeg），排除.txt文件
        assert len(result) == 3
        
        # 检查返回的都是绝对路径
        for path in result:
            assert Path(path).is_absolute()
            assert Path(path).exists()
        
        # 检查文件扩展名
        extensions = {Path(path).suffix.lower() for path in result}
        assert extensions.issubset({'.jpg', '.png', '.jpeg'})
    
    def test_scan_directory_path_object(self, sample_images_dir):
        """测试传入Path对象"""
        result = scan_image_files(sample_images_dir)
        assert len(result) == 3
    
    def test_scan_empty_directory(self, empty_dir):
        """测试扫描空目录"""
        with pytest.raises(ImageNotFoundError, match="目录中没有文件"):
            scan_image_files(str(empty_dir))
    
    def test_scan_nonexistent_directory(self):
        """测试扫描不存在的目录"""
        with pytest.raises(FileOperationError, match="扫描图片文件失败"):
            scan_image_files("/nonexistent/directory")
    
    def test_scan_file_instead_of_directory(self, temp_dir):
        """测试传入文件而非目录"""
        test_file = temp_dir / "test.txt"
        test_file.write_text("test")
        
        with pytest.raises(FileOperationError, match="扫描图片文件失败"):
            scan_image_files(str(test_file))
    
    def test_scan_directory_no_images(self, temp_dir):
        """测试扫描没有图片的目录"""
        no_images_dir = temp_dir / "no_images"
        no_images_dir.mkdir()
        
        # 创建一些非图片文件
        (no_images_dir / "document.txt").write_text("text")
        (no_images_dir / "data.csv").write_text("csv")
        (no_images_dir / "script.py").write_text("python")
        
        with pytest.raises(ImageNotFoundError, match="目录中没有支持的图片文件"):
            scan_image_files(str(no_images_dir))
    
    def test_scan_directory_with_subdirectories(self, temp_dir):
        """测试扫描包含子目录的目录"""
        root_dir = temp_dir / "root"
        root_dir.mkdir()
        
        # 在根目录创建图片
        (root_dir / "root_image.jpg").touch()
        
        # 创建子目录和图片
        sub_dir = root_dir / "subdir"
        sub_dir.mkdir()
        (sub_dir / "sub_image.png").touch()
        
        # 创建更深层的子目录
        deep_dir = sub_dir / "deep"
        deep_dir.mkdir()
        (deep_dir / "deep_image.jpeg").touch()
        
        result = scan_image_files(str(root_dir))
        
        # 应该找到所有3个图片文件
        assert len(result) == 3
        
        # 检查路径
        paths = {Path(path).name for path in result}
        assert paths == {"root_image.jpg", "sub_image.png", "deep_image.jpeg"}
    
    def test_scan_directory_permission_error(self, temp_dir, monkeypatch):
        """测试扫描权限不足的目录"""
        # 模拟权限错误
        def mock_walk(path):
            raise PermissionError("Permission denied")
        
        with monkeypatch.context() as m:
            m.setattr("os.walk", mock_walk)
            
            with pytest.raises(FileOperationError, match="访问目录权限不足"):
                scan_image_files(str(temp_dir))
    
    def test_scan_directory_os_error(self, temp_dir, monkeypatch):
        """测试扫描时的OS错误"""
        # 模拟OS错误
        def mock_walk(path):
            raise OSError("Device not ready")
        
        with monkeypatch.context() as m:
            m.setattr("os.walk", mock_walk)
            
            with pytest.raises(FileOperationError, match="扫描目录时发生错误"):
                scan_image_files(str(temp_dir))
    
    def test_scan_directory_unexpected_error(self, temp_dir, monkeypatch):
        """测试扫描时的意外错误"""
        # 模拟意外错误
        def mock_walk(path):
            raise RuntimeError("Unexpected error")
        
        with monkeypatch.context() as m:
            m.setattr("os.walk", mock_walk)
            
            with pytest.raises(FileOperationError, match="扫描图片文件失败"):
                scan_image_files(str(temp_dir))
    
    def test_scan_directory_with_different_extensions(self, temp_dir):
        """测试扫描不同扩展名的图片"""
        images_dir = temp_dir / "mixed_images"
        images_dir.mkdir()
        
        # 创建不同格式的图片文件
        supported_files = [
            "image1.jpg", "image2.JPG", "image3.jpeg", 
            "image4.JPEG", "image5.png", "image6.PNG"
        ]
        
        unsupported_files = [
            "image7.bmp", "image8.gif", "image9.tiff", 
            "image10.webp", "document.pdf", "data.txt"
        ]
        
        for filename in supported_files + unsupported_files:
            (images_dir / filename).touch()
        
        result = scan_image_files(str(images_dir))
        
        # 应该只找到支持的图片格式
        assert len(result) == len(supported_files)
        
        # 检查所有返回的文件都是支持的格式
        found_names = {Path(path).name.lower() for path in result}
        expected_names = {name.lower() for name in supported_files}
        assert found_names == expected_names
    
    @patch('app.helper.helper.logger')
    def test_scan_directory_logging(self, mock_logger, sample_images_dir):
        """测试扫描时的日志记录"""
        result = scan_image_files(str(sample_images_dir))
        
        # 验证调用了info日志
        mock_logger.info.assert_called_once()
        log_call = mock_logger.info.call_args[0][0]
        assert "总共扫描到" in log_call
        assert "张合法图片" in log_call
        assert str(len(result)) in log_call
    
    def test_scan_directory_empty_string_path(self):
        """测试空字符串路径"""
        with pytest.raises(FileOperationError, match="扫描图片文件失败"):
            scan_image_files("")
    
    def test_scan_directory_none_path(self):
        """测试None路径"""
        with pytest.raises(FileOperationError, match="扫描图片文件失败"):
            scan_image_files(None)


class TestHelperIntegration:
    """测试helper模块的集成功能"""
    
    def test_string_to_list_and_scan_integration(self, sample_images_dir):
        """测试字符串转列表和图片扫描的集成"""
        # 首先测试提示词解析
        prompts_str = "person,car,bus"
        prompts = string_to_list(prompts_str)
        assert len(prompts) == 3
        
        # 然后测试图片扫描
        images = scan_image_files(str(sample_images_dir))
        assert len(images) == 3
        
        # 验证这两个功能可以正常配合使用
        assert isinstance(prompts, list)
        assert isinstance(images, list)
        assert all(isinstance(prompt, str) for prompt in prompts)
        assert all(isinstance(image, str) for image in images)
    
    def test_helper_functions_with_real_workflow(self, sample_images_dir):
        """测试helper函数在真实工作流中的使用"""
        # 模拟真实的使用场景
        user_input = "  dog  ,  cat  ,  person  "
        image_folder = str(sample_images_dir)
        
        # 解析用户输入
        targets = string_to_list(user_input)
        assert targets == ["dog", "cat", "person"]
        
        # 扫描图片文件
        image_files = scan_image_files(image_folder)
        assert len(image_files) > 0
        
        # 验证所有返回的路径都是有效的
        for image_path in image_files:
            assert Path(image_path).exists()
            assert Path(image_path).is_file()
            assert Path(image_path).suffix.lower() in {'.jpg', '.png', '.jpeg'} 