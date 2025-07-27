"""
测试配置管理模块
"""
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, Mock
import configparser

from app.helper.exceptions import ConfigFileNotFoundError, ConfigParseError


class TestConfigClass:
    """测试Config类"""
    
    def test_config_loading_success(self, config_dir, monkeypatch):
        """测试成功加载配置"""
        # 动态导入以使用新的配置路径
        with monkeypatch.context() as m:
            m.setattr("app.helper.config.config_file_path", config_dir / "config.ini")
            
            # 重新导入模块
            import importlib
            import app.helper.config
            importlib.reload(app.helper.config)
            
            from app.helper.config import Config
            
            config = Config()
            assert config.default_conf == 0.5
            assert config.default_model_name == "test-model.pt"
            assert config.default_annotation_format == "Yolo"
            assert config.default_image_extensions == {'.png', '.jpg', '.jpeg'}
            assert config.valid_models == ["test-model.pt", "another-model.pt"]
    
    def test_config_file_not_found(self, temp_dir, monkeypatch):
        """测试配置文件不存在"""
        with monkeypatch.context() as m:
            m.setattr("app.helper.config.config_file_path", temp_dir / "nonexistent.ini")
            
            from app.helper.config import Config
            
            with pytest.raises(ConfigFileNotFoundError, match="配置文件不存在"):
                Config()
    
    def test_config_missing_section(self, temp_dir, monkeypatch):
        """测试配置文件缺少必要section"""
        config_file = temp_dir / "invalid_config.ini"
        config_file.write_text("""[Default]
conf = 0.5
""")  # 缺少Models section
        
        with monkeypatch.context() as m:
            m.setattr("app.helper.config.config_file_path", config_file)
            
            from app.helper.config import Config
            
            with pytest.raises(ConfigParseError, match="配置文件缺少必要的section: Models"):
                Config()
    
    def test_config_invalid_confidence(self, temp_dir, monkeypatch):
        """测试无效的置信度配置"""
        config_file = temp_dir / "invalid_conf.ini"
        config_file.write_text("""[Default]
conf = invalid_value
model_name = test.pt
annotation_format = Yolo
image_extensions = .png .jpg

[Models]
valid_models = test.pt
""")
        
        with monkeypatch.context() as m:
            m.setattr("app.helper.config.config_file_path", config_file)
            
            from app.helper.config import Config
            
            config = Config()
            with pytest.raises(ConfigParseError, match="无效的置信度配置"):
                _ = config.default_conf
    
    def test_config_missing_option(self, temp_dir, monkeypatch):
        """测试配置文件缺少必要选项"""
        config_file = temp_dir / "missing_option.ini"
        config_file.write_text("""[Default]
conf = 0.5
# 缺少model_name

[Models]
valid_models = test.pt
""")
        
        with monkeypatch.context() as m:
            m.setattr("app.helper.config.config_file_path", config_file)
            
            from app.helper.config import Config
            
            config = Config()
            with pytest.raises(ConfigParseError, match="缺少默认模型名称配置"):
                _ = config.default_model_name
    
    def test_config_parse_error(self, temp_dir, monkeypatch):
        """测试配置文件解析错误"""
        config_file = temp_dir / "malformed.ini"
        config_file.write_text("""[Default
conf = 0.5
""")  # 格式错误
        
        with monkeypatch.context() as m:
            m.setattr("app.helper.config.config_file_path", config_file)
            
            from app.helper.config import Config
            
            with pytest.raises(ConfigParseError, match="配置文件解析错误"):
                Config()
    
    def test_config_with_fallback_values(self, temp_dir, monkeypatch):
        """测试使用fallback值的配置"""
        config_file = temp_dir / "minimal.ini"
        config_file.write_text("""[Default]
conf = 0.7
model_name = minimal.pt
annotation_format = Yolo
# 没有image_extensions，应该使用fallback值

[Models]
valid_models = minimal.pt
""")
        
        with monkeypatch.context() as m:
            m.setattr("app.helper.config.config_file_path", config_file)
            
            from app.helper.config import Config
            
            config = Config()
            # 应该使用fallback值
            assert config.default_image_extensions == {'.png', '.jpg', '.jpeg'}


class TestConfigPaths:
    """测试配置路径处理"""
    
    def test_project_root_detection(self):
        """测试项目根目录检测"""
        from app.helper.config import PROJECT_ROOT, APP_PATH
        
        # PROJECT_ROOT应该是项目的根目录
        assert PROJECT_ROOT.exists()
        assert (PROJECT_ROOT / "main.py").exists()
        
        # APP_PATH应该是app目录
        assert APP_PATH.exists()
        assert (APP_PATH / "core").exists()
        assert (APP_PATH / "helper").exists()
    
    def test_path_construction(self):
        """测试路径构造"""
        from app.helper.config import (
            conf_path, core_path, images_folder_path, 
            models_path, outputs_path, config_file_path
        )
        
        # 所有路径都应该是Path对象
        assert isinstance(conf_path, Path)
        assert isinstance(core_path, Path)
        assert isinstance(images_folder_path, Path)
        assert isinstance(models_path, Path)
        assert isinstance(outputs_path, Path)
        assert isinstance(config_file_path, Path)
        
        # 检查路径结构
        assert conf_path.name == "conf"
        assert core_path.name == "core"
        assert images_folder_path.name == "images"
        assert models_path.name == "models"
        assert outputs_path.name == "outputs"
        assert config_file_path.name == "config.ini"


class TestConfigModuleImport:
    """测试配置模块导入和全局变量"""
    
    def test_successful_config_import(self, config_dir, models_dir, monkeypatch):
        """测试成功导入配置"""
        # 设置临时路径
        with monkeypatch.context() as m:
            m.setattr("app.helper.config.PROJECT_ROOT", config_dir.parent)
            m.setattr("app.helper.config.APP_PATH", config_dir.parent / "app")
            m.setattr("app.helper.config.conf_path", config_dir)
            m.setattr("app.helper.config.models_path", models_dir)
            m.setattr("app.helper.config.config_file_path", config_dir / "config.ini")
            
            # 重新导入模块
            import importlib
            import app.helper.config
            importlib.reload(app.helper.config)
            
            # 检查全局变量
            assert app.helper.config.default_conf == 0.5
            assert app.helper.config.default_model_name == "test-model.pt"
            assert app.helper.config.default_annotation_format == "Yolo"
            assert app.helper.config.default_image_extensions == {'.png', '.jpg', '.jpeg'}
            assert app.helper.config.valid_models == ["test-model.pt", "another-model.pt"]
    
    def test_config_import_with_fallback(self, temp_dir, monkeypatch):
        """测试配置导入失败时的fallback"""
        # 设置不存在的配置文件路径
        with monkeypatch.context() as m:
            m.setattr("app.helper.config.config_file_path", temp_dir / "nonexistent.ini")
            
            # 模拟warnings.warn
            with patch("warnings.warn") as mock_warn:
                # 重新导入模块
                import importlib
                import app.helper.config
                importlib.reload(app.helper.config)
                
                # 应该调用了警告
                mock_warn.assert_called_once()
                
                # 应该使用默认值
                assert app.helper.config.default_conf == 0.5
                assert app.helper.config.default_model_name == "yoloe-11l-seg.pt"
                assert app.helper.config.default_annotation_format == "Yolo"
                assert app.helper.config.default_image_extensions == {'.png', '.jpg', '.jpeg'}
                assert app.helper.config.valid_models == ["yoloe-11l-seg.pt", "yoloe-11m-seg.pt", "yoloe-11s-seg.pt"]


class TestConfigValidation:
    """测试配置验证"""
    
    def test_confidence_validation(self, config_dir, monkeypatch):
        """测试置信度验证"""
        # 创建置信度超出范围的配置
        config_file = config_dir / "config.ini"
        config_file.write_text("""[Default]
conf = 2.0
model_name = test.pt
annotation_format = Yolo
image_extensions = .png .jpg

[Models]
valid_models = test.pt
""")
        
        with monkeypatch.context() as m:
            m.setattr("app.helper.config.config_file_path", config_file)
            
            from app.helper.config import Config
            
            config = Config()
            with pytest.raises(ConfigParseError, match="无效的置信度配置"):
                _ = config.default_conf
    
    def test_image_extensions_validation(self, config_dir, monkeypatch):
        """测试图片扩展名验证"""
        # 创建包含空扩展名的配置
        config_file = config_dir / "config.ini"
        config_file.write_text("""[Default]
conf = 0.5
model_name = test.pt
annotation_format = Yolo
image_extensions = 

[Models]
valid_models = test.pt
""")
        
        with monkeypatch.context() as m:
            m.setattr("app.helper.config.config_file_path", config_file)
            
            from app.helper.config import Config
            
            config = Config()
            with pytest.raises(ConfigParseError, match="图片扩展名配置错误"):
                _ = config.default_image_extensions
    
    def test_empty_valid_models(self, config_dir, monkeypatch):
        """测试空的有效模型列表"""
        config_file = config_dir / "config.ini"
        config_file.write_text("""[Default]
conf = 0.5
model_name = test.pt
annotation_format = Yolo
image_extensions = .png .jpg

[Models]
valid_models = 
""")
        
        with monkeypatch.context() as m:
            m.setattr("app.helper.config.config_file_path", config_file)
            
            from app.helper.config import Config
            
            config = Config()
            with pytest.raises(ConfigParseError, match="有效模型列表不能为空"):
                _ = config.valid_models 