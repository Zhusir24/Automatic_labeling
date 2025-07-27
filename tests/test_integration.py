"""
集成测试 - 测试整个系统的端到端功能
"""
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, Mock
import subprocess
import sys

from app.helper.exceptions import AutoLabelingError


class TestEndToEndIntegration:
    """端到端集成测试"""
    
    @pytest.mark.integration
    def test_full_workflow_success(self, temp_dir, monkeypatch):
        """测试完整工作流程成功场景"""
        # 创建测试环境
        images_dir = temp_dir / "test_images"
        images_dir.mkdir()
        
        models_dir = temp_dir / "models"
        models_dir.mkdir()
        
        conf_dir = temp_dir / "conf"
        conf_dir.mkdir()
        
        output_dir = temp_dir / "output"
        
        # 创建测试图片
        (images_dir / "test1.jpg").touch()
        (images_dir / "test2.png").touch()
        
        # 创建测试模型文件
        (models_dir / "test-model.pt").touch()
        
        # 创建配置文件
        config_content = """[Default]
conf = 0.5
model_name = test-model.pt
annotation_format = Yolo
image_extensions = .png .jpg .jpeg

[Models]
valid_models = test-model.pt
"""
        (conf_dir / "config.ini").write_text(config_content)
        
        # 设置环境
        with monkeypatch.context() as m:
            # 修改配置路径
            m.setattr("app.helper.config.PROJECT_ROOT", temp_dir)
            m.setattr("app.helper.config.APP_PATH", temp_dir / "app")
            m.setattr("app.helper.config.conf_path", conf_dir)
            m.setattr("app.helper.config.models_path", models_dir)
            m.setattr("app.helper.config.images_folder_path", images_dir)
            m.setattr("app.helper.config.outputs_path", output_dir)
            m.setattr("app.helper.config.config_file_path", conf_dir / "config.ini")
            
            # 模拟YOLO模型
            with patch('app.core.yoloe.YOLOE') as mock_yoloe_class:
                # 创建模拟的YOLO模型
                mock_model = Mock()
                mock_model.model.names = {0: "person", 1: "car"}
                
                # 模拟预测结果
                mock_result = Mock()
                mock_result.names = {0: "person", 1: "car"}
                
                # 模拟检测框
                mock_box = Mock()
                mock_box.cls.item.return_value = 0  # person
                mock_box.xywhn = [Mock()]
                mock_box.xywhn[0].tolist.return_value = [0.5, 0.5, 0.3, 0.4]
                
                mock_result.boxes = [mock_box]
                mock_model.predict.return_value = [mock_result]
                mock_model.set_classes = Mock()
                mock_model.get_text_pe = Mock()
                
                mock_yoloe_class.return_value = mock_model
                
                # 导入并重新加载模块
                import importlib
                import app.helper.config
                import app.core.yoloe
                import app.helper.helper
                importlib.reload(app.helper.config)
                importlib.reload(app.core.yoloe)
                importlib.reload(app.helper.helper)
                
                from app.core.yoloe import Yoloe
                from app.helper import helper
                
                # 执行完整流程
                yoloe = Yoloe()
                
                # 1. 解析提示词
                prompts = helper.string_to_list("person,car")
                assert prompts == ["person", "car"]
                
                # 2. 扫描图片
                images = helper.scan_image_files(str(images_dir))
                assert len(images) == 2
                
                # 3. 初始化模型
                success = yoloe.init_model("test-model.pt", prompts)
                assert success is True
                
                # 4. 执行预测
                stats = yoloe.predict_image(images, 0.5, str(output_dir))
                
                # 验证结果
                assert stats['total_images'] == 2
                assert stats['successful_predictions'] > 0
                assert output_dir.exists()
                
                # 验证生成的文件
                classes_file = output_dir / "classes.txt"
                assert classes_file.exists()
                
                # 验证标注文件
                annotation_files = list(output_dir.glob("*.txt"))
                annotation_files = [f for f in annotation_files if f.name != "classes.txt"]
                assert len(annotation_files) >= 2
    
    @pytest.mark.integration
    def test_command_line_integration(self, temp_dir, monkeypatch):
        """测试命令行集成"""
        # 创建测试环境
        images_dir = temp_dir / "test_images"
        images_dir.mkdir()
        (images_dir / "test.jpg").touch()
        
        models_dir = temp_dir / "models"
        models_dir.mkdir()
        (models_dir / "test-model.pt").touch()
        
        output_dir = temp_dir / "output"
        
        # 设置参数
        test_args = [
            'main.py',
            '--prompts', 'person,car',
            '--conf', '0.7',
            '--images_folder_path', str(images_dir),
            '--output_folder', str(output_dir),
            '--model_name', 'test-model.pt'
        ]
        
        with monkeypatch.context() as m:
            m.setattr("sys.argv", test_args)
            m.setattr("app.helper.config.models_path", models_dir)
            m.setattr("app.helper.config.valid_models", ["test-model.pt"])
            m.setattr("app.helper.config.default_image_extensions", {'.jpg', '.png', '.jpeg'})
            
            # 模拟YOLO模型
            with patch('main.Yoloe') as mock_yoloe_class:
                mock_yoloe = Mock()
                mock_yoloe.init_model.return_value = True
                mock_yoloe.predict_image.return_value = {
                    'total_images': 1,
                    'successful_predictions': 1,
                    'failed_predictions': 0,
                    'annotation_files_created': 1,
                    'classes_detected': 1,
                    'total_detections': 1,
                    'class_distribution': {0: 1}
                }
                mock_yoloe_class.return_value = mock_yoloe
                
                # 导入并运行主程序
                from main import main
                
                result = main()
                
                # 验证成功执行
                assert result == 0
                
                # 验证模型被调用
                mock_yoloe.init_model.assert_called_once()
                mock_yoloe.predict_image.assert_called_once()
    
    @pytest.mark.integration 
    def test_error_handling_integration(self, temp_dir, monkeypatch):
        """测试错误处理集成"""
        # 测试各种错误场景下的系统行为
        
        # 1. 配置文件错误
        with monkeypatch.context() as m:
            m.setattr("app.helper.config.config_file_path", temp_dir / "nonexistent.ini")
            
            # 重新导入配置模块应该使用fallback值
            import importlib
            import app.helper.config
            with patch("warnings.warn"):
                importlib.reload(app.helper.config)
            
            # 应该能继续工作
            assert app.helper.config.default_conf == 0.5
        
        # 2. 模型文件不存在错误
        with monkeypatch.context() as m:
            m.setattr("app.helper.config.models_path", temp_dir / "nonexistent")
            m.setattr("app.helper.config.valid_models", ["test-model.pt"])
            
            from app.core.yoloe import Yoloe
            from app.helper.exceptions import ModelInitializationError
            
            yoloe = Yoloe()
            with pytest.raises(ModelInitializationError):
                yoloe.init_model("test-model.pt", ["person"])
        
        # 3. 图片目录不存在错误
        from app.helper.helper import scan_image_files
        from app.helper.exceptions import ImageNotFoundError
        
        with pytest.raises(ImageNotFoundError):
            scan_image_files(str(temp_dir / "nonexistent"))
    
    @pytest.mark.integration
    def test_configuration_reload(self, temp_dir, monkeypatch):
        """测试配置重新加载"""
        # 创建初始配置
        conf_dir = temp_dir / "conf"
        conf_dir.mkdir()
        
        initial_config = """[Default]
conf = 0.3
model_name = initial-model.pt
annotation_format = Yolo
image_extensions = .png .jpg

[Models]
valid_models = initial-model.pt
"""
        config_file = conf_dir / "config.ini"
        config_file.write_text(initial_config)
        
        with monkeypatch.context() as m:
            m.setattr("app.helper.config.config_file_path", config_file)
            
            # 第一次导入
            import importlib
            import app.helper.config
            importlib.reload(app.helper.config)
            
            assert app.helper.config.default_conf == 0.3
            assert app.helper.config.default_model_name == "initial-model.pt"
            
            # 修改配置文件
            updated_config = """[Default]
conf = 0.8
model_name = updated-model.pt
annotation_format = Yolo
image_extensions = .png .jpg .jpeg

[Models]
valid_models = updated-model.pt another-model.pt
"""
            config_file.write_text(updated_config)
            
            # 重新加载配置
            importlib.reload(app.helper.config)
            
            assert app.helper.config.default_conf == 0.8
            assert app.helper.config.default_model_name == "updated-model.pt"
            assert app.helper.config.valid_models == ["updated-model.pt", "another-model.pt"]


class TestComponentIntegration:
    """组件集成测试"""
    
    @pytest.mark.integration
    def test_helper_config_integration(self, config_dir, monkeypatch):
        """测试helper模块与config模块集成"""
        with monkeypatch.context() as m:
            m.setattr("app.helper.config.config_file_path", config_dir / "config.ini")
            m.setattr("app.helper.config.default_image_extensions", {'.jpg', '.png', '.jpeg'})
            
            import importlib
            import app.helper.config
            import app.helper.helper
            importlib.reload(app.helper.config)
            importlib.reload(app.helper.helper)
            
            from app.helper.helper import scan_image_files, string_to_list
            
            # 测试扫描功能使用配置的扩展名
            images_dir = config_dir.parent / "images"
            images_dir.mkdir()
            (images_dir / "test.jpg").touch()
            (images_dir / "test.png").touch()
            (images_dir / "test.bmp").touch()  # 不支持的格式
            
            images = scan_image_files(str(images_dir))
            assert len(images) == 2  # 只有jpg和png
            
            # 测试字符串转换
            prompts = string_to_list("person,car,bus")
            assert prompts == ["person", "car", "bus"]
    
    @pytest.mark.integration
    def test_yoloe_config_integration(self, models_dir, config_dir, monkeypatch):
        """测试YOLO模块与config模块集成"""
        with monkeypatch.context() as m:
            m.setattr("app.helper.config.models_path", models_dir)
            m.setattr("app.helper.config.valid_models", ["test-model.pt"])
            m.setattr("app.helper.config.config_file_path", config_dir / "config.ini")
            
            # 模拟YOLO模型
            with patch('app.core.yoloe.YOLOE') as mock_yoloe_class:
                mock_model = Mock()
                mock_model.set_classes = Mock()
                mock_model.get_text_pe = Mock()
                mock_yoloe_class.return_value = mock_model
                
                import importlib
                import app.helper.config
                import app.core.yoloe
                importlib.reload(app.helper.config)
                importlib.reload(app.core.yoloe)
                
                from app.core.yoloe import Yoloe
                
                yoloe = Yoloe()
                success = yoloe.init_model("test-model.pt", ["person", "car"])
                
                assert success is True
                assert yoloe.model_name == "test-model.pt"
                assert yoloe.class_names == ["person", "car"]
    
    @pytest.mark.integration
    def test_validation_integration(self, temp_dir):
        """测试验证器集成"""
        from app.helper.validators import Validator
        from app.helper.exceptions import InvalidParameterError, InvalidPathError
        
        # 创建测试文件
        test_file = temp_dir / "test.jpg"
        test_file.touch()
        
        # 测试各种验证器协同工作
        conf = Validator.validate_confidence(0.5)
        assert conf == 0.5
        
        path = Validator.validate_file_path(test_file)
        assert path == test_file
        
        extensions = Validator.validate_image_extensions(".jpg .png")
        assert extensions == {'.jpg', '.png'}
        
        image_path = Validator.validate_image_file(test_file, extensions)
        assert image_path == test_file
        
        prompts = Validator.validate_prompts("person,car")
        assert prompts == ["person", "car"]
        
        model = Validator.validate_model_name("test-model.pt", ["test-model.pt"])
        assert model == "test-model.pt"


class TestPerformanceIntegration:
    """性能集成测试"""
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_large_image_batch_processing(self, temp_dir, monkeypatch):
        """测试大批量图片处理性能"""
        # 创建大量测试图片
        images_dir = temp_dir / "large_batch"
        images_dir.mkdir()
        
        # 创建100个虚拟图片文件
        for i in range(100):
            (images_dir / f"image_{i:03d}.jpg").touch()
        
        models_dir = temp_dir / "models"
        models_dir.mkdir()
        (models_dir / "test-model.pt").touch()
        
        with monkeypatch.context() as m:
            m.setattr("app.helper.config.models_path", models_dir)
            m.setattr("app.helper.config.valid_models", ["test-model.pt"])
            m.setattr("app.helper.config.default_image_extensions", {'.jpg'})
            
            # 模拟快速的YOLO模型
            with patch('app.core.yoloe.YOLOE') as mock_yoloe_class:
                mock_model = Mock()
                mock_result = Mock()
                mock_result.boxes = []  # 空结果，快速处理
                mock_result.names = {}
                mock_model.predict.return_value = [mock_result]
                mock_model.set_classes = Mock()
                mock_model.get_text_pe = Mock()
                mock_model.model.names = {}
                mock_yoloe_class.return_value = mock_model
                
                import importlib
                import app.core.yoloe
                import app.helper.helper
                importlib.reload(app.core.yoloe)
                importlib.reload(app.helper.helper)
                
                from app.core.yoloe import Yoloe
                from app.helper.helper import scan_image_files
                
                # 测试扫描性能
                import time
                start_time = time.time()
                images = scan_image_files(str(images_dir))
                scan_time = time.time() - start_time
                
                assert len(images) == 100
                assert scan_time < 5.0  # 应该在5秒内完成
                
                # 测试预测性能
                yoloe = Yoloe()
                yoloe.init_model("test-model.pt", ["person"])
                
                start_time = time.time()
                stats = yoloe.predict_image(images, 0.5, str(temp_dir / "output"))
                predict_time = time.time() - start_time
                
                assert stats['total_images'] == 100
                assert predict_time < 30.0  # 应该在30秒内完成
    
    @pytest.mark.integration
    def test_memory_usage_integration(self, temp_dir, monkeypatch):
        """测试内存使用集成（简单检查）"""
        import psutil
        import os
        
        # 获取初始内存使用
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # 创建测试环境
        images_dir = temp_dir / "memory_test"
        images_dir.mkdir()
        
        for i in range(50):
            (images_dir / f"image_{i}.jpg").touch()
        
        models_dir = temp_dir / "models"
        models_dir.mkdir()
        (models_dir / "test-model.pt").touch()
        
        with monkeypatch.context() as m:
            m.setattr("app.helper.config.models_path", models_dir)
            m.setattr("app.helper.config.valid_models", ["test-model.pt"])
            m.setattr("app.helper.config.default_image_extensions", {'.jpg'})
            
            with patch('app.core.yoloe.YOLOE') as mock_yoloe_class:
                mock_model = Mock()
                mock_result = Mock()
                mock_result.boxes = []
                mock_result.names = {}
                mock_model.predict.return_value = [mock_result]
                mock_model.set_classes = Mock()
                mock_model.get_text_pe = Mock()
                mock_model.model.names = {}
                mock_yoloe_class.return_value = mock_model
                
                import importlib
                import app.core.yoloe
                import app.helper.helper
                importlib.reload(app.core.yoloe)
                importlib.reload(app.helper.helper)
                
                from app.core.yoloe import Yoloe
                from app.helper.helper import scan_image_files
                
                # 执行操作
                images = scan_image_files(str(images_dir))
                yoloe = Yoloe()
                yoloe.init_model("test-model.pt", ["person"])
                stats = yoloe.predict_image(images, 0.5, str(temp_dir / "output"))
                
                # 检查内存增长
                final_memory = process.memory_info().rss
                memory_growth = final_memory - initial_memory
                
                # 内存增长应该合理（小于100MB）
                assert memory_growth < 100 * 1024 * 1024  # 100MB 