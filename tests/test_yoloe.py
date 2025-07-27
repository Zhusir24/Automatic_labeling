"""
测试YOLO核心模块
"""
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from app.core.yoloe import Yoloe
from app.helper.exceptions import (
    ModelInitializationError,
    ModelPredictionError,
    FileOperationError
)


class TestYoloeInitialization:
    """测试Yoloe类初始化"""
    
    def test_yoloe_initial_state(self):
        """测试Yoloe初始状态"""
        yoloe = Yoloe()
        assert yoloe.model is None
        assert yoloe.model_name is None
        assert yoloe.class_names is None
        assert yoloe.is_initialized is False
    
    def test_get_model_info_uninitialized(self):
        """测试未初始化状态的模型信息"""
        yoloe = Yoloe()
        info = yoloe.get_model_info()
        
        expected = {
            'is_initialized': False,
            'model_name': None,
            'class_names': None,
            'num_classes': 0
        }
        assert info == expected


class TestYoloeModelInitialization:
    """测试Yoloe模型初始化"""
    
    def test_successful_model_initialization(self, mock_ultralytics, models_dir, monkeypatch):
        """测试成功的模型初始化"""
        with monkeypatch.context() as m:
            m.setattr("app.helper.config.models_path", models_dir)
            m.setattr("app.helper.config.valid_models", ["test-model.pt"])
            
            yoloe = Yoloe()
            result = yoloe.init_model("test-model.pt", ["person", "car"])
            
            assert result is True
            assert yoloe.is_initialized is True
            assert yoloe.model_name == "test-model.pt"
            assert yoloe.class_names == ["person", "car"]
            assert yoloe.model is not None
    
    def test_model_initialization_invalid_model_name(self, models_dir, monkeypatch):
        """测试无效模型名称的初始化"""
        with monkeypatch.context() as m:
            m.setattr("app.helper.config.models_path", models_dir)
            m.setattr("app.helper.config.valid_models", ["test-model.pt"])
            
            yoloe = Yoloe()
            with pytest.raises(ModelInitializationError, match="无效的模型名称"):
                yoloe.init_model("invalid-model.pt", ["person", "car"])
    
    def test_model_initialization_nonexistent_file(self, models_dir, monkeypatch):
        """测试不存在的模型文件"""
        with monkeypatch.context() as m:
            m.setattr("app.helper.config.models_path", models_dir)
            m.setattr("app.helper.config.valid_models", ["nonexistent.pt"])
            
            yoloe = Yoloe()
            with pytest.raises(ModelInitializationError, match="模型文件不存在"):
                yoloe.init_model("nonexistent.pt", ["person", "car"])
    
    def test_model_initialization_invalid_prompts(self, models_dir, monkeypatch):
        """测试无效提示词的初始化"""
        with monkeypatch.context() as m:
            m.setattr("app.helper.config.models_path", models_dir)
            m.setattr("app.helper.config.valid_models", ["test-model.pt"])
            
            yoloe = Yoloe()
            with pytest.raises(ModelInitializationError):
                yoloe.init_model("test-model.pt", [])  # 空提示词列表
    
    def test_model_initialization_loading_error(self, models_dir, monkeypatch):
        """测试模型加载错误"""
        def mock_yoloe_constructor(path):
            raise RuntimeError("Model loading failed")
        
        with monkeypatch.context() as m:
            m.setattr("app.helper.config.models_path", models_dir)
            m.setattr("app.helper.config.valid_models", ["test-model.pt"])
            m.setattr("app.core.yoloe.YOLOE", mock_yoloe_constructor)
            
            yoloe = Yoloe()
            with pytest.raises(ModelInitializationError, match="模型初始化失败"):
                yoloe.init_model("test-model.pt", ["person", "car"])
    
    def test_get_model_info_initialized(self, mock_ultralytics, models_dir, monkeypatch):
        """测试已初始化状态的模型信息"""
        with monkeypatch.context() as m:
            m.setattr("app.helper.config.models_path", models_dir)
            m.setattr("app.helper.config.valid_models", ["test-model.pt"])
            
            yoloe = Yoloe()
            yoloe.init_model("test-model.pt", ["person", "car", "bus"])
            
            info = yoloe.get_model_info()
            expected = {
                'is_initialized': True,
                'model_name': "test-model.pt",
                'class_names': ["person", "car", "bus"],
                'num_classes': 3
            }
            assert info == expected


class TestYoloePrediction:
    """测试Yoloe预测功能"""
    
    def test_prediction_model_not_initialized(self):
        """测试未初始化模型的预测"""
        yoloe = Yoloe()
        
        with pytest.raises(ModelPredictionError, match="模型未初始化"):
            yoloe.predict_image(["image.jpg"], 0.5, "output")
    
    def test_successful_prediction(self, mock_ultralytics, models_dir, temp_dir, sample_images_dir, monkeypatch):
        """测试成功的预测"""
        with monkeypatch.context() as m:
            m.setattr("app.helper.config.models_path", models_dir)
            m.setattr("app.helper.config.valid_models", ["test-model.pt"])
            m.setattr("app.helper.config.default_image_extensions", {'.jpg', '.png', '.jpeg'})
            
            yoloe = Yoloe()
            yoloe.init_model("test-model.pt", ["person", "car", "bus"])
            
            # 准备图片路径
            image_paths = [str(sample_images_dir / "image1.jpg")]
            output_dir = str(temp_dir / "output")
            
            stats = yoloe.predict_image(image_paths, 0.5, output_dir)
            
            # 验证返回的统计信息
            assert isinstance(stats, dict)
            assert 'total_images' in stats
            assert 'successful_predictions' in stats
            assert 'failed_predictions' in stats
            assert 'annotation_files_created' in stats
            assert 'classes_detected' in stats
            assert 'total_detections' in stats
            assert 'class_distribution' in stats
            
            # 验证输出目录已创建
            assert Path(output_dir).exists()
            
            # 验证生成了classes.txt文件
            classes_file = Path(output_dir) / "classes.txt"
            assert classes_file.exists()
    
    def test_prediction_empty_image_list(self, mock_ultralytics, models_dir, temp_dir, monkeypatch):
        """测试空图片列表的预测"""
        with monkeypatch.context() as m:
            m.setattr("app.helper.config.models_path", models_dir)
            m.setattr("app.helper.config.valid_models", ["test-model.pt"])
            
            yoloe = Yoloe()
            yoloe.init_model("test-model.pt", ["person", "car"])
            
            with pytest.raises(ModelPredictionError, match="图片路径列表不能为空"):
                yoloe.predict_image([], 0.5, str(temp_dir))
    
    def test_prediction_invalid_confidence(self, mock_ultralytics, models_dir, temp_dir, monkeypatch):
        """测试无效置信度的预测"""
        with monkeypatch.context() as m:
            m.setattr("app.helper.config.models_path", models_dir)
            m.setattr("app.helper.config.valid_models", ["test-model.pt"])
            
            yoloe = Yoloe()
            yoloe.init_model("test-model.pt", ["person", "car"])
            
            with pytest.raises(ModelPredictionError):
                yoloe.predict_image(["image.jpg"], 2.0, str(temp_dir))  # 无效置信度
    
    def test_prediction_nonexistent_images(self, mock_ultralytics, models_dir, temp_dir, monkeypatch):
        """测试不存在的图片文件"""
        with monkeypatch.context() as m:
            m.setattr("app.helper.config.models_path", models_dir)
            m.setattr("app.helper.config.valid_models", ["test-model.pt"])
            m.setattr("app.helper.config.default_image_extensions", {'.jpg', '.png', '.jpeg'})
            
            yoloe = Yoloe()
            yoloe.init_model("test-model.pt", ["person", "car"])
            
            with pytest.raises(ModelPredictionError, match="没有找到有效的图片文件"):
                yoloe.predict_image(["/nonexistent/image.jpg"], 0.5, str(temp_dir))
    
    def test_prediction_output_directory_creation_error(self, mock_ultralytics, models_dir, sample_images_dir, monkeypatch):
        """测试输出目录创建错误"""
        def mock_mkdir(*args, **kwargs):
            raise OSError("Permission denied")
        
        with monkeypatch.context() as m:
            m.setattr("app.helper.config.models_path", models_dir)
            m.setattr("app.helper.config.valid_models", ["test-model.pt"])
            m.setattr("app.helper.config.default_image_extensions", {'.jpg', '.png', '.jpeg'})
            m.setattr("pathlib.Path.mkdir", mock_mkdir)
            
            yoloe = Yoloe()
            yoloe.init_model("test-model.pt", ["person", "car"])
            
            image_paths = [str(sample_images_dir / "image1.jpg")]
            
            with pytest.raises(FileOperationError, match="创建输出目录失败"):
                yoloe.predict_image(image_paths, 0.5, "/invalid/output/path")
    
    def test_prediction_no_detections(self, models_dir, temp_dir, sample_images_dir, monkeypatch):
        """测试没有检测到目标的预测"""
        # 创建返回空结果的模拟模型
        mock_model = Mock()
        mock_result = Mock()
        mock_result.boxes = []  # 没有检测到任何目标
        mock_result.names = {}
        mock_model.predict.return_value = [mock_result]
        mock_model.set_classes = Mock()
        mock_model.get_text_pe = Mock()
        mock_model.model.names = {}
        
        with monkeypatch.context() as m:
            m.setattr("app.helper.config.models_path", models_dir)
            m.setattr("app.helper.config.valid_models", ["test-model.pt"])
            m.setattr("app.helper.config.default_image_extensions", {'.jpg', '.png', '.jpeg'})
            m.setattr("app.core.yoloe.YOLOE", lambda path: mock_model)
            
            yoloe = Yoloe()
            yoloe.init_model("test-model.pt", ["person", "car"])
            
            image_paths = [str(sample_images_dir / "image1.jpg")]
            output_dir = str(temp_dir / "output")
            
            stats = yoloe.predict_image(image_paths, 0.5, output_dir)
            
            # 验证统计信息
            assert stats['total_detections'] == 0
            assert stats['classes_detected'] == 0
            assert len(stats['class_distribution']) == 0
    
    def test_prediction_model_error_during_prediction(self, mock_ultralytics, models_dir, sample_images_dir, temp_dir, monkeypatch):
        """测试预测过程中的模型错误"""
        # 修改模拟模型使其在predict时抛出异常
        mock_ultralytics.predict.side_effect = RuntimeError("Prediction failed")
        
        with monkeypatch.context() as m:
            m.setattr("app.helper.config.models_path", models_dir)
            m.setattr("app.helper.config.valid_models", ["test-model.pt"])
            m.setattr("app.helper.config.default_image_extensions", {'.jpg', '.png', '.jpeg'})
            
            yoloe = Yoloe()
            yoloe.init_model("test-model.pt", ["person", "car"])
            
            image_paths = [str(sample_images_dir / "image1.jpg")]
            output_dir = str(temp_dir / "output")
            
            # 应该不会抛出异常，而是记录失败并继续
            stats = yoloe.predict_image(image_paths, 0.5, output_dir)
            
            # 验证失败统计
            assert stats['failed_predictions'] > 0
            assert stats['successful_predictions'] == 0


class TestYoloeAnnotationGeneration:
    """测试Yoloe标注生成功能"""
    
    def test_annotation_file_content(self, mock_ultralytics, models_dir, temp_dir, sample_images_dir, monkeypatch):
        """测试标注文件内容正确性"""
        with monkeypatch.context() as m:
            m.setattr("app.helper.config.models_path", models_dir)
            m.setattr("app.helper.config.valid_models", ["test-model.pt"])
            m.setattr("app.helper.config.default_image_extensions", {'.jpg', '.png', '.jpeg'})
            
            yoloe = Yoloe()
            yoloe.init_model("test-model.pt", ["person", "car", "bus"])
            
            image_paths = [str(sample_images_dir / "image1.jpg")]
            output_dir = str(temp_dir / "output")
            
            stats = yoloe.predict_image(image_paths, 0.5, output_dir)
            
            # 检查生成的标注文件
            annotation_file = Path(output_dir) / "image1.txt"
            assert annotation_file.exists()
            
            # 读取并验证标注内容
            content = annotation_file.read_text()
            lines = content.strip().split('\n')
            
            # 应该有两行（两个检测框）
            assert len(lines) == 2
            
            # 验证每行格式：class_id x_center y_center width height
            for line in lines:
                parts = line.split()
                assert len(parts) == 5
                
                # 验证所有值都是数字
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                # 验证坐标范围
                assert 0 <= x_center <= 1
                assert 0 <= y_center <= 1
                assert 0 <= width <= 1
                assert 0 <= height <= 1
                assert class_id >= 0
    
    def test_classes_file_generation(self, mock_ultralytics, models_dir, temp_dir, sample_images_dir, monkeypatch):
        """测试classes.txt文件生成"""
        with monkeypatch.context() as m:
            m.setattr("app.helper.config.models_path", models_dir)
            m.setattr("app.helper.config.valid_models", ["test-model.pt"])
            m.setattr("app.helper.config.default_image_extensions", {'.jpg', '.png', '.jpeg'})
            
            yoloe = Yoloe()
            yoloe.init_model("test-model.pt", ["person", "car", "bus"])
            
            image_paths = [str(sample_images_dir / "image1.jpg")]
            output_dir = str(temp_dir / "output")
            
            yoloe.predict_image(image_paths, 0.5, output_dir)
            
            # 检查classes.txt文件
            classes_file = Path(output_dir) / "classes.txt"
            assert classes_file.exists()
            
            # 读取并验证内容
            content = classes_file.read_text()
            lines = content.strip().split('\n')
            
            # 应该包含检测到的类别
            assert len(lines) > 0
            for line in lines:
                assert line.strip() != ""
    
    def test_annotation_file_writing_error(self, mock_ultralytics, models_dir, sample_images_dir, monkeypatch):
        """测试标注文件写入错误"""
        def mock_open(*args, **kwargs):
            raise PermissionError("Permission denied")
        
        with monkeypatch.context() as m:
            m.setattr("app.helper.config.models_path", models_dir)
            m.setattr("app.helper.config.valid_models", ["test-model.pt"])
            m.setattr("app.helper.config.default_image_extensions", {'.jpg', '.png', '.jpeg'})
            m.setattr("builtins.open", mock_open)
            
            yoloe = Yoloe()
            yoloe.init_model("test-model.pt", ["person", "car"])
            
            image_paths = [str(sample_images_dir / "image1.jpg")]
            
            # 应该处理文件写入错误但不崩溃
            stats = yoloe.predict_image(image_paths, 0.5, "/tmp/output")
            
            # 验证有处理错误的统计
            assert 'annotation_files_created' in stats


class TestYoloeEdgeCases:
    """测试Yoloe边界情况"""
    
    def test_prediction_with_mixed_valid_invalid_images(self, mock_ultralytics, models_dir, temp_dir, monkeypatch):
        """测试混合有效和无效图片的预测"""
        with monkeypatch.context() as m:
            m.setattr("app.helper.config.models_path", models_dir)
            m.setattr("app.helper.config.valid_models", ["test-model.pt"])
            m.setattr("app.helper.config.default_image_extensions", {'.jpg', '.png', '.jpeg'})
            
            # 创建一些测试图片文件
            valid_image = temp_dir / "valid.jpg"
            valid_image.touch()
            
            yoloe = Yoloe()
            yoloe.init_model("test-model.pt", ["person", "car"])
            
            # 混合有效和无效的图片路径
            image_paths = [
                str(valid_image),
                "/nonexistent/invalid.jpg",
                str(temp_dir / "nonexistent.png")
            ]
            output_dir = str(temp_dir / "output")
            
            # 应该只处理有效图片，跳过无效图片
            stats = yoloe.predict_image(image_paths, 0.5, output_dir)
            
            # 验证统计信息合理
            assert stats['total_images'] == 1  # 只有一个有效图片
    
    def test_model_reinitialization(self, mock_ultralytics, models_dir, monkeypatch):
        """测试模型重新初始化"""
        with monkeypatch.context() as m:
            m.setattr("app.helper.config.models_path", models_dir)
            m.setattr("app.helper.config.valid_models", ["test-model.pt", "another-model.pt"])
            
            yoloe = Yoloe()
            
            # 第一次初始化
            result1 = yoloe.init_model("test-model.pt", ["person", "car"])
            assert result1 is True
            assert yoloe.model_name == "test-model.pt"
            assert yoloe.class_names == ["person", "car"]
            
            # 第二次初始化（重新初始化）
            result2 = yoloe.init_model("another-model.pt", ["dog", "cat", "bird"])
            assert result2 is True
            assert yoloe.model_name == "another-model.pt"
            assert yoloe.class_names == ["dog", "cat", "bird"]
            
            # 验证模型信息更新
            info = yoloe.get_model_info()
            assert info['model_name'] == "another-model.pt"
            assert info['class_names'] == ["dog", "cat", "bird"]
            assert info['num_classes'] == 3 