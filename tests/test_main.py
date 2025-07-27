"""
测试主程序功能
"""
import pytest
import sys
from io import StringIO
from unittest.mock import patch, Mock, MagicMock
from pathlib import Path

from main import (
    create_argument_parser,
    validate_arguments,
    run_automatic_labeling,
    main
)
from app.helper.exceptions import (
    InvalidParameterError,
    ConfigError,
    ModelInitializationError,
    ModelPredictionError,
    AutoLabelingError
)


class TestCreateArgumentParser:
    """测试命令行参数解析器创建"""
    
    def test_parser_creation(self):
        """测试解析器创建"""
        parser = create_argument_parser()
        assert parser is not None
        assert parser.description == "自动图片标注程序"
    
    def test_parser_required_arguments(self):
        """测试必需参数"""
        parser = create_argument_parser()
        
        # 测试缺少必需参数时的错误
        with pytest.raises(SystemExit):
            parser.parse_args([])
    
    def test_parser_with_valid_arguments(self, sample_images_dir, temp_dir):
        """测试有效参数解析"""
        parser = create_argument_parser()
        
        args = parser.parse_args([
            '--prompts', 'person,car,bus',
            '--conf', '0.7',
            '--images_folder_path', str(sample_images_dir),
            '--output_folder', str(temp_dir)
        ])
        
        assert args.prompts == 'person,car,bus'
        assert args.conf == 0.7
        assert args.images_folder_path == str(sample_images_dir)
        assert args.output_folder == str(temp_dir)
    
    def test_parser_default_values(self, monkeypatch):
        """测试默认值"""
        # 模拟配置值
        with monkeypatch.context() as m:
            m.setattr("main.config.default_model_name", "test-model.pt")
            m.setattr("main.config.default_conf", 0.5)
            m.setattr("main.config.images_folder_path", Path("/test/images"))
            m.setattr("main.config.outputs_path", Path("/test/outputs"))
            m.setattr("main.config.default_annotation_format", "Yolo")
            m.setattr("main.config.valid_models", ["test-model.pt"])
            
            parser = create_argument_parser()
            args = parser.parse_args(['--prompts', 'person,car'])
            
            assert args.model_name == "test-model.pt"
            assert args.conf == 0.5
            assert args.annotation_format == "Yolo"
    
    def test_parser_help_output(self, capsys):
        """测试帮助信息输出"""
        parser = create_argument_parser()
        
        with pytest.raises(SystemExit):
            parser.parse_args(['--help'])
        
        captured = capsys.readouterr()
        assert "自动图片标注程序" in captured.out
        assert "示例用法" in captured.out


class TestValidateArguments:
    """测试参数验证"""
    
    def test_valid_arguments(self, sample_images_dir, temp_dir, monkeypatch):
        """测试有效参数验证"""
        with monkeypatch.context() as m:
            m.setattr("main.config.valid_models", ["test-model.pt"])
            
            # 创建模拟的命名空间
            args = Mock()
            args.conf = 0.5
            args.model_name = "test-model.pt"
            args.prompts = "person,car,bus"
            args.images_folder_path = str(sample_images_dir)
            args.output_folder = str(temp_dir)
            
            # 应该不抛出异常
            validate_arguments(args)
    
    def test_invalid_confidence(self, sample_images_dir, temp_dir, monkeypatch):
        """测试无效置信度验证"""
        with monkeypatch.context() as m:
            m.setattr("main.config.valid_models", ["test-model.pt"])
            
            args = Mock()
            args.conf = 2.0  # 无效置信度
            args.model_name = "test-model.pt"
            args.prompts = "person,car,bus"
            args.images_folder_path = str(sample_images_dir)
            args.output_folder = str(temp_dir)
            
            with pytest.raises(InvalidParameterError):
                validate_arguments(args)
    
    def test_invalid_model_name(self, sample_images_dir, temp_dir, monkeypatch):
        """测试无效模型名称验证"""
        with monkeypatch.context() as m:
            m.setattr("main.config.valid_models", ["test-model.pt"])
            
            args = Mock()
            args.conf = 0.5
            args.model_name = "invalid-model.pt"  # 无效模型
            args.prompts = "person,car,bus"
            args.images_folder_path = str(sample_images_dir)
            args.output_folder = str(temp_dir)
            
            with pytest.raises(InvalidParameterError):
                validate_arguments(args)
    
    def test_invalid_prompts(self, sample_images_dir, temp_dir, monkeypatch):
        """测试无效提示词验证"""
        with monkeypatch.context() as m:
            m.setattr("main.config.valid_models", ["test-model.pt"])
            
            args = Mock()
            args.conf = 0.5
            args.model_name = "test-model.pt"
            args.prompts = ""  # 空提示词
            args.images_folder_path = str(sample_images_dir)
            args.output_folder = str(temp_dir)
            
            with pytest.raises(InvalidParameterError):
                validate_arguments(args)
    
    def test_empty_paths(self, monkeypatch):
        """测试空路径验证"""
        with monkeypatch.context() as m:
            m.setattr("main.config.valid_models", ["test-model.pt"])
            
            args = Mock()
            args.conf = 0.5
            args.model_name = "test-model.pt"
            args.prompts = "person,car"
            args.images_folder_path = ""  # 空路径
            args.output_folder = "/test/output"
            
            with pytest.raises(InvalidParameterError, match="图片文件夹路径不能为空"):
                validate_arguments(args)
            
            args.images_folder_path = "/test/images"
            args.output_folder = ""  # 空输出路径
            
            with pytest.raises(InvalidParameterError, match="输出文件夹路径不能为空"):
                validate_arguments(args)


class TestRunAutomaticLabeling:
    """测试自动标注流程"""
    
    @patch('main.helper.string_to_list')
    @patch('main.helper.scan_image_files')
    @patch('main.Yoloe')
    def test_successful_labeling(self, mock_yoloe_class, mock_scan, mock_string_to_list, temp_dir):
        """测试成功的自动标注流程"""
        # 设置模拟
        mock_string_to_list.return_value = ["person", "car"]
        mock_scan.return_value = ["/path/to/image1.jpg", "/path/to/image2.jpg"]
        
        mock_yoloe = Mock()
        mock_yoloe.init_model.return_value = True
        mock_yoloe.predict_image.return_value = {
            'total_images': 2,
            'successful_predictions': 2,
            'failed_predictions': 0,
            'annotation_files_created': 2,
            'classes_detected': 2,
            'total_detections': 5,
            'class_distribution': {0: 3, 1: 2}
        }
        mock_yoloe_class.return_value = mock_yoloe
        
        # 创建参数
        args = Mock()
        args.prompts = "person,car"
        args.images_folder_path = "/test/images"
        args.model_name = "test-model.pt"
        args.conf = 0.5
        args.annotation_format = "Yolo"
        args.output_folder = str(temp_dir)
        
        # 运行测试
        stats = run_automatic_labeling(args)
        
        # 验证结果
        assert stats is not None
        assert stats['total_images'] == 2
        assert stats['successful_predictions'] == 2
        assert stats['failed_predictions'] == 0
        
        # 验证调用
        mock_string_to_list.assert_called_once_with(input_str="person,car")
        mock_scan.assert_called_once_with(folder_path="/test/images")
        mock_yoloe.init_model.assert_called_once_with(model_name="test-model.pt", names=["person", "car"])
        mock_yoloe.predict_image.assert_called_once()
    
    @patch('main.helper.string_to_list')
    def test_labeling_string_to_list_error(self, mock_string_to_list):
        """测试提示词解析错误"""
        mock_string_to_list.side_effect = InvalidParameterError("Invalid prompts")
        
        args = Mock()
        args.prompts = "invalid/prompts"
        
        with pytest.raises(InvalidParameterError):
            run_automatic_labeling(args)
    
    @patch('main.helper.string_to_list')
    @patch('main.helper.scan_image_files')
    def test_labeling_scan_images_error(self, mock_scan, mock_string_to_list):
        """测试图片扫描错误"""
        mock_string_to_list.return_value = ["person", "car"]
        mock_scan.side_effect = FileNotFoundError("Images not found")
        
        args = Mock()
        args.prompts = "person,car"
        args.images_folder_path = "/nonexistent/path"
        
        with pytest.raises(AutoLabelingError):
            run_automatic_labeling(args)
    
    @patch('main.helper.string_to_list')
    @patch('main.helper.scan_image_files')
    @patch('main.Yoloe')
    def test_labeling_model_init_error(self, mock_yoloe_class, mock_scan, mock_string_to_list):
        """测试模型初始化错误"""
        mock_string_to_list.return_value = ["person", "car"]
        mock_scan.return_value = ["/path/to/image.jpg"]
        
        mock_yoloe = Mock()
        mock_yoloe.init_model.side_effect = ModelInitializationError("Model init failed")
        mock_yoloe_class.return_value = mock_yoloe
        
        args = Mock()
        args.prompts = "person,car"
        args.images_folder_path = "/test/images"
        args.model_name = "invalid-model.pt"
        
        with pytest.raises(ModelInitializationError):
            run_automatic_labeling(args)
    
    @patch('main.helper.string_to_list')
    @patch('main.helper.scan_image_files')
    @patch('main.Yoloe')
    def test_labeling_prediction_error(self, mock_yoloe_class, mock_scan, mock_string_to_list):
        """测试预测错误"""
        mock_string_to_list.return_value = ["person", "car"]
        mock_scan.return_value = ["/path/to/image.jpg"]
        
        mock_yoloe = Mock()
        mock_yoloe.init_model.return_value = True
        mock_yoloe.predict_image.side_effect = ModelPredictionError("Prediction failed")
        mock_yoloe_class.return_value = mock_yoloe
        
        args = Mock()
        args.prompts = "person,car"
        args.images_folder_path = "/test/images"
        args.model_name = "test-model.pt"
        args.conf = 0.5
        args.output_folder = "/test/output"
        
        with pytest.raises(ModelPredictionError):
            run_automatic_labeling(args)
    
    @patch('main.helper.string_to_list')
    @patch('main.helper.scan_image_files')
    @patch('main.Yoloe')
    def test_labeling_keyboard_interrupt(self, mock_yoloe_class, mock_scan, mock_string_to_list):
        """测试用户中断"""
        mock_string_to_list.side_effect = KeyboardInterrupt()
        
        args = Mock()
        args.prompts = "person,car"
        
        result = run_automatic_labeling(args)
        assert result is None
    
    @patch('main.helper.string_to_list')
    @patch('main.helper.scan_image_files')
    @patch('main.Yoloe')
    def test_labeling_model_init_returns_false(self, mock_yoloe_class, mock_scan, mock_string_to_list):
        """测试模型初始化返回False"""
        mock_string_to_list.return_value = ["person", "car"]
        mock_scan.return_value = ["/path/to/image.jpg"]
        
        mock_yoloe = Mock()
        mock_yoloe.init_model.return_value = False  # 返回False
        mock_yoloe_class.return_value = mock_yoloe
        
        args = Mock()
        args.prompts = "person,car"
        args.images_folder_path = "/test/images"
        args.model_name = "test-model.pt"
        
        with pytest.raises(ModelInitializationError, match="模型初始化返回失败状态"):
            run_automatic_labeling(args)


class TestMainFunction:
    """测试主函数"""
    
    @patch('main.create_argument_parser')
    @patch('main.validate_arguments')
    @patch('main.run_automatic_labeling')
    def test_main_successful_execution(self, mock_run, mock_validate, mock_parser):
        """测试主函数成功执行"""
        # 设置模拟
        mock_parser_instance = Mock()
        mock_parser.return_value = mock_parser_instance
        
        mock_args = Mock()
        mock_parser_instance.parse_args.return_value = mock_args
        
        mock_run.return_value = {'total_images': 5}
        
        # 运行测试
        result = main()
        
        # 验证返回值
        assert result == 0
        
        # 验证调用
        mock_parser.assert_called_once()
        mock_parser_instance.parse_args.assert_called_once()
        mock_validate.assert_called_once_with(mock_args)
        mock_run.assert_called_once_with(mock_args)
    
    @patch('main.create_argument_parser')
    @patch('main.validate_arguments')
    @patch('main.run_automatic_labeling')
    def test_main_user_interrupt(self, mock_run, mock_validate, mock_parser):
        """测试主函数用户中断"""
        mock_parser_instance = Mock()
        mock_parser.return_value = mock_parser_instance
        mock_parser_instance.parse_args.return_value = Mock()
        
        mock_run.return_value = None  # 用户中断
        
        result = main()
        assert result == 1
    
    @patch('main.create_argument_parser')
    @patch('main.validate_arguments')
    def test_main_config_error(self, mock_validate, mock_parser):
        """测试主函数配置错误"""
        mock_parser_instance = Mock()
        mock_parser.return_value = mock_parser_instance
        mock_parser_instance.parse_args.return_value = Mock()
        
        mock_validate.side_effect = ConfigError("Config error")
        
        result = main()
        assert result == 2
    
    @patch('main.create_argument_parser')
    @patch('main.validate_arguments')
    def test_main_invalid_parameter_error(self, mock_validate, mock_parser):
        """测试主函数参数错误"""
        mock_parser_instance = Mock()
        mock_parser.return_value = mock_parser_instance
        mock_parser_instance.parse_args.return_value = Mock()
        
        mock_validate.side_effect = InvalidParameterError("Invalid parameter")
        
        result = main()
        assert result == 3
    
    @patch('main.create_argument_parser')
    @patch('main.validate_arguments')
    @patch('main.run_automatic_labeling')
    def test_main_model_initialization_error(self, mock_run, mock_validate, mock_parser):
        """测试主函数模型初始化错误"""
        mock_parser_instance = Mock()
        mock_parser.return_value = mock_parser_instance
        mock_parser_instance.parse_args.return_value = Mock()
        
        mock_run.side_effect = ModelInitializationError("Model init error")
        
        result = main()
        assert result == 4
    
    @patch('main.create_argument_parser')
    @patch('main.validate_arguments')
    @patch('main.run_automatic_labeling')
    def test_main_model_prediction_error(self, mock_run, mock_validate, mock_parser):
        """测试主函数模型预测错误"""
        mock_parser_instance = Mock()
        mock_parser.return_value = mock_parser_instance
        mock_parser_instance.parse_args.return_value = Mock()
        
        mock_run.side_effect = ModelPredictionError("Prediction error")
        
        result = main()
        assert result == 5
    
    @patch('main.create_argument_parser')
    @patch('main.validate_arguments')
    @patch('main.run_automatic_labeling')
    def test_main_auto_labeling_error(self, mock_run, mock_validate, mock_parser):
        """测试主函数自动标注错误"""
        mock_parser_instance = Mock()
        mock_parser.return_value = mock_parser_instance
        mock_parser_instance.parse_args.return_value = Mock()
        
        mock_run.side_effect = AutoLabelingError("Labeling error")
        
        result = main()
        assert result == 6
    
    @patch('main.create_argument_parser')
    @patch('main.validate_arguments')
    @patch('main.run_automatic_labeling')
    def test_main_unexpected_error(self, mock_run, mock_validate, mock_parser):
        """测试主函数意外错误"""
        mock_parser_instance = Mock()
        mock_parser.return_value = mock_parser_instance
        mock_parser_instance.parse_args.return_value = Mock()
        
        mock_run.side_effect = RuntimeError("Unexpected error")
        
        result = main()
        assert result == 99


class TestMainIntegration:
    """测试主函数集成场景"""
    
    def test_main_with_real_arguments(self, sample_images_dir, temp_dir, monkeypatch):
        """测试使用真实参数的主函数（集成测试）"""
        # 设置测试环境
        test_args = [
            '--prompts', 'person,car',
            '--conf', '0.5',
            '--images_folder_path', str(sample_images_dir),
            '--output_folder', str(temp_dir),
            '--model_name', 'test-model.pt'
        ]
        
        # 模拟配置
        with monkeypatch.context() as m:
            m.setattr("main.config.valid_models", ["test-model.pt"])
            m.setattr("main.config.default_model_name", "test-model.pt")
            m.setattr("main.config.default_conf", 0.5)
            m.setattr("main.config.default_annotation_format", "Yolo")
            m.setattr("main.config.default_image_extensions", {'.jpg', '.png', '.jpeg'})
            m.setattr("sys.argv", ["main.py"] + test_args)
            
            # 模拟YOLO模型
            with patch('main.Yoloe') as mock_yoloe_class:
                mock_yoloe = Mock()
                mock_yoloe.init_model.return_value = True
                mock_yoloe.predict_image.return_value = {
                    'total_images': 3,
                    'successful_predictions': 3,
                    'failed_predictions': 0,
                    'annotation_files_created': 3,
                    'classes_detected': 2,
                    'total_detections': 8,
                    'class_distribution': {0: 5, 1: 3}
                }
                mock_yoloe_class.return_value = mock_yoloe
                
                # 运行主函数
                result = main()
                
                # 验证成功执行
                assert result == 0
                
                # 验证模型调用
                mock_yoloe.init_model.assert_called_once()
                mock_yoloe.predict_image.assert_called_once()
    
    @patch('sys.argv', ['main.py', '--help'])
    def test_main_help_argument(self):
        """测试帮助参数"""
        with pytest.raises(SystemExit) as exc_info:
            main()
        
        # argparse --help 通常返回0
        assert exc_info.value.code == 0
    
    @patch('sys.argv', ['main.py'])
    def test_main_missing_required_arguments(self):
        """测试缺少必需参数"""
        with pytest.raises(SystemExit):
            main() 