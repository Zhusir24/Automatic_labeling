"""
pytest配置和通用测试fixtures
"""
import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Generator, Dict, Any
import pytest
from unittest.mock import Mock, MagicMock

# 添加项目根目录到sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """创建临时目录"""
    temp_path = Path(tempfile.mkdtemp())
    try:
        yield temp_path
    finally:
        shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_images_dir(temp_dir: Path) -> Path:
    """创建包含示例图片的目录"""
    images_dir = temp_dir / "images"
    images_dir.mkdir()
    
    # 创建一些虚拟图片文件
    for name in ["image1.jpg", "image2.png", "image3.jpeg", "not_image.txt"]:
        (images_dir / name).touch()
    
    return images_dir


@pytest.fixture
def empty_dir(temp_dir: Path) -> Path:
    """创建空目录"""
    empty = temp_dir / "empty"
    empty.mkdir()
    return empty


@pytest.fixture
def config_dir(temp_dir: Path) -> Path:
    """创建配置目录和配置文件"""
    conf_dir = temp_dir / "conf"
    conf_dir.mkdir()
    
    config_content = """[Default]
conf = 0.5
model_name = test-model.pt
annotation_format = Yolo
image_extensions = .png .jpg .jpeg

[Models]
valid_models = test-model.pt another-model.pt
"""
    
    config_file = conf_dir / "config.ini"
    config_file.write_text(config_content)
    
    return conf_dir


@pytest.fixture
def models_dir(temp_dir: Path) -> Path:
    """创建模型目录和模型文件"""
    models = temp_dir / "models"
    models.mkdir()
    
    # 创建虚拟模型文件
    (models / "test-model.pt").touch()
    (models / "another-model.pt").touch()
    
    return models


@pytest.fixture
def mock_yoloe_model():
    """创建模拟的YOLOE模型"""
    mock_model = Mock()
    mock_model.model.names = {0: "person", 1: "car", 2: "bus"}
    
    # 模拟预测结果
    mock_result = Mock()
    mock_result.names = {0: "person", 1: "car", 2: "bus"}
    
    # 模拟检测框
    mock_box1 = Mock()
    mock_box1.cls.item.return_value = 0  # person
    mock_box1.xywhn = [Mock()]
    mock_box1.xywhn[0].tolist.return_value = [0.5, 0.5, 0.3, 0.4]
    
    mock_box2 = Mock()
    mock_box2.cls.item.return_value = 1  # car
    mock_box2.xywhn = [Mock()]
    mock_box2.xywhn[0].tolist.return_value = [0.3, 0.7, 0.2, 0.3]
    
    mock_result.boxes = [mock_box1, mock_box2]
    
    mock_model.predict.return_value = [mock_result]
    mock_model.set_classes = Mock()
    mock_model.get_text_pe = Mock(return_value="text_pe_result")
    
    return mock_model


@pytest.fixture
def mock_ultralytics(mock_yoloe_model):
    """模拟ultralytics库"""
    with pytest.MonkeyPatch.context() as m:
        m.setattr("app.core.yoloe.YOLOE", lambda path: mock_yoloe_model)
        yield mock_yoloe_model


@pytest.fixture
def sample_prompts() -> list:
    """示例提示词"""
    return ["person", "car", "bus"]


@pytest.fixture
def sample_config_data() -> Dict[str, Any]:
    """示例配置数据"""
    return {
        'default_conf': 0.5,
        'default_model_name': 'test-model.pt',
        'default_annotation_format': 'Yolo',
        'default_image_extensions': {'.png', '.jpg', '.jpeg'},
        'valid_models': ['test-model.pt', 'another-model.pt']
    } 