import os

from loguru import logger
from ultralytics import YOLOE

from app.helper import config


class Yoloe():
    def init_model(self, model_name:str, names:list):
        self.model = YOLOE(os.path.join(config.models_path, model_name))
        self.model.set_classes(names, self.model.get_text_pe(names))
        logger.success(f'{model_name}初始化成功.')

    def predict_image(self, images_path:list, conf:float):
        for image_path in images_path:
            results = self.model.predict(image_path, conf=conf)
            results[0].show()