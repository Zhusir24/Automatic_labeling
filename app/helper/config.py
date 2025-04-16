import configparser
import os

path_root = os.getcwd()
app_path = os.path.join(path_root, 'app')

conf_path = os.path.join(app_path, 'conf')
core_path = os.path.join(app_path, 'core')
images_folder_path = os.path.join(app_path, 'images')
models_path = os.path.join(app_path, 'models')

config_file_path = os.path.join(path_root, conf_path, 'config.ini')
config = configparser.ConfigParser()
config.read(config_file_path)

default_conf = config.getfloat('Default', 'conf')
default_model_name = config.get('Default', 'model_name')
default_annotation_format = config.get('Default', 'annotation_format')
default_image_extensions = set(ext.strip() for ext in config.get('Default', 'image_extensions', fallback='').split() if ext.strip())

valid_models = config.get('Models', 'valid_models').split()