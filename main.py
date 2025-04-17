import argparse

from loguru import logger

from app.core.yoloe import Yoloe
from app.helper import config, helper

yoloe = Yoloe()

def main():
    parser = argparse.ArgumentParser(description='This is automatic labeling program.')
    parser.add_argument('--visual', action='store_true', help='Enable visual mode.')
    parser.add_argument('--model_name', type=str, default=config.default_model_name, help='Specify the model name (default: yoloev8-s).', choices=config.valid_models)
    parser.add_argument('--prompts', type=str, help='Prompt for detections.', required=True)
    parser.add_argument('--conf', type=float, help='Confidence.', required=False, default=config.default_conf)
    parser.add_argument('--images_folder_path', type=str, help='Picture storage path.', required=False, default=config.images_folder_path)
    parser.add_argument('--annotation_format', type=str, help='Annotation format.', required=False, default=config.default_annotation_format)
    parser.add_argument('--output_folder', type=str, help='Outputs save.', required=False, default=config.outputs_path)
    args = parser.parse_args()

    prompts = helper.string_to_list(input_str=args.prompts)

    logger.info(f'选用模型为{args.model_name}, 提示词为{prompts}')
    logger.info(f'图片存放路径为{args.images_folder_path}')
    logger.info(f'置信度为{args.conf}, 标注格式为{args.annotation_format}')

    images_path = helper.scan_image_files(folder_path=args.images_folder_path)

    yoloe.init_model(model_name=args.model_name,names=prompts)

    yoloe.predict_image(images_path=images_path, conf=args.conf, output_dir=args.output_folder)


if __name__ == '__main__':
    main()