import os
import sys
from typing import List

from loguru import logger
from tqdm import tqdm

from app.helper import config


def string_to_list(input_str:str) -> List[str]:
    """
    将输入字符串转换为列表，如果失败则返回错误原因
    Args:
        input_str (str): 输入字符串，例如 "hello tree boxes"
    Returns:
        tuple: (success, result)
               - success (bool): 是否转换成功
               - result (list or str): 成功返回列表，失败返回错误原因
    """
    if not isinstance(input_str, str):
        logger.error("输入必须是字符串类型")
        logger.info(f'示例输入为： bus,taxi,person')
        sys.exit(1)
    try:
        result = [item.strip() for item in input_str.split(",") if item.strip()]
        return result
    except Exception as e:
        logger.error(f"转换失败: {str(e)}")
        logger.info(f'示例输入为： bus,taxi,person')
        sys.exit(1)


def scan_image_files(folder_path: str) -> List[str]:
    """
    扫描文件夹中的图片文件（png/jpg/jpeg）并返回绝对路径列表
    Args:
        folder_path: 要扫描的文件夹路径
    Returns:
        List[str]: 图片文件的绝对路径列表
    Raises:
        ValueError: 如果文件夹不存在或为空
    """
    if not os.path.exists(folder_path):
        logger.error(f"文件夹不存在: {folder_path}")
        sys.exit(1)

    if not os.listdir(folder_path):
        logger.error(f"文件夹为空: {folder_path}")
        sys.exit(1)

    image_extensions = config.default_image_extensions
    all_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            all_files.append(os.path.join(root, file))

    # 筛选图片文件（带进度条）
    image_files = []
    for file_path in tqdm(all_files, desc="扫描图片", unit="file"):
        ext = os.path.splitext(file_path)[1].lower()
        if ext in image_extensions:
            image_files.append(os.path.abspath(file_path))

    logger.info(f'总共扫描到{len(image_files)}张合法图片')

    return image_files