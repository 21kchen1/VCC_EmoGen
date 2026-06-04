1#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

r"""
@DATE    :   2026-05-29 09:37:17
@Author  :   Chen
@File    :   code/VCC_EmoGen/training/dataset_split.py
@Software:   VSCode
@Description:
    数据集按标签划分
"""

from datetime import datetime
import json
import os
import shutil
import sys
from typing import Dict, List

class AnnoClassifier:
    def __init__(self, property_name: str, root_path: str) -> None:
        self.property_name = property_name
        self.root_path = root_path
        self._class_root_path = os.path.join(root_path, property_name)


    def run(self, anno_json: Dict) -> List[str]:
        return []

class SingleAnnoClassifier(AnnoClassifier):

    def run(self, anno_json: Dict) -> List[str]:
        if anno_json.get(self.property_name) is None:
            return []
        output_paths = [os.path.join(self._class_root_path, anno_json[self.property_name])]
        # output_paths = [anno_json[self.property_name]]

        return output_paths

class MultAnnoClassifier(AnnoClassifier):

    def run(self, anno_json: Dict) -> List[str]:
        if anno_json.get(self.property_name) is None:
            return []

        output_paths = list(set([
            os.path.join(self._class_root_path, property_val)
            for property_val in anno_json[self.property_name]
        ]))
        # output_paths = list(set([
        #     property_val
        #     for property_val in anno_json[self.property_name]
        # ]))

        return output_paths

def split_dataset(root_path: str, output_folder_path: str, annoClassifiers: List[AnnoClassifier]) -> bool:
    """
    按属性与值进行划分

    Args:
        root_path (str): _description_
        output_folder_path (str): _description_
        annoClassifiers (List[AnnoClassifier]): _description_
    """

    anno_root_path = os.path.join(root_path, "annotation")
    image_root_path = os.path.join(root_path, "image")

    try:
        os.makedirs(output_folder_path)

        # 存储
        for annoClassifier in annoClassifiers:
            if os.path.exists(annoClassifier._class_root_path):
                continue
            os.makedirs(annoClassifier._class_root_path)

        # 遍历
        for root, _, file_paths in os.walk(anno_root_path):
            for file_path in file_paths:
                if not file_path.endswith(".json"):
                    continue
                anno_json: Dict = json.load(open(os.path.join(root, file_path), "r"))

                # 分类
                for annoClassifier in annoClassifiers:
                    if anno_json.get(annoClassifier.property_name) is None:
                        continue

                    class_paths = annoClassifier.run(anno_json)

                    image_path = os.path.join(image_root_path, anno_json["emotion"], anno_json["image_id"] + ".jpg")

                    print(f"\rnow: {image_path}", end= "", flush= False)


                    # 存储
                    for class_path in class_paths:
                        if not os.path.exists(class_path):
                            os.makedirs(class_path)

                        shutil.copy(image_path, class_path)

    except Exception as e:
        print(e)
        return False

    return True

def rename_dataset(dataset_root_path: str, annoClassifiers: List[AnnoClassifier]) -> bool:
    """
    修改属性下各类的文件夹名称

    Args:
        root_path (str): _description_

    Returns:
        bool: _description_
    """

    try:
        for annoClassifier in annoClassifiers:
            property_path = os.path.join(dataset_root_path, annoClassifier.property_name)

            # 遍历一级子文件夹
            for folder_name in os.listdir(property_path):
                folder_path = os.path.join(property_path, folder_name)

                if not os.path.isdir(folder_path):
                    continue

                # 统计图像数量
                image_num = len([image_name for image_name in os.listdir(folder_path) if image_name.endswith(".jpg")])
                new_folder_path = os.path.join(property_path, f"({image_num}){folder_name}")

                os.rename(folder_path, new_folder_path)

    except Exception as e:
        print(e)
        return False

    return True

def size_dataset(root_path: str) -> None:
    all_size = 0
    for root, _, file_paths in os.walk(root_path):
            for file_path in file_paths:
                if not file_path.endswith(".jpg"):
                    continue
                all_size += 1

    print(all_size)

DATASET_ROOT_PATH = "/mnt/d/dataset/EmoSet"
# SPLIT_OUTPUT_PATH = f"/mnt/d/dataset/EmoSet/LableSplit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
SPLIT_OUTPUT_PATH = f"/mnt/d/dataset/EmoSet/LableSplit_Test"
propertys = ["object", "scene"]

annoClassifiers = [
    SingleAnnoClassifier("scene", SPLIT_OUTPUT_PATH),
    MultAnnoClassifier("object", SPLIT_OUTPUT_PATH)
]

CHECK_DATASET_PATH = "/mnt/d/dataset/EmoSet/LableSplit_20260529_074548"

def main() -> None:
    # split_dataset(DATASET_ROOT_PATH, SPLIT_OUTPUT_PATH, annoClassifiers)
    # size_dataset(CHECK_DATASET_PATH)
    rename_dataset(SPLIT_OUTPUT_PATH, annoClassifiers)

if __name__ == "__main__":
    main()