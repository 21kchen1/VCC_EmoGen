#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

r"""
@DATE    :   2026-05-30 13:10:06
@Author  :   Chen
@File    :   code/VCC_EmoGen/training/download_model.py
@Software:   VSCode
@Description:
    模型下载
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch


class ModelLoader:

    def __init__(self, pretrained_model_name_or_path: str, cache_dir: str= "") -> None:
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.cache_dir = cache_dir

    def run(self) -> bool:
        return False

class DiffusionLoader(ModelLoader):

    def run(self) -> bool:
        try:
            from diffusers import DiffusionPipeline

            pipe = DiffusionPipeline.from_pretrained(self.pretrained_model_name_or_path, dtype=torch.bfloat16, cache_dir= self.cache_dir if not self.cache_dir == "" else None)

            pipe = pipe.to("cuda")

            prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
            image = pipe(prompt).images[0]

            image.save("here.png")

            return True
        except Exception as e:
            print(e)
            return False

class CLIPLoader(ModelLoader):

    def run(self) -> bool:
        try:
            from transformers import AutoProcessor, AutoModelForZeroShotImageClassification

            processor = AutoProcessor.from_pretrained(
                self.pretrained_model_name_or_path,
                cache_dir=self.cache_dir
            )

            model = AutoModelForZeroShotImageClassification.from_pretrained(
                self.pretrained_model_name_or_path,
                cache_dir=self.cache_dir
            )
            return True
        except Exception as e:
            print(e)
            return False

MODEL_CACHE_ROOT = "/mnt/d/model"
SDV1_5 = "stable-diffusion-v1-5/stable-diffusion-v1-5"
CLIP_VLP14 = "openai/clip-vit-large-patch14"

def main() -> None:
    # dl = DiffusionLoader(SDV1_5, MODEL_CACHE_ROOT)
    # dl.run()

    cl = CLIPLoader(CLIP_VLP14, os.path.join(MODEL_CACHE_ROOT, "CLIP"))
    cl.run()

if __name__ == "__main__":
    main()