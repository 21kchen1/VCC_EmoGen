#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

r"""
@DATE    :   2026-06-01 09:25:44
@Author  :   Chen
@File    :   code/VCC_EmoGen/training/cuda_test.py
@Software:   VSCode
@Description:
    检测显卡使用情况
"""

import torch
# 检查显存使用情况
print(torch.cuda.memory_allocated()) # 输出非零值，即使没有程序运行

torch.cuda.empty_cache() # 清理未使用的显存缓存