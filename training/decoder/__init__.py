# -*- coding: utf-8 -*-
"""
Decoder 训练模块.

提供生成式召回模型的定义、训练和导出功能.
支持 GPT2 (自研) 和 Qwen3-0.6B 两种 backbone.
"""

from .model import GenerativeDecoder
from .qwen3_generative_rec import Qwen3GenerativeRec
from .train import train_decoder
from .export import export_decoder

__all__ = [
    'GenerativeDecoder',
    'Qwen3GenerativeRec',
    'train_decoder',
    'export_decoder',
]
