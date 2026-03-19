# -*- coding: utf-8 -*-
"""
Decoder (GPT2) 训练模块.

提供生成式召回模型的定义、训练和导出功能.
"""

from .model import GenerativeDecoder
from .train import train_decoder
from .export import export_decoder

__all__ = ['GenerativeDecoder', 'train_decoder', 'export_decoder']
