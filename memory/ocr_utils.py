#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RapidOCR 工具脚本
提供本地 OCR 能力，支持中文/英文识别
"""

from pathlib import Path
from typing import List, Tuple, Optional
from rapidocr_onnxruntime import RapidOCR


class OCRError(Exception):
    """OCR 异常"""
    pass


class RapidOCRWrapper:
    """RapidOCR 封装类"""
    
    def __init__(self):
        self.ocr = RapidOCR()
        
    def recognize(self, img_path: str, sort_mode: str = 'tb') -> List[Tuple[str, Tuple[float, float, float, float], float]]:
        """
        识别图片中的文字
        
        Args:
            img_path: 图片路径
            sort_mode: 排序模式，'tb'(从上到下) 或 'lr'(从左到右)
            
        Returns:
            识别结果列表，每个元素为 (文本, 坐标, 置信度)
            
        Raises:
            OCRError: 识别失败
        """
        if not Path(img_path).exists():
            raise OCRError(f"图片不存在: {img_path}")
            
        result, _ = self.ocr(img_path, sort_mode=sort_mode)
        
        if not result:
            raise OCRError(f"未能识别到任何文字")
            
        return result
    
    def recognize_text_only(self, img_path: str, sort_mode: str = 'tb') -> List[str]:
        """
        仅返回识别的文本内容
        
        Args:
            img_path: 图片路径
            sort_mode: 排序模式
            
        Returns:
            文本列表
        """
        result = self.recognize(img_path, sort_mode)
        return [item[0] for item in result]
    
    def recognize_formatted(self, img_path: str, sort_mode: str = 'tb') -> str:
        """
        返回格式化的识别结果（换行符连接）
        
        Args:
            img_path: 图片路径
            sort_mode: 排序模式
            
        Returns:
            格式化文本
        """
        texts = self.recognize_text_only(img_path, sort_mode)
        return "\n".join(texts)


# 全局单例
_ocr_instance = None

def get_ocr() -> RapidOCRWrapper:
    """获取 OCR 单例"""
    global _ocr_instance
    if _ocr_instance is None:
        _ocr_instance = RapidOCRWrapper()
    return _ocr_instance


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python ocr_utils.py <图片路径>")
        sys.exit(1)
        
    img_path = sys.argv[1]
    ocr = get_ocr()
    
    try:
        print(f"正在识别: {img_path}")
        texts = ocr.recognize_text_only(img_path)
        print(f"\n识别到 {len(texts)} 个文本块:")
        for i, text in enumerate(texts, 1):
            print(f"{i}. {text}")
    except OCRError as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)