import os
import json
import logging
from server import PromptServer
import torch
import numpy as np
from typing import List, Union
import glob
from PIL import Image
from PIL import ImageOps
import comfy
import folder_paths
import base64
from io import BytesIO
import hashlib

# 設定基本的日誌記錄格式和級別
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TextBatchNode:
    """
    文本批次處理節點
    用於將大型文本文件按照指定分隔符分割成多個部分
    支持狀態保存和恢復，可以記住上次處理的位置
    """
    def __init__(self):
        # 初始化狀態文件路徑，用於保存處理進度
        self.state_file = os.path.join(os.path.dirname(__file__), "text_batch_state.json")
        self.state = self.load_state()
        self.reset_state()  # 添加初始化重置

    def reset_state(self):
        """重置狀態到初始值"""
        self.state = {
            "prompts": [],
            "current_index": 0,
            "last_input": "",
            "last_input_mode": "",
            "last_separator": "",
            "last_separator_type": "newline",
            "last_start_index": 0,
            "completed": False
        }
        self.save_state()

    @classmethod
    def INPUT_TYPES(cls):
        """
        定義節點的輸入參數類型和預設值
        """
        return {
            "required": {
                # 修改預設值為 text
                "input_mode": (["text", "file"], {"default": "text"}),
                # 文本文件路徑輸入
                "text_file": ("STRING", {"multiline": False, "default": "Enter the path to your text file here"}),
                # 直接輸入文本
                "input_text": ("STRING", {
                    "multiline": True, 
                    "default": "Enter your text here...",
                    "placeholder": "Enter multiple prompts, one per line"
                }),
                # 分隔符類型
                "separator_type": (["newline", "custom"], {"default": "newline"}),
                # 文本分隔符
                "separator": ("STRING", {"default": "---"}),
                # 起始索引位置
                "start_index": ("INT", {"default": 0, "min": 0, "max": 10000}),
                # 新增自動終止選項
                "auto_stop": ("BOOLEAN", {"default": True}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"}
        }

    # 定義輸出類型
    RETURN_TYPES = ("STRING", "STRING", "INT", "INT", "BOOLEAN")
    # 定義輸出名稱
    RETURN_NAMES = ("prompt", "status", "current_index", "total", "completed")
    FUNCTION = "process_text"
    CATEGORY = "TextBatch"
    OUTPUT_NODE = True  # 添加這行

    def process_text(self, input_mode, text_file, input_text, separator_type, separator, start_index, auto_stop, unique_id=None):
        """
        處理文本文件或直接輸入文本的主要方法
        參數:
            input_mode: 輸入模式（file或text）
            text_file: 要處理的文本文件路徑
            input_text: 直接輸入的文本
            separator_type: 分隔符類型（custom或newline）
            separator: 用於分割文本的分隔符
            start_index: 開始處理的索引位置
            auto_stop: 是否自動終止
            unique_id: 節點的唯一識別碼，用於更新節點顯示
        返回:
            tuple: (提示文本, 狀態信息, 當前索引, 總數, 是否完成)
        """
        try:
            # 檔案模式的驗證
            if input_mode == "file":
                if not text_file.strip() or text_file == "Enter the path to your text file here":
                    return ("", "Error: Please provide a valid file path", -1, 0, True)
                if not os.path.exists(text_file):
                    return ("", f"Error: File not found: {text_file}", -1, 0, True)
                with open(text_file, 'r', encoding='utf-8') as file:
                    content = file.read().strip()
                if not content:
                    return ("", "Error: File is empty", -1, 0, True)
                current_input = text_file
            else:
                if not input_text.strip() or input_text == "Enter your text here...":
                    return ("", "Error: Please provide input text", -1, 0, True)
                current_input = input_text
            
            # 檢查是否需要重置
            need_reset = (
                self.state.get("last_input") != current_input or
                self.state.get("last_input_mode") != input_mode or
                self.state.get("last_separator") != separator or
                self.state.get("last_separator_type") != separator_type or
                self.state.get("completed", False)
            )

            # 如果需要重置，重新加載所有數據
            if need_reset:
                self.reset_state()
                if input_mode == "file":
                    self.load_prompts(text_file, separator_type, separator)
                else:
                    self.load_text_input(input_text, separator_type, separator)
                
                # 檢查是否成功加載了提示
                if len(self.state["prompts"]) == 0:
                    return ("", "Error: No valid prompts found", -1, 0, True)
                
                self.state.update({
                    "last_input": current_input,
                    "last_input_mode": input_mode,
                    "last_separator": separator,
                    "last_separator_type": separator_type,
                    "current_index": 0
                })

            total = len(self.state["prompts"])
            if total == 0:
                return ("", "No prompts loaded", 0, 0, True)

            # 安全獲取當前索引
            current_index = min(self.state.get("current_index", 0), total - 1)
            
            try:
                # 安全獲取當前提示
                prompt = self.state["prompts"][current_index]
            except IndexError:
                # 如果發生索引錯誤，重置到最後一個有效索引
                current_index = total - 1
                prompt = self.state["prompts"][current_index]
                self.state["current_index"] = current_index

            # 檢查是否完成
            is_last = current_index >= total - 1
            
            # 更新狀態
            if not is_last and auto_stop:
                self.state["current_index"] = current_index + 1
                completed = False
            else:
                completed = True
            
            self.state["completed"] = completed
            
            # 生成狀態信息
            status = f"Processing {current_index + 1}/{total}"
            if input_mode == "file":
                status += f" | File: {os.path.basename(text_file)}"
            if completed:
                status += " | Completed"

            # 保存狀態
            self.save_state()

            # 更新節點顯示的當前索引
            if not completed:
                PromptServer.instance.send_sync("textbatch-node-feedback", 
                    {"node_id": unique_id, "widget_name": "start_index", "type": "int", "value": self.state["current_index"]})

            return (prompt, status, current_index, total, completed)

        except Exception as e:
            logger.error(f"Error in process_text: {str(e)}")
            return ("", f"Error: {str(e)}", -1, 0, True)

    def load_prompts(self, text_file, separator_type, separator):
        """
        從文件中加載並分割提示文本
        參數:
            text_file: 文本文件路徑
            separator_type: 分隔符類型（custom或newline）
            separator: 用於分割文本的分隔符
        """
        try:
            with open(text_file, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # 根據分隔符類型選擇分割方式
            if separator_type == "newline":
                self.state["prompts"] = [prompt.strip() for prompt in content.splitlines() if prompt.strip()]
            else:
                self.state["prompts"] = [prompt.strip() for prompt in content.split(separator) if prompt.strip()]
            
            logger.info(f"Loaded {len(self.state['prompts'])} prompts from {text_file}")
        except Exception as e:
            logger.error(f"Error loading prompts: {str(e)}")
            raise

    def load_text_input(self, input_text, separator_type, separator):
        """
        處理直接輸入的文本
        參數:
            input_text: 輸入的文本
            separator_type: 分隔符類型（custom或newline）
            separator: 用於分割文本的分隔符
        """
        try:
            # 根據分隔符類型選擇分割方式
            if separator_type == "newline":
                self.state["prompts"] = [prompt.strip() for prompt in input_text.splitlines() if prompt.strip()]
            else:
                self.state["prompts"] = [prompt.strip() for prompt in input_text.split(separator) if prompt.strip()]
            
            logger.info(f"Loaded {len(self.state['prompts'])} prompts from direct input")
        except Exception as e:
            logger.error(f"Error processing input text: {str(e)}")
            raise

    def load_state(self):
        """
        從狀態文件中加載之前的處理狀態
        """
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading state file: {str(e)}")
        return {
            "prompts": [], 
            "current_index": 0, 
            "last_input": "", 
            "last_input_mode": "file",
            "last_separator": "",
            "last_separator_type": "newline",
            "last_start_index": 0,
            "completed": False  # 添加完成狀態標記
        }

    def save_state(self):
        """
        將當前處理狀態保存到狀態文件
        """
        try:
            with open(self.state_file, 'w') as f:
                json.dump(self.state, f)
        except Exception as e:
            logger.error(f"Error saving state file: {str(e)}")

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """
        控制節點的執行頻率
        返回不同的值會觸發節點重新執行
        """
        current_index = kwargs.get("current_index", 0)
        total = kwargs.get("total", 0)
        completed = kwargs.get("completed", False)
        auto_stop = kwargs.get("auto_stop", True)
        
        # 如果啟用了自動停止且未完成，返回 float("nan") 觸發重新執行
        if auto_stop and not completed and current_index < total - 1:
            return float("nan")  # 使用 nan 確保每次都會觸發重新執行
            
        return current_index  # 返回當前索引，不會觸發重新執行

class TextSplitCounterNode:
    """
    文本分割計數節點
    用於計算分割後的文本總數
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_mode": (["file", "text"], {"default": "file"}),
                "text_file": ("STRING", {"multiline": False, "default": "Enter the path to your text file here"}),
                "input_text": ("STRING", {
                    "multiline": True,
                    "default": "Enter your text here...",
                }),
                "separator_type": (["newline", "custom"], {"default": "newline"}),
                "separator": ("STRING", {"default": "---"}),
            }
        }

    RETURN_TYPES = ("INT", "STRING",)
    RETURN_NAMES = ("count", "status")
    FUNCTION = "count_splits"
    CATEGORY = "TextBatch"

    def count_splits(self, input_mode, text_file, input_text, separator_type, separator):
        try:
            if input_mode == "file":
                if not os.path.exists(text_file):
                    return (0, f"Error: File not found: {text_file}")
                with open(text_file, 'r', encoding='utf-8') as file:
                    content = file.read()
            else:
                content = input_text

            # 根據分隔符類型計算分割數
            if separator_type == "newline":
                splits = [x.strip() for x in content.splitlines() if x.strip()]
            else:
                splits = [x.strip() for x in content.split(separator) if x.strip()]

            count = len(splits)
            status = f"Total splits: {count}"
            return (count, status)
        except Exception as e:
            logger.error(f"Error in count_splits: {str(e)}")
            return (0, f"Error: {str(e)}")

class TextQueueProcessor:
    """處理文字佇列的節點"""
    def __init__(self):
        self.state_file = os.path.join(os.path.dirname(__file__), "text_queue_processor_state.json")
        self.state = self.load_state()
        self.reset_state()

    def reset_state(self):
        """重置狀態到初始值"""
        self.state = {
            "current_index": 0,
            "last_input": "",
            "completed": False
        }
        self.save_state()

    def load_state(self):
        """從狀態文件中加載之前的處理狀態"""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading state file: {str(e)}")
        return {
            "current_index": 0,
            "last_input": "",
            "completed": False
        }

    def save_state(self):
        """將當前處理狀態保存到狀態文件"""
        try:
            with open(self.state_file, 'w') as f:
                json.dump(self.state, f)
        except Exception as e:
            logger.error(f"Error saving state file: {str(e)}")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "default": "1girl\n1cat\n1dog",
                    "placeholder": "輸入提示詞，可用分隔符或換行分割"
                }),
                "separator_type": (["newline", "custom"], {"default": "newline"}),
                "separator": ("STRING", {"default": ","}),
                "start_index": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "trigger_next": ("BOOLEAN", {"default": True, "label_on": "Trigger", "label_off": "Don't trigger"}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            }
        }

    RETURN_TYPES = ("STRING", "INT", "INT", "BOOLEAN", "STRING")
    RETURN_NAMES = ("text", "current_index", "total", "completed", "status")
    FUNCTION = "process"
    CATEGORY = "TextBatch"
    OUTPUT_NODE = True

    def process(self, text, separator_type, separator, start_index, trigger_next, unique_id, prompt=None, extra_pnginfo=None):
        try:
            # 檢查是否需要重置
            need_reset = (
                self.state.get("last_input") != text or
                self.state.get("completed", False) or
                (prompt and extra_pnginfo)
            )

            if need_reset:
                self.reset_state()
                self.state["last_input"] = text

            # 根據分隔符類型分割文本
            if separator_type == "newline":
                lines = [line.strip() for line in text.splitlines() if line.strip()]
            else:
                lines = [line.strip() for line in text.split(separator) if line.strip()]
            
            total = len(lines)

            if total == 0:
                return ("", -1, 0, True, "No valid text found")

            # 獲取當前索引
            current_index = min(max(start_index, self.state.get("current_index", 0)), total - 1)
            
            # 獲取當前行
            current_text = lines[current_index]

            # 檢查是否是最後一行
            is_last = current_index >= total - 1
            
            # 更新狀態
            if not is_last and trigger_next:
                self.state["current_index"] = current_index + 1
                completed = False
                # 使用自己的事件名稱
                PromptServer.instance.send_sync("textbatch-add-queue", {})
            else:
                completed = True
                self.state["current_index"] = 0

            self.state["completed"] = completed
            self.save_state()

            # 生成狀態信息
            status = f"Processing {current_index + 1}/{total}"
            if completed:
                status += " | Completed"

            # 更新節點顯示的當前索引
            if not completed:
                PromptServer.instance.send_sync("textbatch-node-feedback", 
                    {"node_id": unique_id, "widget_name": "start_index", "type": "int", "value": self.state["current_index"]})

            return (current_text, current_index, total, completed, status)

        except Exception as e:
            logger.error(f"Error in process: {str(e)}")
            return ("", -1, 0, True, f"Error: {str(e)}")

class ImageQueueProcessor:
    """處理圖片佇列的節點"""
    def __init__(self):
        self.state_file = os.path.join(os.path.dirname(__file__), "image_queue_processor_state.json")
        self.state = self.load_state()
        self.reset_state()

    def reset_state(self):
        self.state = {
            "current_index": 0,
            "last_input": "",
            "completed": False
        }
        self.save_state()

    def load_state(self):
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading state file: {str(e)}")
        return {
            "current_index": 0,
            "last_input": "",
            "completed": False
        }

    def save_state(self):
        try:
            with open(self.state_file, 'w') as f:
                json.dump(self.state, f)
        except Exception as e:
            logger.error(f"Error saving state file: {str(e)}")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),  # 接受多張圖片的輸入
                "start_index": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "trigger_next": ("BOOLEAN", {"default": True, "label_on": "Trigger", "label_off": "Don't trigger"}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "BOOLEAN", "STRING")
    RETURN_NAMES = ("image", "current_index", "total", "completed", "status")
    FUNCTION = "process"
    CATEGORY = "TextBatch"
    OUTPUT_NODE = True

    def process(self, images, start_index, trigger_next, unique_id, prompt=None, extra_pnginfo=None):
        try:
            # 確保輸入是 tensor 並且格式正確
            if not isinstance(images, torch.Tensor):
                return (None, -1, 0, True, "Invalid input: not a tensor")

            # 處理單張圖片的情況
            if len(images.shape) == 3:
                images = images.unsqueeze(0)

            # 獲取總數
            total = images.shape[0]
            if total == 0:
                return (None, -1, 0, True, "No images found")

            # 生成唯一的輸入標識符
            input_hash = str(hash(str(images.shape)))

            # 檢查是否需要重置
            need_reset = (
                self.state.get("last_input") != input_hash or
                self.state.get("completed", False) or
                (prompt and extra_pnginfo)
            )

            if need_reset:
                self.reset_state()
                self.state["last_input"] = input_hash
                current_index = start_index
            else:
                current_index = min(max(start_index, self.state.get("current_index", 0)), total - 1)

            # 獲取當前圖片
            current_image = images[current_index:current_index+1]

            # 檢查是否是最後一張
            is_last = current_index >= total - 1
            
            # 更新狀態
            if not is_last and trigger_next:
                next_index = current_index + 1
                self.state["current_index"] = next_index
                completed = False
                
                # 只有在非最後一張且啟用 trigger_next 時才發送佇列事件
                if next_index < total:
                    PromptServer.instance.send_sync("textbatch-add-queue", {})
            else:
                completed = True
                self.state["current_index"] = 0
                self.state["completed"] = True

            self.state["completed"] = completed
            self.save_state()

            # 生成狀態信息
            status = f"Processing {current_index + 1}/{total}"
            if completed:
                status += " | Completed"

            # 更新節點顯示的當前索引
            if not completed:
                PromptServer.instance.send_sync("textbatch-node-feedback", 
                    {"node_id": unique_id, "widget_name": "start_index", "type": "int", "value": self.state["current_index"]})

            return (current_image, current_index, total, completed, status)

        except Exception as e:
            logger.error(f"Error in process: {str(e)}")
            return (None, -1, 0, True, f"Error: {str(e)}")

class ImageInfoExtractorNode:
    """圖片資訊提取節點
    用於提取圖片的基本資訊，包括尺寸、檔名等
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),  # 接受圖片輸入
                "image_path": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "placeholder": "可選：輸入圖片路徑以獲取更多資訊"
                }),
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("width", "height", "batch_size", "channels", "file_name", "file_format", "file_size", "color_mode", "status")
    FUNCTION = "extract_info"
    CATEGORY = "TextBatch"

    def extract_info(self, images, image_path=""):
        try:
            # 確保輸入是 tensor 並且格式正確
            if not isinstance(images, torch.Tensor):
                return (0, 0, 0, 0, "", "", "", "", "錯誤：輸入不是有效的圖片張量")

            # 處理單張圖片的情況
            if len(images.shape) == 3:
                images = images.unsqueeze(0)

            # 獲取基本資訊
            batch_size = images.shape[0]
            height = images.shape[1]
            width = images.shape[2]
            channels = images.shape[3]

            # 預設值
            file_name = ""
            file_format = ""
            file_size = ""
            color_mode = ""

            # 如果提供了圖片路徑，嘗試獲取額外資訊
            if image_path and os.path.exists(image_path):
                try:
                    file_size = f"{os.path.getsize(image_path) / (1024 * 1024):.2f}MB"
                    file_name = os.path.basename(image_path)
                    
                    # 使用 PIL 讀取圖片以獲取更多資訊
                    with Image.open(image_path) as img:
                        color_mode = img.mode
                        file_format = img.format or os.path.splitext(image_path)[1]
                except Exception as e:
                    return (width, height, batch_size, channels, "", "", "", "", f"讀取檔案資訊時發生錯誤: {str(e)}")

            status = f"成功提取 {batch_size} 張圖片的資訊"
            return (width, height, batch_size, channels, file_name, file_format, file_size, color_mode, status)

        except Exception as e:
            logger.error(f"提取圖片資訊時發生錯誤: {str(e)}")
            return (0, 0, 0, 0, "", "", "", "", f"錯誤: {str(e)}")

class PathParserNode:
    """路徑解析節點
    用於解析檔案路徑，分離出檔名和資料夾路徑
    支援絕對路徑和相對路徑
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "placeholder": "輸入完整檔案路徑"
                }),
                "normalize_path": ("BOOLEAN", {
                    "default": True,
                    "label_on": "是",
                    "label_off": "否"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("file_name_no_ext", "folder_path", "extension", "absolute_path", "status")
    FUNCTION = "parse_path"
    CATEGORY = "TextBatch"

    def parse_path(self, file_path: str, normalize_path: bool = True):
        try:
            if not file_path:
                return ("", "", "", "", "錯誤：未提供檔案路徑")

            # 檢查輸入類型並轉換為字串
            if isinstance(file_path, (list, tuple)):
                # 如果是列表或元組，取第一個元素
                if len(file_path) > 0:
                    file_path = str(file_path[0])
                else:
                    return ("", "", "", "", "錯誤：空列表")
            else:
                file_path = str(file_path)

            # 標準化路徑分隔符
            file_path = os.path.normpath(file_path)
            
            # 轉換為絕對路徑
            absolute_path = os.path.abspath(file_path)
            
            # 取得檔案名稱（含副檔名）和資料夾路徑
            folder_path = os.path.dirname(absolute_path)
            full_filename = os.path.basename(absolute_path)
            
            # 分離檔名和副檔名
            file_name_no_ext, extension = os.path.splitext(full_filename)
            
            # 如果副檔名存在，移除開頭的點
            extension = extension[1:] if extension.startswith('.') else extension
            
            # 處理路徑格式
            if normalize_path:
                # 使用正斜線
                folder_path = folder_path.replace('\\', '/')
                absolute_path = absolute_path.replace('\\', '/')
            
            # 生成狀態信息
            status_parts = []
            status_parts.append(f"檔名: {file_name_no_ext}")
            if extension:
                status_parts.append(f"副檔名: {extension}")
            status_parts.append(f"目錄: {folder_path}")
            
            status = " | ".join(status_parts)

            return (
                file_name_no_ext,  # 無副檔名的檔名
                folder_path,       # 資料夾路徑
                extension,         # 副檔名（不含點號）
                absolute_path,     # 完整絕對路徑
                status            # 狀態信息
            )

        except Exception as e:
            logger.error(f"解析路徑時發生錯誤: {str(e)}")
            return ("", "", "", "", f"錯誤: {str(e)}")
        
class LoadImagesFromDirBatchM:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "directory": ("STRING", {"default": ""}),
            },
            "optional": {
                "image_load_cap": ("INT", {"default": 0, "min": 0, "step": 1}),
                "start_index": ("INT", {"default": 0, "min": -1, "step": 1}),
                "load_always": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("image", "mask", "count", "filenames", "full_paths", "path_only")
    FUNCTION = "load_images"

    CATEGORY = "image"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        if 'load_always' in kwargs and kwargs['load_always']:
            return float("NaN")
        else:
            return hash(frozenset(kwargs))

    def load_images(self, directory: str, image_load_cap: int = 0, start_index: int = 0, load_always=False):
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Directory '{directory} cannot be found.'")
        dir_files = os.listdir(directory)
        if len(dir_files) == 0:
            raise FileNotFoundError(f"No files in directory '{directory}'.")

        # Filter files by extension
        valid_extensions = ['.jpg', '.jpeg', '.png', '.webp']
        dir_files = [f for f in dir_files if any(f.lower().endswith(ext) for ext in valid_extensions)]

        dir_files = sorted(dir_files)
        dir_files = [os.path.join(directory, x) for x in dir_files]

        # start at start_index
        dir_files = dir_files[start_index:]

        images = []
        masks = []
        filenames = []  # 檔名列表（不含路徑）
        full_paths = []  # 完整路徑列表
        path_only = directory  # =輸入路徑
        limit_images = False
        if image_load_cap > 0:
            limit_images = True
        image_count = 0

        has_non_empty_mask = False

        for image_path in dir_files:
            if os.path.isdir(image_path):
                continue
            if limit_images and image_count >= image_load_cap:
                break
            i = Image.open(image_path)
            i = ImageOps.exif_transpose(i)
            image = i.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
                has_non_empty_mask = True
            else:
                mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
            images.append(image)
            masks.append(mask)
            # 儲存檔名（不含路徑）
            filenames.append(os.path.basename(image_path))
            # 儲存標準化的完整路徑
            norm_path = os.path.abspath(image_path).replace('\\', '/')
            full_paths.append(norm_path)
            image_count += 1

        if len(images) == 1:
            # 單張圖片時返回單一檔名和路徑
            return (images[0], masks[0], 1, filenames[0], full_paths[0], path_only)

        elif len(images) > 1:
            image1 = images[0]
            mask1 = None

            for image2 in images[1:]:
                if image1.shape[1:] != image2.shape[1:]:
                    image2 = comfy.utils.common_upscale(image2.movedim(-1, 1), image1.shape[2], image1.shape[1], "bilinear", "center").movedim(1, -1)
                image1 = torch.cat((image1, image2), dim=0)

            for mask2 in masks:
                if has_non_empty_mask:
                    if image1.shape[1:3] != mask2.shape:
                        mask2 = torch.nn.functional.interpolate(mask2.unsqueeze(0).unsqueeze(0), size=(image1.shape[1], image1.shape[2]), mode='bilinear', align_corners=False)
                        mask2 = mask2.squeeze(0)
                    else:
                        mask2 = mask2.unsqueeze(0)
                else:
                    mask2 = mask2.unsqueeze(0)

                if mask1 is None:
                    mask1 = mask2
                else:
                    mask1 = torch.cat((mask1, mask2), dim=0)

            # 多張圖片時返回以逗號分隔的檔名和路徑字串
            return (image1, mask1, len(images), ",".join(filenames), ",".join(full_paths), path_only)

class ImageFilenameProcessor:
    """圖片檔名處理節點
    用於處理單張或多張圖片的檔名，可以根據索引獲取特定檔名
    提供完整檔名、無副檔名、副檔名和完整路徑等多種輸出
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "filenames": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "placeholder": "逗號分隔的檔名或路徑列表"
                }),
                "index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "step": 1,
                    "display": "number"
                }),
                "directory": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "placeholder": "可選：指定檔案目錄路徑"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "INT", "STRING")
    RETURN_NAMES = ("filename", "name_no_ext", "extension", "full_path", "total_files", "status")
    FUNCTION = "process_filename"
    CATEGORY = "TextBatch"

    def process_filename(self, filenames: str, index: int = 0, directory: str = ""):
        try:
            # 處理空輸入
            if not filenames.strip():
                return ("", "", "", "", 0, "錯誤：沒有輸入檔名")

            # 分割檔名列表
            if '\n' in filenames:
                path_list = [f.strip() for f in filenames.split("\n") if f.strip()]
            else:
                path_list = [f.strip() for f in filenames.split(",") if f.strip()]
            total_files = len(path_list)

            # 處理空列表
            if total_files == 0:
                return ("", "", "", "", 0, "錯誤：沒有有效的檔名")

            # 確保索引在有效範圍內
            if index < 0:
                index = 0
            if index >= total_files:
                index = total_files - 1

            # 獲取指定索引的路徑
            selected_path = path_list[index]
            
            # 從路徑中提取檔名
            filename = os.path.basename(selected_path)
            
            # 分離檔名和副檔名
            name_no_ext, extension = os.path.splitext(filename)
            # 確保副檔名不包含點號
            extension = extension[1:] if extension.startswith('.') else extension
            
            # 處理完整路徑
            if directory:
                # 如果提供了目錄，使用該目錄和檔名組合
                directory = os.path.normpath(directory)
                full_path = os.path.join(directory, filename)
            else:
                # 否則使用輸入的路徑
                full_path = selected_path
            
            # 標準化路徑格式（使用正斜線）
            full_path = os.path.normpath(full_path).replace('\\', '/')

            # 生成狀態信息
            status = f"成功獲取第 {index + 1}/{total_files} 個檔名"
            if directory:
                status += f" (目錄: {directory})"

            return (
                filename,          # 完整檔名（含副檔名）
                name_no_ext,       # 無副檔名
                extension,         # 副檔名（不含點號）
                full_path,         # 完整路徑
                total_files,       # 總檔案數
                status            # 狀態信息
            )

        except Exception as e:
            logger.error(f"處理檔名時發生錯誤: {str(e)}")
            return ("", "", "", "", 0, f"錯誤: {str(e)}")

class LoadImageByIndex:
    """根據索引延遲載入單張圖片
    只讀取指定索引的圖片到記憶體，避免一次性載入所有圖片
    適合用於預覽、選擇性處理等場景
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory": ("STRING", {"default": ""}),
                "index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "step": 1,
                    "display": "number"
                }),
            },
            "optional": {
                "load_always": ("BOOLEAN", {
                    "default": False, 
                    "label_on": "enabled", 
                    "label_off": "disabled"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING", "INT", "STRING")
    RETURN_NAMES = ("image", "mask", "filename", "full_path", "total_files", "status")
    FUNCTION = "load_image_by_index"
    CATEGORY = "image"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        if 'load_always' in kwargs and kwargs['load_always']:
            return float("NaN")
        else:
            return hash(frozenset(kwargs.items()))

    def load_image_by_index(self, directory: str, index: int = 0, load_always=False):
        try:
            # 驗證目錄
            if not os.path.isdir(directory):
                raise FileNotFoundError(f"目錄 '{directory}' 不存在")
            
            # 獲取所有圖片檔案列表（只讀取檔名，不載入圖片）
            dir_files = os.listdir(directory)
            valid_extensions = ['.jpg', '.jpeg', '.png', '.webp']
            dir_files = [f for f in dir_files if any(f.lower().endswith(ext) for ext in valid_extensions)]
            
            if len(dir_files) == 0:
                raise FileNotFoundError(f"目錄 '{directory}' 中沒有圖片檔案")
            
            # 排序檔案列表
            dir_files = sorted(dir_files)
            total_files = len(dir_files)
            
            # 確保索引在有效範圍內
            if index < 0:
                index = 0
            if index >= total_files:
                index = total_files - 1
            
            # 只載入指定索引的圖片
            filename = dir_files[index]
            image_path = os.path.join(directory, filename)
            
            # 載入圖片
            i = Image.open(image_path)
            i = ImageOps.exif_transpose(i)
            image = i.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            
            # 處理遮罩
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((image.shape[1], image.shape[2]), dtype=torch.float32, device="cpu")
            
            # 標準化完整路徑
            full_path = os.path.abspath(image_path).replace('\\', '/')
            
            # 狀態信息
            status = f"成功載入第 {index + 1}/{total_files} 張圖片"
            
            return (image, mask, filename, full_path, total_files, status)
            
        except Exception as e:
            logger.error(f"載入圖片時發生錯誤: {str(e)}")
            # 返回空圖片
            empty_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            empty_mask = torch.zeros((64, 64), dtype=torch.float32)
            return (empty_image, empty_mask, "", "", 0, f"錯誤: {str(e)}")

class LoadImagesFromDirLazy:
    """延遲載入模式 - 先獲取檔案列表，返回路徑資訊
    只掃描目錄獲取檔案列表，不載入圖片到記憶體
    配合 LoadImageByIndex 使用實現完整的延遲載入流程
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory": ("STRING", {"default": ""}),
            },
            "optional": {
                "start_index": ("INT", {"default": 0, "min": 0, "step": 1}),
                "max_files": ("INT", {"default": 0, "min": 0, "step": 1}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "INT", "STRING")
    RETURN_NAMES = ("filenames", "full_paths", "directory", "total_files", "status")
    FUNCTION = "scan_directory"
    CATEGORY = "image"

    def scan_directory(self, directory: str, start_index: int = 0, max_files: int = 0):
        try:
            # 驗證目錄
            if not os.path.isdir(directory):
                raise FileNotFoundError(f"目錄 '{directory}' 不存在")
            
            # 獲取所有圖片檔案列表
            dir_files = os.listdir(directory)
            valid_extensions = ['.jpg', '.jpeg', '.png', '.webp']
            dir_files = [f for f in dir_files if any(f.lower().endswith(ext) for ext in valid_extensions)]
            
            if len(dir_files) == 0:
                raise FileNotFoundError(f"目錄 '{directory}' 中沒有圖片檔案")
            
            # 排序
            dir_files = sorted(dir_files)
            
            # 應用起始索引
            if start_index > 0:
                dir_files = dir_files[start_index:]
            
            # 應用數量限制
            if max_files > 0:
                dir_files = dir_files[:max_files]
            
            total_files = len(dir_files)
            
            # 生成檔名列表和完整路徑列表
            filenames = ",".join(dir_files)
            full_paths = ",".join([os.path.abspath(os.path.join(directory, f)).replace('\\', '/') for f in dir_files])
            
            status = f"掃描到 {total_files} 個圖片檔案（未載入到記憶體）"
            if start_index > 0:
                status += f"，起始索引: {start_index}"
            if max_files > 0:
                status += f"，限制數量: {max_files}"
            
            return (filenames, full_paths, directory, total_files, status)
            
        except Exception as e:
            logger.error(f"掃描目錄時發生錯誤: {str(e)}")
            return ("", "", "", 0, f"錯誤: {str(e)}")

class ImageQueueProcessorPro:
    """圖片隊列處理器 Pro 版本
    不需要預先載入圖片，直接從資料夾按需載入
    適合處理大量圖片，記憶體友善
    """
    def __init__(self):
        self.state_file = os.path.join(os.path.dirname(__file__), "image_queue_processor_pro_state.json")
        self.state = self.load_state()

    def reset_state(self, queue_id="default"):
        """重置指定隊列的狀態"""
        if queue_id not in self.state:
            self.state[queue_id] = {}
        self.state[queue_id] = {
            "current_index": 0,
            "last_directory": "",
            "total_files": 0,
            "completed": False
        }
        self.save_state()

    def load_state(self):
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"載入狀態檔案錯誤: {str(e)}")
        return {}

    def save_state(self):
        try:
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(self.state, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"儲存狀態檔案錯誤: {str(e)}")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "圖片資料夾路徑"
                }),
                "start_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10000,
                    "step": 1
                }),
                "trigger_next": ("BOOLEAN", {
                    "default": True,
                    "label_on": "啟用自動觸發",
                    "label_off": "停止自動觸發"
                }),
            },
            "optional": {
                "queue_id": ("STRING", {
                    "default": "default",
                    "multiline": False,
                    "placeholder": "隊列ID（可選，用於多個隊列）"
                }),
                "reset_queue": ("BOOLEAN", {
                    "default": False,
                    "label_on": "重置隊列",
                    "label_off": "繼續處理"
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "BOOLEAN", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("image", "mask", "current_index", "total", "completed", "filename", "full_path", "status")
    FUNCTION = "process"
    CATEGORY = "TextBatch"
    OUTPUT_NODE = True

    def process(self, directory, start_index, trigger_next, queue_id="default", reset_queue=False, unique_id=None):
        try:
            # 驗證目錄
            if not os.path.isdir(directory):
                empty_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
                empty_mask = torch.zeros((64, 64), dtype=torch.float32)
                return (empty_image, empty_mask, -1, 0, True, "", "", f"錯誤：目錄不存在 '{directory}'")
            
            # 掃描目錄獲取圖片列表（只讀取檔名，不載入圖片）
            dir_files = os.listdir(directory)
            valid_extensions = ['.jpg', '.jpeg', '.png', '.webp']
            dir_files = [f for f in dir_files if any(f.lower().endswith(ext) for ext in valid_extensions)]
            
            if len(dir_files) == 0:
                empty_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
                empty_mask = torch.zeros((64, 64), dtype=torch.float32)
                return (empty_image, empty_mask, -1, 0, True, "", "", f"錯誤：目錄中沒有圖片 '{directory}'")
            
            # 排序檔案
            dir_files = sorted(dir_files)
            total_files = len(dir_files)
            
            # 初始化隊列狀態
            if queue_id not in self.state:
                self.reset_state(queue_id)
            
            queue_state = self.state[queue_id]
            
            # 檢查是否需要重置（目錄變更或手動重置）
            directory_changed = queue_state.get("last_directory") != directory
            if reset_queue or directory_changed or queue_state.get("completed", False):
                self.reset_state(queue_id)
                queue_state = self.state[queue_id]
                queue_state["last_directory"] = directory
                queue_state["total_files"] = total_files
                current_index = start_index
            else:
                # 繼續處理
                current_index = queue_state.get("current_index", start_index)
            
            # 確保索引有效
            if current_index < 0:
                current_index = 0
            if current_index >= total_files:
                current_index = total_files - 1
            
            # 載入當前索引的圖片（延遲載入，只載入這一張）
            filename = dir_files[current_index]
            image_path = os.path.join(directory, filename)
            
            try:
                # 載入圖片
                i = Image.open(image_path)
                i = ImageOps.exif_transpose(i)
                image = i.convert("RGB")
                image = np.array(image).astype(np.float32) / 255.0
                image = torch.from_numpy(image)[None,]
                
                # 處理遮罩
                if 'A' in i.getbands():
                    mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                    mask = 1. - torch.from_numpy(mask)
                else:
                    mask = torch.zeros((image.shape[1], image.shape[2]), dtype=torch.float32, device="cpu")
                
                # 標準化完整路徑
                full_path = os.path.abspath(image_path).replace('\\', '/')
                
            except Exception as e:
                logger.error(f"載入圖片錯誤: {str(e)}")
                empty_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
                empty_mask = torch.zeros((64, 64), dtype=torch.float32)
                return (empty_image, empty_mask, current_index, total_files, True, filename, "", f"錯誤：無法載入圖片 {filename}")
            
            # 檢查是否完成
            is_last = current_index >= total_files - 1
            
            # 更新狀態
            if not is_last and trigger_next:
                next_index = current_index + 1
                queue_state["current_index"] = next_index
                queue_state["completed"] = False
                self.save_state()
                
                # 觸發下一個隊列任務
                PromptServer.instance.send_sync("textbatch-add-queue", {})
                completed = False
            else:
                queue_state["current_index"] = 0
                queue_state["completed"] = True
                self.save_state()
                completed = True
            
            # 生成狀態信息
            status = f"處理中 {current_index + 1}/{total_files} | {filename}"
            if completed:
                status += " | ✅ 完成"
            
            # 更新前端顯示
            if not completed and unique_id:
                try:
                    PromptServer.instance.send_sync("textbatch-node-feedback", {
                        "node_id": unique_id,
                        "widget_name": "start_index",
                        "type": "int",
                        "value": next_index
                    })
                except Exception as e:
                    logger.error(f"更新前端顯示錯誤: {str(e)}")
            
            return (image, mask, current_index, total_files, completed, filename, full_path, status)
            
        except Exception as e:
            logger.error(f"處理錯誤: {str(e)}")
            empty_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            empty_mask = torch.zeros((64, 64), dtype=torch.float32)
            return (empty_image, empty_mask, -1, 0, True, "", "", f"錯誤: {str(e)}")

class ImageQueueProcessorPlus:
    """處理圖片佇列的增強節點
    支持兩個圖片輸入，可以將新圖片合併到佇列中實現循環處理
    支持自動停止條件（最大索引或開關控制）
    """
    def __init__(self):
        self.state_file = os.path.join(os.path.dirname(__file__), "image_queue_processor_plus_state.json")
        self.state = self.load_state()

    def calculate_image_md5(self, image_tensor):
        """計算圖片張量的MD5值"""
        try:
            # 將張量轉換為 numpy 陣列並計算 MD5
            img_np = image_tensor.cpu().numpy()
            img_bytes = img_np.tobytes()
            md5_hash = hashlib.md5(img_bytes).hexdigest()
            return md5_hash
        except Exception as e:
            logger.error(f"Error calculating MD5: {str(e)}")
            return ""

    def get_filename_and_path(self, index, filenames, file_paths):
        """從逗號分隔的字串中獲取指定索引的檔名和路徑"""
        try:
            filename = ""
            file_path = ""
            
            if filenames and filenames.strip():
                filename_list = [f.strip() for f in filenames.split(',') if f.strip()]
                if index < len(filename_list):
                    filename = filename_list[index]
            
            if file_paths and file_paths.strip():
                path_list = [p.strip() for p in file_paths.split(',') if p.strip()]
                if index < len(path_list):
                    file_path = path_list[index]
            
            return filename, file_path
        except Exception as e:
            logger.error(f"Error getting filename and path: {str(e)}")
            return "", ""

    def reset_state(self, queue_id="default"):
        """重置狀態到初始值"""
        global _IMAGE_QUEUE_STORAGE
        self.state = {
            "current_index": 0,
            "last_input": "",
            "completed": False,
            "queue_count": 0
        }
        # 清空全局佇列
        if queue_id in _IMAGE_QUEUE_STORAGE:
            del _IMAGE_QUEUE_STORAGE[queue_id]
        self.save_state()

    def load_state(self):
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading state file: {str(e)}")
        return {
            "current_index": 0,
            "last_input": "",
            "completed": False,
            "queue_count": 0
        }

    def save_state(self):
        try:
            with open(self.state_file, 'w') as f:
                json.dump(self.state, f)
        except Exception as e:
            logger.error(f"Error saving state file: {str(e)}")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),  # 初始圖片
                "start_index": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "max_index": ("INT", {"default": 10, "min": 1, "max": 10000}),  # 最大執行索引
                "trigger_next": ("BOOLEAN", {"default": True, "label_on": "Trigger", "label_off": "Don't trigger"}),
                "enable_loop": ("BOOLEAN", {"default": True, "label_on": "Enabled", "label_off": "Disabled"}),  # 循環開關
            },
            "optional": {
                "additional_image": ("IMAGE",),  # 額外的圖片輸入
                "data_tmp_var": ("STRING", {"default": "", "multiline": False}),  # 從 DataTmpSet 讀取的變數名稱
                "use_data_tmp": ("BOOLEAN", {"default": False}),  # 是否使用 DataTmpSet 的資料
                "filenames": ("STRING", {"default": "", "multiline": False}),  # 逗號分隔的檔名列表
                "file_paths": ("STRING", {"default": "", "multiline": False}),  # 逗號分隔的路徑列表
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "BOOLEAN", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("image", "current_index", "total", "completed", "status", "filename", "file_path", "md5")
    FUNCTION = "process"
    CATEGORY = "TextBatch"
    OUTPUT_NODE = True

    def process(self, images, start_index, max_index, trigger_next, enable_loop, unique_id, 
                additional_image=None, data_tmp_var="", use_data_tmp=False, filenames="", file_paths="", prompt=None, extra_pnginfo=None):
        try:
            global _IMAGE_QUEUE_STORAGE, _DATA_TEMP_STORAGE
            
            # 使用 unique_id 作為佇列標識
            queue_id = unique_id if unique_id else "default"
            
            # 確保輸入是 tensor 並且格式正確
            if not isinstance(images, torch.Tensor):
                # 創建一個空白圖片作為錯誤返回
                error_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
                return (error_image, -1, 0, True, "Invalid input: not a tensor", "", "", "")

            # 處理單張圖片的情況
            if len(images.shape) == 3:
                images = images.unsqueeze(0)

            # 生成唯一的輸入標識符
            input_hash = str(hash(str(images.shape)))

            # 檢查是否需要重置（images 輸入變化時重置）
            need_reset = (
                self.state.get("last_input") != input_hash or
                self.state.get("completed", False) or
                (prompt and extra_pnginfo)
            )

            if need_reset:
                self.reset_state(queue_id)
                self.state["last_input"] = input_hash
                self.state["last_data_tmp_size"] = 0
                # 初始化全局佇列（只用 images）
                _IMAGE_QUEUE_STORAGE[queue_id] = [images[i:i+1] for i in range(images.shape[0])]
                current_index = start_index
                logger.info(f"Reset queue for {queue_id}, initialized with {images.shape[0]} images")
            else:
                # 使用全局佇列
                if queue_id not in _IMAGE_QUEUE_STORAGE:
                    _IMAGE_QUEUE_STORAGE[queue_id] = [images[i:i+1] for i in range(images.shape[0])]
                current_index = self.state.get("current_index", start_index)

            # ========== 在處理前先檢查並載入 data_tmp_var 的新內容 ==========
            # 檢查 data_tmp_var 的當前大小
            data_tmp_size = 0
            if use_data_tmp and data_tmp_var and data_tmp_var.strip():
                if data_tmp_var in _DATA_TEMP_STORAGE:
                    tmp_data = _DATA_TEMP_STORAGE[data_tmp_var]
                    if isinstance(tmp_data, list):
                        data_tmp_size = len(tmp_data)
                    elif isinstance(tmp_data, torch.Tensor):
                        data_tmp_size = tmp_data.shape[0] if len(tmp_data.shape) >= 1 else 1

            # 記錄上次的 data_tmp_size
            last_data_tmp_size = self.state.get("last_data_tmp_size", 0)
            data_tmp_changed = (data_tmp_size != last_data_tmp_size)

            # 如果 data_tmp_var 變化了，重新載入
            if use_data_tmp and data_tmp_var and data_tmp_var.strip() and data_tmp_var in _DATA_TEMP_STORAGE:
                if data_tmp_changed or need_reset:
                    # 重置佇列到初始 images（清除舊的 data_tmp 圖片）
                    if data_tmp_changed and not need_reset:
                        _IMAGE_QUEUE_STORAGE[queue_id] = [images[i:i+1] for i in range(images.shape[0])]
                        # 重置索引到 0，因為佇列已經重建
                        current_index = 0
                        self.state["current_index"] = 0
                        logger.info(f"Data tmp var changed ({last_data_tmp_size} -> {data_tmp_size}), reset queue and index")
                    
                    tmp_data = _DATA_TEMP_STORAGE[data_tmp_var]
                    added_count = 0
                    
                    # 處理陣列
                    if isinstance(tmp_data, list):
                        for item in tmp_data:
                            if isinstance(item, torch.Tensor):
                                if len(item.shape) == 3:
                                    item = item.unsqueeze(0)
                                for i in range(item.shape[0]):
                                    _IMAGE_QUEUE_STORAGE[queue_id].append(item[i:i+1])
                                    added_count += 1
                    # 處理單一圖片
                    elif isinstance(tmp_data, torch.Tensor):
                        if len(tmp_data.shape) == 3:
                            tmp_data = tmp_data.unsqueeze(0)
                        for i in range(tmp_data.shape[0]):
                            _IMAGE_QUEUE_STORAGE[queue_id].append(tmp_data[i:i+1])
                            added_count += 1
                    
                    if added_count > 0:
                        logger.info(f"Added {added_count} images from DataTmpSet['{data_tmp_var}'] to queue")
                    
                    # 更新記錄的 data_tmp_size
                    self.state["last_data_tmp_size"] = data_tmp_size

            # 如果有額外圖片輸入，將其添加到佇列
            if additional_image is not None:
                if isinstance(additional_image, torch.Tensor):
                    if len(additional_image.shape) == 3:
                        additional_image = additional_image.unsqueeze(0)
                    # 將額外圖片添加到全局佇列
                    for i in range(additional_image.shape[0]):
                        _IMAGE_QUEUE_STORAGE[queue_id].append(additional_image[i:i+1])
                    logger.info(f"Added {additional_image.shape[0]} images from additional_image to queue")

            total = len(_IMAGE_QUEUE_STORAGE[queue_id])

            # ========== 檢查當前索引是否有效 ==========
            if total == 0:
                # 創建一個空白圖片作為錯誤返回
                error_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
                return (error_image, -1, 0, True, "No images in queue", "", "", "")

            # 如果當前索引超出範圍，表示已經處理完所有圖片
            if current_index >= total:
                logger.info(f"Current index {current_index} >= total {total}, completed")
                self.state["current_index"] = 0
                self.state["completed"] = True
                self.save_state()
                # 返回最後一張圖片
                error_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
                return (error_image, current_index, total, True, f"Completed: index {current_index} >= total {total}", "", "", "")

            # 獲取當前圖片
            current_image = _IMAGE_QUEUE_STORAGE[queue_id][current_index]

            # ========== 決定是否繼續到下一輪 ==========
            # 計算下一個索引
            next_index = current_index + 1
            
            # 檢查停止條件
            should_stop = False
            stop_reason = []
            
            # 條件 1: 檢查下一個索引是否超過 max_index
            if next_index >= max_index:
                should_stop = True
                stop_reason.append(f"next_index({next_index}) >= max_index({max_index})")
            
            # 條件 2: 檢查循環開關
            if not enable_loop:
                should_stop = True
                stop_reason.append("loop_disabled")
            
            # 條件 3: 檢查是否還有圖片（下一個索引是否有效）
            # 注意：這裡不能完全確定，因為下一輪可能會有新圖片加入
            # 所以只在確定沒有 data_tmp_var 的情況下才檢查
            if next_index >= total:
                if not (use_data_tmp and data_tmp_var and data_tmp_var.strip()):
                    # 沒有使用 data_tmp_var，確定沒有新圖片了
                    should_stop = True
                    stop_reason.append(f"next_index({next_index}) >= total({total}), no data_tmp")
                else:
                    # 使用了 data_tmp_var，可能會有新圖片，先繼續
                    logger.info(f"next_index({next_index}) >= total({total}), but data_tmp enabled, will check next round")
            
            # 更新狀態
            if not should_stop and trigger_next:
                self.state["current_index"] = next_index
                completed = False
                logger.info(f"Continue to next index: {next_index}")
                
                # 發送佇列事件
                PromptServer.instance.send_sync("textbatch-add-queue", {})
            else:
                completed = True
                self.state["current_index"] = 0
                self.state["completed"] = True
                if stop_reason:
                    logger.info(f"Stop loop: {', '.join(stop_reason)}")

            self.state["completed"] = completed
            self.state["queue_count"] = total
            self.save_state()

            # 生成狀態信息
            status = f"Processing {current_index + 1}/{total}"
            if next_index >= max_index:
                status += f" | Will reach max_index({max_index})"
            if not enable_loop:
                status += " | Loop disabled"
            if data_tmp_changed:
                status += f" | Data updated ({last_data_tmp_size}->{data_tmp_size})"
            if completed:
                status += " | Completed"
                if stop_reason:
                    status += f" ({', '.join(stop_reason)})"

            # 更新節點顯示的當前索引
            if not completed:
                PromptServer.instance.send_sync("textbatch-node-feedback", 
                    {"node_id": unique_id, "widget_name": "start_index", "type": "int", "value": self.state["current_index"]})

            # ========== 計算 MD5 和獲取檔名/路徑 ==========
            # 計算當前圖片的 MD5
            md5_value = self.calculate_image_md5(current_image)
            
            # 獲取檔名和路徑
            filename, file_path = self.get_filename_and_path(current_index, filenames, file_paths)

            return (current_image, current_index, total, completed, status, filename, file_path, md5_value)

        except Exception as e:
            logger.error(f"Error in ImageQueueProcessorPlus: {str(e)}")
            import traceback
            traceback.print_exc()
            # 創建一個空白圖片作為錯誤返回
            error_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return (error_image, -1, 0, True, f"Error: {str(e)}", "", "", "")

class TextSplitGet:
    """文本分割並獲取指定索引的節點
    支持多種分隔符和註解過濾
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "default": "item1\nitem2\nitem3\n# comment",
                    "placeholder": "輸入文本"
                }),
                "index": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "separator_type": (["newline", "comma", "semicolon", "custom"], {"default": "newline"}),
                "custom_separator": ("STRING", {"default": "|"}),
                "ignore_comment": ("BOOLEAN", {"default": True}),
                "comment_prefix": ("STRING", {"default": "#"}),
                "trim_whitespace": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "INT", "*", "STRING")
    RETURN_NAMES = ("text", "current_index", "total", "array", "status")
    FUNCTION = "process"
    CATEGORY = "TextBatch"

    def process(self, text, index, separator_type, custom_separator, ignore_comment, comment_prefix, trim_whitespace):
        try:
            if not text.strip():
                return ("", 0, 0, [], "Error: Empty text input")

            # 根據分隔符類型選擇分隔符
            separator_map = {
                "newline": "\n",
                "comma": ",",
                "semicolon": ";",
                "custom": custom_separator
            }
            separator = separator_map.get(separator_type, "\n")

            # 分割文本
            if separator_type == "newline":
                parts = text.splitlines()
            else:
                parts = text.split(separator)

            # 處理每個部分
            processed_parts = []
            for part in parts:
                # 去除空白
                if trim_whitespace:
                    part = part.strip()
                
                # 跳過空行
                if not part:
                    continue
                
                # 過濾註解
                if ignore_comment and comment_prefix and part.startswith(comment_prefix):
                    continue
                
                processed_parts.append(part)

            total = len(processed_parts)
            
            if total == 0:
                return ("", 0, 0, [], "No valid text found after filtering")

            # 確保索引在有效範圍內
            index = max(0, min(index, total - 1))
            
            # 獲取指定索引的文本
            result_text = processed_parts[index]
            
            status = f"Retrieved item {index + 1}/{total}"
            
            # 返回包含完整陣列
            return (result_text, index, total, processed_parts, status)

        except Exception as e:
            logger.error(f"Error in TextSplitGet: {str(e)}")
            return ("", 0, 0, [], f"Error: {str(e)}")

class IFMatchCond:
    """數學條件判斷節點
    根據數學條件返回 true 或 false
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value_a": ("FLOAT", {"default": 0.0, "min": -999999.0, "max": 999999.0, "step": 0.01}),
                "operator": (["==", "!=", ">", ">=", "<", "<="], {"default": "=="}),
                "value_b": ("FLOAT", {"default": 0.0, "min": -999999.0, "max": 999999.0, "step": 0.01}),
            },
            "optional": {
                "int_a": ("INT", {"default": 0, "min": -999999, "max": 999999}),
                "int_b": ("INT", {"default": 0, "min": -999999, "max": 999999}),
                "use_int": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("BOOLEAN", "STRING")
    RETURN_NAMES = ("result", "status")
    FUNCTION = "evaluate"
    CATEGORY = "TextBatch"

    def evaluate(self, value_a, operator, value_b, int_a=0, int_b=0, use_int=False):
        try:
            # 選擇使用整數或浮點數
            if use_int:
                a = int_a
                b = int_b
            else:
                a = value_a
                b = value_b

            # 執行比較
            result = False
            if operator == "==":
                result = a == b
            elif operator == "!=":
                result = a != b
            elif operator == ">":
                result = a > b
            elif operator == ">=":
                result = a >= b
            elif operator == "<":
                result = a < b
            elif operator == "<=":
                result = a <= b

            status = f"{a} {operator} {b} = {result}"
            
            return (result, status)

        except Exception as e:
            logger.error(f"Error in IFMatchCond: {str(e)}")
            return (False, f"Error: {str(e)}")

# 全局資料暫存器
_DATA_TEMP_STORAGE = {}

# 全局圖片佇列存儲（用於 ImageQueueProcessorPlus）
_IMAGE_QUEUE_STORAGE = {}

class DataTmpSet:
    """資料暫存器設置節點
    可以存儲文本、圖片或任意資料類型
    支持變數和陣列模式，以及清除功能
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "var_name": ("STRING", {"default": "my_var", "multiline": False}),
                "operation": (["set_variable", "array_push", "array_set_index", "clear_all", "clear_index"], {"default": "set_variable"}),
                "data_type": (["text", "image", "any"], {"default": "text"}),
            },
            "optional": {
                "text_data": ("STRING", {"default": "", "multiline": True}),
                "image_data": ("IMAGE",),
                "any_data": ("*",),
                "array_index": ("INT", {"default": 0, "min": 0, "max": 10000}),
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE", "INT", "STRING")
    RETURN_NAMES = ("status", "preview", "array_size", "info")
    FUNCTION = "set_data"
    CATEGORY = "TextBatch"
    OUTPUT_NODE = True

    def set_data(self, var_name, operation, data_type, text_data="", image_data=None, any_data=None, array_index=0):
        try:
            global _DATA_TEMP_STORAGE
            
            # 預設預覽圖片（空白圖）
            default_preview = torch.zeros((1, 64, 64, 3), dtype=torch.float32)

            # 清除操作
            if operation == "clear_all":
                if var_name in _DATA_TEMP_STORAGE:
                    del _DATA_TEMP_STORAGE[var_name]
                    status = f"Cleared all data in variable '{var_name}'"
                    info = f"Variable '{var_name}' cleared"
                else:
                    status = f"Variable '{var_name}' does not exist, nothing to clear"
                    info = "Nothing to clear"
                logger.info(f"DataTmpSet: {status}")
                return (status, default_preview, 0, info)
            
            elif operation == "clear_index":
                if var_name not in _DATA_TEMP_STORAGE:
                    status = f"Variable '{var_name}' does not exist"
                    return (status, default_preview, 0, status)
                elif not isinstance(_DATA_TEMP_STORAGE[var_name], list):
                    status = f"Variable '{var_name}' is not an array"
                    return (status, default_preview, 0, status)
                elif array_index >= len(_DATA_TEMP_STORAGE[var_name]):
                    status = f"Index {array_index} out of range (array size: {len(_DATA_TEMP_STORAGE[var_name])})"
                    return (status, default_preview, len(_DATA_TEMP_STORAGE[var_name]), status)
                else:
                    _DATA_TEMP_STORAGE[var_name].pop(array_index)
                    new_size = len(_DATA_TEMP_STORAGE[var_name])
                    status = f"Removed item at index {array_index} from array '{var_name}'"
                    info = f"Array size: {new_size}"
                    logger.info(f"DataTmpSet: {status}")
                    return (status, default_preview, new_size, info)

            # 根據資料類型選擇資料
            if data_type == "text":
                data = text_data
                preview = default_preview
            elif data_type == "image":
                data = image_data
                # 使用輸入圖片作為預覽
                if isinstance(data, torch.Tensor):
                    if len(data.shape) == 3:
                        preview = data.unsqueeze(0)
                    else:
                        preview = data
                else:
                    preview = default_preview
            elif data_type == "any":
                data = any_data
                # 如果 any 是圖片，也顯示預覽
                if isinstance(data, torch.Tensor):
                    if len(data.shape) == 3:
                        preview = data.unsqueeze(0)
                    else:
                        preview = data
                else:
                    preview = default_preview
            else:
                return (f"Error: Unknown data type {data_type}", default_preview, 0, "Error")

            # 設置變數
            if operation == "set_variable":
                _DATA_TEMP_STORAGE[var_name] = data
                status = f"Stored data in variable '{var_name}'"
                array_size = 0
                info = f"Type: {data_type} | Mode: variable"
            
            # 陣列操作
            elif operation == "array_push":
                # 初始化陣列（如果不存在或不是列表）
                if var_name not in _DATA_TEMP_STORAGE or not isinstance(_DATA_TEMP_STORAGE[var_name], list):
                    _DATA_TEMP_STORAGE[var_name] = []
                
                # 添加到末尾
                _DATA_TEMP_STORAGE[var_name].append(data)
                array_size = len(_DATA_TEMP_STORAGE[var_name])
                status = f"Pushed data to array '{var_name}' at index {array_size - 1}"
                info = f"Type: {data_type} | Array size: {array_size}"
            
            elif operation == "array_set_index":
                # 初始化陣列（如果不存在或不是列表）
                if var_name not in _DATA_TEMP_STORAGE or not isinstance(_DATA_TEMP_STORAGE[var_name], list):
                    _DATA_TEMP_STORAGE[var_name] = []
                
                # 如果索引超出範圍，擴展陣列
                while len(_DATA_TEMP_STORAGE[var_name]) <= array_index:
                    _DATA_TEMP_STORAGE[var_name].append(None)
                
                _DATA_TEMP_STORAGE[var_name][array_index] = data
                array_size = len(_DATA_TEMP_STORAGE[var_name])
                status = f"Set data in array '{var_name}' at index {array_index}"
                info = f"Type: {data_type} | Array size: {array_size}"

            logger.info(f"DataTmpSet: {status}")
            return (status, preview, array_size, info)

        except Exception as e:
            logger.error(f"Error in DataTmpSet: {str(e)}")
            import traceback
            traceback.print_exc()
            default_preview = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return (f"Error: {str(e)}", default_preview, 0, f"Error: {str(e)}")

class DataTmpGet:
    """資料暫存器讀取節點
    讀取暫存器中的資料
    支持變數和陣列索引
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "var_name": ("STRING", {"default": "my_var", "multiline": False}),
                "data_type": (["text", "image", "any"], {"default": "text"}),
                "read_mode": (["index", "first", "last", "all"], {"default": "index"}),
            },
            "optional": {
                "array_index": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "default_text": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE", "*", "INT", "*", "STRING")
    RETURN_NAMES = ("text", "image", "any", "array_size", "all_elements", "status")
    FUNCTION = "get_data"
    CATEGORY = "TextBatch"

    def get_data(self, var_name, data_type, read_mode, array_index=0, default_text=""):
        try:
            global _DATA_TEMP_STORAGE

            # 創建一個空白圖片作為錯誤時的預設返回值
            default_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)

            # 檢查變數是否存在
            if var_name not in _DATA_TEMP_STORAGE:
                status = f"Variable '{var_name}' not found, using default"
                return (default_text, default_image, None, 0, None, status)

            stored_data = _DATA_TEMP_STORAGE[var_name]
            
            # 獲取陣列大小和所有元素
            array_size = len(stored_data) if isinstance(stored_data, list) else 0
            all_elements = stored_data if isinstance(stored_data, list) else [stored_data]

            # 根據讀取模式處理
            if isinstance(stored_data, list) and array_size > 0:
                if read_mode == "first":
                    # 讀取第一個
                    data = stored_data[0]
                    status = f"Retrieved first item from array '{var_name}' (size: {array_size})"
                
                elif read_mode == "last":
                    # 讀取最後一個
                    data = stored_data[-1]
                    status = f"Retrieved last item from array '{var_name}' (size: {array_size})"
                
                elif read_mode == "all":
                    # 讀取全部（特殊處理圖片類型）
                    if data_type == "image":
                        # 合併所有圖片到一個批次
                        try:
                            valid_images = []
                            for idx, item in enumerate(stored_data):
                                if isinstance(item, torch.Tensor):
                                    # 確保是 4D 張量 [B, H, W, C]
                                    if len(item.shape) == 3:
                                        item = item.unsqueeze(0)
                                    valid_images.append(item)
                                else:
                                    logger.warning(f"Item at index {idx} is not a tensor, skipping")
                            
                            if len(valid_images) == 0:
                                status = f"No valid images found in array '{var_name}'"
                                return ("", default_image, None, array_size, all_elements, status)
                            
                            # 合併所有圖片
                            combined_images = torch.cat(valid_images, dim=0)
                            status = f"Retrieved all {len(valid_images)} images from array '{var_name}'"
                            return ("", combined_images, None, array_size, all_elements, status)
                        except Exception as e:
                            logger.error(f"Error combining images: {str(e)}")
                            status = f"Error combining images: {str(e)}"
                            return ("", default_image, None, array_size, all_elements, status)
                    else:
                        # 非圖片類型，返回整個列表
                        data = stored_data
                        status = f"Retrieved entire array '{var_name}' (size: {array_size})"
                
                elif read_mode == "index":
                    # 讀取指定索引
                    if array_index >= len(stored_data):
                        status = f"Error: Index {array_index} out of range (array size: {len(stored_data)})"
                        return (default_text, default_image, None, array_size, all_elements, status)
                    
                    data = stored_data[array_index]
                    status = f"Retrieved item at index {array_index} from array '{var_name}' (size: {array_size})"
                
                else:
                    status = f"Unknown read_mode: {read_mode}"
                    return (default_text, default_image, None, array_size, all_elements, status)
            
            elif isinstance(stored_data, list) and array_size == 0:
                # 空陣列
                status = f"Array '{var_name}' is empty"
                return (default_text, default_image, None, 0, [], status)
            
            else:
                # 不是陣列，直接返回變數
                data = stored_data
                status = f"Retrieved data from variable '{var_name}' (not an array)"

            # 根據資料類型返回
            if data_type == "text":
                if isinstance(data, str):
                    return (data, default_image, None, array_size, all_elements, status)
                elif isinstance(data, list):
                    # 將列表轉換為字串
                    return (str(data), default_image, None, array_size, all_elements, status)
                else:
                    return (str(data), default_image, None, array_size, all_elements, status)
            elif data_type == "image":
                if isinstance(data, torch.Tensor):
                    # 確保是正確的格式
                    if len(data.shape) == 3:
                        data = data.unsqueeze(0)
                    return ("", data, None, array_size, all_elements, status)
                else:
                    error_msg = f"Error: Data is not an image tensor (type: {type(data).__name__})"
                    logger.error(error_msg)
                    return ("", default_image, None, array_size, all_elements, error_msg)
            elif data_type == "any":
                # 對於 any 類型，如果是圖片也放在 any 輸出
                if isinstance(data, torch.Tensor):
                    return ("", data, data, array_size, all_elements, status)
                else:
                    return ("", default_image, data, array_size, all_elements, status)

            return (default_text, default_image, None, array_size, all_elements, "Error: Unknown data type")

        except Exception as e:
            logger.error(f"Error in DataTmpGet: {str(e)}")
            import traceback
            traceback.print_exc()
            # 創建一個空白圖片作為錯誤返回
            default_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return (default_text, default_image, None, 0, None, f"Error: {str(e)}")

class TextArrayIndex:
    """文本陣列索引節點
    可以輸入多個文本陣列並合併，然後輸出指定索引的文本
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "index": ("INT", {"default": 0, "min": 0, "max": 10000}),
            },
            "optional": {
                "array_1": ("*",),
                "array_2": ("*",),
                "array_3": ("*",),
                "array_4": ("*",),
                "array_5": ("*",),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "STRING")
    RETURN_NAMES = ("text", "total", "status")
    FUNCTION = "process"
    CATEGORY = "TextBatch"

    def process(self, index, array_1=None, array_2=None, array_3=None, array_4=None, array_5=None):
        try:
            # 合併所有輸入的陣列
            merged_array = []
            
            for arr in [array_1, array_2, array_3, array_4, array_5]:
                if arr is None:
                    continue
                
                # 處理不同類型的輸入
                if isinstance(arr, list):
                    # 如果是列表，直接擴展
                    merged_array.extend([str(item) for item in arr])
                elif isinstance(arr, str):
                    # 如果是字串，當作單個元素
                    merged_array.append(arr)
                else:
                    # 其他類型，轉換為字串
                    merged_array.append(str(arr))
            
            total = len(merged_array)
            
            if total == 0:
                return ("", 0, "No arrays provided or all arrays are empty")
            
            # 確保索引在有效範圍內
            if index >= total:
                status = f"Index {index} out of range (total: {total}), returning last item"
                index = total - 1
            else:
                status = f"Retrieved item {index + 1}/{total}"
            
            result_text = merged_array[index]
            
            return (result_text, total, status)

        except Exception as e:
            logger.error(f"Error in TextArrayIndex: {str(e)}")
            import traceback
            traceback.print_exc()
            return ("", 0, f"Error: {str(e)}")

class DataTempManager:
    """資料暫存器管理節點
    可以查看所有變數、狀態、刪除特定變數或索引
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "action": (["list_all", "delete_var", "delete_index", "clear_all"], {"default": "list_all"}),
            },
            "optional": {
                "var_name": ("STRING", {"default": "", "multiline": False}),
                "array_index": ("INT", {"default": 0, "min": 0, "max": 10000}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("variables_list", "status", "total_vars")
    FUNCTION = "manage"
    CATEGORY = "TextBatch"
    OUTPUT_NODE = True

    def manage(self, action, var_name="", array_index=0):
        try:
            global _DATA_TEMP_STORAGE

            if action == "list_all":
                # 列出所有變數及其狀態
                if len(_DATA_TEMP_STORAGE) == 0:
                    return ("No variables stored", "Empty storage", 0)
                
                var_list = []
                for name, value in _DATA_TEMP_STORAGE.items():
                    if isinstance(value, list):
                        size = len(value)
                        types = set(type(item).__name__ for item in value[:5])  # 檢查前5個元素的類型
                        var_info = f"{name}: Array[{size}] - Types: {', '.join(types)}"
                    elif isinstance(value, torch.Tensor):
                        shape = tuple(value.shape)
                        var_info = f"{name}: Tensor{shape}"
                    elif isinstance(value, str):
                        preview = value[:50] + "..." if len(value) > 50 else value
                        var_info = f"{name}: String - '{preview}'"
                    else:
                        var_info = f"{name}: {type(value).__name__}"
                    var_list.append(var_info)
                
                variables_text = "\n".join(var_list)
                status = f"Found {len(_DATA_TEMP_STORAGE)} variables"
                return (variables_text, status, len(_DATA_TEMP_STORAGE))
            
            elif action == "delete_var":
                # 刪除特定變數
                if not var_name:
                    return ("", "Error: Please specify var_name", 0)
                
                if var_name in _DATA_TEMP_STORAGE:
                    del _DATA_TEMP_STORAGE[var_name]
                    status = f"Deleted variable '{var_name}'"
                    logger.info(f"DataTempManager: {status}")
                else:
                    status = f"Variable '{var_name}' not found"
                
                return ("", status, len(_DATA_TEMP_STORAGE))
            
            elif action == "delete_index":
                # 刪除陣列中的特定索引
                if not var_name:
                    return ("", "Error: Please specify var_name", 0)
                
                if var_name not in _DATA_TEMP_STORAGE:
                    return ("", f"Variable '{var_name}' not found", 0)
                
                value = _DATA_TEMP_STORAGE[var_name]
                if not isinstance(value, list):
                    return ("", f"Variable '{var_name}' is not an array", 0)
                
                if array_index >= len(value):
                    return ("", f"Index {array_index} out of range (size: {len(value)})", 0)
                
                value.pop(array_index)
                status = f"Deleted index {array_index} from '{var_name}' (new size: {len(value)})"
                logger.info(f"DataTempManager: {status}")
                return ("", status, len(_DATA_TEMP_STORAGE))
            
            elif action == "clear_all":
                # 清除所有變數
                count = len(_DATA_TEMP_STORAGE)
                _DATA_TEMP_STORAGE.clear()
                status = f"Cleared all {count} variables"
                logger.info(f"DataTempManager: {status}")
                return ("", status, 0)
            
            return ("", f"Unknown action: {action}", len(_DATA_TEMP_STORAGE))

        except Exception as e:
            logger.error(f"Error in DataTempManager: {str(e)}")
            import traceback
            traceback.print_exc()
            return ("", f"Error: {str(e)}", 0)

class GroupController:
    """群組控制器節點
    動態控制所有群組的啟用/停用狀態
    每個群組都有獨立的 BOOLEAN input 來控制
    類似於 rgthree 的 Fast Groups Bypasser
    """
    
    def __init__(self):
        self.group_states_file = os.path.join(os.path.dirname(__file__), "group_controller_states.json")
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "control_mode": (["bypass", "mute"], {"default": "bypass"}),
            },
            "optional": {
                # 這裡會動態添加群組的 input
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "control_groups"
    CATEGORY = "TextBatch"
    OUTPUT_NODE = True

    def control_groups(self, control_mode, unique_id=None, **kwargs):
        """
        控制群組的啟用/停用狀態
        kwargs 會包含所有動態添加的群組控制參數
        """
        try:
            # 收集所有群組的啟用狀態
            group_states = {}
            for key, value in kwargs.items():
                if key.startswith("enable_group_"):
                    group_id = key.replace("enable_group_", "")
                    group_states[group_id] = value
            
            # 生成狀態信息
            enabled_count = sum(1 for v in group_states.values() if v)
            total_count = len(group_states)
            
            status = f"Mode: {control_mode} | Groups: {enabled_count}/{total_count} enabled"
            
            # 將狀態信息發送到前端
            if unique_id:
                PromptServer.instance.send_sync("groupcontroller-update", {
                    "node_id": unique_id,
                    "group_states": group_states,
                    "control_mode": control_mode
                })
            
            return (status,)

        except Exception as e:
            logger.error(f"Error in GroupController: {str(e)}")
            import traceback
            traceback.print_exc()
            return (f"Error: {str(e)}",)

class UniversalLoopController:
    """萬用循環控制器節點
    可以設定循環次數，自動執行指定次數的循環
    支援多個輸入輸出，並在循環完成後輸出所有累積的結果
    """
    def __init__(self):
        self.state_file = os.path.join(os.path.dirname(__file__), "universal_loop_state.json")
        self.state = self.load_state()

    def reset_state(self, loop_id="default"):
        """重置狀態到初始值"""
        self.state = {
            "current_loop": 0,
            "last_config": "",
            "completed": False,
            "accumulated_data": []
        }
        self.save_state()

    def load_state(self):
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading state file: {str(e)}")
        return {
            "current_loop": 0,
            "last_config": "",
            "completed": False,
            "accumulated_data": []
        }

    def save_state(self):
        try:
            with open(self.state_file, 'w') as f:
                json.dump(self.state, f)
        except Exception as e:
            logger.error(f"Error saving state file: {str(e)}")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "loop_count": ("INT", {"default": 3, "min": 1, "max": 1000}),
                "accumulate_mode": (["none", "text", "image", "any"], {"default": "none"}),
                "trigger_next": ("BOOLEAN", {"default": True, "label_on": "Trigger", "label_off": "Don't trigger"}),
                "reset_on_start": ("BOOLEAN", {"default": True, "label_on": "Yes", "label_off": "No"}),
            },
            "optional": {
                # 支援多個輸入
                "input_1": ("*",),
                "input_2": ("*",),
                "input_3": ("*",),
                "input_4": ("*",),
                "input_5": ("*",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            }
        }

    RETURN_TYPES = ("INT", "INT", "BOOLEAN", "*", "*", "*", "*", "*", "*", "STRING")
    RETURN_NAMES = ("current_loop", "total_loops", "completed", 
                    "output_1", "output_2", "output_3", "output_4", "output_5", 
                    "accumulated_results", "status")
    FUNCTION = "process_loop"
    CATEGORY = "TextBatch"
    OUTPUT_NODE = True

    def process_loop(self, loop_count, accumulate_mode, trigger_next, reset_on_start, unique_id,
                    input_1=None, input_2=None, input_3=None, input_4=None, input_5=None,
                    prompt=None, extra_pnginfo=None):
        """
        處理循環邏輯
        """
        try:
            global _DATA_TEMP_STORAGE
            
            # 使用 unique_id 作為循環標識
            loop_id = unique_id if unique_id else "default"
            
            # 生成配置標識
            config_hash = str(hash(f"{loop_count}_{accumulate_mode}_{reset_on_start}"))
            
            # 檢查是否需要重置
            need_reset = (
                self.state.get("last_config") != config_hash or
                self.state.get("completed", False) or
                (prompt and extra_pnginfo and reset_on_start)
            )

            if need_reset:
                self.reset_state(loop_id)
                self.state["last_config"] = config_hash
                current_loop = 0
                logger.info(f"Loop controller reset for {loop_id}, will run {loop_count} times")
            else:
                current_loop = self.state.get("current_loop", 0)

            # 檢查是否已完成
            if current_loop >= loop_count:
                logger.info(f"Loop completed: {current_loop}/{loop_count}")
                self.state["current_loop"] = 0
                self.state["completed"] = True
                self.save_state()
                
                # 返回累積的結果
                accumulated_results = self.state.get("accumulated_data", [])
                return (
                    current_loop, loop_count, True,
                    input_1, input_2, input_3, input_4, input_5,
                    accumulated_results,
                    f"Completed: {loop_count}/{loop_count} loops"
                )

            # 當前循環處理
            logger.info(f"Processing loop {current_loop + 1}/{loop_count}")
            
            # 累積資料（如果啟用）
            if accumulate_mode != "none":
                if "accumulated_data" not in self.state:
                    self.state["accumulated_data"] = []
                
                # 根據模式累積資料
                loop_data = {
                    "loop_index": current_loop,
                    "input_1": self._serialize_data(input_1, accumulate_mode),
                    "input_2": self._serialize_data(input_2, accumulate_mode),
                    "input_3": self._serialize_data(input_3, accumulate_mode),
                    "input_4": self._serialize_data(input_4, accumulate_mode),
                    "input_5": self._serialize_data(input_5, accumulate_mode),
                }
                self.state["accumulated_data"].append(loop_data)

            # 決定是否繼續到下一輪
            next_loop = current_loop + 1
            is_last = next_loop >= loop_count
            
            if not is_last and trigger_next:
                self.state["current_loop"] = next_loop
                completed = False
                logger.info(f"Continue to next loop: {next_loop}")
                
                # 發送佇列事件
                PromptServer.instance.send_sync("textbatch-add-queue", {})
            else:
                completed = True
                self.state["current_loop"] = next_loop
                self.state["completed"] = True

            self.save_state()

            # 生成狀態信息
            status = f"Loop {current_loop + 1}/{loop_count}"
            if accumulate_mode != "none":
                status += f" | Accumulating: {accumulate_mode}"
            if completed:
                status += " | Completed"

            # 更新節點顯示的當前循環
            if not completed:
                PromptServer.instance.send_sync("textbatch-node-feedback", 
                    {"node_id": unique_id, "widget_name": "loop_count", "type": "int", "value": next_loop})

            accumulated_results = self.state.get("accumulated_data", [])
            
            return (
                current_loop, loop_count, completed,
                input_1, input_2, input_3, input_4, input_5,
                accumulated_results,
                status
            )

        except Exception as e:
            logger.error(f"Error in UniversalLoopController: {str(e)}")
            import traceback
            traceback.print_exc()
            return (0, loop_count, True, None, None, None, None, None, [], f"Error: {str(e)}")

    def _serialize_data(self, data, mode):
        """序列化資料以便累積"""
        try:
            if data is None:
                return None
            
            if mode == "text":
                return str(data)
            elif mode == "image":
                if isinstance(data, torch.Tensor):
                    # 對於圖片，我們只儲存形狀信息，不儲存實際資料（太大）
                    return f"Image tensor: shape={tuple(data.shape)}"
                return str(data)
            elif mode == "any":
                if isinstance(data, torch.Tensor):
                    return f"Tensor: shape={tuple(data.shape)}"
                elif isinstance(data, (list, dict)):
                    return data
                else:
                    return str(data)
            else:
                return None
        except Exception as e:
            logger.error(f"Error serializing data: {str(e)}")
            return None

class LoopResultExtractor:
    """循環結果提取器
    用於從 UniversalLoopController 的累積結果中提取特定循環的資料
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "accumulated_results": ("*",),
                "extract_mode": (["specific_index", "all_as_text", "count"], {"default": "specific_index"}),
            },
            "optional": {
                "loop_index": ("INT", {"default": 0, "min": 0, "max": 1000}),
                "output_field": (["input_1", "input_2", "input_3", "input_4", "input_5"], {"default": "input_1"}),
            }
        }

    RETURN_TYPES = ("*", "STRING", "INT", "STRING")
    RETURN_NAMES = ("extracted_data", "text_output", "count", "status")
    FUNCTION = "extract"
    CATEGORY = "TextBatch"

    def extract(self, accumulated_results, extract_mode, loop_index=0, output_field="input_1"):
        """
        從累積結果中提取資料
        """
        try:
            if not accumulated_results or not isinstance(accumulated_results, list):
                return (None, "", 0, "No accumulated results")

            count = len(accumulated_results)

            if extract_mode == "count":
                return (None, str(count), count, f"Total loops: {count}")

            elif extract_mode == "all_as_text":
                # 將所有結果轉換為文本
                text_output = json.dumps(accumulated_results, indent=2, ensure_ascii=False)
                return (accumulated_results, text_output, count, f"Extracted all {count} results as text")

            elif extract_mode == "specific_index":
                # 提取特定索引的資料
                if loop_index >= count:
                    return (None, "", count, f"Index {loop_index} out of range (total: {count})")

                loop_data = accumulated_results[loop_index]
                extracted = loop_data.get(output_field, None)
                
                text_output = str(extracted) if extracted is not None else ""
                status = f"Extracted loop {loop_index + 1}/{count}, field: {output_field}"
                
                return (extracted, text_output, count, status)

            return (None, "", count, f"Unknown extract mode: {extract_mode}")

        except Exception as e:
            logger.error(f"Error in LoopResultExtractor: {str(e)}")
            import traceback
            traceback.print_exc()
            return (None, "", 0, f"Error: {str(e)}")

class JsonQueryNode:
    """JSON 查詢節點
    可以載入 JSON 文本或檔案，並通過 KEY 和欄位路徑來查詢資料
    支援多組輸入和輸出，支援 a.b.c.d 這樣的路徑查詢
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_mode": (["text", "file"], {"default": "text"}),
                "json_text": ("STRING", {
                    "multiline": True,
                    "default": '{"a": {"name": "myname", "prompts": "mycustom prompt"}}',
                    "placeholder": "輸入 JSON 文本"
                }),
                "json_file": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "placeholder": "輸入 JSON 檔案路徑"
                }),
            },
            "optional": {
                # 支援最多 5 組查詢
                "key_1": ("STRING", {"default": "", "multiline": False}),
                "field_1": ("STRING", {"default": "", "multiline": False}),
                
                "key_2": ("STRING", {"default": "", "multiline": False}),
                "field_2": ("STRING", {"default": "", "multiline": False}),
                
                "key_3": ("STRING", {"default": "", "multiline": False}),
                "field_3": ("STRING", {"default": "", "multiline": False}),
                
                "key_4": ("STRING", {"default": "", "multiline": False}),
                "field_4": ("STRING", {"default": "", "multiline": False}),
                
                "key_5": ("STRING", {"default": "", "multiline": False}),
                "field_5": ("STRING", {"default": "", "multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("output_1", "output_2", "output_3", "output_4", "output_5", "status")
    FUNCTION = "query_json"
    CATEGORY = "TextBatch"

    def query_json(self, input_mode, json_text, json_file, 
                   key_1="", field_1="", key_2="", field_2="", 
                   key_3="", field_3="", key_4="", field_4="", 
                   key_5="", field_5=""):
        """
        查詢 JSON 資料
        支援路徑查詢，例如：key='a', field='b.c.d' 或直接 key='a.b.c.d'
        """
        try:
            # 載入 JSON 資料
            json_data = None
            
            if input_mode == "file":
                if not json_file or not json_file.strip():
                    return ("", "", "", "", "", "Error: 請提供 JSON 檔案路徑")
                
                if not os.path.exists(json_file):
                    return ("", "", "", "", "", f"Error: 檔案不存在: {json_file}")
                
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        json_data = json.load(f)
                except json.JSONDecodeError as e:
                    return ("", "", "", "", "", f"Error: JSON 解析錯誤: {str(e)}")
            else:
                if not json_text or not json_text.strip():
                    return ("", "", "", "", "", "Error: 請提供 JSON 文本")
                
                try:
                    json_data = json.loads(json_text)
                except json.JSONDecodeError as e:
                    return ("", "", "", "", "", f"Error: JSON 解析錯誤: {str(e)}")

            # 執行查詢
            results = []
            queries_info = []
            
            query_pairs = [
                (key_1, field_1),
                (key_2, field_2),
                (key_3, field_3),
                (key_4, field_4),
                (key_5, field_5)
            ]
            
            for i, (key, field) in enumerate(query_pairs, 1):
                result = self._query_path(json_data, key, field)
                results.append(result)
                
                # 記錄查詢信息
                if key or field:
                    query_path = self._build_query_path(key, field)
                    if result:
                        queries_info.append(f"Q{i}: {query_path} ✓")
                    else:
                        queries_info.append(f"Q{i}: {query_path} ✗")
            
            # 生成狀態信息
            if queries_info:
                status = " | ".join(queries_info)
            else:
                status = "No queries provided"
            
            return (results[0], results[1], results[2], results[3], results[4], status)

        except Exception as e:
            logger.error(f"Error in JsonQueryNode: {str(e)}")
            import traceback
            traceback.print_exc()
            return ("", "", "", "", "", f"Error: {str(e)}")

    def _query_path(self, data, key, field):
        """
        查詢指定路徑的資料
        支援 key='a', field='b.c' 或 key='a.b.c' 或 field='a.b.c'
        """
        try:
            # 如果 key 和 field 都為空，返回空
            if not key and not field:
                return ""
            
            # 建立完整路徑
            full_path = self._build_query_path(key, field)
            if not full_path:
                return ""
            
            # 分割路徑並逐層查詢
            path_parts = full_path.split('.')
            current = data
            
            for part in path_parts:
                if not part:  # 跳過空字串
                    continue
                
                if isinstance(current, dict):
                    if part in current:
                        current = current[part]
                    else:
                        logger.warning(f"Key '{part}' not found in path '{full_path}'")
                        return ""
                elif isinstance(current, list):
                    # 如果是列表，嘗試轉換索引
                    try:
                        index = int(part)
                        if 0 <= index < len(current):
                            current = current[index]
                        else:
                            logger.warning(f"Index {index} out of range in path '{full_path}'")
                            return ""
                    except ValueError:
                        logger.warning(f"Invalid index '{part}' in path '{full_path}'")
                        return ""
                else:
                    logger.warning(f"Cannot navigate further in path '{full_path}' at '{part}'")
                    return ""
            
            # 轉換結果為字串
            if isinstance(current, (dict, list)):
                return json.dumps(current, ensure_ascii=False)
            else:
                return str(current)
        
        except Exception as e:
            logger.error(f"Error querying path: {str(e)}")
            return ""

    def _build_query_path(self, key, field):
        """建立完整的查詢路徑"""
        parts = []
        if key and key.strip():
            parts.append(key.strip())
        if field and field.strip():
            parts.append(field.strip())
        return '.'.join(parts)

# 節點類映射
NODE_CLASS_MAPPINGS = {
    "TextBatch": TextBatchNode, 
    "TextQueueProcessor": TextQueueProcessor,
    "TextSplitCounter": TextSplitCounterNode,
    "ImageQueueProcessor": ImageQueueProcessor,
    "ImageQueueProcessorPro": ImageQueueProcessorPro,  # 新增：延遲載入的隊列處理器
    "ImageInfoExtractor": ImageInfoExtractorNode,
    "PathParser": PathParserNode,
    "LoadImagesFromDirBatch": LoadImagesFromDirBatchM,
    "ImageFilenameProcessor": ImageFilenameProcessor,
    "LoadImageByIndex": LoadImageByIndex,  # 新增：根據索引延遲載入單張圖片
    "LoadImagesFromDirLazy": LoadImagesFromDirLazy,  # 新增：延遲載入模式 - 先獲取檔案列表
    "ImageQueueProcessorPlus": ImageQueueProcessorPlus,
    "TextSplitGet": TextSplitGet,
    "IFMatchCond": IFMatchCond,
    "DataTmpSet": DataTmpSet,
    "DataTmpGet": DataTmpGet,
    "TextArrayIndex": TextArrayIndex,
    "DataTempManager": DataTempManager,
    "GroupController": GroupController,
    "JsonQuery": JsonQueryNode,
    "UniversalLoopController": UniversalLoopController,  # 新增萬用循環控制器
    "LoopResultExtractor": LoopResultExtractor  # 新增循環結果提取器
}

# 節點顯示名稱映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "TextBatch": "Text Batch", 
    "TextQueueProcessor": "Text Queue Processor",
    "TextSplitCounter": "Text Split Counter",
    "ImageQueueProcessor": "Image Queue Processor",
    "ImageQueueProcessorPro": "Image Queue Processor Pro (延遲載入)",
    "ImageInfoExtractor": "Image Info Extractor",
    "PathParser": "Path Parser",
    "LoadImagesFromDirBatch": "Load Images From Dir Batch",
    "ImageFilenameProcessor": "Image Filename Processor",
    "LoadImageByIndex": "Load Image By Index (延遲載入)",
    "LoadImagesFromDirLazy": "Scan Images Dir (延遲模式)",
    "ImageQueueProcessorPlus": "Image Queue Processor Plus",
    "TextSplitGet": "Text Split Get",
    "IFMatchCond": "IF Match Condition",
    "DataTmpSet": "Data Temp Set",
    "DataTmpGet": "Data Temp Get",
    "TextArrayIndex": "Text Array Index",
    "DataTempManager": "Data Temp Manager",
    "GroupController": "Group Controller",
    "JsonQuery": "JSON Query",
    "UniversalLoopController": "Universal Loop Controller",  # 萬用循環控制器
    "LoopResultExtractor": "Loop Result Extractor"  # 循環結果提取器
}