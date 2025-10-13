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

class ImageQueueProcessorPlus:
    """處理圖片佇列的增強節點
    支持兩個圖片輸入，可以將新圖片合併到佇列中實現循環處理
    支持自動停止條件（最大索引或開關控制）
    """
    def __init__(self):
        self.state_file = os.path.join(os.path.dirname(__file__), "image_queue_processor_plus_state.json")
        self.state = self.load_state()
        self.image_queue = []

    def reset_state(self):
        self.state = {
            "current_index": 0,
            "last_input": "",
            "completed": False,
            "queue_count": 0
        }
        self.image_queue = []
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

    def process(self, images, start_index, max_index, trigger_next, enable_loop, unique_id, 
                additional_image=None, prompt=None, extra_pnginfo=None):
        try:
            # 確保輸入是 tensor 並且格式正確
            if not isinstance(images, torch.Tensor):
                return (None, -1, 0, True, "Invalid input: not a tensor")

            # 處理單張圖片的情況
            if len(images.shape) == 3:
                images = images.unsqueeze(0)

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
                self.image_queue = [images[i:i+1] for i in range(images.shape[0])]
                current_index = start_index
            else:
                current_index = min(max(start_index, self.state.get("current_index", 0)), len(self.image_queue) - 1)

            # 如果有額外圖片輸入，將其添加到佇列
            if additional_image is not None:
                if isinstance(additional_image, torch.Tensor):
                    if len(additional_image.shape) == 3:
                        additional_image = additional_image.unsqueeze(0)
                    # 將額外圖片添加到佇列
                    for i in range(additional_image.shape[0]):
                        self.image_queue.append(additional_image[i:i+1])
                    logger.info(f"Added {additional_image.shape[0]} images to queue")

            total = len(self.image_queue)
            if total == 0:
                return (None, -1, 0, True, "No images in queue")

            # 確保索引有效
            if current_index >= total:
                current_index = total - 1

            # 獲取當前圖片
            current_image = self.image_queue[current_index]

            # 檢查是否應該停止
            # 條件1: 達到最大索引
            # 條件2: 循環開關被關閉
            # 條件3: 已經是最後一張
            should_stop = (
                current_index >= max_index - 1 or
                not enable_loop or
                current_index >= total - 1
            )
            
            # 更新狀態
            if not should_stop and trigger_next:
                next_index = current_index + 1
                self.state["current_index"] = next_index
                completed = False
                
                # 發送佇列事件
                PromptServer.instance.send_sync("textbatch-add-queue", {})
            else:
                completed = True
                self.state["current_index"] = 0
                self.state["completed"] = True

            self.state["completed"] = completed
            self.state["queue_count"] = total
            self.save_state()

            # 生成狀態信息
            status = f"Processing {current_index + 1}/{total}"
            if current_index >= max_index - 1:
                status += " | Reached max_index"
            if not enable_loop:
                status += " | Loop disabled"
            if completed:
                status += " | Completed"

            # 更新節點顯示的當前索引
            if not completed:
                PromptServer.instance.send_sync("textbatch-node-feedback", 
                    {"node_id": unique_id, "widget_name": "start_index", "type": "int", "value": self.state["current_index"]})

            return (current_image, current_index, total, completed, status)

        except Exception as e:
            logger.error(f"Error in ImageQueueProcessorPlus: {str(e)}")
            return (None, -1, 0, True, f"Error: {str(e)}")

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

    RETURN_TYPES = ("STRING", "INT", "INT", "STRING")
    RETURN_NAMES = ("text", "current_index", "total", "status")
    FUNCTION = "process"
    CATEGORY = "TextBatch"

    def process(self, text, index, separator_type, custom_separator, ignore_comment, comment_prefix, trim_whitespace):
        try:
            if not text.strip():
                return ("", 0, 0, "Error: Empty text input")

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
                return ("", 0, 0, "No valid text found after filtering")

            # 確保索引在有效範圍內
            index = max(0, min(index, total - 1))
            
            # 獲取指定索引的文本
            result_text = processed_parts[index]
            
            status = f"Retrieved item {index + 1}/{total}"
            
            return (result_text, index, total, status)

        except Exception as e:
            logger.error(f"Error in TextSplitGet: {str(e)}")
            return ("", 0, 0, f"Error: {str(e)}")

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

class DataTmpSet:
    """資料暫存器設置節點
    可以存儲文本、圖片或任意資料類型
    支持變數和陣列模式
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "var_name": ("STRING", {"default": "my_var", "multiline": False}),
                "storage_mode": (["variable", "array"], {"default": "variable"}),
                "data_type": (["text", "image", "any"], {"default": "text"}),
            },
            "optional": {
                "text_data": ("STRING", {"default": "", "multiline": True}),
                "image_data": ("IMAGE",),
                "any_data": ("*",),
                "array_index": ("INT", {"default": -1, "min": -1, "max": 10000}),  # -1 表示添加到末尾
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "set_data"
    CATEGORY = "TextBatch"
    OUTPUT_NODE = True

    def set_data(self, var_name, storage_mode, data_type, text_data="", image_data=None, any_data=None, array_index=-1):
        try:
            global _DATA_TEMP_STORAGE

            # 根據資料類型選擇資料
            if data_type == "text":
                data = text_data
            elif data_type == "image":
                data = image_data
            elif data_type == "any":
                data = any_data
            else:
                return (f"Error: Unknown data type {data_type}",)

            # 根據存儲模式處理
            if storage_mode == "variable":
                _DATA_TEMP_STORAGE[var_name] = data
                status = f"Stored data in variable '{var_name}'"
            
            elif storage_mode == "array":
                # 初始化陣列（如果不存在）
                if var_name not in _DATA_TEMP_STORAGE or not isinstance(_DATA_TEMP_STORAGE[var_name], list):
                    _DATA_TEMP_STORAGE[var_name] = []
                
                # 添加或設置資料
                if array_index == -1:
                    # 添加到末尾
                    _DATA_TEMP_STORAGE[var_name].append(data)
                    status = f"Appended data to array '{var_name}' (index {len(_DATA_TEMP_STORAGE[var_name]) - 1})"
                else:
                    # 設置指定索引
                    # 如果索引超出範圍，擴展陣列
                    while len(_DATA_TEMP_STORAGE[var_name]) <= array_index:
                        _DATA_TEMP_STORAGE[var_name].append(None)
                    _DATA_TEMP_STORAGE[var_name][array_index] = data
                    status = f"Set data in array '{var_name}' at index {array_index}"

            logger.info(f"DataTmpSet: {status}")
            return (status,)

        except Exception as e:
            logger.error(f"Error in DataTmpSet: {str(e)}")
            return (f"Error: {str(e)}",)

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
            },
            "optional": {
                "array_index": ("INT", {"default": -1, "min": -1, "max": 10000}),  # -1 表示不使用索引
                "default_text": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE", "*", "STRING")
    RETURN_NAMES = ("text", "image", "any", "status")
    FUNCTION = "get_data"
    CATEGORY = "TextBatch"

    def get_data(self, var_name, data_type, array_index=-1, default_text=""):
        try:
            global _DATA_TEMP_STORAGE

            # 檢查變數是否存在
            if var_name not in _DATA_TEMP_STORAGE:
                status = f"Variable '{var_name}' not found, using default"
                return (default_text, None, None, status)

            data = _DATA_TEMP_STORAGE[var_name]

            # 如果是陣列模式
            if array_index >= 0:
                if not isinstance(data, list):
                    status = f"Error: '{var_name}' is not an array"
                    return (default_text, None, None, status)
                
                if array_index >= len(data):
                    status = f"Error: Index {array_index} out of range (array size: {len(data)})"
                    return (default_text, None, None, status)
                
                data = data[array_index]
                status = f"Retrieved data from array '{var_name}' at index {array_index}"
            else:
                status = f"Retrieved data from variable '{var_name}'"

            # 根據資料類型返回
            if data_type == "text":
                if isinstance(data, str):
                    return (data, None, None, status)
                else:
                    return (str(data), None, None, status)
            elif data_type == "image":
                if isinstance(data, torch.Tensor):
                    return ("", data, None, status)
                else:
                    return ("", None, None, f"Error: Data is not an image tensor")
            elif data_type == "any":
                return ("", None, data, status)

            return (default_text, None, None, "Error: Unknown data type")

        except Exception as e:
            logger.error(f"Error in DataTmpGet: {str(e)}")
            return (default_text, None, None, f"Error: {str(e)}")

# 節點類映射
NODE_CLASS_MAPPINGS = {
    "TextBatch": TextBatchNode, 
    "TextQueueProcessor": TextQueueProcessor,
    "TextSplitCounter": TextSplitCounterNode,
    "ImageQueueProcessor": ImageQueueProcessor,
    "ImageInfoExtractor": ImageInfoExtractorNode,
    "PathParser": PathParserNode,
    "LoadImagesFromDirBatch": LoadImagesFromDirBatchM,
    "ImageFilenameProcessor": ImageFilenameProcessor,
    "ImageQueueProcessorPlus": ImageQueueProcessorPlus,  # 新增
    "TextSplitGet": TextSplitGet,  # 新增
    "IFMatchCond": IFMatchCond,  # 新增
    "DataTmpSet": DataTmpSet,  # 新增
    "DataTmpGet": DataTmpGet  # 新增
}

# 節點顯示名稱映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "TextBatch": "Text Batch", 
    "TextQueueProcessor": "Text Queue Processor",
    "TextSplitCounter": "Text Split Counter",
    "ImageQueueProcessor": "Image Queue Processor",
    "ImageInfoExtractor": "Image Info Extractor",
    "PathParser": "Path Parser",
    "LoadImagesFromDirBatch": "Load Images From Dir Batch",
    "ImageFilenameProcessor": "Image Filename Processor",
    "ImageQueueProcessorPlus": "Image Queue Processor Plus",  # 新增
    "TextSplitGet": "Text Split Get",  # 新增
    "IFMatchCond": "IF Match Condition",  # 新增
    "DataTmpSet": "Data Temp Set",  # 新增
    "DataTmpGet": "Data Temp Get"  # 新增
}