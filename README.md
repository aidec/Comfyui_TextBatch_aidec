# Comfyui_TextBatch_aidec
可以將Text每行分割成一組，批量執行(未完成，有明顯bug待修正中)
Text Queue Processor
目前來說第一次執行可能會只執行一筆，等它執行完再跑第二次就正常，每次執行完要將start_index手動重置
有時會出現奇怪的異常，例如：一次添加多個queue，但有時又很正常，目前還在排查中可能的原因(不要一次開好幾個ComfyUI頁面跟工作流，不然可能會出錯)

### 安裝方式：
到 ComfyUI\custom_nodes
用git clone https://github.com/aidec/Comfyui_TextBatch_aidec.git
啟動comfyUI

### 使用方式：

添加 Text Queue Processor 節點
輸入Text每行一組
將text拉到想要的地方，例如：正向提示詞(Clip)


### 新增功能Image Queue Processor(批量圖生圖)：
增加了 Image Queue Processor 節點，可以將圖片每行一組，批量執行(但目前需要搭配Load Image Batch From D(Inspire)這類的使用)
![image](https://github.com/user-attachments/assets/bc264fd8-042f-42c2-b66c-72639ca8a197)

### 新增功能Load Images From Dir Batch(批量讀取圖片從資料夾)：
這個節點的功能是由Load Image Batch From Dir(Inspire)這個節點修改而來，主要是因為Load Image Batch From Dir(Inspire)沒有輸出圖片檔名，無法得知當前的圖片名稱，因此製作了這個改良版。
![image](https://github.com/user-attachments/assets/71d577d4-da75-4c9e-baa1-ccd2d4d5694e)
filenames:輸出的是只有檔名 (圖片1.png,圖片2.png ...) 輸出這樣的字串 
full_paths:輸出的是圖片的完整路徑 (F:AI/ComfyUI/output/圖片1.png,F:AI/ComfyUI/output/圖片2.png ...) 的字串
這個功能主要用途是圖生圖時，可以利用這個讓參照圖跟輸出圖都用同一個檔名，方便識別。

### 新增功能Image Filename Processor(圖片檔名處理器)：
這個節點主要用途就是處理上面 F:AI/ComfyUI/output/圖片1.png,F:AI/ComfyUI/output/圖片2.png ... 這樣的字串，在帶入Index就能得到對應的圖片檔名。
![image](https://github.com/user-attachments/assets/51c68728-ca82-48f7-96df-4ee89b40b963)
用法：
Load Images From Dir Batch 的 filenames 或 full_paths 連結到Image Filename Processor 的filenames
Image Queue Processor 的current_index 連結到Image Filename Processor 的index
這樣就能在批次圖生圖時，獲取對應index的圖片檔名。
![image](https://github.com/user-attachments/assets/7e2468e3-db16-4934-9493-dab2db23e1fb)

### 新增功能 Path Parser (根據路徑解析檔名)：
可以輸出一個file_path，就能輸出檔名、副檔名、目錄路徑、完整路徑
![image](https://github.com/user-attachments/assets/eb9ee272-0284-4ca1-a85e-b57827dcb885)

### 預告功能:
之後應該會出一個Image Queue Processor Plus，把Image Queue Processor跟Load Images From Dir Batch結合一起，讓工作流簡化一點點。

其他節點測試用，無用途

