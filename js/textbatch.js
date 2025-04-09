import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// 用於存儲節點 ID 映射
const nodeIdMap = new Map();

// 註冊自定義事件處理器
api.addEventListener("textbatch-node-feedback", (event) => {
    console.log("Received node feedback:", event);
    try {
        // 從 CustomEvent 中獲取 data
        const data = event.detail;
        
        // 檢查 data 物件的完整性
        if (!data || !data.node_id) {
            console.error("Invalid data received:", data);
            return;
        }

        const nodeId = data.node_id;
        console.log("Looking for node:", nodeId, "Data received:", data);
        
        // 嘗試從 nodeIdMap 中獲取節點
        let node = nodeIdMap.get(nodeId);
        
        // 如果在 Map 中找不到，再嘗試其他方法
        if (!node) {
            node = app.graph._nodes_by_id?.[nodeId] || 
            app.graph.getNodeById?.(parseInt(nodeId)) ||  
            [...(app.graph?.nodes || [])].find(n => n?.id == nodeId);  
        }
                  
        if (!node) {
            console.warn("Node not found by ID:", nodeId, "Available nodes:", 
                        Array.from(nodeIdMap.keys()));
            return;
        }

        console.log("Found node:", node);
        const widget = node.widgets?.find(w => w.name === data.widget_name);
        if (!widget) {
            console.warn("Widget not found:", data.widget_name);
            return;
        }

        if (data.type === "int") {
            console.log("Updating widget value:", data.value);
            widget.value = parseInt(data.value);
        } else {
            widget.value = data.value;
        }
        
        // 觸發小部件的變更事件
        if (widget.callback) {
            widget.callback(widget.value);
        }
    } catch (error) {
        console.error("Error in node feedback handler:", error);
    }
});

// 註冊佇列事件處理器
api.addEventListener("textbatch-add-queue", (data) => {
    try {
        console.log("Received queue event:", data);
        
        // 檢查是否正在處理中
        if (app.isProcessing) {
            console.log("Already processing, queueing next prompt");
        }
        
        // 獲取當前工作流程
        const workflow = app.graph?.serialize?.();  // ✅ 安全访问
        console.log("Current workflow:", workflow);
        
        // 確保在下一個事件循環中執行
        setTimeout(() => {
            try {
                console.log("Executing queued prompt");
                // 使用 queuePrompt 的完整參數
                app.queuePrompt?.(0, 1);  // ✅ 兼容性检查
                console.log("Queue prompt executed");
            } catch (queueError) {
                console.error("Error queueing prompt:", queueError);
            }
        }, 100);
    } catch (error) {
        console.error("Error in textbatch-add-queue handler:", error);
        console.error("Error details:", {
            message: error.message,
            stack: error.stack
        });
    }
});

// 為特定節點添加自定義行為
app.registerExtension({
    name: "TextBatch.TextBatchNode",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        //console.log("Registering node type:", nodeData.name);
        
        if (nodeData.name === "TextBatch" || 
            nodeData.name === "TextQueueProcessor" || 
            nodeData.name === "ZippedPromptBatch" ||
            nodeData.name === "ZippedPromptBatchAdvanced") {
            
            console.log("Adding custom behavior to node:", nodeData.name);
            
            // 添加自定義小部件行為
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const r = onNodeCreated?.apply?.(this, arguments);  // ✅ 安全访问
                
                // 確保節點有有效的 ID
                if (!this?.id || this.id === -1) {  // ✅ 可选链检查
                    console.warn("Invalid node ID detected, waiting for proper initialization");
                    // 等待下一個事件循環再進行初始化
                    setTimeout(() => {
                        console.log("Retrying node initialization:", nodeData.name, "ID:", this.id);
                        // 存儲節點 ID
                        if (this?.id && this.id !== -1) {  // ✅ 双重检查
                            nodeIdMap.set(this.id, this);
                            // 為節點添加自定義標題
                            this.addWidget?.("text", "status", "", (v) => {  // ✅ 兼容性检查
                                console.log("Status widget updated:", v);
                                this.status = v;
                            });
                        }
                    }, 0);
                } else {
                    console.log("Node created:", nodeData.name, "ID:", this.id);
                    nodeIdMap.set(this.id, this);
                    this.addWidget?.("text", "status", "", (v) => {  // ✅ 兼容性检查
                        console.log("Status widget updated:", v);
                        this.status = v;
                    });
                }
                
                return r;
            };

            // 添加節點刪除處理
            const onNodeRemoved = nodeType.prototype.onRemoved;
            nodeType.prototype.onRemoved = function() {
                console.log("Node removed:", this.id);
                if (this?.id) {  // ✅ 关键修复：添加存在性检查
                    console.log("Node removed:", this.id);
                    nodeIdMap.delete(this.id);
                }
                onNodeRemoved?.apply?.(this, arguments);  // ✅ 安全调用
            };
        }
    }
}); 

// api.addEventListener('executed', async ({ detail }) => {
//     console.log('#executed', detail) 
//     console.log(output)
// })

// TextQueueProcessor 節點擴展
class TextQueueProcessorNode {
    constructor() {
        if (!this.properties) {
            this.properties = {};
        }
        this.addCustomWidgets?.(); // ✅ 兼容性检查
    }

    addCustomWidgets() {
        // 添加重置按鈕
        this.addWidget?.("button", "🔄 Reset", null, () => {  // ✅ 安全访问
            this.triggerReset?.();  // ✅ 兼容性检查
        });

        // 添加跳到開頭按鈕
        this.addWidget("button", "⏮️ To Start", null, () => {
            // 將 start_index 設為 0
            this.widgets.find(w => w.name === "start_index").value = 0;
        });

        // 添加跳到結尾按鈕
        this.addWidget("button", "⏭️ To End", null, () => {
            // 獲取 total 值（如果有的話）
            const total = this.outputs?.[2]?.value ?? 0;
            if (total > 0) {
                this.widgets.find(w => w.name === "start_index").value = total - 1;
            }
        });
    }

    triggerReset() {
        // 發送重置事件到後端
        const nodeId = this?.id;  // ✅ 安全访问
        app.graphToPrompt?.().then(workflow => {  // ✅ 兼容性检查
            if (workflow.output) {
                app.queuePrompt(workflow.output, workflow.workflow);
            }
        });
    }
}

// ImageQueueProcessor 節點擴展
class ImageQueueProcessorNode {
    constructor() {
        if (!this.properties) {
            this.properties = {};
        }
        this.addCustomWidgets();
    }

    addCustomWidgets() {
        // 添加重置按鈕
        this.addWidget("button", "🔄 Reset", null, () => {
            // 觸發重置
            this.triggerReset();
        });

        // 添加跳到開頭按鈕
        this.addWidget("button", "⏮️ To Start", null, () => {
            // 將 start_index 設為 0
            this.widgets.find(w => w.name === "start_index").value = 0;
        });

        // 添加跳到結尾按鈕
        this.addWidget("button", "⏭️ To End", null, () => {
            // 獲取 total 值（如果有的話）
            const total = this.outputs?.[3]?.value ?? 0;
            if (total > 0) {
                this.widgets.find(w => w.name === "start_index").value = total - 1;
            }
        });
    }

    triggerReset() {
        // 發送重置事件到後端
        const nodeId = this.id;
        app.graphToPrompt().then(workflow => {
            if (workflow.output) {
                app.queuePrompt(workflow.output, workflow.workflow);
            }
        });
    }
}

// 註冊節點擴展
app.registerExtension({
    name: "rgthree.TextBatch",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "TextQueueProcessor") {
            Object.assign(nodeType.prototype, TextQueueProcessorNode.prototype);
        }
        else if (nodeData.name === "ImageQueueProcessor") {
            Object.assign(nodeType.prototype, ImageQueueProcessorNode.prototype);
        }
    }
});