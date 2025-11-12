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

// ============ 群組控制器節點 ============
// 監聽後端的群組狀態更新事件
api.addEventListener("groupcontroller-update", (event) => {
    try {
        const data = event.detail;
        if (!data || !data.node_id) return;
        
        const node = app.graph.getNodeById(parseInt(data.node_id));
        if (!node) return;
        
        // 應用群組狀態
        if (node.applyGroupStates) {
            node.applyGroupStates(data.group_states, data.control_mode);
        }
    } catch (error) {
        console.error("Error in groupcontroller-update handler:", error);
    }
});

// 註冊群組控制器節點
app.registerExtension({
    name: "TextBatch.GroupController",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "GroupController") {
            console.log("Registering GroupController node");
            
            // 修改 INPUT_TYPES 來動態添加群組 inputs
            const originalGetExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
            nodeType.prototype.getExtraMenuOptions = function(_, options) {
                if (originalGetExtraMenuOptions) {
                    originalGetExtraMenuOptions.apply(this, arguments);
                }
                
                options.unshift(
                    {
                        content: "🔄 重新掃描群組",
                        callback: () => {
                            this.refreshGroupInputs();
                        }
                    },
                    null
                );
            };
            
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const r = onNodeCreated?.apply(this, arguments);
                
                // 初始化屬性
                this.groupsData = [];
                this.lastGroupSignature = "";
                this.lastInputValues = {}; // 存儲上次的 input 值
                
                // 延遲載入群組以確保 graph 已初始化
                setTimeout(() => {
                    this.refreshGroupInputs();
                }, 300);
                
                // 啟動輪詢機制，每 100ms 檢查一次輸入值變化
                this.updateInterval = setInterval(() => {
                    this.checkAndApplyInputChanges();
                }, 100);
                
                console.log(`GroupController ${this.id} created with polling enabled`);
                
                return r;
            };
            
            // 獲取所有群組
            nodeType.prototype.getAllGroups = function() {
                if (!app.canvas || !app.canvas.graph) {
                    return [];
                }

                const groups = [];
                
                // 從 _groups 獲取
                if (app.canvas.graph._groups && Array.isArray(app.canvas.graph._groups)) {
                    groups.push(...app.canvas.graph._groups);
                }
                
                return groups;
            };
            
            // 獲取群組中的所有節點
            nodeType.prototype.getNodesInGroup = function(group) {
                if (!app.canvas || !app.canvas.graph || !app.canvas.graph._nodes) {
                    return [];
                }

                const nodes = [];
                for (const node of app.canvas.graph._nodes) {
                    if (this.isNodeInGroup(node, group)) {
                        nodes.push(node);
                    }
                }
                
                return nodes;
            };
            
            // 判斷節點是否在群組內
            nodeType.prototype.isNodeInGroup = function(node, group) {
                if (!node || !group) return false;
                
                const nodeX = node.pos[0];
                const nodeY = node.pos[1];
                const nodeWidth = node.size[0];
                const nodeHeight = node.size[1];
                
                const groupX = group._pos[0];
                const groupY = group._pos[1];
                const groupWidth = group._size[0];
                const groupHeight = group._size[1];
                
                const nodeCenterX = nodeX + nodeWidth / 2;
                const nodeCenterY = nodeY + nodeHeight / 2;
                
                return nodeCenterX >= groupX && nodeCenterX <= groupX + groupWidth &&
                       nodeCenterY >= groupY && nodeCenterY <= groupY + groupHeight;
            };
            
            // 刷新群組 inputs
            nodeType.prototype.refreshGroupInputs = function() {
                console.log("Refreshing group inputs...");
                
                const groups = this.getAllGroups();
                console.log("Found groups:", groups.length);
                
                if (groups.length === 0) {
                    console.warn("No groups found");
                    return;
                }
                
                // 生成群組簽名以檢測變化
                const groupSignature = groups.map(g => `${g.title}_${g._pos.join(',')}_${g._size.join(',')}`).join('|');
                
                // 如果群組沒有變化，不需要重建
                if (this.lastGroupSignature === groupSignature) {
                    console.log("Groups unchanged, skipping rebuild");
                    return;
                }
                
                this.lastGroupSignature = groupSignature;
                this.groupsData = groups;
                
                // 移除舊的群組 inputs 和 widgets
                const inputsToRemove = [];
                if (this.inputs) {
                    for (let i = 0; i < this.inputs.length; i++) {
                        const input = this.inputs[i];
                        if (input.name.startsWith("enable_group_")) {
                            inputsToRemove.push(i);
                        }
                    }
                }
                
                // 從後往前刪除以避免索引問題
                for (let i = inputsToRemove.length - 1; i >= 0; i--) {
                    this.removeInput(inputsToRemove[i]);
                }
                
                // 移除舊的群組 widgets（保留 control_mode）
                if (this.widgets) {
                    const widgetsToKeep = this.widgets.filter(w => 
                        w.name === "control_mode" || !w.name.startsWith("enable_group_")
                    );
                    this.widgets = widgetsToKeep;
                }
                
                // 為每個群組添加 BOOLEAN input 和對應的 widget
                for (const group of groups) {
                    const groupId = this.sanitizeGroupId(group.title || group.id);
                    const inputName = `enable_group_${groupId}`;
                    const displayName = `Enable ${group.title || groupId}`;
                    
                    // 添加 input（連接點）
                    this.addInput(inputName, "BOOLEAN", {
                        label: displayName
                    });
                    
                    // 添加對應的 toggle widget
                    const widget = this.addWidget(
                        "toggle",
                        inputName,
                        true,
                        (value) => {
                            console.log(`Widget ${inputName} changed to:`, value);
                            // widget 變更時立即應用狀態
                            const modeWidget = this.widgets?.find(w => w.name === "control_mode");
                            const currentMode = modeWidget ? modeWidget.value : "bypass";
                            const groupStates = this.getGroupStates();
                            this.applyGroupStates(groupStates, currentMode);
                        },
                        { on: "yes", off: "no" }
                    );
                    
                    // 將 widget 與 input 關聯
                    widget.linkedInput = inputName;
                    
                    console.log(`Added input and widget: ${inputName}`);
                }
                
                // 更新節點大小
                this.setSize(this.computeSize());
                
                console.log(`Loaded ${groups.length} group inputs`);
            };
            
            // 清理群組 ID（移除特殊字符）
            nodeType.prototype.sanitizeGroupId = function(id) {
                return String(id).replace(/[^a-zA-Z0-9_]/g, '_');
            };
            
            // 應用群組狀態
            nodeType.prototype.applyGroupStates = function(groupStates, controlMode) {
                console.log("Applying group states:", groupStates, "Mode:", controlMode);
                
                for (const group of this.groupsData) {
                    const groupId = this.sanitizeGroupId(group.title || group.id);
                    const enabled = groupStates[groupId] !== undefined ? groupStates[groupId] : true;
                    
                    this.setGroupState(group, enabled, controlMode);
                }
            };
            
            // 設置群組狀態
            nodeType.prototype.setGroupState = function(group, enabled, mode) {
                const nodes = this.getNodesInGroup(group);
                console.log(`Setting group "${group.title}" to ${enabled ? 'enabled' : 'disabled'} (mode: ${mode}), nodes: ${nodes.length}`);
                
                for (const node of nodes) {
                    if (mode === "bypass") {
                        node.mode = enabled ? 0 : 4; // 0 = ALWAYS, 4 = BYPASS
                    } else if (mode === "mute") {
                        node.mode = enabled ? 0 : 2; // 0 = ALWAYS, 2 = NEVER (muted)
                    }
                }
                
                // 重繪畫布
                if (app.canvas) {
                    app.canvas.setDirty(true, true);
                }
            };
            
            // 檢查並應用輸入值變化
            nodeType.prototype.checkAndApplyInputChanges = function() {
                if (!this.inputs || !this.groupsData || this.groupsData.length === 0) {
                    return;
                }
                
                let hasChanged = false;
                const currentValues = {};
                
                // 獲取當前所有輸入值
                for (const input of this.inputs) {
                    if (!input.name.startsWith("enable_group_")) continue;
                    
                    const groupId = input.name.replace("enable_group_", "");
                    let currentValue = true; // 預設值
                    
                    // 檢查是否有連接
                    const link = input.link;
                    if (link !== null && link !== undefined) {
                        const linkInfo = app.graph.links[link];
                        if (linkInfo) {
                            const originNode = app.graph.getNodeById(linkInfo.origin_id);
                            if (originNode) {
                                const outputIndex = linkInfo.origin_slot;
                                
                                // 嘗試從 widget 獲取值
                                if (originNode.widgets && originNode.widgets.length > 0) {
                                    const widget = originNode.widgets.find(w => 
                                        w.name === "value" || w.name === "boolean_value" || 
                                        w.name === "BOOLEAN" || w.type === "toggle"
                                    ) || originNode.widgets[0];
                                    
                                    if (widget && widget.value !== undefined) {
                                        currentValue = widget.value;
                                    }
                                }
                            }
                        }
                    } else {
                        // 沒有連接，使用本地 widget 的值
                        const widget = this.widgets?.find(w => w.name === input.name);
                        if (widget) {
                            currentValue = widget.value !== false;
                        }
                    }
                    
                    currentValues[groupId] = currentValue;
                    
                    // 檢查是否變化
                    if (this.lastInputValues[groupId] !== currentValue) {
                        hasChanged = true;
                        console.log(`GroupController ${this.id}: ${input.name} changed from ${this.lastInputValues[groupId]} to ${currentValue}`);
                    }
                }
                
                // 如果有變化，應用新狀態
                if (hasChanged) {
                    this.lastInputValues = currentValues;
                    
                    const modeWidget = this.widgets?.find(w => w.name === "control_mode");
                    const currentMode = modeWidget ? modeWidget.value : "bypass";
                    
                    this.applyGroupStates(currentValues, currentMode);
                }
            };
            
            // 獲取群組狀態（從 input 連接或 widget）
            nodeType.prototype.getGroupStates = function() {
                const groupStates = {};
                
                if (!this.inputs) return groupStates;
                
                for (const input of this.inputs) {
                    if (!input.name.startsWith("enable_group_")) continue;
                    
                    const groupId = input.name.replace("enable_group_", "");
                    
                    // 檢查是否有連接
                    const link = input.link;
                    if (link !== null && link !== undefined) {
                        // 有連接，嘗試從連接獲取最新值
                        const linkInfo = app.graph.links[link];
                        if (linkInfo) {
                            const originNode = app.graph.getNodeById(linkInfo.origin_id);
                            if (originNode) {
                                const outputIndex = linkInfo.origin_slot;
                                
                                // 先嘗試從 outputs[].value 獲取
                                if (originNode.outputs && originNode.outputs[outputIndex]) {
                                    const output = originNode.outputs[outputIndex];
                                    if (output.value !== undefined) {
                                        groupStates[groupId] = output.value;
                                        continue;
                                    }
                                }
                                
                                // 嘗試從 widget 獲取值（對於 bool 節點）
                                if (originNode.widgets && originNode.widgets.length > 0) {
                                    // 查找 BOOLEAN 類型的 widget
                                    const boolWidget = originNode.widgets.find(w => 
                                        w.name === "value" || w.name === "boolean_value" || w.type === "toggle"
                                    );
                                    if (boolWidget && boolWidget.value !== undefined) {
                                        groupStates[groupId] = boolWidget.value;
                                        continue;
                                    }
                                    // 如果找不到特定的，就使用第一個 widget 的值
                                    if (originNode.widgets[0].value !== undefined) {
                                        groupStates[groupId] = originNode.widgets[0].value;
                                        continue;
                                    }
                                }
                            }
                        }
                    }
                    
                    // 沒有連接或無法從連接獲取值，使用 widget 的值
                    const widget = this.widgets?.find(w => w.name === input.name);
                    if (widget) {
                        groupStates[groupId] = widget.value !== false;
                    } else {
                        groupStates[groupId] = true; // 預設啟用
                    }
                }
                
                return groupStates;
            };
            
            // 監聽執行前事件
            const onExecute = nodeType.prototype.onExecute;
            nodeType.prototype.onExecute = function() {
                // 獲取當前模式
                const modeWidget = this.widgets?.find(w => w.name === "control_mode");
                const currentMode = modeWidget ? modeWidget.value : "bypass";
                
                // 獲取所有群組狀態
                const groupStates = this.getGroupStates();
                
                // 應用狀態到群組
                this.applyGroupStates(groupStates, currentMode);
                
                if (onExecute) {
                    return onExecute.apply(this, arguments);
                }
            };
            
            // 監聽 widget 變更
            const onWidgetChanged = nodeType.prototype.onWidgetChanged;
            nodeType.prototype.onWidgetChanged = function(name, value, oldValue, widget) {
                if (onWidgetChanged) {
                    onWidgetChanged.apply(this, arguments);
                }
                
                // 如果是群組 widget 或 control_mode 變更，立即應用
                if (name.startsWith("enable_group_") || name === "control_mode") {
                    const modeWidget = this.widgets?.find(w => w.name === "control_mode");
                    const currentMode = modeWidget ? modeWidget.value : "bypass";
                    const groupStates = this.getGroupStates();
                    this.applyGroupStates(groupStates, currentMode);
                }
            };
            
            // 監聽連接變更
            const onConnectionsChange = nodeType.prototype.onConnectionsChange;
            nodeType.prototype.onConnectionsChange = function(type, index, connected, link_info) {
                if (onConnectionsChange) {
                    onConnectionsChange.apply(this, arguments);
                }
                
                // 當連接變更時，重新應用狀態
                if (type === 1) { // 1 = input
                    console.log(`GroupController ${this.id}: connection changed`);
                    setTimeout(() => {
                        const modeWidget = this.widgets?.find(w => w.name === "control_mode");
                        const currentMode = modeWidget ? modeWidget.value : "bypass";
                        const groupStates = this.getGroupStates();
                        this.applyGroupStates(groupStates, currentMode);
                    }, 100);
                }
            };
            
            // 清理定時器
            const onRemoved = nodeType.prototype.onRemoved;
            nodeType.prototype.onRemoved = function() {
                console.log(`GroupController ${this.id} removed, stopping polling`);
                if (this.updateInterval) {
                    clearInterval(this.updateInterval);
                    this.updateInterval = null;
                }
                if (onRemoved) {
                    return onRemoved.apply(this, arguments);
                }
            };
        }
    }
});