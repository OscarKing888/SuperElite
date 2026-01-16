--[[
    SuperElite Lightroom Plugin - 连接测试
    测试与 SuperElite GUI 深度分析引擎的连接
]]

local LrHttp = import 'LrHttp'
local LrDialogs = import 'LrDialogs'
local LrTasks = import 'LrTasks'

local API_URL = "http://127.0.0.1:52765"

-- 测试连接
local function testConnection()
    LrTasks.startAsyncTask(function()
        local url = API_URL .. "/status"
        
        -- 发送 GET 请求
        local response, headers = LrHttp.get(url, nil, 5) -- 5秒超时
        
        if response then
            -- 连接成功
            LrDialogs.message(
                "SuperElite 摄影评片",
                "✅ 连接成功！\n\n" ..
                "深度分析引擎正在运行。\n\n" ..
                "现在可以通过「文件 → 导出」使用：\n" ..
                "📷 SuperElite 深度分析",
                "info"
            )
        else
            -- 连接失败
            LrDialogs.message(
                "SuperElite 摄影评片",
                "❌ 无法连接到深度分析引擎\n\n" ..
                "请确保：\n" ..
                "1. SuperElite GUI 应用已打开\n" ..
                "2. 已勾选「深度分析引擎」开关\n\n" ..
                "连接地址: " .. API_URL,
                "warning"
            )
        end
    end)
end

-- 执行
testConnection()
