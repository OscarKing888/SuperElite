--[[
    SuperElite Export Service Provider
    摄影评片 - AI 深度分析引擎
    
    功能：
    - 分析选中的照片
    - 生成关键字、场景描述、场景分类
    - 用户确认后写入元数据
]]

local LrTasks = import 'LrTasks'
local LrApplication = import 'LrApplication'
local LrDialogs = import 'LrDialogs'
local LrLogger = import 'LrLogger'
local LrHttp = import 'LrHttp'
local LrView = import 'LrView'
local LrBinding = import 'LrBinding'
local LrFileUtils = import 'LrFileUtils'

-- 版本信息
local VERSION = "v1.0.0 - SuperElite 摄影评片"

-- 日志
local myLogger = LrLogger('SuperEliteExportServiceProvider')
myLogger:enable("logfile")

-- Binding helper
local bind = LrView.bind

-- Export service provider definition
local exportServiceProvider = {}

-- Required functions for Lightroom SDK
exportServiceProvider.supportsIncrementalPublish = false
exportServiceProvider.canExportVideo = false
exportServiceProvider.exportPresetDestination = "temp"

-- 不需要导出图片，只需获取原图路径
exportServiceProvider.allowFileFormats = nil
exportServiceProvider.allowColorSpaces = nil
exportServiceProvider.hideSections = { 'exportLocation', 'fileNaming', 'fileSettings', 'imageSettings', 'outputSharpening', 'metadata', 'watermarking' }

-- 预设字段
exportServiceProvider.exportPresetFields = {
    { key = 'apiUrl', default = "http://127.0.0.1:52765" },
    { key = 'generateKeywords', default = true },
    { key = 'generateCaption', default = true },
    { key = 'generateTitle', default = true },
    { key = 'writeExif', default = true },
}

-- Unicode转义解码
local function decodeUnicodeEscape(str)
    if not str then return str end
    
    local function unicodeToUtf8(code)
        code = tonumber(code, 16)
        if code < 0x80 then
            return string.char(code)
        elseif code < 0x800 then
            return string.char(
                0xC0 + math.floor(code / 0x40),
                0x80 + (code % 0x40)
            )
        elseif code < 0x10000 then
            return string.char(
                0xE0 + math.floor(code / 0x1000),
                0x80 + (math.floor(code / 0x40) % 0x40),
                0x80 + (code % 0x40)
            )
        end
        return "?"
    end
    
    return str:gsub("\\u(%x%x%x%x)", unicodeToUtf8)
end

-- 简单的JSON解析函数
local function parseJSON(jsonString)
    local result = {}
    
    -- 提取 success 字段
    local success = string.match(jsonString, '"success"%s*:%s*([^,}]+)')
    if success then
        result.success = (success == "true")
    end
    
    -- 提取 status 字段
    local status = string.match(jsonString, '"status"%s*:%s*"([^"]*)"')
    result.status = status
    
    -- 提取 keywords 字段
    local keywords_raw = string.match(jsonString, '"keywords"%s*:%s*"([^"]*)"')
    result.keywords = decodeUnicodeEscape(keywords_raw)
    
    -- 提取 caption 字段
    local caption_raw = string.match(jsonString, '"caption"%s*:%s*"([^"]*)"')
    result.caption = decodeUnicodeEscape(caption_raw)
    
    -- 提取 title 字段
    local title_raw = string.match(jsonString, '"title"%s*:%s*"([^"]*)"')
    result.title = decodeUnicodeEscape(title_raw)
    
    -- 提取 scene 字段
    local scene_raw = string.match(jsonString, '"scene"%s*:%s*"([^"]*)"')
    result.scene = decodeUnicodeEscape(scene_raw)
    
    -- 提取 mood 字段
    local mood_raw = string.match(jsonString, '"mood"%s*:%s*"([^"]*)"')
    result.mood = decodeUnicodeEscape(mood_raw)
    
    -- 提取错误信息
    local error_raw = string.match(jsonString, '"error"%s*:%s*"([^"]*)"')
    result.error = decodeUnicodeEscape(error_raw)
    
    return result
end

-- 简单的JSON编码函数
local function encodeJSON(tbl)
    local parts = {}
    for k, v in pairs(tbl) do
        local key = '"' .. tostring(k) .. '"'
        local value
        if type(v) == "string" then
            value = '"' .. v:gsub('"', '\\"'):gsub('\\', '\\\\') .. '"'
        elseif type(v) == "boolean" then
            value = tostring(v)
        elseif type(v) == "number" then
            value = tostring(v)
        else
            value = '"' .. tostring(v) .. '"'
        end
        table.insert(parts, key .. ":" .. value)
    end
    return "{" .. table.concat(parts, ",") .. "}"
end

-- 分析单张照片
local function analyzeSinglePhoto(photo, apiUrl)
    local photoPath = photo:getRawMetadata("path")
    local photoName = photo:getFormattedMetadata("fileName") or "Unknown"
    
    -- 检查文件是否存在
    if not LrFileUtils.exists(photoPath) then
        return {
            success = false,
            error = "文件不存在: " .. photoName,
            photoName = photoName
        }
    end
    
    -- 构建API请求
    local requestBody = encodeJSON({
        image_path = photoPath,
        generate_keywords = true,
        generate_caption = true,
        generate_title = true,
    })
    
    myLogger:info("发送分析请求: " .. photoPath)
    
    -- 调用API
    local response, headers = LrHttp.post(
        apiUrl .. "/analyze",
        requestBody,
        {
            { field = "Content-Type", value = "application/json" }
        },
        60  -- 60秒超时 (Co-Instruct 较慢)
    )
    
    if not response then
        return {
            success = false,
            error = "API调用失败，请确保 SuperElite GUI 已打开并启用深度分析引擎",
            photoName = photoName
        }
    end
    
    -- 解析响应
    local result = parseJSON(response)
    result.photoName = photoName
    result.photo = photo
    
    return result
end

-- 保存分析结果到照片元数据
local function saveAnalysisResult(photo, title, caption, keywords)
    local catalog = LrApplication.activeCatalog()
    
    catalog:withWriteAccessDo("保存 SuperElite 分析结果", function()
        if title and title ~= "" then
            photo:setRawMetadata("title", title)
        end
        
        if caption and caption ~= "" then
            photo:setRawMetadata("caption", caption)
        end
        
        -- keywords 需要特殊处理 (逗号分隔转数组)
        if keywords and keywords ~= "" then
            local keywordList = {}
            for kw in string.gmatch(keywords, "([^,]+)") do
                local trimmed = kw:match("^%s*(.-)%s*$")
                if trimmed and trimmed ~= "" then
                    table.insert(keywordList, trimmed)
                end
            end
            
            -- 写入关键字
            for _, kw in ipairs(keywordList) do
                local keyword = catalog:createKeyword(kw, {}, true, nil, true)
                photo:addKeyword(keyword)
            end
        end
    end)
end

-- UI配置面板
function exportServiceProvider.sectionsForTopOfDialog(f, propertyTable)
    return {
        {
            title = "🤖 SuperElite 深度分析配置",
            
            synopsis = bind { key = 'apiUrl', object = propertyTable },
            
            f:row {
                spacing = f:control_spacing(),
                
                f:static_text {
                    title = "API 地址:",
                    width = LrView.share "label_width",
                },
                
                f:edit_field {
                    value = bind 'apiUrl',
                    width_in_chars = 30,
                    tooltip = "SuperElite API 服务地址，默认: http://127.0.0.1:52765",
                },
            },
            
            f:row {
                spacing = f:control_spacing(),
                
                f:checkbox {
                    title = "生成关键字 (Keywords)",
                    value = bind 'generateKeywords',
                    tooltip = "AI 自动生成描述性关键字",
                },
            },
            
            f:row {
                spacing = f:control_spacing(),
                
                f:checkbox {
                    title = "生成场景描述 (Caption)",
                    value = bind 'generateCaption',
                    tooltip = "AI 生成详细的场景描述",
                },
            },
            
            f:row {
                spacing = f:control_spacing(),
                
                f:checkbox {
                    title = "生成标题 (Title)",
                    value = bind 'generateTitle',
                    tooltip = "AI 生成简短的诗意标题",
                },
            },
            
            f:separator { fill_horizontal = 1 },
            
            f:row {
                spacing = f:control_spacing(),
                
                f:checkbox {
                    title = "分析后自动写入元数据",
                    value = bind 'writeExif',
                    checked_value = true,
                    unchecked_value = false,
                    tooltip = "分析成功后自动写入元数据（需确认）",
                },
            },
            
            f:row {
                spacing = f:control_spacing(),
                
                f:static_text {
                    title = "💡 提示: 请确保 SuperElite GUI 已打开并启用「深度分析引擎」",
                    text_color = import 'LrColor'(0.5, 0.5, 0.5),
                },
            },
        },
    }
end

-- 主处理函数
function exportServiceProvider.processRenderedPhotos(functionContext, exportContext)
    myLogger:info("📷 SuperElite 深度分析启动 - " .. VERSION)
    
    local exportSettings = exportContext.propertyTable
    local apiUrl = exportSettings.apiUrl or "http://127.0.0.1:52765"
    local writeExif = exportSettings.writeExif
    if writeExif == nil then writeExif = true end
    
    -- 计算照片数量
    local nPhotos = exportContext.nPhotos or 1
    myLogger:info("待处理照片数: " .. nPhotos)
    
    -- 检查照片数量
    if nPhotos == 0 then
        LrDialogs.message("📷 SuperElite 深度分析 - " .. VERSION,
            "❌ 没有选中要处理的照片\n\n请先选择一张照片再进行分析",
            "error")
        return
    elseif nPhotos > 1 then
        LrDialogs.message("📷 SuperElite 深度分析 - " .. VERSION,
            "⚠️ 一次只能分析一张照片\n\n" ..
            "当前选中: " .. nPhotos .. " 张照片\n\n" ..
            "请重新选择，只选中一张照片后再次导出",
            "warning")
        return
    end
    
    -- 检查API服务是否可用
    myLogger:info("检查API服务: " .. apiUrl .. "/status")
    local healthCheck, headers = LrHttp.get(apiUrl .. "/status", nil, 5)
    
    if not healthCheck or string.find(healthCheck, '"status"%s*:%s*"running"') == nil then
        LrDialogs.message("📷 SuperElite 深度分析 - " .. VERSION,
            "❌ 无法连接到 SuperElite 深度分析引擎\n\n" ..
            "请确保:\n" ..
            "1. SuperElite GUI 应用已打开\n" ..
            "2. 已勾选「深度分析引擎」开关\n\n" ..
            "服务地址: " .. apiUrl,
            "error")
        return
    end
    
    myLogger:info("✅ API服务正常，开始分析...")
    
    -- 处理单张照片
    for i, rendition in exportContext:renditions() do
        local photo = rendition.photo
        local result = analyzeSinglePhoto(photo, apiUrl)
        
        if result.success or (result.keywords or result.caption or result.title) then
            myLogger:info("分析成功: " .. (result.photoName or "unknown"))
            
            -- 构建结果消息
            local message = "✅ 分析完成！\n\n"
            
            if result.title and result.title ~= "" then
                message = message .. "📌 标题:\n" .. result.title .. "\n\n"
            end
            
            if result.scene and result.scene ~= "" then
                message = message .. "🏞️ 场景: " .. result.scene .. "\n"
            end
            
            if result.mood and result.mood ~= "" then
                message = message .. "💫 氛围: " .. result.mood .. "\n\n"
            end
            
            if result.keywords and result.keywords ~= "" then
                message = message .. "🏷️ 关键字:\n" .. result.keywords .. "\n\n"
            end
            
            if result.caption and result.caption ~= "" then
                message = message .. "📝 描述:\n" .. result.caption .. "\n"
            end
            
            -- 显示分析结果，询问是否保存
            local action = LrDialogs.confirm(
                "📷 SuperElite 分析完成 - " .. VERSION,
                message .. "\n\n是否保存分析结果到照片元数据？",
                "确认保存",
                "取消"
            )
            
            if action == "ok" and writeExif then
                saveAnalysisResult(photo, result.title, result.caption, result.keywords)
                myLogger:info("✅ 用户确认保存，已写入元数据")
                
                LrDialogs.message("📷 SuperElite 摄影评片",
                    "✅ 元数据写入成功！\n\n" ..
                    "请在元数据面板中查看结果。",
                    "info")
            else
                myLogger:info("❌ 用户取消保存")
            end
            
        else
            local errorMsg = result.error or "未知错误"
            myLogger:info("分析失败: " .. errorMsg)
            
            -- 检查是否是功能未实现
            if result.status == "not_implemented" then
                LrDialogs.message("📷 SuperElite 深度分析 - " .. VERSION,
                    "🚧 AI 分析功能开发中\n\n" ..
                    "当前版本仅支持连接测试。\n" ..
                    "完整的 AI 分析功能将在下一版本发布。",
                    "info")
            else
                LrDialogs.message("📷 SuperElite 分析失败 - " .. VERSION,
                    "❌ 分析过程中出现错误:\n\n" .. errorMsg .. "\n\n" ..
                    "请检查:\n" ..
                    "• 图片文件是否完整\n" ..
                    "• SuperElite 服务是否正常运行",
                    "error")
            end
        end
        break  -- 只处理一张
    end
    
    myLogger:info("📷 SuperElite 深度分析处理完成")
end

return exportServiceProvider
