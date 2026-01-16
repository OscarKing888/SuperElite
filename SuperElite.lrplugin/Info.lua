--[[
    SuperElite Lightroom Plugin - 插件信息
    摄影评片 - AI 深度分析引擎
]]

return {
    LrSdkVersion = 11.0,
    LrSdkMinimumVersion = 8.0,
    
    LrToolkitIdentifier = "com.jamesphotography.superelite",
    LrPluginName = "📷 SuperElite 摄影评片",
    
    LrPluginInfoUrl = "https://github.com/jamesphotography/SuperElite",
    
    -- 导出服务提供商 (主要功能)
    LrExportServiceProvider = {
        {
            title = "📷 SuperElite 深度分析",
            file = "SuperEliteExportServiceProvider.lua",
        },
    },
    
    -- 库菜单项 (测试连接)
    LrLibraryMenuItems = {
        {
            title = "测试连接",
            file = "SuperEliteConnect.lua",
        },
    },
    
    VERSION = {
        major = 1,
        minor = 0,
        revision = 0,
        build = 1,
    },
}
