option("unity_editor_dir")
    set_default("C:/Program Files/Unity/Hub/Editor/6000.5.6f1/Editor")
    set_showmenu(true)
    set_description("Unity Editor directory used to build and package YutrelUnityPlugin")
option_end()

target("YutrelUnityPlugin")
    set_kind("shared")
    set_basename("YutrelUnityPlugin")

    add_files("src/**.cpp")
    add_headerfiles("src/**.h")
    add_includedirs("src")
    add_defines("NOMINMAX", "WIN32_LEAN_AND_MEAN")

    add_deps("YutrelCore", "lc-runtime", "lc-dsl", "lc-vstl")
    add_deps("lc-backend-dx", {inherit = false, links = false})
    add_syslinks("d3d12", "dxgi")

    on_load(function(target)
        if not target:is_plat("windows") or not target:is_arch("x64", "x86_64") then
            raise("YutrelUnityPlugin currently supports Windows x64 only.")
        end

        local unity_editor_dir = get_config("unity_editor_dir")
        local plugin_api_dir = path.join(unity_editor_dir, "Data", "PluginAPI")
        if not os.isfile(path.join(plugin_api_dir, "IUnityGraphicsD3D12.h")) then
            raise("Unity D3D12 PluginAPI headers were not found in " .. plugin_api_dir)
        end
        target:add("includedirs", plugin_api_dir)
    end)

    after_build(function(target)
        local unity_editor_dir = get_config("unity_editor_dir")
        local target_dir = target:targetdir()
        local package_dir = path.join(
            os.projectdir(),
            "unity_package",
            "com.yutrel.path-tracer",
            "Runtime",
            "Plugins",
            "x86_64")
        os.mkdir(package_dir)

        local files = {
            {target:targetfile(), "YutrelUnityPlugin.dll"},
            {path.join(target_dir, "luisa-core.dll"), "luisa-core.dll"},
            {path.join(target_dir, "luisa-runtime.dll"), "luisa-runtime.dll"},
            {path.join(target_dir, "luisa-backend-dx.dll"), "luisa-backend-dx.dll"},
            {path.join(target_dir, "luisa-gui.dll"), "luisa-gui.dll"},
            {path.join(target_dir, "luisa-ext-imgui.dll"), "luisa-ext-imgui.dll"},
            {path.join(target_dir, "luisa-ext-glfw.dll"), "luisa-ext-glfw.dll"},
            {path.join(unity_editor_dir, "Data", "Tools", "dxcompiler.dll"), "dxcompiler.dll"},
            {path.join(unity_editor_dir, "Data", "Tools", "dxil.dll"), "dxil.dll"},
        }

        for _, item in ipairs(files) do
            if not os.isfile(item[1]) then
                raise("Required Unity plugin runtime was not found: " .. item[1])
            end
            os.cp(item[1], path.join(package_dir, item[2]))
        end
    end)
target_end()

target("test_Yutrel_unity_environment_state")
    set_kind("binary")
    set_default(false)
    set_group("tests/Yutrel/unity_plugin")
    set_rundir("$(projectdir)")

    add_files("tests/environment_state.cpp", "src/environment_state.cpp")
    add_includedirs("src")
    add_deps("YutrelCore")
target_end()

target("test_Yutrel_unity_alpha_clip")
    set_kind("binary")
    set_default(false)
    set_group("tests/Yutrel/unity_plugin")
    set_rundir("$(projectdir)")

    add_files("tests/alpha_clip_texture.cpp", "src/alpha_clip_texture.cpp")
    add_includedirs("src")
    add_deps("YutrelCore", "lc-runtime", "lc-dsl", "lc-vstl")
target_end()
