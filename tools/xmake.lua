local reference_dir = get_config("openpbr_reference_dir")
if reference_dir ~= nil and reference_dir ~= "" then
    local glm_dir = get_config("openpbr_glm_dir")
    if glm_dir == nil or glm_dir == "" then
        local candidates = {
            "D:/Project/Vulkan/vkDuck/subprojects/glm-1.0.1",
            "D:/Project/Vulkan/lightweightvk/third-party/deps/src/glm",
        }
        for _, candidate in ipairs(candidates) do
            if os.isdir(path.join(candidate, "glm")) then
                glm_dir = candidate
                break
            end
        end
    end
    if glm_dir == nil or glm_dir == "" then
        add_requires("glm")
    end
    target("generate_openpbr_golden")
        set_kind("binary")
        set_default(false)
        set_group("tools/openpbr")
        set_rundir("$(projectdir)")
        add_files("openpbr_reference/generate_adobe_golden.cpp")
        add_includedirs(reference_dir)
        if glm_dir ~= nil and glm_dir ~= "" then
            add_includedirs(glm_dir)
        else
            add_packages("glm")
        end
        add_defines("YUTREL_OPENPBR_REFERENCE_DIR=\"" .. reference_dir:gsub('\\', '/') .. "\"")
    target_end()
end
