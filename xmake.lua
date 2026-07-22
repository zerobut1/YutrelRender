add_rules("mode.debug", "mode.release", "mode.releasedbg")

set_languages("cxx20")
set_encodings("utf-8")

add_requires("tinyexr")
add_requires("assimp")

if is_host("windows") then
    lc_options = {
        lc_cuda_backend = true,
        lc_dx_backend = true,
        lc_vk_backend = true,
        lc_metal_backend = false,
        lc_enable_gui = true,
        lc_enable_imgui = true,
        lc_enable_tests = false,
        lc_enable_osl = false,
        lc_enable_mimalloc = true,
        lc_enable_py = false,
        lc_enable_clangcxx = false,
        lc_enable_xir = false,
        lc_vk_backend_use_xir_spirv = false,
        lc_vk_backend_use_ast_llvm_spirv = false,
        lc_fallback_backend = false,
        lc_toy_c_backend = false,
        lc_cuda_ext_lcub = false,
        lc_dx_cuda_interop = false,
        lc_vk_cuda_interop = false,
    }
elseif is_host("macosx") then
    lc_options = {
        lc_cuda_backend = false,
        lc_dx_backend = false,
        lc_vk_backend = false,
        lc_metal_backend = true,
        lc_enable_gui = true,
        lc_enable_imgui = true,
        lc_enable_tests = false,
        lc_enable_osl = false,
        lc_enable_mimalloc = false,
        lc_enable_py = false,
        lc_enable_clangcxx = false,
        lc_enable_xir = false,
        lc_fallback_backend = false,
        lc_toy_c_backend = false,
        lc_cuda_ext_lcub = false,
        lc_dx_cuda_interop = false,
        lc_vk_cuda_interop = false,
    }
end

includes("ext/LuisaCompute")
includes("src")
includes("test")

-- Enable CUDA device runtime (cudadevrt) embedding for LuisaCompute CUDA backend.
-- This removes the runtime warning:
--   "CUDA Device Runtime library is not linked. Indirect dispatch will not be available..."
-- We patch the already-defined target from ext/LuisaCompute here.
after_load(function()
    import("core.project.project")

    local cuda_backend = project.target("lc-backend-cuda")
    if cuda_backend == nil then
        return
    end

    -- CUDA只在Windows和Linux上可用
    if not is_plat("windows", "linux") then
        return
    end

    -- CUDA 13.x static NVRTC requires ntdll on Windows.
    if is_plat("windows") then
        local nvrtc = project.target("lc-nvrtc")
        if nvrtc ~= nil then
            nvrtc:add("syslinks", "ntdll")
        end
    end

    -- Prefer xmake's configured CUDA SDK over the process environment, which
    -- may still point to an older Toolkit after an upgrade.
    local cuda_path = get_config("cuda")
    if cuda_path == nil or cuda_path == "" then
        import("cuda_sdkdir", {rootdir = get_config("lc_scripts_path")})
        cuda_path = cuda_sdkdir()
    end
    if cuda_path == nil or cuda_path == "" then
        wprint("CUDA SDK not found; cannot locate cudadevrt. CUDA indirect dispatch will remain unavailable.")
        return
    end

    -- 根据平台构建正确的路径
    local cudadevrt
    if is_plat("windows") then
        cudadevrt = path.join(cuda_path, "lib", "x64", "cudadevrt.lib")
    else
        -- Linux
        cudadevrt = path.join(cuda_path, "lib64", "libcudadevrt.a")
    end

    if not os.isfile(cudadevrt) then
        wprint("cudadevrt library not found at '%s'. CUDA indirect dispatch will remain unavailable.", cudadevrt)
        return
    end

    local gen_dir = path.join(os.projectdir(), "build", "generated", "luisa", "cuda_devrt")
    local gen_cpp = path.join(gen_dir, "cuda_devrt_embedded.cpp")
    os.mkdir(gen_dir)

    local function _generate_bin2c(input_file, output_cpp)
        local fin = io.open(input_file, "rb")
        if fin == nil then
            raise("failed to open file: " .. input_file)
        end
        local data = fin:read("*all")
        fin:close()

        local fout = io.open(output_cpp, "w")
        if fout == nil then
            raise("failed to write file: " .. output_cpp)
        end
        fout:write("// GENERATED FILE. DO NOT EDIT.\n")
        fout:write("extern \"C\" const unsigned char luisa_compute_cudadevrt[" .. tostring(#data) .. "] = {\n")
        for i = 1, #data do
            fout:write(tostring(data:byte(i)))
            if i ~= #data then
                fout:write(",")
            end
            if i % 32 == 0 then
                fout:write("\n")
            end
        end
        fout:write("\n};\n")
        fout:write("extern \"C\" const unsigned long long luisa_compute_cudadevrt_size = " .. tostring(#data) .. "ull;\n")
        fout:close()
    end

    local need_regen = (not os.isfile(gen_cpp)) or (os.mtime(gen_cpp) < os.mtime(cudadevrt))
    if need_regen then
        _generate_bin2c(cudadevrt, gen_cpp)
    end

    cuda_backend:add("defines", "LUISA_COMPUTE_ENABLE_CUDADEVRT=1")
    cuda_backend:add("files", gen_cpp)
end)
