local function yutrel_pbrt_test(name)
    target("test_Yutrel_" .. name)
        set_kind("binary")
        set_default(false)
        set_group("tests/Yutrel/pbrt")
        set_rundir("$(projectdir)")

        add_files("test_" .. name .. ".cpp")
        add_includedirs("$(projectdir)/ext/LuisaCompute/src/tests")
        add_deps("YutrelPbrt")
    target_end()
end

yutrel_pbrt_test("pbrt_parse")
yutrel_pbrt_test("pbrt_import")
yutrel_pbrt_test("environment")
yutrel_pbrt_test("geometry")
