target("YutrelCore")
    set_kind("static")
    add_files("core/**.cpp")
    add_headerfiles("core/**.h")
    add_includedirs("core", {public = true})

    add_deps("lc-dsl", "lc-gui", "stb-image")
    add_packages("tinyexr", "assimp", {public = true})
target_end()

target("YutrelPbrt")
    set_kind("static")
    add_files("importers/pbrt/**.cpp")
    add_headerfiles("importers/pbrt/**.h")
    add_includedirs("importers", {public = true})

    add_deps("YutrelCore", {public = true})
target_end()

target("Yutrel")
    set_kind("binary")
    set_rundir("$(projectdir)/projects/Yutrel")

    add_files("app/**.cpp")
    add_includedirs("app")
    add_deps("YutrelCore", "YutrelPbrt")

target_end()

includes("tests")
