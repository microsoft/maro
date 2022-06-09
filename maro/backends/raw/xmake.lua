add_requires("gtest")

target("test")
	set_kind("binary")
	add_files("test/*.cpp")
    add_files("./*.cpp")
	add_packages("gtest")
