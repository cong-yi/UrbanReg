if(NOT TARGET glad)
	add_subdirectory(${URBANREG_EXTERNAL}/glad glad)
	set_target_properties(glad PROPERTIES FOLDER ${VS_EXTERNAL})
	set_target_properties(glad-generate-files PROPERTIES FOLDER ${VS_EXTERNAL})
endif()

if(NOT TARGET glfw)
	add_subdirectory(${URBANREG_EXTERNAL}/glfw glfw)
endif()

set(VIEWEREXAMPLE_SRC_FILES
    src/main.cpp)
include_directories("${PROJECT_SOURCE_DIR}")
add_executable(ViewerExample ${VIEWEREXAMPLE_SRC_FILES})
target_link_libraries(ViewerExample PUBLIC DataIO glfw glad)
add_definitions(-DNOMINMAX)
file( WRITE "${CMAKE_CURRENT_BINARY_DIR}/ViewerExample.vcxproj.user" 
    "<?xml version=\"1.0\" encoding=\"utf-8\"?>     \
    <Project ToolsVersion=\"4.0\" xmlns=\"http://schemas.microsoft.com/developer/msbuild/2003\">
    <PropertyGroup Condition=\"'$(Configuration)|$(Platform)'=='Debug|x64'\">
        <LocalDebuggerWorkingDirectory>$(OutDir)</LocalDebuggerWorkingDirectory>
        <DebuggerFlavor>WindowsLocalDebugger</DebuggerFlavor>
    </PropertyGroup>
    <PropertyGroup Condition=\"'$(Configuration)|$(Platform)'=='Release|x64'\">
        <LocalDebuggerWorkingDirectory>$(OutDir)</LocalDebuggerWorkingDirectory>
        <DebuggerFlavor>WindowsLocalDebugger</DebuggerFlavor>
    </PropertyGroup>
    </Project>")