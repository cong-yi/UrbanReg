set(REGEXAMPLE_SRC_FILES
    src/main.cpp)
include_directories("${PROJECT_SOURCE_DIR}")

add_executable(RegExample ${REGEXAMPLE_SRC_FILES})
target_link_libraries(RegExample PUBLIC SSICP GPS OverlapTrimmer RegistrationPipeline DataIO)
file( WRITE "${CMAKE_CURRENT_BINARY_DIR}/RegExample.vcxproj.user" 
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