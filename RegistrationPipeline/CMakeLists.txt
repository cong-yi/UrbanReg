set(REGPIPELINE_INC_FILES
	include/prereq.h
	include/registration_pipeline.h
	)

set(REGPIPELINE_SRC_FILES
	src/registration_pipeline.cpp
	)

include_directories("${REGPIPELINE_DIR}/include;${PROJECT_SOURCE_DIR}")
add_library(RegistrationPipeline SHARED ${REGPIPELINE_INC_FILES} ${REGPIPELINE_SRC_FILES})
target_link_libraries(RegistrationPipeline PUBLIC SSICP GPS OverlapTrimmer DataIO GoICP BaseAlg FastGlobalRegistration FeatureAlg)
add_definitions(-DREGPIPELINE_EXPORT)
file( WRITE "${CMAKE_CURRENT_BINARY_DIR}/RegistrationPipeline.vcxproj.user" 
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