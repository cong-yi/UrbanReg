set(REGPIPELINE_INC_FILES
	include/prereq.h
	include/registration_pipeline.h
	)

set(REGPIPELINE_SRC_FILES
	src/registration_pipeline.cpp
	)

include_directories("${REGPIPELINE_DIR}/include;${PROJECT_SOURCE_DIR}")
add_library(RegistrationPipeline SHARED ${REGPIPELINE_INC_FILES} ${REGPIPELINE_SRC_FILES})
target_link_libraries(RegistrationPipeline PUBLIC SSICP GPS OverlapTrimmer DataIO)
add_definitions(-DREGPIPELINE_EXPORT)