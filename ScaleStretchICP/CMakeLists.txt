set(SSICP_INC_FILES
	include/prereq.h
	include/ssicp.h
	)

set(SSICP_SRC_FILES
	src/ssicp.cpp
	)

include_directories("${SSICP_DIR}/include")
add_library(SSICP SHARED ${SSICP_INC_FILES} ${SSICP_SRC_FILES})
add_definitions(-DSSICP_EXPORT)