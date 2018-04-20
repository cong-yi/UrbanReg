if(FLANN_FOUND)
	include_directories(${FLANN_INCLUDE_DIRS})
else()
	message(STATUS "NOTICE: FLANN is not found.") 
endif()

set(SSICP_INC_FILES
	include/prereq.h
	include/ssicp.h
	)

set(SSICP_SRC_FILES
	src/ssicp.cpp
	)

include_directories("${SSICP_DIR}/include")
add_library(SSICP SHARED ${SSICP_INC_FILES} ${SSICP_SRC_FILES})
target_link_libraries(SSICP ${FLANN_LIBRARIES})
add_definitions(-DSSICP_EXPORT)