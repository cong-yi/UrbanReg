set(GPS_INC_FILES
	include/prereq.h
	include/gps.h
	)

set(GPS_SRC_FILES
	src/gps.cpp
	)

include_directories("${GPSLOC_DIR}/include;${PROJECT_SOURCE_DIR}")
add_library(GPS SHARED ${GPS_INC_FILES} ${GPS_SRC_FILES})
target_link_libraries(GPS PUBLIC SSICP DataIO)
add_definitions(-DGPS_EXPORT)