set(GPS_INC_FILES
	include/prereq.h
	include/gps.h
	)

set(GPS_SRC_FILES
	src/gps.cpp
	)

include_directories("${GPSLOC_DIR}/include")
add_library(GPS SHARED ${GPS_INC_FILES} ${GPS_SRC_FILES})
add_definitions(-DGPS_EXPORT)