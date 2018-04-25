if(FLANN_FOUND)
	include_directories(${FLANN_INCLUDE_DIRS})
else()
	message(STATUS "NOTICE: FLANN is not found.") 
endif()

set(OVERLAPTRIMMER_INC_FILES
	include/prereq.h
	include/trimmer.h
	)

set(OVERLAPTRIMMER_SRC_FILES
	src/trimmer.cpp
	)

include_directories("${OVERLAPTRIMMER_DIR}/include")
add_library(OverlapTrimmer SHARED ${OVERLAPTRIMMER_INC_FILES} ${OVERLAPTRIMMER_SRC_FILES})
target_link_libraries(OverlapTrimmer ${FLANN_LIBRARIES})
add_definitions(-DOVERLAPTRIMMER_EXPORT)