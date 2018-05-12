if(FLANN_FOUND)
	include_directories(${FLANN_INCLUDE_DIRS})
else()
	message(STATUS "NOTICE: FLANN is not found.") 
endif()

find_package(PCL 1.3 REQUIRED COMPONENTS common io features registration sample_consensus)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})

set(FGR_INC_FILES
	include/prereq.h
	include/fast_global_registration.h
	)

set(FGR_SRC_FILES
	src/fast_global_registration.cpp
	)

include_directories("${FGR_DIR}/include;${PROJECT_SOURCE_DIR}")
add_library(FastGlobalRegistration SHARED ${FGR_INC_FILES} ${FGR_SRC_FILES})
target_link_libraries(FastGlobalRegistration PUBLIC ${FLANN_LIBRARIES} SSICP ${PCL_SAMPLE_CONSENSUS_LIBRARIES} )
add_definitions(-DFGR_EXPORT)
add_definitions(-D_CRT_SECURE_NO_WARNINGS)