if(FLANN_FOUND)
	include_directories(${FLANN_INCLUDE_DIRS})
else()
	message(STATUS "NOTICE: FLANN is not found.") 
endif()

set(FEATUREALG_INC_FILES
	include/prereq.h
	include/feature_alg.h
	)

set(FEATUREALG_SRC_FILES
	src/feature_alg.cpp
	)
	
find_package(PCL 1.3 REQUIRED COMPONENTS common io features registration)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})

include_directories("${FEATUREALG_DIR}/include")
add_library(FeatureAlg SHARED ${FEATUREALG_INC_FILES} ${FEATUREALG_SRC_FILES})
target_link_libraries(FeatureAlg PUBLIC ${PCL_COMMON_LIBRARIES} ${PCL_IO_LIBRARIES} ${PCL_FEATURES_LIBRARIES} ${PCL_REGISTRATION_LIBRARIES} ${FLANN_LIBRARIES})
add_definitions(-DFEATUREALG_EXPORT)
add_definitions(-D_CRT_SECURE_NO_WARNINGS)