set(GOICP_INC_FILES
	include/prereq.h
	include/goicp.h
	)

set(GOICP_SRC_FILES
	src/goicp.cpp
	src/matrix.cpp
	src/jly_3ddt.cpp
	src/jly_goicp.cpp
	)
	
find_package(PCL 1.3 REQUIRED COMPONENTS common io features)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})

include_directories("${GOICP_DIR}/include")
add_library(GoICP SHARED ${GOICP_INC_FILES} ${GOICP_SRC_FILES})
target_link_libraries(GoICP PUBLIC ${PCL_COMMON_LIBRARIES} ${PCL_IO_LIBRARIES} ${PCL_FEATURES_LIBRARIES})
add_definitions(-DGOICP_EXPORT)
add_definitions(-D_CRT_SECURE_NO_WARNINGS)