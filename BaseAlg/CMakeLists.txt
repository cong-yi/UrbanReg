set(BASEALG_INC_FILES
	include/prereq.h
	include/base_alg.h
	)

set(BASEALG_SRC_FILES
	src/base_alg.cpp
	)

find_package(PCL 1.3 REQUIRED COMPONENTS common io features registration)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})

set(SPECTRA_DIR ${URBANREG_EXTERNAL}/spectra)

include_directories("${BASEALG_DIR}/include" "${SPECTRA_DIR}/include")
add_library(BaseAlg SHARED ${BASEALG_INC_FILES} ${BASEALG_SRC_FILES})
target_link_libraries(BaseAlg PUBLIC ${PCL_COMMON_LIBRARIES} ${PCL_IO_LIBRARIES} ${PCL_FEATURES_LIBRARIES} ${PCL_REGISTRATION_LIBRARIES})
add_definitions(-DBASEALG_EXPORT)