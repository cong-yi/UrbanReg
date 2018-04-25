set(DATAIO_INC_FILES
	include/prereq.h
	include/data_io.h
	)

set(DATAIO_SRC_FILES
	src/data_io.cpp
	)

include_directories("${DATAIO_DIR}/include")
if(WITH_RPLY)
	set(RPLY_DIR ${URBANREG_EXTERNAL}/librply/src/lib)
	add_library(librply STATIC ${RPLY_DIR}/rply.h ${RPLY_DIR}/rplyfile.h ${RPLY_DIR}/rply.c)
else()
	message(FATAL_ERROR "Please git recursively to download librply!")
endif()
include_directories("${URBANREG_EXTERNAL}/librply/src/lib")
add_library(DataIO SHARED ${DATAIO_INC_FILES} ${DATAIO_SRC_FILES})
target_link_libraries(DataIO PUBLIC librply)
add_definitions(-DDATAIO_EXPORT)