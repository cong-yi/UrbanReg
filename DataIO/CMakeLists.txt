set(DATAIO_INC_FILES
	include/prereq.h
	include/data_io.h
	)

set(DATAIO_SRC_FILES
	src/data_io.cpp
	)

if(WITH_RPLY)
	set(RPLY_DIR ${URBANREG_EXTERNAL}/librply/src/lib)
	add_library(librply STATIC ${RPLY_DIR}/rply.h ${RPLY_DIR}/rplyfile.h ${RPLY_DIR}/rply.c)
	set_target_properties(librply PROPERTIES FOLDER ${VS_EXTERNAL})
else()
	message(FATAL_ERROR "Please git recursively to download librply!")
endif()

add_subdirectory(${URBANREG_EXTERNAL}/jsoncpp jsoncpp)
if(BUILD_STATIC_LIBS)
	set_target_properties(jsoncpp_lib_static PROPERTIES FOLDER ${VS_EXTERNAL})
	set(JSONLIB jsoncpp_lib_static)
else()
	set_target_properties(jsoncpp_lib PROPERTIES FOLDER ${VS_EXTERNAL})
	set(JSONLIB jsoncpp_lib)
endif()

#if(JSONCPP_WITH_TESTS)
#	set_target_properties(jsoncpp_test PROPERTIES FOLDER ${VS_EXTERNAL})
#	set_target_properties(jsoncpp_check PROPERTIES FOLDER ${VS_EXTERNAL})
#	set_target_properties(jsoncpp_readerwriter_tests PROPERTIES FOLDER ${VS_EXTERNAL})
#	set_target_properties(jsontestrunner_exe PROPERTIES FOLDER ${VS_EXTERNAL})
#endif()

include_directories("${DATAIO_DIR}/include")
include_directories("${URBANREG_EXTERNAL}/librply/src/lib")
include_directories("${URBANREG_EXTERNAL}/jsoncpp/include")
include_directories("${URBANREG_EXTERNAL}/rapidxml-1.13")
add_library(DataIO SHARED ${DATAIO_INC_FILES} ${DATAIO_SRC_FILES})
target_link_libraries(DataIO PUBLIC librply ${JSONLIB})
add_definitions(-DDATAIO_EXPORT)