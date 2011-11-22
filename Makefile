include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/variables
VORO_DIR = ${HOME}/src/voro/src
VORO_LIB_DIR = ${VORO_DIR}
VORO_INCLUDE_DIR = ${VORO_DIR}
VORO_INCLUDES = -I${VORO_DIR}
VORO_LIBRARIES = -L${VORO_DIR}/lib -Wl,-rpath,${VORO_DIR}/lib -lvoro++

CFLAGS += ${VORO_INCLUDES}

rmeshdual : rmeshdual.o
	${CLINKER} -o $@ $< ${PETSC_MAT_LIB}

vorotest : vorotest.o
	${CLINKER} -o $@ $< -lvoro++ -lstdc++

