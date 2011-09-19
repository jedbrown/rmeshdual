include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/variables

rmeshdual : rmeshdual.o
	${CLINKER} -o $@ $< ${PETSC_MAT_LIB}

