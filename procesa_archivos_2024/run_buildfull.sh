#!/bin/bash

if [ $# -ne 2 ]; then
	echo "$(basename $0) <EXECUTABLE> <output_file>"
	exit 1
fi

DATAFILE="m30_2024_data.txt"
GROUPSFILE="m30_2024_ids_estructura.txt"
N="1 2 3 4"
K="5 10 20 30 40"
EXEC=$1
OUTFILE=$2

echo "Fecha INICIO: $(date +'%Y/%m/%d %H:%M:%S')" > ${OUTFILE}

for n in ${N}
do
	for k in ${K}
	do
		echo "=======================================================================" >> ${OUTFILE}
		echo "./${EXEC} ${DATAFILE} ${GROUPSFILE} ${n} ${k} borrar.txt" >> ${OUTFILE}
		./${EXEC} ${DATAFILE} ${GROUPSFILE} ${n} ${k} borrar.txt >> ${OUTFILE}
		echo "=======================================================================" >> ${OUTFILE}
	done
done

echo "Fecha FIN: $(date +'%Y/%m/%d %H:%M:%S')" >> ${OUTFILE}


