#!/bin/bash

BASEDIR=`dirname $0`/..
BASEDIR=`(cd "$BASEDIR"; pwd)`
MEMORY=10024m

DATADIR=$1

echo "Running ECPF..." > ecpf.log

for dataset in elec covtype pokerhand airlines weatherAUS ; do
	input=${DATADIR}/${dataset}/${dataset}.arff
	ht_output=${DATADIR}/${dataset}/result-ecpf-ht.csv
	arf_output=${DATADIR}/${dataset}/result-ecpf-arf.csv

	echo "Running hoeffding tree on ${dataset}..." >> ecpf.log

	java -Xmx$MEMORY -cp "$BASEDIR/lib/moa-2019.05.0:$BASEDIR/lib/*" -javaagent:$BASEDIR/lib/sizeofag-1.0.4.jar moa.DoTask \
	"EvaluatePrequential -l (meta.ECPF -l trees.HoeffdingTree) -s (ArffFileStream -f $input) -f 1000 -d $ht_output"
	
	
	echo "Running random forest on ${dataset}..." >> ecpf.log
	
	java -Xmx$MEMORY -cp "$BASEDIR/lib/moa-2019.05.0:$BASEDIR/lib/*" -javaagent:$BASEDIR/lib/sizeofag-1.0.4.jar moa.DoTask \
	"EvaluatePrequential -l (meta.ECPF -l (meta.AdaptiveRandomForest -s 60)) -s (ArffFileStream -f $input) -f 1000 -d $arf_output"
done
