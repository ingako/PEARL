#!/bin/bash

BASEDIR=`dirname $0`/..
BASEDIR=`(cd "$BASEDIR"; pwd)`
MEMORY=10024m

DATADIR=/home/hwu344/pearl/data

echo "Running ECPF..." > ecpf.log

for dataset in elec pokerhand weatherAUS insect insect-imbalanced sensor; do
	input=${DATADIR}/${dataset}/${dataset}.arff

	vanilla_arf_output=${DATADIR}/${dataset}/result-moa-arf.csv
	ht_output=${DATADIR}/${dataset}/result-ecpf-ht.csv
	arf_output=${DATADIR}/${dataset}/result-ecpf-arf.csv

	echo "Running ARF on ${dataset}..." >> ecpf.log

    > $vanilla_arf_output
	java -Xmx$MEMORY -cp "$BASEDIR/lib/moa-2019.05.0:$BASEDIR/lib/*" -javaagent:$BASEDIR/lib/sizeofag-1.0.4.jar moa.DoTask \
    "EvaluatePrequential -l (meta.AdaptiveRandomForest -l (ARFHoeffdingTree -l MC) -s 60 -a 1.0 -w) -s (ArffFileStream -f $input) -f 1000 -d $vanilla_arf_output"

	echo "Running hoeffding tree on ${dataset}..." >> ecpf.log

    > $ht_output
	java -Xmx$MEMORY -cp "$BASEDIR/lib/moa-2019.05.0:$BASEDIR/lib/*" -javaagent:$BASEDIR/lib/sizeofag-1.0.4.jar moa.DoTask \
	"EvaluatePrequential -l (meta.ECPF -l trees.HoeffdingTree) -s (ArffFileStream -f $input) -f 1000 -d $ht_output"
	
	echo "Running random forest on ${dataset}..." >> ecpf.log
	
    > $arf_output
	java -Xmx$MEMORY -cp "$BASEDIR/lib/moa-2019.05.0:$BASEDIR/lib/*" -javaagent:$BASEDIR/lib/sizeofag-1.0.4.jar moa.DoTask \
	"EvaluatePrequential -l (meta.ECPF -l (meta.AdaptiveRandomForest -l (ARFHoeffdingTree -l MC) -s 60 -a 1.0 -w)) -s (ArffFileStream -f $input) -f 1000 -d $arf_output"
done
