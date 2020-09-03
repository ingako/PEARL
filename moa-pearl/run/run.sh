#!/bin/bash

BASEDIR=`dirname $0`/..
BASEDIR=`(cd "$BASEDIR"; pwd)`
MEMORY=20024m
DATADIR=/home/hwu344/pearl/data

> run.log

dataset_names=(covtype pokerhand elec weatherAUS airlines)
kappa_vals=(0.4 0.0 0.7 0.1 0.3)
ed_vals=(90 90 120 120 90)

for i in ${!dataset_names[@]} ; do
    dataset=${dataset_names[$i]}
	input=${DATADIR}/${dataset}/${dataset}.arff

	vanilla_arf_output=${dataset}/result-arf.csv
	ht_output=${dataset}/result-ecpf-ht.csv
	arf_output=${dataset}/result-ecpf-arf.csv
	diversity_pool_output=${dataset}/result-diversity-pool.csv

    mkdir -p $dataset

	echo "Running ARF on ${dataset}..." >> run.log
    > $vanilla_arf_output
	nohup java -Xmx$MEMORY -cp "$BASEDIR/lib/moa-2019.05.0:$BASEDIR/lib/*" -javaagent:$BASEDIR/lib/sizeofag-1.0.4.jar moa.DoTask \
    "EvaluatePrequential -l (meta.AdaptiveRandomForest -s 60) -s (ArffFileStream -f $input) -f 1000 -q 100000000 -d $vanilla_arf_output" &


	echo "Running Diversity Pool  on ${dataset}..." >> run.log
    > $diversity_pool_output
	nohup java -Xmx$MEMORY -cp "$BASEDIR/lib/moa-2019.05.0:$BASEDIR/lib/*" -javaagent:$BASEDIR/lib/sizeofag-1.0.4.jar moa.DoTask \
    "EvaluatePrequential -l (meta.DiversityPool -s 60) -s (ArffFileStream -f $input) -f 1000 -q 100000000 -d $diversity_pool_output" &


	echo "Running ECPF with hoeffding tree on ${dataset}..." >> run.log
    > $ht_output
	nohup java -Xmx$MEMORY -cp "$BASEDIR/lib/moa-2019.05.0:$BASEDIR/lib/*" -javaagent:$BASEDIR/lib/sizeofag-1.0.4.jar moa.DoTask \
	"EvaluatePrequential -l (meta.ECPF -l trees.HoeffdingTree) -s (ArffFileStream -f $input) -f 1000 -q 100000000 -d $ht_output" &
	
	echo "Running ECPF with ARF on ${dataset}..." >> run.log
    > $arf_output
	nohup java -Xmx$MEMORY -cp "$BASEDIR/lib/moa-2019.05.0:$BASEDIR/lib/*" -javaagent:$BASEDIR/lib/sizeofag-1.0.4.jar moa.DoTask \
    "EvaluatePrequential -l (meta.ECPF -l (meta.AdaptiveRandomForest -s 60)) -s (ArffFileStream -f $input) -f 1000 -q 100000000 -d $arf_output" &


    cd_kappa=${kappa_vals[$i]}
    edit_distance=${ed_vals[$i]}

    lru_queue_size=10000000
    performance_eval_window_size=50
    enable_state_graph=-g
    lossy_window_size=10000000
    candidate_tree_reuse_rate=0.5
    reuse_window_size=10000

    poarf_output_path="./$dataset/$lru_queue_size/$performance_eval_window_size/$cd_kappa/$edit_distance/"
    pearl_output_path="$poarf_output_path/$lossy_window_size/$reuse_window_size/$candidate_tree_reuse_rate/"
    poarf_output="$poarf_output_path/result.csv"
    pearl_output="$pearl_output_path/result.csv"
    mkdir -p $poarf_output_path
    mkdir -p $pearl_output_path

	echo "Running PEARL with pattern matching only  on ${dataset}..." >> run.log
    > $poarf_output
	nohup java -Xmx$MEMORY -cp "$BASEDIR/lib/moa-2019.05.0:$BASEDIR/lib/*" -javaagent:$BASEDIR/lib/sizeofag-1.0.4.jar moa.DoTask \
    "EvaluatePrequential -l (meta.PEARL -s 60 -k $cd_kappa -e $edit_distance -f $lru_queue_size -z $performance_eval_window_size) \
    -s (ArffFileStream -f $input) -f 1000 -q 100000000 -d $poarf_output" &

	echo "Running PEARL on ${dataset}..." >> run.log

    > $parf_output
	java -Xmx$MEMORY -cp "$BASEDIR/lib/moa-2019.05.0:$BASEDIR/lib/*" -javaagent:$BASEDIR/lib/sizeofag-1.0.4.jar moa.DoTask \
    "EvaluatePrequential -l (meta.PEARL -k $cd_kappa -e $edit_distance -z $performance_eval_window_size \
    $enable_state_graph -d $lossy_window_size -v $candidate_tree_reuse_rate -y $reuse_window_size) \
    -s (ArffFileStream -f $input) -f 1000 -q 100000000 -d $poarf_output"

done
