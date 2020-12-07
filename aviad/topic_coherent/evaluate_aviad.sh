#!/bin/bash

while getopts f:c:n:t: flag
do
    case "${flag}" in
        f) folder=${OPTARG};;
        c) corpus=${OPTARG};;
		n) topics=${OPTARG};;
		t) top=${OPTARG};;
    esac
done
echo "folder: $folder";
echo "corpus: $corpus";
echo "topics: $topics";
echo "top: $top";

# Setup Directories
# =================
dataset_used=(temp)
run_time=(1)
model_used=(aviad)

# Run Topic Coherent
# ===============
if [[ ! -d ./wc ]]; then
	mkdir wc;
	for model_idx in ${!model_used[@]}; do
		model=${model_used[model_idx]}
		mkdir wc/$model
		for idx in ${!dataset_used[@]}; do
			dataset=${dataset_used[idx]}
			mkdir wc/$model/$dataset;
			for i in ${run_time[@]}; do
				mkdir wc/$model/$dataset/run$i;
			done;
		done;
	done;
fi

if [[ ! -d ./oc ]]; then
	mkdir oc;
	for model_idx in ${!model_used[@]}; do
		model=${model_used[model_idx]}
		mkdir oc/$model
		for idx in ${!dataset_used[@]}; do
			dataset=${dataset_used[idx]}
			mkdir oc/$model/$dataset;
			for i in ${run_time[@]}; do
				mkdir oc/$model/$dataset/run$i;
			done;
		done;
	done;
fi

if [[ -d ./wc ]]; then
	for model_idx in ${!model_used[@]}; do
		model=${model_used[model_idx]}
		for idx in ${!dataset_used[@]}; do
			dataset=${dataset_used[idx]}
			for i in ${run_time[@]}; do
				python compute_oc_topwords.py --topns $top --metric npmi --topic_folder $folder \
				--wc_folder wc/$model/$dataset/run$i --ref_corpus_dir $corpus \
				--oc_folder oc/$model/$dataset/run$i --number_topic $topics
			done;
		done;
	done;
fi