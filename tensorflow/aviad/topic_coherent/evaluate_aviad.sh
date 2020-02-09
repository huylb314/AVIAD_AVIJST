#!/bin/bash

# Setup Directories
# =================
dataset_used=(ursa)
labels_used=(3)
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
			labels_used=${labels_used[idx]}
			for i in ${run_time[@]}; do
				python ComputeOcTopwords.py --topic_folder ../results/$dataset \
				--topic_processed ../results/$dataset \
				--wc_folder wc/$model/$dataset/run$i --ref_corpus_dir ../corpus/$dataset \
				--oc_folder oc/$model/$dataset/run$i --number_topic 3 --number_label $labels_used
			done;
		done;
	done;
fi