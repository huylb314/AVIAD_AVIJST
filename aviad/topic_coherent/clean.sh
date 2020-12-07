#!/bin/bash
# Setup Directories
# =================

if [[ -d results ]]; then
	rm -rf results;
fi

if [[ -d predictions ]]; then
	rm -rf predictions;
fi

if [[ -d models ]]; then
	rm -rf models;
fi

if [[ -d ref_corpus ]]; then
	rm -rf ref_corpus;
fi

# Run Topic Coherent
# ===============
if [[ -d topic_processed ]]; then
	rm -rf topic_processed;
fi

if [[ -d wc ]]; then
	rm -rf wc;
fi

if [[ -d oc ]]; then
	rm -rf oc;
fi

if [[ -f final_log.txt ]]; then
	rm final_log.txt;
fi