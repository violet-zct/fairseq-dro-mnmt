#!/bin/bash

SCRIPTS=/jet/home/chuntinz/work/data/wmt4/mosesdecoder/scripts
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl

#CLEAN=$SCRIPTS/training/clean-corpus-n.perl  # clean corpus by min/max lengths and ratios; used after bpe
input_file=$1
lang=$2
cat ${input_file} | perl ${NORM_PUNC} ${lang} | perl ${REM_NON_PRINT_CHAR} >> ${input_file}.clean
