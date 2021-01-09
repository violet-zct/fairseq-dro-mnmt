import numpy as np
import os
import sys

opt_dir = "/home/chuntinz/tir5/data/mnmt_data/opus10"
langs = ["yi", "mr", "oc", "be", "ta", "ka", "gl", "ur", "bg", "is"]
input_dir = "/home/chuntinz/tir5/data/opus-100-corpus/v1.0/supervised"

for lang in langs:
    input_lang_dir = os.path.join(input_dir, "en-{}".format(lang))
    reverse = False
    if not os.path.exists(input_lang_dir):
        input_lang_dir = os.path.join(input_dir, "{}-en".format(lang))
        reverse = True
    os.rmdir(os.path.join(opt_dir, "raw", "{}_en".format(lang)))
    lang_dir = os.path.join(opt_dir, "raw", "{}_en".format(lang))
    os.mkdir(lang_dir)
    for split in ["train", "dev", "test"]:
        for ll in [lang, "en"]:
            if reverse:
                src_file = os.path.join(input_lang_dir, "opus.{}-en-{}.{}".format(lang, split, ll))
            else:
                src_file = os.path.join(input_lang_dir, "opus.en-{}-{}.{}".format(lang, split, ll))
            rename_split = split if split != "dev" else "valid"
            tgt_file = os.path.join(lang_dir, "{}.{}".format(rename_split, ll))
            os.system("cp {} {}".format(src_file, tgt_file))