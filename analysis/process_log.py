import os
import sys
import numpy as np

folder = sys.argv[1]
flog = os.path.join(folder, "log.txt")
ffout = os.path.join(folder, "processed.log.txt")

lang_pairs = "en-yi,en-mr,en-oc,en-be,en-ta,en-ka,en-gl,en-ur,en-bg,en-is"
src_langs = sorted(list(set([langpair.split("-")[0] for langpair in lang_pairs.split(",")])))
tgt_langs = sorted(list(set([langpair.split("-")[1] for langpair in lang_pairs.split(",")])))

src_lang_dict = {lang:i for i, lang in enumerate(src_langs)}
tgt_lang_dict = {lang:i for i, lang in enumerate(tgt_langs)}

final_lang_dict = tgt_lang_dict

with open(flog) as fin, open(ffout, "w") as fout:
    for line in fin:
        if "fairseq.criterions.upper_bounded_alpha_cover_sample_dro | Group loss weights:" in line:
            fields = line.strip().split("weights:")
            normalized = np.array([float(v) for v in fields[-1].strip().split()])
            normalized = " ".join([str(v) for v in normalized / sum(normalized)])
            fout.write("weights: ".join([fields[0], normalized]) + "\n")
        elif "fairseq.data.multilingual.sampled_multi_dataset | [train] Resampled sizes: {" in line:
            dicts = line.strip().split("Resampled sizes: {")[-1].split("};")[0]
            sampled_ratios = np.zeros(len(tgt_langs))
            for field in dicts.split(", "):
                lang = field.split(":")[1].split("-")[-1].strip("\'")
                size = int(field.split(":")[-1])
                sampled_ratios[final_lang_dict[lang]] = size
            sampled_ratios = sampled_ratios / sum(sampled_ratios)
            fout.write("Normalized corresponding samplie ratio: " + " ".join([str(s) for s in sampled_ratios]) + "\n")
            fout.write(line)
        else:
            fout.write(line)