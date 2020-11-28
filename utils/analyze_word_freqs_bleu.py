import os
import sys
from collections import defaultdict
import numpy as np
import math

option = sys.argv[1]

if option == 0:
    root = "/private/home/chuntinz/work/data/mnmt_data/ted/ted8_related"
    langs = "aze,bel,glg,slk,tur,rus,por,ces"
elif option == 1:
    root = "/private/home/chuntinz/work/data/mnmt_data/ted/ted8_diverse"
    langs = "bos,mar,hin,mkd,ell,bul,fra,kor"
else:
    root = "/private/home/chuntinz/work/data/mnmt_data/ted/ted_all"
    langs = "ara,aze,bel,ben,bos,bul,ces,cmn,dan,deu,ell,epo,est,eus,fas,fin,fra,glg,heb,hin,hrv,hun,hye,ind,ita,jpn,kat,kaz,kor,kur,lit,mar,mkd,mon,msa,mya,nld,nob,pol,por,ron,rus,slk,slv,spa,sqi,srp,swe,tam,tha,tur,ukr,urd,vie,XXfr_ca,XXpt_pt,XXzh,XXzh_tw"


vocab = defaultdict(int)
for lang in langs.split(","):
    with open(os.path.join(root, "data", "{}_en".format(lang), "spm.train.{}".format(lang)), "r", encoding="utf-8") as fin:
        for line in fin:
            for word in line.strip().split():
                vocab[word] += 1


sorted_vocab = sorted(vocab.items(), key=lambda item: item[1])
word2id = {word:idx for idx, (word, freq) in enumerate(sorted_vocab)}
total_size = len(vocab)
# total_vocab_counts = sum([c for w, c in sorted_vocab])
# normalized_vocab = [c*1.0/ total_vocab_counts for w, c in sorted_vocab]
# cumsum = np.cumsum(normalized_vocab)
# assert cumsum[-1] >= 1
bucket_ratios = [0.1, 0.3, 0.5, 0.7, 1]
buckets = [math.ceil(total_size * ii) for ii in bucket_ratios]


# the bleus of different languages are not comparable
print(" ".join([str(bb) for bb in bucket_ratios]))
for lang in langs.split(","):
    lang_bucket = [0] * len(bucket_ratios)
    with open(os.path.join(root, "data", "{}_en".format(lang), "spm.train.{}".format(lang)), "r", encoding="utf-8") as fin:
        for line in fin:
            for word in line.strip().split():
                idx = word2id[word]
                for ii, bucket in enumerate(buckets):
                    if idx <= bucket:
                        lang_bucket[ii] += 1
                        break
    print(lang)