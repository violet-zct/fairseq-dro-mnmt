import os
import gcld3
import sys

# filter data by language ids
dirname = sys.argv[1]
prefix = sys.argv[2]

filenames = os.listdir(dirname)
lang = None
for fname in filenames:
    if fname.startswith(prefix):
        lang = fname.split('.')[-2].split('-')[-1]
if lang is None:
    print("Incorrect folder without train files!")
    exit(0)

detector = gcld3.NNetLanguageIdentifier(min_num_bytes=0,
                                        max_num_bytes=1000)


def read_pair_data(p1, p2):
    data = []
    with open(p1, "r", encoding="utf-8") as f1, open(p2, "r", encoding="utf-8") as f2:
        for l1, l2 in zip(f1, f2):
            data.append((l1.strip(), l2.strip()))
    return data


data = read_pair_data(os.path.join(dirname, "{}.en-{}.{}".format(prefix, lang, lang)),
                      os.path.join(dirname, "{}.en-{}.en".format(prefix, lang)))


def check_overlap(s1, s2):
    words1 = set(s1.strip().split())
    words2 = set(s2.strip().split())

    u = words1.union(words2)
    x = words1.intersection(words2)

    if len(x) * 1.0 / len(u) > 0.6:
        return True
    return False

i = 0
keep = 0

with open(os.path.join(dirname, "{}.en-{}.{}.cleanlang".format(prefix, lang, lang)), "w", encoding="utf-8") as fxx, \
        open(os.path.join(dirname, "{}.en-{}.cleanlang".format(prefix, lang)), "w", encoding="utf-8") as fen:
    for xx, en in data:
        if check_overlap(xx, en):
            continue

        xlang = detector.FindLanguage(text=xx).language
        elang = detector.FindLanguage(text=en).language
        print(xlang, elang)
        if xlang == lang and elang == "en":
            fxx.write(xx + "\n")
            fen.write(en + "\n")
            keep += 1

        i += 1
        if i % 100000 == 0:
            print("processed {}, keep {}".format(i, keep))
print("all = {}, keep = {}".format(i, keep))