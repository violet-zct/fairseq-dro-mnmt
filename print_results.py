import os
import sys

dirname = sys.argv[1]

lang_list_dir = "/home/chuntinz/tir5/data/mnmt_data/ted/lang_lists"

if "related" in dirname:
    lang_file = os.path.join(lang_list_dir, "8re.langs.list")
elif "diverse" in dirname:
    lang_file = os.path.join(lang_list_dir, "8di.langs.list")
elif "ted" in dirname and "all":
    lang_file = os.path.join(lang_list_dir, "all.langs.list")
else:
    raise ValueError

langs = []
for lang in open(lang_file).readlines():
    lang = lang.strip()
    if lang == "en":
        continue
    else:
        langs.append(lang)

results = ["-1"] * len(langs)
for model in os.listdir(dirname):
    if model.startswith("test"):
        field = model.strip().split("_")
        if len(field) == 3:
            lang = field[1]
        else:
            lang = "_".join(field[1:3])
        idx = langs.index(lang)

        bleu = open(os.path.join(dirname, model)).readlines()[-1].strip()
        print(lang)
        print(bleu)

        bleu = bleu.split(":")[-1].split("(")[0].split("=")[-1].split()[0]
        results[idx] = bleu.rstrip(",")

print(" ".join(langs) + "\n")
print(" ".join(results))
