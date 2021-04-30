import sys

ipt_file = sys.argv[1]
opt_file = sys.argv[2]

with open(ipt_file, "r", encoding="utf-8") as fin, open(opt_file, "w", encoding="utf-8") as fout:
    for line in fin:
        pieces = line.strip().split()
        detok = ''.join(pieces).replace('▁', ' ')
        fout.write(detok + "\n")