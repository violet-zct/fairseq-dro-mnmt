import os

safe = True
for exp in ["1_analyze_ema0.1_alpha0.5_wu_ub_lang_dro_ted8", "2_analyze_hier_ema0.1_alpha0.5_beta0.5_wu_ub_ted8", "3_sanity_hier_ema0.1_alpha0.5_beta1.0_wu_ub_ted8"]:
    for pp in ["related_o2m", "related_m2o", "diverse_o2m", "diverse_m2o"]:
        dirname = os.path.join("/private/home/ghazvini/chunting/fairseq-dro-mnmt/saved_model", exp + "_" + pp)
        if "related" in pp:
            langs = "aze,bel,glg,slk,tur,rus,por,ces".split(",")
        else:
            langs = "bos,mar,hin,mkd,ell,bul,fra,kor".split(",")

        for lang in langs:
            test_file = os.path.join(dirname, "test_{}_en.log".format(lang))
            if not (os.path.exists(test_file) and os.stat(test_file).st_size != 0):
                safe = False
                break

if safe:
    print("All the test exists, please transfer them via scp exps/scp.sh")
else:
    print("Genetions were not successfully done, please dig into slurm_logs/xx.err (if nothing in the err file, just ignore) and then rerun sbatch exps/test.sh")