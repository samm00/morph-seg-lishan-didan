from segmentation import *
from preprocessing import *
import os
from analysis import *
import pandas as pd

def run():
   # Read the initial lexicon (word list), and convert it into Hex.
   words, encoded_words, hex_chars = process_words('data/lishan-didan/data/lishan-didan.end')
   write_encoded_words(encoded_words, 'data/lishan-didan/pycfg/traindev.txt')

   # Read the initial CFG and append the HEX encoded characters as terminals.
   # encoded_lexicon_path and final_grammar_path then become the input to the PYAGS sampler.
   grammar = read_grammar('grammar.txt')
   appended_grammar = add_chars_to_grammar(grammar, hex_chars)
   write_grammar(appended_grammar, 'data/lishan-didan/pycfg/traindev.cfg1.txt')

   # Seed affixes into the grammar, where the affixes are read from the LK file.
   ss_grammar = prepare_scholar_seeded_grammar(grammar, 'data/lishan-didan/data/lk.txt', 'Prefix', 'Suffix', 1, 1)
   write_grammar(ss_grammar, 'data/lishan-didan/pycfg/traindev.cfg.ss.txt')
   # Append the Hex encoded characters as terminals.
   # encoded_lexicon_path and final_grammar_path then become the input to the PYAGS sampler.
   appended_ss_grammar = add_chars_to_grammar(ss_grammar, hex_chars)
   write_grammar(appended_ss_grammar, 'data/lishan-didan/pycfg/traindev.cfg.ss1.txt')


   input = 'data/lishan-didan/pycfg/traindev.txt'
   grammar = 'data/lishan-didan/pycfg/traindev.cfg1.txt'
   iters = 100
   output = 'data/lishan-didan/pycfg/output/output-traindev.cfg1-s.txt'
   trace_out = 'data/lishan-didan/pycfg/output/trace-traindev.cfg1-s.txt'
   grammar_out = 'data/lishan-didan/pycfg/output/grammar-traindev.cfg1-s.txt'
   os.system(f'cat {input} | data/lishan-didan/pycfg/py-cfg -A {output} -r 0 -d 10 -x 10 -F {trace_out} -G {grammar_out} -D -E -e 1 -f 1 -g 10 -h 0.1 -w 1 -T 1 -m 0 -n {iters} -R -1 {grammar}')

   eval_out = 'data/lishan-didan/pycfg/output/eval_output-traindev.cfg1-s.txt'
   segmentation_model_st = parse_segmentation_output(output, 'Prefix', 'Stem', 'Suffix', eval_out, 'lishan-didan', 3)


   grammar = read_grammar('grammar.txt')
   cascaded_grammar = prepare_cascaded_grammar(grammar, output, 100, 'Prefix', 'Suffix', 'Prefix', 'Suffix', 1, 1)
   write_grammar(cascaded_grammar, 'data/lishan-didan/pycfg/traindev.cfg.c.txt')
   # Append the Hex encoded characters as terminals.
   # encoded_lexicon_path and final_grammar_path then become the input to the PYAGS sampler.
   appended_cascaded_grammar = add_chars_to_grammar(cascaded_grammar, hex_chars)
   write_grammar(appended_cascaded_grammar, 'data/lishan-didan/pycfg/traindev.cfg.c1.txt')


   grammar = 'data/lishan-didan/pycfg/traindev.cfg.c1.txt'
   output = 'data/lishan-didan/pycfg/output/output-traindev.cfg.c1-s.txt'
   trace_out = 'data/lishan-didan/pycfg/output/trace-traindev.cfg.c1-s.txt'
   grammar_out = 'data/lishan-didan/pycfg/output/grammar-traindev.cfg.c1-s.txt'
   os.system(f'cat {input} | data/lishan-didan/pycfg/py-cfg -A {output} -r 0 -d 10 -x 10 -F {trace_out} -G {grammar_out} -D -E -e 1 -f 1 -g 10 -h 0.1 -w 1 -T 1 -m 0 -n {iters} -R -1 {grammar}')

   eval_out = 'data/lishan-didan/pycfg/output/eval_output-traindev.cfg.c1-s.txt'
   segmentation_model_cst = parse_segmentation_output(output, 'Prefix', 'Stem', 'Suffix', eval_out, 'lishan-didan', 3)


   grammar = 'data/lishan-didan/pycfg/traindev.cfg.ss1.txt'
   output = 'data/lishan-didan/pycfg/output/output-traindev.cfg.ss1-s.txt'
   trace_out = 'data/lishan-didan/pycfg/output/trace-traindev.cfg.ss1-s.txt'
   grammar_out = 'data/lishan-didan/pycfg/output/grammar-traindev.cfg.ss1-s.txt'
   os.system(f'cat {input} | data/lishan-didan/pycfg/py-cfg -A {output} -r 0 -d 10 -x 10 -F {trace_out} -G {grammar_out} -D -E -e 1 -f 1 -g 10 -h 0.1 -w 1 -T 1 -m 0 -n {iters} -R -1 {grammar}')

   eval_out = 'data/lishan-didan/pycfg/output/eval_output-traindev.cfg.ss1-s.txt'
   segmentation_model_ss = parse_segmentation_output(output, 'Prefix', 'Stem', 'Suffix', eval_out, 'lishan-didan', 3)


   grammar = read_grammar('grammar.txt')
   cascaded_grammar_ss = prepare_cascaded_grammar(grammar, output, 100, 'Prefix', 'Suffix', 'Prefix', 'Suffix', 1, 1)
   write_grammar(cascaded_grammar_ss, 'data/lishan-didan/pycfg/traindev.cfg.css.txt')
   # Append the Hex encoded characters as terminals.
   # encoded_lexicon_path and final_grammar_path then become the input to the PYAGS sampler.
   appended_cascaded_grammar_ss = add_chars_to_grammar(cascaded_grammar_ss, hex_chars)
   write_grammar(appended_cascaded_grammar_ss, 'data/lishan-didan/pycfg/traindev.cfg.css1.txt')

   grammar = 'data/lishan-didan/pycfg/traindev.cfg.css1.txt'
   output = 'data/lishan-didan/pycfg/output/output-traindev.cfg.css1-s.txt'
   trace_out = 'data/lishan-didan/pycfg/output/trace-traindev.cfg.css1-s.txt'
   grammar_out = 'data/lishan-didan/pycfg/output/grammar-traindev.cfg.css1-s.txt'
   os.system(f'cat {input} | data/lishan-didan/pycfg/py-cfg -A {output} -r 0 -d 10 -x 10 -F {trace_out} -G {grammar_out} -D -E -e 1 -f 1 -g 10 -h 0.1 -w 1 -T 1 -m 0 -n {iters} -R -1 {grammar}')

   eval_out = 'data/lishan-didan/pycfg/output/eval_output-traindev.cfg.css1-s.txt'
   segmentation_model_css = parse_segmentation_output(output, 'Prefix', 'Stem', 'Suffix', eval_out, 'lishan-didan', 3)

   # Segment a white-space tokenized text file.
   # When both <split_marker> and <stem_marker> parameters are set to None, the function does only stemming.
   segment_file('data/lishan-didan/data/lishan-didan.1000', 'dev_output_st.txt', segmentation_model_st, '-', ' ', False, 'lishan-didan' , 3)
   segment_file('data/lishan-didan/data/lishan-didan.1000', 'dev_output_ss.txt', segmentation_model_ss, '-', ' ', False, 'lishan-didan' , 3)
   segment_file('data/lishan-didan/data/lishan-didan.1000', 'dev_output_cst.txt', segmentation_model_cst, '-', ' ', False, 'lishan-didan' , 3)
   segment_file('data/lishan-didan/data/lishan-didan.1000', 'dev_output_css.txt', segmentation_model_css, '-', ' ', False, 'lishan-didan' , 3)


   segs_st = []
   with open('dev_output_st.txt') as f:
      segs_st = f.read().split('\n')[:-1]

   segs_ss = []
   with open('dev_output_ss.txt') as f:
      segs_ss = f.read().split('\n')[:-1]

   segs_cst = []
   with open('dev_output_cst.txt') as f:
      segs_cst = f.read().split('\n')[:-1]

   segs_css = []
   with open('dev_output_css.txt') as f:
      segs_css = f.read().split('\n')[:-1]

   with open('data/lishan-didan/data/lishan-didan.1000') as f:
      dev = f.read().split('\n')[:-1]

   #dev = words.iloc[750:]['Word']

   with open('standard_dev_output.txt', 'w') as f:
      comb = [word + '\t' + seg for word, seg in zip(list(dev), segs_st)]
      f.write('\n'.join(list(comb)))

   with open('scholarly_dev_output.txt', 'w') as f:
      comb = [word + '\t' + seg for word, seg in zip(list(dev), segs_ss)]
      f.write('\n'.join(list(comb)))

   with open('casc_standard_dev_output.txt', 'w') as f:
      comb = [word + '\t' + seg for word, seg in zip(list(dev), segs_cst)]
      f.write('\n'.join(list(comb)))

   with open('casc_scholarly_dev_output.txt', 'w') as f:
      comb = [word + '\t' + seg for word, seg in zip(list(dev), segs_css)]
      f.write('\n'.join(list(comb)))

   morph_info = analyze_output('standard_dev_output.txt', 'data/lishan-didan/data/lishan-didan.1000.gold')
   df_st = pd.DataFrame(morph_info)

   morph_info = analyze_output('scholarly_dev_output.txt', 'data/lishan-didan/data/lishan-didan.1000.gold')
   df_ss = pd.DataFrame(morph_info)

   morph_info = analyze_output('casc_standard_dev_output.txt', 'data/lishan-didan/data/lishan-didan.1000.gold')
   df_cst = pd.DataFrame(morph_info)

   morph_info = analyze_output('casc_scholarly_dev_output.txt', 'data/lishan-didan/data/lishan-didan.1000.gold')
   df_css = pd.DataFrame(morph_info)

   f1, pre, rec = [], [], []
   for df in [df_st, df_ss, df_cst, df_css]:
      pre.append(df.iloc[3].mean())
      rec.append(df.iloc[4].mean())
      f1.append(df.iloc[5].mean())

   return f1, pre, rec


f1s, pres, recs = [], [], []
for i in range(5):
   f1, pre, rec = run()
   print(f'precision: {pre}\t\trecall: {rec}\t\tf1: {f1}')

   f1s.append(f1)
   pres.append(pre)
   recs.append(rec)

all_f1, all_pre, all_rec = pd.DataFrame(f1s), pd.DataFrame(pres), pd.DataFrame(recs)

st_f1 = all_f1[0].mean()
ss_f1 = all_f1[1].mean()
cst_f1 = all_f1[2].mean()
css_f1 = all_f1[3].mean()
st_pre = all_pre[0].mean()
ss_pre = all_pre[1].mean()
cst_pre = all_pre[2].mean()
css_pre = all_pre[3].mean()
st_rec = all_rec[0].mean()
ss_rec = all_rec[1].mean()
cst_rec = all_rec[2].mean()
css_rec = all_rec[3].mean()

print(f'st_f1: {st_f1}\t\tst_precision: {st_pre}\t\tst_recall: {st_rec}')
print(f'ss_f1: {ss_f1}\t\tss_precision: {ss_pre}\t\tss_recall: {ss_rec}')
print(f'cst_f1: {cst_f1}\t\tcst_precision: {cst_pre}\t\tcst_recall: {cst_rec}')
print(f'css_f1: {css_f1}\t\tcss_precision: {css_pre}\t\tcss_recall: {css_rec}')
