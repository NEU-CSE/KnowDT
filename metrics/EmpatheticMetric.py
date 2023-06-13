import numpy as np
from sacrebleu.metrics import BLEU
from rouge import Rouge
from process import process
process(f"results/v1.2.txt")
#
refs_path = './data/refs.txt'
hyps_path = './data/hyps.txt'

refs = []
hyps = []

with open(refs_path, 'r') as refsFile, open(hyps_path, 'r') as hypsFile:
    for ref, hyp in zip(refsFile, hypsFile):
        refs.append(ref.strip('\n'))
        hyps.append(hyp.strip('\n'))

# refs = ['The dog bit the man.', 'It was not unexpected.', 'The man bit him first.']
# hyps = ['The dog bit the man.', "It wasn't surprising.", 'The man had just bitten him.']


#print('标准答案', refs)
#print('待评估句子', hyps)

"""
Bert Score

https://github.com/Tiiiger/bert_score
Bert score will be effected by the version, this version is the version 0.3.12
"""
from bert_score import score
P, R, F1 = score(hyps, refs, lang="en", rescale_with_baseline=True)

print(f"***** Bert_P: {P.mean() * 100} *****")
print(f"***** Bert_R: {R.mean() * 100} *****")
print(f"***** Bert_F1: {F1.mean() * 100} *****")

"""
Distinct

follow the CEM evaluation metric
"""

from nltk.tokenize import word_tokenize


def calc_distinct_n(n, candidates, print_score: bool = True):
    dict = {}
    total = 0
    candidates = [word_tokenize(candidate) for candidate in candidates]
    for sentence in candidates:
        for i in range(len(sentence) - n + 1):
            ney = tuple(sentence[i: i + n])
            dict[ney] = 1
            total += 1
    score = len(dict) / (total + 1e-16)

    if print_score:
        print(f"***** Distinct-{n}: {score * 100} *****")

    return score


def calc_distinct(candidates, print_score: bool = True):
    scores = []
    for i in range(2):
        score = calc_distinct_n(i + 1, candidates, print_score)
        scores.append(score)

    return scores


dist_1, dist_2 = calc_distinct(hyps)


"""
Rouge
follow this repository to calculate rouge score 
https://github.com/pltrdy/rouge
version 1.0.1
"""
scores = Rouge().get_scores(hyps, refs)

print(f"***** Rouge-l : {np.mean([score['rouge-l']['f'] for score in scores]) * 100} *****")


"""
sacrebleu 
follow the https://github.com/mjpost/sacrebleu
version is 2.3.1
"""

print(BLEU().corpus_score(hyps, [refs]))
