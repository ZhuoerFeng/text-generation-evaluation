from utils.query_metric import compute_bert_score, compute_meteror_score

sentence1 = 'The quick brown fox jumps over the lazy dog.'

sentence2 = 'The slow red pig runs over the fast cat.'

# score = compute_bert_score([sentence1] * 5, [sentence2] * 5, language='en', format_score=0.0, score=1.0)
# print(f'BERTScore: {score}')

score = compute_meteror_score(sentence1, [sentence2], format_score=0.0, score=1.0)
print(f'METEOR Score: {score}')
