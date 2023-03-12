import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str)
parser.add_argument('--data_name', type=str)
parser.add_argument('--pair_input', action='store_true')
args = parser.parse_args()

df = pd.read_csv(args.data_path, sep='\t')
print("Dataset lenght:", len(df))

if not args.pair_input:
	sent_lengths = []
	sent_num_words = []
	if 'sst' in args.data_path:
		metric = 'sentence'
	else: # 'kindle'
		metric = 'sentence'
	for i, sent in enumerate(df[metric]):
		try:
			sent_lengths.append(len(sent))
			sent_num_words.append(len(sent.split(' ')))
		except:
			print(i, sent)
		# print(sent.split(' '), len(sent.split(' ')))
	print("Avg sent length:", np.mean(sent_lengths))
	print("Avg/Min/Max num words:", np.mean(sent_num_words), np.min(sent_num_words), np.max(sent_num_words))
	counts, bins = np.histogram(sent_lengths)
	plt.stairs(counts, bins)
	plt.title(f'Sentence Length Distribution of {args.data_name}')
	plt.show()

	scores = []
	if 'sst' in args.data_path:
		metric = 'sentiment'
	else: # 'kindle'
		metric = 'similarity'
	for score in df[metric]:
		scores.append(score)
	print("Avg sentiment:", np.mean(scores))
	counts, bins = np.histogram(scores)
	plt.stairs(counts, bins)
	plt.title(f'Score Distribution of {args.data_name}')
	plt.show()
else:
	sent_lengths = []
	for sent in df['sentence1']:
		try:
			sent_lengths.append(len(sent))
		except:
			pass
	for sent in df['sentence2']:
		try:
			sent_lengths.append(len(sent))
		except:
			pass
	print("Avg sent length:", np.mean(sent_lengths))
	counts, bins = np.histogram(sent_lengths)
	plt.stairs(counts, bins)
	plt.title(f'Sentence Length Distribution of {args.data_name}')
	plt.show()

	scores = []
	num_nan = 0
	if 'sts' in args.data_path or 'SICK' in args.data_path:
		metric = 'similarity'
	else:
		metric = 'is_duplicate'
	for score in df[metric]:
		if np.isnan(score):
			num_nan += 1
			continue
		scores.append(score)
	print("Avg score:", np.mean(scores))
	print("Nan score:", num_nan)
	counts, bins = np.histogram(scores)
	plt.stairs(counts, bins)
	plt.title(f'Score Distribution of {args.data_name}')
	plt.show()



