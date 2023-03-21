import pandas as pd
import random


def prepare_sst_texts(file_path):
	print(f"Reading from {file_path}")
	df = pd.read_csv(file_path , sep='\t')
	sents = []
	for sent in df['sentence']:
		sents.append(sent.strip())
	print(f"Data size: {len(sents)}")
	return sents

def prepare_sentence_pair_texts(file_path):
	print(f"Reading from {file_path}")
	df = pd.read_csv(file_path , sep='\t')
	sents = []
	for sent in df['sentence1']:
		try:
			sents.append(sent.strip())
		except:
			pass # nan
	for sent in df['sentence2']:
		try:
			sents.append(sent.strip())
		except:
			pass # nan
	print(f"Data size: {len(sents)}")
	return sents

def write_to_file(sents, output_path):
	print(f"Writing to {output_path}")
	fout = open(output_path, 'w')
	fout.write("text\n")
	for sent in sents:
		fout.write(sent + '\n')
	fout.close()
	print(f"Data size: {len(sents)}")

if __name__ == "__main__":
	output_path = 'data/quora-sts-pretrain-test.csv'
	# sst_file_path = '../data/ids-sst-dev.csv'
	# sst_rotten_file_path = '../data/ids-sst-rotten15k-train.csv'
	quora_file_path = '../data/quora-dev.csv'
	sts_file_path = '../data/sts-dev.csv'
	sents = []
	# sents.extend(prepare_sst_texts(sst_rotten_file_path))
	sents.extend(prepare_sentence_pair_texts(quora_file_path))
	sents.extend(prepare_sentence_pair_texts(sts_file_path))
	# do shuffling after constructing dataset, so that the same chunk mostly contains close sentences

	write_to_file(sents, output_path)