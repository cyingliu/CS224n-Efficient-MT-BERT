import pandas as pd


def prepare_sst_texts(file_path):
	print(f"Reading from {file_path}")
	df = pd.read_csv(file_path , sep='\t')
	sents = []
	for sent in df['sentence']:
		sents.append(sent)
	print(f"Data size: {len(sents)}")
	return sents

def write_to_file(sents, output_path):
	print(f"Writing to {output_path}")
	fout = open(output_path, 'w')
	for sent in sents:
		fout.write(sent + '\n')
	fout.close()

if __name__ == "__main__":
	output_path = 'data/sst-pretrain.csv'
	sst_file_path = '../data/ids-sst-train.csv'
	sents = []
	sents.extend(prepare_sst_texts(sst_file_path))
	write_to_file(sents, output_path)