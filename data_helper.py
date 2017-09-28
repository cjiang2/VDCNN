import csv
import numpy as np
import random

alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:’"/|_#$%ˆ&*˜‘+=<>()[]{} '
sequence_max_length = 1014
char_dict = {}
for i,c in enumerate(alphabet):
    char_dict[c] = i

def load_csv_file(filename, num_classes):
	"""
	Load CSV file, generate one-hot labels and process text data as Paper did.
	"""
	all_data = []
	labels = []
	with open(filename) as f:
		reader = csv.DictReader(f,fieldnames=['class'],restkey='fields')
		for row in reader:
			# One-hot
			one_hot = np.zeros(num_classes)
			one_hot[int(row['class']) - 1] = 1
			labels.append(one_hot)
			# Text
			data = np.ones(sequence_max_length)*68
			text = row['fields'][1].lower()
			for i in range(0, len(text)):
				if text[i] in char_dict:
					data[i] = char_dict[text[i]]
				else:
					# unknown character set to be 67
					data[i] = 67
				if i > sequence_max_length - 1:
					break
			all_data.append(data)
	f.close()
	return all_data, labels

def load_dataset(dataset_path):
	# Read Classes Info
	with open(dataset_path+"classes.txt") as f:
		classes = []
		for line in f:
			classes.append(line.strip())
	f.close()
	num_classes = len(classes)
	# Read CSV Info
	train_data, train_label = load_csv_file(dataset_path+'train.csv', num_classes)
	test_data, test_label = load_csv_file(dataset_path+'test.csv', num_classes)
	return train_data, train_label, test_data, test_label

def batch_iter(data, batch_size, num_epochs, shuffle=True):
	"""
	Generates a batch iterator for a dataset.
	"""
	data = np.array(data)
	data_size = len(data)
	num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
	for epoch in range(num_epochs):
		# Shuffle the data at each epoch
		if shuffle:
			shuffle_indices = np.random.permutation(np.arange(data_size))
			shuffled_data = data[shuffle_indices]
		else:
			shuffled_data = data
		for batch_num in range(num_batches_per_epoch):
			start_index = batch_num * batch_size
			end_index = min((batch_num + 1) * batch_size, data_size)
			yield shuffled_data[start_index:end_index]