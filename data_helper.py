"""
data helper to preprocess csv format text dataset
"""
import csv
import numpy as np
import random

class data_helper():
	def __init__(self, sequence_max_length=1024):
		self.alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:’"/|_#$%ˆ&*˜‘+=<>()[]{} '
		self.char_dict = {}
		self.sequence_max_length = sequence_max_length
		for i,c in enumerate(self.alphabet):
			self.char_dict[c] = i+1

	def char2vec(self, text):
		data = np.zeros(self.sequence_max_length)
		for i in range(0, len(text)):
			if i > self.sequence_max_length:
				return data
			elif text[i] in self.char_dict:
				data[i] = self.char_dict[text[i]]
			else:
				# unknown character set to be 68
				data[i] = 68
		return data

	def load_csv_file(self, filename, num_classes):
		"""
		Load CSV file, generate one-hot labels and process text data as Paper did.
		"""
		all_data = []
		labels = []
		with open(filename) as f:
			reader = csv.DictReader(f, fieldnames=['class'], restkey='fields')
			for row in reader:
				# One-hot
				one_hot = np.zeros(num_classes)
				one_hot[int(row['class']) - 1] = 1
				labels.append(one_hot)
				# Char2vec
				data = np.ones(self.sequence_max_length)*68
				text = row['fields'][-1].lower()
				all_data.append(self.char2vec(text))
		f.close()
		return np.array(all_data), np.array(labels)

	def load_dataset(self, dataset_path):
		# Read Classes Info
		with open(dataset_path+"classes.txt") as f:
			classes = []
			for line in f:
				classes.append(line.strip())
		f.close()
		num_classes = len(classes)
		# Read CSV Info
		train_data, train_label = self.load_csv_file(dataset_path+'train.csv', num_classes)
		test_data, test_label = self.load_csv_file(dataset_path+'test.csv', num_classes)
		return train_data, train_label, test_data, test_label

	def batch_iter(self, data, batch_size, num_epochs, shuffle=True):
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