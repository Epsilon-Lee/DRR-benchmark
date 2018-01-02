import os
import codecs

path = 'pdtb_data/pdtb'

if os.path.isdir(path):
	print(path + ' is a directory')

# for folder in os.walk(path):
# 	print(folder)

count_implicit = 0
comparison = []
contingency = []
temporal = []
expansion = []
# entrel = []

def extract_example(example):

	if 'Implicit' not in example[1]:
		return

	globals()["count_implicit"] += 1

	label = ''
	arg1 = ''
	arg2 = ''

	idx = 0
	while idx < len(example):
		if 'Comparison' in example[idx]:
			label = 'Comparison'
			idx += 1
			continue
		elif 'Contingency' in example[idx]:
			label = 'Contingency'
			idx += 1
			continue
		elif 'Temporal' in example[idx]:
			label = 'Temporal'
			idx += 1
			continue			
		elif 'Expansion' in example[idx]:
			label = 'Expansion'
			idx += 1
			continue

		if example[idx] == '____Arg1____':
			idx += 4
			arg1 = example[idx]

		if example[idx] == '____Arg2____':
			idx += 4
			arg2 = example[idx]
			break

		idx += 1

	if label == 'Comparison':
		comparison.append([label, arg1, arg2])
	elif label == 'Contingency':
		contingency.append([label, arg1, arg2])
	elif label == 'Temporal':
		temporal.append([label, arg1, arg2])
	elif label == 'Expansion':
		expansion.append([label, arg1, arg2])


if __name__ == '__main__':

	build_split = 'test' # 'train', 'dev', 'test'
	train_range = (2, 20)
	dev_range = (0, 1)
	test_range = (21, 22)

	print('Preprocessing %s data...' % (build_split))
	for folder in os.listdir(path):
		folder_id = int(folder)
		print(folder_id)
		if build_split == 'train':
			if folder_id < train_range[0] or folder_id > train_range[1]:
				continue
		if build_split == 'dev':
			if folder_id < dev_range[0] or folder_id > dev_range[1]:
				continue
		if build_split == 'test':
			if folder_id < test_range[0] or folder_id > test_range[1]:
				continue
		print('Extracting implicit drs from folder' + folder)
		folder_path = os.path.join(path, folder)
		for file in os.listdir(folder_path):
			# print('Extracting implicit drs from file ' + file)
			with codecs.open(os.path.join(folder_path, file), encoding='utf-8', errors='replace') as f:
				# lines = f.readlines()
				lines = []
				line = f.readline()
				while line:
					lines.append(line)
					# print(line)
					line = f.readline()
				buf = []
				count_flag = 0
				for line in lines:
					buf.append(line[:-1])
					if line[:-1] == '________________________________________________________':
						count_flag += 1
					if count_flag == 2:
						# for line_ in buf:
						# 	print(line_)
						# print
						extract_example(buf)
						buf = []
						count_flag = 0

	print('Implicit    count: %d' % (count_implicit))
	print('Comparison  count: %d' % (len(comparison)))
	print('Contingency count: %d' % (len(contingency)))
	print('Temporal    count: %d' % (len(temporal)))
	print('Expansion   count: %d' % (len(expansion)))

	save_to_path = build_split + '.raw.txt'
	data_lst = []
	data_lst.extend(comparison)
	data_lst.extend(contingency)
	data_lst.extend(temporal)
	data_lst.extend(expansion)

	f_save_to = open(save_to_path, 'w')
	for example in data_lst:
		example_str = example[0] + '|||' + example[1] + '|||' + example[2]
		f_save_to.write(example_str + '\n')