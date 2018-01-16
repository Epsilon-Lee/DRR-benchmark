# -*- coding: utf=8 -*-
# Re-implementation of the paper http://aclweb.org/anthology/C/C16/C16-1180.pdf
# 'Implicit Discourse Relation Recognition with Context-aware Character-enhanced 
#  Embeddings'

import torch
import torch.nn as nn
import torch.nn.functional as F

class CCE16(nn.Module):

	def __init__(self, opts):
		self.char_vocab_size = opts.char_vocab_size
		self.word_vocab_size = opts.word_vocab_size
		self.C = opts.num_classes

		super(CCE16, self).__init__()

		self.char_emb = nn.Embeddings(
			self.char_vocab_size,
			30
		)

		self.char_cnn1 = nn.Conv2d(
			1,
			128,
			[2, 30]
		)

		self.char_cnn2 = nn.Conv2d(
			1,
			128,
			[3, 30]
		)

		self.char_cnn3 = nn.Conv2d(
			1,
			128,
			[4, 30]
		)
		self.word_emb = nn.Embeddings(
			self.word_vocab_size,
			300
		)

		self.blstm = nn.LSTM(
			3 * 128,
			50,
			batch_first=True,
			bidirectional=True
		)

		self.word_cnn1 = nn.Conv2d(
			1,
			1024,
			[2, 400]
		)

		self.word_cnn2 = nn.Conv2d(
			1,
			1024,
			[4, 400]
		)

		self.word_cnn3 = nn.Conv2d(
			1,
			1024,
			[8, 400]
		)

		self.linear = self.Linear(6 * 1024, 100)
		self.classifier = self.Linear(100, self.C)

	def forward(self, char_input, word_input):
	"""

	Args
	----------
	char_input : python list
		char_input[0] is torch.LongTensor, size is [N, 80, 20] for arg1
		char_input[1] is torch.LongTensor as well for arg2

	word_input : python list
		word_input[0] is torch.LongTensor, size is [N, 80] for arg1,
		word_input[1] is torch.LongTensor as well for arg2

	Returns
	----------
	logprob    : torch.FloatTensor
		[N, C]

	"""
		N = arg1_char.size(0)

		arg1_char = char_input[0] # [N, 80, 20]
		arg2_char = char_input[1]

		arg1_word = word_input[0]
		arg2_word = word_input[1]

		arg1_char_emb = self.char_emb(arg1_char).view(N * 80, 20, 30) # [N * 80, 1, 20, 30]
		arg2_char_emb = self.char_emb(arg2_char).view(N * 80, 20, 30)

		arg1_word_emb = self.word_emb(arg1_word) # [N, 80, 300]
		arg2_word_emb = self.word_emb(arg2_word)

		arg1_char_conv1 = F.tanh(self.char_cnn1(arg1_char_emb).squeeze(3)) # [N * 80, 128, 19]
		arg1_char_conv2 = F.tanh(self.char_cnn2(arg1_char_emb).squeeze(3)) # [N * 80, 128, 18]
		arg1_char_conv3 = F.tanh(self.char_cnn3(arg1_char_emb).squeeze(3)) # [N * 80, 128, 17]

		arg2_char_conv1 = F.tanh(self.char_cnn1(arg2_char_emb).squeeze(3)) # [N * 80, 128, 19]
		arg2_char_conv2 = F.tanh(self.char_cnn2(arg2_char_emb).squeeze(3)) # [N * 80, 128, 18]
		arg2_char_conv3 = F.tanh(self.char_cnn3(arg2_char_emb).squeeze(3)) # [N * 80, 128, 17]

		arg1_char_pool1 = F.max_pool1d(arg1_char_conv1, arg1_char_conv1.size(2))
		arg1_char_pool2 = F.max_pool1d(arg1_char_conv2, arg1_char_conv2.size(2))
		arg1_char_pool3 = F.max_pool1d(arg1_char_conv3, arg1_char_conv3.size(2))

		arg2_char_pool1 = F.max_pool1d(arg2_char_conv1, arg2_char_conv1.size(2))
		arg2_char_pool2 = F.max_pool1d(arg2_char_conv2, arg2_char_conv2.size(2))
		arg2_char_pool3 = F.max_pool1d(arg2_char_conv3, arg2_char_conv3.size(2))

		arg1_char = torch.cat([arg1_char_pool1, arg1_char_pool2, arg1_char_pool3], dim=1) # [N * 80, 128 * 3]
		arg2_char = torch.cat([arg2_char_pool1, arg2_char_pool2, arg2_char_pool3], dim=1)
		arg1_char = arg1_char.view(N, 80, -1) # [N, 80, 128 * 3]
		arg2_char = arg2_char.view(N, 80, -1)

		arg1_char_hid, _ = self.blstm(arg1_char).view(N, 80, 50, 2) # [N, 80, 50 * 2]
		arg2_char_hid, _ = self.blstm(arg2_char).view(N, 80, 50, 2)

		arg1_enhanced = torch.cat(
			[arg1_char_hid[:, :, :, 0], arg1_word_emb, arg1_char_hid[:, :, :, 1]],
			dim=2
		) # [N, 80, 400]
		arg2_enhanced = torch.cat(
			[arg2_char_hid[:, :, :, 0], arg2_word_emb, arg2_char_hid[:, :, :, 1]],
			dim=2
		)

		arg1_enhanced_conv1 = F.tanh(self.word_cnn1(arg1_enhanced.unsqueeze(1))).squeeze(3) # [N, 1024, 79, 1]
		arg1_enhanced_conv2 = F.tanh(self.word_cnn2(arg1_enhanced.unsqueeze(1))).squeeze(3) # [N, 1024, 77, 1]
		arg1_enhanced_conv3 = F.tanh(self.word_cnn3(arg1_enhanced.unsqueeze(1))).squeeze(3) # [N, 1024, 73, 1]

		arg2_enhanced_conv1 = F.tanh(self.word_cnn1(arg2_enhanced.unsqueeze(1))).squeeze(3) # [N, 1024, 79, 1]
		arg2_enhanced_conv2 = F.tanh(self.word_cnn2(arg2_enhanced.unsqueeze(1))).squeeze(3) # [N, 1024, 77, 1]
		arg2_enhanced_conv3 = F.tanh(self.word_cnn3(arg2_enhanced.unsqueeze(1))).squeeze(3) # [N, 1024, 73, 1]

		arg1_enhanced_pool1 = F.max_pool1d(arg1_enhanced_conv1, arg1_enhanced_conv1.size(2)).squeeze(2) # [N, 1024]
		arg1_enhanced_pool2 = F.max_pool1d(arg1_enhanced_conv1, arg1_enhanced_conv2.size(2)).squeeze(2)		
		arg1_enhanced_pool3 = F.max_pool1d(arg1_enhanced_conv1, arg1_enhanced_conv3.size(2)).squeeze(2)

		arg2_enhanced_pool1 = F.max_pool1d(arg2_enhanced_conv1, arg1_enhanced_conv1.size(2)).squeeze(2)		
		arg2_enhanced_pool2 = F.max_pool1d(arg2_enhanced_conv1, arg1_enhanced_conv2.size(2)).squeeze(2)		
		arg2_enhanced_pool3 = F.max_pool1d(arg2_enhanced_conv1, arg1_enhanced_conv3.size(2)).squeeze(2)

		arg1_s = torch.cat(
			[arg1_enhanced_pool1, arg1_enhanced_pool2, arg1_enhanced_pool3],
			dim=1
		) # [N, 3 * 1024]
		
		arg2_s = torch.cat(
			[arg2_enhanced_pool1, arg2_enhanced_pool2, arg2_enhanced_pool3],
			dim=1
		) # [N, 3 * 1024]

		v = torch.cat([arg1_s, arg2_s], dim=1) # [N, 6 * 1024]
		v_proj = F.tanh(self.linear(v)) # [N, 100]
		logprob = F.log_softmax(self.classifier(v_proj)) # [N, C]

		return logprob