# -*- coding: utf-8 -*-
# Re-implementation of the paper http://aclweb.org/anthology/D/D16/D16-1246.pdf
# 'A Stacking Gated Neural Architecture for Implicit Discourse Relation Classification'

import torch
import torch.nn as nn
import torch.nn.functional as F

class SGN16(nn.Module):

	def __init__(self, opts):

		self.vocab_size = opts.vocab_size
		self.C = opts.num_classes

		super(SGN16, self).__init__()

		self.emb = nn.Embedding(
			self.vocab_size,
			300
		)

		self.cnn1 = self.Conv2d(1, 1024, ([2, 300]))
		self.cnn2 = self.Conv2d(1, 1024, ([2, 300]))
		self.cnn3 = self.Conv2d(1, 1024, ([2, 300]))

		self.input_gate = self.Linear(6 * 1024, 6 * 1024)
		self.output_gate = self.Linear(6 * 1024, 6 * 1024)

		self.proj_concat = self.Linear(6 * 1024, 6 * 1024)

		self.classifier = self.Linear(6 * 1024, self.C)

	def __init__(self, input):
	"""

	Args
	----------
	input   : python list
		input[0] is [N, L1], input[1] is [N, L2]

	Returns
	----------
	logprob : torch.FloatTensor
		[N, C]

	"""
		arg1, arg2 = input
		arg1_emb = self.emb(arg1) # [N, L1, 300]
		arg2_emb = self.emb(arg2) # [N, L2, 300]

		conv1_out_arg1 = self.cnn1(arg1_emb.unsqueeze(1)).squeeze(3)
		conv2_out_arg1 = self.cnn2(arg1_emb.unsqueeze(1)).squeeze(3)
		conv3_out_arg1 = self.cnn3(arg1_emb.unsqueeze(1)).squeeze(3) # [N, 1024, L1']

		conv1_out_arg1 = self.cnn1(arg1_emb.unsqueeze(1)).squeeze(3)
		conv2_out_arg1 = self.cnn2(arg1_emb.unsqueeze(1)).squeeze(3)
		conv3_out_arg1 = self.cnn3(arg1_emb.unsqueeze(1)).squeeze(3) # [N, 1024, L2']

		pool_size_arg1 = conv1_out_arg1.size(2)
		pool_size_arg2 = conv1_out_arg2.size(2)

		conv1_out_arg1_pooled = F.max_pool1d(conv1_out_arg1, pool_size_arg1).squeeze(2)
		conv2_out_arg1_pooled = F.max_pool1d(conv2_out_arg1, pool_size_arg1).squeeze(2)
		conv3_out_arg1_pooled = F.max_pool1d(conv3_out_arg1, pool_size_arg1).squeeze(2)

		conv1_out_arg2_pooled = F.max_pool1d(conv1_out_arg2, pool_size_arg2).squeeze(2)
		conv2_out_arg2_pooled = F.max_pool1d(conv2_out_arg2, pool_size_arg2).squeeze(2)
		conv3_out_arg2_pooled = F.max_pool1d(conv3_out_arg2, pool_size_arg2).squeeze(2)

		v = torch.cat(
			[conv1_out_arg1_pooled, conv2_out_arg1_pooled, conv3_out_arg1_pooled, \
			conv1_out_arg2_pooled, conv1_out_arg2_pooled, conv1_out_arg2_pooled],
			dim=1
		) # [N, 6 * 1024]

		gate_i = F.sigmoid(self.input_gate(v))
		gate_o = F.sigmoid(self.output_gate(v)) # [N, 6 * 1024]

		c_prime = F.tanh(self.proj_concat(v))

		c = c_prime * gate_i

		h = F.tanh(c) * gate_o

		logprob = F.log_softmax(self.classifier(h)) # [N, C]

		return logprob