#-*- coding: utf-8 -*-
# Re-implementation of the paper http://aclweb.org/anthology/P/P17/P17-2040.pdf
# 'A Recurrent Neural Model with Attention for the Recognition of Chinese Implicit
#  Discourse Relations'

import torch
import torch.nn as nn
import torch.nn.functional as F

class RNNAtt17(nn.Module):

	def __init__(self, opts):

		self.vocab_size = opts.vocab_size

		super(RNNAtt17, self).__init__()

		self.emb = nn.Embedding(
			self.vocab_size,
			300
		)

		self.blstm = nn.LSTM(
			300,
			300,
			batch_first=True,
			bidirectional=True
		) # single layer bidirectional lstm

		self.att = nn.Linear(300, 1)
		self.proj2class = nn.Linear(300, 2)

	def forward(self, input)
	"""

	Args
	----------
	input   : [N, 256]
		N is batch size, 256 is the default length of arg1:arg2
		after padding

	Returns
	----------
	logprob : [N, C]
		C means number of classes

	"""
		batch_size = input.size(0)

		embs = self.emb(input) # [N, 256, 300]
		hids, _ = self.blstm(embs) # [N, 256, 300 x 2]
		
		l2r_hids = hids.view(
			batch_size,
			256,
			300,
			2
		)[:, :, :, 0]

		r2l_hids = hids.view(
			batch_size,
			256, 
			300,
			2
		)[:, :, :, 1]
		
		hids_sum = l2r_hids + r2l_hids # [N, 256, 300]
		hids_sum_activated = F.tanh(hids_sum)
		hids_proj = self.att(hids_sum_activated).squeeze(2) # [N, 256]
		alpha = F.softmax(hids_proj) # [N, 256]

		r = torch.bmm(
			hids_sum.transpose(1, 2),
			alpha.unsqueeze(2)
		).squeeze(2) # [N, 300]

		unnormalized_prob = self.proj2class(r) # [N, C]
		logprob = F.log_softmax(unnormalized_prob)

		return logprob