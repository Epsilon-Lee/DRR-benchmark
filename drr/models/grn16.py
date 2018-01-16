#-*- coding: utf-8 -*-
# Re-implementation of the paper http://aclweb.org/anthology/P/P16/P16-1163.pdf
# 'Implicit Discourse Relation Detection via a Deep Architecture with Gated
#  Relevance Network'

import torch
import torch.nn as nn
import torch.nn.functional as F

class GRN16(nn.Module):

	def __init__(self, opts):

		self.vocab_size = opts.vocab_size # default 10000
		self.r = opts.r # 1, 2

		super(GRN16, self).__init__()


		self.emb = nn.Embedding(
			self.vocab_size,
			50
		)

		self.blstm = nn.LSTM(
			50,
			50,
			batch_first=True,
			bidirectional=True
		)

		self.gate = nn.Linear(200,
			self.r
		)

		self.H = nn.Bilinear(
			100,
			100,
			self.r,
			bias=False
		)

		self.V = nn.Linear(
			200,
			self.r,
			bias=False

		)
		self.b = Parameter(torch.zeros(1, 2))

		self.v = nn.Linear(self.r, 1)
		
		self.maxpool2d = nn.MaxPool2d([3, 3])
		self.linear1 = nn.Linear(17 * 17, 50) # [50, 50] --> (3, 3) maxpool --> [17, 17]
		self.linear2 = nn.Linear(50, 2)

	def forward(self, input):
	"""

	Args
	----------
	input   : python list
		input[0] is arg1, input[1] is arg2

	Returns
	----------
	logprob : N x 1

	"""
		arg1, arg2 = input
		arg1_emb = self.emb(arg1)
		arg2_emb = self.emb(arg2)
		arg1_hid, _ = self.blstm(arg1_emb) # [N, 50, 50 * 2]
		arg2_hid, _ = self.blstm(arg2_emb) # [N, 50, 50 * 2]

		cross_hid = self.construct_cross_hid(
			arg1_hid,
			arg2_hid
		) # [N, 50, 50, 200]

		gate = F.sigmoid(self.gate(cross_hid)) # [N, 50, 50, r]
		one_minus_gate = 1 - gate

		bilinear_out = self.H(
			cross_hid[:, :, :, :100].contiguous().view(-1, 100),
			cross_hid[:, :, :, 100:].contiguous().view(-1, 100)
		).view(:, 50, 50, self.r) # [N, 50, 50, r]

		singler_layer_out = F.tanh(self.V(cross_hid)) # [N, 50, 50, r]

		s = gate * bilinear_out + one_minus_gate * singler_layer_out + self.b # [N, 50, 50, r]
		s = self.v(s).transpose(1, 3) # [N. 1, 50, 50]

		s_pooled = self.maxpool2d(s) # [N, 1, 17, 17]
		s_linear = s_pooled.view(-1, 17 * 17) # [N, 249]

		linear1_out = F.tanh(self.linear1(s_linear)) # [N, 50]
		logprob = F.log_softmax(self.linear2(linear1_out)) # [N, 2]

		return logprob

	def construct_cross_hid(arg1_hid, arg2_hid):
	"""Construct cross concatenation of hiddens:
	   that is torch.cat([arg1_hid[i], arg2_hid[j]], dim=2)

	Args
	----------
	arg1_hid  : torch.FloatTensor
		[N, 50, 100]
	arg2_hid  : torch.FloatTensor
		[N, 50, 100]

	Returns
	----------
	cross_hid : torch.FloatTensor
		[N, 50, 50, 200]

	"""
		N = arg1_hid.size(0)

		## RuntimeError: in-place operatioins can be only used on variables that
		## don't share storage with any other variables, but detected that there
		## are 2 objects sharing it
		# cross_hid = Variable(torch.zeros(N, 50, 50, 200))
		
		# for i in range(50):
		# 	for j in range(50):
		# 		hid_cat = torch.cat([arg1_hid[:, i, :], arg2_hid[:, j, :]], dim=1)
		# 		cross_hid[:, i, j, :].copy_(hid_cat)
		hid_cat_stack = []

		for i in range(50):
			arg1_hid_slice = arg1_hid[:, i, :] # [N, 100]
			arg1_hid_slice_repeat = arg1_hid_slice.repeat(1, 50).contiguous().view(N, 50, 100)
			hid_cat = torch.cat([arg1_hid_slice_repeat, arg2_hid], dim=2) # [N, 50, 200]
			hid_cat_stack.append(hid_cat)

		cross_hid = torch.stack(hid_cat_stack, dim=1) # [N, 50, 50, 200]
		return cross_hid
