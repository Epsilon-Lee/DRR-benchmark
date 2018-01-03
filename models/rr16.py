# -*- coding: utf-8 -*-
# Re-implementation of the paper http://aclweb.org/anthology/D/D16/D16-1130.pdf
# 'Recognizing Implicit Discourse Relations via Repeated Reading: Neural Networks
#  with Multi-Level Attention'

# Notice
# ==========
# In the following code, a two-level repeated reading is conducted, since
# it reveals the best result in the experimental results of the paper.

import torch
import torch.nn as nn
import torch.nn.functional as F

class RR16(nn.Module):

	def __init__(self, opts):

		self.vocab_size = opts.vocab_size
		self.C = opts.num_classes

		super(RR16, self).__init__()

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

		self.att1 = nn.Linear(300, 200)
		self.att1_proj_mem = nn.Linear(200, 100)
		self.att1_proj_hid = nn.Linear(100, 100)
		self.att1_proj = nn.Linear(100, 1)

		self.att2 = nn.Linear(500, 200)
		self.att2_proj_mem = nn.Linear(200, 100)
		self.att2_proj_hid = nn.Linear(100, 100)
		self.att2_proj = nn.Linear(100, 1)

		self.classifier = nn.Linear(300, self.C)

	def forward(self, input):
	"""

	Args
	----------
	input   : python list
		input[0] is arg1, input[1] is arg2; each of
		which has size [N, L1] or [N, L2]

	Returns
	----------
	logprob : torch.FloatTensor
		[N, C]

	"""
		arg1, arg2 = input
		N = arg1.size(0)
		L1 = arg1.size(1)
		L2 = arg2.size(1)
		arg1_emb = self.emb(arg1) # [N, L1, 50]
		arg2_emb = self.emb(arg2) # [N, L2, 50]

		arg1_hid, _ = self.blstm(arg1_emb) # [N, L1, 50 * 2]
		arg2_hid, _ = self.blstm(arg2_emb) # [N, L2, 50 * 2]

		arg1_hid_avg = torch.sum(arg1_hid, dim=1) / L1 # [N, 100]
		arg2_hid_avg = torch.sum(arg2_hid, dim=1) / L2 # [N, 100]

		M1 = F.tanh(
			self.att1(
				torch.cat(
					[arg1_hid_avg, arg2_hid_avg, arg1_hid_avg - arg2_hid_avg],
					dim=1
				)
			)
		) # [N, 200]
		M1_repeat_arg1 = M1.repeat(1, L1).view(N, L1, 200)
		M1_repeat_proj_arg1 = self.att1_proj_mem(M1_repeat_arg1) # [N, L1, 100]
		M1_repeat_arg2 = M1.repeat(1, L2).view(N, L2, 200)
		M1_repeat_proj_arg2 = self.att1_proj_mem(M1_repeat_arg2) # [N, L2, 100]

		arg1_hid1_proj = self.att1_proj_hid(arg1_hid) # [N, L1, 100]
		arg2_hid1_proj = self.att1_proj_hid(arg2_hid) # [N, L2, 100]

		o1_arg1 = F.tanh(M1_repeat_proj_arg1 + arg1_hid1_proj)
		a1_arg1 = F.softmax(self.att1_proj(o1_arg1).squeeze(2)) # [N, L1]

		o1_arg2 = F.tanh(M1_repeat_proj_arg2 + arg2_hid1_proj)
		a1_arg2 = F.softmax(self.att1_proj(o1_arg2).squeeze(2)) # [N, L2]

		R1_arg1 = torch.bmm(a1_arg1.unsqueeze(1), arg1_hid).squeeze(1) # [N, 100]
		R1_arg2 = torch.bmm(a1_arg2.unsqueeze(1), arg2_hid).squeeze(1) # [N, 100]

		M2 = F.tanh(
			self.att2(
				torch.cat(
					[R1_arg1, R1_arg2, R1_arg1 - R1_arg2, M1],
					dim=1
				)
			)
		) # [N, 200]
		M2_repeat_arg1 = M2.repeat(1, L1).view(N, L1, 200)
		M2_repeat_proj_arg1 = self.att2_proj_mem(M2_repeat_arg1)
		M2_repeat_arg2 = M2.repeat(1, L2).view(N, L2, 200)
		M2_repeat_proj_arg2 = self.att2_proj_mem(M2_repeat_arg2)

		arg1_hid2_proj = self.att2_proj_hid(arg1_hid)
		arg1_hid2_proj = self.att2_proj_hid(arg2_hid)

		o2_arg1 = F.tanh(M2_repeat_proj_arg1 + arg1_hid2_proj)
		a2_arg1 = F.softmax(self.att2_proj(o2_arg1).squeeze(2))

		o2_arg2 = F.tanh(M2_repeat_proj_arg2 + arg2_hid2_proj)
		a2_arg2 = F.softmax(self.att2_proj(o2_arg2).squeeze(2))

		R2_arg1 = torch.bmm(a2_arg1.unsqueeze(1), arg1_hid).squeeze(1)
		R2_arg2 = torch.bmm(a2_arg2.unsqueeze(1), arg2_hid).squeeze(1)

		logprob = F.log_softmax(
			self.classifier(
				torch.cat(
					[R2_arg1, R2_arg2, R2_arg1 - R2_arg2],
					dim=1
				)
			)
		) # [N, C]

		return logprob