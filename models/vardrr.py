# -*- coding: utf-8 -*-
# Re-implementation of the paper http://aclweb.org/anthology/D/D16/D16-1037.pdf
# 'Variational Neural Discourse Relation Recognizer'

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.distributions as dist

class VarDRR16(nn.Module):

	def __init__(self, opts):

		super(VarDRR16, self).__init__()

		# Encoder part
		self.proj_arg = nn.Embedding(
			10001,
			400,
			padding_idx=opts.padding_idx
		)
		self.proj_arg_prior = nn.Embedding(
			10001,
			400,
			padding_idx=opts.padding_idx
		)
		self.proj_y = nn.Embedding(2, 400)

		self.proj_to_mu = nn.Linear(1200, 20)
		self.proj_to_logsigma = nn.Linear(1200, 20)

		self.proj_to_mu_prior = nn.Linear(800, 20)
		self.proj_to_logsigma_prior = nn.Linear(800, 20)

		self.normal = dist.Normal(torch.zeros(20), torch.ones(20))

		# Decoder part
		self.proj_to_h1_prime = nn.Linear(20, 400)
		self.proj_to_h2_prime = nn.Linear(20, 400)

		self.proj_to_vocab_arg1 = nn.Linear(400, 10001)
		self.proj_to_vocab_arg2 = nn.Linear(400, 10001)

		self.proj_to_hy = nn.Linear(20, 400)
		self.proj_to_y = nn.Linear(400, 2)

	def forward(self, input, input_padded, z):
	"""
	`forward` computes logprob of y and x given z, the probability
	model p(y,x|z) = p(y|z)p(x|z) = p(y|z)p(x_1|z)p(x_2|z)

	Args
	----------
	input[0]  : torch.LongTensor
		[N, 10001]

	input[1]  : torch.LongTensor
		[N, 10001]

	input[2]  : torch.LongTensor
		[N, 2]

	input_padded[0] : torch.LongTensor
		[N, L]

	input_padded[1] : torch.LongTensor
		[N, L]

	z         : torch.FloatTensor	
		[N, 20]


	Returns
	----------
	loss      : torch.FloatTensor
		[1]

	"""
		z, mu_post, logsigma_post = sample_z_from_posterior(input_padded, input[2])
		_, mu_prior, logsigma_prior = sample_z_from_prior(input_padded)

		h1_prime = F.tanh(self.proj_to_h1_prime(z))
		h2_prime = F.tanh(self.proj_to_h2_prime(z))

		arg1_pred = F.sigmoid(self.proj_to_vocab_arg1(h1_prime)) # [N, 10001]
		arg2_pred = F.sigmoid(self.proj_to_vocab_arg2(h2_prime)) # [N, 10001]

		hy = F.tanh(self.porj_to_hy(z)) # [N, 400]
		y_prime = F.log_softmax(self.proj_to_y(z)) # [N, 2]

		log_p_x1_z = input[0] * torch.log(arg1_pred) + (1 - input[0]) * torch.log(1 - arg1_pred) # [N, 10001]
		log_p_x1_z = log_p_x1_z.sum() # [1]

		log_p_x2_z = input[0] * torch.log(arg2_pred) + (1 - input[0]) * torch.log(1 - arg2_pred) # [N, 10001]
		log_p_x2_z = log_p_x2_z.sum() # [1]

		log_p_y_z = F.nll_loss(y_prime, input[2]) # [1]

		# reconstruction loss
		loss_reconstruct = log_p_x1_z + log_p_x2_z + log_p_y_z # [1]

		# kl loss: https://en.wikipedia.org/wiki/Kullback-Leibler_divergence
		# D(N0 || N1) = 1/2 [tr(s2^{-1} dot S1) + (mu1 - mu2)^T s1^{-1} (mu1 - mu2) - k + ln det{s2}/det{s1}]
		loss_kl_batch = 1. / 2 * (torch.exp(logsigma_post / logsigma_prior).sum(1) + \
			((mu_prior - mu_pos) * (mu_prior - mu_post) * torch.exp(logsigma_prior)).sum(1) - 20 + \
			(logsigma_prior - logsigma_post).sum(1)) # [N]
		loss_kl = 1. / N * loss_kl_batch.sum()

		loss_total = loss_reconstruct + loss_kl

		return loss_total

	def sample_z_from_posterior(self, input, label):
	"""
	Method to do z~q(z|x,y) sampling, it is a centered Gaussian
	distribution.

	Args
	----------
	input : python list
		input[0] is arg1, input[1] is arg2 which is the one-hot
		representation of each sentence/argument. A better choice
		of the sentence representation is through padded id sequence
		which is within a batch padded w.r.t. longest sequence.

	label : torch.LongTensor
		one-hot representation of the label with shape [N, Dy],
		and in the paper Dy is set to 2


	Returns
	----------
	z         : torch.FloatTensor
		a batch of high dimentional vectors with shape [N, Dz],
		and in the paper Dz is set to 20

	mu        : torch.FloatTensor
		[N, 20]

	logsigma  : torch.FloatTensor
		[N, 20]

	"""
		N = label.size(0)

		h1 = F.tanh(self.proj_arg(input[0]).sum(1)) # [N, L, 400] ==> [N, 400]
		h2 = F.tanh(self.proj_arg(input[1]).sum(1)) # [N, L, 400] ==> [N, 400]
		hy = F.tanh(self.proj_y(label)).squeeze(1) # [N, 1, 400] ==> [N, 400]

		mu = self.proj_to_mu(torch.cat([h1, h2, hy], dim=1)) # [N, 20]
		logsigma = self.proj_to_logsigma(torch.cat([h1, h2, hy], dim=1)) # [N, 20]

		z = mu + self.normal.sample_n(N) * torch.exp(logsigma)

		return z, mu, logsigma

	def sample_z_from_prior(self, input):
	"""
	Method to do z~p(z|x) which the author claims is the prior of z, we could think
	of it as empirical prior.

	Args
	----------
	input   : torch.LongTensor
		[N, L]

	Returns
	----------
	z       : torch.FloatTensor
		[N, 20]

	mu      : torch.FloatTensor
		[N, 20]

	logsigma: torch.FloatTensor
		[N, 20]

	"""
		N = label.size(0)

		emb1 = self.

		h1 = F.tanh(self.proj_arg_prior(input[0]).sum(1)) # [N, L, 400] ==> [N, 400]
		h2 = F.tanh(self.proj_arg_prior(input[1]).sum(1)) # [N, L, 400] ==> [N, 400]

		mu = self.proj_to_mu_prior(torch.cat([h1, h2], dim=1)) # [N, 20]
		logsigma = self.proj_to_logsigma_prior(torch.cat([h1, h2], dim=1)) # [N, 20]

		z = mu + self.normal.sample_n(N) * torch.exp(logsigma)

		return z, mu, logsigma