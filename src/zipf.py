#!/usr/bin/env python
from collections import Counter
import json
import util
import sys
import random
import numpy as np
import time
import os

try:
	import matplotlib.pyplot as plt
	import matplotlib 
	import matplotlib.gridspec as gridspec
	from matplotlib.ticker import MultipleLocator, FormatStrFormatter
except ImportError:
	print 'could not import matplotlib.'


def harmonic(N,s):
	H = 0
	for n in range(1,N+1):
		H += 1/float(n**s)

	return H

def p_zipf(x,s=1,N=1000):
	H = harmonic(N,s)
	numerator = 1 / float(x**s)
	return numerator / H
	

def calculate_zipf_divergence(zipf_sampler1, zipf_sampler2):

	max_index = max(zipf_sampler1.N, zipf_sampler2.N)
	TV = 0

	for i in range(1, max_index+1):
		TV += abs(zipf_sampler1.pdf(i) - zipf_sampler2.pdf(i))

	return TV


class zipf_sampler(object):
	def __init__(self, s=1, N=1000):
		self.s = s
		self.N = N
		self.H = harmonic(N,s)
		self.cdf = self.make_cdf(self.s, self.N)

	def pdf(self, x):
		if x > self.N:
			return 0
		return 1 / (float(x**self.s) * self.H)

	def make_cdf(self, s, N):
		cdf = []
		last_entry = 0
		for x in range(1, N+1):
			this_entry = last_entry + self.pdf(x)
			cdf.append(this_entry)
			last_entry = this_entry

		return cdf

	def sample(self):
		q = random.random()
		i = 1
		while q > self.cdf[i]:
			i += 1

		return i


def measure_TV_naive(counts1, counts2):
	score = 0
	k = float(sum(counts1.values()))
	keys = set(counts1.keys() + counts2.keys())
	for key in keys:
		score += max(counts1[key], counts2[key])

	return score / k - 1


def measure_TV_adjusted(counts1, counts2):
	score = 0
	k = float(sum(counts1.values()))
	keys = set(counts1.keys() + counts2.keys())
	for key in keys:

		big = max(counts1[key], counts2[key])
		small = min(counts1[key], counts2[key])

		if big == small:
			score += big * (small / k)
		elif big == (small + 1):
			score += big * (k + small)/(2*k)
		else:
			score += big

	return score / k - 1




def test_divergence_methods():
	zs1 = zipf_sampler(N=200, s=1.3)
	zs2 = zipf_sampler(N=300, s=1.2)
	sample1 = [zs1.sample() for i in range(10000)]
	sample2 = [zs2.sample() for i in range(10000)]
	counts1 = Counter(sample1)
	counts2 = Counter(sample2)

	TV_naive = measure_TV_naive(counts1, counts2)
	TV_adjusted = measure_TV_adjusted(counts1, counts2)
	TV_actual = calculate_zipf_divergence(zs1, zs2)

	print 'actual:', TV_actual
	print 'naive:', TV_naive
	print 'adjusted:', TV_adjusted


if __name__ == '__main__':
	test_divergence_methods()
