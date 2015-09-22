#!/usr/bin/env python

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

FIG_PATH = (
	'/Users/enewel3/projects/priming/docs/publications/'
	'2014.09_SCIENCE/figs/normal_example.pdf'
)

def calculate_l1(dist1, dist2):
	l1 = 0
	total_area = 0
	for x1, x2 in zip(dist1, dist2):
		l1 += max(0, x1 - x2)
		total_area += x1

	l1 = l1 / float(total_area)
	return l1


def get_normal_dist(mu,sigma,X):
	sigma = float(sigma)
	return [
		np.e**(-(x - mu)**2 / sigma**2)
		for x in X
	]


def plot_normal_distributions():

	# make a figure with three subplots
	figWidth = 8.7 / 2.54 	# conversion from PNAS spec in cm to inches
	figHeight = 3/5.5*figWidth	# a reasonable aspect ratio
	fig = plt.figure(figsize=(figWidth, figHeight))
	gs = gridspec.GridSpec(1,2)

	# now plot the vocabulary data
	ax1 = plt.subplot(gs[0])

	X = range(1000)
	Y1 = get_normal_dist(428, 150, X)
	Y2 = get_normal_dist(572, 150, X)

	print calculate_l1(Y1, Y2)
	series_1 = ax1.plot(X, Y1, color='0.25')
	series_2 = ax1.plot(X, Y2, color='0.25', linestyle=':')
	plt.tick_params(
		axis='y', which='both', left='off', right='off', labelleft='off')
	plt.tick_params(
		axis='x', which='both', bottom='off', top='off', labelbottom='off')

	Y1 = get_normal_dist(459, 150, X)
	Y2 = get_normal_dist(541, 150, X)

	ax2 = plt.subplot(gs[1], sharey=ax1)
	print calculate_l1(Y1, Y2)
	series_3 = ax2.plot(X, Y1, color='0.25')
	series_2 = ax2.plot(X, Y2, color='0.25', linestyle=':')

	plt.ylim(0, 1.2)
	ax1.set_ylabel('$p(x)$', size=12)
	ax1.set_xlabel('$x$', size=12)
	ax2.set_xlabel('$x$', size=12)

	plt.tick_params(
		axis='y', which='both', left='off', right='off', labelleft='off')
	plt.tick_params(
		axis='x', which='both', bottom='off', top='off', labelbottom='off')

	plt.setp(ax1.get_yticklabels(), visible=False)
	plt.setp(ax2.get_yticklabels(), visible=False)
	plt.setp(ax1.get_xticklabels(), visible=False)
	plt.setp(ax2.get_xticklabels(), visible=False)

	plt.draw()
	plt.tight_layout()
	fig.subplots_adjust(wspace=0.05, top=0.92, right=0.93, left=0.11, 
		bottom=0.14)

	ax1.text(950, 1.17, 'A', va='top', ha='right', size=16, color='0.55')
	ax2.text(950, 1.17, 'B', va='top', ha='right', size=16, color='0.55')
	ax1.text(450, 0.60, '$a_1$', va='top', ha='right', size=11, color='0.25')
	ax1.text(560, 0.25, '$a_2$', va='top', ha='right', size=11, color='0.25')
	ax1.text(680, 0.60, '$a_3$', va='top', ha='right', size=11, color='0.25')

	fig.savefig(FIG_PATH)

if __name__ == '__main__':
	plot_binomial_distributions()

def plot_binomial_distributions():
	N = 10
	X = range(N)
	p1 = 0.25
	p2 = 0.75

	Y1 = [
		util.choose(N, x)*(p1**x)*((1-p1)**(N-x))
		for x in X
	]
	Y2 = [
		util.choose(N,x)*(p2**x)*((1-p2)**(N-x))
		for x in X
	]

	# make a figure with three subplots
	figWidth = 16.78 / 2.54 	# conversion from PNAS spec in cm to inches
	figHeight = 3/5.*figWidth	# a reasonable aspect ratio
	fig = plt.figure(figsize=(figWidth, figHeight))
	gs = gridspec.GridSpec(1,2)

	# now plot the vocabulary data
	series_1 = ax1.plot(X, Y1, color='0.25')
	series_2 = ax1.plot(X, Y2, color='0.55')


if __name__ == '__main__':
	plot_binomial_distributions()
