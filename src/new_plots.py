'''
This module provides a collection of functions that produce the figures used
in the paper
'''

import json
import util
from analysis import get_theta_star, binomial_lower_confidence_p, binomial_confidence_intervals
import sys
import random
import naive_bayes
import ontology
import data_processing
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


# ** Constants ** #
#z-score for two-tailed 99-percent confidence interval
CONFIDENCE_95 = 1.96
CONFIDENCE_99 = 2.975

IMAGE_NAMES = ['test%d' %i for i in range(5)]

DATA_DIR = 'data/new_data'
FIGS_DIR = 'figs/'


def delta_counts_to_table(
		read_fname = 'data/new_data/delta_counts.json',
		write_fname = 'data/new_data/delta_counts_tables.txt'
	):
	'''
		writes the delta food out as latex tables
	'''

	# open files
	data = json.loads(open(read_fname).read())
	write_fh = open(write_fname, 'w')

	START_TABLE_SERIES = r'''
	\setlength{\tabcolsep}{12pt}
	\begin{tabular}{ c c c c c }
	'''

	END_TABLE_SERIES = r'''
	\end{tabular}'''

	START_TABLE = r'''
		\setlength{\tabcolsep}{2pt}
		\begin{tabular}{ r | c }
		\toprule
		\multicolumn{2}{c}{\textit{%s}} \\
		\toprule'''

	END_TABLE = r'''
		\bottomrule
		\end{tabular}'''

	pairs = ['task1', 'frame1', 'echo', 'task2', 'frame2']
	first = True
	write_fh.write(START_TABLE_SERIES)
	for pair_name in pairs:
		deltas = data[pair_name]

		if first:
			first = False
		else:
			write_fh.write('\n&\n')

		write_fh.write(START_TABLE % pair_name)

		for word, delta_count in deltas:
			write_fh.write('\n\t\t%s & %d \\\\' % (word, delta_count))

		write_fh.write(END_TABLE + '\n')

	write_fh.write(END_TABLE_SERIES)


def plot_longit_vocab(
		read_vocab_fname = 'data/new_data/vocabulary.json',
		write_fname = 'figs/longit_vocab.pdf'
	):

	vocab_data = json.loads(open(read_vocab_fname).read())
	exp1_task_food = vocab_data['exp1.task.food']
	exp1_task_cult = vocab_data['exp1.task.cult']

	# make a figure with three subplots
	figWidth = 16.78 / 2.54 	# conversion from PNAS spec in cm to inches
	figHeight = 3/5.*figWidth	# a reasonable aspect ratio
	fig = plt.figure(figsize=(figWidth, figHeight))
	gs = gridspec.GridSpec(1, 3)

	# now plot the vocabulary data
	ax1 = plt.subplot(gs[0])
	width = 0.75

	Y_1 = []
	for image in range(5):
		vocab_food, vocab_cult = exp1_task_food[image], exp1_task_cult[image]
		Y_1.append((vocab_food - vocab_cult) / float(vocab_cult))

	X_1 = range(len(Y_1))

	series_1 = ax1.bar(X_1, Y_1, width, color='0.25')

	padding = 0.25
	xlims = (-padding, len(X_1) - 1 + width + padding)
	plt.xlim(xlims)

	ax1.set_ylabel(r'relative increase in vocabulary', size=12)

	xlabels = ['image %d' % i for i in range(1,6)]

	ax1.set_xticks([x + width/2. for x in X_1])
	ax1.set_xticklabels(xlabels, rotation=45, size=12,
		horizontalalignment='right')
	ax1.set_yticks([0.1,0.2,0.3,0.4])

	plt.draw()
	plt.tight_layout()
	fig.subplots_adjust(wspace=0.35, top=0.82, right=0.99, left=0.10, 
		bottom=0.20)
	fig.savefig(write_fname)


def plot_specificity(
		read_food_fname='data/new_data/food.json',
		read_specificity_fname = 'data/new_data/specificity.json',
		read_vocab_fname = 'data/new_data/vocabulary.json',
		write_fname = 'figs/vocab_specificity.pdf'
	):

	# open files
	food_data = json.loads(open(read_food_fname).read())

	# make a figure with three subplots
	figWidth = 6.5 / 2.54 	# conversion from PNAS spec in cm to inches
	figHeight = 2.6*figWidth	# a reasonable aspect ratio
	fig = plt.figure(figsize=(figWidth, figHeight))
	gs = gridspec.GridSpec(3, 1)

	width = 0.75
	padding = 0.25

	# plot food data first
	ax1 = plt.subplot(gs[0])
		
	pairs = [
		('task1:obj', 'task1:food'),
		('frame1:obj', 'frame1:food'),
		('echo:obj', 'echo:food'),
		('task2:cult', 'task2:food'),
		('frame2:cult', 'frame2:food'),
	]

	Y_0 = [
		(food_data[p[1]]['fract_food'] - food_data[p[0]]['fract_food']) * 100
		for p in pairs
	]
	Y_err = [
		np.sqrt( food_data[p[0]]['std']**2 + food_data[p[1]]['std']**2 )*100 
		for p in pairs
	]
	X_0 = range(len(Y_0))

	series_0 = ax1.bar(
		X_0, Y_0, width, color='0.25', ecolor='0.75', yerr=Y_err)

	# Fuss with axes window
	padding = 0.25
	xlims = (-padding, len(X_0))
	plt.xlim(xlims)

	# Fuss with axes labelling
	ax1.set_xticks([x + width/2.0 for x in X_0])
	plt.ylim(-13, 13)
	ax1.set_ylabel(r'$\Delta$ references $(\%)$', size=12)
	plt.setp(ax1.get_xticklabels(), visible=False)
	ax1.yaxis.labelpad = 0

	vocab_data = json.loads(open(read_vocab_fname).read())


	# plot vocabulary data accross all experiments
	ax2 = plt.subplot(gs[1])

	comparisons = [
		('task1:food', 'task1:obj'),
		('frame1:food', 'frame1:obj'),
		('echo:food', 'echo:obj'),
		('task2:food', 'task2:cult'),
		('frame2:food', 'frame2:cult')
	]

	short_comparisons = [
		'task1',
		'frame1',
		'echo',
		'task2',
		'frame2',
	]
	
	Y_2 = []
	for treatment1, treatment2 in comparisons:
		vocab_1 = sum(vocab_data[treatment1])
		vocab_2 = sum(vocab_data[treatment2])
		Y_2.append(100*(vocab_1 - vocab_2) / float(vocab_2))

	upper_CIs = []
	for c in short_comparisons:
		fname = 'data/new_data/vocabulary/vocabulary_null_%s.json' % c
		data = json.loads(open(fname).read())
		upper_CIs.append(data['upper_CI'])
	
	X_2 = range(len(Y_2))
	X_2_CI = [x + width/2. for x in X_2]

	series_2 = ax2.bar(X_2, Y_2, width, color='0.25')
	series_3 = ax2.plot(
		X_2_CI, upper_CIs, linestyle='None', marker='*', 
		markeredgecolor='0.75', color='0.75'
	)

	xlims = (-padding, len(X_2) - 1 + width + padding)
	plt.xlim(xlims)

	ax2.set_xticks([x + width/2. for x in X_2])
	plt.setp(ax2.get_xticklabels(), visible=False)
	ylims = plt.ylim()
	plt.ylim(-6, 22)
	ax2.set_ylabel(r'$\Delta$ richness $(\%)$', size=12)
	ax2.yaxis.labelpad = 7.5

	# now plot the specificity data.  First get organized
	specificity_data = json.loads(open(read_specificity_fname).read())
	ax3 = plt.subplot(gs[2])
	width = 0.75
	specificity_keys_labels = [
		('task1', 'task1'),
		('frame1', 'frame1'),
		('echo', 'echo'),
		('task2', 'task2'),
		('frame2', 'frame2'),
	]

	# read the data from the bootstrapped specificity data
	specificity_data = []
	specificity_error_low = []
	specificity_error_high = []
	specificity_error = (specificity_error_low, specificity_error_high)
	for key, label in specificity_keys_labels:
		fh = open('data/new_data/specificity/specificity_%s.json' % key)
		data = json.loads(fh.read())
		mean = 100 * data['mean']
		err_high = abs(100 * data['std'])
		err_low = abs(100 * data['std'])
		specificity_data.append(mean)
		specificity_error_high.append(err_high)
		specificity_error_low.append(err_low)

	# Now actually plot the data
	#Y_3 = [specificity_data[k][0]*100 for k,l in specificity_keys_labels]

	Y_3 = specificity_data
	Y_err = specificity_error

	X_3 = range(len(Y_3))
	series_3 = ax3.bar(
		X_3, Y_3, width, color='0.25', ecolor='0.75', yerr=Y_err)

	# fiddle with axes
	xlims = (-padding, len(X_3) - 1 + width + padding)
	plt.xlim(xlims)
	ylims = plt.ylim()
	plt.ylim(ylims[0], 28)
	ax3.set_ylabel(r'$\Delta$ specialization $(\%)$', size=12)
	xlabels = [l for k,l in specificity_keys_labels]
	ax3.set_xticks([x + width/2. for x in X_3])
	ax3.set_xticklabels(xlabels, rotation=45, size=12,
		horizontalalignment='right')
	ax3.yaxis.labelpad = 7.5

	ax1.text(4.7, 11.5, 'A', va='top', ha='right', size=18, color='0.55')
	ax2.text(4.7, 20.5, 'B', va='top', ha='right', size=18, color='0.55')
	ax3.text(4.7, 26.5, 'C', va='top', ha='right', size=18, color='0.55')
	#ax3.set_yticks([0.01,0.02,0.03])

	# control overall layout.  Save and return.
	plt.draw()
	plt.tight_layout()
	fig.subplots_adjust(hspace=0.02, top=0.99, right=0.98, left=0.24, 
		bottom=0.08)
	fig.savefig(write_fname)


def plot_delta_food(
		read_fname='data/new_data/food.json',
		write_fname='figs/delta_food.pdf'
	):

	# open files
	food_data = json.loads(open(read_fname).read())

	# make a figure with two subplots
	figWidth = 8.7 / 2.54 	# conversion from PNAS spec in cm to inches
	figHeight = 4/5.*figWidth	# a reasonable aspect ratio
	fig = plt.figure(figsize=(figWidth, figHeight))
	gs = gridspec.GridSpec(1, 1)

	width = 0.75

	# plot food data first
	ax1 = plt.subplot(gs[0])
		
	pairs = [
		('task1:obj', 'task1:food'),
		('frame1:obj', 'frame1:food'),
		('echo:obj', 'echo:food'),
		('task2:cult', 'task2:food'),
		('frame2:cult', 'frame2:food'),
	]

	# convert accuracy to priming difference (which is what we want to plot)
	Y_0 = [
		(food_data[p[1]]['fract_food'] - food_data[p[0]]['fract_food']) * 100
		for p in pairs
	]
	Y_err = [
		(food_data[p[0]]['std'] + food_data[p[1]]['std'])*100 
		for p in pairs
	]
	X_0 = range(len(Y_0))

	series_0 = ax1.bar(
		X_0, Y_0, width, color='0.25', ecolor='0.75', yerr=Y_err)

	# adjust the padding, then add a horizontal line to indicate significance
	padding = 0.25
	xlims = (-padding, len(X_0))
	plt.xlim(xlims)

	ax1.set_ylabel(r'$\Delta$ % food labels', size=12)

	xlabels = [
		r'$task1$',
		r'$frame1$',
		r'$echo$',
		r'$task2$',
		r'$frame2$', 
	]

	ax1.set_xticks([x + width/2.0 for x in X_0])
	ax1.set_xticklabels(xlabels, rotation=45, size=12,
		horizontalalignment='right')

	plt.draw()
	plt.tight_layout()
	fig.subplots_adjust(top=0.98, right=0.98, left=0.18, 
		bottom=0.22)
	fig.savefig(write_fname)

SUPPLEMENTARY_THETA_FNAMES = [
	'l1.json', 'l1_spell.json', 'l1_nostops_spell.json', 
	'l1_nostops_lem_spell.json', 'l1_split_nostops_lem_spell.json',
	'l1_showpos_split_nostops_lem_spell.json'
]



def plot_theta_supplementary(
		write_fname='theta_sup.pdf'
	):

	# make a figure with two subplots
	figWidth = 6
	figHeight = 8
	fig = plt.figure(figsize=(figWidth, figHeight))
	gs = gridspec.GridSpec(4, 3)
	plot_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']
	width = 0.75
	eps = 0.2

	for i, fname in enumerate(SUPPLEMENTARY_THETA_FNAMES*2):

		# The first set of panels plot naive bayes data, second set plot svm.
		if i >= len(SUPPLEMENTARY_THETA_FNAMES):
			data_dir = 'data/new_data/bound_l1_svm'
		else:
			data_dir = 'data/new_data/bound_l1_naive_bayes'

		# get the data and a new axis for plotting
		path = os.path.join(data_dir, fname)
		data = json.loads(open(path).read())

		if i == 0:
			ax = plt.subplot(gs[i])
			ax0 = ax
		else:
			ax = plt.subplot(gs[i], sharey=ax0)

		# p.lot the data
		plot_theta_panel(ax, data)

		# take care of axis labelling
		if i > 8:
			ax.set_xticks(map(lambda x: x + width/2. + eps, range(5)))
			ax.tick_params(axis='x', colors='0.25')
			xlabels = [
				r'$intertask$', r'$frame$', r'$echo$', '$intertask$', '$frame$'
			]
			ax.set_xticklabels(xlabels, rotation=45, size=12,
				horizontalalignment='right', color='k')
		else:
			plt.setp(ax.get_xticklabels(), visible=False)

		if i % 3 != 0:
			plt.setp(ax.get_yticklabels(), visible=False)
		else:
			ax.set_ylabel(r'$D^-_\mathrm{L1}$', size=12)

		left = 4.7
		height = 59
		plot_label = plot_labels[i]
		ax.text(left, height, plot_label, 
			va='top', ha='right', size=16, color='0.55')

	plt.draw()
	plt.tight_layout()
	fig.subplots_adjust(wspace=0.05, hspace=0.05, top=0.99, right=0.99, 
		left=0.1, bottom=0.1)
	fig.savefig(os.path.join(FIGS_DIR, write_fname))


def plot_theta_panel(ax, data):

	width = 0.75
	this_data = data['aggregates']

	# the img_food_obj test was tried under multiple permutations -- take avg
	this_data['exp2.task'] = np.mean(this_data['exp2.task'])
	test_names = [
		'exp2.task', 'exp2.frame', 'exp2.*', 'exp1.task', 'exp1.frame'
	]
	accuracies = [this_data[tn] for tn in test_names]

	# convert accuracy to priming difference (which is what we want to plot)
	Y_aggregate = [100*(2*a-1) for a in accuracies]

	# construct the confidence intervals
	confidence_intervals = [
		binomial_confidence_intervals(n, int(round(n*a)), alpha=0.15865, as_theta=True)
		for n,a in zip([1190,238,238,238,238], accuracies)
	]

	err_low = [
		y - 100*c[1] for y,c in zip(Y_aggregate, confidence_intervals)
	]
	err_high = [
		100*c[0] - y for y,c in zip(Y_aggregate, confidence_intervals)
	]
	err = [err_low, err_high]

	X = range(len(Y_aggregate))

	series = ax.bar(
		X, Y_aggregate, width, color='0.25', ecolor='0.75', yerr=err
	)

	padding = 0.25
	xlims = (-padding, len(X) - 1 + width + padding)
	plt.xlim(xlims)

	ax.set_yticks([10,20,30,40,50,60])

	ylims = (0, 62)
	plt.ylim(ylims)



def plot_theta(
		read_fname='l1.json',
		write_fname='theta.pdf'
	):

	# open files
	data = json.loads(open(os.path.join(DATA_DIR, read_fname)).read())

	# make a figure with two subplots
	figWidth = 10.0 / 2.54 	# conversion from PNAS spec in cm to inches
	figHeight = 3/5.*figWidth	# a reasonable aspect ratio
	#figWidth = 14.78 / 2.54 	# conversion from PNAS spec in cm to inches
	#figHeight = 2/5.*figWidth	# a reasonable aspect ratio
	fig = plt.figure(figsize=(figWidth, figHeight))
	gs = gridspec.GridSpec(1, 2)
	#gs = gridspec.GridSpec(1, 3, width_ratios=(5,5,5))

	width = 0.75

	# Plot the aggregate data series
	ax1 = plt.subplot(gs[0])
	this_data = data['aggregates']

	# the img_food_obj test was tried under multiple permutations -- take avg
	this_data['exp2.task'] = np.mean(this_data['exp2.task'])

	test_names = ['exp2.task', 'exp2.frame', 'exp2.*', 'exp1.task', 
			'exp1.frame']
	accuracies = [this_data[tn] for tn in test_names]

	# convert accuracy to priming difference (which is what we want to plot)
	Y_aggregate = [100*(2*a-1) for a in accuracies]

	confidence_intervals = [
		binomial_confidence_intervals(n, int(round(n*a)), alpha=0.15865, as_theta=True)
		for n,a in zip([1190,238,238,238,238], accuracies)
	]

	err_low = [
		y - 100*c[1] for y,c in zip(Y_aggregate, confidence_intervals)
	]
	err_high = [
		100*c[0] - y for y,c in zip(Y_aggregate, confidence_intervals)
	]
	err = [err_low, err_high]

	X = range(len(Y_aggregate))

	series = ax1.bar(
		X, Y_aggregate, width, color='0.25', ecolor='0.75', yerr=err)

	# adjust the padding, then add a horizontal line to indicate significance
	padding = 0.25
	xlims = (-padding, len(X) - 1 + width + padding)
	plt.xlim(xlims)
	#theta_star = get_theta_star(119, 0.05)
	#singificance_line = ax1.plot(
	#		xlims, [theta_star, theta_star], color='0.55', linestyle=':')

	ax1.set_ylabel(r'$\theta_\mathrm{NB}\;(\%)$', size=12)

	xlabels = [
		r'$intertask$', r'$frame$', r'$echo$', '$intertask$', '$frame$'
	]

	eps = 0.2
	ax1.set_xticks(map(lambda x: x + width/2. + eps, X))
	ax1.tick_params(axis='x', colors='0.25')
	ax1.set_xticklabels(xlabels, rotation=45, size=10,
		horizontalalignment='right', color='k')

	# Plot the by_pos data series
	ax2 = plt.subplot(gs[1], sharey=ax1)
	plt.setp(ax2.get_yticklabels(), visible=False)
	this_data = data['exp2.task']
	avg_accuracies = [
		np.mean([this_data[img][i] for img in IMAGE_NAMES]) 
		for i in range(5)
	]

	# convert accuracy to priming difference (which is what we want to plot)
	Y_by_pos = [100*(2*a-1) for a in avg_accuracies]
	X = range(len(Y_by_pos))

	err_low = [
		100*(2*binomial_lower_confidence_p(595, int(595*a),alpha=0.15865) - 1)
		for a in avg_accuracies
	]
	err_low = [y-y_err for y, y_err in zip(Y_by_pos, err_low)]
	#err_high = [0 for e in err_low]
	err_high = err_low
	err = [err_low, err_high]

	series = ax2.bar(
		X, Y_by_pos, width, color='0.25', ecolor='0.75', yerr=err)

	# adjust the padding, then add a horizontal line to indicate significance
	padding = 0.25
	xlims = (-padding, len(X) - 1 + width + padding)
	plt.xlim(xlims)

	xlabels = ['1', '2', '3', '4', '5']

	ax2.set_xticks(map(lambda x: x + width/2., X))
	ax2.tick_params(axis='x', colors='0.25')
	ax2.set_xticklabels(xlabels, rotation=0, size=12,
		horizontalalignment='center', color='k')
	ax2.set_xlabel(r'test task position', size=8)

#	# Plot the by_img data series
#	ax3 = plt.subplot(gs[2], sharey=ax1)
#	plt.setp(ax3.get_yticklabels(), visible=False)
#
#	this_data = data['exp2.task']
#	avg_accuracies = [np.mean(this_data[img]) for img in IMAGE_NAMES]
#
#	# convert accuracy to priming difference (which is what we want to plot)
#	Y_by_img = [2*a-1 for a in avg_accuracies]
#	err_low = [
#		2*binomial_lower_confidence_p(595, int(595*a)) - 1
#		for a in avg_accuracies
#	]
#	err_low = [y-y_err for y, y_err in zip(Y_by_img, err_low)]
#	err_high = [0 for e in err_low]
#	err = [err_low, err_high]
#
#	series = ax3.bar(X, Y_by_img, width, color='0.25', ecolor='0.85', yerr=err)
#
#	# adjust the padding, then add a horizontal line to indicate significance
#	padding = 0.25
#	xlims = (-padding, len(X) - 1 + width + padding)
#	plt.xlim(xlims)
#
#
#	xlabels = ['image %d'%i for i in range(1,6)]
#
#	ax3.set_xticks(map(lambda x: x + width/2., X))
#	ax3.set_xticklabels(xlabels, rotation=45, size=12,
#		horizontalalignment='right')

	ax1.set_yticks(range(0,60,10))
	ax2.set_yticks(range(0,60,10))
#	ax3.set_yticks([0.1,0.2,0.3,0.4,0.5])

	ylims = (0, 55)
	plt.ylim(ylims)

	left = 4.7
	height = 52
	#for label, ax in zip(['A', 'B', 'C',], [ax1, ax2, ax3]):
	for label, ax in zip(['A', 'B'], [ax1, ax2]):
		ax.text(left, height, label, 
			va='top', ha='right', size=18, color='0.55')

	plt.draw()
	plt.tight_layout()
	fig.subplots_adjust(wspace=0.05, top=0.99, right=0.99, left=0.15, 
		bottom=0.29)
	fig.savefig(os.path.join(FIGS_DIR, write_fname))


# ****** End of trusted plotters


def plotAllF1Theta(
		readFname='data/new_data/l1.json',
		writeFname='figs/l1_longitudinal.pdf',
		n=125,
		alpha=0.05
	):

	'''
	Plots the theta value for a naive bayes classifier built to distinguish
	between all the interesting pairings of treatments.  See the manuscript
	in <project-root>/docs/drafts for an explanation of theta.

	This function only plots; the data must first be generated by running
	`computeAllF1Accuracy()`
	'''

	subplotLabels = ['A','B','C', 'D', 'E', 'F', 'G']

	# Read the data from file
	f1scores = json.loads(open(readFname, 'r').read())['img_food_obj']

	# Start a figure 
	figWidth = 8.7 / 2.54 	# conversion from PNAS spec in cm to inches
	figHeight = figWidth * 3.	# a reasonable aspect ratio
	fig = plt.figure(figsize=(figWidth, figHeight))

	# calculate the gridspec.  The width ratios are based on the width of
	# each bar-plot-group, spacing between them, and plot padding
	num_subplots = len(f1scores)
	gs = gridspec.GridSpec(num_subplots, 1)

	width = 0.75
	theta_star = get_theta_star(n, alpha)

	for i in range(len(f1scores)):

		image_id = 'test%d' % i

		# Unpack the data for this subplot
		Y_F1s = f1scores[image_id]
		X_F1s = range(len(Y_F1s))

		# Convert from accuracy to theta
		Y_thetas = map(lambda t: t*2 - 1, f1scores[image_id])
		X_thetas = map(lambda x: x+width, X_F1s)

		# Make a set of axes.  
		# Manage axis-sharing directives, and whether ytick-labels are visible
		if i == 0:
			ax = plt.subplot(gs[i])
			ax0 = ax
		else:
			ax = plt.subplot(gs[i], sharey=ax0)
			plt.setp(ax.get_yticklabels(), visible=False)


		ax.set_ylabel(r'$\hat{\theta}_{NB}$', size=9)
		theta_series = ax.bar(X_F1s, Y_thetas, width, color='0.25')

		padding = 0.25
		xlims = (-padding, len(Y_thetas) - 1 + width + padding)
		plt.xlim(xlims)

		# Put together intelligible labels for the x-axis
		ax.tick_params(axis='both', which='major', labelsize=9)
		xlabels = [str(i) for i in range(len(f1scores[image_id]))]
		ax.set_xticks(map(lambda x: x + width/2., X_F1s))
		ax.set_xticklabels(xlabels, rotation=45, size=9,
			horizontalalignment='right')

		zero = ax.plot(
			xlims, [0, 0], color='0.35', linestyle='-', zorder=0)

		significance_bar = ax.plot(
			xlims, [theta_star, theta_star], color='0.55', linestyle=':')

		# Tighten up the layout
		plt.draw()
		if i < 1:
			plt.tight_layout()

	# After plots are made, put labels along the top.  This needs to wait
	# until now so that the axes limits are stable
	for i, (ax, subplotData) in enumerate(zip(fig.axes, f1scores)):

		pass
		#** put the test task name as an inset

#		TREATMENT_NAMES = {
#			'treatment0': 'AMBG'
#			, 'treatment1': 'CULT$_{img}$'
#			, 'treatment2': 'INGR$_{img}$'
#			, 'treatment3': 'INGR$_{fund}$'
#			, 'treatment4': 'INGR$_{fund,img}$'
#			, 'treatment5': 'CULT$_{fund}$'
#			, 'treatment6': 'CULT$_{fund,img}$'
#		}

#		# Label the basis treatments above the subplots
#		basisTreatment = subplotData['basis']
#		basisTreatmentName = TREATMENT_NAMES[basisTreatment]
#		left = len(subplotData['accuracy'])/2.0 + width - 2*padding
#		ylims = plt.ylim()
#
#		# put the label directly above the plot.  
#		# The first label needs to be put a bit higher.
#		height= ylims[1] + (0.02 if i else 0.06)
#		ax.text(left, height, basisTreatmentName, 
#				va='bottom', ha='right', size=9, rotation=-45)


	#y_low, y_high = plt.ylim()
	#plt.ylim(-0.09, y_high)

	fig.subplots_adjust(wspace=0.05, top=0.77, right=0.97, left=0.09, 
		bottom=0.24)
	plt.draw()

	fig.savefig(writeFname)
	plt.show()
# This is an old version that uses two bars to compare each treatment pair
def plot_food_proportions(
		read_food_fname='data/new_data/food.json',
		write_fname='figs/food_proportions.pdf'
	):

	# open files
	food_data = json.loads(open(read_food_fname).read())

	# make a figure with two subplots
	figWidth = 16.78 / 2.54 	# conversion from PNAS spec in cm to inches
	figHeight = 3/5.*figWidth	# a reasonable aspect ratio
	fig = plt.figure(figsize=(figWidth, figHeight))
	gs = gridspec.GridSpec(1, 2)

	width = 0.325


	# plot food data first
	ax1 = plt.subplot(gs[0])
		
	pairs = [
		('task1:obj', 'task1:food'),
		('frame1:obj', 'frame1:food'),
		('echo:obj', 'echo:food'),
		('task2:cult', 'task2:food'),
		('frame2:cult', 'frame2:food'),
	]

	# convert accuracy to priming difference (which is what we want to plot)
	Y_0 = [food_data[p[0]]['fract_food'] for p in pairs]
	Y_1 = [food_data[p[1]]['fract_food'] for p in pairs]
	X_0 = range(len(Y_0))
	X_1 = [x+width for x in X_0]

	series_0 = ax1.bar(X_0, Y_0, width, color='0.25')
	series_1 = ax1.bar(X_1, Y_1, width, color='0.55')

	# adjust the padding, then add a horizontal line to indicate significance
	padding = 0.25
	xlims = (-padding, len(X_1) - 1 + 2*width + padding)
	plt.xlim(xlims)

	ax1.set_ylabel(r'fraction food labels', size=12)

	xlabels = [
		r'$task1$',
		r'$frame1$',
		r'$echo$',
		r'$task2$',
		r'$frame2$', 
	]

	ax1.set_xticks(X_1)
	ax1.set_xticklabels(xlabels, rotation=45, size=12,
		horizontalalignment='right')
	ax1.set_yticks([0.2,0.4,0.6])

	# add a legend
	label_objects = [series_0[0], series_1[0]]
	y_min, y_max = plt.ylim()
	x_min, x_max = plt.xlim()
	ax1.legend(
		label_objects, 
		['non-food priming', 'food priming'], 
		loc=3,
		mode='expand',
		borderaxespad=0.,
		borderpad=0.6,
		prop={'size':11},
		bbox_to_anchor=(0., 1.02, 1., 0.102)
	)   

	plt.draw()
	plt.tight_layout()
	fig.subplots_adjust(wspace=0.35, top=0.82, right=0.99, left=0.10, 
		bottom=0.20)
	fig.savefig(write_fname)



def plot_theta_by_img(
		read_fname='l1.json',
		write_fname='theta_by_img.pdf'
	):

	# open files
	data = json.loads(open(os.path.join(DATA_DIR, read_fname)).read())
	data = data['img_food_obj']

	avg_accuracies = [np.mean(data[img]) for img in IMAGE_NAMES]

	# convert accuracy to priming difference (which is what we want to plot)
	Y = [2*a-1 for a in avg_accuracies]
	X = range(len(Y))
	Y 

	figWidth = 8.7 / 2.54 	# conversion from PNAS spec in cm to inches
	figHeight = figWidth	# a reasonable aspect ratio
	fig = plt.figure(figsize=(figWidth, figHeight))
	gs = gridspec.GridSpec(1, 1)

	width = 0.75

	# make a set of axes, then plot the data
	ax = plt.subplot(gs[0])
	series = ax.bar(X, Y, width, color='0.25')

	# adjust the padding, then add a horizontal line to indicate significance
	padding = 0.25
	xlims = (-padding, len(X) - 1 + width + padding)
	plt.xlim(xlims)
	theta_star = get_theta_star(119*5, 0.05)
	singificance_line = ax.plot(
			xlims, [theta_star, theta_star], color='0.55', linestyle=':')

	ax.set_ylabel(r'$\hat{\theta}_\mathrm{NB}$', size=12)

	xlabels = IMAGE_NAMES

	ax.set_xticks(map(lambda x: x + width/2., X))
	ax.set_xticklabels(xlabels, rotation=45, size=12,
		horizontalalignment='right')

	ax.set_yticks([0.1,0.2,0.3,0.4])
	fig.subplots_adjust(wspace=0.05, top=0.95, right=0.95, left=0.20, 
		bottom=0.20)

	plt.draw()
	fig.savefig(os.path.join(FIGS_DIR, write_fname))


def plot_theta_by_pos(
		read_fname='l1.json',
		write_fname='theta_by_pos.pdf'
	):

	# open files
	data = json.loads(open(os.path.join(DATA_DIR, read_fname)).read())
	data = data['img_food_obj']

	i = 0
	avg_accuracies = [
		np.mean([data[img][i] for img in IMAGE_NAMES]) 
		for i in range(5)
	]

	# convert accuracy to priming difference (which is what we want to plot)
	Y = [2*a-1 for a in avg_accuracies]
	X = range(len(Y))

	figWidth = 8.7 / 2.54 	# conversion from PNAS spec in cm to inches
	figHeight = figWidth	# a reasonable aspect ratio
	fig = plt.figure(figsize=(figWidth, figHeight))
	gs = gridspec.GridSpec(1, 1)

	width = 0.75

	# make a set of axes, then plot the data
	ax = plt.subplot(gs[0])
	series = ax.bar(X, Y, width, color='0.25')

	# adjust the padding, then add a horizontal line to indicate significance
	padding = 0.25
	xlims = (-padding, len(X) - 1 + width + padding)
	plt.xlim(xlims)
	theta_star = get_theta_star(119*5, 0.05)
	singificance_line = ax.plot(
			xlims, [theta_star, theta_star], color='0.55', linestyle=':')

	ax.set_ylabel(r'$\hat{\theta}_\mathrm{NB}$', size=12)

	xlabels = ['1st', '2nd', '3rd', '4th', '5th']

	ax.set_xticks(map(lambda x: x + width/2., X))
	ax.set_xticklabels(xlabels, rotation=45, size=12,
		horizontalalignment='right')

	ax.set_yticks([0.1,0.2,0.3,0.4])
	fig.subplots_adjust(wspace=0.05, top=0.95, right=0.95, left=0.20, 
		bottom=0.20)

	plt.draw()
	fig.savefig(os.path.join(FIGS_DIR, write_fname))



def plot_theta_aggregate(
		read_fname='l1.json',
		write_fname='theta_aggregate.pdf'
	):

	# open files
	data = json.loads(open(os.path.join(DATA_DIR, read_fname)).read())
	data = data['aggregates']
		
	# the img_food_obj test was tried under multiple permutations -- take avg
	data['img_food_obj'] = np.mean(data['img_food_obj'])

	TEST_NAMES = [
		'img_food_cult',
		'img_food_obj',
		'wfrm_food_cult',
		'wfrm_food_obj',
		'sfrm_food_obj'
	]

	accuracies = [data[tn] for tn in TEST_NAMES]

	# convert accuracy to priming difference (which is what we want to plot)
	Y = [2*a-1 for a in accuracies]
	X = range(len(Y))


	figWidth = 8.7 / 2.54 	# conversion from PNAS spec in cm to inches
	figHeight = figWidth	# a reasonable aspect ratio
	fig = plt.figure(figsize=(figWidth, figHeight))
	gs = gridspec.GridSpec(1, 1)

	width = 0.75
	theta_star = get_theta_star(119, 0.05)

	# make a set of axes, then plot the data
	ax = plt.subplot(gs[0])
	series = ax.bar(X, Y, width, color='0.25')

	# adjust the padding, then add a horizontal line to indicate significance
	padding = 0.25
	xlims = (-padding, len(X) - 1 + width + padding)
	plt.xlim(xlims)
	singificance_line = ax.plot(
			xlims, [theta_star, theta_star], color='0.55', linestyle=':')

	
	ax.set_ylabel(r'$\hat{\theta}_\mathrm{NB}$', size=12)

	EXPERIMENT_NAMES = {
		'img_food_cult': 'inter-t. 1',
		'img_food_obj': 'inter-t. 2',
		'wfrm_food_cult': 'frame 1',
		'wfrm_food_obj': 'frame 2',
		'sfrm_food_obj': 'frame 3'
	}
	xlabels = [EXPERIMENT_NAMES[tn] for tn in TEST_NAMES]

	ax.set_xticks(map(lambda x: x + width/2., X))
	ax.set_xticklabels(xlabels, rotation=45, size=12,
		horizontalalignment='right')

	fig.subplots_adjust(wspace=0.05, top=0.95, right=0.95, left=0.20, 
		bottom=0.20)

	plt.draw()
	fig.savefig(os.path.join(FIGS_DIR, write_fname))


def plot_self_specificity(
		read_fname = 'data/new_data/self_specificity.json',
		write_fname = 'figs/self_specificity.pdf'
	):

	self_specificity = json.loads(open(read_fname).read())

	# make a figure with three subplots
	figWidth = 16.78 / 2.54 	# conversion from PNAS spec in cm to inches
	figHeight = 4/5.*figWidth	# a reasonable aspect ratio
	fig = plt.figure(figsize=(figWidth, figHeight))
	gs = gridspec.GridSpec(1, 2)

	width = 0.75

	# now plot the vocabulary data
	ax1 = plt.subplot(gs[0])
	Y_food = np.mean(self_specificity['food'], 1)
	X = range(len(Y_food))
	series_1 = ax1.bar(X, Y_food, width, color='0.25')

	ax1 = plt.subplot(gs[1])
	Y_object = np.mean(self_specificity['object'], 1)
	series_2 = ax1.bar(X, Y_object, width, color='0.25')

