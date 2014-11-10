'''
This module provides a collection of functions that produce the figures used
in the paper
'''

import json
import util
import analysis
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

except:
	print 'matplotlib not installed.  You can do computations but not '\
		'plotting functions.'
	

# z-score for two-tailed 99-percent confidence interval
CONFIDENCE_95 = 1.96
CONFIDENCE_99 = 2.975


TREATMENT_NAMES = {
	'treatment0': 'AMBG'
	, 'treatment1': 'CULT$_{img}$'
	, 'treatment2': 'INGR$_{img}$'
	, 'treatment3': 'INGR$_{fund}$'
	, 'treatment4': 'INGR$_{fund,img}$'
	, 'treatment5': 'CULT$_{fund}$'
	, 'treatment6': 'CULT$_{fund,img}$'
}

TEST_NAMES = [
	'img_food_cult',
	'img_food_obj',
	'wfrm_food_cult',
	'wfrm_food_obj',
	'sfrm_food_obj'
]

EXPERIMENT_NAMES = {
	'img_food_cult': 'inter-t. 1',
	'img_food_obj': 'inter-t. 2',
	'wfrm_food_cult': 'frame 1',
	'wfrm_food_obj': 'frame 2',
	'sfrm_food_obj': 'frame 3'
}

IMAGE_NAMES = ['test%d' %i for i in range(5)]

DATA_DIR = 'data/new_data'
FIGS_DIR = 'figs/'


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



def plot_vocab_specificity(
		read_specificity_fnames = ('data/new_data/specificity_alt.json',
			'data/new_data/specificity_ignore_food.json'),
		read_vocab_fname = 'data/new_data/vocabulary.json',
		write_fname = 'figs/vocab_specificity.pdf'
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


	# plot vocabulary data accross all experiments
	ax2 = plt.subplot(gs[1], sharey=ax1)

	comparisons = [
		('exp1.task.food', 'exp1.task.cult'),
		('exp1.frame.food', 'exp1.frame.cult'),
		('exp2.task.food', 'exp2.task.obj'),
		('exp2.frame.food', 'exp2.frame.obj'),
		('exp2.frame*.food', 'exp2.frame*.obj')
	]

	Y_2 = []
	for treatment1, treatment2 in comparisons:
		vocab_1 = sum(vocab_data[treatment1])
		vocab_2 = sum(vocab_data[treatment2])
		Y_2.append((vocab_1 - vocab_2) / float(vocab_2))
	
	X_2 = range(len(Y_2))

	series_2 = ax2.bar(X_2, Y_2, width, color='0.25')

	xlims = (-padding, len(X_2) - 1 + width + padding)
	plt.xlim(xlims)

	plt.setp(ax2.get_yticklabels(), visible=False)

	xlabels = [
		'$exp1.task$',
		'$exp1.frame$',
		'$exp2.task$',
		'$exp2.frame$',
		'$exp2.frame*$',
	]

	ax2.set_xticks([x + width/2. for x in X_1])
	ax2.set_xticklabels(xlabels, rotation=45, size=12,
		horizontalalignment='right')
	ax2.set_yticks([0.1,0.2,0.3,0.4])

	# original method for plotting vocabulary size
	#ax3 = plt.subplot(gs[2])
	#width = 0.75


	#comparisons = [
	#	(False, 0,1),
	#	(False, 3,5),
	#	(True, 0,5),
	#	(True, 10,11),
	#	(True, 12,13)
	#]

	#Y_3 = []
	#for exp,treatment1,treatment2 in comparisons:
	#	d = data_processing.readDataset(exp)
	#	treatment1 = 'treatment%d' % treatment1
	#	treatment2 = 'treatment%d' % treatment2
	#	vocab_0 = 0
	#	vocab_1 = 0
	#	for image in ['test%d' % i for i in range(5)]:
	#		vocab_0 += len(
	#			d.get_counts_for_treatment_image(treatment1, image))
	#		vocab_1 += len(
	#			d.get_counts_for_treatment_image(treatment2, image))
	#	Y_3.append((vocab_0 - vocab_1) / float(vocab_1))

	#X_3 = range(len(Y_3))

	#series_3 = ax3.bar(X_3, Y_3, width, color='0.25')

	#xlims = (-padding, len(X_3) - 1 + width + padding)
	#plt.xlim(xlims)

	#ax3.set_ylabel(r'relative increase in vocabulary', size=12)

	#xlabels = ['image %d' % i for i in range(1,6)]

	#ax3.set_xticks([x + width/2. for x in X_3])
	#ax3.set_xticklabels(xlabels, rotation=45, size=12,
	#	horizontalalignment='right')
	#ax3.set_yticks([0.1,0.2,0.3,0.4])






	# now plot the specificity data
	specificity_data = json.loads(open(read_specificity_fnames[0]).read())
	specificity_data_no_food = json.loads(
		open(read_specificity_fnames[1]).read())
	ax3 = plt.subplot(gs[2])
	width = 0.75
	specificity_keys_labels = [
			('img_food_cult', r'$exp1.task$'),
			('wfrm_food_cult', r'$exp1.frame$'),
			('img_food_obj', r'$exp2.task$'),
			('wfrm_food_obj', r'$exp2.frame$'),
			('sfrm_food_obj', r'$exp2.frame*$')
	]

	Y_3 = [np.mean(specificity_data[k]) for k,l in specificity_keys_labels]
	X_3 = range(len(Y_3))
	Y_4 = [
		np.mean(specificity_data_no_food[k]) 
		for k,l in specificity_keys_labels
	]


	series_3 = ax3.bar(X_3, Y_3, width, color='0.55')
	series_4 = ax3.bar(X_3, Y_4, width, color='0.25')

	xlims = (-padding, len(X_3) - 1 + width + padding)
	plt.xlim(xlims)

	ax3.set_ylabel(r'relative specificity', size=12)

	xlabels = [l for k,l in specificity_keys_labels]

	# now plot specificity data excluding food tokens




	ax3.set_xticks([x + width/2. for x in X_3])
	ax3.set_xticklabels(xlabels, rotation=45, size=12,
		horizontalalignment='right')
	ax3.set_yticks([0.01,0.02,0.03])


	plt.draw()
	plt.tight_layout()
	fig.subplots_adjust(wspace=0.35, top=0.82, right=0.99, left=0.10, 
		bottom=0.20)
	fig.savefig(write_fname)


def plot_food_specificity(
		read_food_fname='data/new_data/food_alt.json',
		read_specificity_fname='data/new_data/specificity.json',
		write_fname='figs/food_specificity.pdf'
	):

	# open files
	food_data = json.loads(open(read_food_fname).read())
	specificity_data = json.loads(open(read_specificity_fname).read())

	# make a figure with two subplots
	figWidth = 16.78 / 2.54 	# conversion from PNAS spec in cm to inches
	figHeight = 3/5.*figWidth	# a reasonable aspect ratio
	fig = plt.figure(figsize=(figWidth, figHeight))
	gs = gridspec.GridSpec(1, 2)

	width = 0.325


	# plot food data first
	ax1 = plt.subplot(gs[0])
		
	pairs = [
		('2_img_obj', '2_img_food'),
		('1_img_cult', '1_img_food'),
		('2_wfrm_obj', '2_wfrm_food'),
		('1_wfrm_cult', '1_wfrm_food'),
		('2_sfrm_obj', '2_sfrm_food'),
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
		r'$inter$-$t.1$',
		r'$inter$-$t.2$',
		r'$frame1$',
		r'$frame2$', 
		r'$frame$*$3$'
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

	# now plot the specificity data
	ax2 = plt.subplot(gs[1])
	width = 0.75
	d = data_processing.readDataset(False)

	Y_3 = []
	for image in ['test%d' % i for i in range(5)]:
		vocab_0 = len(d.get_counts_for_treatment_image('treatment0', image))
		vocab_1 = len(d.get_counts_for_treatment_image('treatment1', image))
		Y_3.append((vocab_0 - vocab_1) / float(vocab_1))

	X_3 = range(len(Y_3))

	series_3 = ax2.bar(X_3, Y_3, width, color='0.25')

	xlims = (-padding, len(X_3) - 1 + width + padding)
	plt.xlim(xlims)

	ax2.set_ylabel(r'relative increase in vocabulary', size=12)

	xlabels = ['image %d' % i for i in range(1,6)]

	ax2.set_xticks([x + width/2. for x in X_3])
	ax2.set_xticklabels(xlabels, rotation=45, size=12,
		horizontalalignment='right')
	ax2.set_yticks([0.1,0.2,0.3,0.4])

	plt.draw()
	plt.tight_layout()
	fig.subplots_adjust(wspace=0.35, top=0.82, right=0.99, left=0.10, 
		bottom=0.20)
	fig.savefig(write_fname)




def plot_theta(
		read_fname='l1.json',
		write_fname='theta.pdf'
	):

	# open files
	data = json.loads(open(os.path.join(DATA_DIR, read_fname)).read())



	# make a figure with two subplots
	figWidth = 14.78 / 2.54 	# conversion from PNAS spec in cm to inches
	figHeight = 2/5.*figWidth	# a reasonable aspect ratio
	fig = plt.figure(figsize=(figWidth, figHeight))
	gs = gridspec.GridSpec(1, 3)

	width = 0.75



	# Plot the aggregate data series

	ax1 = plt.subplot(gs[0])
	this_data = data['aggregates']
		
	# the img_food_obj test was tried under multiple permutations -- take avg
	this_data['img_food_obj'] = np.mean(this_data['img_food_obj'])

	accuracies = [this_data[tn] for tn in TEST_NAMES]

	# convert accuracy to priming difference (which is what we want to plot)
	Y_aggregate = [2*a-1 for a in accuracies]
	X = range(len(Y_aggregate))

	series = ax1.bar(X, Y_aggregate, width, color='0.25')

	# adjust the padding, then add a horizontal line to indicate significance
	padding = 0.25
	xlims = (-padding, len(X) - 1 + width + padding)
	plt.xlim(xlims)
	theta_star = analysis.get_theta_star(119, 0.05)
	singificance_line = ax1.plot(
			xlims, [theta_star, theta_star], color='0.55', linestyle=':')

	ax1.set_ylabel(r'$\hat{\theta}_\mathrm{NB}$', size=12)

	xlabels = [r'$inter$-$t.1$', r'$inter$-$t.2$', r'$frame1$', r'$frame2$', 
			r'$frame$*$3$']

	ax1.set_xticks(map(lambda x: x + width/2., X))
	ax1.set_xticklabels(xlabels, rotation=45, size=12,
		horizontalalignment='right')




	# Plot the by_pos data series
	ax2 = plt.subplot(gs[1], sharey=ax1)
	plt.setp(ax2.get_yticklabels(), visible=False)
	this_data = data['img_food_obj']
	avg_accuracies = [
		np.mean([this_data[img][i] for img in IMAGE_NAMES]) 
		for i in range(5)
	]

	# convert accuracy to priming difference (which is what we want to plot)
	Y_by_pos = [2*a-1 for a in avg_accuracies]
	X = range(len(Y_by_pos))

	series = ax2.bar(X, Y_by_pos, width, color='0.25')

	# adjust the padding, then add a horizontal line to indicate significance
	padding = 0.25
	xlims = (-padding, len(X) - 1 + width + padding)
	plt.xlim(xlims)
	theta_star = analysis.get_theta_star(119*5, 0.05)
	singificance_line = ax2.plot(
			xlims, [theta_star, theta_star], color='0.55', linestyle=':')

	xlabels = ['1st', '2nd', '3rd', '4th', '5th']

	ax2.set_xticks(map(lambda x: x + width/2., X))
	ax2.set_xticklabels(xlabels, rotation=45, size=12,
		horizontalalignment='right')





	# Plot the by_img data series
	ax3 = plt.subplot(gs[2], sharey=ax1)
	plt.setp(ax3.get_yticklabels(), visible=False)

	this_data = data['img_food_obj']
	avg_accuracies = [np.mean(this_data[img]) for img in IMAGE_NAMES]

	# convert accuracy to priming difference (which is what we want to plot)
	Y_by_img = [2*a-1 for a in avg_accuracies]
	series = ax3.bar(X, Y_by_img, width, color='0.25')

	# adjust the padding, then add a horizontal line to indicate significance
	padding = 0.25
	xlims = (-padding, len(X) - 1 + width + padding)
	plt.xlim(xlims)

	singificance_line = ax3.plot(
			xlims, [theta_star, theta_star], color='0.55', linestyle=':')

	xlabels = ['image %d'%i for i in range(1,6)]

	ax3.set_xticks(map(lambda x: x + width/2., X))
	ax3.set_xticklabels(xlabels, rotation=45, size=12,
		horizontalalignment='right')




	ax1.set_yticks([0.1,0.2,0.3,0.4])
	ax2.set_yticks([0.1,0.2,0.3,0.4])
	ax3.set_yticks([0.1,0.2,0.3,0.4])

	left = 4.7
	height = 0.43
	for label, ax in zip(['A', 'B', 'C',], [ax1, ax2, ax3]):
		ax.text(left, height, label, 
			va='top', ha='right', size=18, color='0.55')


	plt.draw()
	plt.tight_layout()
	fig.subplots_adjust(wspace=0.05, top=0.99, right=0.99, left=0.11, 
		bottom=0.29)
	fig.savefig(os.path.join(FIGS_DIR, write_fname))


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
	theta_star = analysis.get_theta_star(119*5, 0.05)
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
	theta_star = analysis.get_theta_star(119*5, 0.05)
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

	accuracies = [data[tn] for tn in TEST_NAMES]

	# convert accuracy to priming difference (which is what we want to plot)
	Y = [2*a-1 for a in accuracies]
	X = range(len(Y))


	figWidth = 8.7 / 2.54 	# conversion from PNAS spec in cm to inches
	figHeight = figWidth	# a reasonable aspect ratio
	fig = plt.figure(figsize=(figWidth, figHeight))
	gs = gridspec.GridSpec(1, 1)

	width = 0.75
	theta_star = analysis.get_theta_star(119, 0.05)

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

	xlabels = [EXPERIMENT_NAMES[tn] for tn in TEST_NAMES]

	ax.set_xticks(map(lambda x: x + width/2., X))
	ax.set_xticklabels(xlabels, rotation=45, size=12,
		horizontalalignment='right')

	fig.subplots_adjust(wspace=0.05, top=0.95, right=0.95, left=0.20, 
		bottom=0.20)

	plt.draw()
	fig.savefig(os.path.join(FIGS_DIR, write_fname))


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
	theta_star = analysis.get_theta_star(n, alpha)

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
