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

try:
	import matplotlib.pyplot as plt
	import matplotlib 
	import matplotlib.gridspec as gridspec
	from matplotlib.ticker import FixedLocator, LinearLocator, FixedFormatter

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

def plotSpecificityLongitudinal(
		readFnames=[
			'specificity/t0-t1-test%d-overall.json' % i for i in range(5)],
		writeFname='figs/specificity-longitudinal.pdf'
	):

	'''
	Plots the relative specificiy of ambg <-> cult_img for each of the 
	test images.  Comparison uses full ontology (is not restricted to 'food' 
	or 'culture').

	Later, this should get added on to the other longitudinal plots.
	'''

	# we'll open up each file, (there is one per image), and pull out the
	# data that we want.  Assert statements make sure that some assumptions
	# that are built into how we're picking data are met
	pick_data = []
	for fname in readFnames:
		data = json.loads(open(fname, 'r').read())
		data = data[0]
		assert(data['valence'] == 'overall')
		data = data['results'][0]
		assert(data['basis'] == 'treatment0')
		data = data['results'][0]
		assert(data['subject'] == 'treatment1')
		pick_data.append(data)

	Y = [pd['avg'] for pd in pick_data]
	Y_err = [pd['stdev']*CONFIDENCE_95 for pd in pick_data]
	X = range(len(Y))

	width=0.75
	figWidth = 8.7 / 2.54 			# convert from PNAS spec in cm to inches
	figHeight = (4/5.)*figWidth 	# a reasonable aspect ratio
	fig = plt.figure(figsize=(figWidth,figHeight))
	gs = gridspec.GridSpec(1,1)

	ax = plt.subplot(gs[0])
	series = ax.bar(X, Y, width, color='0.25', ecolor='0.55', 
		yerr=Y_err)


	# control the plot X limits
	padding = 0.25
	xlims = (-padding, len(Y) - 1 + width + padding)
	plt.xlim(xlims)

	# Annotate the plot with a line at Y=0
	zero = ax.plot(
		xlims, [0, 0], color='0.35', linestyle='-', zorder=0)

	
	# handle axis labelling
	xlabels = ['image%d' %i for i in range(1,6)]
	ax.tick_params(axis='both', which='major', labelsize=9)
	ax.set_xticks([x + width/2. for x in  X])
	ax.set_xticklabels(xlabels, rotation=45, horizontalalignment='right')
	ax.set_ylabel("relative specificity", size=9)

	plt.draw()
	plt.tight_layout()

	# Post plot adjustments ...

	# Align the y-labels
	ax.yaxis.set_label_coords(-0.15,0.5)

	plt.subplots_adjust(left=0.18, top=0.95, right=0.95, 
		bottom=.18)

	fig.savefig(writeFname)
	plt.show()


def plotSomeSpecificityComparisons(
	readFname='specificity/allComps_allImages_50.json', 
	writeFname='figs/specificity-allImages.pdf',
	normalize=False
	):
	'''
	Plots only the most interesting specificity comparisons between different
	treatments.  It plots only overall specificity comparisons (neither 
	food-specific nor culture-specific specificity comparisons are included).
	'''

	# For this plot, we are interested in very particular data.  We'll be
	# cherry picking the ones we want
	data = json.loads(open(readFname).read())
	picked_data = []
	Y = []

	# We're interested in just the 'overall' comparisons
	data = data[0]
	assert(data['valence'] == 'overall')

	# now find a couple comparisons with the ambg treatment
	with_ambg_as_basis = data['results'][0]
	assert(with_ambg_as_basis['basis'] == 'treatment0')
	with_ambg_as_basis = with_ambg_as_basis['results']

	# ambg <--> cult_img
	picked_data.extend(
		filter(lambda d: d['subject'] == 'treatment1', with_ambg_as_basis))

	# ambg <--> ingr_img
	picked_data.extend(
		filter(lambda d: d['subject'] == 'treatment2', with_ambg_as_basis))

	# and we'll take the comparison between cult_img and ingr_img
	with_cult_img_as_basis = data['results'][1]
	assert(with_cult_img_as_basis['basis'] == 'treatment1')
	with_cult_img_as_basis = with_cult_img_as_basis['results']

	# cult_img <--> ingr_img
	picked_data.extend(
		filter(lambda d: d['subject'] == 'treatment2', 
		with_cult_img_as_basis))

	basisLabels = [TREATMENT_NAMES[basis] 
		for basis in ['treatment0', 'treatment0', 'treatment1']]

	subjectLabels = [TREATMENT_NAMES[basis] 
		for basis in ['treatment1', 'treatment2', 'treatment2']]

	figWidth = 8.7 / 2.54 			# convert from PNAS spec in cm to inches
	figHeight = (5/5.)*figWidth 	# a reasonable aspect ratio
	fig = plt.figure(figsize=(figWidth,figHeight))
	gs = gridspec.GridSpec(1,1)

	Y = [pd['avg'] for pd in picked_data]
	Y_err = [pd['stdev'] * CONFIDENCE_95 for pd in picked_data]
	X = range(len(Y))

	width=0.75

	ax = plt.subplot(gs[0])

	series = ax.bar(X, Y, width, color='0.25', ecolor='0.55', 
		yerr=Y_err)

	# control the plot X limits
	padding = 0.25
	xlims = (-padding, len(Y) - 1 + width + padding)
	plt.xlim(xlims)

	# Annotate the plot with a line at Y=0
	zero = ax.plot(
		xlims, [0, 0], color='0.35', linestyle='-', zorder=0)

	
	# handle x-axis labelling
	ax.tick_params(axis='both', which='major', labelsize=9)
	ax.set_xticks(map(lambda x: x + width/2., X))
	ax.set_xticklabels(subjectLabels, rotation=45, 
		horizontalalignment='right')

	ax.set_ylabel("relative specificity", size=9)


	plt.draw()
	plt.tight_layout()


	# Post plot adjustments ...

	# Align the y-labels
	ax.yaxis.set_label_coords(-0.12,0.5)


	# Label the basis treatments above the subplots
	for i, bl in enumerate(basisLabels):

		left = i + width/2.
		ylims = ax.get_ylim()
		# put the label directly above the plot.  
		# The first label needs to be put a bit higher.
		height= ylims[1] + 0.5
		ax.text(left, height, bl, va='bottom', ha='right', size=9, 
			rotation=-45)

	plt.subplots_adjust(left=0.15, top=0.83, right=0.95, 
		bottom=.18)

	fig.savefig(writeFname)
	plt.show()


def plotAllSpecificityComparisons(
	readFname='specificity/allComps_allImages_50.json', 
	writeFname='figs/specificity-full-allImages.pdf',
	normalize=False
	):
	'''
	Plots all of the interesting specificity comparisons between different
	treatments, in a big multi-pannel figure.  This does overall specificity 
	comparisons as well as food-specific and culture-specific specificity 
	comparisons.

	`normalize` controlls whether the plot is normalized into units of 
	standard deviations of the null comparison.  This was used previously, as
	part of a different way of approaching testing the significance of the 
	spceficity results.
	'''

	data = json.loads(open(readFname).read())

	subplotLabels = [
		'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q',
		'R','S'
	]
	basisLabels = []

	
	figWidth = 17.8 / 2.54 			# convert from PNAS spec in cm to inches
	figHeight = (10/11.)*figWidth 	# a reasonable aspect ratio
	fig = plt.figure(figsize=(figWidth,figHeight))

	num_cols = len(data[0]['results'])
	width_ratios = [5 + s*4 for s in reversed(range(num_cols))]

	gs = gridspec.GridSpec(3,num_cols, width_ratios=width_ratios)
	subplotCounter = 0

	shared_x = []	# keeps references to x-axes for sharing

	for valenceComparison in data:
		valence = valenceComparison['valence']
		
		shared_y = None # keeps references to y-axes for sharing

		for basisComparison in valenceComparison['results']:
			basis = basisComparison['basis']
			normStdev = basisComparison['null']['stdev']

			comparisonData = basisComparison['results']

			width=0.75
			X = range(len(comparisonData))

			if normalize:
				Y = map(lambda x: x['avgNorm'], comparisonData)
				Y_err = map(lambda x: x['stdev']/normStdev, comparisonData)

			else:
				Y = map(lambda x: x['avg'], comparisonData)
				Y_err = map(
					lambda x: x['stdev']*CONFIDENCE_95, comparisonData)

			treatmentNames = map(lambda x: x['subject'], comparisonData)

			row = subplotCounter / num_cols
			col = subplotCounter % num_cols

			# create axes, and keep track of axis sharing, while you're at it
			# if the subplot is on the first row, keep ref for x-sharing
			if row == 0:

				# if subplot is first within its row, keep ref for y-sharing
				if col == 0:
					ax = plt.subplot(gs[subplotCounter])
					shared_x.append(ax)
					shared_y = ax	# keep a reference for sharing y-axis

				else:
					ax = plt.subplot(gs[subplotCounter], sharey=shared_y)
					shared_x.append(ax)

			else:
				if col == 0:
					ax = plt.subplot(gs[subplotCounter], sharex=shared_x[col])
					shared_y = ax

				else:
					ax = plt.subplot(gs[subplotCounter],
						sharex=shared_x[col], sharey=shared_y)

			series = ax.bar(X, Y, width, color='0.25', ecolor='0.55', 
				yerr=Y_err)

			# control the plot X limits
			padding = 0.25
			xlims = (-padding, len(comparisonData) - 1 + width + padding)
			plt.xlim(xlims)

			# Annotate the plot with a line at Y=0
			zero = ax.plot(
				xlims, [0, 0], color='0.35', linestyle='-', zorder=0)

			# for a normalized plot, annotatet the plot with 95% confidence
			# interval for the null comparison
			if normalize:
				confidenceIntervalUpper = ax.plot(
					xlims, [CONFIDENCE_95, CONFIDENCE_95], color='0.35',
					linestyle=':')

				confidenceIntervalLower = ax.plot(
					xlims, [-CONFIDENCE_95, -CONFIDENCE_95], color='0.35',
					linestyle=':')
			
			# determine the basis-treatment label, but don't apply it yet
			basisLabels.append(TREATMENT_NAMES[basis])

			# handle x-axis labelling
			ax.tick_params(axis='both', which='major', labelsize=9)
			if row == 2:
				xlabels = [TREATMENT_NAMES[t] for t in treatmentNames]
				ax.set_xticks(map(lambda x: x + width/2., X))
				ax.set_xticklabels(xlabels, rotation=45, 
					horizontalalignment='right')
			else:
				plt.setp(ax.get_xticklabels(), visible=False)

			# handle y-axis labelling
			if col == 0:
				if row == 0:
					ax.set_ylabel("overall specificity", size=9)

				if row == 1:
					ax.set_ylabel("cultural specificity", size=9)

				if row == 2:
					ax.set_ylabel("food specificity", size=9)

			else:
				plt.setp(ax.get_yticklabels(), visible=False)

			subplotCounter += 1
			plt.draw()
			if row < 2 :
				plt.tight_layout()

	# We need to apply some labels after everything is plotted, once the
	# axis limits are stable
	for i, ax in enumerate(fig.axes):

		row = i / num_cols
		col = i % num_cols

		# Make some adjustments to the scale of the yaxes.  This helps keep
		# the y-tick-labels of vertically adjacent plots from running into 
		# eachother
		if col == 0:
			ylims = ax.get_ylim()
			y_range = ylims[1] - ylims[0]

			if row == 0:
				pad = y_range * -0.1

			if row > 0:
				pad = y_range * 0.04

			ax.set_ylim(ylims[0]-pad, ylims[1]+pad)

			
		# Align the y-labels
		if col == 0:
			ax.yaxis.set_label_coords(-0.25,0.5)


		# Label the basis treatments above the subplots
		# only do this on the first row!
		if row == 0:
			basisTreatment = data[0]['results'][i]['basis']
			basisTreatmentName = TREATMENT_NAMES[basisTreatment]
			num_bars = len(data[0]['results'][i]['results'])
			left = (num_bars -1)/2.0 + width
			ylims = ax.get_ylim()
			# put the label directly above the plot.  
			# The first label needs to be put a bit higher.
			height= ylims[1] + (0.5 if i else 0.5)
			ax.text(left, height, basisTreatmentName, 
					va='bottom', ha='right', size=9, rotation=-45)


		# get the axes limits to position labels well
		xmin, xmax, ymin, ymax = ax.axis()
		x_range = xmax - xmin
		y_range = ymax - ymin
		x_shim = 0.2
		y_inset_shim_factor = 0.05
		y_letter_shim_factor = 0.03


		# Label each pannel with a letter
		letterLabel = subplotLabels[i]
		ax.text(xmin + x_shim, ymax - y_range*y_letter_shim_factor, 
			letterLabel, va='top', ha='left', size=12)



	plt.subplots_adjust(left=0.10, top=0.90, right=0.98, 
		bottom=.11, wspace=0.05, hspace=0.05)
	fig.savefig(writeFname)
	plt.show()


def computeSpecificityLongitudinal(sampleSize=124):

	for image_id in range(5):

		image_name = 'test%d' % image_id

		computeSpecificityComparisons(
			fname='specificity/t0-t1-%s-overall_50.json' % image_name,
			sampleSize=sampleSize,
			comparisons='__first__',
			images = [image_name],
			valences=['overall'],
		)


def computeSpecificityComparisons(
	fname='specificity/allImages.json',
	sampleSize=124,
	comparisons='__all__',
	images=['test%d'%i for i in range(5)],
	valences=['overall', 'cultural', 'food'],
	return_results=False
	):
	'''
	Computes all of the interesting specificity comparisons between different
	treatments, such that they can be plotted  in a big multi-pannel figure.
	This does overall specificity comparisons as well as food-specific and
	culture-specific specificity comparisons.

	This only does the computation and writes the results to file; you need
	to run `plotAllSpecificityComparisons()` to generate the plot.
	'''

	if comparisons == '__first__':
		comparisonSchedule = {'treatment0': ['treatment1']}

		# ordered basis treatments -- used to control the order that each
		# set of comparisons is performed
		ordered_basis_treatments = ['treatment%d' % i for i in [0]]

	elif comparisons == '__all__':
		comparisonSchedule = {
			'treatment0': ['treatment1', 'treatment5', 'treatment6',
				'treatment2', 'treatment3', 'treatment4']

			, 'treatment1': ['treatment5', 'treatment6', 'treatment2', 
				'treatment3', 'treatment4']

			, 'treatment2': ['treatment5', 'treatment6', 
					'treatment3', 'treatment4']

			, 'treatment5': ['treatment6', 'treatment3', 'treatment4']

			, 'treatment3': ['treatment6', 'treatment4']

			, 'treatment6': ['treatment4']
		}

		# ordered basis treatments -- used to control the order that each
		# set of comparisons is performed
		ordered_basis_treatments = ['treatment%d' % i for i in [0,1,2,5,3,6]]

	print '\nComparison based on images: ' + str(images)

	# Make an analyzer object -- it performs the actual comparisons
	a = analysis.Analyzer()

	# A results object to aggregate all the data
	fh = open(fname, 'w')
	results = []


	for valence in valences:
		thisValenceResults = {'valence': valence, 'results': []}
		results.append(thisValenceResults)

		print '   Valence: %s' % valence

		for basisTreatment in ordered_basis_treatments:

			print '****basisTreatment: %s' % basisTreatment
			
			thisBasisResults = {'basis': basisTreatment, 'results':[]}
			thisValenceResults['results'].append(thisBasisResults)

			# we don't compute basis results anymore.  This is here to 
			# not break the plotting function
			nullComparison = {}
			thisBasisResults['null'] = nullComparison

			# Now compare the basis treatment to each subject treatment
			for subjectTreatment in comparisonSchedule[basisTreatment]:

				print '      subjectTreatment: %s' % subjectTreatment

				# compare basis treatment to subject treatment
				rslt = a.compareValenceSpecificity(
					valence, subjectTreatment, basisTreatment, sampleSize,
					images)

				subjectComparison = {
					'subject': subjectTreatment,
					'stdev': rslt['stdMoreMinusLess'],
					'avg': rslt['avgMoreMinusLess']
				}

				thisBasisResults['results'].append(subjectComparison)

	fh.write(json.dumps(results, indent=3))
	fh.close

	if return_results:
		return results


def getTestDataset():
	'''
	Factory method that builds a CleanDataset from a small testing subset
	of the Amazon Mechanical Turk CSV data
	'''

	# Create a new priming-image-label-experiment dataset
	dataset = data_processing.CleanDataset()

	# Read from the raw amt csv files.  
	# Note: order matters!  The older files have duplicates workers that
	# get ignored.  Make sure to read the newer file files earlier
	dataset.read_csv('amt_csv/test100.csv', True)

	# The dataset needs to do some internal calts to refresh its state 
	dataset.aggregateCounts()
	dataset.calc_ktop(5)

	return dataset


def checkOrientationSignificance(
	readFname='orientation/orientation.json',
	writeFname='orientation/significance.json',
	populationSize = 126
	):
	''' 
		Checks for a significant difference between the fraction of food-
		and culture-oriented words for all treatments compared to AMBG.
	'''

	# Read the file.  We're interested in the data in panel 1, which corresponds
	# To panel B of figure 7 in the manuscript
	data = json.loads(open(readFname, 'r').read())['panel1']

	results = []
	for treatment in [1,5,6,2,3,4]:
		result = {}
		result['treatment_id'] = treatment
		result['treatment_name'] = TREATMENT_NAMES['treatment%d'%treatment]
		for valence in ['food', 'cultural', 'excessCultural']:
			difference = (data['avg'][valence][treatment] 
				- data['avg'][valence][0])
			var_mean = (data['std'][valence][treatment]**2 
				+ data['std'][valence][0]**2)
			zscore = difference / np.sqrt(var_mean)

			result['%s_difference'%valence] = difference
			result['%s_var_mean'%valence] = var_mean
			result['%s_zscore'%valence] = zscore

		results.append(result)

	fh = open(writeFname, 'w')
	fh.write(json.dumps(results, indent=3))
	fh.close()


def computeOrientationVsTreatment(
	fname='orientation/orientation.json', useTestData=False):

	'''
	This computes the data for a figure with  3 pannels:

	1) Orientation vs image (aggregating labels from all treatments)
	2) Orientation vs treatment (aggregating all images)
	3) Orientation vs treatment using labels from the image 'test0'
	'''

	fh = open(fname, 'w')


	treatmentIds = [0,1,5,6,2,3,4]
	treatments = ['treatment%d' % i for i in treatmentIds]

	if not useTestData:
		a = analysis.Analyzer()
	else:
		a = analysis.Analyzer(dataset=getTestDataset())

	plotData = {}

	# First we will plot composition vs image, aggregating all treatments
	print '\n\npreparing panel0 data: composition as a function of image'
	plotData['panel0'] = {
		'avg':{'cultural':[], 'food':[], 'both':[]},
		'std':{'cultural':[], 'food':[], 'both':[]}
	}
	for image in ['test%d' % i for i in range(5)]:
		print '\nImage: %s' %image
		result = a.percentValence(treatments, [image])

		plotData['panel0']['avg']['cultural'].append(
			result['mean']['cultural'])
		plotData['panel0']['avg']['food'].append(result['mean']['food'])
		plotData['panel0']['avg']['both'].append(result['mean']['both'])

		plotData['panel0']['std']['cultural'].append(
			result['stdev']['cultural'])
		plotData['panel0']['std']['food'].append(result['stdev']['food'])
		plotData['panel0']['std']['both'].append(result['stdev']['both'])

	# Next we will make plots that examine composition as a function of 
	# treatment
	print '\n\npreparing panel1 and 2 data: composition as a function of '\
		'treatment, with labels from all images, and then only the first '\
		'image.'
	imageSets = [
		['test%d' % i for i in range(5)],
		['test0']
	]
	for i, images in enumerate(imageSets):
		print ('\nAll images' if not i else '\nFirst image')
		plotData['panel%d'%(i+1)] = {
			'avg':{'cultural':[], 'food':[], 'both':[], 'excessCultural':[]},
			'std':{'cultural':[], 'food':[], 'both':[], 'excessCultural':[]}
		}
		thisPlotData =  plotData['panel%d'%(i+1)]

		for treatment in treatments:

			result = a.percentValence([treatment], images)

			thisPlotData['avg']['cultural'].append(result['mean']['cultural'])
			thisPlotData['avg']['food'].append(result['mean']['food'])
			thisPlotData['avg']['both'].append(result['mean']['both'])
			thisPlotData['avg']['excessCultural'].append(
				result['mean']['excessCultural'])

			thisPlotData['std']['cultural'].append(result['stdev']['cultural'])
			thisPlotData['std']['food'].append(result['stdev']['food'])
			thisPlotData['std']['both'].append(result['stdev']['both'])
			thisPlotData['std']['excessCultural'].append(
				result['stdev']['excessCultural'])

	# Finally we make plots that examine the excess culture orientation
	# as a function of image
	print '\n\npreparing panel3,4, and 5 data: excess cultural composition .'

	plotData['panel5'] = {
		'avg': [],
		'std': []
	}

	ambgData = {'avg':[], 'std':[]}
	for i, treatment in enumerate(['treatment1', 'treatment5']):
		plotData['panel%d' % (i+3)] = {
			'avg': [],
			'std': []
		}

		thisPlotData = plotData['panel%d' % (i+3)]
		for image in ['test%d' % j for j in range(5)]:

			if i == 0:
				result = a.percentValence(['treatment0'], [image])

				plotData['panel5']['avg'].append(
					result['mean']['excessCultural'])
				plotData['panel5']['std'].append(
					result['stdev']['excessCultural'])

				ambgData['avg'].append(result['mean']['excessCultural'])
				ambgData['std'].append(result['stdev']['excessCultural'])

			result = a.percentValence([treatment], [image])

			thisPlotData['avg'].append(result['mean']['excessCultural'])
			thisPlotData['std'].append(result['stdev']['excessCultural'])


	# Adjust the excess cultural values to use AMBG treatment as a baseline
	# since we are subtracting means, the resulting std error is the sqrt of
	# the sum of the squares of std errors
	for i in range(5):
		plotData['panel3']['avg'][i] = (
			plotData['panel3']['avg'][i] - ambgData['avg'][i])

		plotData['panel3']['std'][i] = (
			np.sqrt(
				plotData['panel3']['std'][i] + ambgData['std'][i]
			)
		)

		plotData['panel4']['avg'][i] = ( 
			plotData['panel4']['avg'][i] - ambgData['avg'][i])

		plotData['panel4']['std'][i] = (
			np.sqrt(
				(plotData['panel4']['std'][i])**2 + (ambgData['std'][i])**2
			)
		)

	fh.write(json.dumps(plotData, indent=3))
	fh.close()


def plotExcessCultureVsImage(
	readFname='orientation/orientation.json',
	writeFname='figs/excessCultureVsTreatment-t1.pdf',
	num_subplots=1
	):

	images = ['image %d' % i for i in range(1,6)]

	plotData = json.loads(open(readFname, 'r').read())

	width = 0.75

	subplotLabels = ['A', 'B']

	subplots = range(num_subplots)
	gs = gridspec.GridSpec(1,len(subplots))
	if len(subplots) == 1:
		figWidth = 8.7/2.54
		figHeight = figWidth*4/5.
	elif len(subplots) == 2:
		figWidth = 17.4/2.54
		figHeight = figWidth*2/5.
	else:
		raise ValueError('Expected either 1 or 2 subplots')

	fig = plt.figure(figsize=(figWidth, figHeight))

	X = range(len(images))


	for subplot in subplots:

		# If this is the first subplot, keep a reference
		# Also, the first plot has a different x-axis
		if subplot == 0:
			ax = plt.subplot(gs[subplot])
			ax0 = ax

		# If this is the second subplot, make it's y-axis linked to the first
		else:
			ax = plt.subplot(gs[subplot], sharey=ax0)

		X2 = map(lambda x: x + width/2.0, X)

		# first subplot plots panel 3, second plots panel 4
		panel = 'panel%d' % (subplot+3) 

		thisPlotData = plotData[panel]

		series =ax.bar(
			X, thisPlotData['avg'], width, color='0.25', 
			ecolor='0.55', yerr=thisPlotData['std'])

		ax.tick_params(axis='both', which='major', labelsize=9)
		# only label the y-axis of the first sub-plot
		if subplot == (len(subplots)-1):
			ax.set_ylabel("excess cultural orientation (%)", size=9)

		else:
			plt.setp(ax.get_yticklabels(), visible=False)

		# first subplot also has its own kind of x-axis
		xlabels = images
		ax.set_xticks(X2)
		ax.set_xticklabels(xlabels, rotation=45, horizontalalignment='right')

		# Label each pannel
		letterLabel = subplotLabels[subplot]
		ax.text(0.2, 75, letterLabel, 
				va='top', ha='left', size=20)

		padding = 0.25
		xlims = (-padding, len(X) - 1 + width + padding)
		plt.xlim(xlims)

		plt.ylim((0, plt.ylim()[1]+2.5))

		plt.draw()
		plt.tight_layout()

	# Adjustments to figure placement and spacing
	fig.subplots_adjust(bottom=.20)

	#plt.ylim((0,55))

	fig.savefig(writeFname)
	plt.show()
	

def pick(picks, a):
	if picks == '__all__':
		return a
	else:
		return [a[p] for p in picks]


def plotOrientationAndSpecificity(
		readFname=[
			'orientation/orientation.json',
			'specificity/allComps_allImages_50.json'
		],
		writeFname='figs/orientation_specificity.pdf'
	):

	'''
	Plots specificity and orientation for only the major treatments.
	'''

	# within the dataset, panel2 and panel3 have data organized by treatment
	# in this order
	treatmentIds = [0,1,5,6,2,3,4]

	# we only want treatments 0,1, and 2, and so we have to 'pick' the data
	# at positions 0,1, and 4
	PICKS = [0,1,4]
	treatmentIds = pick(PICKS, treatmentIds)

	treatments = ['treatment%d' % i for i in treatmentIds]
	images = ['image %d' % i for i in range(1,6)]

	plotData = json.loads(open(readFname[0], 'r').read())

	width = 0.42
	padding = 1 - 2*width

	figWidth = 17.8 / 2.54
	figHeight = figWidth*5/10.
	fig = plt.figure(figsize=(figWidth,figHeight))

	gs_ratio = 4 * (len(PICKS) + 1) 
	gs = gridspec.GridSpec(1,2, width_ratios=[9,5])

	PANNEL = 2

	ax = plt.subplot(gs[0])
	X = range(len(treatments))

	X2 = [x + width for x in X]
	panel = 'panel%d' % PANNEL

	thisPlotData = plotData[panel]

	picks = PICKS

	# as we plot, we adjust the error bars to plot the 95% confidence 
	# intervals rather than standard errors
	yerr_cult = pick(picks, thisPlotData['std']['cultural'])
	yerr_cult = [y*1.95 for y in yerr_cult]
	seriesCultural = ax.bar(
		X, 
		pick(picks, thisPlotData['avg']['cultural']),
		width, color='0.25', 
		ecolor='0.55', 
		bottom=pick(picks, plotData[panel]['avg']['both']), 
		yerr=yerr_cult)

	yerr_food = pick(picks, thisPlotData['std']['food'])
	yerr_food = [y*1.96 for y in yerr_food]
	seriesFood = ax.bar(
		X2,
		pick(picks, thisPlotData['avg']['food']),
		width, color='0.55',
		ecolor='0.25', 
		bottom=pick(picks, plotData[panel]['avg']['both']), 
		yerr=yerr_food)
		
	seriesBoth1 =ax.bar(
		X, 
		pick(picks, thisPlotData['avg']['both']),
		width, color='0.85', ecolor='0.55')

	seriesBoth2 =ax.bar(
		X2,
		pick(picks, thisPlotData['avg']['both']),
		width, color='0.85', ecolor='0.25')

	ax.set_ylabel("% of labels", size=9)
	ax.tick_params(axis='both', which='major', labelsize=9)


	# Other two subplots have different x-axis from first
	xlabels = [TREATMENT_NAMES[t] for t in treatments]
	ax.set_xticks(X2)
	ax.set_xticklabels(xlabels, rotation=45, 
		horizontalalignment='right', size=9)

	#ylims = plt.ylim()
	#plt.ylim(ylims[0], ylims[1]*1.12)

	xlims = (-padding, len(X) - 1 + 2*width + padding)
	plt.xlim(xlims)

	#plt.draw()
	#plt.tight_layout()

	legend = ax.legend( 
		(seriesCultural[0], seriesFood[0], seriesBoth1[0]), 
		('culture', 'food', 'both'), 
		bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
		prop={'size':9}, labelspacing=0, mode='expand',
		ncol=3, borderaxespad=0.
	)

	ax.yaxis.set_label_coords(-0.10,0.5)

	fig.savefig(writeFname)
	plt.show()


	# *** #


	# For this plot, we are interested in very particular data.  We'll be
	# cherry picking the ones we want
	data = json.loads(open(readFname[1]).read())
	picked_data = []
	Y = []

	# We're interested in just the 'overall' comparisons
	data = data[0]
	assert(data['valence'] == 'overall')

	# now find a couple comparisons with the ambg treatment
	with_ambg_as_basis = data['results'][0]
	assert(with_ambg_as_basis['basis'] == 'treatment0')
	with_ambg_as_basis = with_ambg_as_basis['results']

	# ambg <--> cult_img
	picked_data.extend(
		filter(lambda d: d['subject'] == 'treatment1', with_ambg_as_basis))

	# ambg <--> ingr_img
	picked_data.extend(
		filter(lambda d: d['subject'] == 'treatment2', with_ambg_as_basis))

	# and we'll take the comparison between cult_img and ingr_img
	with_cult_img_as_basis = data['results'][1]
	assert(with_cult_img_as_basis['basis'] == 'treatment1')
	with_cult_img_as_basis = with_cult_img_as_basis['results']

	# cult_img <--> ingr_img
	picked_data.extend(
		filter(lambda d: d['subject'] == 'treatment2', 
		with_cult_img_as_basis))

	basisLabels = [TREATMENT_NAMES[basis] 
		for basis in ['treatment0', 'treatment0', 'treatment1']]

	subjectLabels = [TREATMENT_NAMES[basis] 
		for basis in ['treatment1', 'treatment2', 'treatment2']]

	# taking the negative makes the plot more intuitive
	Y = [-pd['avg'] for pd in picked_data] 
	Y_err = [pd['stdev'] * CONFIDENCE_95 for pd in picked_data]
	X = range(len(Y))

	width=0.75

	ax = plt.subplot(gs[1])

	series = ax.bar(X, Y, width, color='0.25', ecolor='0.55', 
		yerr=Y_err)

	# control the plot X limits
	padding = 0.25
	xlims = (-padding, len(Y) - 1 + width + padding)
	plt.xlim(xlims)

	# Annotate the plot with a line at Y=0
	zero = ax.plot(
		xlims, [0, 0], color='0.35', linestyle='-', zorder=0)
	
	# handle x-axis labelling
	ax.tick_params(axis='both', which='major', labelsize=9)
	ax.set_xticks(map(lambda x: x + width/2., X))
	ax.set_xticklabels(subjectLabels, rotation=45, 
		horizontalalignment='right')

	ax.set_ylabel("relative specificity", size=9)


	plt.draw()
	plt.tight_layout()


	# Post plot adjustments ...

	# Align the y-labels
	ax.yaxis.set_label_coords(-0.18,0.5)


	# Label the basis treatments above the subplots
	for i, bl in enumerate(basisLabels):

		left = i + width/2.
		ylims = ax.get_ylim()
		# put the label directly above the plot.  
		# The first label needs to be put a bit higher.
		height= ylims[1] + 0.5
		ax.text(left, height, bl, va='bottom', ha='right', size=9, 
			rotation=-45)

	plt.subplots_adjust(left=0.08, top=0.80, right=0.98, 
		bottom=.20, wspace=0.30)

	fig.savefig(writeFname)
	plt.show()


def plotOrientationVsTreatment(readFname='orientation/orientation.json',
	writeFname='figs/orientationVsTreatment.pdf'):

	# within the dataset, panel2 and panel3 have data organized by treatment
	# in this order
	treatmentIds = [0,1,5,6,2,3,4]

	# we only want treatments 0,1, and 2, and so we have to 'pick' the data
	# at positions 0,1, and 4
	PICKS = [0,1,4]
	treatmentIds = pick(PICKS, treatmentIds)

	treatments = ['treatment%d' % i for i in treatmentIds]
	images = ['image %d' % i for i in range(1,6)]

	plotData = json.loads(open(readFname, 'r').read())

	width = 0.375

	subplotLabels = ['A', 'B', 'C']

	figWidth = 8.7 / 2.54
	figHeight = figWidth*4/5.
	fig = plt.figure(figsize=(figWidth,figHeight))

	gs_ratio = 4 * (len(PICKS) + 1) 
	gs = gridspec.GridSpec(1,1)

	subplot = 2

	ax = plt.subplot(gs[0])
	X = range(len(treatments))

	X2 = [x + width for x in X]
	panel = 'panel%d' % subplot

	thisPlotData = plotData[panel]

	picks = PICKS

	# as we plot, we adjust the error bars to plot the 95% confidence 
	# intervals rather than standard errors
	yerr_cult = pick(picks, thisPlotData['std']['cultural'])
	yerr_cult = [y*1.95 for y in yerr_cult]
	seriesCultural = ax.bar(
		X, 
		pick(picks, thisPlotData['avg']['cultural']),
		width, color='0.25', 
		ecolor='0.55', 
		bottom=pick(picks, plotData[panel]['avg']['both']), 
		yerr=yerr_cult)

	yerr_food = pick(picks, thisPlotData['std']['food'])
	yerr_food = [y*1.96 for y in yerr_food]
	seriesFood = ax.bar(
		X2,
		pick(picks, thisPlotData['avg']['food']),
		width, color='0.55',
		ecolor='0.25', 
		bottom=pick(picks, plotData[panel]['avg']['both']), 
		yerr=yerr_food)
		
	seriesBoth1 =ax.bar(
		X, 
		pick(picks, thisPlotData['avg']['both']),
		width, color='0.85', ecolor='0.55')

	seriesBoth2 =ax.bar(
		X2,
		pick(picks, thisPlotData['avg']['both']),
		width, color='0.85', ecolor='0.25')

	# only label the y-axis of the first sub-plot
	if subplot == 2:
		ax.set_ylabel("% of labels", size=9)
		ax.tick_params(axis='both', which='major', labelsize=9)


		# Other two subplots have different x-axis from first
		xlabels = [TREATMENT_NAMES[t] for t in treatments]
		ax.set_xticks(X2)
		ax.set_xticklabels(xlabels, rotation=45, 
			horizontalalignment='right', size=9)

	# only put the legend on the last sub-plot
	if subplot == 2:

		padding = 0.25
		#ylims = plt.ylim()
		#plt.ylim(ylims[0], ylims[1]*1.12)

		xlims = (-padding, len(X) - 1 + 2*width + padding)
		plt.xlim(xlims)

		plt.draw()
		plt.tight_layout()

		legend = ax.legend( 
			(seriesCultural[0], seriesFood[0], seriesBoth1[0]), 
			('culture', 'food', 'both'), 
			bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
			prop={'size':9}, labelspacing=0, mode='expand',
			ncol=3, borderaxespad=0.
		)

	# Adjustments to figure placement and spacing
	fig.subplots_adjust(bottom=.20, right=0.97, top=0.88, left=0.12)

	#plt.ylim((0,55))

	fig.savefig(writeFname)
	plt.show()


def plotOrientationVsTreatment_full(readFname='orientation/orientation.json',
	writeFname='figs/orientationVsTreatment_full.pdf'):

	# within the dataset, panel2 and panel3 have data organized by treatment
	# in this order
	treatmentIds = [0,1,5,6,2,3,4]

	# we only want treatments 0,1, and 2, and so we have to 'pick' the data
	# at positions 0,1, and 4
	PICKS = range(7)
	treatmentIds = pick(PICKS, treatmentIds)

	treatments = ['treatment%d' % i for i in treatmentIds]
	images = ['image %d' % i for i in range(1,6)]

	plotData = json.loads(open(readFname, 'r').read())

	width = 0.375

	subplotLabels = ['A', 'B', 'C']

	figWidth = 17.8 / 2.54
	figHeight = figWidth*4/10.
	fig = plt.figure(figsize=(figWidth,figHeight))

	gs_ratio = 4 * (len(PICKS) + 1) 
	gs = gridspec.GridSpec(1,3, width_ratios=[25, gs_ratio, gs_ratio])

	for subplot in [0,1,2]:

		# If this is the first subplot, keep a reference
		# Also, the first plot has a different x-axis
		if subplot == 0:
			ax = plt.subplot(gs[subplot])
			ax0 = ax
			X = range(len(images))

		# If not the first subplot, make it's y-axis linked to the first
		else:
			ax = plt.subplot(gs[subplot], sharey=ax0)
			X = range(len(treatments))

		X2 = [x + width for x in X]
		panel = 'panel%d' % subplot

		thisPlotData = plotData[panel]

		if subplot == 0:
			picks = '__all__'
		else:
			picks = PICKS

		yerr_cult = pick(picks, thisPlotData['std']['cultural'])
		yerr_cult = [y*1.96 for y in yerr_cult]
		seriesCultural = ax.bar(
			X, 
			pick(picks, thisPlotData['avg']['cultural']),
			width, color='0.25', 
			ecolor='0.55', 
			bottom=pick(picks, plotData[panel]['avg']['both']), 
			yerr=yerr_cult)

		yerr_food = pick(picks, thisPlotData['std']['food'])
		yerr_food = [y*1.96 for y in yerr_food]
		seriesFood = ax.bar(
			X2,
			pick(picks, thisPlotData['avg']['food']),
			width, color='0.55',
			ecolor='0.25', 
			bottom=pick(picks, plotData[panel]['avg']['both']), 
			yerr=yerr_food)
			
		seriesBoth1 =ax.bar(
			X, 
			pick(picks, thisPlotData['avg']['both']),
			width, color='0.85', ecolor='0.55')

		seriesBoth2 =ax.bar(
			X2,
			pick(picks, thisPlotData['avg']['both']),
			width, color='0.85', ecolor='0.25')

		# only label the y-axis of the first sub-plot
		if subplot == 0:
			ax.set_ylabel("% labels of orientation", size=9)
			ax.tick_params(axis='both', which='major', labelsize=9)


			# first subplot also has its own kind of x-axis
			xlabels = images
			ax.set_xticks(X2)
			ax.set_xticklabels(xlabels, rotation=45, 
				horizontalalignment='right', size=9)

		else:

			# Other two subplots have different x-axis from first
			xlabels = [TREATMENT_NAMES[t] for t in treatments]
			ax.set_xticks(X2)
			ax.set_xticklabels(xlabels, rotation=45, 
				horizontalalignment='right', size=9)
			# only the first subplot gets y-axis labels
			plt.setp(ax.get_yticklabels(), visible=False)

		# only put the legend on the last sub-plot
		if subplot == 2:
			legend = ax.legend( 
				(seriesCultural[0], seriesFood[0], seriesBoth1[0]), 
				('culture', 'food', 'both'), 
				loc='upper right', prop={'size':9}, labelspacing=0)

		# Label each pannel
		letterLabel = subplotLabels[subplot]
		ax.text(0.2, 75, letterLabel, 
				va='top', ha='left', size=12)

		padding = 0.25
		xlims = (-padding, len(X) - 1 + 2*width + padding)
		plt.xlim(xlims)

		if subplot != 3:
			plt.draw()
			plt.tight_layout()

	# Adjustments to figure placement and spacing
	fig.subplots_adjust(bottom=.25, wspace=0.05, right=0.99, top=0.98,
		left=0.06)

	#plt.ylim((0,55))

	fig.savefig(writeFname)
	plt.show()
	



def computeAllF1Accuracy(
	fname='f1scores/f1-thetas_full_pairwise25_img1',
	testSetSize=25,
	subplotComparisons=None,
	image_nums=[0]
	):
	'''
	Computes the F1 score for a naive bayes classifier built to distinguish
	between all the interesting pairings of treatments.

	This function only computes the data; to generate a plot, use 
	`plotAllDisting()`
	'''

	fh = open(fname, 'w')
	images = ['test%d' % i for i in image_nums]

	if subplotComparisons is None:
		subplotComparisons = [
			{'basis': 'treatment0', 
			'subjects' : ['treatment1', 'treatment5', 'treatment6', 
				'treatment2', 'treatment3', 'treatment4']}

			, {'basis': 'treatment1', 
			'subjects' : ['treatment5', 'treatment6', 
				'treatment2', 'treatment3', 'treatment4']}

			, {'basis': 'treatment2', 
			'subjects' : ['treatment5', 'treatment6', 
				'treatment3', 'treatment4']}

			, {'basis': 'treatment5', 
			'subjects' : ['treatment6', 'treatment3', 'treatment4']}

			, {'basis': 'treatment3', 
			'subjects' : ['treatment6', 'treatment4']}

			, {'basis': 'treatment6', 
			'subjects' : ['treatment4']}
		]

	results = []

	a = analysis.NBCAnalyzer()

	for comp in subplotComparisons:

		thisSubplotData = {
			'basis': comp['basis']
			, 'subjects': comp['subjects']
			, 'f1':[]
			, 'accuracy':[]
		}
		results.append(thisSubplotData)

		basisTreatment = comp['basis']

		# build the set of all comparisons to be presented in one subplot
		# NBCAnalyzer() will perform all these comparisons as a batch
		comparisons = [(basisTreatment, subjectTreatment) for
				subjectTreatment in comp['subjects']]
		
		# Compute the results for the batch of comparisons
		comparisonResults = a.crossComparison(
			comparisons, testSetSize, images)

		# Unpack, and then then repack the batch so it can be saved
		# -- sort of inconvenient isn't it?
		thisSubplotData['f1'] = map(
			lambda c: comparisonResults[c]['f1'], comparisons)

		thisSubplotData['accuracy'] = map(
			lambda c: comparisonResults[c]['accuracy'], comparisons)

	fh.write(json.dumps(results, indent=3))
	fh.close()
	return results


def plotAllF1Theta(
		readFname='f1scores/full_pairwise25_img1.json',
		writeFname='figs/f1-thetas_full_pairwise25_img1.pdf',
		theta_only=True,
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
	f1scores = json.loads(open(readFname, 'r').read())

	# Start a figure 
	figWidth = 17.8 / 2.54 	# conversion from PNAS spec in cm to inches
	figHeight = figWidth * 2/5.	# a reasonable aspect ratio
	fig = plt.figure(figsize=(figWidth, figHeight))

	# calculate the gridspec.  The width ratios are based on the width of
	# each bar-plot-group, spacing between them, and plot padding
	num_subplots = len(f1scores)
	width_ratios = [5 + s*4 for s in reversed(range(num_subplots))]
	gs = gridspec.GridSpec(1,num_subplots, width_ratios=width_ratios)

	if not theta_only:
		width = 0.325
	else:
		width = 0.75

	theta_star = analysis.get_theta_star(n, alpha)

	for i, subplotData in enumerate(f1scores):

		# Unpack the data for this subplot
		basisTreatment = subplotData['basis']
		Y_F1s = subplotData['f1']
		X_F1s = range(len(Y_F1s))

		# Convert from accuracy to theta
		Y_thetas = map(lambda t: t*2 - 1, subplotData['accuracy'])
		X_thetas = map(lambda x: x+width, X_F1s)

		# Make a set of axes.  
		# Manage axis-sharing directives, and whether ytick-labels are visible
		if i == 0:
			ax = plt.subplot(gs[i])
			ax0 = ax
			ax.set_ylabel(r'$\theta_{NB}$', size=9)
		else:
			ax = plt.subplot(gs[i], sharey=ax0)
			plt.setp(ax.get_yticklabels(), visible=False)

		theta_series = ax.bar(X_F1s, Y_thetas, width, color='0.25')


		padding = 0.25
		if not theta_only:
			xlims = (-padding, len(Y_thetas) - 1 + 2*width + padding)
		else:
			xlims = (-padding, len(Y_thetas) - 1 + width + padding)
		plt.xlim(xlims)

		# Put together intelligible labels for the x-axis
		ax.tick_params(axis='both', which='major', labelsize=9)
		xlabels = [TREATMENT_NAMES[t] for t in subplotData['subjects']]
		if not theta_only:
			ax.set_xticks(map(lambda x: x + width, X_F1s))
		else:
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

		# Label the basis treatments above the subplots
		basisTreatment = subplotData['basis']
		basisTreatmentName = TREATMENT_NAMES[basisTreatment]
		left = len(subplotData['accuracy'])/2.0 + width - 2*padding
		ylims = plt.ylim()

		# put the label directly above the plot.  
		# The first label needs to be put a bit higher.
		height= ylims[1] + (0.02 if i else 0.06)
		ax.text(left, height, basisTreatmentName, 
				va='bottom', ha='right', size=9, rotation=-45)


	y_low, y_high = plt.ylim()
	plt.ylim(y_low-0.04, y_high)

	fig.subplots_adjust(wspace=0.05, top=0.77, right=0.97, left=0.09, 
			bottom=0.24)
	plt.draw()
	
	fig.savefig(writeFname)
	plt.show()


def plotAllTheta(
	readFname='f1scores/all.json', 
	writeFname='figs/thetas.pdf',
	theta_only=True
	):

	'''
	Plots the theta value for a naive bayes classifier built to distinguish
	between all the interesting pairings of treatments.  See the manuscript
	in <project-root>/docs/drafts for an explanation of theta.

	This function only plots; the data must first be generated by running
	`computeAllF1Accuracy()`
	'''

	subplotLabels = ['A','B','C']

	# Read the data from file
	f1scores = json.loads(open(readFname, 'r').read())

	# Start a figure 
	figWidth = 17.8 / 2.54 	# conversion from PNAS spec in cm to inches
	figHeight = figWidth * 10/4.	# a reasonable aspect ratio
	fig = plt.figure(figsize=(figWidth,figHeight))
	gs = gridspec.GridSpec(1,3, width_ratios=[25,21,17])
	subplotCounter = 0

	for subplotData in f1scores:

		# Unpack the data for this subplot
		basisTreatment = subplotData['basis']
		# Convert from accuracy to theta
		Y_thetas = map(lambda t: t*2 - 1, subplotData['accuracy'])
		X = range(len(Y_thetas))

		# Make a set of axes.  
		# Manage axis-sharing directives, and whether ytick-labels are visible
		if subplotCounter == 0:
			ax = plt.subplot(gs[subplotCounter])
			ax0 = ax
		else:
			ax = plt.subplot(gs[subplotCounter], sharey=ax0)
			plt.setp(ax.get_yticklabels(), visible=False)

		# Do a bar plot
		width = 0.75
		series = ax.bar(X, Y_thetas, width, color='0.25')

		# Label the y-axis, only on the left-most subplot
		if subplotCounter == 0:
			ax.set_ylabel(r'$\theta_{NB}$', size=14)

		# Let the plot breathe horizontally
		padding = 0.25
		xlims = (-padding, len(Y_thetas) - 1 + width + padding)
		plt.xlim(xlims)

		# Label each pannel
		letterLabel = subplotLabels[subplotCounter]
		ax.text(-0.05,0.98,letterLabel, 
				va='top', ha='left', size=20)

		# Label the basis treatment as an inset
		basisTreatmentName = TREATMENT_NAMES[basisTreatment]
		bbox_props =  {'facecolor': 'white'}
		if not subplotCounter:
			bbox_props['pad'] = 8
		ax.text(len(Y_thetas)-0.4,0.1,basisTreatmentName, 
				ha='right', bbox=bbox_props)

		# Put together intelligible labels for the x-axis
		xlabels = [TREATMENT_NAMES[t] for t in subplotData['subjects']]
		ax.set_xticks(map(lambda x: x + width /2., X))
		ax.set_xticklabels(xlabels, rotation=45, size=12,
			horizontalalignment='right')

		# Increment the subplotCounter, 
		# helps us make the subplots display well
		subplotCounter += 1

		# Tighten up the layout
		plt.draw()
		if subplotCounter < 2:
			plt.tight_layout()
	
	fig.savefig(writeFname)
	plt.show()


def plotAllF1(
	readFname='f1scores/all.json', writeFname='figs/f1scores.pdf'):
	'''
	Plots the F1 score for a naive bayes classifier built to distinguish
	between all the interesting pairings of treatments.

	This function only plots; the data must first be generated by running
	`computeAllF1Accuracy()`
	'''

	subplotLabels = ['A','B','C']

	# Read the data from file
	f1scores = json.loads(open(readFname, 'r').read())

	# Start a figure 
	figWidth = 17.8 / 2.54 	# conversion from PNAS spec in cm to inches
	figHeight = figWidth * 10/4.	# a reasonable aspect ratio
	fig = plt.figure(figsize=(figWidth,figHeight))
	gs = gridspec.GridSpec(1,3, width_ratios=[25,21,17])
	subplotCounter = 0

	for subplotData in f1scores:

		# Unpack the data for this subplot
		basisTreatment = subplotData['basis']
		Y_f1scores = subplotData['f1']
		X = range(len(Y_f1scores))

		# Make a set of axes.  
		# Manage axis-sharing directives, and whether ytick-labels are visible
		if subplotCounter == 0:
			ax = plt.subplot(gs[subplotCounter])
			ax0 = ax
		else:
			ax = plt.subplot(gs[subplotCounter], sharey=ax0)
			plt.setp(ax.get_yticklabels(), visible=False)

		# Do a bar plot
		width = 0.75
		series = ax.bar(X, Y_f1scores, width, color='0.25')

		# Label the y-axis, only on the left-most subplot
		if subplotCounter == 0:
			ax.set_ylabel("$F_1$-score", size=14)

		# Let the plot breathe horizontally
		padding = 0.25
		xlims = (-padding, len(Y_f1scores) - 1 + width + padding)
		plt.xlim(xlims)

		# Label each pannel
		letterLabel = subplotLabels[subplotCounter]
		ax.text(-0.05,0.98,letterLabel, 
				va='top', ha='left', size=20)

		# Label the basis treatment as an inset
		basisTreatmentName = TREATMENT_NAMES[basisTreatment]
		bbox_props =  {'facecolor': 'white'}
		if not subplotCounter:
			bbox_props['pad'] = 8
		ax.text(len(Y_f1scores)-0.4,0.1,basisTreatmentName, 
				ha='right', bbox=bbox_props, size=9)

		# Put together intelligible labels for the x-axis
		xlabels = [TREATMENT_NAMES[t] for t in subplotData['subjects']]
		ax.set_xticks(map(lambda x: x + width /2., X))
		ax.set_xticklabels(xlabels, rotation=45, size=12,
			horizontalalignment='right')

		# Increment the subplotCounter, 
		# helps us make the subplots display well
		subplotCounter += 1

		# Tighten up the layout
		plt.draw()
		if subplotCounter < 2:
			plt.tight_layout()
	
	fig.savefig(writeFname)
	plt.show()

CLASSIFICATION_VS_IMAGE_RESULT_WRITE_FNAME = (
	'f1scores/longitudinal-t0-t1_25.json')

def computeClassificationVsImage_multipass(
		num_replicates=20,
		writeFname=CLASSIFICATION_VS_IMAGE_RESULT_WRITE_FNAME
		):
	'''
	This runs computeClassificationVsImage() many times, and accumulates
	the results from each replicate.  It then averages these results, and
	writes the averages to file.

	Motivation: when computeClassificationVsImage is run, results can differ
	substancially from run to run, due to different partitioning of the 
	test and training set within the cross-validation subroutine of 
	NBCAnalyzer.longitudinal().  Some results look better than others, but
	its not right to cherry-pick.  Instead, take the average of many 
	replicates to present a truly representational view on the results.
	'''

	fh = open(writeFname, 'w')

	# Test the classifier's competance in longitudinal (per-image) mode
	# Aggregate the results
	results = []
	for rep_num in range(num_replicates):
		results.append(computeClassificationVsImage(return_result=True))

	# Average the results
	aggregate_result = {}
	for key, val in results[0].items():

		aggregate_result[key] = []
		for position in range(len(val)):
			aggregate_result[key].append(
				np.mean([r[key][position] for r in results]))

	# Write the averaged results to file
	fh.write(json.dumps(aggregate_result, indent=4))
	fh.close()


def computeClassificationVsImage(
	testSetSize=25,
	treatments=('treatment0','treatment1'), 
	writeFname=CLASSIFICATION_VS_IMAGE_RESULT_WRITE_FNAME,
	return_result=False
	):
	'''
	Measures the F1 score for a classifier built to distingiush between
	<treatments> based on the labels attributed to a specific image, 
	as a function of the images for all 5 test images.

	`return_results`: if true, the results are returned to the calling context
		rather than written to file
	'''
	fh = open(writeFname, 'w')

	# Assess the classifier's ability to classify as a function of the image 
	# from which the labels used for classification are derived
	a = analysis.NBCAnalyzer()

	results = a.longitudinal(treatments, testSetSize)

	# return the results, or write them to file
	if return_result:
		return results
	else:
		fh.write(json.dumps(results, indent=3))
		fh.close()


def plotClassificationVsImage(
	readFname='f1scores/longitudinal-t0-t1_25.json',
	writeFname='figs/longitudinalF1scores-t0-t1_25.pdf',
	theta_only=True,
	n=125,
	alpha=0.05
	):
	'''
	Measures the F1 score for a classifier built to distingiush between
	<treatments> based on the labels attributed to a specific image, 
	as a function of the images for all 5 test images.
	'''

	theta_star = analysis.get_theta_star(n, alpha)

	results = json.loads(open(readFname, 'r').read())

	# Plot the result
	numPics = 5
	pre_space = 0.75
	width = 0.375	# width of bars
	if theta_only:
		width = 0.75




	# Make a plot
	figWidth = 8.7 / 2.54 		# convert from PNAS spec in cm to inches
	figHeight = figWidth * 4/5.
	fig = plt.figure(figsize=(figWidth,figHeight))

	gs = gridspec.GridSpec(1,2, width_ratios=[5,21])


	# *** In the first subplot, the datapoint for classification based on 
	# label from *all* images
	ax0 = plt.subplot(gs[0]) # plots performance using all images
	
	Y_F1s = results['f1'][:1]
	X_F1s = [0]

	Y_thetas = results['accuracy'][:1]
	Y_thetas = [t*2 - 1 for t in  Y_thetas]

	X_thetas_all_images = [width]

	# Plot classifier performance, as F1 and theta
	if not theta_only:
		f1_series = ax0(X_F1s, Y_F1s, width, color='0.25')
		theta_series = ax0.bar(X_thetas, Y_thetas, width, color='0.55')
	else:
		theta_series = ax0.bar(X_F1s, Y_thetas, width, color='0.55')

	# Do some labelling business with the plot
	ax0.tick_params(axis='both', which='major', labelsize=9)
	if not theta_only:
		ax0.set_xticks(X_thetas)
	else:
		ax0.set_xticks([x + width/2. for x in X_F1s])

	xlabels = ['all images']
	ax0.set_xticklabels(xlabels, ha='right', rotation=45, size=9)

	# control the plot limits
	padding = 0.25
	xlims = plt.xlim()
	if not theta_only:
		plt.xlim((-padding, 1))
	else:
		plt.xlim((-padding, 1))

	# plot a significance line
	xlims = plt.xlim()
	significance_bar = ax0.plot(xlims, [theta_star, theta_star],
		color='0.35', linestyle=':', zorder=0)

	# *** Done plotting classification based on labels from all images


	# *** Now, plot the data points for classifications based on each image 
	# seperately
	ax1 = plt.subplot(gs[1], sharey=ax0)	# plots performance on per-image basis
	plt.setp(ax1.get_yticklabels(), visible=False)

	# Unpack the data for this subplot
	Y_F1s = results['f1'][1:]
	X_F1s = range(len(Y_F1s))

	# Convert from accuracy to theta
	Y_thetas = results['accuracy'][1:]
	Y_thetas = [t*2 - 1 for t in Y_thetas]

	X_thetas = map(lambda x: x+width, X_F1s)

	# Plot classifier performance, as F1 and theta
	if not theta_only:
		f1_series = ax1(X_F1s, Y_F1s, width, color='0.25')
		theta_series = ax1.bar(X_thetas, Y_thetas, width, color='0.55')
	else:
		theta_series = ax1.bar(X_F1s, Y_thetas, width, color='0.55')

	# Do some labelling business with the plot
	ax1.tick_params(axis='both', which='major', labelsize=9)
	if not theta_only:
		ax1.set_xticks(X_thetas)
	else:
		ax1.set_xticks([x + width/2. for x in X_F1s])

	xlabels = ['image %d' % (i+1) for i in range(numPics)]
	ax1.set_xticklabels(xlabels, ha='right', rotation=45, size=9)

	# control the plot limits
	padding = 0.25
	xlims = plt.xlim()
	if not theta_only:
		plt.xlim((-padding, numPics -1 + 2*width + padding))
	else:
		plt.xlim((-padding, numPics -1 + width + padding))

	if not theta_only:
		# add a legend
		legend = ax1.legend( 
			(f1_series[0], theta_series[0]), 
			(r'$F_1$ score', r'$\theta_{NB}$'), 
			loc='lower right', prop={'size':9}, labelspacing=0)

	# plot a significance line
	xlims = plt.xlim()
	significance_bar = ax1.plot(xlims, [theta_star, theta_star],
		color='0.35', linestyle=':', zorder=0)
	
	# ** done plotting the results for each image **


	# now make som adjustments
	ylims = plt.ylim()
	plt.ylim(ylims[0], ylims[1] + 0.05)
	plt.draw()
	plt.tight_layout()
	plt.subplots_adjust(wspace=0.05, bottom=.22)
	#fig.subplots_adjust(wspace=0.05, top=0.77, right=0.92, left=0.07, 
	#		bottom=0.24)

	fig.savefig(writeFname)
	plt.show()



def longitudinal_theta_valence(
	readFnames=[
		'f1scores/longitudinal-t0-t1_25.json',
		'orientation/orientation.json'
	],
	writeFname='figs/longitudinal_theta_excess-culture.pdf',
	n=125,
	alpha=0.05
	):
	'''
	Measures the F1 score for a classifier built to distingiush between
	<treatments> based on the labels attributed to a specific image, 
	as a function of the images for all 5 test images.

	This now plots the 95% confidence interval
	'''

	subplotLabels = ['A', 'B']
	image_names = ['image %d' % i for i in range(1,6)]

	# ** Make a shared figure
	figWidth = 8.7 / 2.54 		# convert from PNAS spec in cm to inches
	figHeight = figWidth * 4/5. * 2
	fig = plt.figure(figsize=(figWidth,figHeight))
	gs = gridspec.GridSpec(2,1)



	# ** prepare to plot the classifier data.  
	# Read the data to be plotted and define some constants
	readFname = readFnames[0]
	theta_star = analysis.get_theta_star(n, alpha)
	results = json.loads(open(readFname, 'r').read())
	numPics = 5
	pre_space = 0.75
	width = 0.375	# width of bars
	width = 0.75

	# ** Now, plot the classifier data
	ax_theta = plt.subplot(gs[0])	# plots performance on per-image basis
	plt.setp(ax_theta.get_xticklabels(), visible=False)

	# Unpack the data for this subplot
	Y_F1s = results['f1'][1:]
	X_F1s = range(len(Y_F1s))

	# Convert from accuracy to theta
	Y_thetas = results['accuracy'][1:]
	Y_thetas = [t*2 - 1 for t in Y_thetas]

	X_thetas = map(lambda x: x+width, X_F1s)

	# PLOT!!
	theta_series = ax_theta.bar(X_F1s, Y_thetas, width, color='0.25')

	# Do some labelling business with the plot
	ax_theta.tick_params(axis='both', which='major', labelsize=9)
	ax_theta.set_xticks([x + width/2. for x in X_F1s])

	xlabels = ['image %d' % (i+1) for i in range(numPics)]
	ax_theta.set_xticklabels(xlabels, ha='right', rotation=45, size=9)

	# control the plot limits
	padding = 0.25
	xlims = plt.xlim()
	plt.xlim((-padding, numPics -1 + width + padding))

	# plot a significance line
	xlims = plt.xlim()
	significance_bar = ax_theta.plot(xlims, [theta_star, theta_star],
		color='0.55', linestyle=':')
	
	ax_theta.set_ylabel(r'$\theta_{NB}$', size=9)

	# adjust y axis
	plt.ylim((0, plt.ylim()[1]*1.05))

	# ** done plotting classifier performance

	# ** prepare to plot the excess culture data

	valenceFname = readFnames[1]
	plotData = json.loads(open(valenceFname, 'r').read())

	X = range(len(image_names))

	# If this is the first subplot, keep a reference
	# Also, the first plot has a different x-axis
	ax_valence = plt.subplot(gs[1], sharex=ax_theta)

	X2 = map(lambda x: x + width/2.0, X)

	# first subplot plots panel 3, second plots panel 4
	PANEL = 'panel3'

	thisPlotData = plotData[PANEL]

	yerr = [ye * 1.96 for ye in thisPlotData['std']]
	series =ax_valence.bar(
		X, thisPlotData['avg'], width, color='0.25', 
		ecolor='0.55', yerr=yerr)

	ax_valence.tick_params(axis='both', which='major', labelsize=9)

	# label the y-axis 
	ax_valence.set_ylabel("excess cultural orientation (%)", size=9)

	# adjust y axis
	plt.ylim((0, plt.ylim()[1]*1.05))

	# first subplot also has its own kind of x-axis
	xlabels = image_names
	ax_valence.set_xticks(X2)
	ax_valence.set_xticklabels(xlabels, rotation=45, horizontalalignment='right')

	#Adjustments
	padding = 0.25
	xlims = (-padding, len(X) - 1 + width + padding)
	plt.xlim(xlims)
	ylims = plt.ylim()

	ax_theta.yaxis.set_label_coords(-0.13,0.5)
	ax_valence.yaxis.set_label_coords(-0.15,0.5)

	plt.draw()



	# Label each pannel
	x_pos = 4.4
	ax_theta.text(x_pos, 0.45, 'A', 
			va='top', ha='left', size=20)
	ax_valence.text(x_pos, 35, 'B', 
			va='top', ha='left', size=20)

	plt.tight_layout()
	plt.subplots_adjust(hspace=0.05, left=0.18,bottom=.10)

	fig.savefig(writeFname)
	plt.show()


