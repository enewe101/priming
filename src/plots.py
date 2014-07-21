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
	

#
#TODO: Review whether the computeAllSpecificityComparisons() function does
#		so in a strict mode or not -- that is, whether calculating food
#		specificity is done only for food words that are non-cultural, or 
#		whether it does so for all words having food as a parent in the 
#		ontology, or whether it does so for food words that are not just
#		non-cultural, but also non-anything-else.  I think that the first
#		way (food words that are non-cultural) is most informative.
#
#TODO: Create a plot that shows the deviation in number of cultural (food)
#		references between each treatment and AMBG overall
#

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



def plotAllSpecificityComparisons(
	readFname='specificity/allImages.json', 
	writeFname='figs/specificity-allImages.pdf',
	normalize=False
	):
	'''
	Computes all of the interesting specificity comparisons between different
	treatments, such that they can be plotted  in a big multi-pannel figure.
	This does overall specificity comparisons as well as food-specific and
	culture-specific specificity comparisons.

	This only does the computation and writes the results to file; you need
	to run `plotAllSpecificityComparisons()` to generate the plot.

	`normalize` controlls whether the plot is normalized into units of 
	standard deviations of the null comparison.  This was used previously, as
	part of a different way of approaching testing the significance of the 
	spceficity results.
	'''

	data = json.loads(open(readFname).read())

	subplotLabels = ['A','B','C','D','E','F','G','H','I']
	basisLabels = []

	
	figWidth = 17.8 / 2.54 			# convert from PNAS spec in cm to inches
	figHeight = (10/11.)*figWidth 	# a reasonable aspect ratio
	fig = plt.figure(figsize=(figWidth,figHeight))
	gs = gridspec.GridSpec(3,3, width_ratios=[25,13,9])
	subplotCounter = 0

	for valenceComparison in data:
		valence = valenceComparison['valence']
		
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
				Y_err = map(lambda x: x['stdev'], comparisonData)

			treatmentNames = map(lambda x: x['subject'], comparisonData)

			if subplotCounter == 0:
				ax = plt.subplot(gs[subplotCounter])
				ax0 = ax	# keep a reference for sharing y-axis

			elif subplotCounter == 1:
				ax = plt.subplot(gs[subplotCounter], sharey=ax0)
				ax1 = ax

			elif subplotCounter == 2:
				ax = plt.subplot(gs[subplotCounter], sharey=ax0)
				ax2 = ax

			elif subplotCounter % 3 == 0:
				ax = plt.subplot(gs[subplotCounter], sharey=ax0, sharex=ax0)

			elif subplotCounter % 3 == 1:
				ax = plt.subplot(gs[subplotCounter], sharey=ax0, sharex=ax1)

			elif subplotCounter % 3 == 2:
				ax = plt.subplot(gs[subplotCounter], sharey=ax0, sharex=ax2)

			series = ax.bar(X, Y, width, color='0.25', ecolor='0.55', 
				yerr=Y_err)

			# control the plot X limits
			padding = 0.25
			xlims = (-padding, len(comparisonData) - 1 + width + padding)
			plt.xlim(xlims)

			# Annotate the plot with a line at Y=0
			zero = ax.plot(
				xlims, [0, 0], color='0.35', linestyle='-')

			# for a normalized plot, annotatet the plot with 95% confidence
			# interval for the null comparison
			if normalize:
				confidenceIntervalUpper = ax.plot(
					xlims, [CONFIDENCE_95, CONFIDENCE_95], color='0.35',
					linestyle=':')

				confidenceIntervalLower = ax.plot(
					xlims, [-CONFIDENCE_95, -CONFIDENCE_95], color='0.35',
					linestyle=':')
			
			# Control the Y limits
			#ylims = plt.ylim()
			#ypadding = (ylims[1] - ylims[0]) * 0.1
			#plt.ylim(-12, 10)

			# determine the basis-treatment label, but don't apply it yet
			basisLabels.append(TREATMENT_NAMES[basis])

			# handle x-axis labelling
			ax.tick_params(axis='both', which='major', labelsize=9)
			if subplotCounter > 5:
				xlabels = [TREATMENT_NAMES[t] for t in treatmentNames]
				ax.set_xticks(map(lambda x: x + width/2., X))
				ax.set_xticklabels(xlabels, rotation=45, 
					horizontalalignment='right')
			else:
				plt.setp(ax.get_xticklabels(), visible=False)

			# handle y-axis labelling
			if subplotCounter == 0:
				ax.set_ylabel("overall specificity", size=9)

			elif subplotCounter == 3:
				ax.set_ylabel("cultural specificity", size=9)

			elif subplotCounter == 6:
				ax.set_ylabel("food specificity", size=9)

			else:
				plt.setp(ax.get_yticklabels(), visible=False)

			subplotCounter += 1
			plt.draw()
			if subplotCounter < 6 :
				plt.tight_layout()

	# We need to apply some labels after everything is plotted, once the
	# axis limits are stable
	for i, ax in enumerate(fig.axes):
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

		# Label the basis treatment as an inset
		basisTreatmentName = basisLabels[i]
		bbox_props =  {'facecolor': 'white'}
		if i%3 == 0:
			pad = 10
			bbox_props['pad'] = pad
			new_x_shim = x_shim + pad*x_range/800.
			new_y_shim = y_range*y_inset_shim_factor + pad*y_range/400.
		else:
			new_x_shim = x_shim
			new_y_shim = y_range*y_inset_shim_factor
		ax.text(xmax - new_x_shim, ymin + new_y_shim,
			basisTreatmentName, ha='right', va='bottom', bbox=bbox_props,
			size=9)

	plt.subplots_adjust(left=0.07, top=0.99, right=0.99, 
		bottom=.11, wspace=0.05, hspace=0.05)
	fig.savefig(writeFname)
	plt.show()


def computeAllSpecificities(sampleSize=126, nullSampleSize=63,
		images=['test%d'%i for i in range(5)] ):
	valences = ['overall', 'cultural', 'food']

	comparisonSchedule = {
		'treatment0': ['treatment1', 'treatment5', 'treatment6',
			'treatment2', 'treatment3', 'treatment4']

		, 'treatment1': ['treatment5', 'treatment6', 'treatment2']

		, 'treatment2': ['treatment3', 'treatment4']
	}

	for valence in ['overall', 'cultural', 'food']:
		for basis in ['treatment0', 'treatment1', 'treatment2']:
			subjects = comparisonSchedule[basis]
			
			start = time.time()

			computeSpecificity(
				basis=basis,
				subjects=subjects,
				valence=valence,
				sampleSize=sampleSize,
				nullSampleSize=nullSampleSize,
				images=images)

			print '   that took %d min.\n' % int((time.time() - start)/60)


def csa(	# "csa" = "computesSpecificityAllImages"
	basis='treatment0', 
	subjects=['treatment%d'%i for i in [1,5,6,2,3,4]],
	valence='overall',
	sampleSize=126,
	nullSampleSize=63
	):

	'''
	Runs computeSpecificity with images equal to the set of all images,
	as well as each image in isolation.
	'''
	computeSpecificity(basis, subjects, valence, sampleSize, 
		nullSampleSize, ['test%d'%i for i in range(5)])

	for i in range(5):
		computeSpecificity(basis, subjects, valence, sampleSize, 
			nullSampleSize, ['test%d'%i])


def computeSpecificity(
	basis='treatment0', 
	subjects=['treatment%d'%i for i in [1,5,6,2,3,4]],
	valence='overall',
	sampleSize=126,
	nullSampleSize=63,
	images=['test%d'%i for i in range(5)]
	):

	'''
	Similar to computeSpecificityComparisons, but only one basis and 
	valence are computed at a time.
	'''

	start = time.time()

	# Make an analyzer object -- it performs the actual comparisons
	a = analysis.Analyzer()

	# A results object to aggregate all the data
	fname = 'specificity/%s-%s-%s-%s.json' %(
		basis, ''.join(subjects), ''.join(images), valence)

	fh = open(fname, 'w')
	results = []

	# For the valence passed,
	thisValenceResults = {'valence': valence, 'results': []}
	results.append(thisValenceResults)

	print '   valence: %s' % valence

	# And for the basis treatment passed, 
	print '      basis: %s' % basis

	thisBasisResults = {'basis': basis, 'results':[]}
	thisValenceResults['results'].append(thisBasisResults)

	# first compute the null comparison.  This determines the variance
	# observed when comparing samples when they are both taken from 
	# the basis treatment, and establishes confidence intervals
	rslt = a.compareValenceSpecificity(
		valence, basis, basis, nullSampleSize, images)

	nullComparison = {
		'subject':'null',
		'stdev':rslt['stdMoreMinusLess'],
		'avg':rslt['avgMoreMinusLess']
	}

	thisBasisResults['null'] = nullComparison

	# And for each subject treatment
	for subject in subjects:

		print '      subject: %s' % subject

		# compare basis treatment to subject treatment
		rslt = a.compareValenceSpecificity(
			valence, subject, basis, sampleSize, images)

		# express the avg specificity in terms of the standard 
		# deviations of the null comparison.  Store the result
		avgNorm = rslt['avgMoreMinusLess'] / nullComparison['stdev']

		subjectComparison = {
			'subject': subject,
			'stdev': rslt['stdMoreMinusLess'],
			'avg': rslt['avgMoreMinusLess'],
			'avgNorm': avgNorm
		}

		thisBasisResults['results'].append(subjectComparison)

	fh.write(json.dumps(results, indent=3))
	fh.close

	print '   that took %d min.\n' % int((time.time() - start)/60)
	return


def computeSpecificityComparisons(
	fname='specificity/allImages.json',
	sampleSize=126,
	nullSampleSize=63, 
	images=['test%d'%i for i in range(5)]
	):
	'''
	Computes all of the interesting specificity comparisons between different
	treatments, such that they can be plotted  in a big multi-pannel figure.
	This does overall specificity comparisons as well as food-specific and
	culture-specific specificity comparisons.

	This only does the computation and writes the results to file; you need
	to run `plotAllSpecificityComparisons()` to generate the plot.
	'''

	comparisonSchedule = {
		'treatment0': ['treatment1', 'treatment5', 'treatment6',
			'treatment2', 'treatment3', 'treatment4']

		, 'treatment1': ['treatment5', 'treatment6', 'treatment2']

		, 'treatment2': ['treatment3', 'treatment4']
	}

	print '\nComparison based on images: ' + str(images)

	# Make an analyzer object -- it performs the actual comparisons
	a = analysis.Analyzer()

	# A results object to aggregate all the data
	fh = open(fname, 'w')
	results = []

	for valence in ['overall', 'cultural', 'food']:
		thisValenceResults = {'valence': valence, 'results': []}
		results.append(thisValenceResults)

		print '   Valence: %s' % valence

		for basisTreatment in ['treatment0', 'treatment1', 'treatment2']:

			print '      basisTreatment: %s' % basisTreatment
			
			thisBasisResults = {'basis': basisTreatment, 'results':[]}
			thisValenceResults['results'].append(thisBasisResults)

			# first compute the null comparison.  This determines the variance
			# observed when comparing samples when they are both taken from 
			# the basis treatment, and establishes confidence intervals
			rslt = a.compareValenceSpecificity(
				valence, basisTreatment, basisTreatment, nullSampleSize, 
				images)

			nullComparison = {
				'subject':'null',
				'stdev':rslt['stdMoreMinusLess'],
				'avg':rslt['avgMoreMinusLess']
			}

			thisBasisResults['null'] = nullComparison

			# Now compare the basis treatment to each subject treatment
			for subjectTreatment in comparisonSchedule[basisTreatment]:

				print '      subjectTreatment: %s' % subjectTreatment

				# compare basis treatment to subject treatment
				rslt = a.compareValenceSpecificity(
					valence, subjectTreatment, basisTreatment, sampleSize,
					images)

				# express the avg specificity in terms of the standard 
				# deviations of the null comparison.  Store the result
				avgNorm = rslt['avgMoreMinusLess'] / nullComparison['stdev']

				subjectComparison = {
					'subject': subjectTreatment,
					'stdev': rslt['stdMoreMinusLess'],
					'avg': rslt['avgMoreMinusLess'],
					'avgNorm': avgNorm
				}

				thisBasisResults['results'].append(subjectComparison)

	fh.write(json.dumps(results, indent=3))
	fh.close

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
	

def plotOrientationVsTreatment(readFname='orientation/orientation.json',
	writeFname='figs/orientationVsTreatment.pdf'):

	treatmentIds = [0,1,5,6,2,3,4]
	treatments = ['treatment%d' % i for i in treatmentIds]
	images = ['image %d' % i for i in range(1,6)]

	plotData = json.loads(open(readFname, 'r').read())

	width = 0.375

	subplotLabels = ['A', 'B', 'C']

	figWidth = 17.8 / 2.54
	figHeight = figWidth*4/10.
	fig = plt.figure(figsize=(figWidth,figHeight))
	gs = gridspec.GridSpec(1,3, width_ratios=[25,33,33])

	for subplot in [0,1,2]:

		# If this is the first subplot, keep a reference
		# Also, the first plot has a different x-axis
		if subplot == 0:
			ax = plt.subplot(gs[subplot])
			ax0 = ax
			X = range(len(images))

		# If this is the second subplot, make it's y-axis linked to the first
		else:
			ax = plt.subplot(gs[subplot], sharey=ax0)
			X = range(len(treatments))

		X2 = map(lambda x: x + width, X)
		panel = 'panel%d' % subplot

		thisPlotData = plotData[panel]

		seriesCultural =ax.bar(
			X, thisPlotData['avg']['cultural'], width, color='0.25', 
			ecolor='0.55', 
			bottom=plotData[panel]['avg']['both'], 
			yerr=thisPlotData['std']['cultural'])

		seriesFood = ax.bar(
			X2, thisPlotData['avg']['food'], width, color='0.55',
			ecolor='0.25', 
			bottom=plotData[panel]['avg']['both'], 
			yerr=thisPlotData['std']['food'])
			
		seriesBoth1 =ax.bar(
			X, thisPlotData['avg']['both'], width, color='0.85',
			ecolor='0.55')
			#yerr=thisPlotData['std']['both'])

		seriesBoth2 =ax.bar(
			X2, thisPlotData['avg']['both'], width, color='0.85',
			ecolor='0.25')
			#yerr=thisPlotData['std']['both'])

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
	

# This is deprecated in place of plotOrientationVsTreatmen
def plotValenceComparison(
	fname='../docs/figs/valenceComparison.pdf', image='test0'):

	treatmentIds = [0,1,5,6,2,3,4]
	treatments = ['treatment%d' % i for i in treatmentIds]

	a = analysis.Analyzer()
	percentValences = {'cultural':[], 'food':[], 'both':[]}
	stdValences = {'cultural':[], 'food':[], 'both':[]}

	for treatment in treatments:

		result = a.percentValence(treatment, image)

		percentValences['cultural'].append(result['mean']['cultural'])
		percentValences['food'].append(result['mean']['food'])
		percentValences['both'].append(result['mean']['both'])

		stdValences['cultural'].append(result['stdev']['cultural'])
		stdValences['food'].append(result['stdev']['food'])
		stdValences['both'].append(result['stdev']['both'])

	width = 0.375
	X = range(len(treatments))
	X2 = map(lambda x: x + width, X)
	
	fig = plt.figure(figsize=(5,5))
	ax = plt.subplot(111)

	seriesCultural =ax.bar(
		X, percentValences['cultural'], width, color='0.25', ecolor='0.55', 
		bottom=percentValences['both'], yerr=stdValences['cultural'])

	seriesFood = ax.bar(
		X2, percentValences['food'], width, color='0.55', ecolor='0.25', 
		bottom=percentValences['both'], yerr=stdValences['food'])
		
	seriesBoth1 =ax.bar(
		X, percentValences['both'], width, color='0.85', ecolor='0.55', 
		yerr=stdValences['both'])

	seriesBoth2 =ax.bar(
		X2, percentValences['both'], width, color='0.85', ecolor='0.25', 
		yerr=stdValences['both'])


	# Let the plot breathe horizontally
	padding = 0.25
	xlims = (-padding, len(treatments) - 1 + 2*width + padding)
	plt.xlim(xlims)
	#plt.ylim((0,55))

	ax.set_ylabel("percent of labels having orientation", size=14)

	xlabels = [TREATMENT_NAMES[t] for t in treatments]
	ax.set_xticks(X2)
	ax.set_xticklabels(xlabels, rotation=45, horizontalalignment='right')

	fig.subplots_adjust(bottom=.20)

	legend = ax.legend( (seriesCultural[0], seriesFood[0], seriesBoth1[0]),
			('cultural', 'food', 'both'), loc='upper left', prop={'size':10})

	fig.savefig(fname)
	plt.show()
	
	print percentValences


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
		else:
			ax = plt.subplot(gs[i], sharey=ax0)
			plt.setp(ax.get_yticklabels(), visible=False)

		# Plot the bar plot
		if not theta_only:
			f1_series = ax.bar(X_F1s, Y_F1s, width, color='0.25')
			theta_series = ax.bar(X_thetas, Y_thetas, width, color='0.55')
		else:
			theta_series = ax.bar(X_F1s, Y_thetas, width, color='0.55')

		# Label the y-axis, only on the left-most subplot
		#if i == 0:
		#	ax.set_ylabel(r'$\theta_{NB}$', size=14)

		# Let the plot breathe horizontally
		padding = 0.25
		if not theta_only:
			xlims = (-padding, len(Y_thetas) - 1 + 2*width + padding)
		else:
			xlims = (-padding, len(Y_thetas) - 1 + width + padding)
		plt.xlim(xlims)

		# Label each pannel
		#letterLabel = subplotLabels[i]
		#ax.text(-0.05,0.97,letterLabel, 
		#		va='top', ha='left', size=12)


		# Put together intelligible labels for the x-axis
		ax.tick_params(axis='both', which='major', labelsize=9)
		xlabels = [TREATMENT_NAMES[t] for t in subplotData['subjects']]
		if not theta_only:
			ax.set_xticks(map(lambda x: x + width, X_F1s))
		else:
			ax.set_xticks(map(lambda x: x + width/2., X_F1s))
		ax.set_xticklabels(xlabels, rotation=45, size=9,
			horizontalalignment='right')

		# Add a legend if this is the last panel
		#if i == 0:
		#	legend = ax.legend( 
		#		(f1_series[0], theta_series[0]), 
		#		(r'$F_1$ score', r'$\theta_{NB}$'), 
		#		loc='lower right', prop={'size':9}, labelspacing=0)

		significance_bar = ax.plot(
			xlims, [theta_star, theta_star], color='0.35', linestyle=':', zorder=0)

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
		left = len(subplotData['accuracy'])/2.0 - width
		ylims = plt.ylim()
		# put the label directly above the plot.  
		# The first label needs to be put a bit higher.
		height= ylims[1] + (0.02 if i else 0.06)
		ax.text(left, height, basisTreatmentName, 
				va='bottom', ha='left', size=9, rotation=45)


	y_low, y_high = plt.ylim()
	plt.ylim(y_low-0.04, y_high)

	fig.subplots_adjust(wspace=0.05, top=0.77, right=0.92, left=0.07, 
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

