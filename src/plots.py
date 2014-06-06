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



def plotAllSpecificityComparisons(readFname='specificity/all.json', 
	writeFname='figs/specificity-allImages.pdf'):
	'''
	Computes all of the interesting specificity comparisons between different
	treatments, such that they can be plotted  in a big multi-pannel figure.
	This does overall specificity comparisons as well as food-specific and
	culture-specific specificity comparisons.

	This only does the computation and writes the results to file; you need
	to run `plotAllSpecificityComparisons()` to generate the plot.
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
			Y = map(lambda x: x['avgNorm'], comparisonData)
			Y_err = map(lambda x: x['stdev']/normStdev, comparisonData)

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

			padding = 0.25
			xlims = (-padding, len(comparisonData) - 1 + width + padding)
			plt.xlim(xlims)

			zero = ax.plot(
				xlims, [0, 0], color='0.35', linestyle='-')

			confidenceIntervalUpper = ax.plot(
				xlims, [CONFIDENCE_95, CONFIDENCE_95], color='0.35',
				linestyle=':')

			confidenceIntervalLower = ax.plot(
				xlims, [-CONFIDENCE_95, -CONFIDENCE_95], color='0.35',
				linestyle=':')

			
			ylims = plt.ylim()
			ypadding = (ylims[1] - ylims[0]) * 0.1
			#plt.ylim(-13, 8)
			plt.ylim(-12, 10)
			#plt.ylim(ylims[0], 10 + ypadding)

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

#def computeAllSpecificityComparisons(sampleSize=126, nullSampleSize=63):
#
#	for image in ['test%d' % i for i in range(5)]:
#		computeSpecificityComparisons(
#			fname='specificity/%s.json' % image,
#			sampleSize=sampleSize, nullSampleSize=nullSampleSize,
#			images=[image])

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

			#TODO print time tracking
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
	fname='specificity/allImages.json', sampleSize=126, nullSampleSize=63, 
	images=['test%d'%i for i in range(5)]):
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



def plotValenceVsImage():
	'''
	This creates a figure with three pannels:
	
	1) Aggregation of all treatments orientation vs image
	2) CULT_img treatment vs image
	3) CULT_fund treatment vs image
	'''

	runs = [
			
			]


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


def computeOrientationVsTreatment(
	fname='orientation/orientation.json', useTestData=False):
	#fname='../docs/figs/valenceComparison.pdf'):
	'''
	This creates a figure with  X pannels:
	1) Orientation vs treatment for image 1
	2) Orientation vs treatment for all images
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
	plotData['panel0'] = {
		'avg':{'cultural':[], 'food':[], 'both':[]},
		'std':{'cultural':[], 'food':[], 'both':[]}
	}
	for image in ['test%d' % i for i in range(5)]:
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
	imageSets = [
		['test%d' % i for i in range(5)],
		['test0']
	]
	for i, images in enumerate(imageSets):
		plotData['panel%d'%(i+1)] = {
			'avg':{'cultural':[], 'food':[], 'both':[]},
			'std':{'cultural':[], 'food':[], 'both':[]}
		}
		thisPlotData =  plotData['panel%d'%(i+1)]

		for treatment in treatments:

			result = a.percentValence([treatment], images)

			thisPlotData['avg']['cultural'].append(result['mean']['cultural'])
			thisPlotData['avg']['food'].append(result['mean']['food'])
			thisPlotData['avg']['both'].append(result['mean']['both'])

			thisPlotData['std']['cultural'].append(result['stdev']['cultural'])
			thisPlotData['std']['food'].append(result['stdev']['food'])
			thisPlotData['std']['both'].append(result['stdev']['both'])

	# Finally we make plots that examine the excess culture orientation
	# as a function of image
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
	for i in range(5):
		plotData['panel3']['avg'][i] = (
				plotData['panel3']['avg'][i] - ambgData['avg'][i])
		plotData['panel3']['std'][i] = (
				plotData['panel3']['std'][i] + ambgData['std'][i])

		plotData['panel4']['avg'][i] = ( 
			plotData['panel4']['avg'][i] - ambgData['avg'][i])
		plotData['panel4']['std'][i] = (
			plotData['panel4']['std'][i] + ambgData['std'][i])

	fh.write(json.dumps(plotData, indent=3))
	fh.close()


def plotExcessCultureVsImage(readFname='orientation/orientation.json',
	writeFname='figs/excessCultureVsTreatment-t1.pdf'):

	images = ['image %d' % i for i in range(1,6)]

	plotData = json.loads(open(readFname, 'r').read())

	width = 0.75

	subplotLabels = ['A', 'B']

	figWidth = 17.4/2.54
	figHeight = figWidth*2/5.
	#figWidth = 8.7/2.54
	#figHeight = figWidth*4/5.
	fig = plt.figure(figsize=(figWidth, figHeight))
	gs = gridspec.GridSpec(1,2)

	X = range(len(images))

	subplots = [0,1]

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

		#plt.ylim((0, plt.ylim()[1]))
		plt.ylim((-20, plt.ylim()[1]))

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


def computeAllF1Accuracy(numReplicates=50, fname='f1scores/all.json'):
	'''
	Computes the F1 score for a naive bayes classifier built to distinguish
	between all the interesting pairings of treatments.

	This function only computes the data; to generate a plot, use 
	`plotAllDisting()`
	'''
	fh = open(fname, 'w')

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
			numReplicates, comparisons)

		# Unpack, and then then repack the batch so it can be saved
		# -- sort of inconvenient isn't it?
		thisSubplotData['f1'] = map(
			lambda c: comparisonResults[c]['f1']['avg'], comparisons)

		thisSubplotData['accuracy'] = map(
			lambda c: comparisonResults[c]['accuracy']['avg'], comparisons)

	fh.write(json.dumps(results, indent=3))
	fh.close()
	return results

def plotAllF1Theta(
	readFname='f1scores/all.json', writeFname='figs/f1-thetas.pdf'):

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
	figHeight = figWidth * 2/5.	# a reasonable aspect ratio
	fig = plt.figure(figsize=(figWidth, figHeight))
	gs = gridspec.GridSpec(1,3, width_ratios=[25,21,17])

	width = 0.325

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
		f1_series = ax.bar(X_F1s, Y_F1s, width, color='0.25')
		theta_series = ax.bar(X_thetas, Y_thetas, width, color='0.55')

		# Label the y-axis, only on the left-most subplot
		#if i == 0:
		#	ax.set_ylabel(r'$\theta_{NB}$', size=14)

		# Let the plot breathe horizontally
		padding = 0.25
		xlims = (-padding, len(Y_thetas) - 1 + 2*width + padding)
		plt.xlim(xlims)

		# Label each pannel
		letterLabel = subplotLabels[i]
		ax.text(-0.05,0.97,letterLabel, 
				va='top', ha='left', size=12)

		# Label the basis treatment as an inset
		basisTreatmentName = TREATMENT_NAMES[basisTreatment]
		bbox_props =  {'facecolor': 'white'}

		if i == 0:
			height= 0.32
			left = len(Y_thetas) - 0.40

		else:
			height= 0.05
			left = len(Y_thetas) - 0.32

		if not i:
			bbox_props['pad'] = 8

		ax.text(left,height,basisTreatmentName, 
				va='bottom', ha='right', bbox=bbox_props, size=12)

		# Put together intelligible labels for the x-axis
		ax.tick_params(axis='both', which='major', labelsize=9)
		xlabels = [TREATMENT_NAMES[t] for t in subplotData['subjects']]
		ax.set_xticks(map(lambda x: x + width, X_F1s))
		ax.set_xticklabels(xlabels, rotation=45, size=9,
			horizontalalignment='right')

		# Add a legend if this is the last panel
		if i == 0:
			legend = ax.legend( 
				(f1_series[0], theta_series[0]), 
				(r'$F_1$ score', r'$\theta_{NB}$'), 
				loc='lower right', prop={'size':9}, labelspacing=0)

		# Tighten up the layout
		plt.draw()
		if i < 1:
			plt.tight_layout()

	fig.subplots_adjust(wspace=0.05, top=0.98, right=0.99, left=0.04, 
			bottom=0.24)
	plt.draw()
	
	fig.savefig(writeFname)
	plt.show()


def plotAllTheta(
	readFname='f1scores/all.json', writeFname='figs/thetas.pdf'):

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


def computeClassificationVsImage(numReplicates=50, 
	treatments=('treatment1','treatment2'), 
	writeFname='f1scores/longitudinal-t1-t2.json'):
	'''
	Measures the F1 score for a classifier built to distingiush between
	<treatments> based on the labels attributed to a specific image, 
	as a function of the images for all 5 test images.
	'''
	fh = open(writeFname, 'w')

	# Assess the classifier's ability to classify as a function of the image 
	# from which the labels used for classification are derived
	a = analysis.NBCAnalyzer()

	results = a.longitudinal(numReplicates, treatments)
	fh.write(json.dumps(results, indent=3))
	fh.close()


def plotClassificationVsImage(
	readFname='f1scores/longitudinal-t1-t2.pdf',
	writeFname='figs/longitudinalF1scores-t1-t2.pdf'):
	'''
	Measures the F1 score for a classifier built to distingiush between
	<treatments> based on the labels attributed to a specific image, 
	as a function of the images for all 5 test images.
	'''

	results = json.loads(open(readFname, 'r').read())

	# Plot the result
	numPics = 5
	width = 0.375	# width of bars


	# Unpack the data for this subplot
	Y_F1s = results['f1']['avg']
	X_F1s = range(len(Y_F1s))

	# Convert from accuracy to theta
	Y_thetas = map(lambda t: t*2 - 1, results['accuracy']['avg'])
	X_thetas = map(lambda x: x+width, X_F1s)


	# Make a plot
	figWidth = 8.7 / 2.54 		# convert from PNAS spec in cm to inches
	figHeight = figWidth * 4/5.
	fig = plt.figure(figsize=(figWidth,figHeight))
	ax = plt.subplot(111)

	# Plot classifier performance, as F1 and theta
	f1_series = ax.bar(X_F1s, Y_F1s, width, color='0.25')
	theta_series = ax.bar(X_thetas, Y_thetas, width, color='0.55')

	# Do some labelling business with the plot
	ax.tick_params(axis='both', which='major', labelsize=9)
	ax.set_xticks(X_thetas)
	xlabels = ['all images'] + ['image %d' % (i+1) for i in range(numPics)]
	ax.set_xticklabels( xlabels, ha='right', rotation=45, size=9)

	padding = 0.25
	plt.xlim((-padding, numPics + 2*width + padding))
	plt.ylim((0,1))

	# add a legend
	legend = ax.legend( 
		(f1_series[0], theta_series[0]), 
		(r'$F_1$ score', r'$\theta_{NB}$'), 
		loc='lower right', prop={'size':9}, labelspacing=0)
	
	plt.draw()
	plt.tight_layout()
	plt.subplots_adjust(bottom=.22)

	fig.savefig(writeFname)
	plt.show()

#CROSS_COMPARISONS = [
#	('treatment1', 'treatment0')
#	, ('treatment2', 'treatment0')
#	, ('treatment1', 'treatment2')
#
#	, ('treatment3', 'treatment5')
#	, ('treatment4', 'treatment6')
#]
#
#def plotDisting(numReplicates, fname, comparisons=CROSS_COMPARISONS):
#	a = analysis.NBCAnalyzer()
#
#	f1Results, stdevResults = a.crossComparison(numReplicates, comparisons)
#
#	# Organize the results into arrays for the purpose of plotting
#	PlotF1Results = map(lambda c: f1Results[c], comparisons)
#	PlotF1Stdevs = map(lambda c: stdevResults[c], comparisons)
#
#	X = range(len(comparisons))
#	width = 0.75
#
#	fig = plt.figure(figsize=(5,5))
#	ax = plt.subplot(111)
#
#	series = ax.bar(X, PlotF1Results, width, color='0.25')
#
#	#series = ax.bar(X, PlotF1Results, width, color='0.25', ecolor='0.55',
#	#	yerr=PlotF1Stdevs)
#
#	ax.set_ylabel("$F_1$-score")
#
#	padding = 0.25
#	xlims = (-padding, len(comparisons) - 1 + width + padding)
#	plt.xlim(xlims)
#
#	# Put together intelligible labels for the x-axis
#	xlabels = [TREATMENT_NAMES[secondTreatment] 
#		for firstTreatment, secondTreatment in comparisons]
#	ax.set_xticks(map(lambda x: x + width /2., X))
#	ax.set_xticklabels(xlabels, rotation=45, horizontalalignment='right')
#
#	fig.subplots_adjust(bottom=.20)
#
#	# fig.tight_layout()
#
#	fig.savefig(fname)
#	plt.show()


#def analyze():
#
#	# The effects of priming are persistent.  These plots show how the effect 
#	# priming varies as we move from one picture to the next
#	plotClassificationVsImage(
#		5, ['treatment4', 'treatment6'], 
#		'../docs/figs/longitude_cult-ing-img-fund.pdf')
#
#	plotClassificationVsImage(
#		5, ['treatment3', 'treatment5'], 
#		'../docs/figs/longitude_cult-ing-fund.pdf')
#
#	plotClassificationVsImage(
#		5, ['treatment1', 'treatment2'], 
#		'../docs/figs/longitude_cult-ing-img.pdf')
#
#	# Here we assess whether the cultural and ingredients treatments are
#	# distinguishable, when priming takes place through different mechanisms
#	plotDisting(5, '../docs/figs/cross-classification.pdf')
#
#	plotValenceComparison()
#
#	plotSpecificityComparison(
#		'treatment1', 
#		['treatment5', 'treatment6', 'treatment2'],
#		10, 'figs/spec-treat0-overall.pdf', 'overall'
#	)
#
#
#def plotAllSpecificity():
#	'''
#	Produces many plots in the same style as `plotAllSpecificityComparisons()`,
#	but intstead of being a single large figure with many pannels, its many
#	small figures.  It is obsolete.
#
#	'''
#
#	for dimension in ['overall', 'cultural', 'food']:
#		plotSpecificityComparison(
#			'treatment1', 
#			['treatment5', 'treatment6', 'treatment2'],
#			50, 'figs/spec-treat1-%s.pdf'%dimension, 'overall'
#		)
#
#		plotSpecificityComparison(
#			'treatment2', 
#			['treatment3', 'treatment4', 'treatment1'],
#			50, 'figs/spec-treat2-%s.pdf'%dimension, 'overall'
#		)
#
#		plotSpecificityComparison(
#			'treatment0', 
#			['treatment1', 'treatment5', 'treatment6', 'treatment2', 
#				'treatment3','treatment4'],
#			50, 'figs/spec-treat0-%s.pdf'%dimension, 'overall'
#		)


