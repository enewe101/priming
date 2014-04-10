import json
import util
import analysis
import sys
import random
import naive_bayes
import ontology
import data_processing
import numpy as np
import matplotlib.pyplot as plt
import matplotlib 
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FixedLocator, LinearLocator, FixedFormatter

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

CROSS_COMPARISONS = [
	('treatment1', 'treatment0')
	, ('treatment2', 'treatment0')
	, ('treatment1', 'treatment2')

	, ('treatment3', 'treatment5')
	, ('treatment4', 'treatment6')
]

def analyze():

	# The effects of priming are persistent.  These plots show how the effect 
	# priming varies as we move from one picture to the next
	plotClassificationVsImage(
		5, ['treatment4', 'treatment6'], 
		'../docs/figs/longitude_cult-ing-img-fund.pdf')

	plotClassificationVsImage(
		5, ['treatment3', 'treatment5'], 
		'../docs/figs/longitude_cult-ing-fund.pdf')

	plotClassificationVsImage(
		5, ['treatment1', 'treatment2'], 
		'../docs/figs/longitude_cult-ing-img.pdf')


	# Here we assess whether the cultural and ingredients treatments are
	# distinguishable, when priming takes place through different mechanisms
	plotDisting(5, '../docs/figs/cross-classification.pdf')

	plotValenceComparison()

	plotSpecificityComparison(
		'treatment1', 
		['treatment5', 'treatment6', 'treatment2'],
		10, 'figs/spec-treat0-overall.pdf', 'overall'
	)


def plotAllSpecificity():


	for dimension in ['overall', 'cultural', 'food']:
		plotSpecificityComparison(
			'treatment1', 
			['treatment5', 'treatment6', 'treatment2'],
			50, 'figs/spec-treat1-%s.pdf'%dimension, 'overall'
		)

		plotSpecificityComparison(
			'treatment2', 
			['treatment3', 'treatment4', 'treatment1'],
			50, 'figs/spec-treat2-%s.pdf'%dimension, 'overall'
		)

		plotSpecificityComparison(
			'treatment0', 
			['treatment1', 'treatment5', 'treatment6', 'treatment2', 
				'treatment3','treatment4'],
			50, 'figs/spec-treat0-%s.pdf'%dimension, 'overall'
		)

def plotAllSpecificityComparisons(readFname='specificity/all.json', 
	writeFname='figs/specificity.pdf'):

	data = json.loads(open(readFname).read())

	subplotLabels = ['A','B','C','D','E','F','G','H','I']

	fig = plt.figure(figsize=(10,11))
	gs = gridspec.GridSpec(3,3, width_ratios=[25,13,9])
	subplotCounter = 0


	for valenceComparison in data:
		valence = valenceComparison['valence']

		
		for basisComparison in valenceComparison['results']:
			basis = basisComparison['basis']

			comparisonData = basisComparison['results']

			width=0.75
			X = range(len(comparisonData))
			Y = map(lambda x: x['avgNorm'], comparisonData)
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



			series = ax.bar(X, Y, width, color='0.25')

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
			ypadding = (ylims[1] - ylims[0]) * 0.05
			plt.ylim(ylims[0] - ypadding, ylims[1] + ypadding)

			# Label each pannel
			letterLabel = subplotLabels[subplotCounter]
			ax.text(len(comparisonData) - 0.2, 9.9, letterLabel, 
					va='top', ha='right', size=20)

			# Label the basis treatment as an inset
			basisTreatmentName = TREATMENT_NAMES[basis]
			bbox_props =  {'facecolor': 'white'}
			if subplotCounter%3 == 0:
				bbox_props['pad'] = 8
			ax.text(len(Y)-0.3,-9.4,basisTreatmentName, 
					ha='right', va='bottom', bbox=bbox_props)

			# handle x-axis labelling
			if subplotCounter > 5:
				xlabels = [TREATMENT_NAMES[t] for t in treatmentNames]
				ax.set_xticks(map(lambda x: x + width/2., X))
				ax.set_xticklabels(xlabels, rotation=45, 
					horizontalalignment='right')
			else:
				plt.setp(ax.get_xticklabels(), visible=False)

			# handle y-axis labelling
			if subplotCounter == 0:
				ax.set_ylabel("overall specificity")

			elif subplotCounter == 3:
				ax.set_ylabel("cultural specificity")

			elif subplotCounter == 6:
				ax.set_ylabel("food specificity")

			else:
				plt.setp(ax.get_yticklabels(), visible=False)



			subplotCounter += 1
			plt.draw()
			if subplotCounter < 6 :
				plt.tight_layout()

	plt.subplots_adjust(bottom=.10)
	fig.savefig(writeFname)
	plt.show()


def computeAllSpecificityComparisons(
	fname='specificity/all.json', numToCompare=50):

	comparisonSchedule = {
		'treatment0': ['treatment1', 'treatment5', 'treatment6',
			'treatment2', 'treatment3', 'treatment4']

		, 'treatment1': ['treatment5', 'treatment6', 'treatment2']

		, 'treatment2': ['treatment3', 'treatment4']
	}

	# Make an analyzer object -- it performs the actual comparisons
	a = analysis.Analyzer()

	# A results object to aggregate all the data
	fh = open(fname, 'w')
	results = []

	for valence in ['overall', 'cultural', 'food']:
		thisValenceResults = {'valence': valence, 'results': []}
		results.append(thisValenceResults)

		for basisTreatment in ['treatment0', 'treatment1', 'treatment2']:
			thisBasisResults = {'basis': basisTreatment, 'results':[]}
			thisValenceResults['results'].append(thisBasisResults)

			# first compute the null comparison.  This determines the variance
			# observed when comparing samples when they are both taken from 
			# the basis treatment, and establishes confidence intervals
			rslt = a.compareValenceSpecificity(
				valence, basisTreatment, basisTreatment, numToCompare)

			nullComparison = {
				'subject':'null',
				'stdev':rslt['stdMoreMinusLess'],
				'avg':rslt['avgMoreMinusLess']
			}

			thisBasisResults['null'] = nullComparison

			# Now compare the basis treatment to each subject treatment
			for subjectTreatment in comparisonSchedule[basisTreatment]:

				# compare basis treatment to subject treatment
				rslt = a.compareValenceSpecificity(
					valence, subjectTreatment, basisTreatment, numToCompare)

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





def plotSpecificityComparison(basisTreatment, treatmentsToBeCompared, 
	numToCompare, fname, dimension='overall'):

	# The dimension specifies whether we are comparing specificity overall, or
	# in only with respect to cultural- or food-related tokens
	assert(dimension in ['overall', 'cultural', 'food'])

	# Make an analyzer to carry out the analysis
	a = analysis.Analyzer()

	# Make some containers to aggregate the results
	avgSpecificities = []
	stdSpecificities = []

	# For each treatment to be compared, do the comparison, and record the results
	# A comparison is made for food-words, cultural-words, and overall specificity
	for treatment in treatmentsToBeCompared:

		if dimension == 'overall':
			result = a.compareSpecificity(treatment, basisTreatment, numToCompare)

		elif dimension == 'cultural':
			result = a.compareCulturalSpecificity(
				treatment, basisTreatment, numToCompare)

		elif dimension == 'food':
			result = a.compareFoodSpecificity(
				treatment, basisTreatment, numToCompare)

		avgSpecificities.append(result['avgMoreMinusLess'])
		stdSpecificities.append(result['stdMoreMinusLess'])
	
	# In order to show statistical significance, we do a null-comparison of 
	# the basis treatment to itself

	# null overall comparison
	if dimension == 'overall':
		result = a.compareSpecificity(basisTreatment, basisTreatment, numToCompare)

	elif dimension == 'cultural':
		result = a.compareCulturalSpecificity(
			basisTreatment, basisTreatment, numToCompare)

	elif dimension == 'food':
		result = a.compareFoodSpecificity(
			basisTreatment, basisTreatment, numToCompare)

	avgNullSpecificities = result['avgMoreMinusLess']
	stdNullSpecificities = result['stdMoreMinusLess']

	# convert the specificities into numbers of standard deviations of the null
	# comparison

	avgSpecificities = map(lambda s: s/stdNullSpecificities, avgSpecificities)


	width=0.7
	X = range(len(treatmentsToBeCompared))


	fig = plt.figure(figsize=(5,5))
	ax = fig.add_subplot(1,1,1)

	series = ax.bar(X, avgSpecificities, width, color='0.25')


	padding = len(treatmentsToBeCompared) * 0.05
	xlims = (-padding, len(treatmentsToBeCompared) - 1 + width + padding)
	plt.xlim(xlims)

	zero = ax.plot(
		xlims, [0, 0], color='0.35', linestyle='-')

	confidenceIntervalUpper = ax.plot(
		xlims, [CONFIDENCE_95, CONFIDENCE_95], color='0.35', linestyle=':')

	confidenceIntervalLower = ax.plot(
		xlims, [-CONFIDENCE_95, -CONFIDENCE_95], color='0.35', linestyle=':')

	xlabels = [TREATMENT_NAMES[t] for t in treatmentsToBeCompared]

	
	ylims = plt.ylim()
	ypadding = (ylims[1] - ylims[0]) * 0.05
	plt.ylim(ylims[0] - ypadding, ylims[1] + ypadding)

	ax.set_xticks(map(lambda x: x + width/2., X))
	ax.set_xticklabels(xlabels, rotation=45, horizontalalignment='right')
	ax.set_ylabel("excess specific words")

	fig.subplots_adjust(bottom=.20)
	fig.savefig(fname)

	plt.show()



def plotValenceComparison(fname='../docs/figs/valenceComparison.pdf'):

	treatmentIds = [0,1,5,6,2,3,4]
	treatments = ['treatment%d' % i for i in treatmentIds]

	a = analysis.Analyzer()
	percentValences = {'cultural':[], 'food':[], 'both':[]}
	stdValences = {'cultural':[], 'food':[], 'both':[]}

	for treatment in treatments:

		result = a.percentValence(treatment, 'test0')

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
	plt.ylim((0,55))

	ax.set_ylabel("percent of labels having orientation", size=14)

	xlabels = [TREATMENT_NAMES[t] for t in treatments]
	ax.set_xticks(X2)
	ax.set_xticklabels(xlabels, rotation=45, horizontalalignment='right')

	fig.subplots_adjust(bottom=.20)

	legend = ax.legend( (seriesCultural[0], seriesFood[0], seriesBoth1[0]),
			('cultural', 'food', 'both'), loc='top left', prop={'size':10})

	fig.savefig(fname)
	plt.show()
	
	print percentValences




def computeAllDisting(numReplicates=50, fname='f1scores/all.json'):

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
			, 'results':[]
		}
		results.append(thisSubplotData)

		basisTreatment = comp['basis']

		# build the set of all comparisons to be presented in one subplot
		# NBCAnalyzer() will perform all these comparisons as a batch
		comparisons = [(basisTreatment, subjectTreatment) for
				subjectTreatment in comp['subjects']]

		# Compute the results for the batch of comparisons
		f1Results, stdevResults = a.crossComparison(
			numReplicates, comparisons)

		# Unpack, and then then repack the batch so it can be saved
		# -- sort of inconvenient isn't it?
		thisSubplotData['results'] = map(lambda c: f1Results[c], comparisons)

	fh.write(json.dumps(results, indent=3))
	fh.close()
	return results


def plotAllDisting(
	readFname='f1scores/all.json', writeFname='figs/f1scores.pdf'):

	subplotLabels = ['A','B','C']

	# Read the data from file
	f1scores = json.loads(open(readFname, 'r').read())

	# Start a figure 
	# -- Assumes specific shape of data produced by computeAllDisting()!
	fig = plt.figure(figsize=(10,4))
	gs = gridspec.GridSpec(1,3, width_ratios=[25,21,17])
	subplotCounter = 0

	for subplotData in f1scores:

		# Unpack the data for this subplot
		basisTreatment = subplotData['basis']
		Y_f1scores = subplotData['results']
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



def plotDisting(numReplicates, fname, comparisons=CROSS_COMPARISONS):
	a = analysis.NBCAnalyzer()

	f1Results, stdevResults = a.crossComparison(numReplicates, comparisons)

	# Organize the results into arrays for the purpose of plotting
	PlotF1Results = map(lambda c: f1Results[c], comparisons)
	PlotF1Stdevs = map(lambda c: stdevResults[c], comparisons)

	X = range(len(comparisons))
	width = 0.75

	fig = plt.figure(figsize=(5,5))
	ax = plt.subplot(111)

	series = ax.bar(X, PlotF1Results, width, color='0.25')

	#series = ax.bar(X, PlotF1Results, width, color='0.25', ecolor='0.55',
	#	yerr=PlotF1Stdevs)

	ax.set_ylabel("$F_1$-score")

	padding = 0.25
	xlims = (-padding, len(comparisons) - 1 + width + padding)
	plt.xlim(xlims)

	# Put together intelligible labels for the x-axis
	xlabels = [TREATMENT_NAMES[secondTreatment] 
		for firstTreatment, secondTreatment in comparisons]
	ax.set_xticks(map(lambda x: x + width /2., X))
	ax.set_xticklabels(xlabels, rotation=45, horizontalalignment='right')

	fig.subplots_adjust(bottom=.20)

	# fig.tight_layout()

	fig.savefig(fname)
	plt.show()





def plotClassificationVsImage(numReplicates=50, 
	treatments=('treatment1','treatment2'), fname='figs/longitudinalF1scores.pdf'):

	# Assess the classifier's ability to classify as a function of the image 
	# from which the labels used for classification are derived
	a = analysis.NBCAnalyzer()
	f1Results, stdevResults = a.longitudinal(
			numReplicates, treatments)

	# Plot the result
	numPics = 5
	width = 0.375						# width of bars
	X1 = range(numPics + 1)				# x-position of series1 bars
	X2 = map(lambda x: x + width, X1)	# x-position of series2 bars
	xlabels = ['all images'] + ['image %d' % (i+1) for i in range(numPics)]

	# Make a plot
	fig = plt.figure(figsize=(5,5))
	ax = plt.subplot(111)

	# Plot the performance with position-based labels
	series1 = ax.bar(
		X1, f1Results['withPosition'], width, color='0.25')

	# Plot the performance without position-less labels
	series1 = ax.bar(
		X2, 
		f1Results['withoutPosition'], width, color='0.55')

	# Do some labelling business with the plot
	ax.set_ylabel("$F_1$-score", size=14)
	ax.set_xticks(X2)
	ax.set_xticklabels( xlabels, ha='right', rotation=45 )

	padding = 0.25
	plt.xlim((-padding, numPics + 2*width + padding))
	plt.ylim((0,1))
	
	plt.draw()
	plt.tight_layout()
	plt.subplots_adjust(bottom=.16)

	fig.savefig(fname)
	plt.show()

