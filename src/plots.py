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
from matplotlib.ticker import FixedLocator, LinearLocator, FixedFormatter

# z-score for two-tailed 99-percent confidence interval
CONFIDENCE_99 = 2.975

TREATMENT_NAMES = {
	'treatment0': r'ambg'
	, 'treatment1': 'cult$_{img}$'
	, 'treatment2': 'ingr$_{img}$'
	, 'treatment3': 'ingr$_{fund}$'
	, 'treatment4': 'ingr$_{fund,img}$'
	, 'treatment5': 'cult$_{fund}$'
	, 'treatment6': 'cult$_{fund,img}$'
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
	plotCrossComparison(5, '../docs/figs/cross-classification.pdf')

	treatments = ['treatment0', 'treatment1', 'treatment5', 'treatment6',
			'treatment2', 'treatment3', 'treatment4']
	plotValenceComparison(treatments, '../docs/figs/valenceComparison.pdf')



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
		xlims, [0, 0], color='0.75', linestyle='-')

	confidenceIntervalUpper = ax.plot(
		xlims, [CONFIDENCE_99, CONFIDENCE_99], color='0.75', linestyle=':')

	confidenceIntervalLower = ax.plot(
		xlims, [-CONFIDENCE_99, -CONFIDENCE_99], color='0.75', linestyle=':')

	xlabels = [TREATMENT_NAMES[t] for t in treatmentsToBeCompared]

	
	ylims = plt.ylim()
	ypadding = (ylims[1] - ylims[0]) * 0.05
	plt.ylim(ylims[0] - ypadding, ylims[1] + ypadding)

	ax.set_xticks(map(lambda x: x + width/2., X))
	ax.set_xticklabels(xlabels, rotation=45, horizontalalignment='right')
	ax.set_ylabel("excess specific words")


	fig.subplots_adjust(bottom=.20)

	plt.show()



def plotValenceComparison(treatments, fname):
	a = analysis.Analyzer()
	percentValences = {'cultural':[], 'food':[]}
	stdevValences = {'cultural':[], 'food':[]}
	for treatment in treatments:

		result = a.percentValence('cultural', treatment, 'test0')
		percentValences['cultural'].append(result['mean'])
		stdevValences['cultural'].append(result['stdev'])

		result = a.percentValence('food', treatment, 'test0')
		percentValences['food'].append(result['mean'])
		stdevValences['food'].append(result['stdev'])

	
	width = 0.35
	X = range(len(treatments))
	X2 = map(lambda x: x + width, X)
	
	fig, ax = plt.subplots()

	seriesCultural =ax.bar(
		X, percentValences['cultural'], width, color='0.25', ecolor='0.55', 
		yerr=stdevValences['cultural'])

	seriesFood = ax.bar(
		X2, percentValences['food'], width, color='0.55', ecolor='0.25', 
		yerr=stdevValences['food'])
		

	xlabels = [TREATMENT_NAMES[t] for t in treatments]
	ax.set_ylabel("percent words with valence")


	# ax.set_xticks(map(lambda x: x + width /2., X))
	# ax.set_xticklabels(xlabels, rotation=45)

	xmajorlocator = FixedLocator(map(lambda x: x + width, X))
	ax.xaxis.set_major_locator(xmajorlocator)

	xmajorformatter = FixedFormatter(xlabels)
	ax.xaxis.set_major_formatter(xmajorformatter)

	labels = [tick.label1 for tick in ax.xaxis.get_major_ticks()]
	for label in labels:
		label.set_horizontalalignment('right')
		label.set_rotation(45)

	fig.subplots_adjust(bottom=.20)

	legend = ax.legend((seriesCultural[0], seriesFood[0]), ('cultural', 'food'), 
		loc='lower right')
	

	fig.savefig(fname)
	plt.show()
	
	print percentValences




def plotCrossComparison(numReplicates, fname, comparisons=CROSS_COMPARISONS):
	a = analysis.NBCAnalyzer()
	f1Results, stdevResults = a.crossComparison(numReplicates, comparisons)

	# Organize the results into arrays for the purpose of plotting
	PlotF1Results = map(lambda c: f1Results[c], comparisons)
	PlotF1Stdevs = map(lambda c: stdevResults[c], comparisons)

	print PlotF1Results
	print PlotF1Stdevs

	# Put together intelligible labels for the x-axis
	xlabels = []
	for comparison in comparisons:
		firstTreatment, secondTreatment = comparison
		xlabels.append("%s \nvs %s" % (
			TREATMENT_NAMES[firstTreatment], TREATMENT_NAMES[secondTreatment]))


	X = range(len(comparisons))
	width = 0.35
	
	fig, ax = plt.subplots()

	series = ax.bar(X, PlotF1Results, width, color='0.25', ecolor='0.55',
		yerr=PlotF1Stdevs)

	ax.set_ylabel("F1 Score")


	# ax.set_xticks(map(lambda x: x + width /2., X))
	# ax.set_xticklabels(xlabels, rotation=45)

	xmajorlocator = FixedLocator(map(lambda x: x + width /2., X))
	ax.xaxis.set_major_locator(xmajorlocator)

	xmajorformatter = FixedFormatter(xlabels)
	ax.xaxis.set_major_formatter(xmajorformatter)

	labels = [tick.label1 for tick in ax.xaxis.get_major_ticks()]
	for label in labels:
		label.set_horizontalalignment('right')
		label.set_rotation(45)

	fig.subplots_adjust(bottom=.20)

	# fig.tight_layout()

	fig.savefig(fname)
	plt.show()





def plotClassificationVsImage(numClassificationReplicates, treatments, fname):

	# Assess the classifier's ability to classify as a function of the image 
	# from which the labels used for classification are derived
	a = analysis.NBCAnalyzer()
	f1Results, stdevResults = a.longitudinal(
			numClassificationReplicates, treatments)

	# Plot the result
	numPics = 5
	width = 0.35						# width of bars
	X1 = range(numPics + 1)				# x-position of series1 bars
	X2 = map(lambda x: x + width, X1)	# x-position of series2 bars
	xlabels = ['all images'] + ['Image %d' % i for i in range(numPics)]

	# Make a plot
	fig, ax = plt.subplots()

	# Plot the performance with position-based labels
	series1 = ax.bar(
		X1, f1Results['withPosition'], width, color='0.25',  ecolor='0.55',
		yerr=stdevResults['withPosition'])

	# Plot the performance without position-less labels
	series1 = ax.bar(
		X2, 
		f1Results['withoutPosition'], width, color='0.55', ecolor='0.25',
		yerr=stdevResults['withoutPosition'])

	# Do some labelling business with the plot
	ax.set_ylabel("F1 Score")
	ax.set_xticks(X2)
	ax.set_xticklabels( xlabels )
	

	fig.savefig(fname)
	plt.show()

