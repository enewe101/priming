'''
This module provides two classes that perform certain analyses on the data.
They call on classes in `data_processing`, `naive_bayes`, and `ontology` to do 
so.

`NBCAnalyzer` is used to analyze the capacity of a naive bayes classifier
in being able to distinguish between different treatments.  It provides
different kinds of analyses as methods.  One example is testing the ability of
a classifier to distinguish between treatments as a function of the image
used (i.e., taking into consideration only the labels that worker instances
attribute to a particular image, and varying the image).


`Analyzer` is used to analyze the difference between treatments on the basis
of an ontology of the words used.  So, for example, it can assess to what 
degree the workers in one treatment are using more specific words than the 
workers from another treatment.
'''


# Todo make a __main__ method that gets called when this is run as a python
# script from the command line, which reproductes all of the analysis


import util
import sys
import random
import naive_bayes
import ontology
import data_processing
import numpy as np


# Number of stardard deviations equivalent to the % condifdence for a 
# normal variate
CONFIDENCE_95 = 1.96
CONFIDENCE_99 = 2.975


class Analyzer(object):

	# Constants
	ONTOLOGY_FILE = 'ontology/testOntology.json'

	def __init__(self, dataset=None, ontology=None):
		if dataset is None:
			self.readDataset()
		else:
			self.dataSet = dataset

		if ontology is None:
			self.readOntology()
		else:
			self.ontology = ontology


	def compareNonCulturalSpecificity(
		self, treatment1, treatment2, numToCompare=50):

		# Mask food
		cultureToken = self.ontology.getSynonym('cultural')
		self.ontology.mask(cultureToken)

		# compare entries now that the mask is applied
		result = self.compareSpecificity(
			treatment1, treatment2, numToCompare)

		# remove mask
		self.ontology.clearMask()
		
		return result


	def compareNonFoodSpecificity(
		self, treatment1, treatment2, numToCompare=50):

		# Mask food
		foodToken = self.ontology.getSynonym('food')
		self.ontology.mask(foodToken)

		# compare entries now that the mask is applied
		result = self.compareSpecificity(
			treatment1, treatment2, numToCompare)

		# remove mask
		self.ontology.clearMask()

		return result


	def compareValenceSpecificity(
		self, valence, subjectTreatment, basisTreatment, numToCompare=50, 
		images=['test0']):

		assert(valence in ['overall', 'cultural', 'food'])

		if valence == 'overall':
			return self.compareSpecificity(
				subjectTreatment, basisTreatment, numToCompare, images)

		elif valence == 'cultural':
			return self.compareCulturalSpecificity(
				subjectTreatment, basisTreatment, numToCompare, images)

		elif valence == 'food':
			return self.compareFoodSpecificity(
				subjectTreatment, basisTreatment, numToCompare, images)

		else:
			raise ValueError("In Analyzer.compareValenceSpecificity: valence "\
				"must be one of 'overall', 'cultural', or 'food'. Found '"\
				+ valence + "'.")


	def compareFoodSpecificity(
		self, treatment1, treatment2, numToCompare=50, images=['image0']):

		# Mask all except food
		foodToken = self.ontology.getSynonym('food')
		for token in self.ontology.model['ROOT']:
			token = self.ontology.getSynonym(token)
			if token != foodToken:
				self.ontology.mask(token)

		# compare entries now that the mask is applied
		result = self.compareSpecificity(
			treatment1, treatment2, numToCompare, images)

		# remove mask
		self.ontology.clearMask()

		return result


	def compareCulturalSpecificity(
		self, treatment1, treatment2, numToCompare=50, images=['image0']):

		# Mask all except cultural
		cultureToken = self.ontology.getSynonym('cultural')
		for token in self.ontology.model['ROOT']:
			token = self.ontology.getSynonym(token)
			if token != cultureToken:
				self.ontology.mask(token)

		# compare entries now that the mask is applied
		result = self.compareSpecificity(
			treatment1, treatment2, numToCompare, images)

		# remove mask
		self.ontology.clearMask()

		return result


	def percentValence(self, treatments, images=None, 
		significanceLevel=CONFIDENCE_95):
		'''
		Determine the percentage of culture-related tokens found in entries
		of a given treatment.  If image is specified, restrict counts to
		a specific image
		Input: 
			treatmen (str): key for identifying a treatment, e.g.
				'treatment0'.

			image (str): key identifying an image, e.g. 'test0'.

		Output
			(float): percentage of terms associated to the treatment (and 
				image if specified) that are cultural according to the 
				ontology.
		'''

		# counters
		treatmentCounts = {'cultural':0, 'food':0, 'both':0, 'overall':0}
		percentages = {
			'cultural': [], 'excessCultural': [], 
			'food': [], 'both': []
		}

		for treatment in treatments:
			entries = self.dataSet.entries[treatment]
			for entry in entries:

				# Figure out which tokens we're interested in
				tokenKeys = []

				# if the image wasn't specified, just get all the keys that specify
				# words entered for any test image
				if images is None:
					tokenKeys = filter(
						lambda tokenKey: tokenKey[0].startswith('test'), 
						entry.keys())

				# But if the images are specified, just use that one
				else:
					tokenKeys = filter(
						lambda tokenKey: tokenKey[0] in images, entry.keys())

				if not len(tokenKeys):
					raise Exception('Bad key for image')


				entryCounts = {
					'cultural': 0, 'food': 0, 'both': 0, 
					'excessCultural':0, 'overall': 0
				}

				for tokenKey in tokenKeys:

					treatmentCounts['overall'] += 1
					entryCounts['overall'] += 1

					isCultural = self.isCultural(entry[tokenKey])
					isFood = self.isFood(entry[tokenKey])

					if isCultural:
						entryCounts['cultural'] += 1
						entryCounts['excessCultural'] += 1

					if isFood:
						entryCounts['food'] += 1
						entryCounts['excessCultural'] -= 1

					if isCultural and isFood:
						entryCounts['both'] += 1

						# we want these to count "strictly cultural", etc.
						entryCounts['cultural'] -= 1
						entryCounts['food'] -= 1


				percentages['cultural'].append(
					100 * entryCounts['cultural']/float(entryCounts['overall']))

				percentages['excessCultural'].append(
					100 * entryCounts['cultural']/float(entryCounts['overall']))

				percentages['food'].append(
					100 * entryCounts['food']/float(entryCounts['overall']))

				percentages['both'].append(
					100 * entryCounts['both']/float(entryCounts['overall']))


		meanPercentage = {
			'cultural': np.mean(percentages['cultural'])
			, 'excessCultural': np.mean(percentages['excessCultural'])
			, 'food': np.mean(percentages['food'])
			, 'both': np.mean(percentages['both'])
		}

		# To get the standard deviation of the mean, 
		# take the standard deviation of the samples 
		# and divide by square root of number of samples.
		# Then, multiply by the number of standard deviations needed to 
		# generate the desired significance level.
		stdPercentage = {
			'cultural': significanceLevel * (
				np.std(percentages['cultural']) / np.sqrt(len(entries)))

			, 'excessCultural': significanceLevel * (
				np.std(percentages['excessCultural']) / np.sqrt(len(entries)))

			, 'food': significanceLevel * (
				np.std(percentages['food']) / np.sqrt(len(entries)))

			, 'both': significanceLevel * (
				np.std(percentages['both']) / np.sqrt(len(entries)))
		}

		return {'mean': meanPercentage, 'stdev':stdPercentage}


	def isCultural(self, token):
		cultureToken = self.ontology.getSynonym('cultural')
		if cultureToken in self.ontology.getAncesters(token):
			return True

		return False


	def isFood(self, token):
		foodToken = self.ontology.getSynonym('food')
		if foodToken in self.ontology.getAncesters(token):
			return True

		return False


	def compareSpecificity(
		self, treatment1, treatment2, numToCompare=50, images=['test0']):
		'''
		Compare entries sampled randomly from two treatments, based on the
		relative locations of the words in the entries in the ontology tree.

		Note, an entry i from treatment 1 gets compared against all entries j
		from treatment 2, and this creates the result for that entry i.
		The results for all entries in treatment 1 (which involves all pairs
		being compared) are averaged and reported as the comparison results.
		'''

		# If the two treatments are the same, randomly partition them so we
		# don't get any overlaps.  Each entry is the output of one worker.
		if treatment1 == treatment2:
			entries1, entries2 = util.randomPartition(
				self.dataSet.entries[treatment1], numToCompare, numToCompare)

		# Otherwise, simply randomly sample from the two treatments
		else:
			entries1 = random.sample(
				self.dataSet.entries[treatment1], numToCompare)
			entries2 = random.sample(
				self.dataSet.entries[treatment2], numToCompare)

		# Declare some variables to keep count during the comparison
		firstMoreSpecificCounts = []
		firstLessSpecificCounts = []
		firstMoreMinusLess = []
		uncomparableCounts = []

		util.writeNow('         carrying out comparison ' + str(images))
		
		# Make all pairwise comparisons between the workers subsampled from 
		# in treatment 1 and those subsampled from treatment 2
		for i in range(len(entries1)):

			entry1 = entries1[i]

			util.writeNow('.')

			# Counters that characterize i's comparison to all j entries
			subFirstMoreSpecificCounts = []
			subFirstLessSpecificCounts = []
			subUncomparableCounts = []

			for j in range(len(entries2)):

				entry2 = entries2[j]

				# Now compale the two workers.  All pairs of words are 
				# tried. Words can only be compared if one is the ancestor of
				# the other
				lessSpec, moreSpec, uncomp = self.compareEntries(
					entry1, entry2, images)

				# Record results for this comparison
				subUncomparableCounts.append(uncomp)
				subFirstLessSpecificCounts.append(lessSpec)
				subFirstMoreSpecificCounts.append(moreSpec)
				
			# Compute and record results for all comparisons of ith worker
			numFirstMoreSpecific = np.mean(subFirstMoreSpecificCounts)
			numFirstLessSpecific = np.mean(subFirstLessSpecificCounts)
			numUncomparable = np.mean(subUncomparableCounts)

			firstMoreSpecificCounts.append(numFirstMoreSpecific)
			firstLessSpecificCounts.append(numFirstLessSpecific)
			firstMoreMinusLess.append(
				numFirstMoreSpecific - numFirstLessSpecific)
			uncomparableCounts.append(numUncomparable)

		print ''
		return {
			'avgFirstLessSpecific': np.mean(firstLessSpecificCounts),
			'stdFirstLessSpecific': (
				np.std(firstLessSpecificCounts)/np.sqrt(numToCompare)),

			'avgFirstMoreSpecific': np.mean(firstMoreSpecificCounts),
			'stdFirstMoreSpecific': (
				np.std(firstMoreSpecificCounts)/np.sqrt(numToCompare)),

			'avgMoreMinusLess': np.mean(firstMoreMinusLess),
			'stdMoreMinusLess': (
				np.std(firstMoreMinusLess)/np.sqrt(numToCompare)),

			'avgUncomparable': np.mean(uncomparableCounts),
			'stdUncomparable': (
				np.std(uncomparableCounts)/np.sqrt(numToCompare)),
		}


	def compareEntries(self, entry1, entry2, images=['test0']):
		'''
		Input: two entries from the dataset, which represent the words entered
		by two different workers.
		Get the words that each worker associated to the first image.  Then
		check all possible pairings of a word from the first worker's set to
		a word from the second workers set, and count how many times the
		word from the first worker's set is higher / lower in the ontology
		'''

		num_w1_gt_w2 = 0	# number of times word1 is lower than word2
		num_w1_lt_w2 = 0	# number of times word1 is lower than word2
		num_comparisons = 0

		for image in images:

			# get each entry's words for the picture
			words1 = []
			words2 = []
			for wordPosition in range(self.dataSet.NUM_WORDS_PER_IMAGE):
				words1.append(entry1[(image, wordPosition)])
				words2.append(entry2[(image, wordPosition)])

			# For every word pairing for words in this image, check if 
			# one is higher than the other in the ontology 
			for w1 in words1:
				for w2 in words2:
					num_comparisons += 1
					relation = self.ontology.compare(w1,w2)

					if relation > 0:
						num_w1_gt_w2 += 1
					elif relation < 0:
						num_w1_lt_w2 += 1

		return (num_w1_gt_w2, num_w1_lt_w2,
			num_comparisons - num_w1_gt_w2 - num_w1_lt_w2)


	def readOntology(self):
		self.ontology = ontology.Ontology()
		self.ontology.readOntology(self.ONTOLOGY_FILE)
		

	def readDataset(self):
		'''
		Factory method that builds a CleanDataset from the original Amazon
		Mechanical Turk CSV files
		'''

		# Create a new priming-image-label-experiment dataset
		self.dataSet = data_processing.CleanDataset()

		# Read from the raw amt csv files.  
		# Note: order matters!  The older files have duplicates workers that
		# get ignored.  Make sure to read the newer file files earlier
		self.dataSet.read_csv('amt_csv/amt1_cut.csv', True)
		self.dataSet.read_csv('amt_csv/amt2_cut.csv', True)
		self.dataSet.read_csv('amt_csv/amt3_cut.csv', True)

		# The dataset needs to do some internal calts to refresh its state 
		self.dataSet.aggregateCounts()
		self.dataSet.calc_ktop(5)




class NBCAnalyzer(object):
	'''
	This class is used to assess the ability of a naive bayes classifier in
	being able to distinguish between different treatments.  So for example
	`testNBC()` takes in a set of treatments, and builds a classifier that
	by subsampling labelled points from this projects dataset, and then 
	tests the classifier against unused data from each treatment.  It 
	builds a `ClassifierPerformanceResult` to summarize the performance.

	Another method, `longitudinal()` assesses the classifier performance at 
	distinguishing 
	between treatments, but as a function of the image used (that is using
	only a subset of the features for every data point -- the labels attributed
	to a specific image).

	See the method descriptions for more details!
	'''

	def __init__(self):
		self.readDataset()


	def readDataset(self):
		self.dataSet = data_processing.readDataset()


	def testNBC(self, pTrainingSize=80, pTestSize=20, 
		treatments=['treatment1', 'treatment2'], 
		images=['test0'], pDoConsiderPosition=False, show=False):
		'''
		Builds and trains a naive bayes classifier that classifies instances
		into one of the categories keyed by the input treatments.  Tests
		the resulting classifier on a subset of test instances (which were 
		excluded from training), and returns performance metrics.

		Inputs:
			- pTrainingSize (int): The size of training set that will be
				randomly sampled from all instances found in the dataset for 
				in each treatment served.  The instances left out during random
				sampling form a pool from which test instances are drawn.

			- pTestSize (int): size of the test set, randomly sampled from
				all instances in the dataset less the training set

			- treatments (str): A key used to identify a set of instances
				in the dataset that belong to a particular class.

		Output:
			- (classifer_result): See the class definition.  An object which
				summarizes the performance of the classifier in a test.  This
				object is understood by certain data visualization methods.
		'''

		# Wrap the dataset to make it understandable by the naive bayes 
		# classifier
		naiveBayesDataset = naive_bayes.NBDataset(
			self.dataSet, treatments, images, pDoConsiderPosition,
			pTrainingSize, pTestSize)

		# Make and train a naive bayes classifier
		classifier = naive_bayes.NaiveBayesClassifier()
		classifier.buildFromTrainingSet(naiveBayesDataset)

		# Get test instances from the dataset
		testSet = naiveBayesDataset.getTestSet()

		# Initialize a data structure to store classifier performance results
		r = ClassifierPerformanceResult(treatments)

		resultMsg = ''

		# Test the classifier on the test instances to acertain its competence
		for treatment, instances in testSet.items():

			# Test classification of instances from this treatment
			for instance in instances:

				r.results[treatment]['numTested'] += 1

				verdict = classifier.classify(instance)
				actual = instance.reveal()

				r.results[verdict]['numIdentified'] += 1

				isMatch = '-'
				if verdict == actual:
					r.results[treatment]['numCorrect'] += 1
					isMatch = '@'

				resultMsg += '%s %s %s' % (verdict, actual, isMatch)
		
		if show:
			print resultMsg

		return r


	def crossComparison(self, numReplicates, comparisons):
		'''
		Run the classifier to do pairwise comparisons of various sorts
		'''
		util.writeNow('\nPerforming cross-comparison') 

		results = {}

		for comparison in comparisons:

			util.writeNow('.')
			comparisonF1Scores = []
			comparisonAccuracies = []

			firstTreatment, secondTreatment = comparison

			for i in range(numReplicates):
				allImages = ['test%d' % i for i in range(5)]
				result = self.testNBC(80, 20, comparison, allImages, True)
				try:
					comparisonF1Scores.append(result.getF1(firstTreatment))
					comparisonAccuracies.append(result.getOverallAccuracy())
				except KeyError:
					return result

			results[comparison] = {
				'f1': {
					'avg': np.mean(comparisonF1Scores),
					'std': np.std(comparisonF1Scores)
				},
				'accuracy': {
					'avg': np.mean(comparisonAccuracies),
					'std': np.std(comparisonAccuracies)
				}
			}

		return results


	def longitudinal(self, numReplicates, treatments):
		'''
		Run the classifier to do pairwise classification between the cultural
		images and ingredients images priming, using only one picture as the
		source of features, and repeating for each picture.  
		For each classification, do it with and without keeping track of the 
		position of words, plotting these options next to one another.
		'''

		util.writeNow(
			'\nAnylisis of classification competance as a function of image')

		# Run it for the various classification combinations desired
		f1ScoreStdevs = {'withPosition':[], 'withoutPosition':[]}
		f1Scores = {'withPosition':[], 'withoutPosition':[]}

		results = {
			'f1': {
				'avg': [],
				'std': []
			},
			'accuracy': {
				'avg': [],
				'std': []
			},
		}

		#  We'll run the classifier on various images.  First we run it on 
		# the set of all images, and then on each image on its own
		allImageSet = ['test%d' % i for i in range(5)]
		eachImageSet = [['test%d' % i] for i in range(5)]
		imageSets = [allImageSet] + eachImageSet

		firstTreatment, secondTreatment = treatments

		for imageSet in imageSets:	
			util.writeNow('.')

			# Accumulate results accross replicates for this imageSet
			thisImageSetF1Scores = []
			thisImageSetAccuracies = []

			# Test the classsifier for this imageSet. Do many replicates.
			for j in range(numReplicates):

				# Run and test classification
				result = self.testNBC(
					80, 20, treatments, imageSet, True)

				# Extract the results we're interested in
				thisImageSetF1Scores.append(result.getF1(firstTreatment))
				thisImageSetAccuracies.append(result.getOverallAccuracy())
		
			# record average and standard deviation for classifier F1 score
			results['f1']['avg'].append(
				np.mean(thisImageSetF1Scores))
			results['f1']['std'].append(
				np.std(thisImageSetF1Scores))

			# record average and standard deviation for classifier accuracy
			results['accuracy']['avg'].append(
				np.mean(thisImageSetAccuracies))
			results['accuracy']['std'].append(
				np.std(thisImageSetAccuracies))

		return results


class ClassifierPerformanceResult(object):
	'''
	This class is used to represent the performance of the classifier
	when it has been trained on a subset of the data, and then tested on a
	data not used in training.

	The reason for using a class to represent it is that there are many ways
	to characterise the performance, such as accuracy, precision, F1 score,
	but they are all calculable from the raw hits vs misses outcomes when the
	classifier is tested.
	'''

	def __init__(self, treatments):
		self.treatments = treatments
		self.results = {}
		
		for treatment in treatments:
			self.results[treatment] = {
				'numCorrect':0
				, 'numIdentified':0
				, 'numTested':0
			}


	def getPrecision(self, treatment):
		numerator = self.results[treatment]['numCorrect']
		denominator = float(self.results[treatment]['numIdentified'])
		return numerator / denominator


	def getRecall(self, treatment):
		numerator = self.results[treatment]['numCorrect']
		denominator = float(self.results[treatment]['numTested'])
		return numerator / denominator


	def getOverallAccuracy(self):
		numCorrectOverall = 0
		numTestedOverall = 0
		for treatment, result in self.results.items():
			numCorrectOverall += result['numCorrect']
			numTestedOverall += result['numTested']

		return numCorrectOverall / float(numTestedOverall)

	def getF1(self, treatment):
		recall = self.getRecall(treatment)
		precision = self.getPrecision(treatment)
		return 2*recall*precision / (recall + precision)



