import util
import sys
import random
import naive_bayes
import ontology
import data_processing
import numpy as np


class ClassifierPerformanceResult(object):

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



class Analyzer(object):

	# Constants
	ONTOLOGY_FILE = 'ontology/test0.ontology.json'

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
		self.ontology.mask('cultural')

		# compare entries now that the mask is applied
		result = self.compareSpecificity(
			treatment1, treatment2, numToCompare)

		# remove mask
		self.ontology.clearMask()
		
		return result


	def compareNonFoodSpecificity(
		self, treatment1, treatment2, numToCompare=50):

		# Mask food
		self.ontology.mask('food')

		# compare entries now that the mask is applied
		result = self.compareSpecificity(
			treatment1, treatment2, numToCompare)

		# remove mask
		self.ontology.clearMask()

		return result


	def compareFoodSpecificity(
		self, treatment1, treatment2, numToCompare=50):

		# Mask all except food
		for token in self.ontology.model['ROOT']:
			if token != 'food':
				self.ontology.mask(token)

		# compare entries now that the mask is applied
		result = self.compareSpecificity(
			treatment1, treatment2, numToCompare)

		# remove mask
		self.ontology.clearMask()

		return result


	def compareCulturalSpecificity(
		self, treatment1, treatment2, numToCompare=50):

		# Mask all except cultural
		for token in self.ontology.model['ROOT']:
			if token != 'cultural':
				self.ontology.mask(token)

		# compare entries now that the mask is applied
		result = self.compareSpecificity(
			treatment1, treatment2, numToCompare)

		# remove mask
		self.ontology.clearMask()

		return result


	def percentValence(self, valence, treatment, image=None):
		'''
		Determine the percentage of culture-related tokens found in entries
		of a given treatment.  If image is specified, restrict counts to
		a specific image
		Input: 
			valence (str): root token in the ontology, e.g. 'food', or 
				'cultural'.

			treatmen (str): key for identifying a treatment, e.g.
				'treatment0'.

			image (str): key identifying an image, e.g. 'test0'.

		Output
			(float): percentage of terms associated to the treatment (and 
				image if specified) that are cultural according to the 
				ontology.
		'''

		# counters
		numValenceTokens = 0
		numTokens = 0

		valencePercentages = []

		entries = self.dataSet.entries[treatment]
		for entry in entries:

			# Figure out which images we're interested in
			interestingImages = []

			# if the image wasn't specified, just get all the keys that specify
			# words entered for any test image
			if image is None:
				interestingImages = filter(
					lambda imageKey: imageKey[0].startswith('test'), 
					entry.keys())

			# But if the image was specified, just use that one
			else:
				interestingImages = filter(
					lambda imageKey: imageKey[0] == image, entry.keys())

			if not len(interestingImages):
				raise Exception('Bad key for image')


			thisNumValenceTokens = 0
			thisNumTokens = 0
			for imageKey in interestingImages:
				numTokens += 1
				thisNumTokens += 1

				if valence == 'cultural' and self.isCultural(entry[imageKey]):
					numValenceTokens += 1
					thisNumValenceTokens += 1

				elif valence == 'food' and self.isFood(entry[imageKey]):
					numValenceTokens += 1
					thisNumValenceTokens += 1

			valencePercentages.append(
				100 * thisNumValenceTokens/float(thisNumTokens))


		meanPercentage = np.mean(valencePercentages)

		# To get the standard deviation of the mean, take the standard deviation
		# of the samples and divide by square root of number of samples
		stdvPercentage = (
			np.std(valencePercentages) / np.sqrt(len(valencePercentages)))

		return {'mean': meanPercentage, 'stdev':stdvPercentage}


	def isCultural(self, token):
		if 'cultural' in self.ontology.getAncesters(token):
			return True

		return False


	def isFood(self, token):
		if 'food' in self.ontology.getAncesters(token):
			return True

		return False


	def compareSpecificity(
		self, treatment1, treatment2, numToCompare=50):
		'''
		Compare entries sampled randomly from two treatments, based on the
		relative locations of the words in the entries in the ontology tree
		'''

		# Randomly sample desired number of entries the indicated treatments
		# TODO deal with the case where treatment1 is the same as treatment2
		if treatment1 == treatment2:
			entries1, entries2 = util.randomPartition(
				self.dataSet.entries[treatment1], numToCompare, numToCompare)

		else:
			entries1 = random.sample(
				self.dataSet.entries[treatment1], numToCompare)
			entries2 = random.sample(
				self.dataSet.entries[treatment2], numToCompare)

		firstMoreSpecificCounts = []
		firstLessSpecificCounts = []
		firstMoreMinusLess = []
		uncomparableCounts = []

		for i in range(len(entries1)):

			entry1 = entries1[i]

			# Counters that characterize i's comparison to all j entries
			numFirstMoreSpecific = 0
			numFirstLessSpecific = 0
			numUncomparable = 0

			for j in range(len(entries2)):

				entry2 = entries2[j]

				lessSpec, moreSpec, uncomp = self.compareEntries(entry1, entry2)

				numUncomparable += uncomp
				numFirstLessSpecific += lessSpec
				numFirstMoreSpecific += moreSpec
				
			firstMoreSpecificCounts.append(numFirstMoreSpecific)
			firstLessSpecificCounts.append(numFirstLessSpecific)
			firstMoreMinusLess.append(numFirstMoreSpecific - numFirstLessSpecific)
			uncomparableCounts.append(numUncomparable)

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


	def compareEntries(self, entry1, entry2):
		'''
		Input: two entries from the dataset, which represent the words entered
		by two different workers.
		Get the words that each worker associated to the first image.  Then
		check all possible pairings of a word from the first worker's set to
		a word from the second workers set, and count how many times the
		word from the first worker's set is higher / lower in the ontology
		'''

		# get each entry's words for the first picture
		words1 = []
		words2 = []
		for wordPosition in range(self.dataSet.NUM_WORDS_PER_IMAGE):
			words1.append(entry1[('test0', wordPosition)])
			words2.append(entry2[('test0', wordPosition)])

		num_w1_gt_w2 = 0	# number of times word1 is lower than word2
		num_w1_lt_w2 = 0	# number of times word1 is lower than word2
		num_comparisons = 0

		# For every word pairing between these sets of words, check if
		# one is higher than the other in the ontology of test0 words
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

		f1Scores = {}
		f1ScoreDevs = {}

		for comparison in comparisons:

			util.writeNow('.')
			thisComparisonF1Scores = []
			firstTreatment, secondTreatment = comparison

			for i in range(numReplicates):
				allImages = ['test%d' % i for i in range(5)]
				result = self.testNBC(80, 20, comparison, allImages, True)
				try:
					thisComparisonF1Scores.append(result.getF1(firstTreatment))
				except KeyError:
					return result

			f1Scores[comparison] = np.mean(thisComparisonF1Scores)
			f1ScoreDevs[comparison] = np.std(thisComparisonF1Scores)

		return f1Scores, f1ScoreDevs
			
			
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

		#  We'll run the classifier on various images.  First we run it on 
		# the set of all images, and then on each image on its own
		allImageSet = ['test%d' % i for i in range(5)]
		eachImageSet = [['test%d' % i] for i in range(5)]
		imageSets = [allImageSet] + eachImageSet

		firstTreatment, secondTreatment = treatments

		for imageSet in imageSets:	
			util.writeNow('.')

			thisTreatmentf1Scores = {'withPosition':[], 'withoutPosition':[]}
			for j in range(numReplicates):

				# Run classification with position-based features
				result = self.testNBC(
					80, 20, treatments, imageSet, True)

				thisTreatmentf1Scores['withPosition'].append(
					result.getF1(firstTreatment))

				# Run classification with positionless features
				result = self.testNBC(
					80, 20, treatments, imageSet, False)
				thisTreatmentf1Scores['withoutPosition'].append(
					result.getF1(firstTreatment))

		
			f1Scores['withPosition'].append(
				np.mean(thisTreatmentf1Scores['withPosition']))
			f1ScoreStdevs['withPosition'].append(
				np.std(thisTreatmentf1Scores['withPosition']))

			f1Scores['withoutPosition'].append(
				np.mean(thisTreatmentf1Scores['withoutPosition']))
			f1ScoreStdevs['withoutPosition'].append(
				np.std(thisTreatmentf1Scores['withoutPosition']))

		return f1Scores, f1ScoreStdevs

