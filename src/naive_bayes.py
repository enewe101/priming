'''
This module provides classes that facilitate the building of a naive bayes 
classifier based on the dataset for this project.

As a quick reminder, the dataset consists of words that workers have attributed
to images in an image-labeling task.  So, for a given worker, their use (or
non-use) of a given word as a label for a given image, constitutes a feature.

Now, we can get more specific: workers had to label each image with 5 words.
They were given 5 text inputs.  We can regard the use of a given word for
a given image *in a given text input* as a feature.  That is, using `wine` as
the first label in image 2 is a distinct feature from using `wine` as the
second label in image 2.  (Intuitively, a worker might provide a label sooner
if she was primed to produce that label.)

This module has a few classes:

	`NBDataset` is responsible for wrapping the dataset, (as represented by
		`CleanDataset` in the data_processing module), and answering questions
		about the dataset that are useful in training a naive bayes classifier.

	`NaiveBayesClassifier` is responsible for acting as an actual classifier.
		it takes a `NBDataset` and internalizes a model of a features's 
		probabilities conditionned on the fact that it belongs to an instance
		of a particular class.  This is normal naive bayes stuff.  Once trained
		it also produces classifications in a testing mode, upon input of an
		unlabelled instance taken from one of the classes it was trained to 
		distinguish.

	`TestInstance` wraps an entry taken from the `CleanDataset`, and makes it
		usable as a test-case for the `NaiveBayesClassifier`.  That's because
		the `NaiveBayesClassifier` assumes that test instances know what 
		features are, and what features they have, in straightforward 
		"queriable" way.

These classes get used by the NBCAnalizer, which is responsible for looking at
how distinguishable diffirent treatments are (in various ways) when using a 
naive bayes classifier to do the distinguishing.
'''


import random


class NBDataset(object):
	'''
	Wraps an instance of the CleanDataset class, making an object that
	is suitable for being passed to the constructor of the NaiveBayesModel
	class.  This class satisfies the interface that the NaiveBayesModel expects
	to be able to query in order to build up a model.
	'''
	dataSet = None
	treatments = []
	doConsiderPosition = False
	instances = {}

	def __init__(self, 
		pDataSet, pTreatments=['treatment1','treatment2'], 
		pImages=['test0'], pDoConsiderPosition=False, testSetSize=25):
		'''
		Specify the treatment ids to be included when building the classifier.
		You may not always want to try to build a classifier that can 
		categorize instances from all treatments.  For example, for the image 
		priming data I am first trying to build a classifier for the three 
		first treatments only, since the last three might be quite similar, and
		are of less interest in this respect.
		'''

		self.treatments = pTreatments
		self.dataSet = pDataSet
		self.doConsiderPosition = pDoConsiderPosition
		self.images = pImages
		
		self.dataSet.subsample(testSetSize)


	def getCategories(self):
		return list(self.treatments)


	def rotateSubsample(self):
		self.dataSet.rotateSubsample()


	def getFeatures(self):
		features = set()
		for treatment, image, position in self.dataSet.counts:

			# The labels given to priming images are of no interest
			if image is not None and image.startswith('prime'):
				continue

			# We aren't interested in features that are aggregated accross 
			# many treatments or many images
			if treatment is None or image is None:
				continue

			# If we want to consider the labels given *in specific positions*
			# then the position must be specified
			if self.doConsiderPosition and position is None:
				continue

			# Otherwise position must not be specified
			elif not self.doConsiderPosition and position is not None:
				continue

			# We only interested in features belonging to the treatment and
			# images that were specified in the constructor
			if (image not in self.images 
				or treatment not in self.treatments):
				continue

			# At this point, we're looking at a relavent training instance.
			# We will now extract all the features from it.
			for word in self.dataSet.counts[
				(treatment, image, position)].items():

				features.add((image, position, word[0]))

		return features


	def getProbability(self, pCategory, pFeature):
		treatment = pCategory
		image, position, word = pFeature

		frequency = self.dataSet.getWordFrequency(
			word, treatment, image, position)

		return frequency / float(len(self.dataSet.entries[pCategory]))

	
	def getTestSet(self):
		'''
		Returns a set of instances from the test set, i.e. which were not
		included in the training set, but which are from the same treatments
		as was used to build the training set.

		The test instances are entries from the database which have been 
		wrapped by an TestInstance class, and so can be given to a naive
		bayes classifier as a parameter to its classify() method
		'''
		testSet = {}

		# Get test instances from the underlying dataset
		testInstances = self.dataSet.getTestInstances()

		# return a list of wrapped test instances
		for treatment, instances in testInstances.items():
			if treatment in self.treatments:

				testSet[treatment] = []

				for instance in instances:
					testSet[treatment].append(
						TestInstance(instance, self.images, 
						self.doConsiderPosition))

		return testSet



class NaiveBayesClassifier(object):
	'''
	This represents a naive Bayes classifier.  To function it needs a model
	which can either be provided by a training dataset 
	or read from file (not implemented)
	'''

	def __init__(self):
		self.categories = []
		self.features = set()
		self.condProbs = {}

	def buildFromTrainingSet(self, pTrainingSet):
		self.features = pTrainingSet.getFeatures()
		self.categories = pTrainingSet.getCategories()
		condProbs = {}

		for category in self.categories:
			for feature in self.features:
				self.condProbs[(category, feature)] = (
					pTrainingSet.getProbability(category, feature))


	def classify(self, instance):

		scores = {}
		for category in self.categories:
			scores[category] = 0

		for feature in instance.getFeatures():
			for category in self.categories:

				# If the feature arises for the given category, use the
				# add the conditional probability of its occurrence given the
				# category to the score for that category.  Otherwise add 0
				add_to_score = 0
				try:
					add_to_score = self.condProbs[(category, feature)]
				except KeyError:
					pass

				scores[category] += add_to_score

		verdict = sorted(scores.items(), None, lambda x: x[1], True)[0][0]

		return verdict



class TestInstance(object):
	'''
	Acts as a test instance for the NaiveBayesClassifier.  Provides a function
	getFeatures, which provides an iterable of features.  Can be built from
	a worker instance, of the form stored in CleanDataset or NBDataset
	'''

	def __init__(self, underlyingInstance, images=['test0'], 
		pDoConsiderPosition=False):

		self.underlyingInstance = underlyingInstance
		self.doConsiderPosition = pDoConsiderPosition
		self.images = images


	def getFeatures(self):

		features = []

		for key, value in self.underlyingInstance.items():

			# Only take the properties of the instance that are tuples,
			# because only these represent word-counts
			if not isinstance(key, tuple):
				continue

			image, position = key
			word = value

			if image not in self.images:
				continue

			if self.doConsiderPosition:
				features.append((image, position, word))

			else:
				features.append((image, None, word))

		return features


	def reveal(self):
		return self.underlyingInstance['treatment']




		











