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

import numpy as np
import random
import copy
from collections import Counter, defaultdict

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

	


class NaiveBayesException(Exception):
	pass

class NaiveBayesCrossValidationException(Exception):
	pass

class NaiveBayesCrossValidationTester(object):
	'''
	Given a data set, allows one to perform cross validation on the 
	NaiveBayesTextClassifier.

	The dataset must be a dictionary of examples, where the keys are 
	class names, and the values are lists of examples.  Each example should 
	be a vector (tuple) of features (words), but the first entry should be 
	the the name of the class to which the example belongs (true, that's a 
	bit redundant.

	For the moment, the dataset should have an equal number of examples per
	class.  It's an error to provide an imbalanced dataset for now.

	E.g.:
	dataset [=] {
		'class1': [
			('class1', 'feature1', 'feature2'),
			('class1', 'feature1', 'feature3')
		],
		'class2: [
			('class2', 'feature2', 'feature3'),
			('class2', 'feature3', 'feature4')
	}
	'''

	def __init__(self, dataset):

		# copy the data, initialize state
		self.dataset = copy.copy(dataset)
		self.num_classes = len(self.dataset)
		self.scores = None
		self.are_results_available = False

		# Randomize the examples' ordering.
		# Also check the number of examples per class
		self.num_examples_per_class = None
		for class_name, examples in self.dataset.items():

			random.shuffle(examples)

			# find out the number of examples per class, and ensure its uniform
			if self.num_examples_per_class is None:
				self.num_examples_per_class = len(examples)
			elif len(examples) != self.num_examples_per_class:
				raise NaiveBayesCrossValidationException('The classes must '
					'all have the same number of examples.')

		# as a way to speed computation, we the NBClassifier supports 
		# removing examples from the training set, so that when testing for
		# a new fold, the test set can be removed rather than retraining on
		# the whole training set.  We therefore begin by training on the whole 
		# set.  This has been tested to ensure it does not cause "pollution".
		self.classifier = NewNaiveBayesTextClassifier()
		for class_name in self.dataset:
			self.classifier.train(self.dataset[class_name])


	def extract_test_set(self, fold, test_set_size, is_last=True):
		'''
		this was made as separate function to provide a testable interface.
		This removes a subset of examples from each class to be used as a 
		test set, while ensuring that the examples are not used for training.
		'''

		test_set = {}
		for class_name in self.dataset:

			# select the examples to be used as the test set for this class
			start = fold * test_set_size
			end = start + test_set_size
			if is_last:
				end = None

			test_set[class_name] = self.dataset[class_name][start:end]

			# ensure that these examples are excluded from the training set
			for example in test_set[class_name]:
				self.classifier.remove_example(example)

		return test_set


	def put_test_set_back_in_training_set(self, test_set):
		for class_name in test_set:
			for example in test_set[class_name]:
				self.classifier.add_example(example)


	def cross_validate(self, k):
		'''
		divide the data set set into k equal folds (if k doesn't divide the
		number of examples in each class, then the folds won't all be equal).
		for each 'fold' of cross validation, train a NaiveBayesTextClassifier
		on all the data outside the fold, then test it on the data inside the
		fold, and repeat for all folds.  Keep a running tally of the number
		of classifications correct, for each class.
		'''

		k = int(k)

		# Make sure k is within the allowed range
		if k < 2:
			raise NaiveBayesCrossValidationException('You must have more than '
				'one fold for the cross validation test.')
		if k > self.num_examples_per_class:
			raise NaiveBayesCrossValidationException('Their cannot be more '
				'folds than there are examples in each class!')

		test_set_size = self.num_examples_per_class / k

		# TODO: test to ensure no pollution

		# to save computation time, the NaiveBayesClassifier supports 
		# "removing" examples, which limits the number of conditional 
		# probabilities that need to be re-calculated in each fold. To 
		# take advantage of this, we first train on the full dataset.
		# Testing shows that there is no pollution from doing this.
		self.scores = Counter()

		for fold in range(k):


			is_last = bool(fold == k - 1)
			test_set = self.extract_test_set(fold, test_set_size, is_last)

			for class_name in test_set:
				for example in test_set[class_name]:

					example_class = example[0]
					example_features = example[1:]
					prediction = self.classifier.classify(example_features)

					if prediction == example_class:
						self.scores[example_class] += 1

			self.put_test_set_back_in_training_set(test_set)

		# return the overall accuracy.  
		# Other performance metrics are available through method calls.
		return sum(self.scores.values()) / float(
			self.num_examples_per_class * self.num_classes)



class NewNaiveBayesTextClassifier(object):
	'''
	Training consists of providing the classifier a set of examples having
	the structure:

		('class-name', 'word0', 'word1', 'word2', 'word3', 'word4')

	These can be added in bulk using train(), or singly using add_example().

	Examples can also be quickly removed: because they are stored as tuples
	of strings, they can be hashed, so this can be done quickly.
	'''

	IMPOSSIBLE = None

	def __init__(self):
		self.examples = Counter()

		# global counts is used to keep track of the set of all features
		# if a given feature count goes down to zero, then we don't consider
		# it as existing anymore
		self.global_feature_counts = Counter()
		self.feature_counts = defaultdict(lambda: Counter())
		self.class_counts = Counter()
		self.num_examples = 0

		self.is_num_features_fresh = False
		self._num_features = 0

		self.is_num_classes_fresh = False
		self._num_classes = 0


	def train(self, examples):
		for example in examples:
			self.add_example(example)

	
	def add_example(self, example):
		class_name = example[0]
		features = example[1:]

		self.examples[example] += 1
		self.num_examples += 1
		self.class_counts[class_name] += 1
		self.global_feature_counts.update(features)
		self.feature_counts[class_name].update(features)

		self.is_num_features_fresh = False
		self.is_num_classes_fresh = False


	def remove_example(self, example):

		# raise an exception if attempting to remove an example that the 
		# classifier doesn't actually have
		if example not in self.examples:
			raise NaiveBayesException(
				'The instance of NaiveBayesTextClassifier did not have that'
				' example, so it could not be removed: %s' % str(example)
			)

		features = example[1:]
		class_name = example[0]

		# remove the example and its contribution to feature counts
		self.examples[example] -= 1
		if self.examples[example] == 0:
			del self.examples[example]

		self.num_examples -= 1
		self.class_counts[class_name] -= 1
		self.global_feature_counts.subtract(features)
		self.feature_counts[class_name].subtract(features)

		self.is_num_features_fresh = False
		self.is_num_classes_fresh = False


	def get_num_features(self):
		if self.is_num_features_fresh:
			return self._num_features

		num_features = 0
		for key, val in self.global_feature_counts.iteritems():
			if val>0:
				num_features += 1 

		self._num_features = num_features
		self.is_num_features_fresh = True

		return self._num_features


	def get_num_classes(self):
		if self.is_num_classes_fresh:
			return self._num_classes

		self.refresh_class_counts()
		return self._num_classes


	def refresh_class_counts(self):
		self._num_classes = 0
		self._class_names = set()
		for key, val in self.class_counts.items():
			if val>0:
				self._num_classes += 1
				self._class_names.add(key)

		self.is_num_classes_fresh = True


	def get_class_names(self):
		if self.is_num_classes_fresh:
			return self._class_names

		self.refresh_class_counts()
		return self._class_names



	# TODO: test
	def get_cond_prob(self, feature, class_name, use_add_one_smoothing=True):
		num_features = self.get_num_features()
		num_classes = self.get_num_classes()

		counts_for_feature_in_class = self.feature_counts[class_name][feature]
		num_examples_in_class = self.class_counts[class_name]

		# The technique "add-one-smoothing" helps calculate a "reasonable" 
		# likelihood for features that were never observed for a given class
		if use_add_one_smoothing:
			counts_for_feature_in_class += 1
			num_examples_in_class += num_features


		return counts_for_feature_in_class / float(num_examples_in_class)

	def get_prior(self, class_name):
		num_examples_in_class = self.class_counts[class_name]
		return num_examples_in_class / float(self.num_examples)


	# TODO: test
	# TODO: encorporate add_1_smoothing
	# TODO: encorporate the class prior
	def classify(self, example_features):
		'''
		Takes in an example_feature vector (which is missing the first 
		component, the class_name), and outputs the likelihood maximizing
		class, based on the assumption that features are independant of one
		another.
		'''
		# for each class, calculate a score equal to the likelihood that 
		# that class would produce this feature vector
		class_scores = defaultdict(lambda: 0)
		for class_name in self.get_class_names():
			for feature in example_features:

				cond_prob = self.get_cond_prob(feature, class_name)

				# we have to handle the fact that (if add 1 smoothing is not
				# used, then a class cand be found to be 'impossible'
				if (
					cond_prob == 0 or 
					class_scores[class_name] is self.IMPOSSIBLE
				):
					class_scores[class_name] = self.IMPOSSIBLE

				# summing log likelihoods is like taking the straight product.
				else:
					class_scores[class_name] += np.log(cond_prob)

		# "multiply" each class's score by the class' prior probability 
		for class_name in class_scores:
			class_scores[class_name] += np.log(self.get_prior(class_name))

		# Which class was most likely? Remove impossible classes, then sort.
		feasible_class_scores = filter(
			lambda c: c[1] is not self.IMPOSSIBLE,
			class_scores.items()
		)
		feasible_class_scores.sort(None, lambda c: c[1], True)
		predicted_class, top_score = feasible_class_scores[0]

		return predicted_class




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




		











