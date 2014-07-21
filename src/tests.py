import util
import unittest
import analysis
import naive_bayes as nb
import data_processing as dp
import ontology
import copy

class OntologyTestCase(unittest.TestCase):
	def setUp(self):
		self.ont = ontology.Ontology()
		self.ont.readOntology('ontology/ontology0.json')


	def test_comparison(self):
		# First more general than second returns 1
		self.assertEqual(self.ont.compare('ethnic', 'indian'), 1)

		# First mere specific than second returns 2
		self.assertEqual(self.ont.compare('bread', 'food'), -1)

		# Non-comparable returns 0
		self.assertEqual(self.ont.compare('statue', 'saucy'), 0)


	def test_maskedComparison(self):

		# mask everything in the root of the test ontology except food
		for token in ['thing', 'adj', 'activity', 'cultural']:
			self.ont.mask(token)

		# Now make a food-based comparison
		self.assertEqual(self.ont.compare('food', 'bread'), 1)

		# Now make a cultural comparison.  This yielts 0 because cultural
		# is masked
		self.assertEqual(self.ont.compare('ethnic', 'indian'), 0)

		# Now make a cultural-food comparison.  
		# Default behavior is to do strict comparison, meaning that naan gets
		# masked since it descends from both 'food' and 'cultural' 
		self.assertEqual(self.ont.compare('bread', 'naan'), 0)

		# Now remove the mask for culture.  Naan compares again:
		self.ont.unmask('cultural')
		self.assertEqual(self.ont.compare('bread', 'naan'), 1)
		# but statue still doesn't
		self.assertEqual(self.ont.compare('thing', 'statue'), 0)

		# Remove all masks.  Now statue compares.
		self.ont.clearMask()
		self.assertEqual(self.ont.compare('thing', 'statue'), 1)
		self.assertEqual(self.ont.compare('adj', 'great'), 1)
		
		# Now test dropping tokens.  Compare will ignore tokens
		# whose top-level parents are all dropped.  But, if a token has a 
		# non-dropped top level parent, it would be included so long as at
		# least one top-level-parent isn't dropped, and so long as none of 
		# its top level parents are masked
		self.ont.drop('adj')

		# naan will compare, but something strictly cultural won't
		# borsht
			# food and adj  -- passes
		self.assertEqual(self.ont.compare('food', 'orange'), 1)
			# food and culture and adj -- passes
		self.assertEqual(self.ont.compare('meal', 'indian buffet'), 1)
			# culture -- passes
		self.assertEqual(self.ont.compare('religious', 'buddist'), 1)
			# adj -- fails
		self.assertEqual(self.ont.compare('adj', 'great'), 0)

		# Now we combine masks and drops
		self.ont.mask('culture')

		self.assertEqual(self.ont.compare('food', 'orange'), 1)
		self.assertEqual(self.ont.compare('meal', 'indian buffet'), 0)
		self.assertEqual(self.ont.compare('religious', 'buddist'), 0)
		self.assertEqual(self.ont.compare('adj', 'great'), 0)

		# finally, try removing the drop on adj:
		self.ont.undrop('adj')
		self.assertEqual(self.ont.compare('adj', 'great'), 1)

		self.ont.clearDrop()
		self.ont.clearMask()


	def test_ontologyBuilding(self):

		# start with a fresh ontology
		self.ont = ontology.Ontology()

		# Read a set of words that will be made into an ontology
		# Check that we have loaded the expected words
		self.ont.readWords('test/words.txt')
		self.assertItemsEqual(
			self.ont.getWords(1),
			[('food', 10), ('bread', 5), ('braed', 1), ('naan', 2),
				('colorful', 3), ('colourfull', 2), ('ganesha', 3),
				('ganesh', 2), ('god', 2), ('gods', 1)])


		# Read a set of synonyms
		self.ont.readSynonyms('test/synonyms.txt')

		# the synonyms are mapping correctly
		self.assertTrue(
			self.ont.getSynonym('god'), self.ont.getSynonym('gods'))
		self.assertTrue(
			self.ont.getSynonym('bread'), self.ont.getSynonym('braed'))
		self.assertTrue(self.ont.getSynonym('colorful'), 
			self.ont.getSynonym('colourfull'))
	
		# read the edgelist
		self.ont.readEdgeList('test/edgeList.txt')

		# There should be one orphan, 'food'
		self.assertEqual(self.ont.findOrphans(), ['food'])

		# orphans can't be compared...
		self.assertEqual(self.ont.compare('food', 'bread'), 0)

		# Adding the node 'ROOT', 'food' gets rid of this orphan
		self.ont.addNode('ROOT', 'food')
		self.assertEqual(self.ont.findOrphans(), [])

		# Now bread can be compared.
		# Also, test out the synonyms
		self.assertEqual(self.ont.compare('food', 'bread'), 1)
		self.assertEqual(self.ont.compare('food', 'braed'), 1)
		self.assertEqual(self.ont.compare('braed', 'naan'), 1)

		# There should be one un-placed word
		self.assertEqual(self.ont.getWords(), [('ganesha', 3)])

		# Making 'ganesha' and 'ganesh' synonyms fixes this
		self.ont.addSynonym('ganesha', 'ganesh')
		self.assertEqual(self.ont.getWords(), [])

		# Removing the node ('gods', 'ganesha') works, even though the model 
		# actually contains their synonyms 
		self.ont.removeNode('gods', 'ganesha', False)
		self.assertItemsEqual(
			self.ont.getWords(), [('ganesha',3), ('ganesh', 2)])

		# Remove the synonym ('ganesha', 'ganesh') and add the node (
		# ('god', 'ganesha').  Now ganesh is un-placed
		self.ont.removeSynonym('ganesha', 'ganesh', False)
		self.ont.addNode('god', 'ganesha')
		self.assertEqual(self.ont.getWords(), [('ganesh', 2)])

		# Add the synonym, comparison works; remove the synonym, it doesn't
		self.ont.addSynonym('ganesha', 'ganesh')
		self.assertEqual(self.ont.compare('god', 'ganesh'), 1)
		self.ont.removeSynonym('ganesh', 'ganesha', False)
		self.assertEqual(self.ont.compare('god', 'ganesh'), 0)


class AnalysisTestCase(unittest.TestCase):
	def setUp(self):
		self.dataset = dp.CleanDataset()
		self.dataset.read_csv('amt_csv/test.csv')


		self.a = analysis.Analyzer(self.dataset)


	def test_get_bag_of_labels(self):
		labels = self.a.get_bag_of_labels(
			treatments=['treatment1'], 
			images=['prime%d' % i for i in range(5)])

		self.assertEqual(
			labels, 
			set(['chinese', 'mexico', 'dance', 'mariatchi', 'blue', 
				'sombrero', 'fox', 'asia', 'hang on', 'horses', 'guitar', 
				'hours', 'russian', 'sing', 'tradition', 'stage', 'buck', 
				'new year', 'ride', 'hunt', 'dragon', 'rodeo', 'yeehaw']
			)
		)

		labels = self.a.get_bag_of_labels(
			treatments=['treatment1'], 
			images=['test0'])

		self.assertEqual(
			labels,
			set(['food', 'shiva', 'vishnu', 'ganesh', 'yogurt', 'x'])
		)



	def test_isCultural(self):
		# 'ganesha' is certainly a cultural word (an Indian god)
		self.assertTrue(self.a.isCultural('ganesha'))

		# 'chenese food' is both a cultural and food word
		self.assertTrue(self.a.isCultural('chinese food'))

		# 'russion' is a misspelling of a cultural word, and it is found
		# in the ontology as a 'synonym' for russian.
		self.assertTrue(self.a.isCultural('russion'))

		# not cultural
		self.assertFalse(self.a.isCultural('statue'))
		self.assertFalse(self.a.isCultural('raw'))

	def test_isFood(self):
		# 'ganesha' is not food-related
		self.assertFalse(self.a.isFood('ganesha'))
		self.assertFalse(self.a.isFood('statue'))

		# 'chenese food' is both a cultural and food word
		self.assertTrue(self.a.isFood('chinese food'))

		# Food-related words
		self.assertTrue(self.a.isFood('raw'))
		self.assertTrue(self.a.isFood('pizza'))


	def test_CompareSpecificity(self):

		# In the testing dataset, treatment 3 has strictly more general terms
		# than treatment 4
		result = self.a.compareSpecificity('treatment3','treatment4', 2)
		self.assertEqual(result['avgMoreMinusLess'], -6.25)


	def test_CompareCulturalSpecificity(self):
		# note that "ganesh" gets eliminated in the cultural specificity
		# because of association to "thing" via "elephant"

		result = self.a.compareCulturalSpecificity(
			'treatment1', 'treatment2', 2)
		self.assertEqual(result['avgMoreMinusLess'], 3.0)


	def test_CompareFoodSpecificity(self):

		result = self.a.compareFoodSpecificity('treatment1', 'treatment2', 2)
		self.assertLess(result['avgMoreMinusLess'], 0)




class DataProcessingTestCase(unittest.TestCase):
	def setUp(self):
		self.dataset = dp.CleanDataset()
		self.dataset.read_csv('amt_csv/test100.csv', False)

		self.dataset.aggregateCounts()
		self.dataset.calc_ktop(5)
		self.dataset.uniform_truncate(10)


	def test_NoDuplicates(self):
		seenWorkers = set()

		for treatment, entryset in self.dataset.entries.items():
			for entry in entryset:

				if entry['workerId'] in seenWorkers:
					self.fail()

				seenWorkers.add(entry['workerId'])

	def test_subSampling(self):

		# Take note of the starting number of entries per treatment
		treatments = []
		originalSizes = {}
		for treatment, entries in self.dataset.entries.items():
			treatments.append(treatment)
			originalSizes[treatment] = len(entries)

		# do the subsampling, specify a testSet of 20
		self.dataset.subsample(2)

		for treatment in treatments:
			# We expect 80 entries in each training set
			self.assertEqual(len(self.dataset.trainEntries[treatment]), 8)

			# The sum of entries in partitions should be the original total
			totalInTreatment = (
				len(self.dataset.trainEntries[treatment])
				+ len(self.dataset.testEntries[treatment]))
			self.assertEqual(totalInTreatment, originalSizes[treatment])
			self.assertEqual(len(self.dataset.testEntries[treatment]),2)


	def test_subSampleTooLarge(self):

		'''Specifying a testSetSize that is larger than the dataset --- should
		raise an exception'''

		with self.assertRaises(dp.CleanDatasetException):
			self.dataset.subsample(127)


	def test_areTreatmensEqual(self):

		# Test reading data having equally-sized treatments
		data = dp.CleanDataset()
		data.read_csv('amt_csv/test.csv')
		self.assertTrue(data.areTreatmentsEqual)

		# Test reading data having unequally-sized treatments
		data = dp.CleanDataset()
		data.read_csv('amt_csv/test100.csv')
		self.assertFalse(data.areTreatmentsEqual)


	def test_uniformTruncate(self):	
		''' ensure that after running uniform truncate, treatments have
		the same size, and that this size is size of the smallest treatment
		truncate
		'''
		# read the dataset
		data = dp.CleanDataset()
		data.read_csv('amt_csv/test100.csv')

		# Verify that treatments are being reported as equal
		self.assertFalse(data.areTreatmentsEqual)

		# get the size of the smallest treatment
		min_treatment_size = min([len(t) for t in data.entries.values()])

		# copy the all the entries for each treatment.  This is to check that
		# all entries can be accounted for after truncation
		record_entries = {}
		for treatment, entries in data.entries.items():
			record_entries[treatment] = [e['workerId'] for e in entries]

		# run the function
		data.uniform_truncate()

		# find the min and max treatment size now
		min_t_size = min([len(t) for t in data.entries.values()])
		max_t_size = max([len(t) for t in data.entries.values()])

		# all treatments should be equal in size to the original smallest
		self.assertEqual(min_t_size, min_treatment_size)
		self.assertEqual(max_t_size, min_treatment_size)
		self.assertTrue(data.areTreatmentsEqual)

		self.maxDiff=None

		# The entries removed from each treatment should be found in the
		# unused entries
		for treatment in data.entries.keys():
			treatment_entries = (
				[e['workerId'] for e in data.entries[treatment]]
				+ [e['workerId'] for e in data.unusedEntries[treatment]])
			self.assertItemsEqual(treatment_entries, record_entries[treatment])

	def test_uniformTruncateCustomSize(self):
		'''ensure that you can specify a desired truncation size, and that
		it must not be larger than the size of the smallest treatment'''

		# read the dataset
		data = dp.CleanDataset()
		data.read_csv('amt_csv/test100.csv')

		min_treatment_size = min([len(t) for t in data.entries.values()])

		data.uniform_truncate(min_treatment_size - 1)

		max_treatment_size = max([len(t) for t in data.entries.values()])
		new_min_treatment_size = min([len(t) for t in data.entries.values()])
		
		# The new max and min treatment sizes should be the requested size
		self.assertEqual(max_treatment_size, min_treatment_size - 1)
		self.assertEqual(new_min_treatment_size, min_treatment_size - 1)


	def test_uniformTruncateCustomSizeTooSmall(self):
		'''ensure that you can specify a desired truncation size, and that
		it must not be larger than the size of the smallest treatment'''

		# read the dataset
		data = dp.CleanDataset()
		data.read_csv('amt_csv/test100.csv')

		min_treatment_size = min([len(t) for t in data.entries.values()])

		with self.assertRaises(dp.CleanDatasetException):
			data.uniform_truncate(min_treatment_size + 1)


	def test_subsampleUnequalTreatments(self):
		'''For simpler implementation, subsampling, and sample rotation are
		implemented under the assumption that all treatments are of equal
		size.  This verifies that the assumption is being checked'''

		data = dp.CleanDataset()
		data.read_csv('amt_csv/test100.csv')
		with self.assertRaises(dp.CleanDatasetException):
			data.subsample(80)
	

	def test_subsampleRotation(self):
		data = dp.CleanDataset()
		data.read_csv('amt_csv/test100.csv')
		data.uniform_truncate()
		treatment_size = len(data.entries.values()[0])
		test_set_size = 3
		train_set_size = treatment_size - test_set_size
		k = treatment_size / test_set_size

		used_test_entries = {}
		for treatment in data.entries.keys():
			used_test_entries[treatment] = set()

		first = True
		for fold in range(k):
			if first:
				data.subsample(test_set_size)
				first = False
			else:
				data.rotateSubsample()

			for treatment in data.entries.keys():

				# Each treatment's test set should be the correct size
				self.assertEqual(
					len(data.testEntries[treatment]), test_set_size)

				# test set should not overlap with test set from previous fold
				thisTestSet = set(
					[e['workerId'] for e in data.testEntries[treatment]])
				overlap = (used_test_entries[treatment] & thisTestSet)
				self.assertEqual(len(overlap), 0)

				# add the new test set to the used_test_entries
				used_test_entries[treatment] |= thisTestSet


		# We should not have enough remaining entries that have never been
		# used in a test set, so an error is raised if we try to rotate again
		with self.assertRaises(dp.CleanDatasetRotationException):
			data.rotateSubsample()




class NaiveBayesDatasetTestCase(unittest.TestCase):
	def setUp(self):
		self.dataset = dp.readDataset()

	def test_getFeatures(self):

		# First make a dataste that considers only image0
		# and does not consider spearate labelling positions for each image
		nbDataset = nb.NBDataset(self.dataset, [1,2], [0])

		# Iterate over all the returned features, and keep track of whether 
		# there was ever any bad image or position
		imagesOk = True
		positionsOk = True
		for image, position, token in  nbDataset.getFeatures():
			imagesOk = imagesOk and (image == 'test0')
			positionsOk = positionsOk and (position is None)

		self.assertTrue(imagesOk)
		self.assertTrue(positionsOk)


	def test_getTestInstances(self):
		nbDataset = nb.NBDataset(self.dataset, ['treatment1','treatment2'], 
			[0], False, 25)
		testSet = nbDataset.getTestSet()
		
		# The test set should only contain entries from the treatments listed
		# in the NBDataset constructor
		self.assertItemsEqual(testSet.keys(), ['treatment1', 'treatment2']) 

		# The test sets should have 25
		for treatment, entries in testSet.items():
			self.assertEqual(len(entries), 25)
			

	def test_rotateSubsample(self):
		datasetSize = len(self.dataset.entries.values()[0])
		testsetSize = 25
		numFolds = datasetSize / testsetSize

		treatments = ['treatment0','treatment1']

		# First, we will test running multiple validations without using the
		# rotate function.  This is not true cross-validation, because the
		# test sets from alternate folds should overlap somewhat, since they
		# are randomly sampled.
		hasOverlap = False
		testSet = dict([(t,set()) for t in treatments])
		for fold in range(numFolds):

			nbDataset = nb.NBDataset(self.dataset,
				treatments, [0], True, testsetSize)

			newTestSet = nbDataset.getTestSet()

			for t in treatments:

				if len(testSet[t] & 
					set([entry.underlyingInstance['workerId'] 
					for entry in newTestSet[t]])):
					hasOverlap = True

				testSet[t] |= set([entry.underlyingInstance['workerId'] 
					for entry in newTestSet[t]])

		self.assertTrue(hasOverlap)

		# Next we try using the rotateSubsample function, which alows us to
		# do true cross-validation.  Test sets from alternate folds should not
		# overlap, because they are sampled without replacement. 
		hasOverlap = False
		testSet = dict([(t,set()) for t in treatments])
		nbDataset = nb.NBDataset(self.dataset,
			treatments, [0], True, testsetSize)

		while True:

			newTestSet = nbDataset.getTestSet()

			for t in treatments:

				if len(testSet[t] & 
					set([entry.underlyingInstance['workerId'] 
					for entry in newTestSet[t]])):
					hasOverlap = True

				testSet[t] |= set([entry.underlyingInstance['workerId'] 
					for entry in newTestSet[t]])

			# continue processing folds until there are no more folds
			try:
				nbDataset.rotateSubsample()
			except dp.CleanDatasetRotationException:
				break

		# alternate test sets should not overlap
		self.assertFalse(hasOverlap)

		# the total number of testentries should be the number of folds times
		# the size of a test set
		for t in treatments:
			self.assertEqual(len(testSet[t]), numFolds*testsetSize)




if __name__ == '__main__':
	unittest.main()


