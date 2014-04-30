import util
import unittest
import analysis
import naive_bayes as nb
import data_processing as dp
import ontology


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
		# masked since it descends from 'cultural' 
		self.assertEqual(self.ont.compare('naan', 'bread'), 0)

		# Now make a cultural-food comparison, but with strict=False
		# since naan also descends from food, which is not masked, it 
		# passes under non-strict comparison
		self.assertEqual(self.ont.compare('naan', 'bread', strict=False), -1)


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

		# try a comparison based on the synonyms
		self.assertEqual(self.ont.compare('food', 'braed'), 1)
		self.assertEqual(self.ont.compare('braed', 'naan'), 1)

		# There should be one orphan, 'food'
		self.assertEqual(self.ont.findOrphans(), ['food'])

		# Adding the node 'ROOT', 'food' gets rid of this orphan
		self.ont.addNode('ROOT', 'food')
		self.assertEqual(self.ont.findOrphans(), [])

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
		self.assertEqual(result['avgFirstMoreSpecific'], 0)
		self.assertGreater(
			result['avgFirstLessSpecific'], result['avgFirstMoreSpecific'])


	def test_CompareCulturalSpecificity(self):

		result = self.a.compareCulturalSpecificity(
			'treatment1', 'treatment2', 2)
		self.assertEqual(result['avgFirstMoreSpecific'], 6.0)
		self.assertEqual(result['avgFirstLessSpecific'], 0.0)


	def test_CompareFoodSpecificity(self):

		result = self.a.compareFoodSpecificity('treatment1', 'treatment2', 2)
		self.assertLess(
			result['avgFirstMoreSpecific'], result['avgFirstLessSpecific'])




class DataProcessingTestCase(unittest.TestCase):
	def setUp(self):
		self.dataset = dp.readDataset()

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

		# do the subsampling, first specifying only a training sample size
		self.dataset.subsample(80)

		for treatment in treatments:
			# We expect 80 entries in each training set
			self.assertEqual(len(self.dataset.trainEntries[treatment]), 80)

			# There shouldn't be any unused entries
			self.assertEqual(len(self.dataset.unusedEntries[treatment]), 0)

			# The sum of entries in partitions should be the original total
			totalInTreatment = (
				len(self.dataset.trainEntries[treatment])
				+ len(self.dataset.testEntries[treatment])
				+ len(self.dataset.unusedEntries[treatment]))
			self.assertEqual(totalInTreatment, originalSizes[treatment])

		# Subsample again, this time specifying a test sample size too
		self.dataset.subsample(80, 20)

		for treatment in treatments:
			# We expect 80 entries in each training set
			self.assertEqual(len(self.dataset.trainEntries[treatment]), 80)

			# We expect 20 entries in each test set
			self.assertEqual(len(self.dataset.testEntries[treatment]), 20)

			# The sum of entries in partitions should be the original total
			totalInTreatment = (
				len(self.dataset.trainEntries[treatment])
				+ len(self.dataset.testEntries[treatment])
				+ len(self.dataset.unusedEntries[treatment]))
			self.assertEqual(totalInTreatment, originalSizes[treatment])




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
			[0], False, 80, 20)
		testSet = nbDataset.getTestSet()
		
		# The test set should only contain entries from the treatments listed
		# in the NBDataset constructor
		self.assertItemsEqual(testSet.keys(), ['treatment1', 'treatment2']) 

		for treatment, entries in testSet.items():
			self.assertEqual(len(entries), 20)
			

if __name__ == '__main__':
	unittest.main()


