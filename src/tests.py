from collections import Counter
import util
import unittest
import analysis
import naive_bayes as nb
import data_processing as dp
import ontology
import copy
import numpy as np
import json
import wordnet_analysis as wna
from nltk.corpus import wordnet as wn


class CountsComplimentTestCase(unittest.TestCase):

	def test_get_counts_compliment(self):

		image = 5
		treatments = [0, 4, 3, 2, 1]
		individual_word_counts = {}
		for t in treatments:
			individual_word_counts[t] = analysis.get_word_counts(
					2, [t], [image], balance=0) 

		expected_word_counts_compliment = []
		for t in treatments:

			this_word_counts = individual_word_counts[t]
			other_treatments = [t_ for t_ in treatments if t_ != t]
			other_word_counts = analysis.get_word_counts(
				2, other_treatments, [image], balance=0)

			expected_word_counts_compliment.append({
				'for_i' : this_word_counts,
				'not_i' : other_word_counts
			})

		found_word_counts_compliment = analysis.get_counts_compliment(
			image, balance=0)

		self.assertEqual(
			found_word_counts_compliment, 
			expected_word_counts_compliment
		)


	def test_get_counts_compliment_2(self):

		image = 8
		treatments = [3, 2, 1, 0, 4]
		individual_word_counts = {}
		for t in treatments:
			individual_word_counts[t] = analysis.get_word_counts(
					2, [t], [image], balance=0) 

		expected_word_counts_compliment = []
		for t in treatments:

			this_word_counts = individual_word_counts[t]
			other_treatments = [t_ for t_ in treatments if t_ != t]
			other_word_counts = analysis.get_word_counts(
				2, other_treatments, [image], balance=0)

			expected_word_counts_compliment.append({
				'for_i' : this_word_counts,
				'not_i' : other_word_counts
			})

		found_word_counts_compliment = analysis.get_counts_compliment(
			image, balance=0)

		self.assertEqual(
			found_word_counts_compliment, 
			expected_word_counts_compliment
		)




class SimilarityTestCase(unittest.TestCase):

	def test_similality(self):

		# the same token sets should have a similarity of 1
		fruits = ['apple', 'orange', 'banana']
		self.assertEqual(wna.get_similarity(fruits, fruits), 1.0)


		# these sets have some similarity
		set_1 = ['stone', 'book', 'hydrogen', 'bend']
		set_2 = ['rage', 'liver', 'hydrogen', 'bend']

		size_set_1 = sum([len(wn.synsets(w)) for w in set_1])
		size_set_2 = sum([len(wn.synsets(w)) for w in set_2])
		size_intersection = (
			len(wn.synsets('hydrogen')) + len(wn.synsets('bend')))
		expected_similarity = 2 * size_intersection / float(size_set_1 
			+ size_set_2)

		self.assertEqual(
			wna.get_similarity(set_1, set_2), expected_similarity)


		# these sets have the same words, but aren't that similar in the
		# sense that they differ greatly in the word frequency 
		set_1 = ['stone']*9 + ['book']
		set_2 = ['stone'] + ['book']*9

		size_set_1 = sum([len(wn.synsets(w)) for w in set_1])
		size_set_2 = sum([len(wn.synsets(w)) for w in set_2])
		size_intersection = (
			len(wn.synsets('stone')) + len(wn.synsets('book')))
		expected_similarity = 2 * size_intersection / float(size_set_1 
			+ size_set_2)

		self.assertEqual(
			wna.get_similarity(set_1, set_2), expected_similarity)
		self.assertEqual(wna.get_similarity(set_1, set_1), 1.0)


class WordnetFoodDetectorTestCase(unittest.TestCase):

	def test_food_detection(self):
		food_detector = wna.WordnetFoodDetector()

		# obviously food
		self.assertTrue(food_detector.is_food('apple'))

		# obviously not food
		self.assertFalse(food_detector.is_food('spoon'))

		# less obvious but still food
		self.assertTrue(food_detector.is_food('cornmeal'))
		self.assertTrue(food_detector.is_food('water'))

		# multiple tokens, all are food, so overall label is food
		self.assertTrue(food_detector.is_food('sour apple'))

		# compound word is food, even though individual words are not
		self.assertFalse(food_detector.is_food('granny'))
		self.assertFalse(food_detector.is_food('smith'))
		self.assertTrue(food_detector.is_food('granny smith'))

		# compound word is food, one individual word is food, the other is not
		self.assertTrue(food_detector.is_food('potato'))
		self.assertFalse(food_detector.is_food('skins'))
		self.assertTrue(food_detector.is_food('potato skins'))

		# multiple tokens, one is not food, so overall label is not food
		self.assertFalse(food_detector.is_food('potato masher'))





class CalculateRelativeSpecificityTestCase(unittest.TestCase):

	def test_relative_specificity(self):
		synset_counts_1 = Counter({
			'bread.n.01': 3
		})
		synset_counts_2 = Counter({
			'food.n.01': 1,
			'food.n.02': 2,
			'bun.n.01': 1
		})

		# each instance of bread in counts_1 is more specific than the three 
		# food instances in counts_2, adding 9 to the score; but the instance
		# of bun in counts_2 is more specific than each of the three instances
		# of bread, removing 3... so...
		expected_relative_specificity = 6
		relative_specificity = wna.calculate_relative_specificity(
			synset_counts_1, synset_counts_2)

		self.assertEqual(relative_specificity, expected_relative_specificity)

		# ensure symmetry
		expected_relative_specificity = -6
		relative_specificity = wna.calculate_relative_specificity(
			synset_counts_2, synset_counts_1)

		self.assertEqual(relative_specificity, expected_relative_specificity)


class TestGetCorrectTreatmentForImagePosTestCase(unittest.TestCase):
	TREATMENT = {
		0: [0,4,3,2,1],
		1: [1,0,4,3,2],
		2: [2,1,0,4,3],
		3: [3,2,1,0,4],
		4: [4,3,2,1,0]
	}

	def test_get_correct_treatment(self):
		ds = dp.CleanDataset()
		for image_num in range(ds.NUM_IMAGES):
			for pos in range(ds.NUM_IMAGES):
				t = ds.get_correct_treatment_for_image_pos(image_num,pos)
				expected_treatment_num = self.TREATMENT[image_num][pos]
				expected_treatments = (
					'treatment%d'%expected_treatment_num,
					'treatment%d'%(expected_treatment_num+5)
				)
				self.assertEqual(t,expected_treatments)



class WordnetAnalysisTestCase(unittest.TestCase):

	def test_get_all_synsets(self):
		test_word = 'hot dog'
		found_synsets = wna.get_all_synsets(test_word)
		expected_synsets = set(['hotdog.n.01', 'hotdog.n.02', 'frank.n.02', 
			'hot.a.01', 'hot.s.02', 'hot.a.03', 'hot.s.04', 'hot.s.05', 
			'hot.s.06', 'blistering.s.03', 'hot.s.08', 'hot.s.09', 
			'hot.s.10', 'hot.s.11', 'hot.s.12', 'hot.s.13', 'hot.s.14', 
			'hot.s.15', 'hot.s.16', 'hot.s.17', 'hot.s.18', 'hot.s.19', 
			'hot.s.20', 'hot.s.21', 'dog.n.01', 'frump.n.01', 'dog.n.03', 
			'cad.n.01', 'frank.n.02', 'pawl.n.01', 'andiron.n.01', 
			'chase.v.01'
		])

		self.assertItemsEqual(found_synsets, expected_synsets)

	def test_map_to_synsets(self):
		dataset = dp.CleanDataset()
		dataset.read_csv('test/amt_csv/test.csv')
		found_synsets = wna.map_to_synsets(dataset, 'treatment1', ['test0'])
		expected_synsets = Counter({
			'food.n.03': 2, 'food.n.02': 2, 'food.n.01': 2, 'adam.n.03': 2, 
			'ten.s.01': 2, 'x.n.02': 2, 'siva.n.01': 2, 'shiva.n.01': 2, 
			'ten.n.01': 2, 'ganesh.n.01': 2, 'yogurt.n.01': 1, 
			'vishnu.n.01': 1
		})

		self.assertEqual(found_synsets, expected_synsets)


class SynsetHypoProxy(object):
	def __init__(self, name, children):
		self.name = name
		self.children = children
		self.do_raise = False

	def hyponyms(self):
		if self.do_raise:
			raise Exception('caching should have prevented '
				'this from being called')

		return self.children

class SynsetHyperProxy(object):
	def __init__(self, name, children):
		self.name = name
		self.children = children
		self.do_raise = False

	def hypernyms(self):
		if self.do_raise:
			raise Exception('caching should have prevented '
				'this from being called')

		return self.children


class WordnetRelativesCalculatorTestCase(unittest.TestCase):

	def test_descendant_counter(self):

		I = SynsetHypoProxy('I', [])
		H = SynsetHypoProxy('H', [])
		G = SynsetHypoProxy('G', [])
		F = SynsetHypoProxy('F', [I])
		E = SynsetHypoProxy('E', [])
		D = SynsetHypoProxy('D', [G,H])
		C = SynsetHypoProxy('C', [D,F])
		B = SynsetHypoProxy('B', [D,E])
		A = SynsetHypoProxy('A', [B,C,F])

		synset_counts = Counter(['H', 'I'])

		descendants_counter = wna.WordnetRelativesCalculator(synset_counts)
		count = descendants_counter.count(A)
		expected_count = 4

		self.assertEqual(count, expected_count)

	def test_ancester_counter(self):

		I = SynsetHyperProxy('I', [])
		H = SynsetHyperProxy('H', [])
		G = SynsetHyperProxy('G', [])
		F = SynsetHyperProxy('F', [I])
		E = SynsetHyperProxy('E', [])
		D = SynsetHyperProxy('D', [G,H])
		C = SynsetHyperProxy('C', [D,F])
		B = SynsetHyperProxy('B', [D,E])
		A = SynsetHyperProxy('A', [B,C,F])

		synset_counts = Counter(['H', 'I', 'I'])

		descendants_counter = wna.WordnetRelativesCalculator(
			synset_counts, search_ancesters=True)

		# Before testing the ancester counter, check that the do_raise 
		# flag is working... see below
		A.do_raise = True
		with self.assertRaises(Exception):
			count = descendants_counter.count(A)
		A.do_raise = False

		# now continue with the test
		count = descendants_counter.count(A)
		expected_count = 6
		self.assertEqual(count, expected_count)

		# the descendants counter caches previous results, so that calling
		# this will not actually traverse, but merely retrieve the count at
		# C.  This is checked by setting do_raise, which would raise an 
		# exception if C.hypernyms() was called
		C.do_raise = True
		count = descendants_counter.count(C)
		expected_count = 3
		self.assertEqual(count, expected_count)




class DFSTestCase(unittest.TestCase):
	def setUp(self):
		self.c6 = {'name':'c6'}
		self.c5 = {'name':'c5'}
		self.c4 = {'name':'c4'}
		self.c3 = {
			'name':'c3',
			'children': [self.c5, self.c6]
		}
		self.c2 = {
			'name': 'c2',
			'children': [self.c3, self.c5]
		}
		self.c1 = {
			'name': 'c1',
			'children': [self.c3, self.c4]
		}
		self.c0 = {
			'name': 'c0',
			'children': [self.c1, self.c2]
		}


	def get_node_hash(node):
		return node['name']

	def get_children_callback(node):
		if 'children' in node:
			return node['children']
		else:
			return []

	def inter_node_callback(node, child_vals):
		return node['name'] + ' -> ' + '; '.join(child_vals)

	def leaf_node_callback(node):
		return node['name']

	def abort_branch_callback(node):
		if node['name'] == 'c3':
			return True

		return False


	def test_dfs_cycle_safe(self):
		# make a dfs using the callbacks and 
		self.dfs = wna.DFS(
			self.get_node_hash,
			self.get_children_callback,
			self.inter_node_callback,
			self.leaf_node_callback,
			self.abort_branch_callback,
			allow_double_process=True
		)

		# introduce a cycle
		self.c2['children'].append(self.c0)

		# attempt to walk the graph containing a cycle
		# a max-recursion depth error will occur if cycle protection fails
		expected_walk = 'c0 -> c1 -> c3; c4; c2 -> c3; c5'
		found_walk = self.dfs.start_walk(self.c0)

		self.assertEqual(expected_walk, found_walk)


	def test_dfs_duplicate_allowed(self):
		# make a dfs using the callbacks and 
		self.dfs = wna.DFS(
			self.get_node_hash,
			self.get_children_callback,
			self.inter_node_callback,
			self.leaf_node_callback,
			self.abort_branch_callback,
			allow_double_process=True
		)

		found_walk = self.dfs.start_walk(self.c0)
		expected_walk = 'c0 -> c1 -> c3; c4; c2 -> c3; c5'

		self.assertEqual(expected_walk, found_walk)


	def test_dfs_duplicate_not_allowed(self):
		# make a dfs using the callbacks and 
		self.dfs = wna.DFS(
			self.get_node_hash,
			self.get_children_callback,
			self.inter_node_callback,
			self.leaf_node_callback,
			self.abort_branch_callback,
			allow_double_process=False
		)

		found_walk = self.dfs.start_walk(self.c0)
		expected_walk = 'c0 -> c1 -> c3; c4; c2 -> c5'

		self.assertEqual(expected_walk, found_walk)



class OntologyTestCase(unittest.TestCase):
	def setUp(self):
		self.ont = ontology.Ontology()
		self.ont.readOntology('test/ontology/test_ontology.json')


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
		self.ont.readWords('test/ontology/words.txt')
		self.assertItemsEqual(
			self.ont.getWords(1),
			[('food', 10), ('bread', 5), ('braed', 1), ('naan', 2),
				('colorful', 3), ('colourfull', 2), ('ganesha', 3),
				('ganesh', 2), ('god', 2), ('gods', 1)])

		# Read a set of synonyms
		self.ont.readSynonyms('test/ontology/synonyms.txt')

		# the synonyms are mapping correctly
		self.assertTrue(
			self.ont.getSynonym('god'), self.ont.getSynonym('gods'))
		self.assertTrue(
			self.ont.getSynonym('bread'), self.ont.getSynonym('braed'))
		self.assertTrue(self.ont.getSynonym('colorful'), 
			self.ont.getSynonym('colourfull'))
	
		# read the edgelist
		self.ont.readEdgeList('test/ontology/edgeList.txt')

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
		self.dataset.read_csv('test/amt_csv/test.csv')

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
		self.dataset.read_csv('test/amt_csv/test100.csv', False)

		self.dataset.aggregateCounts()
		self.dataset.calc_ktop(5)
		self.dataset.uniform_truncate(10)

		self.exp_2_dataset = dp.CleanDataset(is_exp_2_dataset=True)
		self.exp_2_dataset.read_csv('test/amt_csv/exp_2_test_13.csv')


	def test_dataset_adaptor(self):
		'''
		make sure that the dataset adaptor converts a clean dataset into
		the expected format for the naive bayes cross validation tester
		'''

		dataset = dp.CleanDataset()
		dataset.read_csv('test/amt_csv/test.csv')
		found_naive_bayes_dataset = dp.clean_dataset_adaptor(dataset,
			treatments=['treatment1', 'treatment2', 'treatment3', 
				'treatment4']
		)

		expected_naive_bayes_dataset = {
			'treatment1': [
				('treatment1', ('test0', 'ganesh'), ('test0', 'shiva'), 
					('test0', 'vishnu'), ('test0', 'food'), 
					('test0', 'yogurt')),
				('treatment1', ('test0', 'ganesh'), ('test0', 'shiva'), 
					('test0', 'x'), ('test0', 'food'), ('test0', 'x'))
			],
			'treatment2': [
				('treatment2', ('test0', 'god'), ('test0', 'hinduism'), 
					('test0', 'dairy'), ('test0', 'pastry'), 
					('test0', 'lamb')),
				('treatment2', ('test0', 'god'), ('test0', 'hinduism'), 
					('test0', 'x'), ('test0', 'pastry'), ('test0', 'x'))
			],
			'treatment3': [
				('treatment3', ('test0', 'cultural'), ('test0', 'food'), 
					('test0', 'thing'), ('test0', 'activity'), 
					('test0', 'adj')),
				('treatment3', ('test0', 'x'), ('test0', 'food'), 
					('test0', 'thing'), ('test0', 'activity'), 
					('test0', 'adj'))
			],
			'treatment4': [
				('treatment4', ('test0', 'god'), ('test0', 'hinduism'), 
					('test0', 'feast'), ('test0', 'naan'), 
					('test0', 'curry')),
				('treatment4', ('test0', 'god'), ('test0', 'hinduism'), 
					('test0', 'feast'), ('test0', 'naan'), ('test0', 'x'))
			]
		}

		self.assertEqual(
			found_naive_bayes_dataset, expected_naive_bayes_dataset)

		# now try restricting the treatments
		found_naive_bayes_dataset = dp.clean_dataset_adaptor(dataset,
			treatments=['treatment2', 'treatment4']
		)
		del expected_naive_bayes_dataset['treatment1']
		del expected_naive_bayes_dataset['treatment3']
		self.assertEqual(
			found_naive_bayes_dataset, expected_naive_bayes_dataset)



	def test_exp_2_data_permutation(self):
		'''
		The test images in experiment 2 were presented in various different
		permutations, so the reader needs to keep track of which permutation
		was used in a given entry, and make sure that labels get assigned
		to the correct images.  Test that this is done properly
		'''

		expected_test0_labels = {
			'treatment0': ['romantic', 'cheese', 'loaf', 'comfy', 'meal'],
			'treatment1': ['bread', 'candle', 'breakfast', 'wine', 'cakes'],
			'treatment2': ['dinner', 'wine', 'cheese', 'bread', 'coffee'],
			'treatment3': ['dinner', 'food', 'meal', 'candles', 'coffee'],
			'treatment4': ['pinnapple', 'apples', 'wine', 'cheese', 'coffee'],
			'treatment5': ['cheese', 'bread', 'candles', 'bread', 
				'cup of tea'],
			'treatment6': ['dinning table', 'dinning items', 
				'dinning and kicten table', 'kicten items', 'kicten table'],
			'treatment7': ['cake', 'cup', 'fork', 'coffee', 'knife'],
			'treatment8': ['drinks', 'pastries', 'breads', 'scrumptious', 
				'cheese'],
			'treatment9': ['crystal wine glasses', 'fresh pastry', 
				'cup of expresso', 'evening supper', 'fresh bread and cheese'],
			'treatment10': ['bread', 'cheese', 'pastries', 'coffee', 'wine'],
			'treatment11': ['food', 'dessert', 'elegant', 'entertainment', 
				'sweets'],
			'treatment12': ['party', 'dessert', 'cheese', 'wine', 'bread'],
			'treatment13': ['food', 'fest', 'dinner', 'romance', 'delicious'],
		}

		expected_test4_labels = {
			'treatment0': ['tea', 'brew', 'brittle', 'cookie', 'orange'],
			'treatment1': ['bread', 'lemon', 'coffee', 'mushroom', 'flowers'],
			'treatment2': ['tea', 'lemon', 'dessert', 'honey', 'cookie'],
			'treatment3': ['tea', 'flowers', 'dessert', 'fruit', 'cup'],
			'treatment4': ['coffee', 'lemon', 'candies', 'cakes', 'pastries'],
			'treatment5': ['tea cup', 'orange slices', 'flowers', 'biscuits', 
				'spoon'],
			'treatment6': ['Snack items', 'Snack and bake items', 'Bake items',
				'Food items', 'Snack and food items'],
			'treatment7': ['Coffee', 'Cake', 'Honey', 'Flower', 'Orange'],
			'treatment8': ['hosting', 'dessert', 'wedding', 'tea', 'formal'],
			'treatment9': ['Cup of Tea', 'Pasteries', 'wedges of Lemon', 
				'Fresh cut flowers', 'Center Piece'],
			'treatment10': ['Orange', 'Flower', 'Pastrie', 'Tea', 'Sunny'],
			'treatment11': ['tea', 'orange', 'cookies', 'flowers', 'spoon'],
			'treatment12': ['Desert', 'Tea', 'Orange slices', 'presentation', 
				'Yum'],
			'treatment13': ['breakfast', 'treats', 'fancy', 'tasty', 
				'delicious'],
		}


		for treatment_id, treatment in self.exp_2_dataset.entries.items():

			# pull out the expeted words for this treatment
			expected_words = expected_test0_labels[treatment_id]

			# get all the test0 words for this treatment
			found_words = []
			for key, value in treatment[0].items():
				if isinstance(key, tuple):
					if key[0] == 'test0':
						found_words.append(value)

			# check that we got the words we expected
			self.assertItemsEqual(found_words, expected_words)






		
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
		data.read_csv('test/amt_csv/test.csv')
		self.assertTrue(data.areTreatmentsEqual)

		# Test reading data having unequally-sized treatments
		data = dp.CleanDataset()
		data.read_csv('test/amt_csv/test100.csv')
		self.assertFalse(data.areTreatmentsEqual)


	def test_uniformTruncate(self):	
		''' ensure that after running uniform truncate, treatments have
		the same size, and that this size is size of the smallest treatment
		truncate
		'''
		# read the dataset
		data = dp.CleanDataset()
		data.read_csv('test/amt_csv/test100.csv')

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
		data.read_csv('test/amt_csv/test100.csv')

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
		data.read_csv('test/amt_csv/test100.csv')

		min_treatment_size = min([len(t) for t in data.entries.values()])

		with self.assertRaises(dp.CleanDatasetException):
			data.uniform_truncate(min_treatment_size + 1)


	def test_subsampleUnequalTreatments(self):
		'''For simpler implementation, subsampling, and sample rotation are
		implemented under the assumption that all treatments are of equal
		size.  This verifies that the assumption is being checked'''

		data = dp.CleanDataset()
		data.read_csv('test/amt_csv/test100.csv')
		with self.assertRaises(dp.CleanDatasetException):
			data.subsample(80)
	

	def test_subsampleRotation(self):
		data = dp.CleanDataset()
		data.read_csv('test/amt_csv/test100.csv')
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


class NaiveBayesCrossValidationTest(unittest.TestCase):
	def setUp(self):

		self.easy_dataset = {
			'fruits': [
				('fruits', 'apple', 'apple', 'orange'),
				('fruits', 'pear', 'apple', 'orange'),
				('fruits', 'apple', 'pear', 'pear'),
				('fruits', 'pear', 'orange', 'pear'),
				('fruits', 'apple', 'pear', 'orange')
			],
			'colours': [
				('colours', 'cyan', 'orange', 'puple'),
				('colours', 'cyan', 'cyan', 'purple'),
				('colours', 'purple', 'cyan', 'orange'),
				('colours', 'purple', 'purple', 'orange'),
				('colours', 'orange', 'cyan', 'cyan'),
			]
		}

		self.unlearnable_dataset = {
			'fruits': [
				('fruits', 'pineapple', 'cherry', 'lemon'),
				('fruits', 'banana', 'lime', 'grapefruit'),
				('fruits', 'tomato', 'watermelon', 'grape'),
			],

			'colours': [
				('colours', 'red', 'green', 'yellow'),
				('colours', 'blue', 'purple', 'brown'),
				('colours', 'pink', 'indigo', 'black'),
			]
		}

		self.adversarial_dataset = {
			'fruits': [
				('fruits', 'pineapple', 'cherry', 'lemon'),
				('fruits', 'banana', 'lime', 'grapefruit'),
				('fruits', 'tomato', 'watermelon', 'grape'),
				('fruits', 'red', 'green', 'yellow'),
				('fruits', 'blue', 'purple', 'brown'),
				('fruits', 'pink', 'indigo', 'black'),
			],

			'colours': [
				('colours', 'pineapple', 'cherry', 'lemon'),
				('colours', 'banana', 'lime', 'grapefruit'),
				('colours', 'tomato', 'watermelon', 'grape'),
				('colours', 'red', 'green', 'yellow'),
				('colours', 'blue', 'purple', 'brown'),
				('colours', 'pink', 'indigo', 'black'),
			]
		}

	def test_performance(self):
		cross_validator = nb.NaiveBayesCrossValidationTester(self.easy_dataset)
		overall_accuracy =  cross_validator.cross_validate(5)
		self.assertEqual(overall_accuracy, 1.0)

		cross_validator = nb.NaiveBayesCrossValidationTester(
			self.unlearnable_dataset)
		overall_accuracy =  cross_validator.cross_validate(3)
		self.assertEqual(overall_accuracy, 0.5)

		cross_validator = nb.NaiveBayesCrossValidationTester(
			self.adversarial_dataset)
		overall_accuracy =  cross_validator.cross_validate(6)
		self.assertTrue(overall_accuracy <= 0.5)



	def test_partitioning(self):
	
		cross_validator = nb.NaiveBayesCrossValidationTester(self.easy_dataset)
		num_examples_per_class = len(self.easy_dataset['fruits'])

		# we'll try a series of different cross validation partitionning 
		# schedules, one for each possible number of folds
		# for each, we check that all examples get used once and only once
		# for testing, and that in a given fold, no example is used in both
		# the test and training sets.
		for num_folds in range(2, num_examples_per_class + 1):

			test_set_size = num_examples_per_class / int(num_folds)
			used_test_examples = set()
			for fold in range(num_folds):

				is_last = bool(fold == num_folds - 1)
				test_set = cross_validator.extract_test_set(
					fold, test_set_size, is_last)

				all_training_examples = list(
					cross_validator.classifier.examples)
				all_test_examples = reduce(lambda x,y: x+y, test_set.values())

				# examples shouldn't get used more than once for testing
				self.assertEqual(
					len(used_test_examples & set(all_test_examples)), 0)

				used_test_examples |= set(all_test_examples)
				for class_name in test_set:
					self.assertTrue(
						len(test_set[class_name]) <=  test_set_size
						or is_last
					)

				# There should be no overlap between the testing and training 
				# examples
				pollution = set(all_test_examples) & set(all_training_examples)
				self.assertEqual(len(pollution), 0)

				cross_validator.put_test_set_back_in_training_set(test_set)

			all_examples = set(
					self.easy_dataset['colours'] + self.easy_dataset['fruits'])

			# all examples should be used once for testing
			self.assertItemsEqual(all_examples, used_test_examples)
			



	def test_overall_accuracy(self):
		pass


class NewNaiveBayesTextClassifierTestCase(unittest.TestCase):
	def setUp(self):
		self.CLASS_1 = 'pies'
		self.CLASS_2 = 'fruits'

		self.examples_for_class_1 = [
			(self.CLASS_1, 'apple', 'rhubarb', 'cherry'),
			(self.CLASS_1, 'sugar', 'apple', 'meat'),
			(self.CLASS_1, 'pecan', 'apple', 'pumpkin'),
		]

		self.examples_for_class_2 = [
			(self.CLASS_2, 'apple', 'orange', 'lemon'),
			(self.CLASS_2, 'cherry', 'apple', 'grape'),
			(self.CLASS_2, 'raspberry', 'apple', 'lime'),
			(self.CLASS_2, 'banana', 'strawberry', 'grape'),
		]

		self.all_examples = (
			self.examples_for_class_1 + self.examples_for_class_2)

		# flatten the examples to keep a set of all features observed;
		# and preserve multiplicity
		self.features_for_class_1 = reduce(lambda x,y: x+y,
				map(lambda x: x[1:], self.examples_for_class_1))

		self.features_for_class_2 = reduce(lambda x,y: x+y,
				map(lambda x: x[1:], self.examples_for_class_2))

		self.all_features_list = (
			self.features_for_class_1 + self.features_for_class_2)
		self.all_features_set = set(self.all_features_list)

		self.classifier = nb.NewNaiveBayesTextClassifier()
		self.classifier.train(self.all_examples)


	def test_predictions(self):
		# An obvious classification as 'CLASS_1'
		self.assertEqual(
			self.classifier.classify(('meat', 'rhubarb', 'pumpkin')),
			self.CLASS_1
		)

		# An obvious classification as 'CLASS_2'
		self.assertEqual(
			self.classifier.classify(('lemon', 'grape', 'banana')),
			self.CLASS_2
		)

		# Less obvious, but should be CLASS_1
		self.assertEqual(
			self.classifier.classify(('lemon', 'rhubarb', 'pumpkin')),
			self.CLASS_1
		)

		# An example that is probable for both classes.
		# Close to a tie, but CLASS_2 has a greater prior, so wins
		self.assertEqual(
			self.classifier.classify(('apple', 'cherry')),
			self.CLASS_2
		)

		# An example that is improbable for both classes.
		# Close to a tie, but CLASS_2 has a greater prior, so wins
		self.assertEqual(
			self.classifier.classify(('rhubarb', 'lemon')),
			self.CLASS_2
		)


	def test_conditional_probability_calculations(self):
		for feature in self.all_features_set:

			#### test conditional probabilities related to class 1

			# test for class 1 without add-1-smoothing
			num_occurences_class_1 = self.features_for_class_1.count(feature)
			num_examples_class_1 = len(self.examples_for_class_1) 

			found_cond_prob = self.classifier.get_cond_prob(
				feature, self.CLASS_1, use_add_one_smoothing=False)

			expected_cond_prob = (
				num_occurences_class_1 / float(num_examples_class_1))

			self.assertEqual(found_cond_prob, expected_cond_prob)

			# test for class 1 *with* add-1-smoothing
			num_occurences_class_1 += 1
			num_examples_class_1 += len(self.all_features_set)

			found_cond_prob = self.classifier.get_cond_prob(
				feature, self.CLASS_1)
			expected_cond_prob = (
				num_occurences_class_1 / float(num_examples_class_1))

			#### reapeat for class 2

			# test for class 2 without add-1-smoothing
			num_occurences_class_2 = self.features_for_class_2.count(feature)
			num_examples_class_2 = len(self.examples_for_class_2) 

			found_cond_prob = self.classifier.get_cond_prob(
				feature, self.CLASS_2, use_add_one_smoothing=False)

			expected_cond_prob = (
				num_occurences_class_2 / float(num_examples_class_2))

			self.assertEqual(found_cond_prob, expected_cond_prob)

			# test for class 2 *with* add-2-smoothing
			num_occurences_class_2 += 1
			num_examples_class_2 += len(self.all_features_set)

			found_cond_prob = self.classifier.get_cond_prob(
				feature, self.CLASS_2)
			expected_cond_prob = (
				num_occurences_class_2 / float(num_examples_class_2))

			self.assertEqual(found_cond_prob, expected_cond_prob)


	def test_train(self):

		# test that the global feature counts are right
		for feature in self.all_features_set:
			self.assertEqual(
				self.classifier.global_feature_counts[feature], 
				reduce(lambda x,y: x+y, self.all_examples).count(feature)
			)

		# test that the feature counts are right on a per-class basis
		for feature in self.all_features_set:
			self.assertEqual(
				self.classifier.feature_counts[self.CLASS_1][feature],
				self.features_for_class_1.count(feature)
			)
			self.assertEqual(
				self.classifier.feature_counts[self.CLASS_2][feature],
				self.features_for_class_2.count(feature)
			)
					
		# test that class counts are correct:
		self.assertEqual(self.classifier.class_counts[self.CLASS_1], 
			len(self.examples_for_class_1))
		self.assertEqual(self.classifier.class_counts[self.CLASS_2], 
			len(self.examples_for_class_2))

		# test the tallies of unique features and represented classes 
		self.assertEqual(self.classifier.get_num_classes(), 2)
		self.assertEqual(self.classifier.get_num_features(), 
			len(self.all_features_set))

		#### try removing examples to ensure that all tallies are updated 
		#### correctly
		self.classifier.remove_example(self.examples_for_class_1[0])

		# test that the global feature counts are right
		for feature in self.all_features_set:
			self.assertEqual(
				self.classifier.global_feature_counts[feature], 
				reduce(lambda x,y: x+y, self.all_examples).count(feature)
				- self.examples_for_class_1[0].count(feature)
			)

		# test that the feature counts are right on a per-class basis
		for feature in self.all_features_set:
			self.assertEqual(
				self.classifier.feature_counts[self.CLASS_1][feature],
				self.features_for_class_1.count(feature) 
				- self.examples_for_class_1[0].count(feature)
			)
			self.assertEqual(
				self.classifier.feature_counts[self.CLASS_2][feature],
				self.features_for_class_2.count(feature)
			)

		# test that class counts are correct:
		self.assertEqual(self.classifier.class_counts[self.CLASS_1], 
			len(self.examples_for_class_1) - 1)
		self.assertEqual(self.classifier.class_counts[self.CLASS_2], 
			len(self.examples_for_class_2))

		# test the tallies of unique features and represented classes
		self.assertEqual(self.classifier.get_num_classes(), 2)
		self.assertEqual(self.classifier.get_num_features(), 
			len(self.all_features_set) - 1)

		#### remove all the examples for class 1!
		for example in self.examples_for_class_1[1:]:
			self.classifier.remove_example(example)

		# test the tallies of unique features and represented classes
		self.assertEqual(self.classifier.get_num_classes(), 1)
		self.assertEqual(self.classifier.get_num_features(), 
			len(set(self.features_for_class_2)))



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


