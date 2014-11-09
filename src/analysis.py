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
import data_processing as dp
import numpy as np
import json
import wordnet_analysis as wna
from collections import defaultdict, Counter
import os
import copy

# Number of stardard deviations equivalent to the % condifdence for a 
# normal variate
CONFIDENCE_95 = 1.96
CONFIDENCE_99 = 2.975
TEST_IMAGES = ['test%d'%i for i in range(5)]

EXP1_TREATMENTS = {
	'1_img_food': [0],
	'1_img_cult': [1],
	'1_img_ingr': [2],
	'1_wfrm_food': [3],
	'1_wfrm_cult': [5],
}
EXP2_TREATMENTS = {
	'2_img_food': range(5),
	'2_img_obj': range(5,10),
	'2_wfrm_food': [10],
	'2_wfrm_obj': [11],
	'2_sfrm_food': [12],
	'2_sfrm_obj': [13],
}


class AnalysisError(Exception):
	pass


class SpellingCorrector(object):
	'''
	loads the specified portion of the dataset, and uses the 
	WordnetSpellChecker to find misspelled words and their plausible correct
	spellings.  This is basically an adaptor between the SimpleDataset, 
	the WordnetSpellChecker, and the SimSweeper (which allows for 
	multiprocessing.
	'''

	def __init__(self):
		pass


	def run(
		self,
		which_experiment=1,
		class_idxs=[0,1],
		img_idxs=range(5,10),
	):
		corpus = dp.SimpleDataset(
			which_experiment=which_experiment,
			show_token_pos=False,
			show_plain_token=True,
			show_token_img=False,
			do_split=False,
			class_idxs=class_idxs,
			img_idxs=img_idxs,
			spellcheck=False,
			get_syns=False,
			balance_classes=False,
		).vocab_list

		spell_checker = wna.WordnetSpellChecker(num_to_return=10)
		return spell_checker.auto_correct(corpus)



def get_all_word_counts(fname='data/new_data/vocabulary.json'):

	results = {}
	treatments = {
		'exp1.task.food': (1,0),
		'exp1.task.cult': (1,1),
		'exp1.task.ingr': (1,2),
		'exp1.frame.ingr': (1,3),
		'exp1.frame.cult': (1,5),
		'exp2.task.food': (2,0),
		'exp2.task.obj': (2,5),
		'exp2.frame.food': (2,10),
		'exp2.frame.obj': (2,11),
		'exp2.frame*.food': (2,12),
		'exp2.frame*.obj': (2,13),
	}

	experiment = 1
	vocab_sizes = {}
	
	for key, (exp, treatment) in treatments.items():

		this_vocabs = []
		vocab_sizes[key] = this_vocabs

		for image in range(5,10):

			this_vocabs.append(
				len(get_word_counts(exp, [treatment], [image])))

	return vocab_sizes



def get_word_counts(experiment, treatments, images, balance=119):
	'''
		set balance=0 to apply no truncation of the dataset (which is 
		normally done to balance the number of data points per treatment).
	'''
	d = dp.SimpleDataset(
		which_experiment=experiment,
		show_token_pos=False,
		show_token_img=False,
		do_split=False,
		class_idxs=treatments,
		img_idxs=images,
	)

	if balance > 0:
		d.balance_classes(119)

	return d.vocab_counts


def calculate_vocabulary_sizes(fname='data/new_data/vocabulary.json'):
	results_file = open(fname, 'w')

	treatments = {
		'exp1.task.food': (1,0),
		'exp1.task.cult': (1,1),
		'exp1.task.ingr': (1,2),
		'exp1.frame.food': (1,3),
		'exp1.frame.cult': (1,5),
		'exp2.task.food': (2,0),
		'exp2.task.obj': (2,5),
		'exp2.frame.food': (2,10),
		'exp2.frame.obj': (2,11),
		'exp2.frame*.food': (2,12),
		'exp2.frame*.obj': (2,13),
	}

	results = {}
	for key, (exp, treatment) in treatments.items():
		results[key] = []
		for image in range(5,10):
			results[key].append(
				len(get_word_counts(exp, [treatment], [image])))

	results_file.write(json.dumps(results, indent=2))
	return results



def calculate_similarity(fname='data/new_data/similarity.json'):

	# open the file that we will write to
	results_file = open(fname, 'w')
	treatments = {
		'exp1.task.food': (1,0),
		'exp1.task.cult': (1,1),
		'exp1.task.ingr': (1,2),
		#'exp1.frame.ingr': (1,3),
		#'exp1.frame.cult': (1,5),
		'exp2.task.food': (2,0),
		'exp2.task.obj': (2,5),
		#'exp2.frame.food': (2,10),
		#'exp2.frame.obj': (2,11),
		#'exp2.frame*.food': (2,12),
		#'exp2.frame*.obj': (2,13),
	}

	test_sets = {
		'exp1': (1, [0,1,2]),
		'exp2': (2, [0,5])
	}

	# get the vocabularies given for initial tasks
	test_vocabularies = {}
	prime_vocabularies = {}

	for key, (exp, treatment) in treatments.items():
		prime_vocabularies[key] = get_word_counts(exp, [treatment], range(5))
		print len(prime_vocabularies[key])

	for key, (exp, treatments) in test_sets.items():
		test_vocabularies[key] = get_word_counts(exp, treatments, range(5,10))

	similarities = {}
	for key, prime_vocab in prime_vocabularies.items():

		test_key = key.split('.')[0]
		test_vocab = test_vocabularies[test_key]

		similarities[key] = wna.get_similarity(prime_vocab, test_vocab)

	results_file.write(json.dumps(similarities, indent=2))
	return similarities



def calculate_longit_self_specificity():
	'''
	This analysis looks at whether the labels attributed to a *test* image
	become *more specific* as the image moves from position 1 in the test
	set through position 5.  The comparison is not between two priming 
	treatments (which is what is usually done), but between the labels
	attributed to the image in the position in question and the pool of
	labels attributed to that image at all other positions.
	'''
	fname = 'data/new_data/self_specificity.json'
	results_fh = open(fname, 'w')

	all_specificities = {}
	for treatment_set in ['food', 'object']:

		specificities = []
		all_specificities[treatment_set] = specificities

		if treatment_set == 'food':
			treatments_of_interest = range(5) 
		elif treatment_set == 'object':
			treatments_of_interest = range(5,10)
		else:
			raise AnalysisError(
				'Inconsistent value for treatmens of interest')

		for image in range(5,10):

			this_image_results = []
			specificities.append(this_image_results)
			word_counts = get_counts_compliment(image, treatments_of_interest)

			for word_count in word_counts:

				this_image_results.append(wna.calculate_relative_specificity(
					word_count['for_i'], word_count['not_i']))

	results_fh.write(json.dumps(all_specificities, indent=2))
	return all_specificities
	


def get_counts_compliment(
		image,
		treatments_of_interest=range(5),
		balance=119
	):
	'''
	for a given image (in the test set of experiment 2), it gets the 
	word counts associated to that image when that image is in position
	i of the test set, as well as the words associated to that image
	in all other positions pooled together, and does that for all i
	'''

	# figure the order of treatments to look at to see this image
	# at each possible position within the test set, consecutively
	treatments = [0]*5
	for treatment in treatments_of_interest:
		treatment_order = dp.permute(range(5,10), treatment).index(image)
		treatments[treatment_order] = treatment

	word_counts = []
	for treatment in treatments:

		other_treatments = copy.deepcopy(treatments_of_interest)
		other_treatments.remove(treatment)

		word_counts.append({
			'for_i': get_word_counts(2, [treatment], [image], balance),
			'not_i': get_word_counts(2, other_treatments, [image], balance)
		})

	return word_counts



def calculate_all_relative_specificities(ignore_food=False):

	fname= 'data/new_data/specificity.json'
	if ignore_food:
		fname= 'data/new_data/specificity_ignore_food.json'

	write_fh = open(fname, 'w')

	#images = ['test%d'% i for i in range(5)]
	images = range(5,10)
	results = {}

	## start with experiment 1
	results['img_food_cult'] = []
	results['img_food_ingr'] = []
	results['img_ingr_cult'] = []
	results['frm_food_cult'] = []

	results['wfrm_food_cult'] = []
	results['img_food_obj'] = []
	results['wfrm_food_obj'] = []
	results['sfrm_food_obj'] = []
	results['img_food_sfrm_food'] = []
	results['img_obj_sfrm_obj'] = []
	results['img_cult_wfrm_cult'] = []
	results['img_food_wfrm_food'] = []

	for image in images:
		# compare the img_priming
		counts_food = get_word_counts(1, [0], [image])
		counts_cult = get_word_counts(1, [1], [image])
		results['img_food_cult'].append(wna.calculate_relative_specificity(
			counts_food, counts_cult, ignore_food))

		# compare the img_priming
		counts_food = get_word_counts(1, [0], [image])
		counts_ingr = get_word_counts(1, [2], [image])
		results['img_food_ingr'].append(wna.calculate_relative_specificity(
			counts_food, counts_ingr, ignore_food))

		# compare the img_priming
		counts_ingr = get_word_counts(1, [2], [image])
		counts_cult = get_word_counts(1, [1], [image])
		results['img_ingr_cult'].append(wna.calculate_relative_specificity(
			counts_ingr, counts_cult, ignore_food))

		# compare the img_priming
		counts_food = get_word_counts(1, [3], [image])
		counts_cult = get_word_counts(1, [5], [image])
		results['frm_food_cult'].append(wna.calculate_relative_specificity(
			counts_food, counts_cult, ignore_food))

		# compare the exp1 framing 
		counts_food = get_word_counts(1, [3], [image])
		counts_cult = get_word_counts(1, [5], [image])
		results['wfrm_food_cult'].append(wna.calculate_relative_specificity(
			counts_cult, counts_food, ignore_food))

		# next work on experiment 2
		d = dp.readDataset(True)

		# compare the img_priming
		counts_food = get_word_counts(2, [0], [image])
		counts_obj = get_word_counts(2, [5], [image])
		#counts_food = get_word_counts(2, range(5), [image])
		#counts_obj = get_word_counts(2, range(5,10), [image])
		results['img_food_obj'].append(wna.calculate_relative_specificity(
			counts_food, counts_obj, ignore_food))

		# compare the weak framing
		counts_food = get_word_counts(2, [10], [image])
		counts_obj = get_word_counts(2, [11], [image])
		results['wfrm_food_obj'].append(wna.calculate_relative_specificity(
			counts_food, counts_obj, ignore_food))

		# compare strong framing
		counts_food = get_word_counts(2, [12], [image])
		counts_obj = get_word_counts(2, [13], [image])
		results['sfrm_food_obj'].append(wna.calculate_relative_specificity(
			counts_food, counts_obj, ignore_food))

		# compare img food to sfrm food
		counts_img_food = get_word_counts(2, [0], [image])
		#counts_img_food = get_word_counts(2, range(5), [image])
		counts_sfrm_food = get_word_counts(2, [12], [image])
		results['img_food_sfrm_food'].append(
			wna.calculate_relative_specificity(counts_img_food, 
				counts_sfrm_food, ignore_food))

		# compare img obj to sfrm obj
		counts_img_obj = get_word_counts(2, [5], [image])
		#counts_img_obj = get_word_counts(2, range(5,10), [image])
		counts_sfrm_obj = get_word_counts(2, [13], [image])
		results['img_obj_sfrm_obj'].append(
			wna.calculate_relative_specificity(counts_img_obj, 
				counts_sfrm_obj, ignore_food))

		# compare img food to wfrm food
		counts_img_food = get_word_counts(1, [0], [image])
		counts_wfrm_food = get_word_counts(1, [3], [image])
		results['img_food_wfrm_food'].append(
			wna.calculate_relative_specificity(counts_img_food, 
				counts_wfrm_food, ignore_food))

		# compare img obj to sfrm obj
		counts_img_cult = get_word_counts(2, [1], [image])
		counts_wfrm_cult = get_word_counts(2, [5], [image])
		results['img_cult_wfrm_cult'].append(
			wna.calculate_relative_specificity(counts_img_cult, 
				counts_wfrm_cult, ignore_food))

	#for comparison in results:
	#	results[comparison] = np.mean(results[comparison])

	write_fh.write(json.dumps(results, indent=2))
	return results


def get_food_proportions():
	fname = 'data/new_data/food.json'
	write_fh = open(fname, 'w')

	food_detector = wna.WordnetFoodDetector()

	experiment = 1
	result = {}
	for i, experiment_group in enumerate([EXP1_TREATMENTS, EXP2_TREATMENTS]):
		for exp_name, treatment_idxs in experiment_group.items():

			d = dp.SimpleDataset(
				which_experiment=i+1,
				show_token_pos=False,
				show_token_img=False,
				class_idxs=treatment_idxs,
				img_idxs=range(5,10)
			)

			result[exp_name] = {
				'num_words':0,
				'num_food_words':0,
			}
			this_result = result[exp_name]

			for word, count in d.vocab_counts.items():
				this_result['num_words'] += count
				if food_detector.is_food(word):
					this_result['num_food_words'] += count

			this_result['fract_food'] = (
				this_result['num_food_words'] / float(this_result['num_words']))

	write_fh.write(json.dumps(result, indent=2))
	return result





def avg_l1():
	fnames = os.popen('ls data/new_data/l1_spellcorrected | grep ^l1').read().split()
	print 'y'
	print fnames

	for fname in fnames:
		data = json.loads(open('data/new_data/l1/'+fname).read())
		aggregates = data['aggregates']
		exp2_data = data['img_food_obj']
		exp2_data = reduce(lambda x,y: x + y[1], exp2_data.items(), [])
		print os.path.split(fname)[-1], np.mean(exp2_data)
		print aggregates
		print ''


def try_everything():
	for show_token_pos in [True, False]:
		for do_split in [True, False]:
			for remove_stops in [True, False]:
				for lemmatize in [True, False]:
					for spellcheck in [True, False]:

						# get the file name sorted out
						fname = 'l1'
						fname += 'showpos' if show_token_pos else ''
						fname += 'split' if do_split else ''
						fname += 'nostops' if remove_stops else ''
						fname += 'lem' if lemmatize else ''
						fname += '.json'

						# now do it
						bound_l1(
							fname='data/new_data/l1_spellcorrected/' + fname,
							show_token_pos=show_token_pos,
							do_split=do_split,
							remove_stops=remove_stops,
							lemmatize=lemmatize,
							spellcheck=spellcheck
						)



def bound_l1(
		fname='data/new_data/l1.json',
		show_token_pos=True,
		show_plain_token=True,
		do_split=True,
		remove_stops=True,
		lemmatize=True,
		spellcheck=False,
		#balance_classes
	):
	''' 
	also known as determine the classifier's accuracy using 
	cross-validation
	'''
	# first things first: try to open the output file.  No sense doing 
	# calculations if there's nowhere to put them!
	output_fh = open(fname, 'w')

	# first, do this for the old data
	ds_exp1 = dp.readDataset(is_exp_2_dataset=False)
	food_cult_accuracy = []
	for image in TEST_IMAGES:
		food_cult_accuracy.append(
			_do_cross_validation(
				ds_exp1, ['treatment0', 'treatment1'], [image]))

	wfrm_food_cult = []
	for image in TEST_IMAGES:
		wfrm_food_cult.append(
			_do_cross_validation(
				ds_exp1, ['treatment3', 'treatment6'], [image]))

	# next, do this for the new data
	ds_exp2 = dp.readDataset(is_exp_2_dataset=True)

	# look at the distinguishability of IMG:FOOD and IMG:OBJ on a per-image
	# per-position basis
	img_food_obj_accuracy = defaultdict(lambda: [])
	for image_num in range(5):
		for pos in range(5):
			treatments = ds_exp2.get_correct_treatments(image_num, pos)
			accuracy = _do_cross_validation(
				ds_exp2, treatments, ['test%d'%image_num])
			img_food_obj_accuracy['test%d'%image_num].append(accuracy)

	# now determine the distinguishability of wFRM:FOOD and wFRM:OBJ on a 
	# per-image basis
	wfrm_food_obj_accuracy = []
	for image in TEST_IMAGES:
		wfrm_food_obj_accuracy.append(
			_do_cross_validation(
				ds_exp2, ['treatment10', 'treatment11'], [image]))

	# now do the same, for sFRM:FOOD and sFRM:OBJ on a per-image basis
	sfrm_food_obj_accuracy = []
	for image in TEST_IMAGES:
		sfrm_food_obj_accuracy.append(
			_do_cross_validation(
				ds_exp2, ['treatment12', 'treatment13'], [image]))

	# now test the distinguishability of the treatment pairs based on all
	# images
	aggregates = {}

	aggregates['img_food_cult'] = _do_cross_validation(
		ds_exp1, ['treatment0','treatment1'], TEST_IMAGES)
	aggregates['wfrm_food_cult'] = _do_cross_validation(
		ds_exp1, ['treatment3','treatment5'], TEST_IMAGES)

	# TODO: use all the data from img_food_obj, not just one treatment pair
	aggregates['img_food_obj'] = []
	for idx in range(5):
		aggregates['img_food_obj'].append(
			_do_cross_validation(
				ds_exp2, ['treatment%d'%idx,'treatment%d'%(idx+5)],
				TEST_IMAGES))

	aggregates['wfrm_food_obj']  = _do_cross_validation(
		ds_exp2, ['treatment10','treatment11'], TEST_IMAGES)
	aggregates['sfrm_food_obj']  = _do_cross_validation(
		ds_exp2, ['treatment12','treatment13'], TEST_IMAGES)

	# gather the results
	results = {
		'aggregates': aggregates,
		'img_food_cult': food_cult_accuracy,
		'img_food_obj': img_food_obj_accuracy,
		'wfrm_food_obj': wfrm_food_obj_accuracy,
		'sfrm_food_obj': sfrm_food_obj_accuracy,
		'wfrm_food_cult': wfrm_food_cult
	}

	# write results to file
	output_fh.write(json.dumps(results, indent=2))

	# display results for debugging only
	print json.dumps(results, indent=2)

	# also return the results
	return results


def _do_cross_validation(clean_dataset, treatments, images, use_pos=True):

	print 'doing cross-validation:', treatments, images

	nb_dataset = dp.clean_dataset_adaptor(
		clean_dataset, treatments, images)
	cross_validator = naive_bayes.NaiveBayesCrossValidationTester(nb_dataset)
	overall_accuracy = cross_validator.cross_validate()

	return overall_accuracy




def theta_NB_significance(n,k_star):
	significance = 0

	for k in range(k_star, n+1):
		significance += util.choose(n,k) * (0.5**n)
	
	print significance


def get_theta_star(n, alpha):
	k_star = get_k_star(n,alpha)
	return 2 * k_star / float(n) - 1


def get_k_star(n, alpha):

	k = n
	prob_tot = prob_k_successes(n,k)
	k_star = None

	while prob_tot < alpha:
		k_star = k
		k -= 1
		prob_tot += prob_k_successes(n,k)

	return k_star


def prob_k_successes(n,k):
	return util.choose(n,k) * (0.5**n)




class Analyzer(object):

	# Constants
	ONTOLOGY_FILE = 'ontology/ontology.json'

	def __init__(self, dataset=None, ontology=None):
		if dataset is None:
			self.readDataset()
		else:
			self.dataSet = dataset

		if ontology is None:
			self.readOntology()
		else:
			self.ontology = ontology


	def compare_image_sets(self, fname='data/similarity_new.txt'):

		print 'yo'

		fh = open(fname, 'w')
		label_sets = {}
		similarities = {}

		# first, collect the bag of labels for the initial image sets

		# The label specs indicate where to collect the labels for each 
		# image set.  Each label spec has the form:
		# ('image-set-name', [ <list of treatments> ], [ <list of images>])
		label_specs = [
			('ambiguous', ['treatment0'], ['prime%d' % i for i in range(5)]),
			('cultural', ['treatment1'], ['prime%d' % i for i in range(5)]),
			('ingredients', ['treatment2'], ['prime%d' % i for i in range(5)]),
			('test', ['treatment%d' % i for i in range(3)], 
				['test%d' % i for i in range(5)])
		]

		# Get the bag of labels for each image set specified in label_specs
		for set_name, treatments, images in label_specs:
			print set_name, treatments, images
			label_sets[set_name] = self.get_bag_of_labels(
				treatments=treatments, images=images)

		# Calculate the number of unique labels, and the similarity between 
		# the bag of labels for each image_set
		for set_name_i, treatments_i, images_i in label_specs:

			# For each image set, record the number of unique labels
			similarities[set_name_i] = {'size': len(label_sets[set_name_i])}

			# In comparison with each other set, find the fraction of overlap
			for set_name_j, treatments_j, images_j in label_specs:
				union = len(label_sets[set_name_i] | label_sets[set_name_j])
				intersection = len(
					label_sets[set_name_i] & label_sets[set_name_j])
				similarities[set_name_i][set_name_j] = (
					intersection / float(union))

		print similarities

		fh.write(json.dumps(similarities, indent=4))
		fh.close()




#		ambiguous_labels = self.get_bag_of_labels(
#			treatments=['treatment0'],
#			images=['prime%d' %i for i in range(5)])
#
#		fh.write('ambg: %d\n' % len(ambiguous_labels))
#
#		cultural_labels = self.get_bag_of_labels(
#			treatments=['treatment1'],
#			images=['prime%d' %i for i in range(5)])
#
#		fh.write('cult: %d\n' % len(cultural_labels))
#
#		ingredients_labels = self.get_bag_of_labels(
#			treatments=['treatment2'],
#			images=['prime%d' %i for i in range(5)])
#
#		fh.write('ingr: %d\n' % len(ingredients_labels))
#
#		test_labels = self.get_bag_of_labels(
#			treatments=['treatment0', 'treatment1', 'treatment2'],
#			images=['test%d' %i for i in range(5)])
#
#		fh.write('test: %d\n' % len(test_labels))
#
#		# print the size of intersection for each
#		ambg_intersection = len(ambiguous_labels & test_labels)
#		ambg_union = len(ambiguous_labels | test_labels)
#		ambg_jacc = ambg_intersection / float(ambg_union) * 100
#
#		fh.write('ambg & test: %d / %d = %2.2f\n' % (
#			ambg_intersection, ambg_union, ambg_jacc))
#
#		cult_intersection = len(cultural_labels & test_labels)
#		cult_union = len(cultural_labels | test_labels)
#		cult_jacc = cult_intersection / float(cult_union) * 100
#
#		fh.write('cult & test: %d / %d = %2.2f\n' % (
#			cult_intersection, cult_union, cult_jacc))
#
#		ingr_intersection = len(ingredients_labels & test_labels)
#		ingr_union = len(ingredients_labels | test_labels)
#		ingr_jacc = ingr_intersection / float(ingr_union) * 100
#
#		fh.write('ingr & test: %d / %d = %2.2f\n' % (
#			ingr_intersection, ingr_union, ingr_jacc))
#
#		fh.close()


	def get_bag_of_labels(
		self,
		treatments=['treatment0', 'treatment3', 'treatment5'],
		images=['prime%d' %i for i in range(5)]
		):

		label_set = set()

		for treatment in treatments:
			for entry in self.dataSet.entries[treatment]:
				for image_id, label in entry.items():

					# image labels have keys that are tuples
					# but other entry properties exist
					if not isinstance(image_id, tuple):
						continue;

					# only include the labels specified by the images param
					if not image_id[0] in images:
						continue

					label_set.add(label)

		return label_set




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


	def compareFoodSpecificity(
		self, treatment1, treatment2, numToCompare=50, images=['test0']):

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
		self, treatment1, treatment2, numToCompare=50, images=['test0']):

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
			treatment (str): key for identifying a treatment, e.g.
				'treatment0'.

			image (str): key identifying an image, e.g. 'test0'.

		Output
			(float): percentage of terms associated to the treatment (and 
				image if specified) that are cultural according to the 
				ontology.
		'''

		# counters
		overall_counts = {'cultural':0, 'food':0, 'both':0, 'overall':0}
		percentages = {
			'cultural': [], 'excessCultural': [], 
			'food': [], 'both': []
		}

		for treatment in treatments:
			util.writeNow('\n\ttreatment: %s' % str(treatment))
			entries = self.dataSet.entries[treatment]
			for entry in entries:
				util.writeNow('.')

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

					overall_counts['overall'] += 1
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
					100 * entryCounts['excessCultural']/float(entryCounts['overall']))

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
		stdPercentage = {
			'cultural': (
				np.std(percentages['cultural']) / np.sqrt(
				overall_counts['overall']))

			, 'excessCultural': (
				np.std(percentages['excessCultural']) / np.sqrt(
				overall_counts['overall']))

			, 'food': (
				np.std(percentages['food']) / np.sqrt(
				overall_counts['overall']))

			, 'both': (
				np.std(percentages['both']) / np.sqrt(
				overall_counts['overall']))
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
		firstMoreMinusLess = []
		uncomparableCounts = []

		util.writeNow('         carrying out comparison ' + str(images))

		# Make all pairwise comparisons between the workers subsampled from 
		# in treatment 1 and those subsampled from treatment 2
		for i in range(len(entries1)):

			entry1 = entries1[i]

			util.writeNow('.')

			# Counters that characterize i's comparison to all j entries
			subRelativeSpecificities = []
			subUncomparableCounts = []

			for j in range(len(entries2)):

				entry2 = entries2[j]

				# Now compare the two workers.  All pairs of words are 
				# tried. Words can only be compared if one is the ancestor of
				# the other
				lessSpec, moreSpec, uncomp = self.compareEntries(
					entry1, entry2, images)

				# Record results for this comparison
				subRelativeSpecificities.append(moreSpec - lessSpec)
				subUncomparableCounts.append(uncomp)

			# Compute and record results for all comparisons of ith worker
			uncomparableCounts.append(subUncomparableCounts)
			firstMoreMinusLess.append(subRelativeSpecificities)

		# calculate the covariance of comparisons involving the same jth
		# worker
		cov_i_estimates = []
		num_pairs = len(firstMoreMinusLess)/2 * 2 # guaranteed to be even
		for i in range(0, num_pairs, 2):
			avg_1 = np.mean(firstMoreMinusLess[i])
			avg_2 = np.mean(firstMoreMinusLess[i+1])
			cov_sum = 0
			for j in range(len(firstMoreMinusLess[0])):
				A = firstMoreMinusLess[i][j] - avg_1
				B = firstMoreMinusLess[i+1][j] - avg_2
				cov_sum += A*B

			cov_i_estimate = cov_sum / float(len(firstMoreMinusLess[i]) - 1)
			cov_i_estimates.append(cov_i_estimate)

		cov_i = np.mean(cov_i_estimates)
		print 'cov_i', cov_i

		# calculate the covariance of comparisons involving the same ith
		# worker
		cov_j_estimates = []
		num_pairs = len(firstMoreMinusLess[0])/2*2 # guaranteed to be even
		for j in range(0, num_pairs, 2):

			avg_1 = np.mean([f[j] for f in firstMoreMinusLess])
			avg_2 = np.mean([f[j+1] for f in firstMoreMinusLess])

			cov_sum = 0
			for i in range(len(firstMoreMinusLess)):
				A = firstMoreMinusLess[i][j] - avg_1
				B = firstMoreMinusLess[i][j+1] - avg_2
				cov_sum += A*B

			cov_j_estimate = cov_sum / float(len(firstMoreMinusLess)-1)
			cov_j_estimates.append(cov_j_estimate)

		cov_j = np.mean(cov_j_estimates)
		print 'cov_j', cov_j

		# calculate the overall straight variance
		all_comparisons = []
		for comparison_column in firstMoreMinusLess:
			all_comparisons.extend(comparison_column)

		straight_var = np.std(all_comparisons)**2

		N = float(len(firstMoreMinusLess))
		M = float(len(firstMoreMinusLess[0]))
		true_var = (straight_var + (N-1)*cov_i + (M-1)*cov_j) /(N*M)
		old_var = np.std([np.mean(i) for i in firstMoreMinusLess])**2

		print 'straight_var', straight_var
		print 'true_var', true_var
		print 'old_var', old_var


		print ''
		return {
			'avgMoreMinusLess': 
				np.mean([np.mean(i) for i in firstMoreMinusLess]),
			'stdMoreMinusLess': np.sqrt(true_var),

			'avgUncomparable': 
				np.mean([np.mean(i) for i in uncomparableCounts]),
			'stdUncomparable': (np.std(uncomparableCounts)
				/ np.sqrt(numToCompare)),
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
		self.dataSet = dp.readDataset()




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
		self.dataSet = dp.readDataset()


	def testNBC(self, testSetSize=25,
		treatments=['treatment1', 'treatment2'], 
		images=['test0'], pDoConsiderPosition=True, show=False):
		'''
		Builds and trains a naive bayes classifier that classifies instances
		into one of the categories keyed by the input treatments.  Tests
		the resulting classifier on a subset of test instances (which were 
		excluded from training), and returns performance metrics.

		Inputs:
			- testSetSize (int): The size of test set that will be
				randomly sampled from all instances found in the dataset for 
				in each treatment served.  The instances left out during random
				sampling are used for training

			- treatments (str): A key used to identify a set of instances
				in the dataset that belong to a particular class.

		Output:
			- (classifer_result): See the class definition.  An object which
				summarizes the performance of the classifier in a test.  This
				object is understood by certain data visualization methods.
		'''

		# Temporary to catch if I am using a different training set
		# size somewhere

		# Wrap the dataset to make it understandable by the naive bayes 
		# classifier
		naiveBayesDataset = naive_bayes.NBDataset(
			self.dataSet, treatments, images, pDoConsiderPosition,
			testSetSize)

		# Make and train a naive bayes classifier
		classifier = naive_bayes.NaiveBayesClassifier()
		classifier.buildFromTrainingSet(naiveBayesDataset)

		# Get test instances from the dataset
		testSet = naiveBayesDataset.getTestSet()

		# Initialize a data structure to store classifier performance results
		r = ClassifierPerformanceResult(treatments)

		# Test the classifier on the test instances to acertain its competence
		for treatment, instances in testSet.items():

			# Test classification of instances from this treatment
			for instance in instances:

				r.results[treatment]['numTested'] += 1

				verdict = classifier.classify(instance)
				actual = instance.reveal()

				r.results[verdict]['numIdentified'] += 1

				if verdict == actual:
					r.results[treatment]['numCorrect'] += 1

		return r

	def crossValidateNBC(self, 
		testSetSize=50,
		treatments=('treatment0', 'treatment1'),
		images=['test0'],
		pDoConsiderPosition=True):

		'''
		Tests the naive bayes classifier using cross validation.  Given
		the testSetSize, the set of labeled entries in the dataset will be
		randomly partitioned into a test set and a training set.  After
		training on the training set, the classifier is tested on the test
		set, and statistics about its performance (accuracy, recall, precision,
		F1) are collected.

		This is then repeated, possibly several times, using different 
		entries for the test set.  Test sets from different iterations are
		guaranteed to be mutually exclusive. As many rounds ("folds") as 
		possible are performed, given the dataset size and test set size.
		'''

		# Wrap the dataset to make it understandable by the naive bayes 
		# classifier
		naiveBayesDataset = naive_bayes.NBDataset(
			self.dataSet, treatments, images, pDoConsiderPosition,
			testSetSize)

		# Initialize a data structure to store classifier performance results
		r = ClassifierPerformanceResult(treatments)

		foldNumber = 1
		while True:
			# This loop executes as long as cross-fold validation process has 
			# not performed all folds
			foldNumber += 1

			# Make and train a naive bayes classifier
			classifier = naive_bayes.NaiveBayesClassifier()
			classifier.buildFromTrainingSet(naiveBayesDataset)

			# Get test instances from the dataset
			testSet = naiveBayesDataset.getTestSet()

			# Test the classifier on the test instances to acertain its 
			# competence
			for treatment, instances in testSet.items():

				# Test classification of instances from this treatment
				for instance in instances:

					r.results[treatment]['numTested'] += 1

					verdict = classifier.classify(instance)
					actual = instance.reveal()

					r.results[verdict]['numIdentified'] += 1

					if verdict == actual:
						r.results[treatment]['numCorrect'] += 1

			# Keep doing tests as long as another fold is available
			try:
				naiveBayesDataset.rotateSubsample()
			except dp.CleanDatasetRotationException:
				break
				
		return r


	def crossComparison(
		self,
		comparisons,
		testSetSize=50,
		images=['test%d' % i for i in range(1)] 
		):
		'''
		Run the classifier to do pairwise comparisons of various sorts
		'''

		util.writeNow('\nPerforming cross-comparison') 

		dataSetSize = len(self.dataSet.entries.values()[0])
		trainingSetSize = dataSetSize - testSetSize
		numFolds = dataSetSize / testSetSize

		# prepare a dataset to hold the classifier performance info
		f1scores = {}
		accuracies = {}
		results = {}

		# do cross-fold validation 
		# this simultaneously tests several binary classifiers, one for each
		# tuple of treatments in `comparisons'
		first = True
		for comparison in comparisons:

			util.writeNow('.')
			firstTreatment, secondTreatment = comparison
			result = self.crossValidateNBC(
				testSetSize, comparison, images, True)

			f1scores[comparison] = result.getF1(firstTreatment)
			accuracies[comparison] = result.getOverallAccuracy()

		for comparison in comparisons:
			results[comparison] = {
				'f1': f1scores[comparison],
				'accuracy': accuracies[comparison]
			}

		return results


	def longitudinal(self, treatments, testSetSize=50):
		'''
		Run the classifier to do pairwise classification between the cultural
		images and ingredients images priming, using only one picture as the
		source of features, and repeating for each picture.  
		For each classification, do it with and without keeping track of the 
		position of words, plotting these options next to one another.
		'''

		util.writeNow(
			'\nAnylisis of classification competance as a function of image')

		results = {
			'f1': [],
			'accuracy': []
		}

		# We'll run the classifier on various images.  First we run it on 
		# the set of all images, and then on each image on its own
		allImageSet = ['test%d' % i for i in range(5)]
		eachImageSet = [['test%d' % i] for i in range(5)]
		imageSets = [allImageSet] + eachImageSet

		firstTreatment, secondTreatment = treatments

		dataSetSize = len(self.dataSet.entries.values()[0])
		trainingSetSize = dataSetSize - testSetSize
		numFolds = dataSetSize / testSetSize

		for imageSet in imageSets:	
			util.writeNow('.')

			# Accumulate results accross replicates for this imageSet
			thisImageSetF1Scores = []
			thisImageSetAccuracies = []

			# Run and test classification
			result = self.crossValidateNBC(
				testSetSize, treatments, imageSet, True)

			# record average and standard deviation for classifier F1 score
			results['f1'].append(result.getF1(firstTreatment))
			results['accuracy'].append(result.getOverallAccuracy())

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



