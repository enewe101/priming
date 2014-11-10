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
		This is an adaptor between the SimpleDataset, 
		the WordnetSpellChecker, and the SimSweeper (which makes 
		multiprocessing easy).

		loads the specified portion of the dataset, and uses the 
		WordnetSpellChecker to find misspelled words and their plausible 
		correct spellings.
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
		class_idxs=treatments,
		img_idxs=images,
		balance_classes=balance
	)

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

	fname= 'data/new_data/specificity_alt.json'
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
	fname = 'data/new_data/food_alt.json'
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
				img_idxs=range(5,10),
				spellcheck=True
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


def avg_l1(dir_suffix='l1'):
	DIR = 'data/new_data/' + dir_suffix
	fnames = os.popen('ls %s | grep ^l1' % DIR).read().split()
	print 'y'
	print fnames

	for fname in fnames:
		data = json.loads(open(os.path.join(DIR,fname)).read())
		aggregates = data['aggregates']
		exp2_data = data['img_food_obj']
		exp2_data = reduce(lambda x,y: x + y[1], exp2_data.items(), [])
		print os.path.split(fname)[-1], np.mean(exp2_data)
		print aggregates
		print ''


def try_everything(use_simple=True):
	
	fname_prefix = 'data/new_data/l1_spellcorrected'

	for show_token_pos in [True, False]:
		for do_split in [True, False]:
			for remove_stops in [True, False]:
				for lemmatize in [True, False]:
					for spellcheck in [True, False]:

						# get the file name sorted out
						fname = 'l1'
						fname += '_showpos' if show_token_pos else ''
						fname += '_split' if do_split else ''
						fname += '_nostops' if remove_stops else ''
						fname += '_lem' if lemmatize else ''
						fname += '_spell' if spellcheck else ''
						fname += '.json'

						# now do it
						simple_bound_l1(
							fname=fname_prefix + fname,
							show_token_pos=show_token_pos,
							do_split=do_split,
							remove_stops=remove_stops,
							lemmatize=lemmatize,
							spellcheck=spellcheck
						)



def do_cross_val(simple_dataset):
	nb_dataset = dp.simple_dataset_2_naive_bayes(simple_dataset)
	cross_validator = naive_bayes.NaiveBayesCrossValidationTester(nb_dataset)
	overall_accuracy = cross_validator.cross_validate()
	return overall_accuracy


def simple_bound_l1(
		fname='data/new_data/l1_alt.json',
		show_token_pos=True,
		show_plain_token=True,
		do_split=True,
		remove_stops=True,
		lemmatize=True,
		spellcheck=False,
		balance_classes=119
	):
	'''
		determine a naive bayes classifier's accuracy when trying to 
		distinguish workers various experimental treatments, using cross-
		validation.  The result is a bound on the L1-distance between the
		workers' responses from these treatments.
	'''

	output_fh = open(fname, 'w')

	# shortcut for making a simple dataset with defaults as specified
	def make_simple_dataset(which_experiment, class_idxs, images):
		return dp.SimpleDataset(
			which_experiment=which_experiment,
			show_token_pos=show_token_pos,
			show_plain_token=show_plain_token,
			do_split=do_split,
			class_idxs=class_idxs,
			img_idxs=images,
			spellcheck=spellcheck,
			lemmatize=lemmatize,
			remove_stops=remove_stops,
			balance_classes=balance_classes
		)

	# shortcut for making a simple dataset that aggregates all of the food
	# and all of the object treatments, in exp2, into two metatreatments.
	def simple_dataset_aggregated_img_treatments():
		ds = dp.SimpleDataset(
			which_experiment=2,
			show_token_pos=show_token_pos,
			show_plain_token=show_plain_token,
			do_split=do_split,
			class_idxs=range(10),
			img_idxs=range(5,10),
			spellcheck=spellcheck,
			lemmatize=lemmatize,
			remove_stops=remove_stops,
			balance_classes=balance_classes
		)

		data = {
			'food': reduce(lambda x,y: x + ds.data[y], range(5), []),
			'obj': reduce(lambda x,y: x + ds.data[y], range(5,10), []),
		}

		class AggregatedSimpleDs(object):
			def __init__(self, data):
				self.data = data

		return AggregatedSimpleDs(data)


	# first, do this for the old data
	food_cult_accuracy = []
	for image in range(5,10):
		ds = make_simple_dataset(1, [0,1], [image])
		food_cult_accuracy.append(do_cross_val(ds))

	wfrm_food_cult = []
	for image in range(5,10):
		ds = make_simple_dataset(1, [3,5], [image])
		wfrm_food_cult.append(do_cross_val(ds))

	# look at the distinguishability of IMG:FOOD and IMG:OBJ on a per-image
	# per-position basis
	img_food_obj_accuracy = defaultdict(lambda: [])
	for image in range(5,10):
		for pos in range(5):
			treatments = dp.get_correct_treatments(image, pos)
			ds = make_simple_dataset(2, treatments, [image])
			accuracy = do_cross_val(ds)
			img_food_obj_accuracy['test%d'%(image-5)].append(accuracy)

	wfrm_food_obj_accuracy = []
	for image in range(5,10):
		ds = make_simple_dataset(2, [10,11], [image])
		wfrm_food_obj_accuracy.append(do_cross_val(ds))

	# now do the same, for sFRM:FOOD and sFRM:OBJ on a per-image basis
	sfrm_food_obj_accuracy = []
	for image in range(5,10):
		ds = make_simple_dataset(2, [12,13], [image])
		sfrm_food_obj_accuracy.append(do_cross_val(ds))

	# now test the distinguishability of the treatment pairs based on all
	# images
	aggregates = {}

	ds = make_simple_dataset(1, [0,1], range(5,10))
	aggregates['img_food_cult'] = do_cross_val(ds)

	ds = make_simple_dataset(1, [3,5], range(5,10))
	aggregates['wfrm_food_cult'] = do_cross_val(ds)

	# TODO: use all the data from img_food_obj, not just one treatment pair
	aggregates['img_food_obj'] = []
	for idx in range(5):
		ds = make_simple_dataset(2, [idx, idx+5], range(5,10))
		aggregates['img_food_obj'].append(do_cross_val(ds))

	ds = make_simple_dataset(2, [10,11], range(5,10))
	aggregates['wfrm_food_obj']  = do_cross_val(ds)

	ds = make_simple_dataset(2, [12,13], range(5,10))
	aggregates['sfrm_food_obj']  = do_cross_val(ds)

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

