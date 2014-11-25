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
import csv

CORRECTIONS_PATH = 'data/new_data/dictionaries/with_allrecipes/'
DICTIONARY_FNAMES = ['dictionary_1.json', 'dictionary_2.json']

# Number of stardard deviations equivalent to the % condifdence for a 
# normal variate
CONFIDENCE_95 = 1.96
CONFIDENCE_99 = 2.975
TEST_IMAGES = ['test%d'%i for i in range(5)]

# note that our chronologically first experiment is presented as 
# experiment 2 in the science paper for expository flow
EXP1_TREATMENTS = {
	'task2:food': [0],
	'task2:cult': [1],
	#'1_img_ingr': [2],
	'frame2:food': [3],
	'frame2:cult': [5],
}

# note that our chronologically second experiment is presented as 
# experiment 1 in the science paper for expository flow
EXP2_TREATMENTS = {
	'task1:food': range(5),
	'task1:obj': range(5,10),
	'frame1:food': [10],
	'frame1:obj': [11],
	'echo:food': [12],
	'echo:obj': [13],
}


PAIRS = [
	[
		('task2:food', 'task2:cult'),
		('frame2:food', 'frame2:cult')
	], [ 
		('task1:food', 'task1:obj'),
		('frame1:food', 'frame1:obj'),
		('echo:food', 'echo:obj')
	], 
]

COMPARISONS = [
	{'name':'exp1.task', 'experiment':1, 'treatments': [0,1]},
	{'name':'exp1.frame', 'experiment':1, 'treatments': [3,5]},
	{'name':'exp2.task', 'experiment':2, 'treatments': [0,5]},
	{'name':'exp2.frame', 'experiment':2, 'treatments': [10,11]},
	{'name':'exp2.*', 'experiment':2, 'treatments': [12,13]}
]

class AnalysisError(Exception):
	pass


def assess_raters(
		read_fname='data/new_data/food_coding_test/all_raters.csv',
		write_fname='data/new_data/food_coding_test/rater_assessment.json'
	):
	'''
		Calculate interrater reliability, and accuracy for the food coding
		and spelling correction.
	'''

	rater_test_fh = open(read_fname)
	rater_test = csv.reader(rater_test_fh)
	rater_assessment_fh = open(write_fname, 'w')

	# skip the line containing column headings
	rater_test.next()

	num_terms = 0
	num_food_terms = 0	# based on majority human-coder consensus
	num_misspelled = 0
	num_machine_corrections = 0
	num_machine_should_not_have_corrected = 0
	num_correct_machine_corrections = 0
	TP = 0				# True positives for machine classification as food
	FP = 0				# False positives
	TN = 0				# True negatives
	FN = 0				# False negatives
	num_correct_machine_food_ratings = 0	# machine tries to detect food
	num_correct_machine_interp = 0			# machine recognizes words and 
											# tries to interpret misspellings
	for row in rater_test:

		# parse the row
		term, machine_correction = row[:2]
		human_food_ratings = [t.strip() for t in row[2:5]]
		machine_food_rating = row[5]
		human_corrections = row[6:]

		# is the majority rating food, and does the machine coding match?
		num_positive_human_ratings = sum(
			[int(fr.strip()=='y') for fr in human_food_ratings])
		num_terms += 1
		if num_positive_human_ratings > 1:
			num_food_terms += 1
			if machine_food_rating == 'y':
				TP += 1
				num_correct_machine_food_ratings += 1
			else:
				FN += 1

		else:
			if machine_food_rating == '':
				TN += 1
				num_correct_machine_food_ratings += 1
			else:
				FP += 1

		# get the term that the machine used
		machine_accepted = (machine_correction or term).strip()
		human_accepted = [hc.strip() for hc in human_corrections]

		# if the machine made a correction
		if machine_correction.strip():
			# but the humans think it was okay
			if sum([hc == term for hc in human_accepted])>1:
				num_machine_should_not_have_corrected += 1
			# and the humans agree it needs correcting
			else:
				num_misspelled += 1

		# the machine did not make a correction...
		else:
			# but did the humans?
			if sum([bool(hc) for hc in human_accepted])>2:
				num_misspelled += 1

		# are most human corrections different from the machine_accepted one?
		num_diff = sum([
			(bool(hc) and hc != machine_accepted) 
			for hc in human_corrections
		])

		if machine_correction.strip():
			num_machine_corrections += 1
			if num_diff < 2:
				num_correct_machine_corrections += 1

		if num_diff < 2:
			num_correct_machine_interp += 1

	# now extract the just the food-codings and convert to a 0-1 matrix
	rater_test_fh.seek(0)	# rewind the csv file
	rater_test.next()		# skip the first row containing column headings
	food_ratings = [row[2:6] for row in rater_test]
	food_ratings = [
		[int(bool(a.strip())) for a in row] for row in food_ratings
	]

	# compute inter-rater reliability using that
	reliability = compute_reliability(food_ratings)

	net_machine_fixes = (num_correct_machine_corrections 
			- num_machine_should_not_have_corrected)

	recall = TP / float(TP + FN)
	precision = TP / float(TP + FP)

	results = {
		'num_food_terms': num_food_terms,
		'fract_misspelled': num_misspelled / float(num_terms),
		'fract_misspelled_after_correction': (num_misspelled - 
			net_machine_fixes)/(float(num_terms)),
		'num_misspelled': num_misspelled,
		'num_machine_corrections': num_machine_corrections,
		'num_correct_machine_corrections': num_correct_machine_corrections,
		'net_machine_fixes': net_machine_fixes,
		'fract_correct_machine_corrections': (num_correct_machine_corrections 
			/ float(num_machine_corrections)),
		'num_machine_should_not_have_corrected': (
			num_machine_should_not_have_corrected),
		'fract_net_machine_fixes': net_machine_fixes / float(num_misspelled),
		'num_terms': num_terms,
		'num_correct_machine_food_ratings': num_correct_machine_food_ratings,
		'fract_correct_machine_food_ratings':  (
			num_correct_machine_food_ratings / float(num_terms)),
		'food_coding_reliability': reliability,
		'machine_food_TP': TP,
		'machine_food_FP': FP,
		'machine_food_TN': TN,
		'machine_food_FN': FN,
		'recall': recall,
		'precision': precision,
		'F1': 2*precision*recall / float(precision + recall),
	}

	# TODO: also calculate interrater reliability
	rater_assessment_fh.write(json.dumps(results, indent=2))


def compute_reliability(rows):
	'''
		computes interrater reliability when ratings are binary and there
		are four raters.
	'''
	num_units = len(rows)
	num_coders = len(rows[0])
	values_by_units = [(num_coders - sum(row), sum(row)) for row in rows]
	values = sum([sum(row) for row in rows])
	values = (num_coders * num_units - values, values)

	expected_agreements = values[0] * values[1]
	disagreements = 0

	for row in values_by_units:
		disagreements += 1/float(num_coders) * row[0] * row[1] 

	reliability = 1 - (num_coders * num_units - 1) * (
			disagreements / float(expected_agreements))

	return reliability




def make_inter_rater_test(
		k=50,
		fname='data/new_data/rater_test.csv'
	):
	'''
		produce a csv of labels, randomly chosen from each dataset for
		both experiments, to be independantly rated by human raters.
		the csv includes the original label, and the spell-corrected one
	'''

	# open a reader so we can write out the rater test as a csv
	fn = fname
	writer = csv.writer(open(fn, 'w'))

	# randomly select 50 entries from experiment 1 and experiment 2
	for experiment in [1,2]:

		# get words on an image-by-image basis
		for image in range(5,10):

			# get the words
			entries = get_entries_to_rate(experiment,[image],k)

			#write a bunch of empty rows (an image will be placed here)
			for i in range(30):
				writer.writerow(())

			# write the words to file
			for entry in entries:
				writer.writerow(entry)

	return


def get_entries_to_rate(
		which_experiment,
		images,
		k=50,
	):
	'''
		Get random labels from the indicated experiment, in both 
		spell-corrected and raw form, for the purpose of inter-rater 
		reliability testing.
	'''

	# the experiment must be 1 or 2
	assert(which_experiment in [1,2])

	# we're going to be doing spell correction.  Get the dictionaries.
	dict_fname = os.path.join(
		CORRECTIONS_PATH, 
		DICTIONARY_FNAMES[which_experiment-1]
	)
	dictionary = json.loads(open(dict_fname).read())

	# the dictionary has separate volumes.  Amalgamate them.
	master_dict = {}
	for vol in dictionary:
		master_dict.update(vol['results'])

	# use all of the treatments in a given experiment. That's 7 and 14.
	if which_experiment == 1:
		treatments = range(7)
	elif which_experiment == 2:
		treatments = range(14)
	else:
		raise ValueError('which_experiment must be either 1 or 2.')

	# load the data for a given experiment; then randomly pull out k words
	ds = dp.SimpleDataset(
			which_experiment=which_experiment,
			show_token_pos=False,
			show_token_img=False,
			do_split=True,
			class_idxs=treatments,
			img_idxs=images,
			spellcheck=False,
			balance_classes=True
		)
	word_raffle = list(ds.vocab_list)
	selected_words = random.sample(word_raffle, k)

	# get the spelling correction for each selected word
	spell_corrected = [
		master_dict[w] if w in master_dict else w 
		for w in selected_words
	]

	# zip up the two word lists into pairs of raw and spell-corrected words
	return zip(selected_words, spell_corrected)



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


def calculate_count_differences(
		relative_difference=False,
		fname='data/new_data/delta_counts.json'
	):

	# open a file to write the results
	write_fh = open(fname, 'w')

	results = {}

	# get the treatments to which they correspond
	treatment_map = EXP1_TREATMENTS
	treatment_map.update(EXP2_TREATMENTS)

	# look at both experiments, for each we have a variety of treatment pairs
	for exp in range(len(PAIRS)):

		# we're going to look at all the usual comparison pairs
		pairs = PAIRS[exp]

		for pair in pairs:

			# we refer to a pair by the part of the name before the ':'
			generic_pair_name = pair[0].split(':')[0]

			# get the treatments for this particular pair
			treatments = [treatment_map[p] for p in pair]

			print treatments

			# get the counts for each pair
			counts1, counts2 = [
				get_word_counts(exp+1, t, [5]) 
				for t in treatments
			]

			if relative_difference:
				counts_diff = util.counter_relative_diff(counts1, counts2)
			else:
				counts_diff = util.counter_subtract(counts1,counts2)

			counts_diff = counts_diff.items()
			counts_diff.sort(None, lambda x: x[1], True)

			results[generic_pair_name] = counts_diff[:5] + counts_diff[-5:]

	# write and return the results
	write_fh.write(json.dumps(results, indent=2))
	return results


def calculate_vocabulary_sizes(
		include_food=True,
		include_nonfood=False,
		fname='data/new_data/vocabulary.json'
	):
	results_file = open(fname, 'w')

	treatments = {
		'task1:food': (2,0),
		'task1:obj': (2,5),
		'frame1:food': (2,10),
		'frame1:obj': (2,11),
		'echo:food': (2,12),
		'echo:obj': (2,13),
		'task2:food': (1,0),
		'task2:cult': (1,1),
		'frame2:food': (1,3),
		'frame2:cult': (1,5),
	}

	results = {}
	for key, (exp, treatment) in treatments.items():
		results[key] = []
		for image in range(5,10):
			word_counts = get_word_counts(exp, [treatment], [image])
			word_counts = wna.filter_word_counts(
				word_counts, include_food, include_nonfood)
			results[key].append(len(word_counts))

	results_file.write(json.dumps(results, indent=2))
	return results


def bootstrap_relative_specificity(
		(exp1,tmts1), 
		(exp2,tmts2), 
		num_bootstraps
	):

	dataset_specs = [
		{'exp': exp1, 'treatments': tmts1},
		{'exp': exp2, 'treatments': tmts2},
	]
	datasets = [[],[]]

	# load the datasets into memory
	for i, spec in enumerate(dataset_specs):
		for image in range(5,10):
			datasets[i].append(dp.SimpleDataset(
				which_experiment=spec['exp'],
				show_token_pos=False,
				show_token_img=False,
				class_idxs=spec['treatments'],
				img_idxs=[image],
				balance_classes=False
			))

	# reorganize the data
	for i in datasets:
		for j in datasets[i]:

			# pool the workers from all treatments
			datasets[i][j] = reduce(
				lambda x,y: x + datasets[i][j].data[y],
				datset_specs[i]['treatments'],
				[]
			)

		# Currently, we have the datasets sorted by image, then worker
		# Make it sorted by worker, then image
		datasets[i] = zip(*datasets[i])


	for b in range(num_bootstraps):

		# resample both datasets (the bootstrapping principle)
		resample1 = np.random.choice(entries1, 119, replace=True)
		resample2 = np.random.choice(entries1, 119, replace=True)


	# pool all the workers from ds1 and ds2
	entries1 = reduce(lambda x: x + ds1.data[y], ds1.data.keys(), [])
	entries2 = reduce(lambda x: x + ds2.data[y], ds2.data.keys(), [])

	for b in range(num_bootstraps):

		for image in range(5,10):


			# collect the words from each resampling
			tokens1 = reduce(lambda x,y: x+y['features'], resample1, [])
			tokens2 = reduce(lambda x,y: x+y['features'], resample2, [])

			# count the words
			counts1 = Counter(tokens1)
			counts2 = Counter(tokens2)

			counts_food = get_word_counts(1, [0], [image])
			counts_cult = get_word_counts(1, [1], [image])

			results['img_food_cult'].append(wna.calculate_relative_specificity(
				counts_food, counts_cult, ignore_food))

	results[comparison] = np.mean(results[comparison])
	write_fh.write(json.dumps(results, indent=2))

	return results


def calculate_all_relative_specificities(
		include_food=True,
		include_nonfood=False,
		normalize=True,
		average=False
	):

	fname= 'data/new_data/specificity.json'
	write_fh = open(fname, 'w')
	images = range(5,10)
	results = defaultdict(lambda: [])

	for image in images:

		# compare the img_priming
		counts_food = get_word_counts(2, [0], [image])
		counts_obj = get_word_counts(2, [5], [image])
		results['task1'].append(wna.calculate_relative_specificity(
			counts_food, counts_obj, include_food, include_nonfood, normalize))

		# compare the weak framing
		counts_food = get_word_counts(2, [10], [image])
		counts_obj = get_word_counts(2, [11], [image])
		results['frame1'].append(wna.calculate_relative_specificity(
			counts_food, counts_obj, include_food, include_nonfood, normalize))

		# compare strong framing
		counts_food = get_word_counts(2, [12], [image])
		counts_obj = get_word_counts(2, [13], [image])
		results['echo'].append(wna.calculate_relative_specificity(
			counts_food, counts_obj, include_food, include_nonfood, normalize))

		# compare the img_priming
		counts_food = get_word_counts(1, [0], [image])
		counts_cult = get_word_counts(1, [1], [image])
		results['task2'].append(wna.calculate_relative_specificity(
			counts_food, counts_cult, include_food, include_nonfood, normalize))

		# compare the exp1 framing 
		counts_food = get_word_counts(1, [3], [image])
		counts_cult = get_word_counts(1, [5], [image])
		results['frame2'].append(wna.calculate_relative_specificity(
			counts_food, counts_cult, include_food, include_nonfood, normalize))

	if average:
		for comparison in results:
			results[comparison] = np.mean(results[comparison])

	write_fh.write(json.dumps(results, indent=2))
	return results


def get_food_proportions():
	fname = 'data/new_data/food.json'
	write_fh = open(fname, 'w')

	food_detector = wna.WordnetFoodDetector()

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
				'fract_food': 0,
				'std': 0
			}
			this_result = result[exp_name]

			fract_food_scores = []
			for treatment in treatment_idxs:
				for entry in d.data[treatment]:
					num_tokens = len(entry['features'])
					num_food_tokens = reduce(
						lambda x,y: x+food_detector.is_food(y),
						entry['features'], 
						0
					)
					fract_food_scores.append(num_food_tokens/float(num_tokens))

			this_result['fract_food'] = np.mean(fract_food_scores)
			this_result['std'] = np.std(fract_food_scores)/np.sqrt(
				len(fract_food_scores))

			for word, count in d.vocab_counts.items():
				this_result['num_words'] += count
				if food_detector.is_food(word):
					this_result['num_food_words'] += count

	write_fh.write(json.dumps(result, indent=2))
	return result


def bound_l1(
		fname='data/new_data/l1.json',
		show_token_pos=True,
		show_plain_token=True,
		do_split=True,
		remove_stops=True,
		lemmatize=True,
		spellcheck=True,
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
	# most of the arguments are derived from those of the outer function
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

	results = {}

	# calculate naive bayes accuracy for pairwise treatment comparisons
	for c in COMPARISONS:
		name, exp, treatments = c['name'], c['experiment'], c['treatments']

		# we handle this treatment differently because of multiple replicates
		if name == 'exp2.task':
			continue

		results[name] = []
		for image in range(5,10):
			ds = make_simple_dataset(exp, treatments, [image])
			results[name].append(do_cross_val(ds))

	# calculate the same, but for experiment 2's inter-task comparison we
	# have multiple replicates under different permutations of test tasks
	results['exp2.task'] = defaultdict(lambda: [])
	for image in range(5,10):
		for pos in range(5):
			treatments = dp.get_correct_treatments(image, pos)
			ds = make_simple_dataset(2, treatments, [image])
			accuracy = do_cross_val(ds)
			results['exp2.task']['test%d' % (image-5)].append(accuracy)
	
	# calculate naive bayes accuracy for pairwise treatment comparisons
	# when the classifier sees the labels attributed to all images
	results['aggregates'] = {}
	for c in COMPARISONS:
		name, exp, treatments = c['name'], c['experiment'], c['treatments']

		# we handle this treatment differently because of multiple replicates
		if name == 'exp2.task':
			continue

		ds = make_simple_dataset(exp, treatments, range(5,10))
		results['aggregates'][name] = do_cross_val(ds)


	# calculate the same, but for experiment 2's inter-task comparison we
	# have multiple replicates under different permutations of test tasks
	results['aggregates']['exp2.task'] = []
	for idx in range(5):
		ds = make_simple_dataset(2, [idx, idx+5], range(5,10))
		results['aggregates']['exp2.task'].append(do_cross_val(ds))

	## first, do this for the old data
	#food_cult_accuracy = []
	#for image in range(5,10):
	#	ds = make_simple_dataset(1, [0,1], [image])
	#	food_cult_accuracy.append(do_cross_val(ds))

	#wfrm_food_cult = []
	#for image in range(5,10):
	#	ds = make_simple_dataset(1, [3,5], [image])
	#	wfrm_food_cult.append(do_cross_val(ds))

	# look at the distinguishability of IMG:FOOD and IMG:OBJ on a per-image
	# per-position basis
	#img_food_obj_accuracy = defaultdict(lambda: [])
	#for image in range(5,10):
	#	for pos in range(5):
	#		treatments = dp.get_correct_treatments(image, pos)
	#		ds = make_simple_dataset(2, treatments, [image])
	#		accuracy = do_cross_val(ds)
	#		img_food_obj_accuracy['test%d'%(image-5)].append(accuracy)

	#wfrm_food_obj_accuracy = []
	#for image in range(5,10):
	#	ds = make_simple_dataset(2, [10,11], [image])
	#	wfrm_food_obj_accuracy.append(do_cross_val(ds))

	## now do the same, for sFRM:FOOD and sFRM:OBJ on a per-image basis
	#sfrm_food_obj_accuracy = []
	#for image in range(5,10):
	#	ds = make_simple_dataset(2, [12,13], [image])
	#	sfrm_food_obj_accuracy.append(do_cross_val(ds))

	# now test the distinguishability of the treatment pairs based on all
	# images
	#aggregates = {}

	#ds = make_simple_dataset(1, [0,1], range(5,10))
	#aggregates['img_food_cult'] = do_cross_val(ds)

	#ds = make_simple_dataset(1, [3,5], range(5,10))
	#aggregates['wfrm_food_cult'] = do_cross_val(ds)

	## TODO: use all the data from img_food_obj, not just one treatment pair
	#aggregates['img_food_obj'] = []
	#for idx in range(5):
	#	ds = make_simple_dataset(2, [idx, idx+5], range(5,10))
	#	aggregates['img_food_obj'].append(do_cross_val(ds))

	#ds = make_simple_dataset(2, [10,11], range(5,10))
	#aggregates['wfrm_food_obj']  = do_cross_val(ds)

	#ds = make_simple_dataset(2, [12,13], range(5,10))
	#aggregates['sfrm_food_obj']  = do_cross_val(ds)

	## gather the results
	#results = {
	#	'aggregates': aggregates,
	#	'img_food_cult': food_cult_accuracy,
	#	'img_food_obj': img_food_obj_accuracy,
	#	'wfrm_food_obj': wfrm_food_obj_accuracy,
	#	'sfrm_food_obj': sfrm_food_obj_accuracy,
	#	'wfrm_food_cult': wfrm_food_cult
	#}

	# write results to file
	output_fh.write(json.dumps(results, indent=2))

	# display results for debugging only
	print json.dumps(results, indent=2)

	# also return the results
	return results


def do_cross_val(simple_dataset):
	'''
		This is a shortcut for performing cross-validation using a NaiveBayes
		classifier, based on the classes and examples defined in the 
		simple_dataset argument.
	'''
	nb_dataset = dp.simple_dataset_2_naive_bayes(simple_dataset)
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


def binomial_lower_confidence_p(n,k,alpha=0.05, tolerance=1e-6):
	'''
		Given a bernouli variable B(p), that has been sampled n times, find 
		the threshold probability p, at or below which, 95% of the time,
		we would find less than k successes
	'''

	high_p = 1
	high_prob = binom_upper_tail_prob(n,k,high_p)

	low_p = 0
	low_prob = binom_upper_tail_prob(n,k,low_p)

	cur_p = 0.5
	cur_prob = binom_upper_tail_prob(n,k,cur_p)

	while abs(high_p - low_p) > tolerance:

		# if the probability is bigger than alpha, reduce cur_p
		if cur_prob > alpha:
			high_p = cur_p
			high_prob = cur_prob

		# if the probability is smaller than alpha, reduce cur_p
		elif cur_prob < alpha:
			low_p = cur_p
			low_prob = cur_prob

		# if it's dead on, break out
		else:
			break

		# take another guess at cur_p
		cur_p = (high_p + low_p)/ 2.0
		cur_prob = binom_upper_tail_prob(n,k,cur_p)


	return cur_p

	
def binom_upper_tail_prob(n,k,p):
	'''
		What is the probability of having at least k successes for the 
		binomial variable Bin(n,p)?
	'''
	total_prob = 0
	
	for k_prime in range(k,n+1):
		total_prob += prob_k_successes(n,k_prime,p)

	return total_prob


# Test
def prob_k_successes(n,k,p=0.5):
	return util.choose(n,k) * (p**k)*(1-p)**(n-k)


#####
#		Deprecated Functions
#####


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
						bound_l1(
							fname=fname_prefix + fname,
							show_token_pos=show_token_pos,
							do_split=do_split,
							remove_stops=remove_stops,
							lemmatize=lemmatize,
							spellcheck=spellcheck
						)



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


