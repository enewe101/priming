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

try:
	from svm import calc_priming_diff_svm 
except ImportError:
	pass

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
from scipy.stats import chi2_contingency as chi2

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


def do_all_chi_squared_tests(
		data_fname='data/new_data/chi2.json',
		table_fname='data/new_data/chi2.tex'
	):

	data_fh = open(data_fname, 'w')
	table_fh = open(table_fname, 'w')
	comparisons = {
		'intertask-food-objects':[0,5], 
		'frame-food-objects':[10,11], 
		'echo-food-objects':[12,13], 
		'intertask-food-culture':[0,1], 
		'frame-food-culture':[3,5]
	}
	treatments = {
		'intertask-food-objects': {
			'food': (2,0),
			'objects': (2,5),
		},
		'frame-food-objects': {
			'food': (2,10),
			'objects': (2,11),
		},
		'echo-food-objects': {
			'food': (2,12),
			'objects': (2,13),
		},
		'intertask-food-culture': {
			'food': (1,0),
			'culture': (1,1),
		},
		'frame-food-culture': {
			'food': (1,3),
			'culture': (1,5),
		}
	}
	experiments = [
		['intertask-food-culture', 'frame-food-culture'], 
		['intertask-food-objects', 'frame-food-objects', 'echo-food-objects']
	]

	# Test within-treatment homogeneity 
	within_treatment_chi2_vals = {}
	table_fh.write(
		'\\begin{table}\n\\centering\n\\begin{tabular}{c c c c c}\n'
		'\\toprule\nExperiment & Treatment & Degrees of freedom & $\chi^2$ & $p$-value'
		'\\\\\n\\toprule\n'
	)
	for exp_idx in [1,0]:
		for key in experiments[exp_idx]:
			spec = treatments[key]
			table_fh.write(
				'\\noalign{\\smallskip}\n\\multirow{2}{*}{\\textit{%s}}' % key
			)
			for treatment_name, (exp, treatment) in spec.items():
				chi_val, p_val, dof  = within_treatment_chi_squared_test(
					exp, treatment)
				within_treatment_chi2_vals[treatment_name] = {
					'chi2': chi_val, 'dof':dof, 'p_val': p_val}
				p_val_str = util.as_scientific_latex(p_val, 2)
				table_fh.write(
					' & \\textit{%s} & %d & %3.1f & %s\\\\\n' % (
						treatment_name, dof, chi_val, p_val_str)
				)

	table_fh.write(
		'\\noalign{\\smallskip}\n'
		'\\bottomrule\n\\end{tabular}\n\\end{table}\n\n\n'
	)

	# Test homogeneity between the treatments of given experiments
	table_fh.write(
		'\\begin{table}\n\\centering\n\\begin{tabular}{c c c c}\n'
		'\\toprule\nExperiment & Degrees of freedom & $\chi^2$ & $p$-value'
		'\\\\\n\\toprule\n'
	)
	between_treatment_chi2_vals = {}
	for exp in [1,0]:
		for key in experiments[exp]:
			treatments = comparisons[key]
			chi_val, p_val, dof  = between_treatment_chi_squared_test(
				experiment=exp+1,
				treatment_1 = treatments[0],
				treatment_2 = treatments[1]
			)
			between_treatment_chi2_vals[key] = {
				'chi2': chi_val, 'dof':dof, 'p_val': p_val}
			p_val_str = util.as_scientific_latex(p_val, 2)
			table_fh.write(
				'\\textit{%s} & %d & %3.1f & %s\\\\\n' % (
					key, dof, chi_val, p_val_str)
			)

	table_fh.write('\\bottomrule\n\\end{tabular}\n\\end{table}\n\n\n')

	chi2_vals = {
		'within_treatment': within_treatment_chi2_vals,
		'between_treatment': between_treatment_chi2_vals,
	}

	data_fh.write(json.dumps(chi2_vals, indent=2))
	data_fh.close()
	table_fh.close()

	return chi2_vals


def within_treatment_chi_squared_test(experiment=1, treatment=0):
	'''
		Performs a chi-squared test, to see if a random division of 
		workers from a treatment have homogeneous word frequencies.
	'''
	ds = dp.SimpleDataset(
		which_experiment=experiment,
		show_token_pos=False,
		show_token_img=True,
		class_idxs=[treatment],
		img_idxs=range(5,10),
		balance_classes=119
	)
	data = [d['features'] for d in ds.data[treatment]]

	# randomly partition workers between two sets
	list1, list2 = [], []
	for d in data:
		if random.randint(0,1):
			list1.extend(d)
		else:
			list2.extend(d)

	c1 = Counter(list1) 
	c2 = Counter(list2) 

	chi_val, p_val, dof = chi_squared_test(c1, c2)
	return chi_val, p_val, dof


def between_treatment_chi_squared_test(
		experiment=1,
		treatment_1=0,
		treatment_2=1
	):
	'''
		Performs a chi-squared test, to see if the word frequencies used
		in treatment_1 are significantly different from those in treatment_2.
		Uses Pearson's chi-squared test.
	'''

	# get the word frequencies for two different treatments
	c1 = get_word_counts(experiment, [treatment_1], range(5,10), balance=119,
		show_token_img=True)
	c2 = get_word_counts(experiment, [treatment_2], range(5,10), balance=119,
		show_token_img=True)

	chi_val, p_val, dof = chi_squared_test(c1, c2)
	return chi_val, p_val, dof


	
def chi_squared_test(counts1, counts2):
	''' 
		accepts two word frequency Counters, and performs a chi-squared 
		test of homogeneity between them.  It handles the case where
		less than 5 occurences are expected (i.e. less than 10 observations 
		between both treatments), by lumping infrequent words into an OTHER
		category.
	'''
	# get the full set of all words
	vocab = set(counts1.keys() + counts2.keys())

	# to create a contingency table for the chi2 test, we flatten the counts
	# into lists.  We only include words that appear at least 10 times between
	# both treatments.  All other words are added to an "other" bin
	other_count_1, other_count_2 = 0,0
	list_1, list_2 = [],[]

	for w in vocab:
		if counts1[w] + counts2[w] < 10:
			other_count_1 += counts1[w]
			other_count_2 += counts2[w]
			del counts1[w]
			del counts2[w]

		else:
			list_1.append(counts1[w])
			list_2.append(counts2[w])

	list_1.append(other_count_1)
	list_2.append(other_count_2)

	contingency_table = np.array([list_1, list_2])

	chi_val, p_val, dof, expected = chi2(contingency_table, correction=True)

	return chi_val, p_val, dof



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


def get_word_counts(
		experiment, 
		treatments, 
		images, 
		balance=119, 
		show_token_img=False
	):
	'''
		set balance=0 to apply no truncation of the dataset (which is 
		normally done to balance the number of data points per treatment).
	'''
	d = dp.SimpleDataset(
		which_experiment=experiment,
		show_token_pos=False,
		show_token_img=show_token_img,
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


def vocabulary_null_model(
		comparison='task1',
		include_food=True,
		include_nonfood=False,
		num_bootstraps=1000,
		num_workers=119,
		images=range(5,10),
	):

	fname = 'data/new_data/vocabulary/vocabulary_null_%s.json' % comparison
	comparisons = {
		'task1': ((2,0), (2,5)),
		'frame1': ((2,10), (2,11)),
		'echo': ((2,12), (2,13)),
		'task2': ((1,0), (1,1)),
		'frame2': ((1,3), (1,5))
	}
	assert(comparison in comparisons)
	fh = open(fname, 'w')

	
	upper_CI_idx = int(np.ceil(0.975 * num_bootstraps)) - 1
	lower_CI_idx = int(np.floor(0.025 * num_bootstraps))

	# unpack the comparison specifications
	spec1, spec2 = comparisons[comparison]
	exp1, treatment1 = spec1
	exp2, treatment2 = spec2

	# get the comparison data.
	ds1 = get_worker_features(exp1, [treatment1], images, num_workers)
	ds2 = get_worker_features(exp2, [treatment2], images, num_workers)

	# mix the two datasets to create the null model
	ds = ds1 + ds2

	boot_results = []
	for b in range(num_bootstraps):
		print '%2.1f %%' % (100 * b / float(num_bootstraps))
		rel_diff = compare_vocab_size_random_partition(
			ds, num_workers, images, include_food, include_nonfood)
		boot_results.append(rel_diff)

	boot_results.sort()
	result = {
		'mean': np.mean(boot_results),
		'upper_CI': boot_results[upper_CI_idx],
		'lower_CI': boot_results[lower_CI_idx]
	}

	fh.write(json.dumps(result, indent=2))
	fh.close()
	return result





def compare_vocab_size_random_partition(
		dataset,
		num_workers,
		images,
		include_food,
		include_nonfood
	):

	result = []

	# randomly partition the dataset in two
	ds1_indices = random.sample(range(2*num_workers), num_workers)
	ds2_indices = list(set(range(2*num_workers)) - set(ds1_indices))
	split1 = [dataset[i] for i in ds1_indices]
	split2 = [dataset[i] for i in ds2_indices]

	vocab_size1, vocab_size2 = 0, 0
	for image in range(len(images)):
		vocab_size1 += get_vocab_size(
			image,split1,include_food,include_nonfood)

		vocab_size2 += get_vocab_size(
			image,split2,include_food,include_nonfood)

	diff = vocab_size1 - vocab_size2
	avg = (vocab_size1 + vocab_size2) / 2.0
	relative_percent_diff = 100 * diff / avg
	return relative_percent_diff

				

def get_vocab_size(image,split,include_food, include_nonfood):
	counts = Counter(reduce(lambda x,y: x + y[image], split, []))
	counts = wna.filter_word_counts(counts, include_food, include_nonfood)
	vocab_size = len(counts)
	return vocab_size



def get_worker_features(exp, treatments, images, num_workers):

	# load the datasets into memory
	datasets = []
	for im_no, image in enumerate(images):
		add_dataset = dp.SimpleDataset(
			which_experiment=exp,
			show_token_pos=False,
			show_token_img=False,
			class_idxs=treatments,
			img_idxs=[image],
			balance_classes=num_workers
		)

		# pool the workers records from all treatments
		add_dataset = reduce(
			lambda x,treatment: x + add_dataset.data[treatment],
			treatments,
			[]
		)

		# Now the records for workers from different treatments were 
		# pooled.  Next, extract the labels ('features') from those 
		# records
		# extract just the labels given by workers from each record
		add_dataset = [w['features'] for w in add_dataset]

		datasets.append(add_dataset)

	# The desired data was extracted.  The data are organized first by
	# image, then by worker.  Instead, organize by worker, then image
	datasets = zip(*datasets)

	return datasets


def bootstrap_relative_specificity(
		(exp1,tmts1), 
		(exp2,tmts2), 
		images=[5],
		num_bootstraps=1000,
		resample_size=119,
		fname='data/new_data/specificity_bootstrap.json'
	):

	write_fh = open(fname, 'w')

	dataset_specs = [
		{'exp': exp1, 'treatments': tmts1},
		{'exp': exp2, 'treatments': tmts2},
	]
	datasets = [[],[]]

	# load the datasets into memory
	for i, spec in enumerate(dataset_specs):
		for image in images:
			datasets[i].append(dp.SimpleDataset(
				which_experiment=spec['exp'],
				show_token_pos=False,
				show_token_img=False,
				class_idxs=spec['treatments'],
				img_idxs=[image],
				balance_classes=resample_size
			))

	# reorganize the data.
	#
	# Currently, separate datasets for each image, within which we have
	# separate workers.  Make the worker the dominant organizing division.
	# This enables us to do worker-based resampling, which is important for
	# the integrity of the bootstrap, since each worker is an HPU, and we
	# are measuring the distribution over HPU characteristics.
	for ds in range(len(datasets)):
		for image in range(len(images)):

			# pool the workers records from all treatments
			datasets[ds][image] = reduce(
				lambda x,treatment: x + datasets[ds][image].data[treatment],
				dataset_specs[ds]['treatments'],
				[]
			)

			# Now the records for workers from different treatments were 
			# pooled.  Next, extract the labels ('features') from those 
			# records
			# extract just the labels given by workers from each record
			datasets[ds][image] = [w['features'] for w in datasets[ds][image]]

		# The desired data was extracted.  The data are organized first by
		# image, then by worker.  Instead, organize by worker, then image
		datasets[ds] = zip(*datasets[ds])

	boot_results = []
	for b in range(num_bootstraps):

		# show progress
		if b % 10 == 0:
			print '%2.1f %%' % (100 * b / float(num_bootstraps))

		# resample workers from both datasets (the bootstrapping principle)
		resample1= [
			random.choice(datasets[0]) 
			for i in range(resample_size*min(len(tmts1),len(tmts2)))
		]
		resample2= [
			random.choice(datasets[1]) 
			for i in range(resample_size*min(len(tmts1), len(tmts2)))
		]

		# Debug: try doing sampling w/o replacement -- should give the non
		# bootstrap result for each bootstrap
		#resample1 = random.sample(datasets[0], resample_size)
		#resample2 = random.sample(datasets[1], resample_size)

		this_boot_results = []
		for image in range(len(images)):

			# extract the features related to this image
			tokens1 = reduce(lambda x,y: x + y[image], resample1, [])
			tokens2 = reduce(lambda x,y: x + y[image], resample2, [])

			# convert the token lists to counts
			counts1 = Counter(tokens1)
			counts2 = Counter(tokens2)

			# calculate the relative specificity for this image
			this_boot_results.append(
				wna.calculate_relative_specificity(counts1, counts2)
			)

		boot_results.append(np.mean(this_boot_results))

	# Find the 95% confidence interval using the 5th and 95th percentile
	boot_results.sort()
	idx_percentile_2 = int(np.floor(0.025*num_bootstraps))
	idx_percentile_98 = int(np.ceil(0.975*num_bootstraps)) - 1
	mean = np.mean(boot_results)
	stderr = np.std(boot_results)
	upper_CI = boot_results[idx_percentile_98]
	lower_CI = boot_results[idx_percentile_2]

	results = {
		'mean': mean,
		'std': stderr,
		'upper_CI': upper_CI,
		'lower_CI': lower_CI
	}

	write_fh.write(json.dumps(results, indent=2))

	return results


def calculate_all_relative_specificities(
		include_food=True,
		include_nonfood=False,
		normalize=True,
		average=False,
		fname='data/new_data/specificity.json'
	):

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


def try_everything(
		classifier='naive_bayes',	# 'svm'
	):
	
	if classifier == 'naive_bayes':
		fname_prefix = 'data/new_data/bound_l1_naive_bayes/'

	elif classifier == 'svm':
		fname_prefix = 'data/new_data/bound_l1_svm/'

	else:
		raise ValueError('classifier must be either `naive_bayes` or `svm`.')

	i = 0
	for show_token_pos in [True, False]:
		for do_split in [True, False]:
			for remove_stops in [True, False]:
				for lemmatize in [True, False]:
					for spellcheck in [True, False]:

						print '%2.1f %%' % (100 * i / float(2**5))

						# get the file name sorted out
						fname = 'l1'
						fname += '_showpos' if show_token_pos else ''
						fname += '_split' if do_split else ''
						fname += '_nostops' if remove_stops else ''
						fname += '_lem' if lemmatize else ''
						fname += '_spell' if spellcheck else ''
						fname += '.json'

						# now do it
						if classifier == 'naive_bayes':
							bound_l1(
								fname=fname_prefix + fname,
								show_token_pos=show_token_pos,
								do_split=do_split,
								remove_stops=remove_stops,
								lemmatize=lemmatize,
								spellcheck=spellcheck
							)

						elif classifier == 'svm':
							calc_priming_diff_svm(
								fname=fname_prefix + fname,
								show_token_pos=show_token_pos,
								do_split=do_split,
								remove_stops=remove_stops,
								lemmatize=lemmatize,
								spellcheck=spellcheck
							)

						else:
							raise ValueError(
								'classifier must be either `naive_bayes` or '
								'`svm`.'
							)

						i += 1


class TryEverything(object):
	'''
		This makes it possible to run the classifiers at all settings
		using the simsweep paralellizer
	'''
	def __init__(self):
		pass

	def run(
		self,
		classifier='naive_bayes',
		show_token_pos=True,
		do_split=True,
		remove_stops=True,
		lemmatize=True,
		spellcheck=True
	):

		if classifier == 'naive_bayes':
			fname = 'data/new_data/bound_l1_naive_bayes/'

		elif classifier == 'svm':
			fname = 'data/new_data/bound_l1_svm/'

		else:
			raise ValueError(
				'classifier must be either `naive_bayes` or `svm`.')

		# get the file name sorted out
		fname += 'l1'
		fname += '_showpos' if show_token_pos else ''
		fname += '_split' if do_split else ''
		fname += '_nostops' if remove_stops else ''
		fname += '_lem' if lemmatize else ''
		fname += '_spell' if spellcheck else ''
		fname += '.json'

		# now do it
		if classifier == 'naive_bayes':
			bound_l1(
				fname=fname,
				show_token_pos=show_token_pos,
				do_split=do_split,
				remove_stops=remove_stops,
				lemmatize=lemmatize,
				spellcheck=spellcheck
			)

		elif classifier == 'svm':
			calc_priming_diff_svm(
				fname=fname,
				show_token_pos=show_token_pos,
				do_split=do_split,
				remove_stops=remove_stops,
				lemmatize=lemmatize,
				spellcheck=spellcheck
			)

		else:
			raise ValueError(
				'classifier must be either `naive_bayes` or '
				'`svm`.'
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


