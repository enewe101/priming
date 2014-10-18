import os
import numpy as np
from sklearn import cross_validation
from sklearn import svm
from collections import defaultdict
import data_processing as dp
import naive_bayes as nb
import random as r
import json

SVM_EXP2_L1_FNAME = 'exp2.l1.svm.json'
SVM_OPTIMIZATION_DIR = 'data/new_data/svm_optimization'
BEST_PARAMS_FNAME = 'exp1.best.json'
DATA_DIR = 'data/new_data'


def test_func(C,gamma):
	'''
	This is a function whose optimal values are known, and which serves
	as a tester to make sure that the simulated annealer can find its
	maxima properly
	'''
	# set the optimal values.  Change these to make a different test.
	optimal_gamma = 0.000021
	optimal_C = 49

	A = np.e**(-(np.log2(gamma)-np.log2(optimal_gamma))**2/100.)
	B = np.e**(-(np.log2(C)-np.log2(optimal_C))**2/100.)
	return A*B


def find_best_svm_params(write_fname=BEST_PARAMS_FNAME):
	'''
	Once the LognormalSimulatedAnnealer has been run a few times to find
	good parameters for svm classfier, we take the best candidates 
	among those and then independantly run cross validation several times
	on each, and take the best result.  
	
	This is to defend against the fact that variance in reported accuracy
	during cross validation can to a particular parameter value seeming 
	better than the others when it is not.
	'''

	# open a file where results will be written
	write_fh = open(os.path.join(SVM_OPTIMIZATION_DIR, write_fname), 'w')

	# get a listing of the simulated annealing results which will be ranked
	fnames = os.popen(
		'ls %s | grep exp1.run' % SVM_OPTIMIZATION_DIR).read().split()

	# read all simulated annealing results.  Take the best 5 from each.
	top_params = []
	for fname in fnames:

		# read one simulated annealing results file
		fh = open(os.path.join(SVM_OPTIMIZATION_DIR, fname))
		params = json.loads(fh.read())
		fh.close()

		# sort the params by accuracy, and keep the best 5
		top_params.extend(
			sorted(params, None, lambda x: x['accuracy'], True)[:5])

	# we'll now independantly re-score the top parameters

	# for better performance, use the same function that was designed for
	# testing in simulated annealing (it loads the data vectors into memory
	# only once)
	test_func = get_annealing_func(CV=40)

	# independantly re-score all the top parameters
	re_scored_params = []
	for param_record in top_params:
		print 'testing', param_record
		params = param_record['params'] 

		# take the average of three replicates
		new_score = np.mean([test_func(**params) for i in range(3)])
		print 'independantly scored:', new_score
		re_scored_params.append({'params':params, 'accuracy':new_score})

	# write these to file, in ranked order
	re_scored_params.sort(None, lambda x: x['accuracy'], True)
	write_fh.write(json.dumps(re_scored_params, indent=2))
	write_fh.close()








class LognormalSimulatedAnnealer(object):

	RESULTS_PATH = SVM_OPTIMIZATION_DIR

	def __init__(self, 
		objective_func,
		ranges={'C':[1,20000],'gamma':[1e-8,1]},
		sigma=1,
		fname='test4.json',
		num_steps=20,
		B=50,
		max_tries=100,
		temp_rise=1.1,
	):

		self.objective_func = objective_func
		self.ranges = ranges
		self.params = {}
		self.sigma=sigma
		self.num_steps = num_steps
		self.B = B
		self.path = []
		self.max_tries = max_tries
		self._record = 0.5
		self.temp_rise=temp_rise

		# start the parameters in their midpoints
		for param_name in ranges:
			min_val, max_val = ranges[param_name]
			lg_avg = ( np.log2(min_val) + np.log2(max_val) ) / 2.
			self.params[param_name] = 2**lg_avg


	def walk(self):
		# first evaluate the function at the new val
		self.old_val = self.objective_func(**self.params)

		for i in range(self.num_steps):
			self.step()

		return self.get_best_result()


	def resume(self):
		for i in range(self.num_steps):
			self.step()

		return self.get_best_result()


	def write_path(self, fname):
		write_fh = open(os.path.join(self.RESULTS_PATH, fname), 'w')
		data = [{'params':d[0], 'accuracy':d[1]} for d in self.path]
		write_fh.write(json.dumps(data, indent=2))
		write_fh.close()


	def record(self, success):
		self._record = self._record * (9/10.) + int(success)*(1/10.)

	def get_record(self):
		return self._record

	def step(self):
		step_success = False
		tries_counter = 0
		while not step_success:
			step_success = self.try_step()

			# keep a record of the frequency of successes.
			# reduce the step size if unsuccessful too often, increase
			# if successful too often
			self.record(step_success)
			if self.get_record() < 0.05:
				print 'reducing step size'
				self.sigma *= 0.8

			if self.get_record() > 0.9:
				self.B += 5

			tries_counter += 1
			if tries_counter > self.max_tries:
				print 'exceeded number of tries'
				break

		print self.B, self.old_val, self.params, self.sigma
		self.B *= self.temp_rise


	def try_step(self):

		# generate a new set of parameters
		new_params = {}
		for param_name in self.params:
			lg_old_val = np.log2(self.params[param_name])
			new_params[param_name] = 2**(lg_old_val + r.gauss(0,self.sigma))

		# try at this val
		new_val = self.objective_func(**new_params)
		self.path.append((new_params, new_val))

		# set up a test to decide whether to keep the new value
		uniform = r.uniform(0,1)
		bar = min(1, (new_val / self.old_val)**self.B)

		# reject step with high probability if it makes the value worse
		if bar < uniform:
			return False

		# keep the step
		self.old_val = new_val
		self.params = new_params
		return True


	def get_best_result(self):

		best_val = None
		for params, val in self.path:

			if best_val is None:
				best_val = val
				best_params = params

			if val > best_val:
				best_val = val
				best_params = params

		return best_val, best_params


class SvmCvalTest(object):
	'''
	A wrapper around the cross-validation of svm on experimental data.
	This makes it possible to run using my sim_sweep parallelization utility
	In the end, I found it better to do simulated annealing, for which this
	is not needed.  However, it is useful to try an exhaustive combination
	of settings for the svm classfier.
	'''

	def __init__(self):
		pass

	def run(
			self,
			which_experiment=2,
			show_token_pos=True,
			show_plain_token=True,
			class_idxs=[0,5],
			img_idxs=range(5,10),
			weights='tfidf',
			kernel='linear',
			C=1,
			gamma=1e-3
		):

		return calc_priming_svc(
			which_experiment=which_experiment,
			show_token_pos=show_token_pos,
			show_plain_token=show_plain_token,
			class_idxs=class_idxs,
			img_idxs=img_idxs,
			weights=weights,
			kernel=kernel,
			C=C,
			gamma=gamma,
			CV=20
		)


def get_annealing_func(reps=1, CV=20):
	'''
	wrapper for svm cross_validation on experiment 1 data.
	It's output is used as the annealing_func argument to the simulated 
	annealer, to search for optimal svm settings.
	'''

	datas = [
		dp.SimpleDataset(
			which_experiment=1,
			show_token_pos=True,
			show_plain_token=True,
			class_idxs=[1,2],
			img_idxs=[img_idx],
			spellcheck=False,
			get_syns=False
		).as_vect(weights='tfidf')
		for img_idx in range(5,10)
	]

	def wrap_for_annealing_separate(C=1, gamma=1e-3):
		scores = []

		for r in range(reps):
			for i in range(len(datas)):
				feature_vectors, outputs = datas[i]
				scores.append(
					cross_val_svc(feature_vectors, outputs,C,gamma,CV)
				)

		return np.mean(scores)

	return wrap_for_annealing_separate



def calc_priming_svc(
		which_experiment=2,
		show_token_pos=True,
		show_plain_token=True,
		class_idxs=[0,5],
		img_idxs=range(5,10),
		weights='tfidf',
		spellcheck=False,
		get_syns=False,
		kernel='rbf',
		C=1,
		gamma=1e-3,
		CV=20
	):
	'''
	make the prescribed data representation, and then do cross validation
	on it.  This is like `cross_val_svc`, but you don't need to first make the
	data representation and then call cross_val_svc separately
	'''
	# load up the right data
	data = dp.SimpleDataset(
		which_experiment=which_experiment,
		show_token_pos=show_token_pos,
		show_plain_token=show_plain_token,
		class_idxs=class_idxs,
		img_idxs=img_idxs,
		spellcheck=spellcheck,
		get_syns=get_syns
	)

	# extract the feature vectors
	feature_vectors, outputs = data.as_vect(weights=weights)

	# do cross-validation of an svm classifier
	scores = []
	clf = svm.SVC(kernel=kernel, C=C, gamma=gamma)
	scores.extend(cross_validation.cross_val_score(
		clf, feature_vectors, outputs, cv=CV))

	# return the average accuracy
	return np.mean(scores)


def cross_val_svc(feature_vectors, outputs, C, gamma, CV=20):
	'''
	An abbreviation for doing cross validation using an svm classifier in
	with an rbf kernel.  It returns a single accuracy result, averaging over
	all of the folds.  
	'''

	# do cross-validation of an svm classifier
	clf = svm.SVC(kernel='rbf', C=C, gamma=gamma)
	scores = cross_validation.cross_val_score(
		clf, feature_vectors, outputs, cv=CV)

	# return the average accuracy
	return np.mean(scores)


def calc_priming_diff_svm(fname=SVM_EXP2_L1_FNAME):
	'''
	Using the best svm settings (as determined on experiment 1 data) estimate
	the l1 distance between various classes for each image
	'''
	# open a file to write the results
	write_fh = open(os.path.join(DATA_DIR, fname), 'w')

	# get the best parameters for SVM
	best_params_path = os.path.join(SVM_OPTIMIZATION_DIR, BEST_PARAMS_FNAME)
	best_params = json.loads(open(best_params_path).read())[0]['params']

	l1_vals = {'intertask':{}, 'framing':{}}

	# first measure priming differences between intertask treatments
	for class_idx in range(5):

		# treatments separated by 5 have the same image permutation but
		# different initial tasks
		class_idxs = (class_idx,class_idx+5)
		print 'measuring intertask priming for treatments %d and %d.' % class_idxs
		l1_vals['intertask']['%d_%d' % class_idxs] = {'by_pos':[], 'by_idx':[]}
		cur_results = l1_vals['intertask']['%d_%d' % class_idxs]

		for img_idx in range(5,10):
			print '\tcalculating priming for image %d' % img_idx

			data = dp.SimpleDataset(
				which_experiment=2,
				class_idxs=class_idxs,
				img_idxs=[img_idx])
			features, outputs = data.as_vect()

			# Do leave-one-out CV
			num_examples = data.num_examples
			scores = cross_val_svc(
				features, outputs, CV=num_examples, **best_params)

			# keep the results
			overall_accuracy = np.mean(scores)
			print '\t\tpriming difference:', overall_accuracy
			cur_results['by_idx'].append(overall_accuracy)

		# also keep the results ordered by position (i.e. in permuted order)
		# before we can get the permuted order, we add 5 to each images
		# index, because these are the test images, and so are offset by 5
		cur_results['by_pos'] = dp.permute(cur_results['by_idx'], class_idx)

	# now measure the priming differences between framing treatments
	for class_idxs in [(10,11), (12,13)]:
		print 'measuring intertask priming for treatments %d and %d.' % class_idxs
		l1_vals['framing']['%d_%d' % class_idxs] = []
		cur_results = l1_vals['framing']['%d_%d' % class_idxs] 

		for img_idx in range(5,10):

			print '\tcalculating priming for image %d' % img_idx
			data = dp.SimpleDataset(
				which_experiment=2,
				class_idxs = class_idxs,
				img_idxs = [img_idx])
			features, outputs = data.as_vect()

			# Do leave-one-out cross-validation
			num_examples = data.num_examples
			scores = cross_val_svc(
				features, outputs, CV=num_examples, **best_params)

			# keep the results
			overall_accuracy = np.mean(scores)
			print '\t\tpriming difference:', overall_accuracy
			cur_results.append(overall_accuracy)

	write_fh.write(json.dumps(l1_vals, indent=2))
	write_fh.close()







def calc_priming_differences():
	'''
	Calculates a whole bunch of priming differences based on binary 
	classification using a naive Bayes classifier and the CleanDataset
	representation.

	This is basically obsolete because I have moved to using SVM as a
	classifier.
	'''

	OLD_DATASET = False
	NEW_DATASET = True
	NUM_FOLDS = 125
	NUM_IMAGES = 5

	# this provides a mapping that shows which treatment should be used to
	# find a given image at a given position.  The image is the key, and 
	# the positions are the array index.  Example: the image 'test0' was in
	# position 0 for treatment 0, in position 1 for treatment 4, in position
	# 3 for treatment 3 etc. (this is reading off the first line).
	treatment_offsets = {
			'test0': [0,4,3,2,1],
			'test1': [1,0,4,3,2],
			'test2': [2,1,0,4,3],
			'test3': [3,2,1,0,4],
			'test4': [4,3,2,1,0]
	}

	results = {}

	# first read the old dataset, and make a naive bayes dataset using 
	# labels from the first image as features
	old_results = {}
	print 'calculation'
	old_results['cult_vs_ambg'] = test_binary_classification(
		False, ['treatment0', 'treatment1'], ['test0'])

	print 'calculation'
	old_results['cult_vs_ingr'] = test_binary_classification(
		False, ['treatment1', 'treatment2'], ['test0'])

	results['old_results'] = old_results
	
	# next read the new dataset.  Calculate the distinguishability for
	# the two image primings, the 
	new_results = {}
	image_priming_results =  defaultdict(lambda:[])

	for image in ['test%d'%i for i in range(NUM_IMAGES)]:
		for treatment_offset in treatment_offsets[image]:
			print 'calculation'
			overall_accuracy = test_binary_classification(
				True, 
				['treatment%d'%treatment_offset, 
					'treatment%d'%(5+treatment_offset)], 
				[image]
			)
			image_priming_results[image].append(overall_accuracy)
	new_results['image_priming_results'] = image_priming_results

	results['new_results'] = new_results
	
	return results


def test_binary_classification(use_new_dataset, treatments, images):
	dataset = dp.readDataset(use_new_dataset)
	naive_bayes_dataset = dp.clean_dataset_adaptor(
		dataset, treatments=treatments, images=images)
	cross_validator = nb.NaiveBayesCrossValidationTester(
		naive_bayes_dataset)

	overall_accuracy = cross_validator.cross_validate()

	return overall_accuracy
