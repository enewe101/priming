import numpy as np
from sklearn import cross_validation
from sklearn import svm
from collections import defaultdict
import data_processing as dp
import naive_bayes as nb
import random as r


def test_func(C,gamma):
	A = np.e**(-(np.log2(gamma)-np.log2(0.000021))**2/100.)
	B = np.e**(-(np.log2(C)-np.log2(49))**2/100.)
	return A*B



class LognormalSimulatedAnnealer(object):

	def __init__(self, 
		objective_func,
		ranges={'C':[1,20000],'gamma':[1e-8,1]},
		sigma=1,
		fname='test4.json',
		num_steps=100,
		B=50,
		max_tries=100
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
		self.B += 10


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


def get_annealing_func():

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

		for i in range(len(datas)):
			feature_vectors, outputs = datas[i]
			scores.append(
				cross_val_svc(feature_vectors, outputs,C,gamma)
			)

		return np.mean(scores)

	return wrap_for_annealing_separate


def wrap_for_annealing(C=1, gamma=1e-3):
	return calc_priming_svc(
		which_experiment=1,
		show_token_pos=True,
		show_plain_token=True,
		class_idxs=[1,2],
		img_idxs=range(5,10),
		weights='tfidf',
		kernel='rbf',
		C=C,
		gamma=gamma,
		CV=20
	)


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
		for i in range(2):
			clf = svm.SVC(kernel=kernel, C=C, gamma=gamma)
			scores.extend(cross_validation.cross_val_score(
				clf, feature_vectors, outputs, cv=CV))

		# return the average accuracy
		return np.mean(scores)


def cross_val_svc(feature_vectors, outputs, C, gamma):

	# do cross-validation of an svm classifier
	clf = svm.SVC(kernel='rbf', C=C, gamma=gamma)
	scores = cross_validation.cross_val_score(
		clf, feature_vectors, outputs, cv=20)

	# return the average accuracy
	return np.mean(scores)


def calc_priming_differences():
	'''
	Based on Naive Bayes
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
