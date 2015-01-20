import os
import sys
import numpy as np
from sklearn import cross_validation
from sklearn import svm
from collections import defaultdict
import data_processing as dp
import naive_bayes as nb
import random as r
import json

BEST_PARAMS_FNAME = 'best.json'
SVM_EXP2_L1_FNAME = 'exp2.l1.svm.json'
SVM_OPTIMIZATION_DIR = 'data/new_data/svm_optimization'
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
	during cross validation can lead to a particular parameter value seeming 
	better than the others when it is not.
	'''

	# open a file for writing
	write_fh = open(os.path.join(SVM_OPTIMIZATION_DIR, write_fname), 'w')

	# for exp1 and exp2, find the best parameter settings
	rescored_params = {}
	for exp in [1, 2]:

		# get a listing of the simulated annealing results for this exp
		fnames = os.popen(
			'ls %s | grep exp%d.run' % (SVM_OPTIMIZATION_DIR, exp)
		).read().split()

		# independantly rescore and rank these parameter settings
		rescored_params['exp%d' % exp] = rescore_svm_params(fnames, exp)

	write_fh.write(json.dumps(rescored_params, indent=2))
	write_fh.close()


def rescore_svm_params(fnames, exp):
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

	if exp == 1:
		class_idxs = [0,1]
	elif exp == 2:
		class_idxs = [0,5]

	test_func = get_annealing_func(exp, class_idxs, CV=40)

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
	return re_scored_params



class LognormalSimulatedAnnealer(object):

	RESULTS_PATH = SVM_OPTIMIZATION_DIR

	def __init__(self, 
		objective_func,
		ranges={'C':[1,20000],'gamma':[1e-8,1]},
		sigma=1,
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


def get_annealing_func(exp, class_idxs, reps=1, CV=20):
	'''
	wrapper for svm cross_validation on experiment 1 data.
	It's output is used as the annealing_func argument to the simulated 
	annealer, to search for optimal svm settings.
	'''

	datas = [
		dp.SimpleDataset(
			which_experiment=exp,
			show_token_pos=True,
			show_plain_token=True,
			show_token_img=True,
			do_split=True,
			class_idxs=class_idxs,
			img_idxs=[img_idx],
			spellcheck=True,
			lemmatize=True,
			remove_stops=True,
			balance_classes=True,
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
		# These arguments are passed to the SimpleDataset constructor
		which_experiment=2,
		show_token_pos=True,
		show_plain_token=True,
		show_token_img=True,
		do_split=True,
		class_idxs=[0,5],
		img_idxs=range(5,10),
		spellcheck=True,
		lemmatize=True,
		remove_stops=True,
		balance_classes=True,

		# These arguments are passed to the classifier constructor
		kernel='rbf',
		C=1,
		gamma=1e-3,

		# Number of cross-validation folds to use
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
		show_token_img=True,
		do_split=True,
		class_idxs=class_idxs,
		img_idxs=img_idxs,
		spellcheck=spellcheck,
		lemmatize=True,
		remove_stops=True,
		balance_classes=True,
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

	outputs = np.array(outputs)
	feature_vectors = np.array(feature_vectors)

	# do cross-validation of an svm classifier
	clf = svm.SVC(kernel='rbf', C=C, gamma=gamma)
	scores = cross_validation.cross_val_score(
		clf, feature_vectors, outputs, cv=CV)

	# return the average accuracy
	return np.mean(scores)


COMPARISONS = [
	{'name':'exp1.task', 'experiment':1, 'treatments': [0,1]},
	{'name':'exp1.frame', 'experiment':1, 'treatments': [3,5]},
	{'name':'exp2.task', 'experiment':2, 'treatments': [0,5]},
	{'name':'exp2.frame', 'experiment':2, 'treatments': [10,11]},
	{'name':'exp2.*', 'experiment':2, 'treatments': [12,13]}
]



def calc_priming_diff_svm(
		fname='data/new_data/bound_l1_svm/l1_temp.json',
		show_token_pos=True,
		show_plain_token=True,
		do_split=True,
		remove_stops=True,
		lemmatize=True,
		spellcheck=True,
		balance_classes=119,
	):
	'''
	Using the best svm settings (as determined on experiment 1 data) estimate
	the l1 distance between various classes for each image
	'''

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

	# open a file to write the results
	write_fh = open(fname, 'w')

	# get the best parameters for SVM
	best_params_path = os.path.join(SVM_OPTIMIZATION_DIR, BEST_PARAMS_FNAME)
	best_params = json.loads(open(best_params_path).read())
	params_exp1 = best_params['exp1'][0]['params']
	params_exp2 = best_params['exp2'][0]['params']

	results = {}

	for c in COMPARISONS:
		name, exp, treatments = c['name'], c['experiment'], c['treatments']
		if exp == 1:
			params = params_exp1
		elif exp == 2:
			params = params_exp2

		# we handle this treatment differently because of multiple replicates
		if name == 'exp2.task':
			continue

		results[name] = []
		for image in range(5,10):

			# do leave one out cross-val
			ds = make_simple_dataset(exp, treatments, [image])
			features, outputs = ds.as_vect()
			num_examples = ds.num_examples
			scores = cross_val_svc(
				features, outputs, CV=num_examples, **params)

			# keep the results
			overall_accuracy = np.mean(scores)
			results[name].append(overall_accuracy)

	# calculate the same, but for experiment 2's inter-task comparison we
	# have multiple replicates under different permutations of test tasks
	results['exp2.task'] = defaultdict(lambda: [])
	for image in range(5,10):
		for pos in range(5):
			treatments = dp.get_correct_treatments(image, pos)
			ds = make_simple_dataset(2, treatments, [image])
			features, outputs = ds.as_vect()
			num_examples = ds.num_examples
			accuracy = np.mean(cross_val_svc(
				features, outputs, CV=num_examples, **params_exp2))

			results['exp2.task']['test%d' % (image-5)].append(accuracy)
	

	# calculate naive bayes accuracy for pairwise treatment comparisons
	# when the classifier sees the labels attributed to all images
	results['aggregates'] = {}
	for c in COMPARISONS:
		name, exp, treatments = c['name'], c['experiment'], c['treatments']
		if exp == 1:
			params = params_exp1
		elif exp == 2:
			params = params_exp2

		# we handle this treatment differently because of multiple replicates
		if name == 'exp2.task':
			continue

		ds = make_simple_dataset(exp, treatments, range(5,10))
		features, outputs = ds.as_vect()
		num_examples = ds.num_examples
		results['aggregates'][name] = np.mean(cross_val_svc(
				features, outputs, CV=num_examples, **params))

	# calculate the same, but for experiment 2's inter-task comparison we
	# have multiple replicates under different permutations of test tasks
	results['aggregates']['exp2.task'] = []
	for idx in range(5):
		ds = make_simple_dataset(2, [idx, idx+5], range(5,10))
		features, outputs = ds.as_vect()
		num_examples = ds.num_examples
		results['aggregates']['exp2.task'].append(np.mean(cross_val_svc(
				features, outputs, CV=num_examples, **params_exp2)))


	write_fh.write(json.dumps(results, indent=2))
	write_fh.close()


if __name__ == '__main__':

	# get command line args
	exp = int(sys.argv[1])
	run = sys.argv[2]

	fname = 'exp%d.run%s.json' % (exp, run)

	if exp == 1:
		class_idxs = [0,1]
	elif exp == 2:
		class_idxs = [0,5]
	else:
		raise ValueError(
			'First argument, which specifies the experimental '
			'data to use, must be 1 or 0.'
		) 

	afunc = get_annealing_func(exp=exp, class_idxs=class_idxs, reps=1, CV=20)
	lsa = LognormalSimulatedAnnealer(afunc)
	lsa.walk()
	lsa.write_path(fname)


