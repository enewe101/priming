import sys
from svm import SvmCvalTest
from sim_sweep import sim_sweep

SCHEDULE = {
	'kernel':['rbf'],
	'C':[100, 200, 500, 1000, 2000, 5000, 10000, 20000],
	'gamma':[1e-4, 2e-4, 5e-4, 1e-5, 2e-5, 5e-5, 1e-6, 2e-6, 5e-6, 1e-7, 2e-7,
		5e-7, 1e-8, 2e-8]
}

DEFAULTS = {
	'which_experiment':1,
	'show_token_pos':True,
	'show_plain_token':True,
	'weights':'tfidf',
	'class_idxs':[1,2],
	'img_idxs':range(5,10),
}

FNAME = 'test3.json'

def do_cross_validation_test_suite(num_procs=None):
	if num_procs is not None:
		print 'using %d processes.' % num_procs
	else:
		print 'using all available processors.'

	sim_sweep(SvmCvalTest, FNAME, SCHEDULE, sim_defaults=DEFAULTS, num_procs=num_procs)

if __name__ == '__main__':
	if len(sys.argv)>1:
		num_procs = int(sys.argv[1])
	else:
		num_procs = None

	do_cross_validation_test_suite(num_procs)

