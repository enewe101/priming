import sys
from analysis import TryEverything
from sim_sweep import sim_sweep

SCHEDULE = {
	'classifier': ['svm'],
	'show_token_pos': [True, False],
	'do_split': [True, False],
	'remove_stops': [True, False],
	'lemmatize': [True, False],
	'spellcheck': [True, False],
}

FNAME = 'try_everything.json'


def try_everything(num_procs=None):
	if num_procs is not None:
		print 'using %d processes.' % num_procs
	else:
		print 'using all available processors.'

	sim_sweep(
		TryEverything,
		FNAME,
		SCHEDULE,
		num_procs=num_procs
	)


if __name__ == '__main__':
	if len(sys.argv)>1:
		num_procs = int(sys.argv[1])
	else:
		num_procs = None

	try_everything(num_procs)

