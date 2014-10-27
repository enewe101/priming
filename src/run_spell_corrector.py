import sys
import analysis as a
from sim_sweep import sim_sweep

SIM_CLASS = a.SpellingCorrector

FNAME_1 = 'data/new_data/dictionary_1.json'
FNAME_2 = 'data/new_data/dictionary_2.json'

SCHEDULE_1 = {
	'which_experiment': [1],
	'class_idxs' : [[0,1,2,3,5]],
	'img_idxs': [[5],[6],[7],[8],[9],[10]]
}

SCHEDULE_2 = {
	'which_experiment': [2],
	'class_idxs': [range(14)],
	'img_idxs': [[5],[6],[7],[8],[9],[10]]
}

RUNS = [(FNAME_1, SCHEDULE_1), (FNAME_2, SCHEDULE_2)]

def do_corrections(num_procs):
	for fname, schedule in RUNS:
		sim_sweep(SIM_CLASS, fname, schedule, num_procs=num_procs)


if __name__ == '__main__':

	# The number of processes can be passed as a command line argument.
	# By default the number of processes will equal the number of processors.
	num_procs = None
	if len(sys.argv) == 2:
		num_procs = sys.argv[1]
	elif len(sys.argv) > 2:
		raise ValueError('Expected at most one command line argument.  '
			'Found %d: %s.' % (len(sys.argv)-1, ', '.join(sys.argv)))

	do_corrections(num_procs)
