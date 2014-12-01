'''
	This runnable module is a convenience to call the vocabulary_null_model
	analysis function from the command line.  The first argument should
	be one of the following:

		task1, frame1, echo, task2, frame2

'''

import sys
from analysis import vocabulary_null_model

if __name__ == '__main__':
	treatment_pair = sys.argv[1]
	vocabulary_null_model(treatment_pair)
