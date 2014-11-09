import os
import sys
import numpy as np
import copy
import json
import time
import utils as u
from Queue import Empty as QueueEmpty
from multiprocessing import Process, Pool, Queue, Value, Manager


def sim_sweep(sim_class, fname, schedule, sim_defaults={}, 
	sim_constructor_kwargs={}, num_procs=None, chunksize=10):

	'''Runs many simulators (of sim_class) in parallel, at all parameter
	values supplied in schedule, and writes results to fname.

	sim_defualts allows simulation parameters to be specified which are 
	constant from one run to the next.  If schedule specifies a value also
	specified in sim_defualts, schedule takes precedence.

	num_procs determines the number of worker processes dedicated to 
	simulations.  If num_procs is None, the number of simulation workers will
	match the number of processers. 

	In addition to simulation processes, there is the main process, and a 
	writing process that writes results to file in json format as they become 
	available.
	'''

	# TODO: allow no fname, in which case sim data returned as array.


	# Open file.  If already exists, consult user.
	fh = u.copen(fname) # <- automatically checks if exists and consults
	if not fh:
		print 'Aborted!'
		return
	fh.write('[\n')		# Start the json list of simulation data
	fh.close()

	# Shared data
	manager = Manager()
	write_queue = manager.Queue()
	all_done = manager.Value('h', 0)

	task_args = {'sim_class': sim_class, 'sim_defaults':sim_defaults, 
		'sim_constructor_kwargs': sim_constructor_kwargs,
		'write_queue':write_queue}

	# Use a pool of workers to run sims at all param values.  They put results
	# in the write_queue
	params_iter = ParamIterator(schedule)
	task_iter = TaskIter(params_iter, task_args)
	pool = Pool(num_procs)
	pool.map_async(run_sim, task_iter)
	pool.close()

	# Start a writer process which writes to file from the write_queue
	p = Process(target=write_results, args=(fname, write_queue, all_done))
	p.start() 	# This consumes items from the write_queue

	# Close the pool and join all processes
	pool.join()	# joins once: all sim tasks done AND write_queue fully consumed
	all_done.value = True
	p.join()

	print 'Completed!'



class ParamIterator:
	def __init__(self, sweep):
		self.keys = []
		self.values = []
		self.lengths = []
		self.cursors = [0]*len(sweep)
		for key, values in sweep.items():
			self.keys.append(key)
			self.values.append(values)
			self.lengths.append(len(values))

		self.no_iterations_left = False

	def __iter__(self):
		return self

	def next(self):
		if self.no_iterations_left:
			raise StopIteration

		cur_vals = [self.values[i][j] for i, j in enumerate(self.cursors)]
		self.increment_cursors()
		return dict(zip(self.keys, cur_vals))


	def increment_cursors(self):
		done_increment_cursor = False

		curs_idx = 0
		while not done_increment_cursor:
			# If you are pointing beyond the last cursor, it means the
			# last cursor rolled over, so all iteration is complete
			if curs_idx == len(self.cursors):
				self.no_iterations_left = True
				done_increment_cursor = True

			# If the pointed-at cursor is at end, roll over and increment
			# the next cursor in the cursor list
			elif self.cursors[curs_idx] + 1 == self.lengths[curs_idx]:
				self.cursors[curs_idx] = 0
				curs_idx += 1

			# Otherwise succussfully increment the pointed-at cursor.
			else:
				self.cursors[curs_idx] += 1
				done_increment_cursor = True


class TaskIter:
	def __init__(self, iterable, task_args):
		self.iterable = iterable
		self.task_args = task_args

	def __iter__(self):
		return self

	def next(self):
		next_item = self.iterable.next()
		return (next_item, self.task_args)



def run_sim((params, task_args)):

	start = time.clock()

	sim_class = task_args['sim_class']
	sim_defaults = task_args['sim_defaults']
	write_queue = task_args['write_queue']
	constructor_kwargs = task_args['sim_constructor_kwargs']

	params = dict(sim_defaults.items() + params.items())
	print 'Started: %s' % str(params)
	sim = sim_class(**constructor_kwargs)
	run_data = sim.run(**params)

	elapsed = time.clock() - start

	add_data = {'params':params, 'results':run_data, 'run_time': elapsed}

	write_queue.put(add_data)
	print 'Finished: %s in %f seconds.' % (str(params), elapsed)


def write_results(fname, write_queue, all_done):
	fh = open(fname, 'a')
	first_line = True

	while not all_done.value or not write_queue.empty():

			try:
				result = write_queue.get(timeout=1)

			except QueueEmpty:
				continue

			else:
				if first_line:
					first_line = False
				else:
					fh.write(',\n')

				fh.write(json.dumps(result, indent=2))
	
	fh.write('\n]')


def debug_exit():
	sys.exit(0)
