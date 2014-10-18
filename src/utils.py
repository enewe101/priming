import random as r
import numpy as np
import os

def all_none(lst):
	for a in lst:
		if a is not None:
			return False
	
	return True

def update_options():
	pass

def apply_options():
	pass

def merge_series(s1, s2, nonp=False):
	''' Given two lists of vectors as 2d arrays, merge them into one
	list of longer vectors. returned as an np array, unless nonp is set true
	in which case it is a python list of lists.'''

	if len(s1) != len(s2):
		raise ValueError('Series are not the same length: %d & %d.' % (len(s1),
			len(s2)))
	
	merged = []
	for row1, row2 in zip(s1,s2):
		merged.append(list(row1) + list(row2))
	
	if nonp:
		return merged

	return np.array(merged)


def sindex(lst, val):
	'''\'silent\' version of [].index(). gets the index of the first occurence
	of val in lst.  Returns None instead of raising ValueError if val is not
	in lst'''
	try:
		idx = lst.index(val)
	except ValueError:
		return None

	return idx

def nappend(lst, val):
	'''Appends to list only if the list doesn't already have the value.'''
	if sindex(lst, val) is None:
		lst.append(val)

def nextend(lst, e_lst):
	'''Extends lst with those elements in e_lst that aren't already in lst'''
	for elm in e_lst:
		nappend(lst, elm)

def unpend(lst, val):
	idx = sindex(lst,val)
	while idx is not None:
		del lst[idx]
		idx = sindex(lst,val)

def unextend(lst, e_lst):
	'''removes all occurences of the elements in e_lst from lst'''
	for elm in e_lst:
		unpend(lst, elm)

def copen(fname, can_append=False):
	if os.path.isfile(fname):
		if can_append:
			prompt = 'File (%s) exists.  overWrite, Append, or Cancel?'
			accept_vals = ['w','a','c']
		else:
			prompt = 'File (%s) exists.  overWrite, or Cancel?'
			accept_vals = ['w','c']

		open_mode = user_input(prompt, accept_vals)
		if open_mode == 'c':
			return False
	
	else:
		open_mode = 'w'

	return open(fname, open_mode)


def user_input(prompt, accept_vals, strip_reply=True, 
		is_retry=False, do_retry=True, preprocess=None, append_prompt=True, 
		case_sensitive=False):

		reply = None

		if append_prompt:
			prompt = '%s (%s): ' % (prompt, '/'.join(accept_vals)) 

		while reply not in accept_vals:
			if reply is not None:
				print 'Sorry didn\'t catch that.'

			reply = raw_input(prompt)

			if not case_sensitive:
				reply = reply.lower()

			if strip_reply:
				reply = reply.strip()

			if preprocess is not None:
				reply = prerpocoss(reply)

			if not do_retry:
				break

		return reply


def as_ranges(idxs, do_sort=False):
	if not isinstance(idxs, list):
		idxs = list(idxs)
	
	if do_sort:
		idxs.sort()

	if not len(idxs):
		return []

	ranges = []
	first = idxs[0]
	last = idxs[0]
	for idx in idxs[1:]:
		if idx == last+1:
			last = idx
			continue

		else:
			ranges.append((first, last))
			first = idx
			last = idx

	ranges.append((first, last))

	return ranges


def str_ranges(idxs, do_sort=False):
	'''Takes a list of indexes, and makes a more compact string description.
	E.g. [1,2,3,6,8,9,10] => '1-3, 6, 8-10'.  It will show the multiplicity
	of entries correctly.  pass in a set() if this is not desired'''

	ranges = as_ranges(idxs, do_sort)
	str_range = []
	for r in ranges:
		if r[0] == r[1]:
			str_range.append(str(r[0]))
		else:
			str_range.append('%d-%d' % r)

	return ', '.join(str_range)




def gather(args, **kwargs):
	'''Flattens an list of lists, also permitting primitives.  Veryfies that
	all elements are of type given by the check_type keyword argument.
	'''

	check_type = kwargs.pop('check_type', None)
	if check_type is None:
		legal_primitive_types = (basestring, nmbr.Number, type(None))
	else:
		legal_primitive_types = check_type

	flat_args = []
	for arg in args:

		# Catch primitive args
		if isinstance(arg, legal_primitive_types):
			flat_args.append(arg)

		# Catch iterable args
		else:
			try:
				list_arg = list(arg)
			except TypeError:
				return False
			flat_args.extend(list_arg)

	# do verification if a checktype was specified
	for a in flat_args:
		if not isinstance(a, legal_primitive_types):
			return False


	return flat_args
		

def product(l):
	'''Computes product of elements in a list
	'''
	return reduce(lambda x,y: x*y,l,1)


def get_pairs(pop, k):
	'''Produces a list of pairings of elements in pop.  Each element gets 
	paired k times.  If pop does not have an even number of elements, 
	ValueError is raised.'''

	try:
		pop = list(pop)
	except TypeError:
		raise TypeError('pop must be iterable.')

	n = len(pop)
	if n & 1: # if n is odd
		raise ValueError('pop must contain an even number of elements')

	pairs = []
	num_pairs = [0]*n	# ith entry gives the number of pairs containing ith
						# item
	
	# Randomly generate k pairings for each item
	while True:
		# Randomly choose an item from among those having the fewest pairs
		min_int = min(num_pairs)
		few_interactions = filter(lambda x: x[1] == min_int, 
				enumerate(num_pairs))
		idx1, num_pairs1 = r.choice(few_interactions)

		# if this agent has had k pairs, then all agents must have k
		# pairs and we are done
		if num_pairs1 == k:
			break

		# randomly choose any agent that doesn't have k pairs
		available_agents = filter(
				lambda x: num_pairs[x]<k and x!=idx1,
				range(n))

		# randomly choose an agent from the list
		idx2 = r.choice(available_agents)

		# create the pair
		pairs.append((idx1, idx2))

		# record the interaction for both agents
		num_pairs[idx1] += 1
		num_pairs[idx2] += 1

	return pairs

def choose(n, k):
	"""
	A fast way to calculate binomial coefficients by Andrew Dalke (contrib).
	"""
	if 0 <= k <= n:
		ntok = 1
		ktok = 1
		for t in xrange(1, min(k, n - k) + 1):
			ntok *= n
			ktok *= t
			n -= 1
		return ntok // ktok
	else:
		return 0

class Lottery:
	def __init__(self, **kwargs):
		self.defaults = {
			'lots': None,
			'total': None
		}

		for key, val in self.defaults.items():
			setattr(self, key, kwargs.pop(key,val))

		if len(kwargs):
			raise ValueError('Unexpected keyword argument in '\
				'games.utils.Lottery.draw(): %s' % ', '.join(
				[str(k) for k in kwargs.keys]))

		if self.lots is not None:
			self.set_lottery(self.lots, self.total)


	def set_lottery(self, lots, total=None):
		self.lots = list(lots)
		self.num_lots = len(lots)

		# User can leave some lots with unspecified value using NoneType
		# If a total value is given, then the amount remaining in the total
		# beyond the sum of the lots is divided between the unspecified lots
		null_lots = filter(
				lambda x: self.lots[x] is None, range(len(self.lots)))
		num_null = len(null_lots)

		# The difference between sum of all lots and `total' gets divided
		# between the null lots
		if num_null:

			# A total had to be specified if there are null lots
			if total is None:
				ValueError('Could not set lottery in '\
					'games.utils.set_lottery(): NoneType lots supplied '\
					'without specifying lot total.')

			# Divide difference between current sum and total between null lots
			partial_total = sum(filter(lambda lot: lot is not None, lots))
			null_lot_value = (total - partial_total) / float(num_null)
			for idx in null_lots:
				self.lots[idx] = null_lot_value 	

		if total is None:
				self.total = sum(self.lots)

		else:
			self.total = total

		self.cumulative = [sum(self.lots[0:i+1]) for i in range(self.num_lots)]


	def draw(self, lots=None, total=None):
		if lots is not None:
			self.set_lottery(lots, total)

		if self.lots is None or self.total is None:
			raise ValueError('Could not draw in games.utils.Lottery.draw():'\
					' self.lots must be iterable of numeric types.')

		pick = r.uniform(0, self.total)
		pick_idx = 0
		while pick_idx < self.num_lots and pick > self.cumulative[pick_idx]:
			pick_idx += 1
		
		if pick_idx == self.num_lots:
			pick_idx = None

		return pick_idx
		

