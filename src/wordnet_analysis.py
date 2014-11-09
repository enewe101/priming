import os
import copy
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from collections import Counter
import re
import numpy as np
import json


SPLIT_RE = re.compile('[^-a-zA-Z]')



def filter_misspelled(labels):
	words = set(reduce(lambda x,y: x + y.split(), labels, []))
	misspelled = filter(lambda x: len(wn.synsets(x)) == 0, words)
	return misspelled
	

def find_misspelled_words(dataset):

	misspelled = {}
	for image in ['test%d' % i for i in range(dataset.NUM_IMAGES)]:
		words = dataset.get_counts_for_treatment_image(None, image).keys()
		misspelled[image] = filter_misspelled(words)

	return misspelled


def list_unknown_words():
	'''
		looks through the dictionaries (which provide corrections to 
		misspelled words), and identifies which of those are not in wordnet.
		These words will need to be placed into the ontology.
	'''
	DICTIONARY_DIR = 'data/new_data/dictionaries/with_allrecipes/'
	DICTIONARY_FNAMES = ['dictionary_1.json', 'dictionary_2.json']
	NEW_WORDS_FNAME = 'new_words.json'
	stops = stopwords.words('english')

	fnames = [os.path.join(DICTIONARY_DIR, d) for d in DICTIONARY_FNAMES]
	new_words_fh = open(os.path.join(DICTIONARY_DIR, NEW_WORDS_FNAME), 'w')
	
	new_words = []
	for fname in fnames:

		print 'Processing %s' % fname

		fh = open(fname)
		dictionary = json.loads(fh.read())

		for volume in dictionary:

			classes = volume['params']['class_idxs']
			image = volume['params']['img_idxs'][0]
			exp = volume['params']['which_experiment']
			print 'experiment: %d, image: %s, classes%s' % (
				exp, str(classes), image)

			entries = volume['results']

			# get all the correction words.  Omit None's, and empty strings
			words = filter(lambda x: x and x.strip(), entries.values())

			# split words
			words = reduce(lambda x,y: x + y.split(), words, [])
			words = filter(
				lambda x: len(wn.synsets(x))<1 and x not in stops, 
				words
			)
			words = [
				{'experiment': exp, 'image': image, 'word': w} 
				for w in words
			]
			new_words.extend(words)
	
	new_words_fh.write(json.dumps(words, indent=2))
	new_words_fh.close()

	return new_words

	



class WordnetSpellChecker(object):

	ALPH = 'abcdefghijklmnopqrstuvwxyz_'
	W = re.compile(r'[^a-z ]') # note a space is included
	stops = stopwords.words('english')

	CORPUS_DIR = 'data/html/'
	AUGMENTED_CORPUS_FNAMES = ['corpus.txt', 'recipe_corpus.txt']
	TRIMED_AUX_CORPUS_FNAME = 'data/new_data/auxiliary_corpus.txt'

	SPLIT_REGEX = re.compile(r"[^a-z]", re.I) # no spaces, no punctuation

	def __init__(self, num_to_return=1):
		self.aux_corpus = None 
		self.num_to_return = num_to_return


	def is_ok(self, word):

		return (
			len(wn.synsets(word))>0 
			or word in self.stops 
			or word in self.aux_corpus
		)


	def all_ok(self, word):
		return all([self.is_ok(token) for token in word.split()])


	def get_all_neighbors(self, words_and_scores):
		all_neighbors = set()
		for word, score in words_and_scores:
			all_neighbors.update(self.get_neighbors(word, score))

		return all_neighbors


	def get_neighbors(self, word, score):

		w = word

		splits = [(w[:i], w[i:]) for i in range(len(w) + 1)]

		deletes = [a + b[1:] for a,b in splits if b]
		transposes = [a + b[1] + b[0] + b[2:] for a,b in splits if len(b)>1]
		replaces = [a + c + b[1:] 
			for a,b in splits for c in self.ALPH if b]
		inserts = [a + c + b for a,b in splits for c in self.ALPH]

		# if there are non-word characters, try replacing with spaces
		if self.W.search(w):
			separations = [self.W.sub(' ', w)]

		# otherwise try inserting spaces in all possible positions
		else:
			separations = [a + ' ' + b for a,b in splits]

		# return all the possibilities, but the separations get half-score
		# as a penalty
		return_neighbors = set(
			[(s,score*0.50) for s in deletes + transposes + replaces + inserts]
		)
		return_neighbors |= set([(s,score*0.1) for s in separations])

		return return_neighbors


	def read_auxiliary_corpus(self):
		'''
			reads a bunch of raw text stripped from some source (in this case
			currently the allrecipes.com website) and makes a list of all the
			new words seen therein.  Eliminates duplicates, punctuation (but
			preserves appostrophes), and normalizes to lower case.

			After processing in the abovementionned way, only those tokens that
			are *not* in wordnet are actually kept.  The reason is because 
			we already use wordnet, and we only need to store what is above
			and beyond wordnet's vocabulary (keeping loading time in mind).

			Prints the trimmed version of this to a trimmed coprus file.
		'''

		aux_tokens = set()
		trimmed_corpus_fh = open(self.TRIMED_AUX_CORPUS_FNAME, 'w')

		# read auxiliary corpus files, and aggregate the all tokens found
		for corpus_fname in self.AUGMENTED_CORPUS_FNAMES:
			fname = os.path.join(self.CORPUS_DIR, corpus_fname)
			corpus_fh = open(fname)

			print 'Getting tokens from corpus...'
			for line in corpus_fh:
				tokens = self.SPLIT_REGEX.split(line.lower())
				tokens = filter(lambda x: len(x)>2, tokens)
				aux_tokens.update(tokens)

		# only keep the aux_tokens that aren't found in wordnet
		print "Culling tokens (don't include anything already in wordnet)"
		culled_aux_tokens = set(
			filter(lambda x: len(wn.synsets(x)) == 0, aux_tokens)
		)

		print 'writing culled tokens to file'
		trimmed_corpus_fh.write('\n'.join(culled_aux_tokens) + '\n')

		return culled_aux_tokens


	def auto_correct(self, corpus):

		# read in the auxiliary corpus.
		if os.path.isfile(self.TRIMED_AUX_CORPUS_FNAME):
			self.aux_corpus = set(
				open(self.TRIMED_AUX_CORPUS_FNAME).read().split()
			)

		# if the auxiliary corpus is not found, look for the raw stripped
		# words
		else:
			self.aux_corpus = self.read_auxiliary_corpus()

		# Now, read all of the words in the argument corpus.
		# Sort out the ones that are misspelled.  Among the ones that 
		# are correctly spelled, accumulate the frequencies of occurence
		# as a guide to the spell checker

		frequencies = Counter()
		misspelled = set()

		for phrase in corpus:
			for word in phrase.split():

				if self.is_ok(word):
					frequencies[word] += 1

				else:
					misspelled.add(word)

		print 'found %d misspelled words...' % len(misspelled)

		# Now try to correct the spellings
		corrections = {}
		for w in misspelled:

			candidates = self.get_neighbors(w,1)
			candidates |= self.get_all_neighbors(candidates)

			# remove nonsense candidates
			candidates = filter(lambda x: self.all_ok(x[0]), candidates)

			# remove empty candidates
			candidates = filter(lambda x: len(x[0].split())>0, candidates)

			# multiply scores by the frequency of occurence in the corpus
			#
			# words that have been split get afforded the minimum frequency
			# of the resulting words.  All words get at least a score of 1
			scored_candidates = [
				(c, s*(1 + min([frequencies[x] for x in c.split()])))
				for c,s in candidates
			]

			try:
				sorted_candidates = sorted(
					scored_candidates, None, 
					lambda x: x[1],
					True
				)
				# take the top words
				best_words = sorted_candidates[:self.num_to_return]

				# discard the score
				best_words = [b[0] for b in best_words]

			except IndexError:
				best_words = None

			print w, '->', best_words
			corrections[w] = best_words

		return corrections

	


def get_similarity(counts_1, counts_2):

	all_keys = set(counts_1.keys() + counts_2.keys())
	intersection_size = sum([min(counts_1[k], counts_2[k]) for k in all_keys])
	union_size = sum(counts_1.values()) + sum(counts_2.values())

	return 2 * intersection_size / float(union_size)
	


def get_all_synsets(label):
	'''
	Given a label, get all the synsets to which it maps.  If a label consists
	of multiple words, get the mapping for each word, as well as for the
	label treated as a single compound word.  There are sometimes synsets for
	compound words.
	'''

	all_tokens_to_try = label.split() 
	# compound words occur in wordnet as words glued together with underscores
	all_tokens_to_try.append('_'.join(label.split()))

	synsets = set()
	for token in all_tokens_to_try:
		synsets.update([s.name for s in wn.synsets(token)])

	return synsets


def map_to_synsets(word_counts):
	'''
	given a CleanDataset, a treatment name, and image name, 
	get all the labels that workers from the given treatment attribute to
	the given image, and map those labels into synsets.  Return a Counter
	with synset names as keys, and values as the number of occurances.
	(Note that the map from labels to synsets is many to many.)
	'''

	synset_counts = Counter()
	for word in word_counts:
		count = word_counts[word]
		synsets = get_all_synsets(word)
		add_to_counts = dict([(s, count) for s in synsets])
		synset_counts.update(add_to_counts)

	return synset_counts


def strip_food_words(word_counts):
	food_detector = WordnetFoodDetector()

	#Make a new copy of the word counts, but only include non-food words
	new_counts = {}
	for key in word_counts.keys():
		if not food_detector.is_food(key):
			new_counts[key] = word_counts[key]

	return new_counts



def calculate_relative_specificity(
		word_counts_1, 
		word_counts_2,
		ignore_food=False
	):
	'''
	returns the relative specificity between to sets of synsets.
	This is positive when synset_counts_1 is more specific overall than
	synset_counts_2.  Reversing the order of the arguments reverses the sign, 
	but gives the same result.
	'''

	if ignore_food:
		print len(word_counts_1), len(word_counts_2)
		word_counts_1 = strip_food_words(word_counts_1)
		word_counts_2 = strip_food_words(word_counts_2)
		print len(word_counts_1), len(word_counts_2)

	synset_counts_1 = map_to_synsets(word_counts_1)
	synset_counts_2 = map_to_synsets(word_counts_2)

	# Make relative counters based on synset_counts_2
	ancester_counter = WordnetRelativesCalculator(synset_counts_2, True)
	descendant_counter = WordnetRelativesCalculator(synset_counts_2, False)

	specificity_score = 0

	# for every synset in synset_counts_1, add the number of descendants
	# minus the number of ancesters to get the net relative specificity

	for synset_name, count in synset_counts_1.iteritems():
		syn = wn.synset(synset_name)
		specificity_score += count * ancester_counter.count(syn)
		specificity_score -= count * descendant_counter.count(syn)

	return specificity_score / float(sum(word_counts_1.values()) * 
		sum(word_counts_2.values()))


class WordnetFoodDetector(object):
	def __init__(self):
		self.food_synsets = [
			'food.n.02', 
			'food.n.01', 
			'helping.n.01',
			'taste.n.01', 
			'taste.n.05', 
			'taste.n.06', 
			'taste.n.07'
		]

		self.foodish_cache = dict([(s,True) for s in self.food_synsets])

		# We will build a DFS walker to help with the calculation
		# To do so, we need to define a bunch of callbacks
		def get_node_hash(node):
			return node.name

		# We alter the get children callback to get parents if we want to
		# count the number of ancestors instead of descendants.
		def get_children_callback(node):
			return node.hypernyms()


		def inter_node_callback(node, child_vals=[]):
			node_name = node.name

			if node_name in self.foodish_cache:
				return self.foodish_cache[node_name]

			# if any of the "children" (actually, in this case hypernyms are
			# more like ancesters) are foodish, then this node is foodish
			cur_node_val = any(child_vals)

			# cache this for next time
			self.foodish_cache[node_name] = cur_node_val

			return cur_node_val 


		def abort_branch_callback(node):
			# if we have a cached value for this node, no need to continue
			# the traversal any deeper
			if node.name in self.foodish_cache:
				return True
			return False

		allow_double_process = True

		# we build a walker with these callbacks.
		self.foodish_detecting_walker = DFS(
			get_node_hash,
			get_children_callback,
			inter_node_callback,
			inter_node_callback,	# this callback works for leaves too
			abort_branch_callback,
			allow_double_process
		)


	def get_food_vocab(self):

		# we will bulid a DFS that accumulates all of the food-related lemmas
		# begin by defining the DFS's callback functions
		def get_node_hash(node):
			return node.name

		def get_children_callback(node):
			return node.hyponyms()
		
		def inter_node_callback(node, child_vals=[]):
			this_result = [l.name for l in node.lemmas]

			# each child returns a list, flatten this to one list
			child_results = reduce(lambda x,y: x + y, child_vals, [])

			print this_result + child_results

			return this_result + child_results

		def abort_branch_callback(node):
			return False

		allow_double_process = False

		# build the DFS
		food_vocab_counter = DFS(
			get_node_hash,
			get_children_callback,
			inter_node_callback,
			inter_node_callback,	# this callback works for leaves too
			abort_branch_callback,
			allow_double_process
		)

		# for each top-level food synset, 
		# walk the DFS to collect the vocabulary
		vocab = []
		for syn_name in self.food_synsets:
			syn = wn.synset(syn_name)
			vocab += food_vocab_counter.start_walk(syn)

		# eliminate duplicates
		vocab = set(vocab)

		# done!
		return vocab


	def is_food(self, label):

		tokens = SPLIT_RE.split(label)

		# if the label has multiple words, consider the compound word first
		if len(tokens) > 1:
			if self.is_token_food('_'.join(tokens)):
				return True

		# otherwise *all* of the tokens should be food
		return all([self.is_token_food(t) for t in tokens])


	def is_token_food(self, token):
		synsets = wn.synsets(token)
		return any([self.is_synset_food(s) for s in synsets])


	def is_synset_food(self, synset):
		return self.foodish_detecting_walker.start_walk(synset)

		

class WordnetRelativesCalculator(object):

	def __init__(
			self,
			synset_counts,
			search_ancesters=False
		):

		self.search_ancesters = search_ancesters
		self.num_relatives_cache = {}

		# We will build a DFS walker to help with the calculation
		# To do so, we need to define a bunch of callbacks
		def get_node_hash(node):
			return node.name

		# We alter the get children callback to get parents if we want to
		# count the number of ancestors instead of descendants.
		if self.search_ancesters:
			def get_children_callback(node):
				return node.hypernyms()

		else:
			def get_children_callback(node):
				return node.hyponyms()

		def inter_node_callback(node, child_vals=[]):
			node_name = node.name

			if node_name in self.num_relatives_cache:
				# print node_name, '(:)', self.num_relatives_cache[node_name]
				return self.num_relatives_cache[node_name]

			cur_node_val = synset_counts[node_name] + sum(child_vals)

			# cache this for next time
			self.num_relatives_cache[node_name] = cur_node_val

			# print node_name, ' : ', cur_node_val

			return cur_node_val 


		def abort_branch_callback(node):
			if node.name in self.num_relatives_cache:
				return True
			return False

		allow_double_process = True

		# we build a walker with these callbacks.
		self.relative_counting_walker = DFS(
			get_node_hash,
			get_children_callback,
			inter_node_callback,
			inter_node_callback,	# this callback works for leaves too
			abort_branch_callback,
			allow_double_process
		)


	def count(self, synset):
		return self.relative_counting_walker.start_walk(synset)

		
class DFSError(Exception):
	pass

# test that no state is carried from one walk to the next
class DFS(object):
	'''
	provides a depth-first-search walk for any kind of nodes that form a graph.
	Safe against cycles when allow_double_process=False.  
	
	The graph should be implied by the get_children_callback, which, when
	called on a node, must yield a list of children, or an empty list.
	
	The inter_node_callback and leaf_node_callback will be called on inter
	nodes and leaf nodes respectively.  The inter_node callback should take
	a second argument, which is a list of the return values from the 
	callbacks run on its children.

	The abort_branch_callback is fired on a node before it's children
	get processed.  If it returns true, the children will not be processed
	and it will be treated as a leaf.

	When allow_double_process is False, any node that has been seen before 
	will not be processed again.  No calback is fired on it, and so no value
	for it is included in the children values passed to the second argument
	of its parent's inter_node_callback.
	'''

	CHILDREN_NOT_PROCESSED = 0
	CHILDREN_PROCESSED = 1

	def __init__(self, 
		get_node_hash,
		get_children_callback,
		inter_node_callback,
		leaf_node_callback,
		abort_branch_callback,
		allow_double_process=True,
	):
		self.callbacks = {
			'get_node_hash': get_node_hash,
			'get_children_callback': get_children_callback,
			'inter_node_callback': inter_node_callback,
			'leaf_node_callback': leaf_node_callback,
			'abort_branch_callback': abort_branch_callback
		}
		self.allow_double_process = allow_double_process

		self.seen_nodes = set()
		self.nodes_in_process = []


	def reset(self):
		self.seen_nodes = set()
		self.nodes_in_process = []


	def do_callback(self, callback_name, *args):
		func = self.callbacks[callback_name]
		if hasattr(func, 'im_func'):
			return func.im_func(*args)

		return func(*args)


	def queue(self, node_hash):
		self.nodes_in_process.append(node_hash)


	def dequeue(self, node_hash):
		found_node_hash = self.nodes_in_process.pop()
		if found_node_hash != node_hash:
			raise DFSError('Invalid traversal order.')


	def start_walk(self, start_node):
		# at the begining of the walk clear any history from previous walks
		self.reset()
		return self.walk(start_node)


	def walk(self, cur_node):

		# mark this node as seen, and in process
		cur_node_hash = self.do_callback('get_node_hash', cur_node)
		self.seen_nodes.add(cur_node_hash)
		self.queue(cur_node_hash)

		# decide whether this node needs to be followed
		# if not, treat it like a leaf
		if self.do_callback('abort_branch_callback', cur_node):
			self.dequeue(cur_node_hash)
			return self.do_callback('leaf_node_callback', cur_node)

		# get the children
		children = self.do_callback('get_children_callback', cur_node)

		# don't double process (if that's what we want)
		if not self.allow_double_process:
			children = filter(
				lambda c: self.do_callback(
					'get_node_hash', c) not in self.seen_nodes,
				children
			)

		# Recursively process all the child nodes, keep their returned vals
		# But don't process children that create a cycle!
		child_vals = [
			self.walk(c) 
			for c in children 
			if self.do_callback('get_node_hash', c) 
				not in self.nodes_in_process
		]

		# depending on if node was a leaf, call the appropriate callback
		if len(children)>0:
			self.dequeue(cur_node_hash)
			return self.do_callback(
				'inter_node_callback', cur_node, child_vals)


		# remove this node from the in process list, and do integrity check
		self.dequeue(cur_node_hash)
		return self.do_callback('leaf_node_callback', cur_node)




