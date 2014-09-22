from nltk.corpus import wordnet as wn
from collections import Counter
import re


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


class WordnetSpellChecker(object):

	ALPH = 'abcdefghijklmnopqrstuvwxyz_'
	W = re.compile(r'[^a-z ]')

	def __init__(self):
		pass

	def is_ok(self, word):
		return bool(len(wn.synsets(word))>0)


	def get_neighbors(self, word):

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

		return set(deletes + transposes + replaces + inserts + separations)


	def auto_correct(self, corpus):

		# First, we read all of the words.
		# Sort out the ones that are misspelled.  Among the ones that 
		# are correctly spelled, accumulate the frequencies of occurence
		# to guide the spell checker

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

			single_edits = self.get_neighbors(w)
			double_edits = reduce(
				lambda x,y: x | self.get_neighbors(y), 
				single_edits, 
				set()
			)

			first_candidates = filter(
				lambda x: 
					all([self.is_ok(y) for y in x.split()])
					and len(x.split())>0,
				single_edits)

			second_candidates = filter(
				lambda x: 
					all([self.is_ok(y) for y in x.split()])
					and len(x.split())>0, 
				double_edits)

			scored_candidates = (
				[(fc, min([frequencies[x] for x in fc.split()])) 
					for fc in first_candidates] + 
				[(sc, min([frequencies[x] for x in sc.split()]) / 2.0 )
					for sc in second_candidates]
			)

			try:
				sorted_candidates = sorted(
					scored_candidates, None, 
					lambda x: x[1],
					True
				)
				best_word = sorted_candidates[0][0]

			except IndexError:
				best_word = None

			print w, '->', best_word
			corrections[w] = best_word

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


def map_to_synsets(clean_dataset, treatment, images):
	'''
	given a CleanDataset, a treatment name, and image name, 
	get all the labels that workers from the given treatment attribute to
	the given image, and map those labels into synsets.  Return a Counter
	with synset names as keys, and values as the number of occurances.
	(Note that the map from labels to synsets is many to many.)
	'''

	# get a counter for the words associated to the treatment and image
	label_counts = Counter()
	for image in images:
		label_counts += clean_dataset.get_counts_for_treatment_image(
			treatment, image)

	synset_counts = Counter()
	for label in label_counts:
		synsets = get_all_synsets(label)
		add_to_counts = dict([(s, label_counts[label]) for s in synsets])
		synset_counts.update(add_to_counts)

	return synset_counts


def calculate_relative_specificity(synset_counts_1, synset_counts_2):

	# Make relative counters based on synset_counts_2
	ancester_counter = WordnetRelativesCalculator(synset_counts_2, True)
	descendant_counter = WordnetRelativesCalculator(synset_counts_2, False)

	specificity_score = 0

	# for every synset in synset_counts_1, add the number of descendants
	# minus the number of ancesters to get the net relative specificity

	for synset_name, count in synset_counts_1.iteritems():
		synset = wn.synset(synset_name)
		specificity_score += count * ancester_counter.count(synset)
		specificity_score -= count * descendant_counter.count(synset)

	return specificity_score


class WordnetFoodDetector(object):
	def __init__(self):
		self.foodish_cache = {
			'food.n.02': True, 
			'food.n.01': True, 
			'helping.n.01': True,
			'taste.n.01': True, 
			'taste.n.05': True, 
			'taste.n.06': True, 
			'taste.n.07': True
		}

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
		return self.foodish_detecting_walker.walk(synset)

		

class WordnetRelativesCalculator(object):

	def __init__(self, synset_counts, search_ancesters=False):

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
		return self.relative_counting_walker.walk(synset)
		


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
		allow_double_process=True
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


	def do_callback(self, callback_name, *args):
		func = self.callbacks[callback_name]
		if hasattr(func, 'im_func'):
			return func.im_func(*args)

		return func(*args)


	def walk(self, cur_node):

		# mark this node as seen
		cur_node_hash = self.do_callback('get_node_hash', cur_node)
		self.seen_nodes.add(cur_node_hash)

		# decide whether this node needs to be followed
		# if not, treat it like a leaf
		if self.do_callback('abort_branch_callback', cur_node):
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

		# recursively process all the child nodes, keep their returned vals
		child_vals = map(lambda c: self.walk(c), children)

		# depending on if node was a leaf, call the appropriate callback
		if len(children)>0:
			return self.do_callback(
				'inter_node_callback', cur_node, child_vals)

		return self.do_callback('leaf_node_callback', cur_node)



















