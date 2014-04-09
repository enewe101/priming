import json
import re

class Ontology(object):
	def __init__(self):
		self.model = {}
		self.translator = {}
		self._mask = set([])		# nodes in the mask affect how compare works


	def readEdgeList(self, fname):
		fh = open(fname, 'r')
		edgeList = fh.read().split('\n')
		for line in edgeList:
			cells = line.split('\t')
			

			# Ignore lines with an empty first cell
			if cells[0].strip() == '':
				continue

			parents = map(lambda x: self.strip(x), cells[0].split(','))
			child = self.strip(cells[1])

			# check if there is a synonym for the child
			if child in self.translator:
				child = self.translator[child]

			for p in parents:

				# First check if there is a synonym for the parent
				p = self.getSynonym(p)

				if p not in self.model:
					self.model[p] = []
				self.model[p].append(child)


	def mask(self, tokenToMask):
		self._mask.add(self.getSynonym(tokenToMask))


	def clearMask(self):
		self._mask = set()

	def unmask(self, tokenToUnmask):
		self._mask.remove(self.getSynonym(tokenToUnmask))


	def compare(self, token1, token2, strict=True):

		# Resolve synonyms, then get the ancesters for the tokens
		token1, token2 = map(self.getSynonym, [token1, token2])
		ancesters1, ancesters2 = map(self.getAncesters, [token1, token2])


		if strict:
			# each token must have no masked ancesters at all
			roots1 = (ancesters1 | set([token1])) & set(self.model['ROOT'])
			roots2 = (ancesters2 | set([token2])) & set(self.model['ROOT'])
			if (roots1 & self._mask) or (roots2 & self._mask):
				return 0


		else:
			# each token has to have at least one non-masked ancester
			roots1 = (ancesters1 | set([token1])) & set(self.model['ROOT'])
			roots2 = (ancesters2 | set([token2])) & set(self.model['ROOT'])
			if not (roots1 - self._mask) or not (roots2 - self._mask):
				return 0

		if token1 in self.getAncesters(token2):
			return 1

		if token2 in self.getAncesters(token1):
			return -1

		return 0


	def getAncesters(self, token):

		ancesters = set()

		# Use the canonical synonym
		token = self.getSynonym(token)

		parents = self.getParents(token)
		for p in parents:
			if p != 'ROOT':
				ancesters.add(p)
				ancesters |= self.getAncesters(p)

		return ancesters


	def getParents(self, token):
		token = self.getSynonym(token)

		parents = []
		for parent, children in self.model.items():
			if token in children:
				parents.append(parent)

		return parents


	def writeOntology(self, fname):
		'''
		writes out the ontology including the hierarchy of tokens and the
		synonyms as a single json file
		'''
		representation = {'model':self.model, 'translator':self.translator}
		fh = open(fname, 'w')
		fh.write(json.dumps(representation, indent=3))
		fh.close()


	def readOntology(self, fname):
		'''
		reads a json-encoded ontology file
		'''
		representation = json.loads(open(fname, 'r').read())
		self.model = representation['model']
		self.translator = representation['translator']


	def getSynonym(self, token):

		# If there's an entry in the translator, return the canonical synonym
		# otherwise, just use the token as is
		try:
			token = self.translator[token]
		except KeyError:
			pass

		return token


	def readSynonyms(self, fname):
		if len(self.model):
			raise Exception('Ontology data has already been read. '\
				'You must read synonym data before ontology data!')

		# There can be many words that mean the same thing.  They form an 
		# equivalence class -- a synonym set.  Here, we also have the notion
		# of a canonical representative of the class.  These are chosen
		# completely arbitrarily -- no guarantees!

		fh = open(fname, 'r')
		synonymList = fh.read().split('\n')
		synonyms = {}
		for line in synonymList:
			cells = line.split('\t')
			if cells[0].strip() == '':
				continue

			term1, term2 = map(self.strip, cells)

			# These terms may already have some entries in the synonym
			# set.  We now need to pool these entries into one synonym set
			synonymSet = set([term1, term2])
			for term in [term1, term2]:
				if term in synonyms:
					synonymSet |= set(synonyms[term])

			# Now that they are pooled, make this synonym set addressable by
			# any of the aliases that occur within it
			for term in synonymSet:
				synonyms[term] = list(synonymSet)

		# with synonym sets created, we arbitrarily choose a representative
		# of each synonym set.
		self.translator = {}
		for word, synonymSet in synonyms.items():

			# arbitrarily pick the first word from the synonym set to be
			# the representative.  When we run accross other representatives
			# entries will already be made in the translator, so they can be
			# skipped
			if word in self.translator:
				continue

			for alias in synonymSet:
				if alias != word:
					self.translator[alias] = word



	def __str__(self):

		roots = self.model['ROOT']
		string = ''
		for r in roots:
			string += self.recurseStr(r, 0)

		return string


	def findOrphans(self):
		'''
		Every parent must itself have a parent, otherwise, our tree isn't 
		rooted.  The exception is when the node in question is itself ROOT.
		This function searches for nodes that have no parent
		'''

		orphans = []
		for parent in self.model.keys():

			# Don't look for the parent of ROOT
			if parent == 'ROOT':
				continue

			# parent must be a child of someone
			foundParent = False
			for children in self.model.values():
				if parent in children:
					foundParent = True

			if not foundParent:
				orphans.append(parent)

		return orphans


	def addNode(self, parent, child):
		if parent not in self.model:
			self.model[parent] = []

		self.model[parent].append(child)




	def strip(self, string):

		# Remove quotes
		if string.startswith('"'):
			string = string[1:]
		if string.endswith('"'):
			string = string[:-1]

		# Remove trailing numbers
		trailDigitRegx = re.compile(r'([^0-9]*)')
		match = trailDigitRegx.match(string)
		if match:
			string = match.group(0)
		
		# Strip whitespace
		string = string.strip()

		return string

	def recurseStr(self, p, depth):
		string = ('\t'*depth) + p + '\n'

		if p not in self.model:
			return string

		for child in self.model[p]:
			string += self.recurseStr(child, depth + 1)

		return string



