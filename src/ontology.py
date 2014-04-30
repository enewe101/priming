'''
This module has just one class, `Ontology` which handles two basic concerns.

The first is to represent an ontology of words, which is used by the 
`analysis.Analyzer` class as a basis for comparing treatments.  So, given an
ontology, `Ontology` answers questions like "is 'naan' a type of 'bread'?".

The second concern is the actual building of ontologies, which is somewhat
onerous.  `Ontology` provides methods that allow one to incrementally add
words and relationships between words to build up an ontology.  You will notice
various methods having short (often one-letter) names, which are the methods
mainly anticipated for this use-case.

Finally, `Ontology` provides methods for reading and writing representations
of instances of itself.

What is an ontology? it is a directed acyclic graph of words, wherein one
word "points to" words that are specific cases.  For example, bread --> bun.
No cycles are allowed.  The ontology has one root called "ROOT".  One of
the top-level nodes is called "d" and it stands for "discarded", so words
that it points to are meant to be discarded.
'''


import copy
import json
import re


#
# TODO:
#	- I am unclear as to how the mask is being applied when it comes to
#		specificity and to valence.  I want to look at this again to make 
#		sure that is recorded somewhere.
#


class Ontology(object):

	def __init__(self):

		self.wordList = set()

		self.edgeList = set()
		self.model = {}
		self.isModelStale = False

		self.synonymList = set()
		self.translator = {}
		self.isTranslatorStale = False

		self._mask = set()	# nodes in the mask affect how compare works


	def writeOntology(self, fname):
		'''
		writes out the ontology including the hierarchy of tokens and the
		synonyms as a single json file
		'''

		self.refresh()

		representation = {
			'wordList': list(self.wordList),
			'edgeList': list(self.edgeList),
			'model':self.model,
			'synonymList': list(self.synonymList),
			'translator':self.translator
		}

		fh = open(fname, 'w')
		fh.write(json.dumps(representation, indent=3))
		fh.close()


	def readOntology(self, fname):
		'''
		reads a json-encoded ontology file
		'''
		representation = json.loads(open(fname, 'r').read())
		self.wordList = set([tuple(w) for w in representation['wordList']])
		self.edgeList = set([tuple(e) for e in representation['edgeList']])
		self.synonymList = set(
			[tuple(s) for s in representation['synonymList']])

		# Refresh the ontology
		self.isModelStale = True
		self.isTranslatorStale = True
		self.refresh()


	def readWords(self, fname):
		'''
		Reads a list of words and their frequencies (number of occurrences).  
		The file should have one word followed by its count per line.
		One word and 
		'''

		wordList = set()
		frequencyRegx = re.compile(r'([^0-9]*)([0-9]*)')
		fh = open(fname, 'r')
		words = fh.read().split('\n')


		# parse out the words and the counts
		for wordEntry in words:
			match = frequencyRegx.match(wordEntry)
			if match:
				word = self.strip(match.group(1))
				count = match.group(2)
				wordList.add((word, count))

			else:
				print 'Warning -- word `%s` has no count' % wordEntry
				continue

		self.wordList |= wordList


	def getWords(self, thresholdCount=2):

		sortedWordList = sorted(
			list(self.wordList), None, lambda w: w[1], True)

		returnWords = []
		for word, count in sortedWordList:
			try:
				count = int(count)
			except ValueError:
				continue

			if int(count) < thresholdCount:
				continue

			synonym = self.getSynonym(word)
			if synonym not in self.model:
				returnWords.append((word, count))

		return returnWords


	def readEdgeList(self, fname):
		fh = open(fname, 'r')
		edgeListFile = fh.read().split('\n')
		edgeList = set()

		for line in edgeListFile:

			cells = line.split('\t')

			# Ignore lines with an empty first cell
			if cells[0].strip() == '':
				continue

			# There can be more than one parent, but we'll store each 
			# edge as a single (parent, child) entry
			parents = map(lambda x: self.strip(x), cells[0].split(','))
			child = self.strip(cells[1])

			for parent in parents:
				edgeList.add((parent, child))

		self.edgeList |= edgeList
		self.isModelStale = True
		self.refresh()



	def addSynonym(self, w1, w2):
		# for duplicate detection, put the two words in alphabetical order
		w1, w2 = sorted((w1, w2), key=str.lower)

		if (w1, w2) in self.synonymList:
			print '(`%s`, `%s`) is already in the synonym list.' % (w1, w2)

		else: 
			self.synonymList.add((w1,w2))
			self.isModelStale = True
			self.isTranslatorStale = True

		self.isTranslatorStale = True
		self.isModelStale = True
		self.refresh()

	def asyn(self, w1, w2):
		self.addSynonym(w1,w2)


	def removeSynonym(self, w1, w2, verbose=True):
		# for duplicate detection, put the two words in alphabetical order
		w1, w2 = sorted((w1, w2), key=str.lower)

		if (w1,w2) in self.synonymList:
			self.synonymList.remove((w1,w2))
			if verbose:
				print 'removed (`%s`, `%s`).' % (w1, w2)

			self.isModelStale = True
			self.isTranslatorStale = True
			self.refresh()

		else:
			raise ValueError('The entry (`%s`, `%s`) does not exist in the'\
				' synonymList.' % (w1,w2))

	def rs(self, w1, w2, verbose=True):
		self.removeSynonym(w1, w2, verbose)


	def readSynonyms(self, fname):
		'''
		Reads a tab-delimeted file containing two tokens per line.  The tokens
		are separated by tabs.  Either token may have trailing numbers 
		indicating their frequency.  These get trimmed and ignored.  The
		synonyms in the file are stored as a synonymList, which is list of 
		2-tuples, where the two terms in each tuple are considered synonymous.

		Reading multiple synonym files simply appends the synonym tuples
		onto the existing synonymList.

		This function also triggers re-building of the translator.
		'''

		if len(self.model):
			raise Exception('Ontology data has already been read. '\
				'You must read synonym data before ontology data!')

		# There can be many words that mean the same thing.  They form an 
		# equivalence class -- a synonym set.  
		# One representative is arbitrarily chosen from the synonym set to
		# be the canonical representative. 
		fh = open(fname, 'r')
		synonymFile = fh.read().split('\n')
		synonymList = set()
		for line in synonymFile:
			cells = line.split('\t')
			if cells[0].strip() == '':
				continue

			term1, term2 = map(self.strip, cells)

			# for duplicate detection, put the two words in alphabetical order
			term1, term2 = sorted((term1, term2))

			synonymList.add((term1, term2))

		self.synonymList |= synonymList
		self.isTranslatorStale = True
		self.isModelStale = True
		self.refresh()



	def addNode(self, parent, child):
		if (parent, child) in self.edgeList:
			print '(`%s`, `%s`) is already in the edge list.' % (parent, child)
			return

		else:
			if(self.getSynonym(parent) not in self.model):
				print 'Error `%s` not in model.' % parent
				return

			self.edgeList.add((parent, child))

			self.isModelStale = True
			self.refresh()

	def an(self, parent, child):
		self.addNode(parent, child)


	def removeNode(self, parent, child, verbose=True):
		'''
		'''
		# First get all possible synonyms (not just the cononical one) for 
		# parent and child.
		parentSyns = self.getSynSet(parent)
		childSyns = self.getSynSet(child)

		# Based on the possible synonyms for the parent, find the possible
		# edges that this might refer to
		candidateEdges = []
		for p in parentSyns:
			candidateEdges.extend(filter(lambda e: e[0] == p, self.edgeList))

		# among the candidate edges, delete any that have the child token 
		# synonymous to the passed child
		foundEdge = False
		for e in candidateEdges:
			if e[1] in childSyns:
				self.edgeList.remove((e[0],e[1]))
				foundEdge = True
				if verbose:
					print 'removed (`%s`, `%s`).' % e

		# If any edges were removed, the model is stale
		if foundEdge:
			self.isModelStale = True
			self.refresh()

		# If no edges were removed, provide that feedback
		elif verbose:
			print '(`%s`, `%s`) is not in the edge list.' % (parent, child)


	def rn(self, parent, child, verbose=True):
		self.removeNode(parent, child, verbose)


	def mv(self, curParent, newParent, child, verbose=True):
		self.removeNode(curParent, child, verbose)
		self.addNode(newParent, child)


	def refresh(self):
		if self.isTranslatorStale:
			self.buildTranslator()

		if self.isModelStale:
			self.buildModel()


	def buildTranslator(self):
		synonyms = {}
		for term1, term2 in self.synonymList:

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

		self.isTranslatorStale = False


	def buildModel(self):
		'''
		The model that is actually used for the ontology collapses all words
		into their connonical synonym to prevent ambiguity.  Since this
		operation is destructive to the edgelist on which it was based, we
		keep the original edgelist.  This prevents the loss of information
		if two different things are accidentally marked as synonymous
		'''

		self.model = {}
		for parent, child in self.edgeList:

			# check if there is a synonym for the child
			child = self.getSynonym(child)

			# check if there is a synonym for the parent
			parent = self.getSynonym(parent)

			if parent not in self.model:
				self.model[parent] = []

			# by including placeholder for a token's children, even it has
			# none, we can easily check if a token is in the model by looking
			# only at keys, and we can immediately tell if a token is a leaf
			if child not in self.model:
				self.model[child] = []

			self.model[parent].append(child)

		self.isModelStale = False


	def checkTranslatorStale(self):
		if self.isTranslatorStale:
			raise OntologyError('The ontology is in a stale state.  Run '\
				'ontology.refresh().')
	

	def checkIfStale(self):
		if self.isTranslatorStale or self.isModelStale:
			raise OntologyError('The ontology is in a stale state.  Run '\
				'ontology.refresh().')


	def mask(self, tokenToMask):
		self._mask.add(self.getSynonym(tokenToMask))


	def clearMask(self):
		self._mask = set()

	def unmask(self, tokenToUnmask):
		self._mask.remove(self.getSynonym(tokenToUnmask))


	def compare(self, token1, token2, strict=True):

		self.checkIfStale()

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

		self.checkIfStale()

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

		self.checkIfStale()

		token = self.getSynonym(token)

		parents = []
		for parent, children in self.model.items():
			if token in children:
				parents.append(parent)

		return parents

	def getChildren(self, token):
		self.checkIfStale()

		token = self.getSynonym(token)

		if token in self.model:
			return copy.copy(self.model[token])

		else:
			print '`%s` could not be found in the ontology.' % token


	def getSynonym(self, token):

		self.checkTranslatorStale()

		# If there's an entry in the translator, return the canonical synonym
		# otherwise, just use the token as is
		try:
			token = self.translator[token]
		except KeyError:
			pass

		return token


	def __str__(self):

		self.checkIfStale()

		roots = self.model['ROOT']
		string = ''
		for r in roots:
			string += self.recurseStr(r, 0)

		return string


	def recurseStr(self, p, depth=0):
		self.checkIfStale()

		string = ('\t'*depth) + p + '\n'

		if p not in self.model:
			return string

		for child in self.model[p]:
			string += self.recurseStr(child, depth + 1)

		return string


	def findOrphans(self):
		'''
		Every parent must itself have a parent, otherwise, our tree isn't 
		rooted.  The exception is when the node in question is itself ROOT.
		This function searches for nodes that have no parent
		'''

		self.checkIfStale()

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


	def getSynSet(self, token):
		self.checkTranslatorStale()

		token = self.getSynonym(token)
		synSet = set([token])

		synEntries = filter(lambda s: s[1] == token, self.translator.items())
		synSet |= set([s[0] for s in synEntries])

		return synSet




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


	def w(self, numToShow=5, thresholdCount=2):
		'''
		Print out all the words that are not yet added to the model, and
		which occur at least thresholdCount (default 2) number of times.
		'''
		self.checkIfStale()
		wordsToShow = ['%s\t%s' % w for w in self.getWords(thresholdCount)]
		print '\n'.join(wordsToShow[0:numToShow])


	def p(self, token):
		'''
		Print out the parents for the token.
		'''
		self.checkIfStale()
		print '\n'.join(self.getParents(token))


	def c(self, token):
		'''
		Print out the children for the token.
		'''
		self.checkIfStale()
		print '\n'.join(sorted(self.getChildren(token)))


	def cs(self, token):
		'''
		Print out the children of the token as synonym sets.
		'''
		children = self.getChildren(token)
		for c in children:
			print ', '.join(self.getSynSet(c))


