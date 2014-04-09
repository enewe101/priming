import util
import random
import json
import csv

def readDataset():
	'''
	Factory method that builds a CleanDataset from the original Amazon
	Mechanical Turk CSV files
	'''

	# Create a new priming-image-label-experiment dataset
	dataSet = CleanDataset()

	# Read from the raw amt csv files.  
	# Note: order matters!  The older files have duplicates workers that
	# get ignored.  Make sure to read the newer file files earlier
	dataSet.read_csv('amt_csv/amt1_cut.csv', True)
	dataSet.read_csv('amt_csv/amt2_cut.csv', True)
	dataSet.read_csv('amt_csv/amt3_cut.csv', True)

	# The dataset needs to do some internal calts to refresh its state 
	dataSet.aggregateCounts()
	dataSet.calc_ktop(5)

	return dataSet


class CleanDataset(object):

	# Constants
	K = 5
	NUM_IMAGES = 5
	NUM_WORDS_PER_IMAGE = 5

	def __init__(self):

		# State flags
		self.hasKtop = False		# whether ktop is up-to-date
		self.isAggregated = False	# whether aggregated values are up-to-date

		# Data
		self.dictionary = set()		# a list of all words that occur 
		self.entries = {}			# an entry holds the data for one worker
		self.testEntries = {}		# holds the unused entries when sub-sampled
		self.trainEntries = {}
		self.workerIds = set()		# to prevent duplicates
		self.counts = {}			# stores words and frequency of occurrence
		self.ktops = {}


	# In principle This should be moved to the NBDataset class, because this 
	# separation
	# is really a concern introduced by the desire to train a Naive Bayes
	# Classifier on the data.  But for now it is here.  The reason is because
	# I had written the function aggregateCounts also here, but really both
	# should be moved to Naive Bayes Classifier.
	def subsample(self, trainingSetSize, testSetSize=None):
		'''
		Partitions the dataset into a training set, and test set.  
		'''
		self.trainEntries = {}
		self.testEntries = {}
		self.unusedEntries = {}

		# Split the entries of each treatment into testing and training sets
		for treatment, entries in self.entries.items():

			if testSetSize is None:
				lTestSetSize = len(entries) - trainingSetSize
				unusedSize = 0

			else:
				lTestSetSize = testSetSize
				unusedSize = len(entries) - trainingSetSize - testSetSize

			train, test, unused = util.randomPartition(
				entries, trainingSetSize, lTestSetSize, unusedSize)

			self.trainEntries[treatment] = train
			self.testEntries[treatment] = test
			self.unusedEntries[treatment] = unused

		# Now that partitioning is done, recalculate aggregates
		self.aggregateCounts()
		self.calc_ktop()


	def getTestInstances(self):
		return self.testEntries


	def clearAggregateCounts(self):
		
		# We want to clear all the aggregated counts.  But we don't want to
		# clear the counts that are specific to a treatment-image-position
		for treatment, image, position in self.counts.keys():

			# We can tell that a count is aggregated if the 
			# position coordinate is None.  If so, delete it.
			if position is None:
				del self.counts[(treatment, image, position)]


	def aggregateCounts(self):

		# First we need to clear out stale counts
		self.clearAggregateCounts()

		# We are going to iterate over all the counts that are associated
		# to a specific treatment, image, and word-position, and aggregate
		# these counts 
		for treatment, image, position in self.counts.keys():
			count_dict = self.counts[(treatment, image, position)]

			# Don't consider counts that are themselves aggregate.  These are
			# recognizeable because the word-position in the key is None
			if position is None:
				continue

			for word, frequency in count_dict.items():

				# Depending on which of the entries in the key-tuple is left
				# as None, we aggregate counts to various degrees
				self._aggregateCount((treatment, None, None), word, frequency)
				self._aggregateCount((treatment, image, None), word, frequency)
				self._aggregateCount((None, image, None), word, frequency)

				# This entry in counts is a data-set wide count. Maybe that
				# means I should drop the self.dictionary
				self._aggregateCount((None, None, None), word, frequency)

		self.isAggregated = True


	def _aggregateCount(self, key, word, count):

		# If this is a new key, make it
		if key not in self.counts:
			self.counts[key] = {}

		# If this is a new word, make a new word-count entry
		if word not in self.counts[key]:
			self.counts[key][word] = count

		# Otherwise, just increment the word-count entry
		else:
			self.counts[key][word] += count



	def read_csv(self, fname, hold=False):
		self.hasKtop = False
		self.isAggregated = False

		fh = open(fname, 'r')
		reader = csv.DictReader(fh)

		for record in reader:

			# Skip duplicate workers.
			workerId = record['WorkerId']
			if workerId in self.workerIds:
				continue

			newEntry = {}

			# Note the new worker id
			self.workerIds.add(workerId)
			newEntry['workerId'] = workerId

			# Note the experimental treatment for this worker
			tmt_id = 'treatment' + record['Answer.treatment_id']
			newEntry['treatment'] = tmt_id

			if tmt_id not in self.entries:
				self.entries[tmt_id] = []

			self.entries[tmt_id].append(newEntry)

			# Record the image files used to prime this worker
			newEntry['primingImageFiles'] = []

			# Do the following for both the priming and testing image-sets
			for sub_treatment in ['prime', 'test']:

				# Iterate over all the images in the image-set
				for img_num in range(self.NUM_IMAGES):

					# The test images are numbered sequentially, following the
					# priming images, so we need to apply an offset
					offset = 0 if sub_treatment=='prime' else self.NUM_IMAGES
					amt_img_num = img_num + offset

					# This is how we name images in the dataset
					img_id = sub_treatment + str(img_num)

					# Record the name of the file for this image
					newEntry['primingImageFiles'].append(
						record['Answer.img_%d_id' % amt_img_num])

					# Now, within the data recorded for each image, 
					# iterate over each position.  These correspond to the
					# text-inputs in the HIT
					for word_pos in range(self.NUM_WORDS_PER_IMAGE):

						# Get the word from the csv file, normalize to 
						# lowercase, store it in the record, and add it to
						# the data-set-wide dictionary
						word = record['Answer.img_%d_word_%d' 
							% (amt_img_num, word_pos)].lower()
						newEntry[(img_id, word_pos)] = word
						self.dictionary.add(word)

						# If there is no entry in self.counts for this 
						# treatment, image, and word-position, make one
						self._aggregateCount(
							(tmt_id, img_id, word_pos), word, 1)

		# Make the training entry set be the full entry set, and make the
		# test set empty.  Subsampling changes this partitioning
		for treatment, entries in self.entries.items():
			self.trainEntries[treatment] = list(entries)
			self.testEntries[treatment] = []

		# Update the aggregated counts and the k-top words (except if held)
		if not hold:
			self.aggregateCounts()
			self.calc_ktop(self.K)


	def write_counts(self, directory):

		# Make sure that the directory ends with a slash
		# TODO check that it exists
		if not directory.endswith('/'):
			directory += '/'

		for image_id in self.counts_by_image.keys():

			# Open a file for image counts for this image
			# Write the counts, then close.
			fh_img_counts = open(directory + image_id + '.txt', 'w')
			fh_img_counts.write(self.list_counts_for_img(image_id))
			fh_img_counts.close()

			for treatment_id in self.counts_by_tmt_image.keys():


				# Open a file for image counts for this image and treatment
				# combination.  Write the counts, then close.
				fh_tmt_img_counts = open(
					directory + image_id + '_' + treatment_id + '.txt', 'w')
				fh_tmt_img_counts.write(
					self.list_counts_for_tmt_img(treatment_id, image_id))
				fh_tmt_img_counts.close()


	def getWordFrequency(
		self, pWord, pTreatment=None, pImage=None, pPosition=None):
		'''
		Returns the number of occurrences of a word, when looking within a
		specific treatment, at the tags attributed to a specific image, and
		in a specific text input (e.g first, second,..., fifth)

		We can regard the treatment_id, image_id, and position as coordinates
		that successively zero in on a more specificly designed feature of 
		the dataset.  But it makes sense to be able to omit some or all of 
		these.  So, for example, by specifying only a word, but not a 
		treatment_id, image_id, or position, we should receive the frequency
		of occurrence of that word over the entire dataset.  On the other hand
		if we specify the treatment_id and image, but not position, then we 
		should get the number of times a word was attributed to the indicated
		image under the indicated treatment, but without regard for which 
		position it was in.

		For the moment, I will support only specifying a parameter provided
		that the ones earlier in the list are specified.  So, e.g. it is an 
		error to specify which position, but not which image.
		'''
		numOccurrences = 0

		if not self.isAggregated:
			self.aggregateCounts()

		if pWord not in self.counts[(pTreatment, pImage, pPosition)]:
			return 0

		else:
			return self.counts[(pTreatment, pImage, pPosition)][pWord]

				
	def list_counts_for_img(self, img_id):
		'''
		Return a string that lists one word per line followed by its frequency
		sorted in descending order of frequency
		'''
		counts = sorted(
			self.counts[(None, img_id, None)].items(), 
			None, lambda x: x[1], True)

		string = ''
		for word, frequency in counts:
			string += word + ' ' + str(frequency) + '\n'

		return string


	def list_counts_for_tmt_img(self, treatment_id, image_id):
		'''
		Return a string that lists one word per line followed by its frequency
		sorted in descending order of frequency
		'''
		counts = sorted(
			self.counts[(treatment_id, image_id, None)].items(), 
			None, lambda x: x[1], True)

		string = ''
		for word, frequency in counts:
			string += word + ' ' + str(frequency) + '\n'

		return string


	def calc_ktop(self, k=K):
		'''
		Determine the most frequent k words in each word position of each
		image for each treatment.  This is stored in self.ktops which has a
		structure that is analogous to the self.counts.
		'''

		# Iterate over all of the counts for specific treatments, images, and
		# word positions
		for treatment, image, position in self.counts.keys():

			# We only want to consider the minimally aggregated data, that is
			# we want counts that are specific to a word-position
			if position is None:
				continue

			word_count_dict = self.counts[(treatment, image, position)]
			ranked_word_counts = sorted(word_count_dict.items(),
					None, lambda x: x[1], True)

			top_k_words = ranked_word_counts[0:k]
			self.ktops[(treatment, image, position)] = top_k_words

		hasKtop = True






