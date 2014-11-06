import random as r
import time
import json
import os
from urlparse import urlparse, urlunparse, urljoin
from collections import deque, OrderedDict
from bs4 import BeautifulSoup as bs
import requests
import re


START = 'http://allrecipes.com/Recipes/World-Cuisine/'
HOST = 'http://allrecipes.com'


class Crawler(object):

	URL_QUEUE_FNAME = 'data/html/url_queue.json'
	RECIPE_URL_QUEUE_FNAME = 'data/html/recipe_url_queue.json'
	CORPUS_FNAME = 'data/html/corpus.txt'
	MAX_TRIES = 3
	TIMEOUT = 3

	SLEEP = 2
	REST = 10
	BACKOFF = 60 * 5
	COOL_OFF = 60 * 30


	def __init__(self, start=START):

		# initialize the url queue based on logs stored on disk
		self.url_queue = self.read_queue()

		# if no url logi is found use initialize with the starting url 
		if self.url_queue is None:
			self.url_queue = OrderedDict({
				self.normalize(start, start): {
					'url':start,
					'visited':False,
					'tries':0
				}
			})

		self.strings = []
		self.sync_disk()
		self.check_files()


	def crawl(self):

		i = 0
		num_failures = 0
		for url_key in self.url_queue:

			url_obj = self.url_queue[url_key]
			if url_obj['visited'] or url_obj['tries'] > self.MAX_TRIES:
				continue


			print 'Crawling %s...' % url_key

			url_obj = self.url_queue[url_key]
			url = url_obj['url']

			success = self.process_page(url)

			if success:
				url_obj['visited'] = True
				num_failures = 0
			else:
				url_obj['tries'] += 1
				num_failures += 1

			# sleep a bit
			time.sleep(self.get_sleep_time())

			# periodically give the site a rest, and record progress
			if i % 20 == 19:
				self.sync_disk()
				print 'Resting...'
				time.sleep(self.REST)

			# if we've had too many failures, backoff for awhile
			if num_failures > 30:
				print 'Cooling off...'
				time.sleep(self.COOL_OFF)

			elif num_failures > 10:
				print 'Backing off...'
				time.sleep(self.BACKOFF)

			i += 1

		self.sync_disk()
		print 'DONE!'


	def get_sleep_time(self):
		return r.uniform(0,2) * self.SLEEP


	def sync_disk(self):
		'''
			Updates the url log on disk to reflect recent crawling
			progress.  Any new urls that have been discovered will be
			added, and the status of urls that have been crawled, or tried
			will be updated
		'''
		# record the food-related strings found
		corpus_fh = open(self.CORPUS_FNAME, 'a')
		for string in self.strings:
			corpus_fh.write(string + '\n')

		corpus_fh.close()

		# purge the cache of found strings
		self.strings = []

		# record progress in the url log
		url_log = json.dumps(self.url_queue, indent=2)

		url_log_fh = open(self.URL_QUEUE_FNAME, 'w')
		url_log_fh.write(url_log)
		url_log_fh.close()


	def check_files(self):
		''' 
			Tries to open the file in append mode, then closes it again.
			The purpose of this is to catch an IO problem before we start 
			crawling, for example, if the path for the corpus file doesn't 
			exist.
		'''

		# make sure we can open the corpus file
		corpus_fh = open(self.CORPUS_FNAME, 'a')
		corpus_fh.close()

		# make sure we can open the url log file
		url_log_fh = open(self.URL_QUEUE_FNAME, 'a')
		url_log_fh.close()

		# make sure we can open the recipe urls log file
		recipe_url_log_fh = open(self.URL_QUEUE_FNAME, 'a')
		recipe_url_log_fh.close()


	def read_queue(self):

		if not os.path.isfile(self.URL_QUEUE_FNAME):
			return None

		queue_dict = json.loads(open(self.URL_QUEUE_FNAME).read())
		return OrderedDict(queue_dict)


	def absolutize(self, url, base):
		''' ensure the url is absolute '''
		return urljoin(base, url)


	def strip_query(self, url):
		''' remove the query string '''
		return urlunparse(urlparse(url)[:3] + ('',)*3)

	
	def normalize(self, url, base):
		return self.strip_query(self.absolutize(url, base))


	def add_recipe_urls(self, urls, base):

		recipe_url_fh = open(self.RECIPE_URL_QUEUE_FNAME, 'a')
		for recipe_url in urls:
			absolutized = self.absolutize(recipe_url, base)
			recipe_url_fh.write(absolutized + '\n')

		recipe_url_fh.close()


	def add_urls(self, urls, base):

		for url in urls:

			absolutized = self.absolutize(url, base)
			normalized = self.normalize(url, base)

			if normalized not in self.url_queue:
				self.url_queue[normalized] = {
					'url': absolutized,
					'visited': False,
					'tries': 0
				}


	def process_page(self, url):

		# get a page
		try:
			r = requests.get(url, timeout=self.TIMEOUT)
			r.raise_for_status()

		except requests.exceptions.HTTPError:
			print 'Warning: HTTP response code %d' % r.status_code
			return False

		except requests.exceptions.Timeout:
			print 'Warning: timed out.'
			return False

		except requests.exceptions.ConnectionError:
			print 'Warning: connection error.'
			return False

		except requests.exceptions.RequestException:
			print 'Warning: other request error.'
			return False

		soup  = bs(r.text)

		# get interesting things from the page
		self.add_urls(find_nav_urls(soup), url)
		self.add_urls(find_collection_urls(soup), url)
		self.add_urls(find_next_urls(soup), url)

		self.add_recipe_urls(find_recipe_urls(soup), url)

		self.strings.extend(find_recipe_strings(soup))

		return True





def find_next_urls(soup):
	'''
		Accepts a soup object, and looks for the link to the 'next' page
		in order to paginate through allrecipe.com's recipe pages
	'''
	next_nodes = soup.find_all(text=re.compile('NEXT'))
	next_links = [nn.parent['href'] for nn in next_nodes]

	return next_links



def find_nav_urls(soup):
	'''
		Accepts a soup object, and looks for the navigation bar that 
		allows drilling down to specific food collections.  It only looks
		at the most specific (i.e. textually last) navigation heading, which
		provides hrefs for finer grained ethnic food categories.

		This embeds assumptions about the allrecipes.com page structure.
	'''
	# navigate to the list of urls under the last (most specific) navitem
	nav = soup.find('nav', class_='hub-nav')
	last_navitem = nav.find('ul').find_all('li', recursive=False)[-1]
	urls = last_navitem.find_all('a')

	# get all the hrefs under that navitem
	hrefs = [a['href'] for a in urls]
	return hrefs


def find_collection_urls(soup):
	'''
		Accepts a soup object, and finds the navigation elements that allow
		you to look at sub-collections.  This is conceptually similar to 
		find_nav_urls, but it looks in a different place on the web page.

		returns an empty array if there are no collection urls
	'''
	# get to the right place by finding a certain comment in the html
	try:
		target_comment = soup.find_all(text=re.compile(r'^ Collections $'))[0]

	# if the target comment isn't found, then there are no collection urls
	except IndexError:
		return []

	# we move to a unique point in the html that is close to the collection
	# urls
	siblings = target_comment.next_siblings

	# walk along the siblings until the container that holds collection urls
	container = None
	for s in siblings:
		if s.name == 'div' and 'grid-view' in s['class']:
			container = s
			break

	# find the urls inside that container.  They are wrapped by a 
	# div with a particular class
	url_wrappers = container.find_all('div', class_='grid-result-cntnr')

	# and extract the hrefs.  Look at the url bound to the image.
	hrefs = [lw.find('a', class_='img-link')['href'] for lw in url_wrappers]

	return hrefs


def find_recipe_strings(soup):

	try:
		target_anchor = soup.find_all('a', id='recipes')[0]
	except IndexError:
		print 'Warning: did not find any recipes'
		return []

	container = None
	siblings = target_anchor.next_siblings

	for s in siblings:

		if s.name == 'div' and 'grid-view' in s['class']:
			container = s
			break

	recipe_wrappers = container.find_all('div', class_='grid-result-cntnr')
	recipe_strings = [rw.find('img')['alt'].strip() for rw in recipe_wrappers]

	return recipe_strings


def find_recipe_urls(soup):

	try:
		target_anchor = soup.find_all('a', id='recipes')[0]
	except IndexError:
		print 'Warning: did not find any recipes'
		return []

	container = None
	siblings = target_anchor.next_siblings

	for s in siblings:

		if s.name == 'div' and 'grid-view' in s['class']:
			container = s
			break

	recipe_wrappers = container.find_all('div', class_='grid-result-cntnr')
	recipe_urls = [rw.find('a')['href'] for rw in recipe_wrappers]

	return recipe_urls

