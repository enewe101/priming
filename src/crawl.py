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
	RECIPE_URL_LIST_FNAME = 'data/html/recipe_url_list.json'
	RECIPE_URL_QUEUE_FNAME = 'data/html/recipe_url_queue.json'
	CORPUS_FNAME = 'data/html/corpus.txt'
	RECIPE_CORPUS_FNAME = 'data/html/recipe_corpus.txt'
	MAX_TRIES = 3
	TIMEOUT = 3

	SLEEP = 2
	REST = 10
	BACKOFF = 60 * 5
	COOL_OFF = 60 * 30

	PAGE_REGEX = re.compile(r'.*(Page=\d+)')
	URL_OK_REGEX = re.compile(	
		r'http://allrecipes.com/Recipes/(US-Recipes|world-cuisine)', re.I)
	RECIPE_BASE_URL = 'http://allrecipes.com/Recipes/'
	RECIPE_MATCH = re.compile(r'.*detail.aspx', re.I)

	CRAWL_RECIPES = 0
	CRAWL_LISTINGS = 1

	def __init__(self, start=START):

		# initialize the url queue based on logs stored on disk
		self.make_url_queue()
		self.make_recipe_url_queue()

		# initialize accumulators
		self.strings = []
		self.recipe_strings = []

		# synchronize log files
		self.sync_disk(self.CRAWL_RECIPES)
		self.sync_disk(self.CRAWL_LISTINGS)
		self.check_files()


	def make_url_queue(self):

		print 'loading listing url queue...'

		if not os.path.isfile(self.URL_QUEUE_FNAME):
			self.url_queue = OrderedDict({
				self.normalize(start, start): {
					'url':start,
					'visited':False,
					'tries':0
				}
			})
			return self.url_queue

		queue_dict = json.loads(open(self.URL_QUEUE_FNAME).read())
		self.url_queue = OrderedDict(queue_dict)

		return self.url_queue


	def make_recipe_url_queue(self):

		print 'loading recipe url queue...'

		# if a recipe queue already exists on disk, just load that
		if os.path.isfile(self.RECIPE_URL_QUEUE_FNAME):
			self.recipe_url_queue = OrderedDict(
				json.loads(open(self.RECIPE_URL_QUEUE_FNAME).read())
			)
			return self.recipe_url_queue

		# otherwise, make a queue from the listing
		self.recipe_url_queue = OrderedDict()
		recipe_urls = open(self.RECIPE_URL_LIST_FNAME).read().split()
		self.add_recipe_urls(recipe_urls, self.RECIPE_BASE_URL)

		return self.recipe_url_queue


	def crawl_recipes(self):
		queue = self.recipe_url_queue
		mode = self.CRAWL_RECIPES
		self.crawl(queue, mode)

		
	def crawl_listings(self):
		queue = self.url_queue
		mode = self.CRAWL_LISTINGS
		self.crawl(queue, mode)


	def crawl(self, queue, mode):

		i = 0
		num_failures = 0
		for url_key in queue:

			url_obj = queue[url_key]
			if url_obj['visited'] or url_obj['tries'] > self.MAX_TRIES:
				continue


			print 'Crawling %s...' % url_key

			url_obj = queue[url_key]
			url = url_obj['url']

			# the way that the page gets processed depends on whether it is
			# a recipe page, or a listing of recipes
			if self.RECIPE_MATCH.match(url):
				result_summary = self.process_recipe_page(url)
			else:
				result_summary = self.process_listing_page(url)

			if result_summary:
				url_obj['visited'] = True
				url_obj['result_summary'] = result_summary
				num_failures = 0
			else:
				url_obj['tries'] += 1
				num_failures += 1

			# sleep a bit
			time.sleep(self.get_sleep_time())

			# periodically give the site a rest, and record progress
			if i % 20 == 19:
				self.sync_disk(mode)
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

		self.sync_disk(mode)
		print 'DONE!'


	def get_sleep_time(self):
		return r.uniform(0,2) * self.SLEEP


	def sync_disk(self, mode):
		'''
			Updates the url log on disk to reflect recent crawling
			progress.  Any new urls that have been discovered will be
			added, and the status of urls that have been crawled, or tried
			will be updated
		'''

		# the crawl mode deterimines where we'll store progress on disk
		corpus_fname = None
		queue_fname = None

		if mode == self.CRAWL_LISTINGS:
			corpus_fname = self.CORPUS_FNAME
			queue_fname = self.URL_QUEUE_FNAME
			url_queue = self.url_queue

			# copy and then purge the buffer
			strings = self.strings
			self.strings = []

		elif mode == self.CRAWL_RECIPES:
			corpus_fname = self.RECIPE_CORPUS_FNAME
			queue_fname = self.RECIPE_URL_QUEUE_FNAME
			url_queue = self.recipe_url_queue

			# copy and then purge the strings buffer
			strings = self.recipe_strings
			self.recipe_strings = []

		else:
			raise ValueError('mode must be self.CRAWL_LISTINGS or '
				'self.CRAWL_RECIPES.')

		# record the food-related strings found
		corpus_fh = open(corpus_fname, 'a')
		for string in strings:
			try:
				corpus_fh.write(string + '\n')
			except UnicodeEncodeError:
				print 'Skipping weird character...'

		corpus_fh.close()

		# record progress in the url log
		url_log = json.dumps(url_queue, indent=2)

		url_log_fh = open(queue_fname, 'w')
		url_log_fh.write(url_log)
		url_log_fh.close()


	def check_files(self):
		''' 
			Tries to open the file in append mode, then closes it again.
			The purpose of this is to catch an IO problem before we start 
			crawling, for example, if the path for the corpus file doesn't 
			exist.
		'''

		# make sure we can open the corpus files
		corpus_fh = open(self.CORPUS_FNAME, 'a')
		corpus_fh.close()
		corpus_fh = open(self.RECIPE_CORPUS_FNAME, 'a')
		corpus_fh.close()

		# make sure we can open the url log file
		url_log_fh = open(self.URL_QUEUE_FNAME, 'a')
		url_log_fh.close()
		url_log_fh = open(self.RECIPE_URL_QUEUE_FNAME, 'a')
		url_log_fh.close()

		# make sure we can open the recipe urls log file
		recipe_url_log_fh = open(self.RECIPE_URL_LIST_FNAME, 'a')
		recipe_url_log_fh.close()


	def absolutize(self, url, base):
		''' ensure the url is absolute '''
		return urljoin(base, url)


	def strip_query(self, url):
		''' remove the query string, but not if it has a page #! '''
		page_param = self.PAGE_REGEX.match(url)
		stripped = urlunparse(urlparse(url)[:3] + ('',)*3)

		# we don't want to strip page numbers though
		if page_param:
			stripped += '?' + page_param.groups()[0]

		return stripped


	def normalize(self, url, base):
		return self.strip_query(self.absolutize(url, base))


	def append_recipe_urls(self, urls, base):

		recipe_url_fh = open(self.RECIPE_URL_LIST_FNAME, 'a')
		for recipe_url in urls:
			absolutized = self.absolutize(recipe_url, base)
			recipe_url_fh.write(absolutized + '\n')

		recipe_url_fh.close()


	def add_recipe_urls(self, urls, base):

		for url in urls:

			absolutized = self.absolutize(url, base)
			normalized = self.normalize(url, base)

			if normalized not in self.recipe_url_queue:
				self.recipe_url_queue[normalized] = {
					'url': absolutized,
					'visited': False,
					'tries': 0
				}


	def add_urls(self, urls, base):

		for url in urls:

			absolutized = self.absolutize(url, base)
			normalized = self.normalize(url, base)

			if self.URL_OK_REGEX.match(normalized) is None:
				continue

			if normalized not in self.url_queue:
				self.url_queue[normalized] = {
					'url': absolutized,
					'visited': False,
					'tries': 0
				}


	def get_page(self, url):

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

		return r


	def process_listing_page(self, url):

		# get a page (returns a request object, or False on failure)
		r = self.get_page(url)
		if not r:
			return False

		soup  = bs(r.text)

		# get interesting things from the page
		nav_urls = find_nav_urls(soup)
		collection_urls = find_collection_urls(soup)
		next_urls = find_next_urls(soup)
		recipe_urls = find_recipe_urls(soup)
		strings = find_recipe_strings(soup)

		self.add_urls(nav_urls + collection_urls + next_urls, url)
		self.append_recipe_urls(recipe_urls, url)
		self.strings.extend(strings)
		
		result_summary = {
			'nav_urls': len(nav_urls),
			'collection_urls': len(collection_urls),
			'next_urls': len(next_urls),
			'recipe_urls': len(recipe_urls),
			'strings': len(strings)
		}

		return result_summary


	def process_recipe_page(self, url):

		# get a page (returns a request object, or False on failure)
		r = self.get_page(url)
		if not r:
			return False

		soup  = bs(r.text)

		# get interesting things from the page
		ingredients_strings = get_recipe_ingredients(soup)
		directions_strings = get_recipe_directions(soup)

		self.recipe_strings.extend(ingredients_strings)
		self.recipe_strings.extend(directions_strings)
		
		result_summary = {
			'ingredients_strings': len(ingredients_strings),
			'directions_strings': len(directions_strings)
		}

		return result_summary





def get_recipe_ingredients(soup):
	ingredient_elements = soup.find_all(class_='ingredient-name')
	ingredients = [i.text for i in ingredient_elements]
	return ingredients


def get_recipe_directions(soup):
	instructions_wrappers = soup.find_all('div', itemprop='recipeInstructions')
	instructions = [' '.join(i.text.split()) for i in instructions_wrappers]
	return instructions



def find_next_urls(soup):
	'''
		Accepts a soup object, and looks for the link to the 'next' page
		in order to paginate through allrecipe.com's recipe pages
	'''
	next_nodes = soup.find_all(text=re.compile('NEXT'))
	next_links = []
	for nn in next_nodes:
		try:
			next_links.append(nn.parent['href'])
		except KeyError:
			print 'Finished a collection!'

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

