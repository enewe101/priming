from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support.ui import WebDriverWait 
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.select import Select
from selenium.webdriver.common.by import By

def get_job_descriptions():
	driver = webdriver.Firefox()
	url=( 
		"https://www.mturk.com/mturk/viewhits?"
		"searchWords=&"
		"selectedSearchType=hitgroups"
		"&sortType=LastUpdatedTime%3A1"
		"&pageNumber=1"
		"&searchSpec=HITGroupSearch%23T%232%2310%23-1%23T%23!%23!LastUpdatedTime!1!%23!"
	)
	driver.get(url)
	table = driver.find_elements_by_tag_name('table')[6]
	for i, td in enumerate(table.find_elements_by_tag_name('td')):
		print td.text
