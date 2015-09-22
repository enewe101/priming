




def extract_post(soup):
	comments_container = soup.find('div', class_='snapComments')
	post['comments'] = [
		extract_comment(c) 
		for c in comment_container.find_all('ul', class_='comments')
	]

def extract_comment(comment_div)
	comment['body'] = 'stuff found in comment'

	replies_container = comment_div.find('li').find('ul')
	comment['comments'] = [
		extract_comment(c)
		for c in replies_container.find_all('ul', class_='comments')
	]

def extract_deep_comments()
	pass
