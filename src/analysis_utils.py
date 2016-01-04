import numpy as np
import util

CONFIDENCE_95 = 1.96
CONFIDENCE_99 = 2.975

def prob_k_successes(n,k,p=0.5):
	return util.choose(n,k) * (p**k)*(1-p)**(n-k)

def get_k_star(n, alpha):

	k = n
	prob_tot = prob_k_successes(n,k)
	k_star = None

	while prob_tot < alpha:
		k_star = k
		k -= 1
		prob_tot += prob_k_successes(n,k)

	return k_star

def get_theta_star(n, alpha):
	k_star = get_k_star(n,alpha)
	return 2 * k_star / float(n) - 1


def binomial_confidence_intervals(
		n,k,alpha=0.05, tolerance=1e-6, as_theta=False
	):
	'''
		Gets both the upper and lower confidence intervals for 
		the single-experiment probability of success, for a binomial
		variable sampled n times, observed to have k succeses, according 
		to significance level alpha.  The numerical error on the calculation
		(not the statistical error mind you) is given by tolerance.
		as_theta controls whether to return the results as the probability
		of success (p) or as theta = 2p-1
	'''
	try:
		upper = binomial_upper_confidence_p(n,k,alpha, tolerance, as_theta)
		lower = binomial_lower_confidence_p(n,k,alpha, tolerance, as_theta)

	except OverflowError:
		# get it from normal distribution
		if alpha==0.05:
			z = CONFIDENCE_95

		elif alpha==0.15865:
			z = 1.0

		else:
			raise NotImplementedError(
				'this function currently only supports alpha=0.05 when the '
				'sample size is large.  Youll need to code up something to '
				'produce z-scores for non alpha=0.05 cases.'
			)

		std_dev = np.sqrt(
			1/float(n)*(k/float(n))*(1-k/float(n))
		)
		upper = k/float(n) + z * std_dev
		lower = k/float(n) - z * std_dev

		if as_theta:
			upper = 2*upper - 1
			lower = 2*lower - 1

	return upper, lower


def binomial_upper_confidence_p(
	n,k,alpha=0.05, tolerance=1e-6, as_theta=False):
	'''
		For a Binomial RV Bin(n,p), with unknown p,
		the largest probability p for which we expect to observe at least
		k successes with probability 1 - alpha/2.
	'''

	high_p = 1
	high_prob = binom_upper_tail_prob(n,k,high_p)

	low_p = 0
	low_prob = binom_upper_tail_prob(n,k,low_p)

	cur_p = 0.5
	cur_prob = binom_upper_tail_prob(n,k,cur_p)

	while abs(high_p - low_p) > tolerance:

		# if the probability is bigger than alpha, reduce cur_p
		if cur_prob > 1 - alpha/2.:
			high_p = cur_p
			high_prob = cur_prob

		# if the probability is smaller than alpha, increase cur_p
		elif cur_prob < 1 - alpha/2.:
			low_p = cur_p
			low_prob = cur_prob

		# if it's dead on, break out
		else:
			break

		# take another guess at cur_p
		cur_p = (high_p + low_p)/ 2.0
		cur_prob = binom_upper_tail_prob(n,k,cur_p)

	if as_theta:
		return 2*cur_p - 1

	return cur_p



def binomial_lower_confidence_p(
	n,k,alpha=0.05, tolerance=1e-6, as_theta=False):
	'''
		For a Binomial RV Bin(n,p), with unknown p,
		the smallest probability p for which we expect to observe at least
		k successes with probability alpha/2.
	'''

	high_p = 1
	high_prob = binom_upper_tail_prob(n,k,high_p)

	low_p = 0
	low_prob = binom_upper_tail_prob(n,k,low_p)

	cur_p = 0.5
	cur_prob = binom_upper_tail_prob(n,k,cur_p)

	while abs(high_p - low_p) > tolerance:

		# if the probability is bigger than alpha, reduce cur_p
		if cur_prob > alpha/2.:
			high_p = cur_p
			high_prob = cur_prob

		# if the probability is smaller than alpha, increase cur_p
		elif cur_prob < alpha/2.:
			low_p = cur_p
			low_prob = cur_prob

		# if it's dead on, break out
		else:
			break

		# take another guess at cur_p
		cur_p = (high_p + low_p)/ 2.0
		cur_prob = binom_upper_tail_prob(n,k,cur_p)

	if as_theta:
		return 2*cur_p - 1

	return cur_p



def binom_upper_tail_prob(n,k,p):
	'''
		What is the probability of having at least k successes for the 
		binomial variable Bin(n,p)?
	'''
	total_prob = 0
	
	for k_prime in range(k,n+1):
		total_prob += prob_k_successes(n,k_prime,p)

	return total_prob

