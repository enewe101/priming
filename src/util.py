import sys
import random

def as_scientific(float_number, precision):
	formatter = '%%1.%de' % (precision-1)
	as_str = formatter % float_number
	digits, exp = as_str.split('e')
	return float(digits), int(exp)

def as_scientific_latex(
		float_number,
		precision,
		min_exp=None, 
		math_delimiters=True
	):
	'''
		renders a float as latex-formatted string with pretty scientific 
		notation.  min_exp is the minimum absolute value of the exponent,
		below which the string will be prented in plain decimals.
	'''

	if min_exp is None:
		min_exp = precision

	digits, exp = as_scientific(float_number, precision)

	if abs(exp) < min_exp:
		formatter = '%%1.%df' % (abs(exp) + precision - 1)
		formatted =  formatter % (float_number)

	else:
		formatter = '%%1.%df \\times 10^{%%d}' % (precision-1)
		formatted = formatter % (digits, exp)

	if math_delimiters:
		return '$%s$' % formatted

	return formatted


def writeNow(string):
	sys.stdout.write(string)
	sys.stdout.flush()


def counter_relative_diff(c1, c2):
	result = {}
	for key in set(c1.keys() + c2.keys()):
		diff = c1[key] - c2[key]
		avg = (c1[key] + c2[key]) / 2.0
		result[key] = diff / avg

	return result


def counter_subtract(c1, c2):
	result = {}
	for key in set(c1.keys() + c2.keys()):
		result[key] = c1[key] - c2[key]

	return result

def randomPartition(collection, *args):
	firstPartitionSize = args[0]

	# randomly sample the first partition
	firstPartition = random.sample(collection, firstPartitionSize)

	# Set aside the remainin not-sampled elements
	remaining = []
	for i in collection:
		if i not in firstPartition:
			remaining.append(i)

	# recursively partition the remainder
	if len(args) > 1:
		remainingPartitions = randomPartition(remaining, *args[1:])
	else:
		remainingPartitions = []

	return [firstPartition] + remainingPartitions


def choose(n, k):
    """
    A fast way to calculate binomial coefficients by Andrew Dalke (contrib).
    """
    if 0 <= k <= n:
        ntok = 1
        ktok = 1
        for t in xrange(1, min(k, n - k) + 1):
            ntok *= n
            ktok *= t
            n -= 1
        return ntok // ktok
    else:
        return 0
