import sys
import random

def writeNow(string):
	sys.stdout.write(string)
	sys.stdout.flush()

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
