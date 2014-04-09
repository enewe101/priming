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


