import sys
import analysis as a

COMPARISONS = {
	'task1': ((2,[0]), (2,[5])),
	'frame1': ((2,[10]), (2,[11])),
	'echo': ((2,[12]), (2,[13])),
	'task2': ((1,[0]), (1,[1])),
	'frame2': ((1,[3]), (1,[5])),
}

def bootstrap_specificity(comparison):
	spec1, spec2 = COMPARISONS[comparison]

	a.bootstrap_relative_specificity(
		spec1, 
		spec2, 
		images=[5],
		num_bootstraps=1000,
		resample_size=119,
		fname=('data/new_data/specificity/specificity_%s.json'%comparison)
	)

if __name__ == '__main__':
	treatment_pair = sys.argv[1]
	bootstrap_specificity(treatment_pair)
