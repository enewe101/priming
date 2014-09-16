from collections import defaultdict
import data_processing as dp
import naive_bayes as nb

def calc_priming_differences():
	OLD_DATASET = False
	NEW_DATASET = True
	NUM_FOLDS = 125
	NUM_IMAGES = 5

	treatment_offsets = {
			'test0': [0,4,3,2,1],
			'test1': [1,0,4,3,2],
			'test2': [2,1,0,4,3],
			'test3': [3,2,1,0,4],
			'test4': [4,3,2,1,0]
	}

	results = {}

	# first read the old dataset, and make a naive bayes dataset using 
	# labels from the first image as features
	old_results = {}
	print 'calculation'
	old_results['cult_vs_ambg'] =  test_binary_classification(
		False, ['treatment0', 'treatment1'], ['test0'])

	print 'calculation'
	old_results['cult_vs_ingr'] = test_binary_classification(
		False, ['treatment1', 'treatment2'], ['test0'])

	results['old_results'] = old_results
	
	# next read the new dataset.  Calculate the distinguishability for
	# the two image primings, the 
	new_results = {}
	image_priming_results =  defaultdict(lambda:[])

	for image in ['test%d'%i for i in range(NUM_IMAGES)]:
		for treatment_offset in treatment_offsets[image]:
			print 'calculation'
			overall_accuracy = test_binary_classification(
				True, 
				['treatment%d'%treatment_offset, 
					'treatment%d'%(5+treatment_offset)], 
				[image]
			)
			image_priming_results[image].append(overall_accuracy)
	new_results['image_priming_results'] = image_priming_results

	results['new_results'] = new_results
	
	return results


def test_binary_classification(use_new_dataset, treatments, images):
	dataset = dp.readDataset(use_new_dataset)
	naive_bayes_dataset = dp.clean_dataset_adaptor(
		dataset, treatments=treatments, images=images)
	cross_validator = nb.NaiveBayesCrossValidationTester(
		naive_bayes_dataset)

	overall_accuracy = cross_validator.cross_validate()

	return overall_accuracy
