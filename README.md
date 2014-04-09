## Premise


The concep is to measure the effect of priming on humans engaged in an image-
labelling task.  Obviously priming does occur.  But we want to quantify it.
To what degree are people affected by the images they just labelled?

Another way that people can be primed is by what they are told as they prepare 
to engage the task.
For example, if they are told the funding agency behind the research, this will
likely lead them to assume a particular intent, and they will perhaps focus on
specific aspects of the content.

One interesting comparison to make is between these two types of priming.  
That is, between priming due to earlier sub-tasks, and priming due to something
in the *explanation context* that precedes the task. 

We imagine that the *explanation context* priming is both more severe, and
more avoidable.  But is this the case? Here we put these two priming treatments
head-to-head to compare their impact on a population of AMT workers who are
engaged in labeling pictures.

## First Experiment (week of 24 March 2014) ##

We collected a four sets of pictures, in the folder `/data/images/set_1`.
These images follow a naming convention that identifies subsets used to set
up the various experimental treatments.  Here's an explanation of the subsets
and the naming convention:

- `test_#.jpg`:
  These are images of food, chosen because they also have identifiable 
  cultural signal (e.g. cultural symbols are also in the frame and / or
  the food is recognizeably associated to a particular culture).  Labellers
  might focus on aspects of culture, but they will probably also focus on
  describing the objective contents of the food.  This set is a test set,
  designed such that the task of labeling them can be approached by focusing
  on different aspects of their content.

- `cultural_#.jpg`:
  These are images chosen because they are very obviously tied to a specific
  culture and because the cultural content of the image is predominant.  This
  subset serves to prime users to look for cultural elements.

- `ingredients_#.jpg`:
  These are images chosen because they focus on the particular ingredients of
  food, and because they have no obvious connection to particular cultures.  
  Generally we see ingredients separated out, as if someone is ready
  to cook or bake.  This subset serves to prime users to look for the contents
  of food.

- `ambiguous_#.jpg`:
  These are images of prepared food, chosen because, althogh they could be
  identified as belonging to a particular culture, this identification is
  not predominant.  This subset is a control set, designed to provide neutral
  priming.

The original urls for the pictures can be found in the file 
`/data/images/image_scavenging/images3.json`.  The file 
`/data/images_in_experiment_1.txt` lists out which of the images in the latter
were actually used to construct AMT tasks.  The AMT task actually used for
experiment 1 can be found in `/src/turkHit_27032014.html`.



A brief recap of thoughts leading up to this experiment.


