## Project: Search and Sample Return
### Writeup Template: This file can be used as a template for writeup if want to submit it as a markdown file.

=============================================================================================================================


**The steps of this project are the following:**  


===============================================================================================================================
**Training and  Calibration**  

* Download the simulator and take data in "Training Mode"
* Test out the functions in the Jupyter Notebook provided
* Add functions to detect obstacles and samples of interest for rocks
* Fill in the 'process_image()' function with the appropriate image processing steps (perspective transform, color threshold etc.) 
  to get from raw images to a map.  The 'output_image' you create in this step should demonstrate that your mapping pipeline works.
* Use 'moviepy' to process the images in your saved dataset with the 'process_image()' function.  Include the video you produce as 
  part of your submission.
==================================================================================================================================
**Autonomous Navigation / Mapping**

* Fill in the 'perception_step()' function within the 'perception.py'script with the appropriate image processing functions to create
  a map and update 'Rover()' data in the notebook.
* Fill in the 'decision_step()' function within the 'decision.py' script with conditional statements that take into consideration the 
  outputs of the 'perception_step()' in deciding how to issue throttle, brake and steering commands. 
* Iterate on your perception and decision function until your rover does a meaningful job of navigating and mapping.  
===================================================================================================================================
[//]: # (Image References)

[image1]: ./misc/rover_image.jpg
[image2]: ./calibration_images/example_grid1.jpg
[image3]: ./calibration_images/example_rock1.jpg 

## [Rubric](https://review.udacity.com/#!/rubrics/916/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

=====================================================================================================================================
### Writeup / README

1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as 
   markdown or pdf.  

=====================================================================================================================================
### Notebook Analysis

1. Run the functions provided in the notebook on test images (first with the test data provided, next on data you have recorded). 
   Add/modify functions to allow for color selection of obstacles and rock samples.
   
Here is an example of how to include an image in your writeup.


### process 

1. Populate the 'process_image()' function with the appropriate analysis steps to map pixels identifying navigable terrain, obstacles 
   and rock samples into a worldmap.  Run 'process_image()' on your test data using the 'moviepy' functions provided to create video 
   output of your result.
   
=======================================================================================================================================
![alt text][image2]
### Autonomous Navigation and Mapping

1. Fill in the 'perception_step()' and 'decision_step()' functions in the autonomous mapping scripts and an explanation is provided in 
   the writeup of how and why these functions were modified as they were.

2. Launching in autonomous mode your rover can navigate and map autonomously.  Explain your results and how you might improve them in 
   your writeup.  
========================================================================================================================================
