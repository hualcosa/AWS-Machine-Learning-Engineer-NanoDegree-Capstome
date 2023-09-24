**NOTE:** This file is a template that you can use to create the README for your project. The **TODO** comments below will highlight the information you should be sure to include.

# Inventory Monitoring at Distribution Centers

Advances in Artificial Intelligence, specially in computer vision, allowed novel approaches for monitoring inventory in real time. This project is about one of them: Imagine  warehouses were robots put and pick itens to be delivered in bins. These robots have cameras attached to them, and take photos of theses bins. The photos are then uploaded to the cloud and are the inputs to a machine learning pipeline that classify the amount of objects present in the bins. This output paired with other metadata like SKUs name, can be send to the companie's inventory system to update the product stock. The described situation is approximately what happens at Amazon's distribution center, and this project will use an Image dataset provided by them.
<br>
The goal of the project is to develop a machine learning pipeline that leverages AWS services to build and Image Classifier that counts the amount of objects present in the distribution center's bins. Besides that,  profilling and debugging will be performed, in order to evaluate how the pipeline could be improved


## Project Set Up and Installation
1. Create a Notebook instance in Sagemaker, or a new Sagemaker Studio;
2. Clone this repository and follow the sagemaker.ipynb notebook

## Dataset
The Amazon Bin Image Dataset contains over 500,000 images and metadata from bins of a pod in an operating Amazon Fulfillment Center. The bin images in this dataset are captured as robot units carry pods as part of normal Amazon Fulfillment Center operations.
We are working with a subset of this data tha contains:
- 1228 images with 1 objects
- 2299 images with 2 objects
- 2666 images with 3 objects
- 2373 images with 4 objects
- 1875 images with 5 objects

This dataset is provided by Amazon and is located in a Publicly acessible S3 Bucket.
![image](https://github.com/hualcosa/AWS-Machine-Learning-Engineer-NanoDegree-Capstome/assets/46836901/bc4a83aa-7600-4235-b5d8-1f8fb6c93394)

## data preprocessing
I have used 80% data for training, 10% test and 10% validation. The data must be split into train, valid, and test folders, so It can be properly used by sagemaker estimators
![image](https://github.com/hualcosa/AWS-Machine-Learning-Engineer-NanoDegree-Capstome/assets/46836901/b3d68bdf-ba7b-45d8-9376-0b292ea4585b)

## upload to S3
Data can be easily uploaded to an s3 bucket using AWS CLI: 
![image](https://github.com/hualcosa/AWS-Machine-Learning-Engineer-NanoDegree-Capstome/assets/46836901/70490e1e-95a0-45d0-8d0a-8d8d2da2236c)

# Model Training
Before starting the training itself, it is necessary to set hyperparameters values, debugger and profiler rules and configurations:
![image](https://github.com/hualcosa/AWS-Machine-Learning-Engineer-NanoDegree-Capstome/assets/46836901/fe870474-9949-4f63-8e42-aee99587933c)

The training code(entry point) is located in a file called train.py. In this file, you can find the model definitions, train and test loops, and data augmentation and normalization pipelines

### model architecture
![image](https://github.com/hualcosa/AWS-Machine-Learning-Engineer-NanoDegree-Capstome/assets/46836901/dfc5f99b-a636-4cf9-8ec9-74d8a78212a1)

### data transformation pipelines
![image](https://github.com/hualcosa/AWS-Machine-Learning-Engineer-NanoDegree-Capstome/assets/46836901/b7aab12e-bd2b-49a6-b8a6-90a7dedf434d)

### creating and fitting an estimator:

![image](https://github.com/hualcosa/AWS-Machine-Learning-Engineer-NanoDegree-Capstome/assets/46836901/7c6b98f8-dc95-4812-96ef-f849bacf0c1f)

As you can see in the above screenshots, images in the valid and test sets are only resized and cropped, but images in the training set go through some data augmentations like rotation and HorizontalFlip, that can increase the model's generalization capabilities. Note also that I am performing Multi-instance training, because the job is run in two ml.c5.xlarge instances. I am using a pre-trained resnet50 model, and I changed the last part of the network to suit the problem at hand: The final layer has 5 output neurons since our images can have 1-5 objects in it.<br>

After the job is completed, we have the following outputs:
<br>
![image](https://github.com/hualcosa/AWS-Machine-Learning-Engineer-NanoDegree-Capstome/assets/46836901/e79ca133-1bea-43c2-a15e-55d7a2ea34f1)
![image](https://github.com/hualcosa/AWS-Machine-Learning-Engineer-NanoDegree-Capstome/assets/46836901/bf806506-87de-4964-a139-d33c6e4de398)

# Training results
As can be seen, The model does not do a great job at classfying the images. The test set accuracy is only about 27%. Resnet is mainly a feature extractor model. In order to achieve higher performance, a more complex architecture like FasterRCNN should be used, once identifying the number of objects in an image is closer to an object detection task. Since the focus of this project is not on producing a highly accurate model, but ratherthe training pipeline construction using Sagemaker, I will leave FasterRCNN transfer-learning as an exercise for the reader. 

# Profilling and Debugging

 
 

**TODO**: What kind of model did you choose for this experiment and why? Give an overview of the types of hyperparameters that you specified and why you chose them. Also remember to evaluate the performance of your model.

## Machine Learning Pipeline
**TODO:** Explain your project pipeline.

## Standout Suggestions
**TODO (Optional):** This is where you can provide information about any standout suggestions that you have attempted.
