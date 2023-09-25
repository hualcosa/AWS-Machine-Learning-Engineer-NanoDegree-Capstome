# Inventory Monitoring at Distribution Centers

Advances in Artificial Intelligence, especially in computer vision, allowed novel approaches for monitoring inventory in real time. This project is about one of them: Imagine  warehouses where robots put and pick items to be delivered in bins. These robots have cameras attached to them and take photos of these bins. The photos are then uploaded to the cloud and are the inputs to a machine-learning pipeline that classifies the amount of objects present in the bins. This output, paired with other metadata like SKU names, can be sent to the company's inventory system to update the product stock. The described situation is approximately what happens at Amazon's distribution center, and this project will use an Image dataset provided by them.
<br>
The project aims to develop a machine-learning pipeline that leverages AWS services to build an Image Classifier that counts the amount of objects present in the distribution center's bins. Besides that,  profiling and debugging will be performed, in order to evaluate how the pipeline could be improved.

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

This dataset is provided by Amazon and is located in a Publicly acessible S3 Bucket.<br><br>
![image](https://github.com/hualcosa/AWS-Machine-Learning-Engineer-NanoDegree-Capstome/assets/46836901/bc4a83aa-7600-4235-b5d8-1f8fb6c93394)

## Data preprocessing
I have used 80% data for training, 10% test and 10% validation. The data must be split into train, valid, and test folders, so It can be properly used by sagemaker estimators
![image](https://github.com/hualcosa/AWS-Machine-Learning-Engineer-NanoDegree-Capstome/assets/46836901/b3d68bdf-ba7b-45d8-9376-0b292ea4585b)

## Upload to S3
Data can be easily uploaded to an s3 bucket using AWS CLI: 
![image](https://github.com/hualcosa/AWS-Machine-Learning-Engineer-NanoDegree-Capstome/assets/46836901/70490e1e-95a0-45d0-8d0a-8d8d2da2236c)

# Model Training
Before starting the training itself, it is necessary to set hyperparameters values, debugger and profiler rules and configurations:
<br><br>
![image](https://github.com/hualcosa/AWS-Machine-Learning-Engineer-NanoDegree-Capstome/assets/46836901/fe870474-9949-4f63-8e42-aee99587933c)

The training code(entry point) is located in a file called train.py. In this file, you can find the model definitions, train and test loops, and data augmentation and normalization pipelines

### model architecture
![image](https://github.com/hualcosa/AWS-Machine-Learning-Engineer-NanoDegree-Capstome/assets/46836901/dfc5f99b-a636-4cf9-8ec9-74d8a78212a1)

### Data transformation pipelines
![image](https://github.com/hualcosa/AWS-Machine-Learning-Engineer-NanoDegree-Capstome/assets/46836901/b7aab12e-bd2b-49a6-b8a6-90a7dedf434d)

### Creating and fitting an estimator:

![image](https://github.com/hualcosa/AWS-Machine-Learning-Engineer-NanoDegree-Capstome/assets/46836901/7c6b98f8-dc95-4812-96ef-f849bacf0c1f)

As you can see in the above screenshots, images in the validation and testing sets are only resized and cropped, but images in the training set go through some data augmentations, like rotation and HorizontalFlip, that can increase the model's generalization capabilities. Note also that I am performing Multi-instance training because the job is run in two ml.c5.xlarge instances. I am using a pre-trained resnet50 model, and I changed the last part of the network to suit the problem at hand: The final layer has 5 output neurons since our images can have 1-5 objects in it.<br>

After the job is completed, we have the following outputs:
<br>
![image](https://github.com/hualcosa/AWS-Machine-Learning-Engineer-NanoDegree-Capstome/assets/46836901/e79ca133-1bea-43c2-a15e-55d7a2ea34f1)
![image](https://github.com/hualcosa/AWS-Machine-Learning-Engineer-NanoDegree-Capstome/assets/46836901/bf806506-87de-4964-a139-d33c6e4de398)

# Training results
As can be seen, The model does not do a great job at classifying the images. The test set accuracy is only about 27%. Resnet is mainly a feature extractor model. In order to achieve higher performance, a more complex architecture like FasterRCNN should be used, once identifying the number of objects in an image is closer to an object detection task. Since the focus of this project is not on producing a highly accurate model, but rather on constructing the training pipeline using Sagemaker, I will leave FasterRCNN transfer-learning as an exercise for the reader. 

# Profilling and Debugging
In order to evaluate possible improvements to the training pipeline, I have attached debugger and training rules to the estimator (see the above prints).
Sagemaker debugger allows us to monitor the weights, biases, and gradients tensors through the training process:
<br><br>
![image](https://github.com/hualcosa/AWS-Machine-Learning-Engineer-NanoDegree-Capstome/assets/46836901/f091c21b-d71f-4a44-8bd0-c8071c9cc556)

Accessing the tensors, we can, for instance, plot train and validation losses along the optimization steps:
<br><br>
![image](https://github.com/hualcosa/AWS-Machine-Learning-Engineer-NanoDegree-Capstome/assets/46836901/a2372020-4410-4af2-ad69-1c70381a6f00)

Sagemaker profiler generates a report  saved in the same S3 bucket where the training model artifacts are saved. You can download it 
and check whether any of the alerts were triggered during the training process:
<br><br>
![image](https://github.com/hualcosa/AWS-Machine-Learning-Engineer-NanoDegree-Capstome/assets/46836901/5b0bb317-e54a-484b-b7a0-0fb26e98b4ad)

# Cost Analysis
An important point in industrialized machine-learning applications is to use the proper resources. Overprovisioning resources may lead to a substantial amount of money wasted. I monitored the instance metrics during the training process and found the following:

![image](https://github.com/hualcosa/AWS-Machine-Learning-Engineer-NanoDegree-Capstome/assets/46836901/a22e5dd3-4441-434b-893c-fc9d6ed8fd2b)

We can see that the maximum workloads for disk, CPU, and memory utilization were 2.21%, 20.1%, and 20.7%, respectively. These metrics point out that we could run our training jobs in smaller, cheaper instances if we were to orchestrate such a pipeline to run, for instance, once a week. Doing this would save some costs and make our bosses happier!
