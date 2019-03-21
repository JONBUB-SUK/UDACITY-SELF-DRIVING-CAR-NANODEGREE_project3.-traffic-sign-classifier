# UDACITY-SELF-DRIVING-CAR-NANODEGREE_project3.-traffic-sign-classifier

[//]: # (Image References)

[image1-1]: ./images/2-train-dataset-graph.png "HISTOGRAM OF TRAIN DATA"
[image1-2]: ./images/3-test-dataset-graph.png "HISTOGRAM OF TEST DATA"
[image1-3]: ./images/4-valid-dataset-graph.png "HISTOGRAM OF VALID DATA"

[image2]: ./images/결과.png "RESULT OF PREDICTION"





# Introduction
This project onject is making traffic sign classifier by using tensorflow and accuracy of validation set have to upper than 93%

Traffic sign data set is given by 'German Traffic Sign Dataset'

The dataset is already devided into 3 parts

```train.p``` : 34799 images

```test.p``` : 12630 images

```validation.p``` : 4410 images

Each set have 43 classes, that means there are 43 sort of traffic signs

And shape of each images is (32,32,3), that means image size is (32,32) and it has RGB channels

This is histogram of each dataset classes distribution

![alt text][image1-1]

![alt text][image1-2]

![alt text][image1-3]

(사진 3개  출처 : Ryein Goddard github)


I defined LeNet function that has already knwon architecture to train model

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Convolution 5x5x3     	| 1x1 stride, valid padding, outputs 28x28x20 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x20 				|
| Convolution 5x5	    | 2x2 stride, valid padding, outputs 10x10x50    |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x50 				|
| Flatten  	    |       outputs 1x1x1250    |
| RELU					|												|
| Fully connected		| input 1250, output 200        									|
| RELU					|												|     									|
| Fully connected		| input 200, output 100        									|
| RELU					|												|    									|
| Fully connected		| input 100, output 43        									|

And finally I tested this model to classify 5 really German traffic sign download at google image


# Background Learning



# Results

* validation set accuracy of 94.9%
* test set accuracy of 92.6%

I used this model to predict new traffic sign images downloaded at goole

I searched German traffic sign, thats because I trained by german traffic sign images

Below is result of prediction

(사진으로 설명 : 5장의 실제 표지판과 예측한 표지판의 비교)





# Conclusion & Discussion

As a result, although accuracy of test set was almost 93%, and I thought it was enough high to predict new traffic sign,

my model was not good at prediction for new images

Below is the reason I thought why prediction for new images is not good

1. I had to preprocess train images like turn it to gray scale

Because I trained using RGB images, that means it has 3 times diversities, complexities more than gray scale images

So it maybe more difficult to apply new one

2. Like the preceding, new german traffic sign 



Drop out 은 오히려 성능이 안좋아 지더라

sub sampling 으로 max pooling 이 적절했다

여러번 튜닝해본 결과 12epoch 이 과적합을 피할 수 있었다


