# healthy-or-unhealthy-food-cnn

This project focuses on distinguishing a images of food items using CNN's. Predicting the healthy or unhealthy food is done by android app. It will soon be integrated as front end along with the project.  

# Requirements

Food dataset. I have downloaded the 101 Food Dataset from Kaggle or you can download it from https://www.vision.ee.ethz.ch/datasets_extra/food-101/.
As the dataset contains 101 food items each containing only 1000 images per category. Transfer Learning has been used to prevent excessive overfitting in the model. The VGG 16 model for keras has been adapted. 

Details about the network architecture can be found in the following arXiv paper:
            
            Very Deep Convolutional Networks for Large-Scale Image Recognition
            K. Simonyan, A. Zisserman
            arXiv:1409.1556
            
Please cite the paper if you use the models.

Weights are available for keras 3 as 'vgg16_weights_tf_dim_ordering_tf_kernels'

# Libraries

Keras library implemented with numpy and open-cv.



# Details

Divide your dataset folder into two parts training and test folder. Each training and test folder should contain the folders with names of food-items, which then contain the images of the food.

Part 1 consists of training the model without using Transfer Learning(Bottleneck features) with training accuracy of 77% and test accuracy of 82%.
Part 2 consists of using Transfer Learning with training accuracy of 92% ad test accuracy of 87%.
