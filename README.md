# Leaf Segmentation in field images using FCN network #

Fine tuning of pre-trained VGG16 model on CVPPP dataset using FCN8 model and testing it on some sample outdoor field images


# Steps involved #   

## Training ##
1. Creation of CVPPP dataset for training and validation
2. Trained the model for 15 epochs on **Google Colab** 
3. Example command for training : python main.py --gpu 0 --resume ""

## Testing ##
1. Took some random outdoor images from web and used [labelme](https://github.com/wkentaro/labelme) to label these images and create ground truth labelled data
2. Observed best **Mean IoU : 0.82** with the 14th epoch 
3. Example command for testing : python test.py

# Sample Outputs
Some of the predicted labels are shown in the *sample_output.png* image
