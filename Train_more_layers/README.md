
![](docs/_static/Deeper_architecture.png)
# Deeper architecture 

In order to improve F1-score from previous models, we unfroze additional deeper layers from the [VGG16](./Histo_train_moreLayers_VGG16.ipynb) and [EffNetB4](./Histo_train_moreLayers_EffnetB4.ipynb) models for training: one or two convolutional layers for the [VGG16](./VGG_train_more_layers.png) model and batch normalization and convolutional layers for [EffNetB4](./ENB4_train_more_layers.png) model, respectively.

To avoid overfitting when increasing the number of trainable parameters, we expanded the training dataset sevenfold by adding deterministic augmentations as brightened and blurred versions of the images (blurring filters with (2,2) and (3,3) kernel sizes and adjusting brightness to 0.8 and 1.3 and combinations of both blurring with kernel size of (3,3) and change of brightness to 0.8 and 1.3) added to original dataset and also applied additional random augmentations such as random rotation, flipping, and changes in brightness during training with “ImageDataGenerator” function during training.  

The results for best models are summarized in [VGG16](./VGG_train_more_layers.png) and [EffNetB4](./ENB4_train_more_layers.png).