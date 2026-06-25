#n script for reading train and validation data set and train the network
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.layers import Conv2D, Flatten, Input, MaxPooling2D, Activation, BatchNormalization, Concatenate, Dropout, Add, Dense,LeakyReLU
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2
import pandas as pd
from pathlib import Path
from tensorflow.keras.utils import plot_model
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from matplotlib import pyplot
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
from tensorflow.keras.optimizers import SGD,RMSprop
from tensorflow.keras.applications import EfficientNetB0
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input, Flatten, Dense, Lambda
from keras.models import Model
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import glob
import cv2
from pathlib import Path
from tensorflow.keras.utils import plot_model
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from keras.models import model_from_json
from plot import multi_plot
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
import keras
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
import pandas as pd
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.python.keras.layers import InputSpec, Layer

def right_angle_rotate(input_image):
    angle = random.choice([0, 90, 180, 270])
    if angle != 0:
        input_image = apply_affine_transform(
            input_image, theta=angle, fill_mode='nearest')
    return input_image


class MulticlassTruePositives(tf.keras.metrics.Metric):
    def __init__(self, name='multiclass_true_positives', **kwargs):
        super(MulticlassTruePositives, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.reshape(tf.argmax(y_pred, axis=1), shape=(-1, 1))
        values = tf.cast(y_true, 'int32') == tf.cast(y_pred, 'int32')
        values = tf.cast(values, 'float32')
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, 'float32')
            values = tf.multiply(values, sample_weight)
        self.true_positives.assign_add(tf.reduce_sum(values))

    def result(self):
        return self.true_positives

    def reset_state(self):
        # The state of the metric will be reset at the start of each epoch.
        self.true_positives.assign(0.)

##########################################################################################################################
import os

from keras.callbacks import Callback
import matplotlib.pyplot as plt
import numpy as np
from scikitplot.metrics import plot_confusion_matrix, plot_roc, plot_precision_recall_curve, plot_precision_recall
import keras
from tensorflow.keras import utils
from keras.preprocessing.image import ImageDataGenerator, apply_affine_transform
from skimage import io
import random

class PerformanceVisualizationCallback(Callback):
    def __init__(self, model, validation_data, image_dir):
        super().__init__()
        self.model = model
        self.validation_data = validation_data
        
        os.makedirs(image_dir, exist_ok=True)
        self.image_dir = image_dir

    def on_epoch_end(self, epoch, logs={}):
        y_pred = np.asarray(self.model.predict(self.validation_data[0]))
        y_true = self.validation_data[1]             
        y_pred_class = np.argmax(y_pred, axis=1)

        # plot and save confusion matrix
        fig, ax = plt.subplots(figsize=(16,12))
        plot_confusion_matrix(y_true, y_pred_class,labels=[0,1,2],true_labels=[0,1,2], ax=ax)
        fig.savefig(os.path.join(self.image_dir, f'confusion_matrix_epoch_{epoch}'))

from tensorflow.keras.applications import EfficientNetB0,EfficientNetB4,EfficientNetB7


################################################################################
# Model definition for classification

def pretrained_model(img_shape, num_classes,layer_type):
    model_vgg16_conv = VGG16(weights='imagenet', include_top=False)

    #Input format
    keras_input = Input(shape=(224,224,3), name = 'image_input')
    
    #Use the generated model 
    output_vgg16_conv = model_vgg16_conv(keras_input)
    
    for layer in model_vgg16_conv.layers:
        layer.trainable = False
    model_vgg16_conv.layers[-2].trainable = True  # First conv layer is unfreezed
    model_vgg16_conv.layers[-3].trainable = True  # Second conv layer is unfreezed 
   

    print(model_vgg16_conv.layers[-1].trainable)

    #Add the fully-connected layers 
    x = Flatten(name='flatten')(output_vgg16_conv)
    x = Dense(128, activation=layer_type, name='fc1')(x)
    x = Dense(128, activation=layer_type, name='fc2')(x)
    x = Dropout(0.5, name='dropout')(x)
    x = Dense(num_classes, activation='softmax', name='predictions')(x)
    
    METRICS = [
      keras.metrics.SparseCategoricalAccuracy(),
      MulticlassTruePositives()]
    
    
    #Create your own model 
    pretrained_model = Model(inputs=keras_input, outputs=x)
    pretrained_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=METRICS)
    pretrained_model.summary()

    return pretrained_model








max_score = 48
min_score  = 2 


low = (10-(min_score))/(max_score-min_score)
#10-(min_score)/(max_score-min_score)
mid = (25-(min_score))/(max_score-min_score)

#25-(min_score)/(max_score-min_score)  ################################################################################
# Training data (total_score)


val_accuracy_dict  = {}
train_accuracy_dict = {}
val_loss_dict = {}
train_loss_dict = {}
validation_slide = ['1st_part','2nd_part','3rd_part']
for val_s in validation_slide:
    val_dir  = val_s

    train_DATA_PATH = '../analysed_data/All_samples_group_1_2_3_total_score_Median_aug/'+val_dir+'_val_total_score_cropped/train_aug/'
    validation_DATA_PATH = '../analysed_data/All_samples_group_1_2_3_total_score_Median_aug/'+val_dir+'_val_total_score_cropped/validation/'

    input_shape = (224,224,3)
	#classes = ['low','medium','high']

    train_images      = glob.glob(train_DATA_PATH+'*.tif')
    validation_images = glob.glob(validation_DATA_PATH+'*.tif')

    n_t= len(train_images)
    print('hiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii---train data size is {}'.format(n_t))
    train_samples = np.zeros((n_t, input_shape[0], input_shape[1], 3), dtype=np.float32)
    train_label = np.zeros((n_t,), int)


    with open('../analysed_data/All_samples_group_1_2_3_total_score_Median_aug/'+val_dir+'_val_total_score/train/train_newlabels_aug_corrected.txt','r') as f:
        for ctr,line in zip(range(n_t),f):
            if cv2.imread(train_DATA_PATH+(line.split()[0])) is not None:
                train_samples[ctr,:,:,:] = cv2.imread(train_DATA_PATH+(line.split()[0]))
                la = float(line.split(' ')[1])
                if la > mid:
                    train_label[ctr] = 2
                elif la > low:
                    train_label[ctr] = 1
                else:
                    train_label[ctr] = 0
            else:
               print(line.split(' ')[0],'is None')
    #train_label = tf.one_hot(train_label,depth=3)
    for i in range(n_t):
        mod_img = train_samples[i,:, :].astype(np.float32)
        train_samples[i, :, :, :] = mod_img

    #print(samples.dtype)
    train_samples = np.clip(train_samples, 0, 255)
    #train_samples /= 255
    #train_samples -= 0.5
    print(val_dir,'trainnnnn  is ddoooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooone!!')  
    
    ################################################################################
    # Validation dataset
    n_v= len(validation_images)
    print('hiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiivalidation data size is {}'.format(n_v))
    validation_samples = np.zeros((n_v, input_shape[0], input_shape[1], 3), dtype=np.float32)
    validation_label = np.zeros((n_v,), int)

    with open('../analysed_data/All_samples_group_1_2_3_total_score_Median_aug/'+val_dir+'_val_total_score/validation/'+'validation_newlabels.txt','r') as f:
        for ctr,line in zip(range(n_v),f):
            if cv2.imread(validation_DATA_PATH+(line.split()[0])) is not None:
                validation_samples[ctr,:,:,:] = cv2.imread(validation_DATA_PATH+(line.split()[0]))
                la = float(line.split()[1])
                if la > mid:
                    validation_label[ctr] = 2
                elif la > low:
                    validation_label[ctr] = 1
                else:
                    validation_label[ctr] = 0
            else:
                print(line.split(' ')[0],'is Noneeee')

    #validation_label = tf.one_hot(validation_label,depth = 3)  

    print(val_dir,'is ddoooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooone!!')  
    print(validation_label)      
    for i in range(n_v):
        mod_img = validation_samples[i,:, :].astype(np.float32)
        validation_samples[i, :, :, :] = mod_img

    validation_samples = np.clip(validation_samples, 0, 255)
    #validation_samples /= 255
    #validation_samples -= 0.5

    ################################################################################
    datagen = ImageDataGenerator(
        horizontal_flip=True, 
        vertical_flip=True,
        brightness_range=(0.8, 1.3),
        fill_mode='nearest',
        preprocessing_function=right_angle_rotate,
        validation_split=False)
       
        


    #############################################################
	## Training the model
    model = pretrained_model(train_samples.shape[1:], 3,'relu') ## 
    model.summary()

    for layer in model.layers:
        print(layer, layer.trainable)
   
    mcp_save = ModelCheckpoint(val_dir+'_best_weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', mode='min')

    performance_viz_cbk = PerformanceVisualizationCallback(
                                       model=model,
                                       validation_data=(validation_samples, validation_label),
                                       image_dir=val_dir+'_perorfmance_charts')

    my_callbacks = [
    #tf.keras.callbacks.EarlyStopping(patience=2),
    tf.keras.callbacks.ModelCheckpoint(filepath=val_dir+'_best_weights.{epoch:02d}-{val_loss:.2f}.hdf5'),
    tf.keras.callbacks.TensorBoard(log_dir=val_dir+'_logs'),
    performance_viz_cbk]
      
  
    hist = model.fit(datagen.flow(train_samples, train_label, batch_size=512),epochs=25, callbacks=my_callbacks, validation_data=(validation_samples, validation_label),workers=4,use_multiprocessing=True)
    

    print("Training done for validation set of <<{}>>!".format(val_dir))
    model.save_weights('final_weights_'+val_dir+'.h5')
    df = pd.DataFrame.from_dict(hist.history, orient="index")
    df.to_csv(val_dir+"_raw.csv")
        #############################################
    print(hist.history.keys())

    from plot import multi_plot
   
    multi_plot(None, [hist.history['loss'], hist.history['val_loss']],val_dir+'Loss.png', 'Epochs', 'Loss', legend=['Training', 'Validation'])
    multi_plot(None, [hist.history['multiclass_true_positives'], hist.history['val_multiclass_true_positives']],val_dir+'TP.png', 'Epochs', 'True positive', legend=['Training', 'Validation'])
    multi_plot(None, [hist.history['sparse_categorical_accuracy'], hist.history['val_sparse_categorical_accuracy']],val_dir+'_ACC.png', 'Epochs', 'Accuracy', legend=['Training', 'Validation'])


#################################################################################################
    
######################################################################################
        
    
        # fits the model on batches with real-time data augmentation:
   





