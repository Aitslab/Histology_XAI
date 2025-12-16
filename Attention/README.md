
![](../docs/_static/Attention.png)

# Attention 


Attention mechanism with convolutional and locally connected layers are added to the main architecture. The convolutional layers were utilized as filters to extract features from images. This then refined to attention weights by a ‘locally Connected’ layer with ‘sigmoid’ activation function. The attention weights then scaled and multiplied with image data. The masked features and the scaled attention weights were tracked in each epoch of training.  Randomly selected attention maps for the best model for each fold were inspected manually. The architecture of models with attention layers has following layers. 

✅ Model with attention Architecture & Trainable Parameters
1. Input Layer
Trainable: No (input layers have no parameters)

2. Base models (Pretrained, include_top=False)
All convolutional blocks of base (Conv + ReLU + MaxPool)
Trainable: ❌ All frozen
base_pretrained_model.trainable = False
Trainable params: 0
Outputs: Feature map with depth = pt_depth (typically 512 for VGG16)

3. BatchNormalization
BatchNormalization()(pt_features)
Trainable: ✔️ (BN has gamma & beta parameters)
Trainable params: Yes (small number relative to rest)

4. Attention Mechanism

    4.1 First attention conv
    Conv2D(64, (1,1), activation='relu')
    Trainable: ✔️
    Params: (1×1×pt_depth×64) + 64

    4.2 Second attention conv
    Conv2D(16, (1,1), activation='relu')
    Trainable: ✔️
    Params: (1×1×64×16) + 16

    4.3 LocallyConnected2D
    LocallyConnected2D(1, kernel_size=(1,1), activation='sigmoid')
    Trainable: ✔️
Params: depends on spatial size (locally connected = no weight sharing)

5. Channel Expansion Layer
Up-projection conv
Conv2D(pt_depth, (1,1), use_bias=False, weights=[up_c2_w])
Trainable: ❌ (explicitly frozen)
Params: 0 trainable (weights are fixed to 1s)

6. Masking & Pooling
multiply([attn_layer, bn_features]) → no parameters
GlobalAveragePooling2D() for mask and features → no parameters

7. Rescaling Layer
Lambda layer dividing GAP features by GAP mask
Trainable: ❌ (no parameters)

8. Dropout Layers
Dropout(0.5) and Dropout(0.25)
Trainable: ❌ (no trainable params)

9. Fully Connected Layers
Dense(1024, activation='elu')
Trainable: ✔️
Params:
1024 × feature_dim + 1024

10. Output Dense(3, linear)
Trainable: ✔️
Params:
3 × 1024 + 3


The [Training](./Training/Attention.py) code and [visualization](./Attention_visualization.ipynb) of sample attention maps are described. results of models with adding attention layers are also 
discussed [here](./Attention_results_table.ipynb) and [here](./Attention_results_table_EffNet.ipynb). 
