import matplotlib.pyplot as plt
from keras.models import *
from keras.layers import *
from keras.applications import *
from keras.preprocessing.image import *
from keras.callbacks import *
from keras.optimizers import *
from keras.utils import *


def build_model(MODEL, image_size, train_data_dir, valid_data_dir, lambda_func=None):
    #构造模型
    width = image_size[0]
    height = image_size[1]
    x_input = Input((height, width, 3))
    if lambda_func:
        x_input = Lambda(lambda_func)(x_input)
    
    base_model = MODEL(input_tensor=x_input, weights='imagenet', include_top=False, pooling = 'avg')
        
    x = Dropout(0.5)(base_model.output)
    x = Dense(1, activation='sigmoid',kernel_regularizer=regularizers.l2(0.001))(x)
    model = Model(base_model.input, x)
    adam = optimizers.Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer=adam,
             loss='binary_crossentropy',
             metrics=['accuracy'])
    
    gen = ImageDataGenerator(rotation_range=40,  #旋转数据增强
                            width_shift_range=0.2,
                            height_shift_range=0.2,
                            shear_range=0.2,
                            zoom_range=0.2,
                            horizontal_flip=True)
    val_gen = ImageDataGenerator()
    train_generator = gen.flow_from_directory(train_data_dir, (height, width), shuffle=True, 
                                              batch_size=64,class_mode='binary')
    valid_generator = val_gen.flow_from_directory(valid_data_dir, (height, width), shuffle=True, 
                                              batch_size=32,class_mode='binary')
    
    return model,train_generator,valid_generator
    
def show_learning_curve(history):
    plt.figure(1)  
       
    # summarize history for accuracy  
    plt.subplot(211)  
    plt.plot(history.history['acc'])  
    plt.plot(history.history['val_acc'])  
    plt.title('model accuracy')  
    plt.ylabel('accuracy')  
    plt.xlabel('epoch')  
    plt.legend(['train', 'test'], loc='upper left')  
      
    # summarize history for loss  
      
    plt.subplot(212)  
    plt.plot(history.history['loss'])  
    plt.plot(history.history['val_loss'])  
    plt.title('model loss')  
    plt.ylabel('loss')  
    plt.xlabel('epoch')  
    plt.legend(['train', 'test'], loc='upper left')  
    plt.show()  

def lock_layers(model, locked_layer_nums):
    for i in range(len(model.layers)):
        print(i,model.layers[i].name)

    for layer in model.layers[:locked_layer_nums]:  #冻结前N层
        layer.trainable = False
    
def predict_on_modle(n, width, heigth, test_data_dir, model, weight, output_name):
    x_test = np.zeros((n,width,heigth,3),dtype=np.uint8)

    for i in tqdm(range(n)):
    #for i in range(n):
        img = load_img(test_data_dir+"/test/"+'/%d.jpg' % (i+1)) 
        x_test[i,:,:,:] = img_to_array(img.resize((width,heigth),Image.ANTIALIAS))
    
#     x_test = xception.preprocess_input(x_test)
    model.load_weights(weight)
    y_test = model.predict(x_test, verbose=1)
    y_test = y_test.clip(min=0.005, max=0.995)
    
    df = pd.read_csv("sample_submission.csv")
    for i in tqdm(range(n)):
        df.set_value(i, 'label', y_test[i])
    df.to_csv(output_name, index=None)
    df.head(10)