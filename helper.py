import matplotlib.pyplot as plt
from keras.models import *
from keras.layers import *
from keras.applications import *
from keras.preprocessing.image import *
from keras.callbacks import *
from keras.optimizers import *
from keras.utils import *
from tqdm import tqdm   #进度条
from PIL import Image
import pandas as pd
import cv2

#image_dir含dog，cat子目录
def read_images_to_memory(image_dir, width, height, test=False):
    dog_filenames = os.listdir(image_dir+'/dog') #dog
    cat_filenames = os.listdir(image_dir+'/cat') #cat
    print("dog_nums={}, cat_nums={}".format(len(dog_filenames), len(cat_filenames)))
    if test:
        n=(len(dog_filenames)+len(cat_filenames))*2//100  #取5%的数据
    else:
        n=len(dog_filenames)+len(cat_filenames)
    print("total images:", n)
    X = np.zeros((n, width, height, 3), dtype=np.uint8)
    Y = np.zeros(n, dtype=np.uint8)
    
    i=0
    for filename in tqdm(dog_filenames):
        fullname=image_dir+'/dog/'+filename
        img = cv2.imread(fullname)  
        X[i] = cv2.resize(img, (width, height))  
        Y[i] = 1
        i=i+1
        if test:
            if i>n//2:
                break
    
    for filename in tqdm(cat_filenames):
        fullname=image_dir+'/cat/'+filename
        img = cv2.imread(fullname)  
        X[i] = cv2.resize(img, (width, height))  
        Y[i] = 0
        i=i+1
        
        if test:
            if i==n:
                break
    return X, Y


def load_test_data(n, width, heigth, test_data_dir):
    x_test = np.zeros((n,width,heigth,3),dtype=np.uint8)

    for i in tqdm(range(n)):
        img = load_img(test_data_dir+"/test/"+'/%d.jpg' % (i+1)) 
        x_test[i,:,:,:] = img_to_array(img.resize((width,heigth),Image.ANTIALIAS))
    
    return x_test

    
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
        model.layers[i].trainable = True  #刚开始没有这行处理，导致lock_layers有误
        
    for layer in model.layers[:locked_layer_nums]:  #冻结前N层
        layer.trainable = False   

def predict_on_model(x_test, model, weight, output_name):
    n=len(x_test)
    
    model.load_weights(weight)
    y_test = model.predict(x_test, verbose=1)
    y_test = y_test.clip(min=0.005, max=0.995)
    
    df = pd.read_csv("sample_submission.csv")
    for i in tqdm(range(n)):
        df.set_value(i, 'label', y_test[i])
    df.to_csv(output_name, index=None)


def predict_on_model2(x_test, test_dir, batchSize, model, weight, output_name):
    n=len(x_test)
    
    model.load_weights(weight)
    y_test = model.predict(x_test, verbose=1)
    y_test = y_test.clip(min=0.005, max=0.995)
    
    gen = ImageDataGenerator()
    test_generator = gen.flow_from_directory(test_dir, (224, 224), shuffle=False, 
                                         batch_size=batchSize, class_mode=None)

    df = pd.read_csv("sample_submission.csv")
    for i, fname in enumerate(test_generator.filenames):
        index = int(fname[fname.rfind('/')+1:fname.rfind('.')])
        df.set_value(index-1, 'label', y_test[i])
    
    df.to_csv(output_name, index=None)