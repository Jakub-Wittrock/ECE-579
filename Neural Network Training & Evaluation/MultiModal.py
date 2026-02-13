#Helper Imports
import os
import pandas as pd
import pickle
import numpy as np
import cv2
from datetime import datetime as dt

#Sklearn Imports For Data Preprocessing
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import shuffle

#Tensorflow and Keras Imports
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras import layers, models, Input

#Performance Visualization Imports
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


import warnings


class BimodalNN:
    #=================================================
    # Initialization Method
    #=================================================
    def __init__(self,BASE_PATH:str,  LIBROSA_PATH:str, TEST_DIR:str = 'test', TRAIN_DIR:str = 'train',SAVE_DIR:str = 'bimodal save'):
        
        #Make Dirs
        self.BASE_PATH = BASE_PATH
        self.TRAIN_PATH = os.path.join(self.BASE_PATH,TRAIN_DIR)
        self.TEST_PATH = os.path.join(self.BASE_PATH,TEST_DIR)
        self.SAVE_DIR = SAVE_DIR

        #Make sure the save dir exists
        os.makedirs(self.SAVE_DIR,exist_ok=True)

        #Create checkpoint dirs
        checkpoint_dir = os.path.join(r"C:\Users\JTWit\Documents\ECE 579",'Training Checkpoints')
        self.checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}.weights.h5")

        # Create the directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)

        #Configure Variables
        self.model = None
        self.le = LabelEncoder()
        self.scaler = StandardScaler()
        

        #Save path to the librosa dataset 
        self.LIBROSA_PATH = LIBROSA_PATH 
        self.__read_librosa()

    
    #=================================================
    # Network Model Configuration
    #=================================================
    def __get_default_model(self,img_shape:tuple = (128,128,3), feat_shape:tuple = (57,), classes_count:int = 10)->None:
        
        #Convolutional layer fo the nerual network
        image_input = Input(shape=img_shape, name="image_input")

        x = layers.Conv2D(16, (3, 3), activation='relu')(image_input)
        x = layers.MaxPooling2D((2, 2))(x)

        x = layers.Conv2D(32, (3, 3), activation='relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.3)(x)

        x = layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)

        x = layers.Conv2D(128, (3, 3), activation='relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.2)(x)

        x = layers.Conv2D(256, (3, 3), activation='relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)


        x = layers.Flatten()(x)

        x = layers.Dense(32, activation='relu')(x)
        
        #Multilayer perceptron layer of the neural network
        feat_input = Input(shape=feat_shape, name="librosa_input")
        
        y = layers.Dense(64, activation='relu')(feat_input)
        y = layers.Dropout(0.2)(y)

        y = layers.Dense(32, activation='relu')(y)
        y = layers.Dropout(0.4)(y)

        y = layers.Dense(16, activation='relu')(y)

        #Concatenate the brances of the neural network
        combined = layers.Concatenate()([x, y])
        
        #Final output layer
        z = layers.Dense(64, activation='relu')(combined)
        output = layers.Dense(classes_count, activation='softmax')(z) # 10 genres

        #Create a model from the combined archetecture
        self.model = models.Model(inputs=[image_input, feat_input], outputs=output)
        
    def load_model(self,model)->None:
        self.model = model

    def load_from_path(self, path:str)->None:
        pass

    #=================================================
    # Prepare Data
    #=================================================
    def __prepare_data(self,dir:str):
        images = []
        features = []
        labels = []
        
        for root, dirs, files in os.walk(dir):
            for file in files:
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path)
                if img is None: continue
                img = cv2.resize(img, (128, 128)) # Resize for consistency
    
                feat = self.df.loc[self.df['image_file'] == file].values.reshape(-1)[4:61]
                
                if len(feat) > 0:
                    images.append(img)
                    features.append(feat.astype(np.float32))
                    labels.append(os.path.basename(root)) # Folder name as label

        return np.array(images), np.array(features), np.array(labels)


    def __normalize_data(self,X_img,X_feat,y ):
        y_encoded = self.le.fit_transform(y)
        X_feat_scaled = self.scaler.fit_transform(X_feat)

        X_img_norm = X_img.astype('float32') / 255.0

        return X_img_norm,X_feat_scaled,y_encoded

    def __read_librosa(self):
        self.df = pd.read_csv(self.LIBROSA_PATH)
        
    #=================================================
    # Train Neural Network
    #=================================================
    def train(self,model_name:str, EPOCHS:int = 50,LEARNING_RATE:float = 1e-3, VAL_SPLIT:float = 0.2,BATCH_SIZE:int = 16, callbacks:list = []):

        #Check to see if the model has been defined for this instance of the object
        if not self.model:
            warnings.warn("Custom model has not been loaded. Defaulting to predefined model.", UserWarning)
            self.__get_default_model()
        
        #Compile the model
        self.model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
        
        #Prepare the training data
        X_img_train, X_feat_train, y_train = self.__prepare_data(self.TRAIN_PATH)
        X_img_train_norm,  X_feat_train_scaled ,y_train_encoded = self.__normalize_data(X_img_train, X_feat_train, y_train)

        X_img_train, X_feat_train, y_train_encoded = shuffle(
            X_img_train_norm, 
            X_feat_train_scaled, 
            y_train_encoded, 
            random_state=42)   

        #Configure default callbacks if none are specified
        if not callbacks:
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,      
                patience=5,      
                min_lr=1e-6,     
                verbose=1
            )

            earlystop = EarlyStopping(
                monitor='val_accuracy',
                mode="max", 
                patience=10,     
                restore_best_weights=True 
            )

            checkpoint = ModelCheckpoint(
                filepath=self.checkpoint_prefix,
                save_weights_only=True,  
                monitor='val_loss',      
                save_best_only=False,    
                verbose=1                
            )

            callbacks = [reduce_lr,earlystop] 

        #Train the neural network model
        history = self.model.fit(
            x=[X_img_train,X_feat_train], 
            y=y_train_encoded,
            validation_split=VAL_SPLIT, 
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks
        )

        #Keep track of the training and acccuracy of the model and create the name string        
        train_accuracy = history.history['accuracy'][-1]
        val_acc = history.history['val_accuracy'][-1]
        date_str = dt.today().strftime('%Y-%m-%d')
        name_string = f"{model_name}(train accuracy = {train_accuracy:.4f}) (val accuracy = {val_acc:.4f})(date = {date_str}).keras"

        save_file = os.path.join(self.SAVE_DIR,name_string)

        # Save the model
        self.model.save(save_file)


    #=================================================
    # Saving and Loading pickle 
    #=================================================
    def pickle_compress(self,network_path:str, pickle_path:str)->None:
        
        if self.model is not None:
            self.model.save(network_path)

            # Save the Scaler (X_feat_train_scaled)
            with open('scaler.pkl', 'wb') as f:
                pickle.dump(self.scaler, f)

            # Save the LabelEncoder (le)
            with open('label_encoder.pkl', 'wb') as f:
                pickle.dump(self.le, f)

            print("All components saved successfully!")

        else:
            raise Exception("No neural network loaded to export. Load a neural network using one of the load methods")

    def set_from_pickle(self,network_path:str, pickle_path:str)->None:

        if os.path.exists(network_path):
            # Load the components
            self.model = tf.keras.models.load_model(network_path)
        else:
            raise Exception("File path specified does not exist. Verify that the path provided links to a pickle export")

        if os.path.exists(pickle_path):
            with open('scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)

            with open('label_encoder.pkl', 'rb') as f:
                self.le = pickle.load(f)
        else:
            raise Exception("File path specified does not exist. Verify that the path provided links to a pickle export")


    #=================================================
    # Make predictions with the neural network
    #=================================================
    def predict(self,image_path:str, librosa_features)->tuple:
        raise NotImplementedError("Not implemented just yet")


    #=================================================
    # Evaluate Neural Network
    #=================================================
    def evaluate(self, test_path:str = "", verbose:bool = True):

        if self.model is not None:
            
            #Check to see if a unique test patch has been specified
            if test_path:
                X_img_test, X_feat_test, y_test = self.__prepare_data(test_path)
            
            #Use the default test set 
            else:
                X_img_test, X_feat_test, y_test = self.__prepare_data(self.TEST_PATH)
            
            #Normalize the images using the scalar and the label encoder
            X_img_test_norm, X_feat_test_scaled, y_test_encoded = self.__normalize_data(X_img_test, X_feat_test, y_test)

            #Make predictions
            predictions = self.model.predict(x=[X_img_test_norm, X_feat_test_scaled])

            #Extract the predicted class labels 
            predicted_classes = np.argmax(predictions, axis=1)

            results = self.model.evaluate(
                x=[X_img_test_norm, X_feat_test_scaled], 
                y=y_test_encoded
            )
            if verbose:
                print(f"Test Loss: {results[0]}")
                print(f"Test Accuracy: {results[1] * 100:.2f}%")

            return y_test_encoded,predicted_classes
        else:
            raise Exception("Error. No model loaded. Please call a load model method to evaluate your model")


    def confusion_matrix(self, title:str = 'Music Genre Confusion Matrix')->None:
        
        y_test_encoded,predicted_classes = self.evaluate(verbose=False)

        cm = confusion_matrix(y_test_encoded, predicted_classes)

        genre_names = self.le.classes_

        plt.figure(figsize=(12, 8))
        sns.heatmap(
            cm, 
            annot=True,              
            fmt='d',                 
            cmap='Blues',            
            xticklabels=genre_names, 
            yticklabels=genre_names
        )

        plt.title(title)
        plt.ylabel('Actual Genre')
        plt.xlabel('Predicted Genre')
        plt.show()
