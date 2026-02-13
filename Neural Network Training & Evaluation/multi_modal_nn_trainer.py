#Helper Imports 
import os

#My Imports 
from MultiModal import BimodalNN

#-----------------------------------------------------------
# Main 
#-----------------------------------------------------------
def main():

    #Configure Paths for Training
    BASE_PATH = r"C:\Users\JTWit\Documents\ECE 579\Datasets\Split GTZAN Dataset 3s"
    LIBROSA_PATH = r"C:\Users\JTWit\Documents\ECE 579\Datasets\GTZAN Dataset\audio_features.csv"
    TEST_PATH = os.path.join(BASE_PATH,'test')
    TRAIN_PATH = os.path.join(BASE_PATH,'train')
    SAVE_DIR = os.path.join(r"C:\Users\JTWit\Documents\ECE 579","Custom DNN Models")

    #Specify training hyperparameters
    EPOCHS=50
    LEARNING_RATE=1e-4
    VAL_SPLIT=0.2
    BATCH_SIZE=32

    #Create an object of the Bimodal Neural Network Class
    net = BimodalNN(BASE_PATH=BASE_PATH,LIBROSA_PATH=LIBROSA_PATH, SAVE_DIR = SAVE_DIR )

    #Train the neural network
    net.train(model_name='bimodal network',EPOCHS=EPOCHS,LEARNING_RATE=LEARNING_RATE,VAL_SPLIT=VAL_SPLIT,BATCH_SIZE=BATCH_SIZE)

    #Evaluate the neural network
    net.evaluate()

    #Plot the confusion matrix
    net.confusion_matrix()

#-----------------------------------------------------------
# Main Guard
#-----------------------------------------------------------
if __name__ =='__main__':
    main()