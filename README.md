# ECE 579 - Intelligent Systems

## Project Initiative Report (02/05/2026)

### Project Title: 
A Machine Learning Algorithm for Classification of Music by Genre

### Students in the project group:

Jakub Wittrock - Neural Network Architecture Design and Research 
Ben Pollatz - Data Preprocessing and Validation
John Soltis - Test Validation and Results Analysis 

### Project Description: 
In this project we will develop an algorithm that will classify music genres. We intend to train several neural networks using MEL spectrogram representations of the songs in our dataset. We will perform a comparative study between our own custom neural network architectures and models that we train taking advantage of transfer learning. Our deliverables for this project are a set of neutral networks trained to classify songs by genre and the results of the comparative study between trained networks.

### Data Description 
Type of Data: We will be using audio data for this project, our primary dataset will be the GTZAN dataset which we will source from kaggle. If we require further samples we will use the Free Music Archive (FMA) as an additional source of data. The links to both datasets can be found below.

### Data Sets: 

- D1: GTZAN: https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification
- D2: FMA:  https://github.com/mdeff/fma 

### Data Sets Size:
- D1: 1,000 samples
- D2: 106,574 samples

Total = 107,574 samples

### Number of Attributes and Classes: For each dataset, format is <attributes,classes>
- D1: <58,10>
- D2: <518,161>

### Additional Information:

#### D1 
The GTZAN dataset can be likened to being the MNIST dataset for sounds as it contains highly usable song samples belonging to 10 classes or genres. Each class contains 100 samples, each of which are 30 second clips. The 10 classes are as follows: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, and rock. This dataset also provides helpful mel spectrograms for each of the song samples which show how the spectrum of frequencies vary with time. Each song is mapped to its own spectrogram which will be helpful in training our deep neural networks.

#### D2 
The FMA dataset is much larger than the GTZAN dataset, it has four versions that we could use to supplement the GTZAN dataset. It has a small version which contains 8 classes of 1000 samples each. It has a medium version which contains 16 unbalanced classes and 25,000 samples. It has a large version that contains 161 unbalanced classes and 106,574 samples. To this point all the samples are 30 second audio clips. The full dataset contains 161 unbalanced classes and 106,574 untrimmed samples. Unlike the GTZAN dataset, the FMA does not have pregenerated mel spectrograms. We would have to generate our own spectrograms for any of the samples that we use from this dataset. If we do use mel to supplement the GTZAN we will use the small or the medium datasets due to resource and timing constraints.

## Project Status Update (xx/xx/xx):


## Technical Survey (xx/xx/xx)
