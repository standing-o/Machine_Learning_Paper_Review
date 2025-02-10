## Speech emotion recognition: Emotional models, databases, features, preprocessing methods, supporting modalities, and classifiers
- Authors : Ak{\c{c}}ay, Mehmet Berkehan and O{\u{g}}uz, Kaya
- Journal : Speech Communication
- Year : 2020
- Link : https://www.sciencedirect.com/science/article/pii/S0167639319302262

## Abstract
- We define speech emotion recognition (SER) systems as a collection of methodologies that process and classify speech signals to detect the embedded emotions. SER is not a new field, it has been around for over two decades.

## Introduction
- Although it has many applications, emotion detection is a challenging task, because emotions are subjective.
- Such a supervised system brings the necessity of labeled data that have emotions embedded in them.
➔ The data requires preprocessing before their features can be extracted.

- An overview of speech emotion recognition systems

<img src='https://github.com/standing-o/Machine_Learning_Paper_Review/assets/57218700/804dd83e-b343-43ff-b01c-5cccfcbf6f2d'>

------------
## Emotions
- We need to define and model emotion carefully.
- According to Plutchik, more than ninety definitions of emotion were proposed in the twentieth century (Plutchik, 2001)
- Based on these definitions, two models have become common in speech emotion recognition: discrete emotional model,
and dimensional emotional model.

### 1. Discrete emotion theory
- Discrete emotion theory is based on the six categories of basic emotions; sadness, happiness, fear, anger, disgust, and surprise, as described by Ekman and Oster (1979); Ekman et al. (2013).
- Other emotions are obtained by the combination of the basic ones. Most of the existing SER systems focus on these basic emotional categories.
- Nonetheless, these discrete categories of emotions are not able to define some of the complex emotional states observed in daily
communication.

### 2. Dimensional emotional model
- Dimensional emotional model is an alternative model that uses a small number of latent dimensions to characterize emotions such as valence, arousal, control, power (Russell and Mehrabian, 1977; Watson et al., 1988).
- The three-dimensional model includes a dimension of dominance or power, which refers to the seeming strength of the person that is between weak and strong.
- It is not intuitive enough and special training may be needed to label each emotion (Zeng et al., 2009).
- Some of the emotions become identical, such as fear and anger, and some emotions like surprise cannot be categorized and lie outside of the dimensional space since surprise emotion may have positive or negative valence depending on the context.
- Emotions are not independent of each other; instead, they are analogous to each other in a systematic way.

#### 2-1) Uses arousal, activation, or excitation on one dimension
- Arousal dimension defines the strength of the felt emotion. It may be excited or apathetic, and it ranges from boredom to frantic excitement.

#### 2-2) Uses valence, appraisal, or evaluation on the other
- Valence dimension describes whether an emotion is positive or negative, and it ranges between unpleasant and pleasant.

--------------
## Speech Emotion Recognition
### 1. Preprocessing
- First step after collecting data
- Some of these preprocessing techniques are used for feature extraction, while others are used to normalize the features so that variations of speakers and recordings would not affect the recognition process.

#### 1-1) Framing
- Signal framing, also known as speech segmentation, is the process of partitioning continuous speech signals into fixed length segments to overcome several challenges in SER.
- Emotion can change in the course of speech since the signals are non-stationary.
- Continuous speech signals restrain the usage of processing techniques such as Discrete Fourier Transform (DFT) for feature extraction.
- Fixed size frames are suitable for classifiers.

#### 1-2) Windowing
- The windowing function is used to reduce the effects of leakages that occurs during Fast Fourier Transform (FFT) of data caused by discontinuities at the edge of the signals.
- Hamming window function:
$$w(n) = 0.54 - 0.46 \cos(\frac{2 \pi n}{M-1})$$ with $$0 < n < M-1$$
,where the window size is M for the frame w(n).

#### 1-3) Voice Activity Detection
- An utterance consists of three parts; voiced speech, unvoiced speech, and silence. Voiced speech is generated with the vibration of vocal folds that creates periodic excitation to the vocal tract during the pronunciation of phonemes which are perceptually distinct units of sound that distinguish one word from another; such as bag, tag, tab.
- Unvoiced speech is the result of air passing through a constriction in the vocal tract, producing transient and turbulent noises that are aperiodic excitations of the vocal tract. Due to its periodic nature, voiced speech can be identified and extracted. The detection of the presence of voiced speech among various unvoiced speech and silence is called endpoint detection, speech detection or voice activity detection.
- It’s hard to model silence and noise accurately in a dynamic environment; if voice and noise frames are removed, it will be easier to model speech.
- Removal of these frames decreases the complexity and increases accuracy.
  - **Zero crossing rate**: Zero crossing rate is the rate at which a signal changes its sign from positive to negative or vice versa within a given time frame. In voiced speech, the zero crossing count is low whereas it has a high count in unvoiced speech.
  - **Short time energy**: The voiced speech has high energy due to its periodicity while low energy is observed in the unvoiced speech.
  - **Auto-correlation method**: The auto-correlation method provides a measure of similarity between a signal and itself as a function of delay. It is used to find repeating patterns. Because of its periodic nature, voiced signals can be detected using the auto-correlation method.

#### 1-4) Normalization
- Feature normalization is an important step which is used to reduce speaker and recording variability without losing the discriminative strength of the features.
- The generalization ability of features are increased. Normalization can be done at different levels, such as function level and corpus level.
- z-normalization:

$$z = \frac{x-\mu}{\sigma}$$

#### 1-5) Noise reduction
- The noise present in the environment is captured along with the speech signal. This affects the recognition rate, hence some
noise reduction techniques must be used to eliminate or reduce the noise.
- Minimum mean square error (MMSE) and log-spectral amplitude MMSE (LogMMSE) estimators are most successfully applied methods for noise reduction

#### 1-6) Feature selection and dimension reduction

-------------
### 2. Features
- Various features have been used for SER systems; however, there is no generally accepted set of features for
precise and distinctive classification.
- Speech is a continuous signal of varying length that carries both information and emotion.
- Global or local features can be extracted depending on the required approach.
  - **Global features**, also called long-term or supra-segmental features, represent the gross statistics such as mean, minimum and maximum values, and standard deviation.
  - **Local features**, also known as short-term or segmental features, represent the temporal dynamics, where the purpose is to approximate a stationary state.
- These stationary states are important because emotional features are not uniformly distributed over all positions of the speech signal
- These local and global features of SER systems are analyzed in the following four categories
  - Prosodic, spectral, voice-quality features and teager energy operator (TEO) based features
  - Prosodic and spectral features are used more commonly in SER systems.
  - TEO features are specifically designed for recognizing stress and anger.

#### 2-1) Prosodic features
- Prosodic features are those that can be perceived by humans, such as intonation and rhythm.
- The most widely used prosodic features are based on fundamental frequency, energy, and duration. The fundamental frequency, F0, is created by the vibrations in the vocal cord. It yields rhythmic and tonal characteristics of the speech. The change of the fundamental frequency over the course of an utterance yields its fundamental frequency contour whose statistical properties can be used as features.
- The energy of the speech signal, sometimes referred as volume or the intensity, provides a representation which reflects amplitude variation of speech signals over time.

#### 2-2) Spectral features
- When sound is produced by a person, it is filtered by the shape of the vocal tract. The sound that comes out is determined by this shape. An accurately simulated shape may result in an accurate representation of the vocal tract and the sound produced.
- Spectral features are obtained by transforming the time domain signal into the frequency domain signal using the Fourier transform.
  - **Mel Frequency Cepstral Coefficients (MFCC)** feature represents the short term power spectrum of the speech signal.
  - **Linear Prediction Cepstral Coefficients(LPCC)** also embodies vocal tract characteristics of speakers.
  - **Log-Frequency Power Coefficients (LFPC)**, mimics logarithmic filtering characteristics of the human auditory system by measuring spectral band energies using Fast Fourier Transform.
  - **Gammatone Frequency Cepstral Coefficients (GFCC)** is also a spectral feature obtained by a similar technique of MFCC extraction.
  - **Formants** are the frequencies of the acoustic resonance of the vocal tract.

#### 2-3) Voice quality features
- Voice quality is determined by the physical properties of the vocal tract.
- Involuntary changes may produce a speech signal that might differentiate emotions using properties such as the jitter, shimmer, and
harmonics to noise ratio (HNR).

#### 2-4) Teager energy operator based features
- According to Teager, speech is formed by a non-linear vortex-airflow interaction in the human vocal system. A stressful situation affects the muscle tension of the speaker that results in an alteration of the airflow during the production of the sound.

-------------
### 3. Supporting modalities
- A large number of audio-visual databases are available to use for multimodal classification.
- Facial expressions, gestures, posture, body movements
- Biosignals
- Word recognition technology
- Keystroke dynamics, mouse movement, and touch behavior

----------------
### 4. Classifier
#### 4-1) Traditional classifier
- Most preferred algorithms are Hidden Markov Model (HMM), Gaussian Mixture Model (GMM), Support Vector Machines (SVM), and Artificial Neural Networks (ANN).
- There are also classification methods based on Decision Trees (DT), k-Nearest Neighbor (k-NN), k-means, and Naive Bayes Classifiers.

#### 4-2) Ensemble of classifier

#### 4-3) Deep learning based classifier
- Recurrent neural networks (RNNs): RNN, LSTM-RNN

#### 4-4) Convolutional neural networks
- CNN-LSTM, 2D-CNN, 3D-CNN

#### 4-5) Machine learning techniques for classification enhancement
- Variational autoencoder (VAE), denoising autoencoder (DAE), sparse autoencoder (SAE), adversarial autoencoder (AAE).
- Conditional variational autoencoder (CVAE), autoencoder-LSTM, semi-supervised autoencoder

#### 4-6) Multitask learning
- Kim et al. proposed an MTL approach that uses emotion recognition as primary task; and gender and naturalness as auxiliary ones.
-  Mangalam et al. used spontaneity classification as an auxiliary task to MTL.
- Parthasarathy et al. using the interrelation between the dimensions proposed a unified framework to jointly predict the arousal,
valence and dominance dimensions.
- Lotfian et al. used primary and secondary emotions for MTL within spontaneous emotion recognition context.
- Le et al. used MTL BLSTM-RNN to classify emotional speech signals.

#### 4-7) Attention Mechanism
- Emotions are not evenly distributed over the whole utterances, rather they are observed on the specific portion of the utterances as mentioned earlier.
- In speech emotion recognition, this attention mechanism is used to focus on the emotionally salient portion of the given utterance.
- Huang et al. proposed a Deep Convolutional Recurrent Neural Network (CLDNN) with an attention mechanism.
- Mirsamadi et al. used RNN to learn features for SER. They introduced a novel weighted-polling method inspired by the attention mechanisms of neural machine translation.

#### 4-8) Transfer learning
- Deng et al. used a sparse autoencoder to transfer knowledge from one model to another in order to boost the performance.
- Gideon et al. transferred the knowledge between emotion, speaker, and gender recognition tasks.
- Pre-trained Convolutional Neural Networks such as AlexNet or ImageNet which are trained by millions of images are extensively used for image classification tasks.
  - Stolar et al. by using spectrograms formulated SER task as an image classification problem.
- Transfer learning can also be used in cross-corpus and cross-language setting in order to transfer information gained from one corpus can be transferred for another one.

#### 4-9) Adversarial traning
- Large perturbations in model output are penalized by the Adversarial training when small perturbation are added to training samples
- Abdelwahab et al. used Domain Adversarial Neural Network (DANN) to find a common representation between training data and test data.
- Han et al. proposed a conditional adversarial training framework to predict dimensional emotional representation namely arousal and valence.
- Sahu et al. proposed a system to smoothing model predictions using adversarial training.

------------
## Challenge
- Generation of the dataset
- The real-life data is noisy and has far more different characteristics than the others. Although natural data sets are also available, they are fewer in numbers.
- There are also cultural and language effects on SER. There are several studies available working on cross-language SER.

------------
## Conclusion
- The signals are then preprocessed to make them fit for feature extraction. SER systems most commonly use prosodic and
spectral features since they support a wider range of emotion and yield better results.
- Once all the features are extracted, SER systems have a wide range of classification algorithms to choose from.
- All of these preprocessing and feature extraction are done to detect the emotion in the speech signal, yet emotions are still an open problem in psychology.
  - SER systems use manual labeling for their training data, which, as mentioned earlier, is not always exactly correct.
- Although there are systems and realizations of real-time emotion recognition, SER systems are not yet part of our every day life, unlike speech recognition systems that are now easily accessible even with mobile devices