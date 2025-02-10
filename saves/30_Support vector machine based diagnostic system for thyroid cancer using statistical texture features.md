## Support vector machine based diagnostic system for thyroid cancer using statistical texture features
- Author: Gopinath, B and Shanthi, N
- Journal: Asian Pacific Journal of Cancer Prevention
- Year: 2013
- Link: https://pdfs.semanticscholar.org/1172/4368de105f50369e866d8064449477bb4ebf.pdf

### Abstract
- The aim of this study was to develop an automated computer-aided diagnostic system for diagnosis of thyroid cancer pattern in fine needle aspiration cytology (FNAC) microscopic images with high degree of sensitivity and specificity using statistical texture features and a SVM.
- A training set of 40 benign and 40 malignant FNAC images and a testing set of 10 benign and 20 malignant FNAC images were used to perform the diagnosis of thyroid cancer. 
  - Initially, segmentation of region of interest (ROI) was performed by region-based morphology segmentation. The developed diagnostic system utilized statistical texture features derived from the segmented images using a Gabor filter bank at various wavelengths and angles. 
  - Finally, the SVM was used as a machine learning algorithm to identify benign and malignant states of thyroid nodules.
- The SVM achieved a diagnostic accuracy of 96.7% with sensitivity and specificity of 95% and 100%, respectively.


### Materials and Methods
- The training image set consists of 40 benign and 40 malignant images whereas the testing image set consists of 10 benign and 20 malignant images.
- The thyroid FNAC images are stained by various types of stains and they are multi-stained cytological FNAC images. 
  -  In automated computer-aided diagnosis, the diagnostic accuracy is affected by the unwanted background staining information and hence it is very important to select an appropriate segmentation method to carefully remove the unwanted background information in multi-stained FNAC images.

### Image Segmentation
- The diagnostic accuracy depends on the efficient segmentation methodology and can be improved, if the segmentation of the regions of interest is clearly defined.

### Preprocessing
- The preprocessing stage consists of converting RGB image into gray-scale image, cropping 256x256 pixel image from the input image by auto-cropping and thresholding.
-  Since morphology concentrates only on the shape of the cell objects, RGB components of the given input image are not required in this study. 
- In automated segmentation and classification methods, high density cell regions in slide images are identified before segmentation and classification for simple and fast computing.
- An appropriate threshold value is calculated using Otsu’s method to classify image pixels into one of two classes; i.e. objects and background.

### Gabor Filter
- Each image in the training set and testing set are convolved with the filters. Then, the statistical features mean, standard deviation, entropy, variance, energy, homogeneity, contrast and correlation are calculated and stored in feature library.
- These eight statistical features are derived for 80 images in training set and 30 images in testing set so that the total number of feature vectors becomes 880.

### Training and Testing
- In training phase, the statistical texture features are calculated from automatically cropped and segmented training set images of benign and malignant thyroid nodules and stored in the feature library. 
- In the testing phase, the same set of statistical texture features are extracted from automatically cropped and segmented
testing set images of benign and malignant thyroid nodules. 
- These extracted texture features are compared with the feature library and classified using k-NN and SVM classifiers.

### Results
- The statistical texture features are again extracted and the features are compared with the features available
in the feature library and classified using k-NN and SVM classifier.
- From the diagnostic results, it is clearly observed that SVM classifier performed better than the k-NN classifier. 
- The highest diagnostic accuracy of 96.66% is reported with sensitivity of 95% and specificity of 100% by SVM classifier when it uses the results of morphology segmentation and the texture features derived from Gabor filter with wavelength of 4 and angle of 45.
