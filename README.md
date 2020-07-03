# Solutions of Clinical BCI Challenge-WCCI2020 -- HUSTBCI

We introduce our method in this competition briefly from three aspects: Experimental Settings, running the code and references. If there are any questions, please contact wenz@hust.edu.cn.

## I. Experimental Settings

### 1) Within Subject Classification

#### Preprocessing
We used **leave-one-out cross validation** to determine whether to filter the signals and the optimal time-window to extract trials for each subject. We performed 4-40Hz band-pass filtering after removing baseline drift if filtering is required and then extracted trials using different time-window selected in leave-one-out cross validation for different subjects.


#### Feature Extraction
**Common Spatial Pattern(CSP)** filters are used to extract features from the covariance matrices of each subject.

#### Classification Approach
**Linear Discriminant Analysis** is used for classification in our solution.


### 2) Cross Subject Classification

#### Preprocessing

We use **leave one subject out cross validation** to choose the best frequency bands and the selected trial time for all the subjects. In our solution, the signals were first detrended to remove polynomial trend in each channel, then band-pass filtered to 4-40 Hz. The signals from 0.5 to 3.5s after the cue onsets were extracted as trials.

#### Feature Extraction

We use tangent space mapping to extract tangent features. Considering there are 12 electrodes in the dataset, thus we can extract the 78-dimensional features.

#### Classification Approach

We used two transfer learning approaches in our experiments: **Centroid Alignment (CA)** [1] approach to align the covariance matrices of different subjects firstly, and then use and **Selective Pseudo-Labeling (SPL)** [2] approach to facilitate accurate pseudo-labeling by structured prediction to predict the labels of the unseen subject in an unsupervised setting. The hyperparameters of SPL were selected according to the leave one subject out validation results. In our experiments, we set the subspace dimension as 10, iteration number as 10, and the regularization parameter $\alpha$ as 1.



## II. Running the code

The platform in our experiments are MATLAB 2019b and 2020a in 64-bit Windows 10 and MacOS system.

Code files introduction:

**CrossValidation.m** -- demo file. It performs within-subject leave-one-out cross validation to determine the optimal parameters for each subject. After running this file, a mat format file containing the selected parameters in cross validation would be saved.

**demo_withSubject.m** -- demo file. It first loads the parameter file saved from **CrossValidation.m** and then estimates the labels of test data of subject P01~P08.

**demo_crossSubject.m** -- demo file. It's the implementation of the cross subject validation. We use the eight training subjects to learn an adaptive classifier for predicting the labels of test data of subject P09 and P10.

**/utils/CSPfeature** --  function file that performs CSP filtering.

**/utils/centroid_align.m** -- function file that implements the CA approach. Please find the specific input/output instructions in the function comments.

**/utils/SPL.m** -- function file that implements the SPL approach. Please find the specific input/output instructions in the function comments.



## III. References

This code refers to the following papers:
[1] Wen Zhang, Dongrui Wu, “Manifold Embedded Knowledge Transfer for Brain-Computer Interfaces,” IEEE Trans. on Neural Systems & Rehabilitation Engineering, 28(5), pp. 1117-1127, 2020.
[2] Qian Wang, Toby P. Breckon, Unsupervised Domain Adaptation via Structured Prediction Based Selective Pseudo-Labeling, In proceedings of AAAI Conference on Artificial Intelligence, 2020.