This is the open-source code of the pre-processing and the technical validation for the FACED
Dataset

# Codes for pre-processing and validation

Preprocessing： 

main.py: 

the main file for the data preprocessing pipeline：
1. unit_adjust ： # Different subjects may have different units (uV or V) of the recorded EEG signals， all were adjusted into uV
2. get the last 30-s epoch for each video clip
3. downsample：250 Hz if necessary 
4. band-pass filter ：0.05-47Hz
5. bad_channels_interpolate: The detected bad channels were interpolated with the MNE interpolate_bads() function
6. ICA: following the practice in the MNE toolbox, the independent component analysis (ICA) method was performed. The independent components (ICs) containing ocular artifacts were defined and rejected automatically by using FP1/2 as the proxy for electro-oculogram with the MNE ica.find_bads_eog() and ica.exclude functions. You can also reject ICs manually.
7. re-referenced to the common average reference
8. data_organized：The electrodes in two cohorts are at the same spatial position on the scalp while 6 of them have different names due to the device setting. The code was used to adjust the order of electrodes according to the Cohort 2# :see Electrode_Location.xlsx for details
9. Save the pre-processed data for each subject as .pkl files
For each subject, the pre-processed EEG data is a 3-dimensional matrix of VideoNum*ElecNum*(TrialDur*SampRate). The number of video clips is 28. The order of video clips in the pre-processed data was reorganized according to the index of video clips as reported in Stimuli_info.xlsx. The number of electrodes is 32. The order of electrodes is provided in Electrode_Location.xslx. The duration of each EEG trial is 30 seconds, and the sampling rate of pre-processed EEG data is 250 Hz.

the classes and functions for preprocessing can be found in the Preprocessing.py

All required packages are listed in torch_ubuntu.yml or torch_win.yml file (depends on your operating system). We recommend users to use Anaconda for the virture enviornment.
$ conda env create -f torch.yml


Then, the .pkl file was used in the validation parts (1. Correlation analysis: the topographies of the average correlation coefficients between the spectral powers  and self-report ratings on emotions; 2. Classification analysis for DE+svm)

Note that the present dataset also provide codes for  a Contrastive Learning for Inter-Subject Alignment (CLISA) algorithm for cross-subject emotion recognition
Shen, X., Liu, X., Hu, X., Zhang, D. & Song, S. Contrastive learning of subject-invariant EEG representations for cross-subject emotion recognition. IEEE Transactions on Affective Computing (2022).
However, the preprocessing pipeline is a bit different according to the clisa paper, so we provide the clisa preprocessing option(--clisa_or_not)

## You could run as follow:
$ mkdir Processed_data; mkdir Clisa_data
$ python main.py --clisa-or-not yes


# Feature calculation:

features_extract.py:

This code calculated the commonly-used EEG features including differential entropy (DE) and power spectral density (PSD).The DE and PSD features were obtained from the pre-processed data within each non-overlapping second at 5 frequency bands (delta, theta, alpha, beta and gamma).The formula to calculate DE and PSD followed the practice in the SEED dataset (https://bcmi.sjtu.edu.cn/home/seed/seed-iv.html).

For each subject, the DE and PSD feature is a 4-dimensional matrix of VideoNum*ElecNum*TrialDur*FreqBand.


Validation

# Correlation_analysis:

Behavioral_Data_validation.py: 
The codes read the self-report ratings on  emotion items after each video to test 
whether the video-watching paradigm elicit targeted emotions with repeated measures analysis of variance (rmANOVA) and post-hoc tests.


The self-report ratings on  emotion items  can be found in Data/subXXX/ After_remarks.mat for each
subject. For the ith video, the After_remark{i}.score is the  self-report ratings on 12 items["Joy","Tenderness","Inspiration","Amusement","Anger","Disgust","Fear","Sadness","Arousal","Valence","Familiarity","Liking"]. 
After_remark{i}.vid is the video idx for this video, see Stimuli_info.xlsx for details of the video idx
 

EEG_Correlation_Analysis.py:
Pearson correlation between the relative spectral powers and the self-reported emotional ratings was conducted to illustrate the neural activation patterns to these emotions. For every 30-s EEG epoch for each film clip, the relative spectral powers over the frequency bands of interest ( delta: 1–4 Hz, theta: 4–8 Hz, alpha: 8–14 Hz, beta: 14–30 Hz and gamma: 30–47 Hz) were calculated using Fourier Transform for each 1-s segment and averaged across segments. Then, since all 28 film clips had ratings on each emotion item, the correlation were conducted between each participant’s 28 ratings on a certain emotion item and her/his relative spectral powers for 28 film clips and repeated for all the electrodes. The topographies of the across-participant average Pearson correlation coefficients were demonstrated

pos.npy: The position of electrodes



# Classification_validation:

# SVM_analysis

Classification analysis was conducted to validate the utility of the data records in two parts: 1) a binary classification of positive and negative emotional states to make a direct comparison with previous studies and 2) a classification of the nine-category emotional states ('Anger','Disgust','Fear','Sadness','Neutral','Amusement','Inspiration','Joy','Tenderness') to test whether the present dataset supports a more fine-grained emotion recognition. The classification of emotional states was conducted on a 1-second time scale.


Here, the classical method combing differential entropy (DE) features with the support vector machine (SVM) was used for both intra-subject emotion recognition and cross-subject emotion recognition.  
In the intra-subject emotion recognition, for all positive/negative video clips, 90% of EEG data in each video clip was used as the training sets, and the remaining 10% in each video clip was used as the testing sets for each subject. 
In the cross-subject emotion recognition, the subjects were divided into 10 folds (12 subjects for the first nine folds, and 15 subjects for the 10th fold). Then, nine-fold subjects were used as the training sets, and the remaining subjects were used as the testing sets. 
The procedure was repeated 10 times and the classification performances were obtained by averaging accuracies for 10 folds.

The io.utils.py, reorder_vids.py, and load_data.py include relevant classes or functions needed for the program.

1. DE feature calculation
$ python save_de.py 
2. Running_norm calculation (--n-vids 24 for binary classification; --n-vids 28 for nine-category classification, same below)
$ python running_norm_fea.py --n-vids 28    
3. Using LDS to smooth the data
$ python smooth_lds.py --n-vids 28
4. Using SVM to do the classification（--subjects-type intra for analysis with intra-subjects; --valid-method  loo for leave-one-subject-out analysis)
$ python main_de_svm.py  --subjects-type cross --valid-method 10-folds --n-vids 28 
5. To evaluate the accuracy across subjects inside the dataset. It's noted that the final accuracy is calculated inside a subject then averaged across subjects.
$ python main_de_ svm.py --train-or-test test


# Clisa_analysis
We also employed a Contrastive Learning for Inter-Subject Alignment (CLISA) algorithm to achieve a classification among nine emotion categories cross subjects
See details in：
Shen, X., Liu, X., Hu, X., Zhang, D. & Song, S. Contrastive learning of subject-invariant EEG representations for cross-subject emotion recognition. IEEE Transactions on Affective Computing (2022).

To run the program, please extract the Others/Clisa_data.zip into the Clisa_data folder under Code/Validation/Classification_validation/Clisa_analysis first.
The io.utils.py, train_utils, model.py, reorder_vids.py, simCLR.py and train_utils.py include relevant classes or functions needed for the program.

1. Pretraining section, you can choose to 
$ python main_pretrain.py  --training-fold all 
(The training-fold could be chose to 'all' for the whole 10-folds operation or separate into pieces for a multi-gpu server by using --training-fold 5 --gpu-index 5;
the binary classification choice is provided using --cls 2)
2. Extract the feature from the CNN
$ python extract_pretrainFeat.py  --cls 9
3. Running norm section 
$  python running_norm.py --cls 9  
4. Use lds to smooth the data
$ python smooth_lds.py --cls 9  
5. Fine-tuning section
In the fine-tuning section, both the ten-fold procedure and  leave-one-subject-out procedure were conducted for comparison
$ python main_classify.py --cls 9  
6. To evaluate the accuracy across subjects inside the dataset. It's noted that the final accuracy is calculated inside a subject then averaged across subjects.
$ python main_classify.py --train-or-test test

It should be noted that since there is a running normalization section, the aligned data should be reordered  according to the order in which subjects received them during the experiment.
For this purpose, the /After_remarks which contains the after_remarks.mat from different subjects is provided under the folders of each classification method.
Also, to draw the accuracies of different subjects, evl.py is provided. You should move the relevant .csv file and change the filepath inside the code by yourself.



