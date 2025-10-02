# Classification_validation:

# SVM_analysis

Classification analysis was conducted to validate the utility of the data records in two parts: 
1) a binary classification of positive and negative emotional states to make a direct comparison with previous studies and 
2) a classification of the nine-category emotional states ('Anger','Disgust','Fear','Sadness','Neutral','Amusement','Inspiration','Joy','Tenderness') to test whether the present dataset supports a more fine-grained emotion recognition. The classification of emotional states was conducted on a 1-second time scale.


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
