# Fuzzy-KNN-Algorithm
An improved KNN Algorithm based on Fuzzy set theory.

We propose a Fuzzy KNN Algorithm which uses fuzzy set theory to extract the fuzzy vector of the sample, then calculate the distance of the fuzzy vector of the samples and find out the K-nearest neighbors. A weighted vote is assigned to each nearest neighbor by using a reciprocal of the index. Finally, calculate the probability belong to each category. Select the class with the highest probability as the predict label.

This algorithm is applied to malware detection and network intrusion detection. And verified on the public dataset "ClaMP" and "KDD CUP 99". And compared with other machine learning algorithms(Classical KNN, Local Mean KNN, SVM).

An introduction to the "Local Mean KNN" algorithm, please refer to: https://www.researchgate.net/publication/288716151_An_improved_KNN_algorithm_for_imbalanced_data_based_on_local_mean

DataSet Link:  
[ClaMP](https://github.com/urwithajit9/ClaMP)  
[KDD CUP 99](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html)



## Directory Structure
![Directory Structure](https://github.com/tangdongdong/Fuzzy-KNN-Algorithm/blob/master/Directory_Structure.png)

  
## Notes
The Classical KNN algorithm and SVM algorithm is implement with Weka, so you need to using Weka dependency jar package. download link:  
https://www.cs.waikato.ac.nz/ml/weka/downloading.html  
Also, the project involves the processing of .csv files, you need to download the dependency package too. download link:  
http://www.java2s.com/Code/Jar/j/Downloadjavacsv20jar.htm
