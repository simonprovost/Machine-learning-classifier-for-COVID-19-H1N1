# Machine Learning Classifier for COVID-19 & FLU Data

<p style='text-align: justify;'>
The goal of this project is to demonstrate an implementation of an optimised decision tree capable of classifying the
data fed to it with ~90% accuracy using Python and the Scikit Learn as Machine Learning Library. More information can
be found in the sections that follow.
</p>

[Python](http://python.org)

[Scikit Learn](http://scikit-learn.org)


## How to run the project 🏃🏻‍♂️

First and foremost, ensure that all the required packages are installed (see `requirement.txt`), otherwise, use the script
`packages-required-install.sh` to have them installed in your Python packages env.

- `python(3) main.py` (It reads the `data.csv` file at the project's root automatically).

## Documentation 📖

The project is fully commented and accessible through a pre-compiled version at the project's root folder named: `/pre-compiled-documentation`.

Make sure you have the python package `pdoc` installed before regenerating the documentation.

- `./documentation.sh` and see the output.

## Code source and contribution 👀

The project is open and free to use, however, feel free to add some scenarios to the main file `main.py` and submit a
[pull-request](https://github.com/simonprovost/Machine-Learning-COVID19andFLUE/pulls). I would be happy to make that short study of the data more flourished with your ML experience.


# Quick report of the study

## Data 🤝

<p style='text-align: justify;'>
The data are provided from the recent research article of Li, W.T _et.al_ [1] on `Using machine learning of clinical data to diagnose COVID-19: a systematic review and meta-analysis`.
</p>

[1].`
Li, W.T., Ma, J., Shende, N., Castaneda, G., Chakladar, J., Tsai, J.C., Apostol, L., Honda, C.O., Xu, J., Wong, L.M. and Zhang, T., 2020. Using machine learning of clinical data to diagnose COVID-19: a systematic review and meta-analysis. BMC medical informatics and decision making, 20(1), pp.1-13.
`

## Pre-Processing 🔬

<p style='text-align: justify;'>
The dataset provided was also derived from previously published works on the internet, and it contained a
significant number of missing values or categorical features. For the well-processing of the classifiers
shown below (Fig: 1), the Label encoding and Hand-impute methods were used.
</p>

<figure>
  <img src="https://raw.githubusercontent.com/simonprovost/Machine-Learning-COVID19andFLUE/main/assets/DataPreprocessing.png?token=AHZ3AMCM7JXA7EL7TNOI623AND6SI" alt="Pre-processing" style="width:50%">
  <figcaption>Fig: 1 - Pre-processing of the data.</figcaption>
</figure>

## Simple Decision-tree

<p style='text-align: justify;'>
To begin, a first decision tree was created in order to observe the behaviour of its output in relation to the
data that was fed into it. The outcome was disappointing because the tree's depth was too shallow, implying that the
model is not even slightly correlated with the training data.  All the classifier's parameters, as well as the
outcome metrics (Conf. Matrix, Classifier reports, and so on), are shown below (Fig: 2).
The representation of the output tree is shown below (Fig: 3).
</p>

<figure>
  <img src="https://raw.githubusercontent.com/simonprovost/Machine-Learning-COVID19andFLUE/main/assets/SimpleDecisiontree-information.png?token=AHZ3AMGHUJGVTOAEEAMV6ETAND722" alt="Simple decision tree hyper parameters and metrics outcome" style="width:50%">
  <figcaption>Fig: 2 - Simple decision tree outcomes.</figcaption>
</figure>

<figure>
  <img src="https://raw.githubusercontent.com/simonprovost/Machine-Learning-COVID19andFLUE/main/assets/SimpleDecisiontree-Representation.png?token=AHZ3AMEF222EDNYJXCZZ5ADAND7YK" alt="Simple decision tree representation" style="width:50%">
  <figcaption>Fig: 3 - Simple decision tree representation.</figcaption>
</figure>

## Optmised Decision-tree

<p style='text-align: justify;'>
After the poor results shown above, a more in-depth analysis of the hyperparameters has been performed using a
K-cross-validation method. Having the highest score, on the other hand, increases the risk of getting an over-fitted model.
To avoid a model that is overly correlated with the training dataset, hyperparameter range has been intentionally
hand-decreased (range(x, 21) -> range(x, 10)), illustrated in (Fig: 4). The exhibit model is effective while also
being easily interpretable: _best depth found was 9 with gini for the impurity method.
There is less miss-classification and showed strong results with the harmonic mean of precision/recall (Fig: 4).
Finally, the sensitive features observed are shown below (Fig: 5).
</p>

The tree representation is available [here](https://github.com/simonprovost/Machine-Learning-COVID19andFLUE/blob/main/assets/Optimised-decision-tree.png).


<figure>
  <img src="https://raw.githubusercontent.com/simonprovost/Machine-Learning-COVID19andFLUE/main/assets/OptimizedDecisionTree.png?token=AHZ3AMGTWJBL3YGEQ6SB5S3ANEBMO" alt="Optimised decision tree with hyper parameters tunning as well as metrics outcome" style="width:50%">
  <figcaption>Fig: 4 - Optimised decision tree outcomes with hyper parameters tuning using Cross-validation.</figcaption>
</figure>

<figure>
  <img src="https://raw.githubusercontent.com/simonprovost/Machine-Learning-COVID19andFLUE/main/assets/ODTSenstivefeatures.png?token=AHZ3AMEUDJ4RCBIR25XDFZDANEBRU" alt="Optimised decision tree sensitive features" style="width:50%">
  <figcaption>Fig: 5 - Optimised decision tree sensitive features.</figcaption>
</figure>

## Benchmarking Learning curves Simple/Optimised Decision-tree / Bias-variance trade-off 🧪

<p style='text-align: justify;'>
Regarding the Scikit Learn Classification report or the confusion matrix available above, it was difficult to determine
which model had a better fit of the data (i.e.:not under/over but good fit). The figure below describes how good the model is
with the training and cross-validated datasets in the first row. The simple decision tree appears to be very close with
both curves but has a low accuracy, whereas the optimised decision tree on the right has a slight over fit but a high accuracy, which
is not too bad in the current context because it can be applied to unseen data and still recovers good prediction.
The missing values in the data appear to be the source of the overfitting in this case. The two last rows are about
the performance regarding the time of the training etc which presents a huge gap between the simple DT and the 
optimised DT with cross-validation. (Fig: 6). On the other hand, the bias and variance of each model presents
a huge gap after having performing cross-validation for hyper parameters tuning. (Fig: 7).
</p>

<figure>
  <img src="https://raw.githubusercontent.com/simonprovost/Machine-Learning-COVID19andFLUE/main/assets/Benchmarking-Learning-curves.png?token=AHZ3AMBOGRRJ5KA5D67XORDANEDPE" alt="learning curves Simple/Optimised DT" style="width:50%">
  <figcaption>Fig: 6 - Simple/Optimised DT learning curves</figcaption>
</figure>

<figure>
  <img src="https://raw.githubusercontent.com/simonprovost/Machine-Learning-COVID19andFLUE/main/assets/Impactbiasvariancefromadecisontreetoanoptimisedone.png?token=AHZ3AMBOOZLSGN3QM2CX2U3ANED3G" alt="Impact bias variance trade off" style="width:50%">
  <figcaption>Fig: 7 - Impact bias variance trade off from the Simple to the Optimised decision tree</figcaption>
</figure>

## Classifiers comparison 📊

<p style='text-align: justify;'>
Other classifiers are available throughout the Scikit Learn library, and there are likely other ways to manage that
data as well. Following the completion of the preceding experiments, a final one on the comparison of several classifiers
was carried out. According to the findings, the Random Forest classifier is extremely powerful, but it is clearly less
interpretable than a simple decision tree with hyper parameters tuning (using cross-validation). Personally,
I would use the optimised decision tree because of its interpretability and high accuracy. Results are shown below (Fig: 8).
</p>

<figure>
  <img src="https://raw.githubusercontent.com/simonprovost/Machine-Learning-COVID19andFLUE/main/assets/algorithms-benchmarking.png?token=AHZ3AMGDWN6LW2YHZNUNR33ANEEA6" alt="Classifiers comparision" style="width:50%">
  <figcaption>Fig: 8 - Simple/Optimised DT learning curves</figcaption>
</figure>


## Limitation and further study 🚨

<p style='text-align: justify;'>
Some subjects within the data are outliers, as one may have tested positive for one or another disease without mentioning any
symptoms (only age and sex). Hence, more research on outliers could reduce the number of miss-classifications or
better-fit the slightly over-fitted model.
</p>
