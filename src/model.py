import numpy as np
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns;
from sklearn.tree import export_graphviz
from io import StringIO
from IPython.display import Image
import pydotplus
from mlxtend.evaluate import bias_variance_decomp
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import learning_curve
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import _tree
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.tree import export_text

sns.set_theme()


class SupervisedModel:
    """
    The SupervisedModel class instance is about gathering a number of algorithms as well as benchmarking methods
    for helping with the classification of an input and target given during the instantiation of the class.

    - Classification Algorithms: Decision-Tree / Decision Tree with hyperparameters tunning / K-NN (with optimized way) / Naive-Bayes / Random Forest.
    - Benchmarking: Between all algorithms ; Between Simple and Optimized Decision tree ;
    The variance decomposition difference between a Simple Decision tree and an Optimized Decision tree.

    Init
    ----------
    inputData : numpy.ndArray()
        input values that will be used for splitting the data as well as computing a classifier or generate metrics scores.
    target : numpy.ndArray()
        target values that will be used for splitting the data as well as teaching a classifier or generate metrics scores.

    Returns
    -------
    An instance of the SupervisedModel() with all the public methods available below.
    An instance of SupervisedModel() is composed of:
        - __input = private attribute = input value stored during the instantiation of the class.
        - __target = private attribute = input value stored during the instantiation of the class.
        - model = Any of the algorithms implemented in the following class stored the classifier into this attribute.
        - X_train = Training input acquired via the splitting method.
        - X_test = Testing input acquired via the splitting method.
        - y_train = Training teacher acquired via the splitting method.
        - y_test = Testing teacher acquired via the splitting method.
        - y_pred = Value of prediction from a classifier.
        - mse = Mean squared error of a particular classifier acquired via the bias-variance-decom method.
        - bias = Bias of a particular classifier acquired via the bias-variance-decom method.
        - var = Variance of a particular classifier acquired via the bias-variance-decom method.
        - accuracy = Accuracy of a classifier ran.
    """

    def __init__(self, inputData, target):
        self.__input = inputData
        self.__target = target
        self.model = None
        self.X_train = np.array([])
        self.X_test = np.array([])
        self.y_train = np.array([])
        self.y_test = np.array([])
        self.y_pred = np.array([])
        self.mse = None
        self.bias = None
        self.var = None
        self.accuracy = None

    ##### UTILS

    def splitData(self, testSize):  # testsize=0.3 --> 70% training and 30% test
        """
        The splitData public method give the possibility to split the data and stored the output into the class.
        [train_test_split from scikit learn is used](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)

        Parameters
        ----------
        testSize : int
            Example: testsize=0.3 --> 70% training and 30% test.

        Returns
        -------
        (void)
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.__input,
            self.__target,
            test_size=testSize,
            random_state=0
        )

    def displayClassificationReport(self):
        """
        The displayClassificationReport public method print the classification report of a particular classifier ran.
        [classification_report from scikit learn is used](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html)

        Parameters
        ----------
        (void)

        Returns
        -------
        (void)
        """
        if self.y_pred.size == 0:
            raise TypeError("run a classifier before showing the report!")
        print(classification_report(self.y_test, self.y_pred, target_names=['H1N1', 'COVID']))
        print("Accuracy: %.3f" % self.accuracy)

    def displayConfusionMatrix(self, cliOrPlot="both"):
        """
        The displayConfusionMatrix public method plot the confusion matrix of a particular classifier ran.
        [confusion_matrix from scikit learn is used](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)

        Parameters
        ----------
        cliOrPlot: string<"both","cli"> DEFAULT:"both"
            The parameter allow a little bit of verbose, CLI will only display the confusion matrix in the
            command line interpreter. However, both will also plot the confusion matrix with the aid of matplotlib.

        Returns
        -------
        (void)
        """
        mat = confusion_matrix(self.y_test, self.y_pred)

        if cliOrPlot in ["both", "cli"]:
            print(mat)

        if cliOrPlot in ["both", "plot"]:
            names = ['H1N1', 'COVID']

            sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False,
                        xticklabels=names, yticklabels=names)
            plt.xlabel('Truth')
            plt.ylabel('Predicted')
            plt.show()

    def biasVarianceTradeOff(self, lossFunction="mse", numRounds=200, display=True):
        """
        The biasVarianceTradeOff public method print the bias variance trade off (i.e.: The mean square error, the Bias, the variance) of a particular classifier ran.
        [bias_variance_decomp from mlxtend is used](http://rasbt.github.io/mlxtend/user_guide/evaluate/bias_variance_decomp/)

        Parameters
        ----------
        lossFunction: string<"mse", "0-1_loss">
            Allow to use one of the above loss function for the bias_variance_decomp API method.
        numRounds: int range(1, inf) DEFAULT=200
            Allow to give the number of bootstrapping that the API should do on the data for evaluating the model.
        display: Boolean DEFAULT=True
            Display or not the values, in the case of False, it will just stored the result into the class to use it
            later.

        Returns
        -------
        (void)
        """
        self.mse, self.bias, self.var = bias_variance_decomp(
            self.model,
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
            loss=lossFunction,
            num_rounds=numRounds,
            random_seed=123)

        # summarize results
        if display:
            print('mse Loss: %.3f' % self.mse)
            print('Bias: %.3f' % self.bias)
            print('Variance: %.3f' % self.var)
            print("Accuracy: %.3f" % self.accuracy)

    def plotValidationModelCurves(self, estimator, title, X, y, axes=None, ylim=None, cv=None,
                                  n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
        """
        The plotValidationModelCurves public method plot the validation of a model according to it's bias variance
        trade off curve (overfit, underfit, goodfit), as well as the scalability and performance of the model acording
        to the time it took for making it.
        [Inspired from the doc/tutorials available in scikitlearn](https://scikit-learn.org/stable/auto_examples/index.html)

        Parameters
        ----------
        estimator: scikitLearn classifier
            The estimator is the classifier that will be test and plotted.
        title: string
            The title of the plot figure for a better genericity.
        X: numpy.ndArray
            Input stored in the class.
        y: numpy.ndArray
            Target class stored in the class.
        axes: array[]
            Axes of where to plot the result (used for subplot).
        cv: caller || int
            Number of cross-validation or a caller that will split the data with a method.

        Returns
        -------
        (void)
        """
        if axes is None:
            _, axes = plt.subplots(1, 3, figsize=(20, 5))

        axes[0].set_title(title)
        if ylim is not None:
            axes[0].set_ylim(*ylim)
        axes[0].set_xlabel("Training samples")
        axes[0].set_ylabel("Score")

        train_sizes, train_scores, test_scores, fit_times, _ = \
            learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                           train_sizes=train_sizes,
                           return_times=True)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        fit_times_mean = np.mean(fit_times, axis=1)
        fit_times_std = np.std(fit_times, axis=1)

        axes[0].grid()
        axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                             train_scores_mean + train_scores_std, alpha=0.1,
                             color="r")
        axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.1,
                             color="g")
        axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                     label="Training score")
        axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                     label="Cross-validation score")
        axes[0].legend(loc="best")

        axes[1].grid()
        axes[1].plot(train_sizes, fit_times_mean, 'o-')
        axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                             fit_times_mean + fit_times_std, alpha=0.1)
        axes[1].set_xlabel("Training examples")
        axes[1].set_ylabel("fit_times")
        axes[1].set_title("Scalability of the model")

        axes[2].grid()
        axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
        axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.1)
        axes[2].set_xlabel("fit_times")
        axes[2].set_ylabel("Score")
        axes[2].set_title("Performance of the model")

        return plt

    def decisionTreeToPic(self, featureColumns=np.array([])):
        """
        The decisionTreeToPic public method save as image the decision tree model stored in the class.
        WARNING: Do not work with other classifier.
        [export_graphviz from scikitlearn is used](https://scikit-learn.org/stable/modules/generated/sklearn.tree.export_graphviz.html)

        Parameters
        ----------
        featureColumns: numpy.ndArrray<string>
           A list of string that contains the column names of the input data stored in the class.

        Returns
        -------
        (void)
        """
        if featureColumns.size == 0:
            raise TypeError("Feature Columns are needed")

        dot_data = StringIO()
        export_graphviz(self.model, out_file=dot_data,
                        filled=True, rounded=True,
                        special_characters=True, feature_names=featureColumns, class_names=['H1N1', 'COVID'])
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        graph.write_png('output/tree.png')
        Image(graph.create_png())

    def showValidationCurveMaxDepth(self, classifierBench, param_dist):
        """
        The showValidationCurveMaxDepth public method plot the impact of the hyperparameter max_depth during the training
        for a decision tree classifier with gridSearchCV (cross validation) results.

        Parameters
        ----------
        classifierBench : GridSearchCV()
            CrossValidated "gridSearch" from scikitLearn results.
        param_dist : Object
            Hyperparameters used for the GridSearchCV.

        Returns
        -------
        (void)
        """
        train_scores_mean = classifierBench.cv_results_['mean_train_score']
        train_scores_std = classifierBench.cv_results_['std_train_score']
        test_scores_mean = classifierBench.cv_results_['mean_test_score']
        test_scores_std = classifierBench.cv_results_['std_test_score']

        datas = param_dist['max_depth']

        plt.figure()
        plt.title('Model')
        plt.xlabel('max_depth')
        plt.ylabel('Score')

        plt.semilogx(datas, train_scores_mean, label='Mean Train score',
                     color='navy')
        plt.gca().fill_between(datas,
                               train_scores_mean - train_scores_std,
                               train_scores_mean + train_scores_std,
                               alpha=0.2,
                               color='navy')
        plt.semilogx(datas, test_scores_mean,
                     label='Mean Test score', color='darkorange')

        plt.gca().fill_between(datas,
                               test_scores_mean - test_scores_std,
                               test_scores_mean + test_scores_std,
                               alpha=0.2,
                               color='darkorange')

        plt.legend(loc='best')
        plt.show()

    ##### ALGORITHMS

    def decisionTree(self, crit, depth):
        """
        The decisionTree public method is the wrapper of a Decision tree Classifier implemented by SciKitLearn.
        The method instantiate the tree; fit and predict the tree; and store the accuracy score in the class.

        Parameters
        ----------
        crit : string<"entropy","gini">
            criterion used for validated a node of the tree (impurity, etc.).
        depth : int
            How deep could go the model.

        Returns
        -------
        (void)
        """
        if self.X_train.size == 0 or self.y_train.size == 0:
            raise TypeError("split Data before running a classifier!")
        self.model = DecisionTreeClassifier(criterion=crit, max_depth=depth, random_state=0)
        self.model = self.model.fit(self.X_train, self.y_train)
        self.y_pred = self.model.predict(self.X_test)
        self.accuracy = metrics.accuracy_score(self.y_test, self.y_pred)

    def decisionTreeOptimizedDepth(self, plotVisualisation=False):
        """
        The decisionTreeOptimizedDepth public method is the wrapper of a Decision tree Classifier implemented by SciKitLearn.
        A GridSearchCV for getting the best hyper parameters for the decision tree is also running.

        The method instantiate the tree; fit and predict the tree; and store the accuracy score in the class.
        The method could also show the impact of using max_depth as hyper parameters on the training. (uncomment the
        appropriate line for getting access to the plot).

        Parameters
        ----------
        plotVisualisation : Boolean DEFAULT=False
            If true, a plot of the mean test score regarding the max depth value tested will be plotted.

        Returns
        -------
        (void)
        """
        if self.X_train.size == 0 or self.y_train.size == 0:
            raise TypeError("split Data before running a classifier!")

        param_dist = {
            "max_depth": range(3, 10),
            "criterion": ["entropy", "gini"],
            #  "min_samples_split": [2, 5, 10, 15, 20], #Do not produce relevant results.
            #  "min_samples_leaf": [1, 3, 5, 7, 10], #Do not produce relevant results.
            #  "max_leaf_nodes": [None, 3, 5, 7, 10, 15, 20], #Do not produce relevant results.
        }

        classifierBench = GridSearchCV(DecisionTreeClassifier(), param_dist, cv=10, n_jobs=-1, return_train_score=True)
        print(self.X_train)
        classifierBench.fit(X=self.X_train, y=self.y_train)

        # Visualisation of the K-FOLD cross validation
        if plotVisualisation:
            for i in ['mean_test_score', 'std_test_score', 'param_max_depth']:
                print(i, " : ", classifierBench.cv_results_[i])

            for ind, i in enumerate(classifierBench.cv_results_['param_max_depth']):
                xGraph = classifierBench.cv_results_['param_max_depth'][ind]
                yGraph = classifierBench.cv_results_['mean_test_score'][ind]
                plt.scatter(xGraph, yGraph, label='Depth: ' + str(classifierBench.cv_results_['param_max_depth'][ind]))

            plt.legend()
            plt.xlabel('Param Max Depth')
            plt.ylabel('Mean (Test) score')
            plt.show()

        print("Tuned Decision Tree Parameters: {}".format(classifierBench.best_params_))

        self.model = DecisionTreeClassifier(criterion=classifierBench.best_params_['criterion'],
                                            max_depth=classifierBench.best_params_['max_depth'],
                                            # min_samples_split=classifierBench.best_params_['min_samples_split'], #Do not produce relevant results.
                                            # min_samples_leaf=classifierBench.best_params_['min_samples_leaf'], #Do not produce relevant results.
                                            # max_leaf_nodes=classifierBench.best_params_['max_leaf_nodes'], #Do not produce relevant results.
                                            )
        self.model = self.model.fit(self.X_train, self.y_train)
        self.y_pred = self.model.predict(self.X_test)
        self.accuracy = metrics.accuracy_score(self.y_test, self.y_pred)

    def randomForestClassifier(self, crit):
        """
        The randomForestClassifier public method is the wrapper of a Random Forest classifier implemented by SciKitLearn.
        The method instantiate the trees; fit and predict the trees; and store the accuracy score in the class.

        Parameters
        ----------
        crit : string<"entropy","gini">
            criterion used for validated a node of the tree (impurity, etc.).

        Returns
        -------
        (void)
        """
        if self.X_train.size == 0 or self.y_train.size == 0:
            raise TypeError("split Data before running a classifier!")
        self.model = RandomForestClassifier(n_estimators=100, criterion=crit, random_state=0)

        self.model = self.model.fit(self.X_train, self.y_train)
        self.y_pred = self.model.predict(self.X_test)
        self.accuracy = metrics.accuracy_score(self.y_test, self.y_pred)

    def knearestneighnour(self, optimized=False):
        """
        The knearestneighnour public method is the wrapper of a K-NN classifier implemented by SciKitLearn.
        With an optimized way, which get the best n hyperparameter via the aid of a loop between 40 K-NN with K=i, which
        one produce the best score on the data given.

        Parameters
        ----------
        optimized : Bool DEFAULT=FALSE
            If True, the K-NN will run an optimized loop for having the best "n" to classify with.

        Returns
        -------
        (void)
        """
        if self.X_train.size == 0 or self.y_train.size == 0:
            raise TypeError("split Data before running a classifier!")

        scaler = StandardScaler()
        scaler.fit(self.X_train)

        self.X_train = scaler.transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

        # Calculating error for K values between 1 and 40
        n_hyper_param = 5

        if optimized:
            error = []
            for i in range(1, 40):
                knn = KNeighborsClassifier(n_neighbors=i)
                knn.fit(self.X_train, self.y_train)
                pred_i = knn.predict(self.X_test)
                error.append(np.mean(pred_i != self.y_test))

            plt.figure(figsize=(12, 6))
            plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
                     markerfacecolor='blue', markersize=10)
            plt.title('Error Rate K Value')
            plt.xlabel('K Value')
            plt.ylabel('Mean Error')

            plt.show()
            n_hyper_param = input("Enter the n: ")

        self.model = KNeighborsClassifier(n_neighbors=int(n_hyper_param))
        self.model = self.model.fit(self.X_train, self.y_train)
        self.y_pred = self.model.predict(self.X_test)
        self.accuracy = self.model.score(self.X_test, self.y_test)

    def naiveBayes(self):
        """
        The naiveBayes public method is the wrapper of a Naive Bayes classifier implemented by SciKitLearn.
        The method instantiate the model; fit and predict the model; and store the accuracy score in the class.

        Parameters
        ----------
        (void)

        Returns
        -------
        (void)
        """
        if self.X_train.size == 0 or self.y_train.size == 0:
            raise TypeError("split Data before running a classifier!")
        self.model = GaussianNB()

        self.model = self.model.fit(self.X_train, self.y_train)
        self.y_pred = self.model.predict(self.X_test)
        self.accuracy = self.model.score(self.X_test, self.y_test)

    ##### BENCH ALGORITHMS

    def simpleAndOptimizedDecisionTreeBench(self):
        """
        The simpleAndOptimizedDecisionTreeBench public method is the benchmarking of a simple DT and Optimized DT.

        The process is as follow:
        - Create the cross validation caller (ShugffleSplit from scikit learn for going over n iteration with test_size%
        randomly selection as a validation set).
        - Create the estimator.
        - Plot the validation model curves
        - Doing the above step both for the Simple DT and Optimized DT and observe the results on a subplot.

        Parameters
        ----------
        (void)

        Returns
        -------
        (void)
        """
        fig, axes = plt.subplots(3, 2, figsize=(10, 15))
        fig.suptitle('Simple DT and Optimized DT Model - Model Validation')

        X = self.__input
        y = self.__target

        title = "Learning Curves (Decision Tree)"
        cv = ShuffleSplit(n_splits=100, test_size=0.3, random_state=0)

        estimator = DecisionTreeClassifier(criterion="entropy", max_depth=2)
        self.plotValidationModelCurves(estimator, title, X, y, axes=axes[:, 0], ylim=(0.7, 1.01),
                                       cv=cv, n_jobs=4)

        title = "Learning Curves (Decision tree with hyperparameters tunning"

        cv = ShuffleSplit(n_splits=100, test_size=0.3, random_state=0)

        param_dist = {
            "max_depth": range(3, 10),
            "criterion": ["entropy", "gini"],
            # "min_samples_split": [2, 5, 10, 15, 20],
            # "min_samples_leaf": [1, 3, 5, 7, 10],
            # "max_leaf_nodes": [None, 3, 5, 7, 10, 15, 20],
        }

        estimator = GridSearchCV(DecisionTreeClassifier(), param_dist, cv=10, n_jobs=-1, return_train_score=True)
        # gives [{max_depth: 9}, {criterion: 'gini'}]. Use the following estimator for computing faster.
        # estimator = DecisionTreeClassifier(criterion='gini',
        #                                   max_depth=9,
        #                                   )
        self.plotValidationModelCurves(estimator, title, X, y, axes=axes[:, 1], ylim=(0.7, 1.01),
                                       cv=cv, n_jobs=4)

        plt.show()

    def SimpleDTandOptimizedDTVarianceDecomp(self):
        """
        The SimpleDTandOptimizedDTVarianceDecomp public method is the gain in variance and bias of passing from
        a simple Decision Tree to a Optimized Decision Tree.
        [Inspired from the doc/tutorials available in scikitlearn](https://scikit-learn.org/stable/auto_examples/index.html)

        The process is as follow:
        - Create the estimator
        - Evaluate his bias variance decomposition using mlxtend.
        - Doing the above step twice for the Simple and Optimized Decision tree.
        - Display the reduction of the variance from the first classifier to the second.
        - Display the introduction of the bias from the first classifier to the second.

        Parameters
        ----------
        (void)

        Returns
        -------
        (void)
        """
        dt = DecisionTreeClassifier(criterion="entropy", max_depth=2)
        error_dt, bias_dt, var_dt = bias_variance_decomp(
            dt,
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
            'mse',
            random_seed=123
        )

        param_dist = {
            "max_depth": range(3, 10),
            "criterion": ["entropy", "gini"],
        }

        OptDt = GridSearchCV(DecisionTreeClassifier(), param_dist, cv=10, n_jobs=-1, return_train_score=True)
        error_dt_pruned, bias_dt_pruned, var_dt_pruned = bias_variance_decomp(
            OptDt,
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
            'mse',
            random_seed=123
        )

        print("Variance Impact from the first to the second classifier:",
              str(np.round((var_dt_pruned / var_dt - 1) * 100, 2)) + '%')
        print("Bias Impact from the first to the second classifier:",
              str(np.round((bias_dt_pruned / bias_dt - 1) * 100, 2)) + '%')

        # fig, ax = plt.subplots(nrows=1, ncols=2)

        print(var_dt_pruned)
        print(var_dt)
        print(bias_dt_pruned)
        print(bias_dt)

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 8))

        algorithms = ['Simple DT', 'Optimised DT']
        biases = [bias_dt, bias_dt_pruned]
        ax[0].bar(algorithms, biases, color='lightblue')
        ax[0].set_ylabel('Bias')
        ax[0].set_title('Bias impact through a simple to an optimised DT')
        ax[0].set_xticks(algorithms)
        ax[0].set_xticklabels(algorithms)
        ax[0].legend(['Bias'])

        variances = [var_dt, var_dt_pruned]
        ax[1].bar(algorithms, variances, color='#69b3a2')
        ax[1].set_ylabel('Variance')
        ax[1].set_title('Variance impact through a simple DT to an optimised DT')
        ax[1].set_xticks(algorithms)
        ax[1].set_xticklabels(algorithms)
        ax[1].legend(['Variance'])

        plt.show()

    def benchAlgorithms(self):
        """
        The benchAlgorithms public method is the benchmarking of all algorithms available in this class together with
        the same input/class data. The output is a box plot which shows the outliers as well as where is the classifier
        regarding is F1-score on a range of 0-1.

        The process is as follow:
        - Create the estimators.
        - Evaluate the estimators with a KFold method.
        - Append the results.
        - Display the results on a box plot graph.

        Parameters
        ----------
        (void)

        Returns
        -------
        (void)
        """
        param_dist = {
            "max_depth": range(3, 10),
            "criterion": ["entropy", "gini"],
        }

        models = [
            ('NB', GaussianNB()),
            ('KNN', KNeighborsClassifier(n_neighbors=5)),
            ('RF', RandomForestClassifier(n_estimators=100, criterion="entropy", random_state=0)),
            ('DT', DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=0)),
            ('ODT', GridSearchCV(DecisionTreeClassifier(), param_dist, cv=10, n_jobs=-1, return_train_score=True))
        ]

        results = []
        names = []
        for name, model in models:
            kfold = model_selection.KFold(n_splits=100, shuffle=True, random_state=0)
            cv_results = model_selection.cross_val_score(model, self.__input, self.__target, cv=kfold, scoring='f1')
            results.append(cv_results)
            names.append(name)

        fig = plt.figure()
        fig.suptitle('Algorithms Comparison')
        ax = fig.add_subplot(111)
        plt.boxplot(results)
        plt.ylabel('F1-Score')
        ax.set_xticklabels(names)
        plt.show()

    def tree_to_code(self, tree, feature_names, pseudoCode=False):
        """
        Outputs a decision tree model as a Python function

        Parameters:
        -----------
        tree: decision tree model
            The decision tree to represent as a function
        feature_names: list
            The feature names of the dataset used for building the decision tree
        """
        if pseudoCode:
            tree_rules = export_text(self.model, feature_names=list(feature_names))
            print(tree_rules)
            return

        tree_ = tree.tree_
        feature_name = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]
        print ("def tree({}):".format(", ".join(feature_names)))

        def recurse(node, depth):
            indent = "  " * depth
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_name[node]
                threshold = tree_.threshold[node]
                print ("{}if {} <= {}:".format(indent, name, threshold))
                recurse(tree_.children_left[node], depth + 1)
                print ("{}else:  # if {} > {}".format(indent, name, threshold))
                recurse(tree_.children_right[node], depth + 1)
            else:
                print ("{}return {}".format(indent, tree_.value[node]))

        recurse(0, 1)
