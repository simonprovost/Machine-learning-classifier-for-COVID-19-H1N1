from src.preprocessing import PreProcess
from src.model import SupervisedModel


def genericMetric(classifier, verbose):
    """
    GenericMetric display metrics values according to the verbose value given as parameter.

    - Verbose level 1 will only show classification report of the given model.
    [Classification report from scikitlearn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html)

    - Verbose level 2 will verbose 1 as well as the confusion matrix of the given model.
    [Conf. matrix from scikitlearn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)

    - Verbose level 3 will verbose 1,2 as well as the bias variance trade of the given model.
    [bias variance trade-off library used](http://rasbt.github.io/mlxtend/user_guide/evaluate/bias_variance_decomp/)

    Parameters
    ----------
    classifier : class instance SupervisedModel()
        A classifier previously instantiated, split into training and test subset, as well as trained with one of
        the classifier algorithms available in model.py.
    verbose : int
        level of verbose (see above).

    Returns
    -------
    void()

    """
    if verbose in [1, 2, 3]:
        classifier.displayClassificationReport()
    if verbose in [2, 3]:
        classifier.displayConfusionMatrix()
    if verbose == 3:
        classifier.biasVarianceTradeOff()


def scenario_1(data, verbose):
    """
    The scenario number one show the performance of a simple decision tree classifier according to the data available
    in the classifier instantiated. Data are given in parameters.

    Parameters
    ----------
    data : class instance PreProcess()
        inputs are given to the scenario via the help of a pre-step called pre-processing in this context, use the
        PreProcess() class for reading and pre-process a csv file to give to the scenario.
    verbose : int
        level of verbose used for the scenario.

    Returns
    -------
    void()

    """
    classifier = SupervisedModel(data.input, data.classData)
    classifier.splitData(0.3)

    classifier.decisionTree("entropy", 2)
    classifier.decisionTreeToPic(data.getColumnsWithoutClass())

    genericMetric(classifier, verbose)


def scenario_2(data, verbose):
    """
    The scenario number two show the performance of an optimized decision tree classifier with hyperparameters tunning
    via the use of a gridSearchCV() (cross-validation). Data (i.e.: inputs) are given in parameters.

    Parameters
    ----------
    data : class instance PreProcess()
        inputs are given to the scenario via the help of a pre-step called pre-processing in this context, use the
        PreProcess() class for reading and pre-process a csv file to give to the scenario.
    verbose : int
        level of verbose used for the scenario.

    Returns
    -------
    void()

    """
    classifier = SupervisedModel(data.input, data.classData)
    classifier.splitData(0.3)

    classifier.decisionTreeOptimizedDepth(plotVisualisation=True)
    classifier.decisionTreeToPic(data.getColumnsWithoutClass())
    classifier.tree_to_code(classifier.model, data.getColumnsWithoutClass())
    # genericMetric(classifier, verbose)


def scenario_3(data):
    """
    The scenario number three show a benchmarking between a simple decision-tree and an optimized decision-tree previously
    seen in the scenario_1 and scenario_2. Data used for this benchmark is given as parameters of this scenario.


    WARNING: The following scenario can take time according to your computer.

    Parameters
    ----------
    data : class instance PreProcess()
        inputs are given to the scenario via the help of a pre-step called pre-processing in this context, use the
        PreProcess() class for reading and pre-process a csv file to give to the scenario.

    Returns
    -------
    void()

    """
    classifier = SupervisedModel(data.input, data.classData)
    classifier.splitData(0.3)

    classifier.simpleAndOptimizedDecisionTreeBench()


def scenario_4(data):
    """
    The scenario number four show the benchmarking of multiple algorithms (Naive Bayes, K-NN, Simple DT, Optimized DT)
    Data used for this benchmark is given as parameters of this scenario.

    WARNING: The following scenario can take time according to your computer.

    Parameters
    ----------
    data : class instance PreProcess()
        inputs are given to the scenario via the help of a pre-step called pre-processing in this context, use the
        PreProcess() class for reading and pre-process a csv file to give to the scenario.
    verbose : int
        level of verbose used for the scenario.

    Returns
    -------
    void()

    """
    classifier = SupervisedModel(data.input, data.classData)
    classifier.splitData(0.3)

    classifier.benchAlgorithms()


def scenario_5(data, verbose):
    """
    The scenario number five show the performance of the algorithm that you have to uncomment.
    Data (i.e.: inputs) are given in parameters as well as the verbose level.

    WARNING: Do not run the scenario without instantiated an algorithm available in the SuperviseModel() class instance.

    Parameters
    ----------
    data : class instance PreProcess()
        inputs are given to the scenario via the help of a pre-step called pre-processing in this context, use the
        PreProcess() class for reading and pre-process a csv file to give to the scenario.
    verbose : int
        level of verbose used for the scenario.

    Returns
    -------
    void()

    """
    classifier = SupervisedModel(data.input, data.classData)
    classifier.splitData(0.3)

    ## Choose an algorithm

    ### Random Forest
    # classifier.randomForestClassifier("entropy")

    ### K-NN
    # classifier.knearestneighnour()

    ### Naive-Bayes
    # classifier.naiveBayes()

    genericMetric(classifier, verbose)


def scenario_6(data):
    """
    The scenario number six show the bias variance decomposition between a simple decision tree and an optimized decision tree
    previously explained in the scenario_1 and scenario_2. Data used for both classifier are given as parameter of the
    scenario.

    WARNING: The following scenario can take time according to your computer.

    Parameters
    ----------
    data : class instance PreProcess()
        inputs are given to the scenario via the help of a pre-step called pre-processing in this context, use the
        PreProcess() class for reading and pre-process a csv file to give to the scenario.

    Returns
    -------
    void()

    """
    classifier = SupervisedModel(data.input, data.classData)
    classifier.splitData(0.3)

    classifier.SimpleDTandOptimizedDTVarianceDecomp()


def main():
    """
    The main function is used for instantiate the pre-processing via acquiring a csv and making it ready for
    being classified. Do not forget to load the dataset, clean the attributes as well as encoding the inputs (see
    Preprocess() class instance).
    A - sort of - switch (IF-ELSE cond.) is then used for showing the scenario chosen with his verbose value set at
    the beginning of the function.

    Parameters
    ----------
    scenario : int
        number of the scenario you would like to run with. DEFAULT: -1 (need to changed).
    verbose : int
        level of verbose used for the scenario chosen previously. DEFAULT: -1 (need to changed).

    Returns
    -------
    void()

    """

    scenario = 2
    verbose = 3

    ### Generic for all scenario - Data Pre processing -
    ### Removal of ['neutrophil', 'serumLevelsOfWhiteBloodCell', 'lymphocytes'] due to the significant lack of information.
    data = PreProcess("./data.csv", ['neutrophil', 'serumLevelsOfWhiteBloodCell', 'lymphocytes'])
    data.loadDataset()
    data.cleanDataAttributes()
    data.labelEncodings()

    if scenario == 1:
        scenario_1(data, verbose)
    elif scenario == 2:
        scenario_2(data, verbose)
    elif scenario == 3:
        scenario_3(data)
    elif scenario == 4:
        scenario_4(data)
    elif scenario == 5:
        scenario_5(data, verbose)
    elif scenario == 6:
        scenario_6(data)
    else:
        help(main)


if __name__ == "__main__":
    """
    The following is the programme made for the assessment 2 of the module [Data Mining and Knowledge and discovery - CO832](https://www.kent.ac.uk/courses/modules/module/CO832)
    The main goal was to use WEKA or Python for classifying given data by the professor in charge of the assessment.
    
    Data is about COVID-19 and H1N1, how accurately we are able to classify the given data and explain them through
    a scrutinize report.
    
    Student
    -------
    Provost Simon - sgp28
    MSc Advanced Computer Science.
    
    """
    main()
