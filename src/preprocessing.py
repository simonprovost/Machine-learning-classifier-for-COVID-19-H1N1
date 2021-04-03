from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn import preprocessing
import math


class LabelEncoder:
    """
    The LabelEncoder class instance below is a derived of the sklearn.preprocessing.LabelEncoder() class, forked for
    a better application of the current context. Given the data its running a [label encoding method](https://www.geeksforgeeks.org/ml-label-encoding-of-datasets-in-python/)
    for having all features in a numerical way for feeding any machine learning algorithms.

    Init
    ----------
    Nothing is necessary to give during the instantiation of the class.

    Returns
    -------
    An instance of the LabelEncoder() with all the public methods available below.

    """

    def __init__(self):
        self.columns = None
        self.led = defaultdict(preprocessing.LabelEncoder)

    def fit(self, X):
        """
        The fit method is looping through the data to be sure that NaN value are changed in "None" value (string).

        Parameters
        ----------
        X: Pandas.series()
            X is the dataframe that you are looking for being fit with the method.

        """
        self.columns = X.columns
        for col in self.columns:
            val = X[col].unique()
            val = [x if x is not None else "None" for x in val]
            self.led[col].fit(val)
        return self

    def fit_transform(self, X):
        """
        The fit_transform method is a faster way of doing LabelEncoder().fit ; LabelEncoder().transform.

        Parameters
        ----------
        X: Pandas.series()
            X is the dataframe that you are looking for being fit with the method.

        """
        if self.columns is None:
            self.fit(X)
        return self.transform(X)

    def transform(self, X):
        """
        The transform method is moving all X's data into a label encoded.

        Parameters
        ----------
        X: Pandas.series()
            X is the dataframe that you are looking for being fit with the method.

        """
        return X.apply(lambda x: self.led[x.name].transform(x.apply(lambda e: e if e is not None else "None")))


class PreProcess:
    """
    The PreProcess class instance below is about pre-processing. It is incorporating a few method from the loading of
    the csv to the label encoding of the features available in the data.

    Init
    ----------
    path: string
        The path to the CSV that the class is going to load.
    columnsToDrop: list<string>
        A list of string which represent the colums to initially drop after having loaded the data.

    Returns
    -------
    An instance of the PreProcess() with all the public methods available below.
    An instance of PreProcess() is composed of:
        - path = path from the filed saved.
        - columnsToDrop = columns to Drop saved.
        - input = the data without the classData.
        - classData = the classData without the input.
        - encodedData = data labelled with the LabelEncoder() Method.
    """

    def __init__(self, path, columnsToDrop):
        if not path:
            raise TypeError("Path is needed!")
        self.path = path
        self.columsToDrop = columnsToDrop
        self.dataset = []
        self.input = []
        self.classData = []
        self.encodedData = []

    def __normalizeAttribute(self, attr, validationAttr, posAttr):
        """
        The __normalizeAttribute generic private method is used for modifying a value in the data according to the
        approval of a condition.

        Parameters
        ----------
        attr : string
            The column too look at in the dataset stored in the class.
        validationAttr : string || int
            According to the dtype of the column `attr` in the dataset stored in the class, the validationAttr is the value
            that will be the right side of the condition.
                Example: self.dataset['Age'][0] == validationAttr(=42)
            means that if the value 0 of the colum 'Age' is equal to 42 then ... It works same with string 'male'/'female'.

        posAttr : string || int
            PosAttr is that attribute that the value x of the column `attr` that will be changed with if the condition is
            return a True value.

        Returns
        -------
        (void)
        """
        self.dataset[attr] = np.where(
            (self.dataset[attr] == validationAttr), posAttr, self.dataset[attr]
        )

    def loadDataset(self):
        """
        The loadDataset public method is used for loading the dataset with the path saved during the instantiation
        of the class. It also drop the column given during the instantiation.

        Parameters
        ----------
        (void)

        Returns
        -------
        (void))
        """
        self.dataset = pd.read_csv(self.path)
        self.dataset.drop(self.columsToDrop, axis=1, inplace=True)

    def getColumnsWithoutClass(self):
        """
        The getter getColumnsWithoutClass is a public method for getting the names of all columns available
        in the dataset stored in the class without the class Data.


        Parameters
        ----------
        (void)

        Returns
        -------
        Return a list of string (list<string>).
        """
        datas = self.dataset.drop(['Diagnosis'], axis=1, inplace=False)
        return datas.columns

    def labelEncodings(self):
        """
        The labelEncodings public method is pre-processing the data in a way to:

        - Running a LabelEncoder() in all non-numerical and class data.
        - Making all numerical columns pure without missing values.
        - Making the class data a numerical column.
        - Apply and store the change.

        Parameters
        ----------
        (void)

        Returns
        -------
        (void)
        """

        ### Label Encoding without class data and numerical columns
        df = self.dataset.drop(['Diagnosis', 'Age'], axis=1, inplace=False)

        ### Impurity of the numerical columns with the median of the column 'Age' for its missing values
        ages = self.dataset["Age"].values
        ages = [float(xx) for xx in ages]
        medianAges = np.nanmedian(ages)
        ages = [medianAges if math.isnan(x) else x for x in ages]

        i = 0
        while i < len(ages):
            if isinstance(ages[i], str):
                i += 1
                continue
            if ages[i] <= 18:
                ages[i] = "child"
            elif 18 < ages[i] <= 35:
                ages[i] = "youngAdult"
            elif 35 < ages[i] <= 55:
                ages[i] = "adult"
            else:
                ages[i] = "older"
            i += 1

        df['Age'] = ages

        self.encodedData = LabelEncoder()
        self.encodedData.fit(df)
        transformed = self.encodedData.transform(df)

        ### Class data label encoding to 1,0 for COVID=1 and H1N1=0
        self.dataset['Diagnosis'] = np.where(
            (self.dataset['Diagnosis'] == 'COVID19'), 1, 0
        )

        y = self.dataset['Diagnosis']

        ### Apply changes and stored them to self.input and self.classData
        self.input = transformed.values
        self.classData = y.values

    def cleanDataAttributes(self):
        """
        The cleanDataAttributes public method is pre-processing the data in a way to:

        - Normalise every non-numerical data that have missing values knows as "b'?'" in the data to NaN value.
        - Fill every None value to "Nan".

        This choice is for a better performance with the LabelEncoder().

        Parameters
        ----------
        (void)

        Returns
        -------
        (void)
        """
        self.__normalizeAttribute('Sex', "b'?'", "NaN")
        self.__normalizeAttribute('CTscanResults', "b'?'", "NaN")
        self.__normalizeAttribute('XrayResults', "b'?'", "NaN")
        self.__normalizeAttribute('Diarrhea', "b'?'", "NaN")
        self.__normalizeAttribute('Fever', "b'?'", "NaN")
        self.__normalizeAttribute('Coughing', "b'?'", "NaN")
        self.__normalizeAttribute('SoreThroat', "b'?'", "NaN")
        self.__normalizeAttribute('NauseaVomitting', "b'?'", "NaN")
        self.__normalizeAttribute('Fatigue', "b'?'", "NaN")
        self.__normalizeAttribute('RenalDisease', "b'?'", "NaN")
        self.__normalizeAttribute('diabetes', "b'?'", "NaN")
        self.dataset = self.dataset.fillna("NaN")
