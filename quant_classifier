'''
Created on Mar 2, 2011

@author: zul110
'''
import optparse
import logging
import logging.handlers
import sys

import numpy as np
import pylab as pl
import os
import csv
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.svm import SVC
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.grid_search import GridSearchCV

INPUT = 'inputf'
OUTPUT = 'outputf'
TRUTH = 'truthf'
DEFAULTLOG = './qlog'

class QClassifier():
    def __init__(self, input, output, truthfile):
        if not os.path.exists(input) :
            _log.error('Cannot find the input %s or output %s' % (input, output))
            sys.exit(-1)
        
        self.queryfile = input
        self.outputfile = output
        self.truthfile = truthfile
        
        self.train_data = None
        self.test_data = None
        self.train_target = None
        self.test_target = None
        
        self.test_data_aid = []
        
        self.np_sample_train = 0
        self.np_sample_test = 0
        self.np_feature = 0
        
        self.csv_output = None
        
        self.scaler = None
        self.predict_model = None
    
    def load_data(self):
        """
        load data from disk.
        """
        
        #process input file
        datafile = csv.reader(open(self.queryfile), delimiter = ' ')        
        #get the first row
        temp = datafile.next()        
        self.np_sample_train = int(temp[0])
        self.np_feature = int(temp[1])
        self.train_data = np.empty((self.np_sample_train, self.np_feature))
        self.train_target = np.empty((self.np_sample_train, ), dtype = np.int)
        
        for i, ir in enumerate(datafile):
            if i < self.np_sample_train:
                self.train_target[i] = ir[1]
                self.train_data[i] = [ele.split(':')[1] for ele in ir[2:]]
            elif i == self.np_sample_train:
                self.np_sample_test = int(ir[0])
                self.test_data = np.empty((self.np_sample_test, self.np_feature))
            else:
                self.test_data[i-self.np_sample_train-1] =  [ele.split(':')[1] for ele in ir[1:]]
                self.test_data_aid.append(ir[0])
        
        
        #process output file
        self.csv_output = csv.writer(open(self.outputfile, 'wb'), delimiter = ' ')
                
        #process truth file
        truthfile_file = csv.writer(open(self.truthfile), delimiter = ' ')
        self.test_target = np.empty((self.np_sample_test, ), dtype = np.int)        
        for i, ir in enumerate(truthfile_file):
            print i, ir
            self.test_target[i] = ir[1]
            if i >= self.np_sample_test:
                break

       
        print self.train_data.shape
        print self.train_target.shape
        
        print self.test_data.shape
        print self.test_target.shape

    pass

        
    def preprocessing(self):
        """
        preprocessing steps.
        """
        from sklearn import preprocessing
        
        """"step1: Standardization: got the z-score"""
        self.scaler = preprocessing.Scaler().fit(self.train_data + self.test_data)
        print "scaler parameters. mean: %s,  stand deviation: %s" % (self.scaler.mean_, self.scaler.std_)

        train_data = self.scaler.transform(self.train_data)
        test_data = self.scaler.transform(self.test_data)


        """analyze features using Univariate Feature Selection and SVM Weights"""
        #E = np.random.normal(size=(len(train_data), 2))
        #
        ## Add the noisy data to the informative features
        #x = np.hstack((train_data, E))
        #y = train_target
        #
        #pl.figure(1)
        #pl.clf()
        #
        #x_indices = np.arange(x.shape[-1])
        #
        ## Univariate feature selection
        #from sklearn.feature_selection import SelectFpr, f_classif
        ## As a scoring function, we use a F test for classification
        ## We use the default selection function: the 10% most significant
        ## features
        #
        #selector = SelectFpr(f_classif, alpha=0.1)
        #selector.fit(x, y)
        #scores = -np.log10(selector._pvalues)
        #scores /= scores.max()
        #pl.bar(x_indices-.45, scores, width=.3,
        #        label=r'Univariate score ($-Log(p_{value})$)',
        #        color='g')
        #
        ## Compare to the weights of an SVM
        #clf = svm.SVC(kernel='linear')
        #clf.fit(x, y)
        #
        #svm_weights = (clf.coef_**2).sum(axis=0)
        #svm_weights /= svm_weights.max()
        #pl.bar(x_indices-.15, svm_weights, width=.3, label='SVM weight',
        #        color='r')
        #
        #
        #pl.title("Comparing feature selection")
        #pl.xlabel('Feature number')
        #pl.yticks(())
        #pl.axis('tight')
        #pl.legend(loc='upper right')
        #pl.show()

        """Now we concluded that all features are correlated with the class, let's do the pca"""
        
        pca = PCA(n_components=80)
        selector = pca.fit(train_data)
        print pca.explained_variance_ratio_
        
        self.train_data = selector.transform(self.train_data)
        self.test_data = selector.transform(self.test_data)

    
    def feature_selection(self):
        """analyze features using:            
            filtering method: univariate feature selection 
            embeded method: SVM weights
        """
                
        # Add noisy feature
        E = np.random.normal(size=(len(self.train_data), 2))        
        x = np.hstack((self.train_data, E))
        y = self.train_target        
        pl.figure(1)
        pl.clf()
        
        x_indices = np.arange(x.shape[-1])
        
        from sklearn.feature_selection import SelectFpr, f_classif
        selector = SelectFpr(f_classif, alpha=0.1)
        selector.fit(x, y)
        scores = -np.log10(selector._pvalues)
        scores /= scores.max()
        pl.bar(x_indices-.45, scores, width=.3,
                label=r'Univariate score ($-Log(p_{value})$)',
                color='g')
       
        
        # Compare to the weights of an SVM
        clf = svm.SVC(kernel='linear')
        clf.fit(x, y)
        
        svm_weights = (clf.coef_**2).sum(axis=0)
        svm_weights /= svm_weights.max()
        pl.bar(x_indices-.15, svm_weights, width=.3, label='SVM weight',
                color='r')
        
        
        pl.title("Comparing feature selection")
        pl.xlabel('Feature number')
        pl.yticks(())
        pl.axis('tight')
        pl.legend(loc='upper right')
        pl.show()
        pass
    
    
    def model_selection(self):   
        """
        step 1: use svc with linear kernel, which tune the parameter using grid search. 
        """ 
        trainset, testset = iter(StratifiedKFold(self.train_target, 2)).next()        
        tuned_parameters = [{'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

        scores = [
            ('precision', precision_score),
            ('recall', recall_score),
        ]

        for score_name, score_func in scores:
            clf = GridSearchCV(SVC(C=1), tuned_parameters, score_func=score_func)
            clf.fit(self.train_data[trainset], self.train_target[trainset], cv=StratifiedKFold(self.train_target[trainset], 5))
            y_true, y_pred = self.train_target[testset], clf.predict(self.train_data[testset])
        
            print "Classification report for the best estimator: "
            print clf.best_estimator_
            print "Tuned for '%s' with optimal value: %0.3f" % (
                score_name, score_func(y_true, y_pred))
            print classification_report(y_true, y_pred)
            print "Grid scores:"
            
            from pprint import pprint
            pprint(clf.grid_scores_)
            print

        """
        step 2: train the model using all data. 
        """
        self.predict_model = clf.best_estimator_
        self.predict_model.fit(self.train_data, self.train_target)
        
#        kf = cross_validation.KFold(self.np_sample_train, k=20)
#        for train_index, test_index in kf:
#            X_train = self.train_data[train_index]
#            X_test = self.train_data[test_index]
#            y_train = self.train_target[train_index]
#            y_test = self.train_target[test_index]
#            
#            clf = SVC()
#            clf.fit(X_train, y_train) 
#            res = clf.predict(self.train_data)
#            
#            rightcount = 0
#            totalcount = 0
#            for i in range(len(res)):
#                totalcount += 1
#                if int(res[i]) == self.train_target[i]:
#                    rightcount += 1
#            
#            print rightcount/float(totalcount)


    def predict(self):  
        res = self.predict_model.predict(self.test_data)    
                   
        for i in range(len(res)):
            self.csv_output.writerow([self.test_data_aid[i], res[i]])
                    
        rightcount = 0
        totalcount = 0
        for i in range(len(self.test_data)):
            totalcount += 1
            if int(res[i]) == self.test_target[i]:
                rightcount += 1
        
        print rightcount/float(totalcount)
        pass



def main(options, args):
    qc = QClassifier(options.input, options.output, options.truthfile)
    qc.load_data()
    qc.preprocessing()
    qc.feature_selection()
    qc.model_selection()
    qc.predict()    
    pass



if __name__ == '__main__':
    global _log
    global mailer_inst
    
    option_parser = optparse.OptionParser()

    option_parser.add_option('-v', '--verbosity', metavar='LVL',
        action='store', type='int', dest='verbosity', default=3,
        help='debug verbosity level (0-3), default is 3')
    
    option_parser.add_option('-l', '--log-filename', metavar='LOG_FN',
        action='store', type='string', dest='log_filename', default=DEFAULTLOG,
        help='path to the base log filename. ')
    
    
    option_parser.add_option('-i', '--input', metavar='INPUT_FN',
        action='store', type='string', dest='input', default=INPUT,
        help='the input file of the model.')
    
    option_parser.add_option('-o', '--output', metavar='OUTPUT_FN',
        action='store', type='string', dest='output', default=OUTPUT,
        help='the output file of the model.')
    
    option_parser.add_option('-t', '--truthfile', metavar='TRUTH_FN',
        action='store', type='string', dest='truthfile', default=TRUTH,
        help='the truth file is used to verify the predition model.')
    
    
    (options, args) = option_parser.parse_args()

    #logging
    if options.verbosity == 0:
        llevel = logging.WARN
    elif options.verbosity == 1:
        llevel = logging.INFO
    elif options.verbosity == 2:
        llevel = logging.DEBUG
    elif options.verbosity >= 3:
        llevel = logging.DEBUG

    if options.log_filename:
        log_filename = os.path.realpath(options.log_filename)
        handler = logging.handlers.RotatingFileHandler(log_filename)
        print >> sys.stderr, 'logging to %s' % (log_filename,)
    else:
    # log to stdout if nothing is mentioned
        handler = logging.StreamHandler()

    handler.setFormatter(
        logging.Formatter(
        '%(pathname)s(%(lineno)d): [%(name)s] '
        '%(levelname)s %(message)s'))

    _log = logging.getLogger('')
    _log.addHandler(handler)
    _log.setLevel(level=llevel)
    _log.info('Quant Classifier Started')

    main(options, args)
    
    





