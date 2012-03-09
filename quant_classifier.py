'''
Created on Mar 2, 2012 

Here is an example to run this program: 
>> python quant_classifier.py -i ~/inputFile -o ~/outputFile -t ~/truthFile -l ./logFile

-i :   the path of the input file
-o:   the path of the output file
-t:    the path of the truth file (optional)
-l:    the path of the log file. 


comment the line if you want to use the full feature sets:       qc.feature_selection()
uncomment the line if want to a simple feature analysis:         #qc.feature_analysis()

'''


import optparse
import logging.handlers
import sys

import numpy as np
import pylab as pl
import os
import csv
#from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.metrics import zero_one
from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import RFECV

INPUT = '~/inputB'
OUTPUT = '~/output'
TRUTH = '~/outputB'
DEFAULTLOG = './qlog'


class QClassifier():
    def __init__(self, input, output, truthfile):
        if not os.path.exists(input) :
            _log.error('Cannot find the input %s or output %s' % (input, output))
            print 'Error: Cannot find the input %s or output %s' % (input, output)
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
        
        self.featuremask = None
    
    def load_data(self):
        """
        load data from disk
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
                
        #process truth file, if the truth file is provided. 
        if self.truthfile and os.path.exists(self.truthfile):
            truthfile_file = csv.reader(open(self.truthfile), delimiter = ' ')
            self.test_target = np.empty((self.np_sample_test, ), dtype = np.int)        
            for i, ir in enumerate(truthfile_file):
                self.test_target[i] = ir[1]
                if i >= self.np_sample_test:
                    break
       
        _log.info("number of trainning example is: %d" %(self.np_sample_train))
        _log.info("number of dimensions is: %d" %(self.np_feature))
        _log.info("number of testing example is: %d" %(self.np_sample_test))

        
    def preprocessing(self):
        """
        preprocessing steps: Standardize all features using z-score
        """
        from sklearn import preprocessing
        self.scaler = preprocessing.Scaler().fit(self.train_data)
        _log.info("scaler mean: %s" % (self.scaler.mean_))
        _log.info("scaler stand deviation: %s" % (self.scaler.std_))
        
        self.train_data = self.scaler.transform(self.train_data)
        self.test_data = self.scaler.transform(self.test_data)

    
    def feature_analysis(self):
        """analyze features using:     
            1. generate two random features and mit them with informative features.
            2. apply a linear classifier and get the weights of all features. 
        """                
        # Add one noisy feature for comparison
        E = np.random.normal(size=(len(self.train_data), 2))        
        x = np.hstack((self.train_data, E))
        y = self.train_target        
        pl.figure(1)
        pl.clf()
              
        x_indices = np.arange(x.shape[-1])               
        clf = svm.SVC(kernel='linear')
        clf.fit(x, y)        
        svm_weights = (clf.coef_**2).sum(axis=0)
        svm_weights /= svm_weights.max()
        pl.bar(x_indices-.15, svm_weights, width=.3, label='SVM weight',color='b')
                        
        pl.title("feature weights analysis")
        pl.xlabel('Feature number')
        pl.yticks(())
        pl.axis('tight')
        pl.show()

    
    def feature_selection(self):
        """
        select features using SVM-RFE. see papar :
            [1] Guyon, I., Weston, J., Barnhill, S., & Vapnik, V., "Gene selection for cancer classification 
            using support vector machines", Mach. Learn., 46(1-3), 389--422, 2002.
        """
        svc = SVC(kernel="linear")
        rfecv = RFECV(estimator=svc,
        step=1,
        cv=StratifiedKFold(self.train_target, 2),
        loss_func=zero_one)
        rfecv.fit(self.train_data, self.train_target)
        
        self.featuremask = rfecv.support_
        _log.info("after svm-rfe, the feature selection mask is shown as following:")
        _log.info(self.featuremask)
        self.train_data = self.train_data[:, self.featuremask] 
        self.test_data = self.test_data[:, self.featuremask]          
        pass

    
    def model_selection(self):   
        """
        step 1: use svc with linear kernel, which tunes Penalty C using grid search. 
        """ 
        tuned_parameters = [{'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
        clf = GridSearchCV(SVC(C=1), tuned_parameters, score_func=f1_score)
        clf.fit(self.train_data, self.train_target, cv=StratifiedKFold(self.train_target, 5))
        
        _log.info('Best SVC Classifier is:')
        _log.info(clf.best_estimator_)
        
        """
        step 2: train the model using all data. 
        """
        self.predict_model = clf.best_estimator_
        self.predict_model.fit(self.train_data, self.train_target)
    
    
    def model_without_selection(self):           
        """
        to save time, skip the grid search.
        """
        self.predict_model = svm.SVC(kernel='linear', C=1)
        self.predict_model.fit(self.train_data, self.train_target)
        
        

    def predict(self):  
        res = self.predict_model.predict(self.test_data)                   
        for i in range(len(res)):
            if int(res[i]) == 1:
                ostr = '+1'
            else:
                ostr = '-1'
            self.csv_output.writerow([self.test_data_aid[i], ostr])
        
        #if the truthfile is provided, we can estimate the accuracy.
        if self.test_target != None:            
            rightcount = 0
            totalcount = 0
            for i in range(len(self.test_data)):
                totalcount += 1
                if int(res[i]) == self.test_target[i]:
                    rightcount += 1            
            print "Total number of queries is %d, and number of right predictions is %d" %(totalcount, rightcount)
            print "The accuracy againt the truth file is: %f" % (rightcount/float(totalcount))
            print
            from sklearn.metrics import classification_report
            print classification_report(self.test_target, res)


def main(options, args):
    print
    print "==> initilizing the classifier, reading data files..."
    qc = QClassifier(options.input, options.output, options.truthfile)
    qc.load_data()
    print
    
    print "==> Preprocessing data by applying z-score scaler..."
    qc.preprocessing()
    print
    
    print "==> Feature analysis...skiped (uncomment the line 'qc.feature_analysis()' to activate feature analysis)"
    #qc.feature_analysis()
    print
    
    print "==> Conducting feature selection by applying svm-rfe ..."
    qc.feature_selection()
    print
    
    print "==> uncomment the line '#qc.model_selection()' to conduct model selection by applying gridsearch..."
    #qc.model_selection()
    qc.model_without_selection()
    print
    
    print "==> Predicting queries..."
    qc.predict()  
    print  

    print "==> The job is done. Please check the output file."
    print

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
        handler = logging.StreamHandler()

    handler.setFormatter(
        logging.Formatter(
        '%(pathname)s(%(lineno)d): [%(name)s] '
        '%(levelname)s %(message)s'))

    _log = logging.getLogger('')
    _log.addHandler(handler)
    _log.setLevel(level=llevel)
    _log.info('Classifier Started')

    main(options, args)
    
    





