

import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


class ExponentialDistribution():

    '''
        @param X the feature vectors x (n by d)
        @param T the vector (n by 1) amount of time until the next delay for the airports corresponding to entries of X
        @return beta (1 by d) the coefficeints used to derive this distribution from a feature vector,
                likelihood_vals the values of the likelihood during training, gradient_norm_vals, the norm of the gradient
                at each iteration of training
        Note: converged with learning_rate=.0000001
    '''
    def fit(self, X, T, regularization=1, learning_rate=.0000001, early_stopping=True, max_iter=35000, tol=.01):
        if len(X) == 0:
            return
        X = np.array(X)
        T = np.array(T)
        if T.ndim == 2:
            T = np.squeeze(T, axis=1)
        self.betas = np.zeros(len(X[0]))
        grad_norm = float("inf")
        num_iter = 0
        likelihood_vals = []
        gradient_norm_vals = []
        num_iters_with_decreasing_likelihood = 0
        while grad_norm > tol and num_iter < max_iter:
            # if num_iter == 100000:
            #     learning_rate *= 10
            # if num_iter == 500000:
            #     learning_rate *= 1000
            likelihood = self.log_likelihood(X, T, regularization)

            gradient = self.gradient(X, T, regularization)
            grad_norm = np.linalg.norm(gradient)

            likelihood_vals.append(likelihood)
            if len(likelihood_vals) > 2:
                if likelihood_vals[-1] < likelihood_vals[-2]:
                    num_iters_with_decreasing_likelihood += 1
                else:
                    num_iters_with_decreasing_likelihood = 0
            gradient_norm_vals.append(grad_norm)
            if early_stopping and num_iters_with_decreasing_likelihood >= 5:
                print "Stopped training early at iteration {} because likelihood stopped increasing...returning".format(num_iter)
                break
            #print "Iteration: {}, Likelihood = {}, Gradient Norm = {}".format(num_iter, likelihood, grad_norm)

            self.betas = self.betas + learning_rate * gradient
            num_iter += 1

        return (self.betas, likelihood_vals, gradient_norm_vals)



    # assumes X is an n by d matrix
    def log_likelihood(self, X, T, regularization=0):
        return X.dot(self.betas.transpose()).sum() - np.exp(self.betas.dot(X.transpose())).dot(T) - regularization * self.betas.dot(self.betas.transpose())

    # returns a d by 1 vector
    def gradient(self, X, T, regularization=0):
        # print np.expand_dims(np.exp(X.dot(self.betas.transpose())), axis=1).shape
        #print T.shape
        # if T.ndim > 1:
        #     T = np.squeeze(T, axis=1)
        # print "X:",X.shape
        # print "T:",T.shape
        # print "first term:",np.sum(X, axis=0).shape
        # print np.multiply(np.expand_dims(np.exp(X.dot(self.betas.transpose())), axis=1), T).shape
        # print X.transpose().dot(np.multiply(np.expand_dims(np.exp(X.dot(self.betas.transpose())), axis=1), T)).shape
        # print "reg shape:",(2 * regularization * np.absolute(self.betas).transpose()).shape
        # a = np.sum(X, axis=0)
        # #b = X.transpose().dot(np.multiply(np.expand_dims(np.exp(X.dot(self.betas.transpose())), axis=1), T))
        # print "exp shape:",np.exp(X.dot(self.betas.transpose())).shape
        # print "prod shape:",(np.exp(X.dot(self.betas.transpose())) * T).shape
        # print "const:",np.multiply(np.exp(X.dot(self.betas.transpose())), np.squeeze(T, axis=1)).shape
        # b = X.transpose().dot(np.multiply(np.exp(X.dot(self.betas.transpose())), np.squeeze(T, axis=1)))
        # print "b shape:",b.shape
        # print "a:",a
        # print "b: ",b
        # print "a - b:",(a - b)
        #return np.sum(X, axis=0) - X.transpose().dot(np.multiply(np.expand_dims(np.exp(X.dot(self.betas.transpose())), axis=1), T)).sum(axis=0) - 2 * regularization * np.absolute(self.betas).transpose()
        return np.sum(X, axis=0) - X.transpose().dot(np.multiply(np.exp(X.dot(self.betas.transpose())), T)) - 2 * regularization * np.absolute(self.betas).transpose()
        #print "Return shape:",(np.sum(X, axis=0) - X.transpose().dot(np.multiply(np.expand_dims(np.exp(X.dot(self.betas.transpose())), axis=1), T))).shape
        #return np.sum(X, axis=0).expand_dims(axis=1) - X.transpose().dot(np.multiply(np.exp(X.dot(self.betas.transpose())), T))

        #sys.exit(0)

    # returns an n by 1 vector
    def pdf(self, X, T):
        X = np.array(X)
        T = np.array(T)

        lam_vals = np.expand_dims(np.exp(X.dot(self.betas.transpose())), axis=1)
        return np.multiply(np.exp(-1.0 * np.multiply(lam_vals, T)), lam_vals)

    # returns an n by 1 vector
    def cdf(self, X, T):
        X = np.array(X)
        T = np.array(T)
        ''' FIX THIS USING SQUEEZE INSTEAD OF EXPAND DIMS'''
        lam_vals = np.exp(X.dot(self.betas.transpose()))
        return (1 - np.exp(-1.0 * np.multiply(lam_vals, T)))


    # @param X, T the test feature vectors and labels (number of hours until the next delay) respectively
    # @param num_hours the number of ROC curves to make, defaults to 24, representing a single day
    def makeROCCurvesForTestData(self, X, T, num_hours=5):
        X = np.array(X)
        T = np.array(T)
        plt.figure()
        for i in xrange(1, num_hours + 1):
            labels = T >= i # 1 if there is no delay within the next i hours, 0 otherwise
            probabilities = self.cdf(X, T)
            fpr, tpr, _ = roc_curve(labels, probabilities)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label='Predicting Delays in the next {} hours:'.format(i) + ' AUC (area = %0.2f)' % roc_auc)

        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title("Predicting Delays for the Next {} Hours".format(num_hours))       
        plt.legend(loc="lower right")

        plt.savefig("./ROCCurvesForDelayPredictionUpTo{}Hours".format(num_hours), format="png")

def TestExponentialEstimation():
    n = 100
    d = 15
    fake_X = np.random.randint(0, 2, size=(n, d))
    fake_T = np.random.randint(1, 10, size=(n, 1))

    dist = ExponentialDistribution()

    # set early_stopping to True to stop training at optimal likelihood
    betas, likelihood_vals, gradient_norm_vals = dist.fit(fake_X, fake_T, regularization=1, early_stopping=False, max_iter=10000)
    iters = range(len(likelihood_vals))

    #print "PDF:",dist.pdf(fake_X, fake_T)
    #print "CDF:",dist.cdf(fake_X, fake_T)
    print "Final Betas:",betas
    print "Final Likelihood:",likelihood_vals[-1]
    print "Final Gradient Norm:",gradient_norm_vals[-1]

    #dist.makeROCCurvesForTestData(fake_X, fake_T)
    plt.figure(1)
    plt.subplot(211)
    plt.plot(iters, likelihood_vals, label="Log Likelihood")
    #plt.plot(iters, gradient_norm_vals, label="Gradient Norm")
    plt.title("Log Likelihood for Estimation with No Regularization")
    plt.xlabel("Iteration Number")
    
    plt.subplot(212)
    plt.plot(iters, gradient_norm_vals, label="Gradient Norm")
    plt.xlabel("Iteration Number")
    #plt.ylabel("Value")
    plt.title("Gradient Norm for Estimation with No Regularization")
    #plt.legend(loc="upper right")
    plt.show()


<<<<<<< Updated upstream
def TestEstimationOnTrainingData():
    X, T = loadTrainingData()
    dist = ExponentialDistribution()

    betas, likelihood_vals, gradient_norm_vals = dist.fit(X, T, learning_rate=1e-10, regularization=0, early_stopping=False, max_iter=1000000)
    iters = range(len(likelihood_vals))
    print "Final betas:",betas
    print "Final Likelihood:",likelihood_vals[-1]
    print "Final Gradient Norm:",gradient_norm_vals[-1]

    dist.makeROCCurvesForTestData(X, T)
    # plt.figure(1)
    # plt.subplot(211)
    # plt.plot(iters, likelihood_vals, label="Log Likelihood")
    # #plt.plot(iters, gradient_norm_vals, label="Gradient Norm")
    # plt.title("Log Likelihood for Estimation with No Regularization")
    # plt.xlabel("Iteration Number")
    
    # plt.subplot(212)
    # plt.plot(iters, gradient_norm_vals, label="Gradient Norm")
    # plt.xlabel("Iteration Number")
    # #plt.ylabel("Value")
    # plt.title("Gradient Norm for Estimation with No Regularization")
    # #plt.legend(loc="upper right")
    # plt.show()


def loadTrainingData():
    f = open("./training_data/X.txt", "r")
    X = []
    T = []
    for line in f:
        l = [float(x) for x in line.split(',')]
        X.append(l)
    f.close()
    f = open("./training_data/T.txt", "r")
    for line in f:
        T.append(float(line))
    f.close()

    print len(X)
    print len(T)
    return X, T

if __name__ == '__main__':
  TestExponentialEstimation()

