import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from time import time
from datetime import datetime
import tensorflow as tf
from tensorflow.python.ops.variables import global_variables_initializer

#Computes and returns:
#x_train, x_test, y_train, y_test
#x,y_train are the test set
#x,y_test are the holdout set aka validation set
#to try to avoid overfitting, we generate a random train/test split each time
def get_data():
        
    df = pd.read_csv('titanic_train.csv') #from kaggle
    
    #sanitize age and sex - fill out missing data
    femalemed = df[df['Sex'] == 'female']['Age'].median()
    malemed = df[df['Sex'] == 'male']['Age'].median()
    df['Age'] = df['Age'].replace(df[(df['Sex']=='male') & (df['Age'].isnull())]['Age'], malemed)
    df['Age'] = df['Age'].replace(df[(df['Sex']=='female') & (df['Age'].isnull())]['Age'], femalemed)
    
    df['fam'] = df['SibSp'] + df['Parch']
    
    #These are the features we care about
    features = ['Pclass', 'Sex', 'Age', 'Fare', 'fam', 'Embarked']
    
    
    #Categorical vars - we'll eventually one hot encode them
    catvars = ['Pclass', 'Sex', 'Embarked']
    
    #continuous variables - could be a number
    contvars = list(features)
    [contvars.remove(x) for x in catvars]
    
    X = df[features].fillna(0)
    y = df['Survived']
    
    for var in catvars:
        #replace a column containing a categorical var with N columns
        #containing its one hot encoding. Clearly, N = num of categories
        X = X.join(pd.get_dummies(X[var]))
        X = X.drop(var, axis=1)
    
    #scale continuous vars to a gaussian distribution
    scaler = StandardScaler()
    X[contvars] = scaler.fit_transform(X[contvars].values)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
     
    #Reshape Y - it's of shape (N,), tensorflow expects (N,1)
    return X_train.values, y_train.values.reshape(len(y_train),1), X_test.values, y_test.values.reshape(len(y_test),1)


#Batch helper function - returns batches of data for training
def train_next_batch(xt,yt,i):
    
    sz = xt.shape[0]
    step = 10
    
    i = i%sz
    n1 = i*step
    n2 = n1 + step
    
    if((n2 >= sz -1) or (n1 >= sz-1)):
        n2 = n2-n1
        n1 = 0
    
    bx = xt[n1:n2]
    by = yt[n1:n2]
    
    bx = bx.reshape(step,12)
    by = by.reshape(step,1)
    return bx,by

#Helper function for decaying learning rate
#We could theoretically encode this in tensorflow too
def getlr(maxlr,i):
    minlr = maxlr/30.0
    decay_speed = 2000.0
    rv = minlr + (maxlr - minlr) * math.exp(-i/decay_speed)
    return rv

#returns weights and biases tensors, given input dims
def wbhelper2(a,b):
    W = tf.Variable(tf.truncated_normal([a,b], stddev=0.1))
    bi = tf.Variable(tf.ones([b])/10)
    return W,bi

#Batch normalization function - THIS ONE PERFORMS WORSE!!!!
#input: logits of a layer (pre-activation)
#returns: batch-normalized logits, intended to be fed into activation function 
def mybn(Yl,is_test,offset,iteration):
    
    ema = tf.train.ExponentialMovingAverage(0.999,iteration)
    eps = 1e-5
    mean,variance = tf.nn.moments(Yl,[0])
    
    uma = ema.apply([mean,variance])  
    m = tf.cond(is_test, lambda:ema.average(mean), lambda:mean)
    v = tf.cond(is_test, lambda:ema.average(variance), lambda:variance)
    
    Ybn = tf.nn.batch_normalization(Yl, m, v, offset, None, eps)
    
    return Ybn,uma

#Dummy function that doesn't to batch norm - for ease of parametrization
def nobn(Yl,is_test,offset,iteration):
    return Yl, tf.no_op()

#Main function that implements a 3 layer fully connected neural network to fit input data
#xtr,ytr = training X and Y
#xts,yts = ditto for test
#M1, M2 are middle layer sizes
#bnf = batch norm function (could use a dummy batch norm here too to test without batch norm
def nn_sol(xtr,ytr,xts,yts,numepochs,trainpk,alpha,M1,M2,bnf):
    numiters = numepochs*xtr.shape[0]
    M0 = xtr.shape[1]   #works out of the box on 2-D arrays of input features
    ytr_reshaped = ytr.reshape(ytr.shape[0],1)   #necessary to reshape from (N,) to (N,1), dunno why tf can't handle that
    
    
    X = tf.placeholder(tf.float32, [None,M0])
    
    W1,b1 = wbhelper2(M0,M1)
    W2,b2 = wbhelper2(M1,M2)
    W3,b3 = wbhelper2(M2,1)
    
    acf = tf.nn.sigmoid #use sigmoid, can easily change to relu, etc.
    pkeep = tf.placeholder(tf.float32)
    iteration = tf.placeholder(tf.int32)
    is_test = tf.placeholder(tf.bool)
    
    #build each layer of NN, using weights and biases
    
    Y1l = tf.matmul(X,W1)
    Y1b,uma1 = bnf(Y1l,is_test,b1,iteration)
    Y1f = acf(Y1b)
    Y1 = tf.nn.dropout(Y1f, pkeep)
    
    Y2l = tf.matmul(Y1,W2)
    Y2b,uma2 = bnf(Y2l,is_test,b2,iteration)
    Y2f = acf(Y2b)
    Y2 = tf.nn.dropout(Y2f, pkeep)
    
    Y = tf.matmul(Y2,W3) + b3
    
    uma = tf.group(uma1,uma2)
    
    Y_ = tf.placeholder(tf.float32, [None,1])
    lr = tf.placeholder(tf.float32)
    
    #cross entropy
    xe = tf.reduce_sum(abs(Y - Y_))
    train_step = tf.train.AdamOptimizer(lr).minimize(xe)
    
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    
    is_correct = tf.equal(tf.round(Y),Y_)
    accuracy = tf.reduce_mean(tf.cast(is_correct,tf.float32))
    
    train_next_batch.i = 0
    for i in range(numiters):
        lra = getlr(alpha,i)
        
        bx,by = train_next_batch(xtr,ytr,i)
        train_data = {X:bx, Y_:by, lr:lra, pkeep: trainpk, is_test:False, iteration:i}    
        sess.run(train_step,feed_dict=train_data)
        sess.run(uma,train_data)
        
    
    #at this point, we have a trained model - test it on all data, train and hold out set
    tal = sess.run([accuracy,xe], feed_dict={X:xtr,Y_:ytr_reshaped,pkeep:1.0,is_test:True})
    hal = sess.run([accuracy,xe], feed_dict={X:xts, Y_:yts, pkeep: 1.0,is_test:True})
    
    print("training acc,loss: %.2f " % tal[0], round(tal[1]/xtr.shape[0],2))
    print("hold out acc,loss: %.2f " % hal[0], round(hal[1]/xts.shape[0],2))
    
    return hal[0]


#Wrapper function for repeating runs with a single set of parameters
def dorun(N, nepochs, M1, M2, alpha, trainpk, bn):
    print("Starting run with params:\nepochs: ", nepochs, "\nlayers: ", M1, "-", M2, "\nlr: ", alpha, "trainpk: ", trainpk, "\nbn: ", bn)
    
    avaccuracy = 0
    avtime = 0

    for i in range(N):
        starttime = datetime.now()
        xtr,ytr,xts,yts = get_data()
        soln = nn_sol(xtr,ytr,xts,yts,nepochs,trainpk,alpha,M1,M2,bn)
        tottime = datetime.now() - starttime
        
        print("run ", i+1, " of ", N, ":", soln,tottime.seconds)
        avaccuracy = avaccuracy + soln
        avtime = avtime + tottime.seconds
    
    avaccuracy = avaccuracy/N
    avtime = avtime/N
    
    print("average accuracy and training time for : \nepochs: ", nepochs, "\nlayers: ", M1, "-", M2, "\nlr: ", alpha, "\nbn: ", bn, ":\n", avaccuracy, avtime)


#Execution starts here
print("Welcome")
#dorun(N=10, nepochs=100, M1=32, M2=32, alpha=0.1, trainpk=0.5, bn=mybn)

#1 call of dorun does N runs with that param set
#It's then a simple matter to grep and sort output data to see which param set
#had the best performance
dorun(N=10, nepochs=100, M1=32, M2=32, alpha=0.1, trainpk=0.5, bn=nobn)

#dorun(N=10, nepochs=100, M1=320, M2=32, alpha=0.1, trainpk=0.5, bn=mybn)
dorun(N=10, nepochs=100, M1=320, M2=32, alpha=0.1, trainpk=0.5, bn=nobn)

#dorun(N=10, nepochs=100, M1=32, M2=12, alpha=0.1, trainpk=0.5, bn=mybn)
dorun(N=10, nepochs=100, M1=32, M2=12, alpha=0.1, trainpk=0.5, bn=nobn)

#dorun(N=10, nepochs=100, M1=32, M2=32, alpha=0.03, trainpk=0.5, bn=mybn)
dorun(N=10, nepochs=100, M1=32, M2=32, alpha=0.03, trainpk=0.5, bn=nobn)

#dorun(N=10, nepochs=100, M1=320, M2=32, alpha=0.03, trainpk=0.5, bn=mybn)
dorun(N=10, nepochs=100, M1=320, M2=32, alpha=0.03, trainpk=0.5, bn=nobn)

#dorun(N=10, nepochs=100, M1=32, M2=12, alpha=0.03, trainpk=0.5, bn=mybn)
dorun(N=10, nepochs=100, M1=32, M2=12, alpha=0.03, trainpk=0.5, bn=nobn)

"""
testy = sess.run(Y, feed_dict={X: xts, pkeep:1.0})
#print myy
predictions = np.round(testy)
plen = len(predictions)
predictions = predictions.reshape(plen,)
submission = pd.DataFrame({
    "PassengerId": testy["PassengerId"],
    "Survived": predictions
})

submission.to_csv("titanic-submission.csv", index=False)
"""


