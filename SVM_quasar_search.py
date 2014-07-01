"""
    Identify quasars behind M31 using SVM and logistic regression, including 
    1. Formatting the data
    2. Feature selection
    3. Feature visualization
    4. Parameter tuning in classifiers
    5. Classification
    6. Validation
 
    -- Input files:
    1. Variable candidates: /Users/litfeix/m31qso/data/
       10984 stats_variable_PTFIDE-coadd_wise_ref_SDSS_Massey_filled.txt
    
    2. Quasar and stellar catalogs: /Users/litfeix/m31qso/catalogs/
        39 known_qso_catalog_f100043.txt
        77 qso_catalog_f100043.txt
      5369 stellar_catalog.txt

        56 quasars detected in variable-stats-wise-ref-sdss-massey (out of 76)
       215 stars detected in variable-stats-wise-ref-sdss-massey (out of 5368)

    -- Output file: 
     M31_quasars.txt
 
     Sumin Tang, Oct 6, 2013; April 2014
"""

import pylab
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn import cross_validation
from sklearn import svm
from sklearn import feature_selection
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import RFECV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import metrics
from sklearn.lda import LDA
import time
tic = time.time()


matplotlib.rcParams.update({'font.size': 14})
matplotlib.rc("axes", linewidth=1.0)

basedir = '/Users/litfeix/m31qso/data/'
infile = basedir + 'stats_variable_PTFIDE-coadd_wise_ref_SDSS_Massey_filled.txt'  

catdir = '/Users/litfeix/m31qso/catalogs/'
catqso = catdir + 'qso_catalog_f100043.txt'
catqso2 = catdir + 'known_qso_catalog_f100043.txt'
catstar = catdir + 'stellar_catalog.txt'
catfailedqso = catdir + 'myfailed-qso.txt'

# output files:
outfile = basedir + 'M31_quasars.txt'
regionfile = basedir + 'M31_quasars.reg'


# =====================================
# =====    Part 1. Format data    =====
# =====================================

# ===== Load qsos and stars
raqso, decqso = np.loadtxt(catqso, unpack = True, skiprows = 1)
rastar, decstar = np.loadtxt(catstar, unpack=True, skiprows = 1)
raqso2, decqso2 = np.loadtxt(catqso2, unpack = True, skiprows = 1)
rafqso, decfqso, magfqso = np.loadtxt(catfailedqso, unpack = True, skiprows = 1)

# ==== Load the variable-stats-wise-ref-sdss-massey table
dt = [('ra', 'f'), ('dec', 'f'), ('ccdidp', 'i'), ('objidp', 'i'), ('rap', 'f'), ('decp','f'),  ('nallp', 'i'),  ('lcmedianp', 'f'),  ('lcrmsp', 'f'),  ('lcerrp', 'f'),  ('rchi2p', 'f'), ('lcampp', 'f'),  ('lcamp1p', 'f'),  ('lcamp2p', 'f'),  ('lcamp5p', 'f'),  ('amp1op', 'f'),  ('amp2op', 'f'),  ('amp3op', 'f'),  ('amp1dp', 'f'),  ('amp2dp', 'f'),  ('amp3dp', 'f'),  ('nd1p', 'i'),  ('nd2p', 'i'),  ('nd3p', 'i'),  ('nd4p', 'i'),  ('nd5p', 'i'),  ('ndc1p', 'i'),  ('ndc2p', 'i'),  ('ndc3p', 'i'),  ('ndc4p', 'i'),  ('ndc5p', 'i'),  ('no1p', 'i'),  ('no2p', 'i'),  ('no3p', 'i'),  ('no4p', 'i'),  ('no5p', 'i'),  ('noc1p', 'i'),  ('noc2p', 'i'),  ('noc3p', 'i'),  ('noc4p', 'i'),  ('noc5p', 'i'), ('rmsratio1p', 'f'), ('rmsratio2p', 'f'),  ('rmsratio3p', 'f'), ('Sp', 'f'), ('Kp', 'f'), ('JBp', 'f'), ('qchi2p', 'f'), ('qchi2qsop', 'f'), ('qchi2nonp', 'f'), ('qsigvarp', 'f'), ('qsigqsop', 'f'), ('qsignonp', 'f'), ('varprobp', 'f'), ('ccdidc', 'i'), ('rac', 'f'), ('decc','f'),  ('nallc', 'i'), ('ngoodc', 'i'), ('rb2_maxc', 'f'), ('rb2_medianc', 'f'), ('lcmedianc', 'f'),  ('lcrmsc', 'f'),  ('lcerrc', 'f'),  ('rchi2c', 'f'), ('lcampc', 'f'),  ('lcamp1c', 'f'),  ('lcamp2c', 'f'),  ('lcamp5c', 'f'),  ('amp1oc', 'f'),  ('amp2oc', 'f'),  ('amp3oc', 'f'),  ('amp1dc', 'f'),  ('amp2dc', 'f'),  ('amp3dc', 'f'),  ('nd1c', 'i'),  ('nd2c', 'i'),  ('nd3c', 'i'),  ('nd4c', 'i'),  ('nd5c', 'i'),  ('ndc1c', 'i'),  ('ndc2c', 'i'),  ('ndc3c', 'i'),  ('ndc4c', 'i'),  ('ndc5c', 'i'),  ('no1c', 'i'),  ('no2c', 'i'),  ('no3c', 'i'),  ('no4c', 'i'),  ('no5c', 'i'),  ('noc1c', 'i'),  ('noc2c', 'i'),  ('noc3c', 'i'),  ('noc4c', 'i'),  ('noc5c', 'i'), ('rmsratio1c', 'f'), ('rmsratio2c', 'f'),  ('rmsratio3c', 'f'), ('Sc', 'f'), ('Kc', 'f'), ('JBc', 'f'), ('qchi2c', 'f'), ('qchi2qsoc', 'f'), ('qchi2nonc', 'f'), ('qsigvarc', 'f'), ('qsigqsoc', 'f'), ('qsignonc', 'f'), ('varprobc', 'f'), ('wisedsep', 'f'), ('w1', 'f'), ('w1err', 'f'), ('w1mw2', 'f'), ('w2mw3', 'f'), ('w3mw4', 'f'), ('refdsep', 'f'), ('refmagmw1', 'f'), ('sdssdsep', 'f'), ('sdssr', 'f'), ('umg', 'f'), ('gmr', 'f'), ('rmi', 'f'), ('imz', 'f'), ('masseydsep', 'f')]
data = np.loadtxt(infile, unpack=False, dtype = dt, skiprows = 1)
nrow = len(data['ra'])


# ==== format data: features
featurenames = ['nallp', 'lcrmsp', 'rchi2p', 'lcampp', 'lcamp1p', 'lcamp2p', 'lcamp5p', 'amp1op', 'amp2op', 'amp3op', 'amp1dp', 'amp2dp', 'amp3dp', 'nd1p', 'nd2p', 'nd3p', 'nd4p', 'ndc1p', 'ndc2p', 'ndc3p', 'ndc4p', 'no1p', 'no2p', 'no3p', 'no4p', 'noc1p', 'noc2p', 'noc3p', 'noc4p', 'rmsratio1p', 'rmsratio2p', 'rmsratio3p', 'Sp', 'Kp', 'JBp', 'qchi2p', 'qchi2qsop', 'qchi2nonp', 'varprobp', 'nallc', 'ngoodc', 'lcrmsc', 'rchi2c', 'lcampc', 'lcamp1c', 'lcamp2c', 'lcamp5c', 'amp1oc', 'amp2oc', 'amp3oc', 'amp1dc', 'amp2dc', 'amp3dc', 'nd1c', 'nd2c', 'nd3c', 'nd4c', 'nd5c', 'ndc1c', 'ndc2c', 'ndc4c', 'ndc5c', 'no1c', 'no2c', 'no3c', 'no4c', 'no5c', 'noc1c', 'noc2c', 'noc4c', 'noc5c', 'rmsratio1c', 'rmsratio2c', 'rmsratio3c', 'Sc', 'Kc', 'JBc', 'qchi2c', 'qchi2qsoc', 'qchi2nonc', 'varprobc', 'w1', 'w1mw2', 'w2mw3', 'w3mw4', 'refmagmw1', 'umg', 'gmr', 'rmi', 'imz']
# featurenames = ['qsigqsop', 'rmsratio1p', 'rmsratio2p', 'rmsratio3p', 'qsigqsoc', 'w1mw2', 'gmr']
"""
1. ndc5p, feature 22 (nan)
2. noc5p, feature 32 (nan)
3. ndc3c, feature 64 (nan)
4. noc3c, feature 74 (nan)
5. no5p, feature 27 (nan)
6. nd5p, feature 17 (nan)
"""

X_data0 = np.zeros((nrow, len(featurenames)))
X_data = np.zeros((nrow, len(featurenames)))
for i in range(len(featurenames)):
    ftname = featurenames[i]
    datacolumn = data[ftname]
    datacolumn[(datacolumn<-900)| (datacolumn>9000)] = 0
    datacolumn[np.isnan(datacolumn)] = 0
    X_data0[:,i] = datacolumn
    ix = np.where((data[ftname]>-90) & (data[ftname]<9000))[0]
    xmean = np.mean(data[ftname][ix])
    xstd = np.std(data[ftname][ix])
    if xstd<0.01:
        xstd = 1.
    X_data[:,i] = (datacolumn - xmean)/xstd
    

print X_data.shape
#print X_data.mean(axis=0)
#print X_data.std(axis=0)

# === scale the data to mean of 0 and variance of 1
#X_data = preprocessing.scale(X_data0)

# ==== format training set
myflag = -1.*np.ones(nrow) # -1 for non-labeled objects
for i in range(nrow):
    ra1 = data['ra'][i]
    dec1 = data['dec'][i]
    dist = (((raqso-ra1)*np.cos(dec1*np.pi/180.))**2 + (decqso-dec1)**2 )**0.5*3600.0
    ix = np.where(dist<1)[0]
    nmatch1 = len(ix)
    
    dist = (((rastar-ra1)*np.cos(dec1*np.pi/180.))**2 + (decstar-dec1)**2 )**0.5*3600.0
    ix = np.where(dist<1)[0]
    nmatch2 = len(ix)
    if nmatch1 == 1:
        myflag[i] = 1  # 1 for quasars
    elif nmatch2 == 1:
        myflag[i] = 0  # 0 for stars

# All labeled objects
ix = np.where(myflag>=0)[0]
data_labeled = data[ix]
Y_labeled = myflag[ix]
n_labeled = len(ix)
X_labeled = np.zeros((n_labeled, len(featurenames)))
for i in range(len(featurenames)):
    ftname = featurenames[i]
    X_labeled[:,i] = data_labeled[ftname]

# All un-labeled objects
ix = np.where(myflag<0)[0]
data_unlabeled = data[ix]
Y_unlabeled = myflag[ix]
n_unlabeled = len(ix)
X_unlabeled = np.zeros((n_unlabeled, len(featurenames)))
for i in range(len(featurenames)):
    ftname = featurenames[i]
    X_unlabeled[:,i] = data_unlabeled[ftname]


# Quasars
ix = np.where(myflag==1)[0]
data_qso = data[ix]
Y_qso = myflag[ix]
n_qso = len(ix)
X_qso = np.zeros((n_qso, len(featurenames)))
for i in range(len(featurenames)):
    ftname = featurenames[i]
    X_qso[:,i] = data_qso[ftname]

# Stars
ix = np.where(myflag==0)[0]
data_star = data[ix]
Y_star = myflag[ix]
n_star = len(ix)
X_star = np.zeros((n_star, len(featurenames)))
for i in range(len(featurenames)):
    ftname = featurenames[i]
    X_star[:,i] = data_star[ftname]

print "Number of qsos: %i" % n_qso
print "Number of stars: %i" % n_star
print "Number of qsos + stars: %i" % (n_qso + n_star)

# Quasars known before my study:
# ==== format training set
myflag2 = -1.*np.ones(nrow) # -1 for non-labeled objects
for i in range(nrow):
    ra1 = data['ra'][i]
    dec1 = data['dec'][i]
    dist = (((raqso2-ra1)*np.cos(dec1*np.pi/180.))**2 + (decqso2-dec1)**2 )**0.5*3600.0
    ix = np.where(dist<1)[0]
    nmatch1 = len(ix)
    if nmatch1 == 1:
        myflag2[i] = 1  # 1 for quasars

ix = np.where(myflag2==1)[0]
data_qso2 = data[ix]
Y_qso2 = myflag[ix]
n_qso2 = len(ix)
X_qso2 = np.zeros((n_qso2, len(featurenames)))
for i in range(len(featurenames)):
    ftname = featurenames[i]
    X_qso2[:,i] = data_qso2[ftname]


"""
ix = np.where(data['w1mw2']>0.8)[0]
print 'Number of objects with w1-w2>0.8: %i (out of %i)' % (len(ix), len(data))
ix = np.where(data_qso['w1mw2']>0.8)[0]
print 'Number of qsos with w1-w2>0.8: %i (out of %i)' % (len(ix), len(data_qso))
ix = np.where(data_star['w1mw2']>0.8)[0]
print 'Number of stars with w1-w2>0.8: %i (out of %i)' % (len(ix), len(data_star))

ix = np.where((data['w1mw2']>0.8) & (data['qsigqsop']>1))[0]
print 'Number of objects with w1-w2>0.8 and sigqso>1: %i (out of %i)' % (len(ix), len(data))
ix = np.where((data_qso['w1mw2']>0.8) & (data_qso['qsigqsop']>1))[0]
print 'Number of qsos with w1-w2>0.8 and sigqso>1: %i (out of %i)' % (len(ix), len(data_qso))
ix = np.where((data_star['w1mw2']>0.8) & (data_star['qsigqsop']>1))[0]
print 'Number of stars with w1-w2>0.8 and sigqso>1: %i (out of %i)' % (len(ix), len(data_star))
"""


# ===========================================
# =====    Part 2. Feature selection    =====
# ===========================================

# Too many features, and too few objects in the training set; need to reduce the number of features


# ==== 2.1 Feature importances with forests of trees

ntree = 1000
forest = ExtraTreesClassifier(n_estimators=1000, random_state=0)

forest.fit(X_labeled, Y_labeled)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking (random forest):")
for f in range(10):
    print("%d. %s, feature %d (%f)" % (f + 1, featurenames[indices[f]], indices[f], importances[indices[f]]))

# Plot the feature importances of the top N features
Ntop = 20
figname = 'feature-importance-top' + str(Ntop) + '_forest.png' 
fig = plt.figure()
ax = fig.add_subplot(111)
pylab.ylabel("Feature Importances", fontsize = 18)
pylab.bar(range(Ntop), importances[indices[0:Ntop]], color="r", yerr=std[indices[0:Ntop]]/(ntree**0.5), align="center")
ax.set_xticks(range(Ntop))
topfeature = []
for i in range(Ntop):
    topfeature.append(featurenames[indices[i]])
print topfeature
xtickNames = plt.setp(ax, xticklabels=topfeature)
plt.setp(xtickNames, rotation=45, fontsize=8)
ax.set_xlim([-1, Ntop])
ax.text(Ntop*0.2, importances[indices[0]], 'From random forest (%i trees; Gini criterion)' % ntree, fontsize = 12)
pylab.savefig(basedir + figname)
pylab.close()


# ==== 2.2 Feature selection based on univariate feature selection
kfeature = len(Y_labeled)
featureSelector = SelectKBest(score_func=feature_selection.f_classif,k=kfeature)
featureSelector.fit(X_labeled,Y_labeled)
importances = featureSelector.scores_
indices = np.argsort(importances)[::-1]


# Print the feature ranking
print("Feature ranking (selectkbest):")
for f in range(10):
    print("%d. %s, feature %d (%f)" % (f + 1, featurenames[indices[f]], indices[f], importances[indices[f]]))


# Plot the feature importances of the top N features
Ntop = 20
figname = 'feature-importance-top' + str(Ntop) + '_selectkbest.png' 
fig = plt.figure()
ax = fig.add_subplot(111)
pylab.ylabel("Feature Importances", fontsize = 18)
pylab.bar(range(Ntop), importances[indices[0:Ntop]], color="r", align="center")
ax.set_xticks(range(Ntop))
topfeature = []
for i in range(Ntop):
    topfeature.append(featurenames[indices[i]])
print topfeature
xtickNames = plt.setp(ax, xticklabels=topfeature)
plt.setp(xtickNames, rotation=45, fontsize=8)
ax.set_xlim([-1, Ntop])
ax.text(Ntop*0.2, importances[indices[0]], 'From SelectKBest', fontsize = 12)
pylab.savefig(basedir + figname)
pylab.close()


# ==== 2.3 cross-validation using logistic regression and support vector machine; no parameter tuning yet
nftuse = len(featurenames)
t0 = time.time()
cv = cross_validation.KFold(X_labeled.shape[0], 10, shuffle=True)#, random_state=0)
clf1 = LogisticRegression(class_weight='auto')
clf2 = svm.SVC(kernel='linear', class_weight='auto')
clf3 = svm.SVC(kernel='rbf', class_weight='auto')
score_lr = np.zeros(nftuse)
score_svmlinear = np.zeros(nftuse)
score_svmrbf = np.zeros(nftuse)
topfeature2 = []
for i in range(nftuse):
    kfeature = i + 1
    featureSelector = SelectKBest(score_func=feature_selection.f_classif, k=kfeature)
    featureSelector.fit(X_labeled,Y_labeled)
    ixselect = featureSelector.get_support(indices = True)
    if i<10:
        print "Top %i features: " % kfeature
        print ixselect
        print np.asarray(featurenames)[ixselect]  
    
    # format data using selected features:
    X_labeled2 = X_labeled[:, ixselect]

    # logistic regression 
    score_lr[i] = np.mean(cross_validation.cross_val_score(clf1, X_labeled2, Y_labeled, cv=cv))

    # support vector machine
    score_svmlinear[i] = np.mean(cross_validation.cross_val_score(clf2, X_labeled2, Y_labeled, cv=cv))
    score_svmrbf[i] = np.mean(cross_validation.cross_val_score(clf3, X_labeled2, Y_labeled, cv=cv))

t1 = time.time()
print "Time elapsed in feature selection: %.2f" % (t1 - t0)

figname = 'feature_selection_score-vs-nfeature_test.eps'
fig = plt.figure()
ax = fig.add_subplot(111)
kfeature = np.linspace(1, nftuse, nftuse)
ax.plot(kfeature, score_lr, 'r-o', markersize = 5)
ax.plot(kfeature, score_svmlinear, 'g-o', markersize = 5) 
ax.plot(kfeature, score_svmrbf, 'b-o', markersize = 5) 
ax.legend(('Logistic Regression', 'SVM Linear', 'SVM RBF'), numpoints = 1, prop={'size':10}, loc='lower left')
ax.set_xlabel('Number of Features Used', fontsize = 18)
ax.set_ylabel('Cross Validation Score', fontsize = 18)
pylab.savefig(basedir + figname)
plt.close()


# =======================================================
# =====        Part 3. Feature Visualization        =====
# =======================================================

# Linear Discriminant Analysis (LDA)
lda = LDA(n_components=2)
lda.fit(X_labeled, Y_labeled)
X_lda = lda.transform(X_labeled)

figname = 'lda_2components.png'
fig = plt.figure()
ax = fig.add_subplot(111)
ix = np.where(Y_labeled==0)[0]
ax.plot(X_lda[ix, 0], X_lda[ix, 1]+np.random.randn(len(ix))*0.1, 'r.', markeredgecolor='r', markersize = 8, alpha=0.8)
ix = np.where(Y_labeled==1)[0]
ax.plot(X_lda[ix, 0], X_lda[ix, 1]+np.random.randn(len(ix))*0.1, 'g.', markeredgecolor='g', markersize = 8, alpha=0.8)
ax.legend(('stars', 'quasars'), numpoints = 1, prop={'size':14}, loc='lower left')
ax.set_xlabel('Component 1', fontsize = 18)
ax.set_ylabel('Component 2', fontsize = 18)
#ax.set_xlim([-2, 2])
#ax.set_ylim([-2, 2])
ax.text(-6, 35, "Linear Discriminant Analysis (LDA) Projection")
pylab.savefig(basedir + figname)
plt.close()


# plot the best two features
kfeature = 2
featureSelector = SelectKBest(score_func=feature_selection.f_classif,k=kfeature)
featureSelector.fit(X_labeled,Y_labeled)
importances = featureSelector.scores_
indices = np.argsort(importances)[::-1]
ft1 = featurenames[indices[0]]
ft2 = featurenames[indices[1]]

figname = 'best-2-features.png'
fig = plt.figure()
ax = fig.add_subplot(111)
#ax.plot(data[ft1], data[ft2], 'k.', markersize = 3)
ax.plot(data_star[ft1], data_star[ft2], 'r.', markeredgecolor='r', markersize = 8)
ax.plot(data_qso[ft1], data_qso[ft2], 'g.', markeredgecolor='g', markersize = 8)
ax.legend(('stars', 'quasars'), numpoints = 1, prop={'size':14}, loc='upper right')
ax.set_xlabel('Feature 1', fontsize = 18)
ax.set_ylabel('Feature 2', fontsize = 18)
ax.set_ylim([-0.5, 3])
pylab.savefig(basedir + figname)
plt.close()



# ===============================================================================
# =====    Part 4. Feature selection and parameter tuning in classifiers    =====
# ===============================================================================


# Method 1: using recursive feature elimination
# Create the RFE object and compute a cross-validated score.

# === Logistic Regression; It takes 10min
t0 = time.time()
clf = LogisticRegression(class_weight='auto')

h = open(basedir + 'score-vs-nfeature-lr.txt', 'w')
h.write("# nfeature, score\n")

niter = 50
for i in range(niter):
    cv = cross_validation.KFold(X_labeled.shape[0], 10, shuffle=True)
    rfecv = RFECV(estimator=clf, step=1, cv=cv, scoring='accuracy')
    rfecv.fit(X_labeled,Y_labeled)
    #print("Optimal number of features : %d" % rfecv.n_features_)

    for j in range(len(rfecv.grid_scores_)):
        h.write("%i  %f\n" % (j+1, rfecv.grid_scores_[j]))
h.close()

t1 = time.time()
print "Time elapsed in feature selection (Logistic Regression): %.2f" % (t1 - t0)


# === SVM Linear; It takes 1h for niter = 4
t0 = time.time()
h = open(basedir + 'score-vs-nfeature-svmlinear.txt', 'w')
h.write("# C, nfeature, score\n")

Cgrid = 10.**np.linspace(-2, 2, 11)
ncgrid = len(Cgrid)
niter = 20
featurerank = np.zeros((X_labeled.shape[1], ncgrid, niter))
for i in range(niter):
    cv = cross_validation.KFold(X_labeled.shape[0], 10, shuffle=True)    
    for ii in range(ncgrid):
        myc = Cgrid[ii]
        clf = svm.SVC(C = myc, kernel='linear', class_weight='auto')
        rfecv = RFECV(estimator=clf, step=1, cv=cv, scoring='accuracy')
        rfecv.fit(X_labeled,Y_labeled)

        for j in range(len(rfecv.grid_scores_)):
            h.write("%f  %i  %f\n" % (myc, j+1, rfecv.grid_scores_[j]))
            featurerank[j, ii, i] = rfecv.ranking_[j]

h.close()

avgrank = featurerank.mean(axis=2)
fname = basedir + 'feature-ranking-svmlinear.txt'
h = open(fname, 'w')
np.savetxt(fname, avgrank)
h.close()
t1 = time.time()
print "Time elapsed in feature selection (SVM Linear): %.2f" % (t1 - t0)

# load results for logistic regression
nfeature, score = np.loadtxt(basedir + 'score-vs-nfeature-lr.txt', unpack=True, skiprows=1)
fnlr = np.linspace(1, max(nfeature)+1, max(nfeature))
myscorelr = np.zeros(len(fnlr))
for i in range(len(fnlr)):
    kfeature = i+1
    ix = np.where(nfeature==kfeature)[0]
    myscorelr[i] = np.mean(score[ix])

# load results for SVM Linear
Cgrid = 10.**np.linspace(-2, 2, 11)
ncgrid = len(Cgrid)
Cvalue, nfeature, score = np.loadtxt(basedir + 'score-vs-nfeature-svmlinear.txt', unpack=True, skiprows=1)
fn = np.linspace(1, max(nfeature)+1, max(nfeature))
myscoresvm = np.zeros(len(fn))
mycvalue = np.zeros(len(fn))
myscorec = np.zeros(ncgrid)
for i in range(len(fn)):
    kfeature = i+1
    # select the best C value
    for j in range(len(Cgrid)):
        myc = Cgrid[j]
        ix = np.where((nfeature==kfeature)&(np.abs(Cvalue-myc)<myc*0.01))[0]
        myscorec[j] = np.mean(score[ix])
    iy = np.where(myscorec == np.max(myscorec))[0]
    myscoresvm[i] = np.max(myscorec)
    mycvalue[i] = np.mean(Cgrid[iy])


ix=np.where(myscoresvm==np.max(myscoresvm))[0]
svmbestn = fn[ix[0]]
svmbestc = mycvalue[ix[0]]
print "SVM Linear, optimal number of features: %i" % (svmbestn)
print "SVM Linear, optimal C: %f" % (svmbestc)

# select the best-ranking features
# (feature, cvalue)
ftrank = np.loadtxt(basedir + 'feature-ranking-svmlinear.txt')
logc = np.log10(svmbestc/Cgrid)
ixc = np.where(abs(logc)==np.min(abs(logc)))[0]
ftrank2 = ftrank[:, ixc].flatten()
indices = np.argsort(ftrank2)
print ftrank2.shape
print indices

ixselect = indices[0:int(svmbestn)]
print ixselect
print np.asarray(featurenames)[ixselect]


figname = 'score-vs-nfeature-lr-svmlinear.png'
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel("Number of features selected", fontsize = 18)
ax.set_ylabel("Cross validation score", fontsize = 18)
ax.plot(fnlr, myscorelr, 'r-o')
ax.plot(fn, myscoresvm, 'g-o')
ax.legend(('Logistic Regression', 'SVM Linear'), numpoints = 1, prop={'size':14}, loc='upper right')
pylab.savefig(basedir + figname)
plt.close()




# Method 2: using SelectKbest
# It takes 3 hours for 11x11 grids, up to 40 features, 10 iterations

"""
t0 = time.time()

# 10x10 grid of C and gamma
Cgrid = 10.**np.linspace(-2, 2, 11)
gammagrid = 10.**np.linspace(-2, 2, 5)
ncgrid = len(Cgrid)
ngamma = len(gammagrid)

h = open(basedir + 'feature_selection_score-vs-nfeature.txt', 'w')
h.write("# nfeature, score_lr, score_svmlinear, score_svmrbf\n")

nftuse = 40
niter = 50
clf1 = LogisticRegression(class_weight='auto')
clf2 = svm.SVC(kernel='linear', class_weight='auto')
clf3 = svm.SVC(kernel='rbf', class_weight='auto')
score_lr = np.zeros(nftuse)
score_svmlinear = np.zeros(nftuse)
score_svmrbf = np.zeros(nftuse)
score1 = np.zeros(niter)
score2b = np.zeros(niter)
score3b = np.zeros(niter)
score2 = np.zeros(ncgrid)
score3 = np.zeros((ncgrid, ngamma))
for i in range(nftuse):
    kfeature = i + 1
    featureSelector = SelectKBest(score_func=feature_selection.f_classif, k=kfeature)
    featureSelector.fit(X_labeled,Y_labeled)
    ixselect = featureSelector.get_support(indices = True)
    
    # format data using selected features:
    X_labeled2 = X_labeled[:, ixselect]
    
    for ii in range(niter):   
        cv = cross_validation.KFold(X_labeled.shape[0], 5, shuffle=True)
        # logistic regression     
        score1[ii] = np.mean(cross_validation.cross_val_score(clf1, X_labeled2, Y_labeled, cv=cv))
        
        # support vector machine
        for jj in range(len(Cgrid)):
            myC = Cgrid[jj]
            clf2 = svm.SVC(C = myC, kernel='linear', class_weight='auto')
            score2[jj] = np.mean(cross_validation.cross_val_score(clf2, X_labeled2, Y_labeled, cv=cv))
            for kk in range(len(gammagrid)):
                mygamma = gammagrid[kk]
                clf3 = svm.SVC(C = myC, kernel='rbf', class_weight='auto', gamma = mygamma)
                score3[jj, kk] = np.mean(cross_validation.cross_val_score(clf3, X_labeled2, Y_labeled, cv=cv))
        
        score2b[ii] = np.max(score2)
        score3b[ii] = max(score3.flatten())
    
    score_lr[i] = np.mean(score1)
    score_svmlinear[i] = np.mean(score2b)
    score_svmrbf[i] = np.mean(score3b)
    h.write('%i  %f  %f  %f\n' % (kfeature, score_lr[i], score_svmlinear[i], score_svmrbf[i]))
h.close()

t1 = time.time()
print "Time elapsed in parameter/feature selection: %.2f" % (t1 - t0)

kfeature, score_lr, score_svmlinear, score_svmrbf = np.loadtxt(basedir + 'feature_selection_score-vs-nfeature.txt', unpack=True, skiprows=1)
figname = 'feature_selection_score-vs-nfeature.png'
fig = plt.figure()
ax = fig.add_subplot(111)
# kfeature = np.linspace(1, nftuse, nftuse)
ax.plot(kfeature, score_lr, 'r-o', markersize = 5)
ax.plot(kfeature, score_svmlinear, 'g-o', markersize = 5) 
# ax.plot(kfeature, score_svmrbf, 'b-o', markersize = 5) 
# ax.legend(('Logistic Regression', 'SVM Linear', 'SVM RBF'), numpoints = 1, prop={'size':14}, loc='lower left')
ax.legend(('Logistic Regression', 'SVM Linear'), numpoints = 1, prop={'size':14}, loc='lower left')
ax.set_xlabel('Number of Features Used', fontsize = 18)
ax.set_ylabel('Cross Validation Score', fontsize = 18)
pylab.savefig(basedir + figname)
plt.close()
"""


# Parameter fine tuning of Cvalue
# using SVM linear, 6 features
# Turned out C = 1. is the best    
# 40 grid of C
"""
Cgrid = 10**np.linspace(-2, 2, 41)
ngrid = len(Cgrid)

kfeature = 6
niter = 20
score2 = np.zeros((niter, ngrid))

featureSelector = SelectKBest(score_func=feature_selection.f_classif, k=kfeature)
featureSelector.fit(X_labeled,Y_labeled)
ixselect = featureSelector.get_support(indices = True)
# format data using selected features:
X_labeled2 = X_labeled[:, ixselect]

importances = featureSelector.scores_
indices = np.argsort(importances)[::-1]
# Print the top features
print("Feature ranking (selectkbest):")
for f in range(kfeature):
    print("%d. %s (%f)" % (f + 1, featurenames[indices[f]], importances[indices[f]]))


for ii in range(niter):    
    cv = cross_validation.KFold(X_labeled.shape[0], 10, shuffle=True)#, random_state=0)
    
    # support vector machine with different C
    for jj in range(len(Cgrid)):
        myC = Cgrid[jj]
        clf2 = svm.SVC(C = myC, kernel='linear', class_weight='auto')
        score2[ii, jj] = np.mean(cross_validation.cross_val_score(clf2, X_labeled2, Y_labeled, cv=cv))

score_svmlinear = np.mean(score2, axis=0)
figname = 'svm-linear-score-vs-C.png'
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(Cgrid, score_svmlinear, 'g-o', markersize = 5)
ax.set_xlabel('Penalty parameter C (SVM Linear)', fontsize = 18)
ax.set_ylabel('Cross Validation Score', fontsize = 18)
ax.set_ylim([score_svmlinear.min()-0.005, score_svmlinear.max()+0.005])
ax.set_xscale('log')
pylab.savefig(basedir + figname)
plt.close()
"""


# ========================================
# =====    Part 5. Classification    =====
# ========================================
# using SVM linear, adopting C = 1., k=6

clf = svm.SVC(C=1.0, kernel='linear', class_weight='auto', probability=True) 

# select features and format labeled data
featureSelector = SelectKBest(score_func=feature_selection.f_classif, k=6)
featureSelector.fit(X_labeled,Y_labeled)
ixselect = featureSelector.get_support(indices = True)
print ixselect
print np.asarray(featurenames)[ixselect]
X_labeled2 = X_labeled[:, ixselect]

# Format the full, qso, and star datasets
X_qso_new = X_qso[:, ixselect]
X_qso2_new = X_qso2[:, ixselect] # only include previously known quasars
X_star_new = X_star[:, ixselect]
X_new = X_data[:, ixselect]
X_unlabeled_new = X_unlabeled[:, ixselect]
nqso = X_qso_new.shape[0]
nqso2 = X_qso2_new.shape[0]
nstar = X_star_new.shape[0]
nall = X_new.shape[0]
nunlabeled = X_unlabeled_new.shape[0]

#data_train, data_test, target_train, target_test = cross_validation.train_test_split(X_labeled2, Y_labeled, test_size=0.2)

# Bootstrap aggregating (bagging)
# do 10 folds, and then average the probability scores
nfolds = 10
kf = cross_validation.KFold(len(Y_labeled), n_folds=nfolds, shuffle=True, random_state=0)

cm = np.zeros((2,2))
qsoprob_all = np.zeros(nall)
qsoprob_qso = np.zeros(nqso)
qsoprob_qso2 = np.zeros(nqso2)
qsoprob_star = np.zeros(nstar)
qsoprob_unlabeled = np.zeros(nunlabeled)

for train_index, test_index in kf:
    data_train, data_test = X_labeled2[train_index], X_labeled2[test_index]
    target_train, target_test = Y_labeled[train_index], Y_labeled[test_index]

    # fit the training set
    clf.fit(data_train, target_train)

    # confusion matrix
    predicted = clf.predict(data_test)
    expected = target_test
    cm = cm + metrics.confusion_matrix(expected, predicted)
    
    # prob of being quasars in the full, qso, and star samples
    qsopredict_prob = clf.predict_proba(X_new)
    qsoprob_all = qsoprob_all + qsopredict_prob[:, 1]
    
    qsopredict_prob = clf.predict_proba(X_qso_new)
    qsoprob_qso = qsoprob_qso + qsopredict_prob[:, 1]

    qsopredict_prob = clf.predict_proba(X_qso2_new)
    qsoprob_qso2 = qsoprob_qso2 + qsopredict_prob[:, 1]
    
    qsopredict_prob = clf.predict_proba(X_star_new)
    qsoprob_star = qsoprob_star + qsopredict_prob[:, 1]
    
    qsopredict_prob = clf.predict_proba(X_unlabeled_new)
    qsoprob_unlabeled = qsoprob_unlabeled + qsopredict_prob[:, 1]


# average the probability of being quasar
qsoprob_all_norm = 1.*qsoprob_all/nfolds
qsoprob_qso_norm = 1.*qsoprob_qso/nfolds
qsoprob_qso2_norm = 1.*qsoprob_qso2/nfolds
qsoprob_star_norm = 1.*qsoprob_star/nfolds
qsoprob_unlabeled_norm = 1.*qsoprob_unlabeled/nfolds


# plot the confusion matrix
figname = 'confusion-matrix.png'
labels = ['stars', 'quasars']
cm2 = np.zeros((2,2))
cm2[0, :] = 1.*cm[0, :]/cm[0,:].sum()
cm2[1, :] = 1.*cm[1, :]/cm[1,:].sum()
print(cm)
print(cm2)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm2, cmap='gray', vmin=0, vmax=1)
ax.set_title('Confusion matrix of the SVM linear classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
ax.set_xlabel('Predicted', fontsize=18)
ax.set_ylabel('True', fontsize=18)
for i in range(2):
    for j in range(2):
        ax.text(j, i, '%.3f' % (cm[i,j]/cm[i,:].sum()), horizontalalignment='center', verticalalignment='center', fontsize = 18, color='blue')
pylab.savefig(basedir + figname)
plt.close()


# Number of quasars selected from the full sample: 
print "Number of quasars selected in the full sample: %i (out of %i)" % (len(np.where(qsoprob_all_norm>=0.5)[0]), X_new.shape[0])

# Plot the Nselect vs prob curves
probrate = np.linspace(0., 1., 101)
nqso_select = np.zeros(len(probrate))
nqso2_select = np.zeros(len(probrate))
nstar_select = np.zeros(len(probrate))
nall_select = np.zeros(len(probrate))
nunlabeled_select = np.zeros(len(probrate))

for i in range(len(probrate)):
    prob0 = probrate[i]
    nall_select[i] = len(np.where(qsoprob_all_norm>=prob0)[0])
    nqso_select[i] = len(np.where(qsoprob_qso_norm>=prob0)[0])
    nqso2_select[i] = len(np.where(qsoprob_qso2_norm>=prob0)[0])
    nstar_select[i] = len(np.where(qsoprob_star_norm>=prob0)[0])
    nunlabeled_select[i] = len(np.where(qsoprob_unlabeled_norm>=prob0)[0])
    

figname = 'QSO-Nselect-ratio-vs-prob-rate.png'
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(probrate, nqso_select/nqso, 'g-o', markersize = 5)
#ax.plot(probrate, nqso2_select/nqso2, 'y-o', markersize = 5)
ax.plot(probrate, nstar_select/nstar, 'r-o', markersize = 5)
ax.plot(probrate, nunlabeled_select/nunlabeled, 'k-o', markersize = 5)
#ax.legend(('%i quasars' % nqso, '%i pre-known quasars' % nqso2, '%i stars' % nstar, '%i unlabeled' % nunlabeled), numpoints = 1, loc = 'lower left', prop={'size':14})
ax.legend(('quasars', 'stars', 'unlabeled'), numpoints = 1, loc = 'lower left', prop={'size':14})
#ax.plot([0.5, 0.5], [0.003, 1], 'y--')
ax.set_xlabel('$P_{quasar}$ (bagging/averaging 10 models)', fontsize = 14)
ax.set_ylabel('Ratio of Selected Objects with $P>=P_{quasar}$', fontsize = 14)
ax.set_yscale('log')
pylab.savefig(basedir + figname)
plt.close()

print len(raqso)
print len(raqso2)


# Output quasar candidates
h = open(outfile, 'w')
h.write("# ra, dec, flag (1: previously known; 2: my newly identified; -1: ruled out quasar; 0: no spec)\n")
ix = np.where(qsoprob_all_norm>=0.5)[0]
candprob = qsoprob_all_norm[ix]
myra0 = data['ra'][ix]
mydec0 = data['dec'][ix]
ncand = len(ix)
qflag = np.zeros(ncand)
for i in range(ncand):
    myra = myra0[i]
    mydec = mydec0[i]
    
    # match with previously known quasars    
    dist = (((raqso2-myra)*np.cos(mydec*np.pi/180.))**2 + (decqso2-mydec)**2 )**0.5*3600.0
    ix = np.where(dist<1)[0]
    nmatch1 = len(ix)

    # match with my newly identified quasars    
    dist = (((raqso-myra)*np.cos(mydec*np.pi/180.))**2 + (decqso-mydec)**2 )**0.5*3600.0
    ix = np.where(dist<1)[0]
    nmatch2 = len(ix)
    
    # match with failed quasars    
    dist = (((rafqso-myra)*np.cos(mydec*np.pi/180.))**2 + (decfqso-mydec)**2 )**0.5*3600.0
    ix = np.where(dist<1)[0]
    nmatch3 = len(ix)
    
    # match with stars   
    dist = (((rastar-myra)*np.cos(mydec*np.pi/180.))**2 + (decstar - mydec)**2 )**0.5*3600.0
    ix = np.where(dist<1)[0]
    nmatch4 = len(ix)

    if nmatch1 == 1:
        qflag[i] = 1
    elif nmatch2 == 1:
        qflag[i] = 2
    elif (nmatch3 == 1) | (nmatch4==1):
        qflag[i] = -1

    h.write("%f  %f  %i\n" % (myra, mydec, qflag[i]))

h.close()

ix = np.where(qflag==1)[0]; ncan1 = len(ix)
print "Number of previously known quasars recovered: %i; mean(prob)=%.2f" % (len(ix), np.mean(candprob[ix]))
ix = np.where(qflag==2)[0]; ncan2 = len(ix)
print "Number of my quasars recovered: %i; mean(prob)=%.2f" % (len(ix), np.mean(candprob[ix]))
ix = np.where(qflag==-1)[0]; ncan3 = len(ix)
print "Number of candidates turned out to be non-quasars: %i; mean(prob)=%.2f" % (len(ix), np.mean(candprob[ix]))
ix = np.where(qflag==0)[0]; ncan4 = len(ix)
print "Number of un-observed candidates: %i; mean(prob)=%.2f" % (len(ix), np.mean(candprob[ix]))


# =====================================================
# =====    Part 6. Evaluation and region files    =====
# =====================================================

# plot pie chart
# The slices will be ordered and plotted counter-clockwise.
figname = 'QSO-piechart_selected.png'
pylab.figure(1, figsize=(8,6))
ax = pylab.axes([0.1, 0.1, 0.7, 0.8])
labels = ['Known quasars (%i)' % ncan1, 'New quasars (%i)' % ncan2, 'False (%i)' % ncan3, 'New candidates (%i)' % ncan4]
sizes = [ncan1, ncan2, ncan3, ncan4]
colors = ['blue', 'green', 'red', 'yellow']
explode = (0.05, 0.05, 0.05, 0.05) # only "explode" the 2nd slice
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%')
# Set aspect ratio to be equal so that pie is drawn as a circle.
plt.axis('equal')
#ax.set_title('Quasars selected by the classifier', horizontalalignment='center')
pylab.savefig(basedir + figname)
plt.close()

figname = 'QSO-piechart_all.png'
pylab.figure(1, figsize=(8,6))
ax = pylab.axes([0.15, 0.1, 0.7, 0.8])
labels = ['Known quasars (%i)' % len(raqso2), 'New quasars (%i)' % (len(raqso)-len(raqso2)), 'False (%i)' % ncan3, 'New candidates (%i)' % ncan4]
sizes = [len(raqso2), (len(raqso)-len(raqso2)), ncan3, ncan4]
colors = ['blue', 'green', 'red', 'yellow']
explode = (0.05, 0.05, 0.05, 0.05) # only "explode" the 2nd slice
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%')
# Set aspect ratio to be equal so that pie is drawn as a circle.
plt.axis('equal')
ax.set_title('Quasars behind the Andromeda Galaxy', horizontalalignment='center')
pylab.savefig(basedir + figname)
plt.close()

# Quasars behind M31 within 3 sq deg
# (dec_2 - dec_1)*(ra_2-ra_1)*np.cos(41.*np.pi/180.) = 3.0905357310122437
ra_1 = 9.6
ra_2 = 11.7
dec_1 = 40.35
dec_2 = 42.3

myra = raqso2
mydec = decqso2
ix = np.where((myra>ra_1) & (myra<ra_2) & (mydec>dec_1) & (mydec<dec_2))[0]
nc1 = len(ix)
myra = raqso
mydec = decqso
ix = np.where((myra>ra_1) & (myra<ra_2) & (mydec>dec_1) & (mydec<dec_2))[0]
nc2 = len(ix) - nc1

racand, deccand, flagcand = np.loadtxt(outfile, unpack = True, skiprows = 1)
myra = racand
mydec = deccand
ix = np.where((myra>ra_1) & (myra<ra_2) & (mydec>dec_1) & (mydec<dec_2) & (flagcand==-1))[0]
nc3 = len(ix)
ix = np.where((myra>ra_1) & (myra<ra_2) & (mydec>dec_1) & (mydec<dec_2) & (flagcand==0))[0]
nc4 = len(ix)


figname = 'QSO-piechart_all_3sd.png'
pylab.figure(1, figsize=(8,6))
ax = pylab.axes([0.1, 0.1, 0.7, 0.8])
labels = ['Known quasars (%i)' % nc1, 'New quasars (%i)' % nc2, 'False (%i)' % nc3, 'New candidates (%i)' % nc4]
sizes = [nc1, nc2, nc3, nc4]
colors = ['blue', 'green', 'red', 'yellow']
explode = (0.05, 0.05, 0.05, 0.05) # only "explode" the 2nd slice
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%')
# Set aspect ratio to be equal so that pie is drawn as a circle.
plt.axis('equal')
ax.set_title('In the central region (3 sq deg)', horizontalalignment='center')
pylab.savefig(basedir + figname)
plt.close()

# region file
h = open(regionfile, 'w')
h.write("#region file for M31 QSOs, last updated April 2014\n")
h.write("global width=2\n")
for i in range(len(racand)):
    myra = racand[i]
    mydec = deccand[i]
    myflag = flagcand[i]
    if myflag == 1:
        h.write("circle  %fd  %fd  2\' # color = blue\n" % (myra, mydec))
    elif myflag == 2:
        h.write("circle  %fd  %fd  2\' # color = green\n" % (myra, mydec))
    elif myflag == -1:
        h.write("circle  %fd  %fd  2\' # color = red\n" % (myra, mydec))
    elif myflag == 0:
        h.write("circle  %fd  %fd  2\' # color = yellow\n" % (myra, mydec))

for i in range(len(raqso)):
    myra = raqso[i]
    mydec = decqso[i]
    dist = (((racand-myra)*np.cos(mydec*np.pi/180.))**2 + (deccand-mydec)**2 )**0.5*3600.0
    ix1 = np.where(dist<1.)[0]
    dist2 = (((raqso2-myra)*np.cos(mydec*np.pi/180.))**2 + (decqso2-mydec)**2 )**0.5*3600.0
    ix2 = np.where(dist2<1.)[0]        
    if (len(ix1)==0) & (len(ix2)==0):
        h.write("circle  %fd  %fd  2\' # color = green\n" % (myra, mydec))

# known qsos not recovered
for i in range(len(raqso2)):
    myra = raqso2[i]
    mydec = decqso2[i]
    dist = (((racand-myra)*np.cos(mydec*np.pi/180.))**2 + (deccand-mydec)**2 )**0.5*3600.0
    ix = np.where(dist<1.)[0]
    if len(ix)==0:
        h.write("box  %fd  %fd  3\' 3\' # color = blue\n" % (myra, mydec))


h.close()


h = open(basedir + 'known-quasars.reg', 'w')
h.write("# region file for previously known M31 QSOs, last updated April 2014\n")
h.write("global width=2\n")
for i in range(len(racand)):
    myra = racand[i]
    mydec = deccand[i]
    myflag = flagcand[i]
    if myflag == 1:
        h.write("circle  %fd  %fd  2\' # color = blue\n" % (myra, mydec))

# known qsos not recovered
for i in range(len(raqso2)):
    myra = raqso2[i]
    mydec = decqso2[i]
    dist = (((racand-myra)*np.cos(mydec*np.pi/180.))**2 + (deccand-mydec)**2 )**0.5*3600.0
    ix = np.where(dist<1.)[0]
    if len(ix)==0:
        h.write("box  %fd  %fd  3\' 3\' # color = blue\n" % (myra, mydec))


h.close()


toc = time.time()
dt = toc - tic
print 'Total time elapsed: %5.2f s' % dt
