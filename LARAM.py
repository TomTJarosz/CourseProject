import numpy as np
from numpy.random import normal

class LARAM:
    def __init__(self, documents, reviews):
        '''
        k = # topics
        w = # words
        intrinsic model parameters:
            eps: k x w, eps[i,j] = p(word_j | topic_i)
            gamma: k, postulates the prior distribution of aspects in the whole corpus.
            beta: k x w, beta[i,j] = sentiment of word j for topic i
            mu: k, vector representing mean of normal distribution from which aspect weight is drawn
            sigma: k, vector representing st. dev. of normal distribution from which aspect weight is drawn
            deltaSquared: 1, vector representing st. dev. of normal distribution from final rating iis drawn

        :param documents:[string]
        :param reviews:[float]
        '''
        self.numToWordMap, self.documents = self.wordToNum(documents)
        self.reviews = reviews

    def wordToNum(self, documents):
        '''
        Parse documents, and create new "documents" which are easier to work with. Transform each document (string)
        to a list of numbers, where each number is mapped to a  word. Return these documents in this new representation,
        along with a map from number to the word it represents
        '''
        newDocs = []
        wordToNumMap = {}
        numToWordMap = {}
        c = 0
        for doc in documents:
            newDoc = []
            doc = doc.lower().strip()
            doc = doc.split()
            for w in doc:
                if w not in wordToNumMap:
                    wordToNumMap[w] = c
                    numToWordMap[c] = w
                    c += 1
                newDoc.append(wordToNumMap[w])
            newDocs.append(newDoc)
        return numToWordMap, newDocs

    def initialize_params(self, k):
        '''
        All the parameters are first randomly initialized
        '''
        self.beta = np.random.random((k, len(self.numToWordMap)))
        self.gamma = np.random.random((k))
        self.eta = np.random.random((k, len(self.numToWordMap)))
        self.mu = np.random.random((k))
        self.sigma = np.random.random((k))
        self.deltaSquared = np.random.random()

    def fit(self, k=5):
        '''
        All the parameters are first randomly initialized to obtain
        Θ(0) (subscript indicates the iteration step) and then the
        following EM algorithm is applied to iteratively update and
        improve the parameters by alternatively executing the Estep
        and M-step in each iteration until the log-likelihood
        defined in Eq (12) converges.
        '''
        self.initialize_params(k)
        while not self.hasConverged():
            self.eStep()
            self.mStep()

    def hasConverged(self):
        '''
        Compute the current likelyhood of our document/review pairs. If the computed likelyhood differs from our
        previous likelyhood by a smaller quantity than our threshold, we consider our EM algorithm to have converged.
        :return: true if we have converged, else false
        '''
        lastLogLikelyhood = self.lastLogLikelyhood
        curLikelyhood = self.computeLikelyhood()
        self.lastLogLikelyhood = curLikelyhood
        return np.abs(lastLogLikelyhood - curLikelyhood) < .0001

    def computeLikelyhood(self):
        '''
        L(D) = ∑ Ld(ϕ, η, λ, σ2)
        '''
        return sum([self.documentLikelyhood(d) for d in self.documents])


    def documentLikelyhood(self, d):
        '''
        Compute the likelyhood of a document 'd' given our current model parameters
        :param d: a document
        :return: likelyhood of the  document
        '''
        theta = ...
        z = ...
        term1 = np.log(self.probThetaGivenGamma(theta) * np.prod([self.probZGivenTheta(z[w,:], theta) * self.probWordGivenZAndEps(w,z[w,:]) for w in d]))
        term2 = np.log(...)
        return term1 + term2 + ...

    def probThetaGivenGamma(self, theta):
        '''
        Used to compute document Likelyhood
        '''
        ...

    def probZGivenTheta(self, z, theta):
        '''
        Used to compute document Likelyhood
        '''
        ...

    def probWordGivenZAndEps(self, w, z):
        '''
        Used to compute document Likelyhood
        '''
        ...

    def mStep(self):
        '''
        M-Step: Given the sufficient statistics collected from each
        review in E-Step, find the updated model parameters Θ(t+1)
        by using Eq (13) to Eq (19).
        '''
        self.updateEps()
        self.updateBeta()

    def updateBeta(self):
        '''
        e apply the gradient-based optimization technique to find
        the optimal solution of β with the following gradients:
        ∂L(βi)∂βij=∑Dd[(λ T d ¯sd−rd)λdi+σ 2 di¯sdi+(λ 2 di+σ 2 reviewi)βijw j dn(1−ϕdni)]ϕdniw j dn
        '''
        ...

    def updateEps(self):
        '''
        ϵij ∝ ∑D d ∑Nd n ϕdniw j dn
        '''
        newEps = np.zeros(self.eps.shape)
        for i,d in enumerate(self.documents):
            for j,w in enumerate(d):
                newEps[:,w]  = self.eps[:,w] * self.Zs[i][j,:]
        self.eps = newEps

    def eStep(self):
        '''
        E-Step: For each review d in the corpus, infer aspect weight
        α and topic assignments {zn} based on the current parameter
        Θ(t) by using Eq (8) to Eq (11) and compute aspect
        rating s by Eq (2).
        '''
        Alphas = []
        Zs = []
        Ss = []
        for d in self.documents:
            alpha, z = self.inferAlphaAndZ(d)
            s = self.computeAspectRatings(d, alpha, z)
            Alphas.appennd(alpha)
            Zs.appennd(z)
            Ss.appennd(s)
        self.Alphas = Alphas
        self.Zs = Zs
        self.Ss = Ss

    def computeAspectRatings(self, z):
        '''
        si = ∑|d| n=1 βij∆[wn = vj , zn = i]
        '''
        ret = 0
        for i,w in enumerate(d):
            ret += self.beta[np.argmax(z[i,:]),w]
        return ret

    def inferAlphaAndZ(self, d):
        '''
        For each review d in the corpus, infer aspect weight
        α and topic assignments {zn}
        |d|
        n=1 based on the current parameter Θ(t) by using Eq (8) to Eq (11) and compute aspect
        rating s by Eq (2).

        ηˆi = γi + ∑ |d| n=1 ϕni

        σ2i = δ 2 V ar[si] + ¯s 2 i + δ 2Σ −1 ii
        '''
        nu = np.zeros(self.gamma.shape)
        for i,g in enumerate(self.gamma):
            n = g
            for w in d:
                n += self.phi(w)[i]
            nu[i] = n
        self.nu = nu

        smallSigmaSquared = np.zeros(self.gamma.shape)
        for i,s in enumerate(self.Ss):
            sss = self.deltaSquared / (self.varS(i, d) + self.sBar(i, d) ** 2 + ((self.deltaSquared **2) * (1/self.sigma[i,i])))
            smallSigmaSquared[i] = sss
        self.smallSigmaSquared = smallSigmaSquared

        return self.generateAlpha(), self.generateZ(d)

    def generateZ(self, d):
        '''
        the aspect assignment z for each word in d is specified
        by a k-dimensional multinomial distribution Mul(ϕ)
        :param d:
        :return:
        '''
        z = np.zeros((len(d), self.phi.shape[1]))
        for i, w in enumerate(d):
            z[i,:] = self.phi[w,:]
        return z

    def generateAlpha(self):
        '''
        we employ a multivariate Normal distribution as the prior for aspect weight α, i.e., α ∼ N(µ, Σ)
        :return: a
        '''
        ret = np.zeros(self.mu.shape)
        for i in range(ret.shape[0]):
            ret[i] = normal(self.mu[i], np.sqrt(self.sigma[i]))
        return ret/p.linalg.norm(ret)

    def varS(self, i, d):
        '''
        V ar[si] = ∑|d|n=1(βijw j n) 2ϕni(1−ϕni)
        '''
        return sum([self.beta[i,w]**2 * self.phi[i,w] * (1  - self.phi[i,w]) for w in d])

    def sBar(self, i, d):
        '''
        s¯i = ∑|d| n=1 βijw j nϕni
        '''
        return sum([self.beta[i,w] * self.phi[i,w]  for w in d])

