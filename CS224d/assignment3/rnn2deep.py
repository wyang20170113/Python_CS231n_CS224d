import numpy as np
import collections
import pdb


# This is a 2-Layer Deep Recursive Neural Netowrk with two ReLU Layers and a softmax layer
# You must update the forward and backward propogation functions of this file.

# You can run this file via 'python rnn2deep.py' to perform a gradient check

# tip: insert pdb.set_trace() in places where you are unsure whats going on


class RNN2:

    def __init__(self,wvecDim, middleDim, outputDim,numWords,mbSize=30,rho=1e-4):
        self.wvecDim = wvecDim
        self.outputDim = outputDim
        self.middleDim = middleDim
        self.numWords = numWords
        self.mbSize = mbSize
        self.defaultVec = lambda : np.zeros((wvecDim,))
        self.rho = rho

    def initParams(self):
        np.random.seed(12341)

        # Word vectors
        self.L = 0.01*np.random.randn(self.wvecDim,self.numWords)

        # Hidden activation weights for layer 1
        self.W1 = 0.01*np.random.randn(self.wvecDim,2*self.wvecDim)
        self.b1 = np.zeros((self.wvecDim))

        # Hidden activation weights for layer 2
        self.W2 = 0.01*np.random.randn(self.middleDim,self.wvecDim)
        self.b2 = np.zeros((self.middleDim))

        # Softmax weights
        self.Ws = 0.01*np.random.randn(self.outputDim,self.middleDim) # note this is " U " in the notes and the handout.. there is a reason for the change in notation
        self.bs = np.zeros((self.outputDim))

        self.stack = [self.L, self.W1, self.b1, self.W2, self.b2, self.Ws, self.bs]

        # Gradients
        self.dW1 = np.empty(self.W1.shape)
        self.db1 = np.empty((self.wvecDim))
        
        self.dW2 = np.empty(self.W2.shape)
        self.db2 = np.empty((self.middleDim))

        self.dWs = np.empty(self.Ws.shape)
        self.dbs = np.empty((self.outputDim))

    def costAndGrad(self,mbdata,test=False): 
        """
        Each datum in the minibatch is a tree.
        Forward prop each tree.
        Backprop each tree.
        Returns
           cost
           Gradient w.r.t. W1, W2, Ws, b1, b2, bs
           Gradient w.r.t. L in sparse form.

        or if in test mode
        Returns 
           cost, correctArray, guessArray, total
        """
        cost = 0.0
        correct = []
        guess = []
        total = 0.0

        self.L, self.W1, self.b1, self.W2, self.b2, self.Ws, self.bs = self.stack
        # Zero gradients
        self.dW1[:] = 0
        self.db1[:] = 0
        
        self.dW2[:] = 0
        self.db2[:] = 0

        self.dWs[:] = 0
        self.dbs[:] = 0
        self.dL = collections.defaultdict(self.defaultVec)

        # Forward prop each tree in minibatch
        for tree in mbdata: 
            c,tot = self.forwardProp(tree.root,correct,guess)
            cost += c
            total += tot
            
        if test:
            return (1./len(mbdata))*cost,correct, guess, total

        # Back prop each tree in minibatch
        for tree in mbdata:
            self.backProp(tree.root)

        # scale cost and grad by mb size
        scale = (1./self.mbSize)
        for v in self.dL.itervalues():
            v *=scale
        
        # Add L2 Regularization 
        cost += (self.rho/2)*np.sum(self.W1**2)
        cost += (self.rho/2)*np.sum(self.W2**2)
        cost += (self.rho/2)*np.sum(self.Ws**2)

        return scale*cost,[self.dL,scale*(self.dW1 + self.rho*self.W1),scale*self.db1,
                                   scale*(self.dW2 + self.rho*self.W2),scale*self.db2,
                                   scale*(self.dWs+self.rho*self.Ws),scale*self.dbs]

    def forwardProp(self,node, correct=[], guess=[]):
        cost  =  total = 0.0
        # this is exactly the same setup as forwardProp in rnn.py
        if (node.fprop == False):
            if (node.isLeaf == True):
                node.hActs1 = self.L[:,node.word]
                #t = self.Ws.dot(node.hActs1) + self.bs
                node.hActs2 = np.maximum(self.W2.dot(node.hActs1) + self.b2,0)
                node.probs = self.softmax(self.Ws.dot(node.hActs2) + self.bs)
                
                cost += -np.log(node.probs[node.label])
            else:
                if (node.left != None and node.left.fprop == False):
                    cost_left, total_left = self.forwardProp(node.left,correct,guess)
                    cost += cost_left
                    total += total_left
                if (node.right != None and node.right.fprop == False):
                    cost_right, total_right = self.forwardProp(node.right,correct,guess)
                    cost += cost_right
                    total += total_right
                if (node.left != None and node.left.fprop == True and node.right != None and node.right.fprop == True):
                    h_pre = np.hstack((node.left.hActs1,node.right.hActs1))
                    node.hActs1 = np.maximum(self.W1.dot(h_pre) + self.b1,0)
                    node.hActs2 = np.maximum(self.W2.dot(node.hActs1) + self.b2,0)
                    node.probs = self.softmax(self.Ws.dot(node.hActs2) + self.bs)
                    cost += -np.log(node.probs[node.label])
                if (node.left != None and node.left.fprop == True and node.right == None):
                    h_pre = np.hstack((node.left.hActs1,np.zeros(self.wvecDim,)))
                    node.hActs1 = np.maximum(self.W1.dot(h_pre) + self.b1,0)
                    node.hActs2 = np.maximum(self.W2.dot(node.hActs1) + self.b2,0)
                    node.probs = self.softmax(self.Ws.dot(node.hActs2) + self.bs)
                    cost += -np.log(node.probs[node.label])
                if (node.left == None and node.right != None and node.right.fprop == True):
                    h_pre = np.hstack((np.zeros(self.wvecDim,),node.right.hActs1))
                    node.hActs1 = np.maximum(self.W1.dot(h_pre) + self.b1,0)
                    node.hActs2 = np.maximum(self.W2.dot(node.hActs1) + self.b2,0)
                    node.probs = self.softmax(self.Ws.dot(node.hActs2) + self.bs)
                    cost += -np.log(node.probs[node.label])
            ## update the node.fprop
            node.fprop = True
        correct.append(node.label)
        guess.append(np.argmax(node.probs))

        return cost, total + 1

    def backProp(self,node,error=None):

        # Clear nodes
        node.fprop = False

        # this is exactly the same setup as backProp in rnn.py
        
        
        if (node.isLeaf == False):
             if(node.left != None or node.right != None):
                mask = np.zeros(self.outputDim,)
                mask[node.label] = 1.0
                delta = node.probs - mask
                self.dbs += delta
                self.dWs += np.outer(delta,node.hActs2)
                ##
                pre_dh2 = self.Ws.T.dot(delta)
                dh2 = np.sign(node.hActs2) * pre_dh2
                
                ##
                self.dW2 += np.outer(dh2,node.hActs1)
                self.db2 += dh2
                pre_dh = self.W2.T.dot(dh2)
                
                ##
                
                if(error == None):
                    error_t = 0.0
                    pre_dh += error_t
                else:
                    pre_dh += error
                ## update the self.dW
                
            
                ## propagate the error downward
                error_down = np.sign(node.hActs1) * (pre_dh)
                self.db1 += np.sign(node.hActs1) * (pre_dh)
                if(node.left != None):
                    self.dW1[:,0:self.wvecDim] += np.outer(error_down,node.left.hActs1)
                    self.backProp(node.left,self.W1[:,0:self.wvecDim].T.dot(error_down))
            
                if(node.right != None):
                    self.dW1[:,self.wvecDim:] += np.outer(error_down,node.right.hActs1)
                    self.backProp(node.right,self.W1[:,self.wvecDim:].T.dot(error_down))
        else:
            mask = np.zeros(self.outputDim,)
            mask[node.label] = 1.0
            delta = node.probs - mask
            self.dWs += np.outer(delta,node.hActs2)
            self.dbs += delta
            #pre_dh = self.Ws.T.dot(delta)
            pre_dh2 = self.Ws.T.dot(delta)
            dh2 = np.sign(node.hActs2) * pre_dh2
                
            ##
            self.dW2 += np.outer(dh2,node.hActs1)
            self.db2 += dh2
            pre_dh = self.W2.T.dot(dh2)
            
            if(error == None):
                error_t = 0.0
                pre_dh += error_t
            else:
                pre_dh += error
            #self.db += np.sign(node.hActs1) * (pre_dh)
            self.dL[node.word] += (pre_dh) 
            pass
        ## end of backprop.
        
    def softmax(self,x):
        x = x - np.max(x)
        return np.exp(x) / sum(np.exp(x))
        
    def updateParams(self,scale,update,log=False):
        """
        Updates parameters as
        p := p - scale * update.
        If log is true, prints root mean square of parameter
        and update.
        """
        if log:
            for P,dP in zip(self.stack[1:],update[1:]):
                pRMS = np.sqrt(np.mean(P**2))
                dpRMS = np.sqrt(np.mean((scale*dP)**2))
                print "weight rms=%f -- update rms=%f"%(pRMS,dpRMS)

        self.stack[1:] = [P+scale*dP for P,dP in zip(self.stack[1:],update[1:])]

        # handle dictionary update sparsely
        dL = update[0]
        for j in dL.iterkeys():
            self.L[:,j] += scale*dL[j]

    def toFile(self,fid):
        import cPickle as pickle
        pickle.dump(self.stack,fid)

    def fromFile(self,fid):
        import cPickle as pickle
        self.stack = pickle.load(fid)

    def check_grad(self,data,epsilon=1e-6):

        cost, grad = self.costAndGrad(data)

        err1 = 0.0
        count = 0.0
        print "Checking dWs, dW1 and dW2..."
        for W,dW in zip(self.stack[1:],grad[1:]):
            W = W[...,None] # add dimension since bias is flat
            dW = dW[...,None] 
            for i in xrange(W.shape[0]):
                for j in xrange(W.shape[1]):
                    W[i,j] += epsilon
                    costP,_ = self.costAndGrad(data)
                    W[i,j] -= epsilon
                    numGrad = (costP - cost)/epsilon
                    err = np.abs(dW[i,j] - numGrad)
                    err1+=err
                    count+=1
        if 0.001 > err1/count:
            print "Grad Check Passed for dW"
        else:
            print "Grad Check Failed for dW: Sum of Error = %.9f" % (err1/count)
        # check dL separately since dict
        dL = grad[0]
        L = self.stack[0]
        err2 = 0.0
        count = 0.0
        print "Checking dL..."
        for j in dL.iterkeys():
            for i in xrange(L.shape[0]):
                L[i,j] += epsilon
                costP,_ = self.costAndGrad(data)
                L[i,j] -= epsilon
                numGrad = (costP - cost)/epsilon
                err = np.abs(dL[j][i] - numGrad)
                err2+=err
                count+=1

        if 0.001 > err2/count:
            print "Grad Check Passed for dL"
        else:
            print "Grad Check Failed for dL: Sum of Error = %.9f" % (err2/count)

if __name__ == '__main__':

    import tree as treeM
    train = treeM.loadTrees()
    numW = len(treeM.loadWordMap())

    wvecDim = 10
    middleDim = 10
    outputDim = 5

    rnn = RNN2(wvecDim,middleDim,outputDim,numW,mbSize=4)
    rnn.initParams()

    mbData = train[:4]
    
    print "Numerical gradient check..."
    rnn.check_grad(mbData)






