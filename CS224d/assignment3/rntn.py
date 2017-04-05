import numpy as np
import collections
np.seterr(over='raise',under='raise')

class RNTN:

    def __init__(self,wvecDim,outputDim,numWords,mbSize=30,rho=1e-6):
        self.wvecDim = wvecDim
        self.outputDim = outputDim
        self.numWords = numWords
        self.mbSize = mbSize
        self.defaultVec = lambda : np.zeros((wvecDim,))
        self.rho = rho

    def initParams(self):
        np.random.seed(12341)
        
        # Word vectors
        self.L = 0.01*np.random.randn(self.wvecDim,self.numWords)

        # Hidden activation weights
        self.V = 0.01*np.random.randn(self.wvecDim,2*self.wvecDim,2*self.wvecDim)
        self.W = 0.01*np.random.randn(self.wvecDim,self.wvecDim*2)
        self.b = np.zeros((self.wvecDim))

        # Softmax weights
        self.Ws = 0.01*np.random.randn(self.outputDim,self.wvecDim)
        self.bs = np.zeros((self.outputDim))

        self.stack = [self.L, self.V, self.W, self.b, self.Ws, self.bs]

        # Gradients
        self.dV = np.empty((self.wvecDim,2*self.wvecDim,2*self.wvecDim))
        self.dW = np.empty(self.W.shape)
        self.db = np.empty((self.wvecDim))
        self.dWs = np.empty(self.Ws.shape)
        self.dbs = np.empty((self.outputDim))

    def costAndGrad(self,mbdata,test=False): 
        #cost = 0.0
        #correct = []
        #guess = []
        #total = 0.0
        #self.L,self.V,self.W,self.b,self.Ws,self.bs = self.stack

        # Zero gradients
        #self.dV[:] = 0
        #self.dW[:] = 0
        #self.db[:] = 0
        #self.dWs[:] = 0
        #self.dbs[:] = 0
        #self.dL = collections.defaultdict(self.defaultVec)

        # Forward prop each tree in minibatch
        #for tree in mbdata: 
        #    c,tot = self.forwardProp(tree.root)
        #    cost += c
        #    total += tot
        #if test:
        #    return (1./len(mbdata))*cost,correct,guess,total

        # Back prop each tree in minibatch
        #for tree in mbdata:
        #    self.backProp(tree.root)

        # scale cost and grad by mb size
        #scale = (1./self.mbSize)
        #for v in self.dL.itervalues():
        #    v *=scale
        
        # Add L2 Regularization 
        #cost += (self.rho/2)*np.sum(self.V**2)
        #cost += (self.rho/2)*np.sum(self.W**2)
        #cost += (self.rho/2)*np.sum(self.Ws**2)

        #return scale*cost,[self.dL,scale*(self.dV + self.rho*self.V),scale*(self.dW + self.rho*self.W),scale*self.db,
        #                   scale*(self.dWs+self.rho*self.Ws),scale*self.dbs]
        #return cost, []
        cost = 0.0
        correct = []
        guess = []
        total = 0.0

        self.L,self.V,self.W,self.b,self.Ws,self.bs = self.stack
        # Zero gradients
        self.dV[:] = 0
        self.dW[:] = 0
        self.db[:] = 0
        self.dWs[:] = 0
        self.dbs[:] = 0
        self.dL = collections.defaultdict(self.defaultVec)

        # Forward prop each tree in minibatch
        for tree in mbdata:
            c,tot = self.forwardProp(tree.root,correct,guess)
            cost += c
            #correct += corr
            total += tot
        if test:
            return (1./len(mbdata))*cost,correct,guess,total

        # Back prop each tree in minibatch
        for tree in mbdata:
            self.backProp(tree.root)

        # scale cost and grad by mb size
        scale = (1./self.mbSize)
        for v in self.dL.itervalues():
            v *=scale
        
        # Add L2 Regularization 
        cost += (self.rho/2)*np.sum(self.V**2)
        cost += (self.rho/2)*np.sum(self.W**2)
        cost += (self.rho/2)*np.sum(self.Ws**2)

        return scale*cost,[self.dL,scale*(self.dV+self.rho*self.V),
                           scale*(self.dW + self.rho*self.W),scale*self.db,
                           scale*(self.dWs+self.rho*self.Ws),scale*self.dbs]

    def forwardProp(self,node,correct,guess):
        cost = total = 0.0
        ## recursive to update the cost
        if (node.fprop == False):
            if (node.isLeaf == True):
                #node.hActs1 = np.maximum(self.L[:,node.word] + self.b,0)
                node.hActs1 = self.L[:,node.word]
                t = self.Ws.dot(node.hActs1) + self.bs
                node.probs = self.softmax(t)
                
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
                    t = np.zeros((self.wvecDim))
                    for i in xrange(self.wvecDim):
                        t[i] = h_pre.T.dot(self.V[i,:,:].dot(h_pre))
                    #node.hActs1 = np.maximum(self.V.dot(h_pre).dot(h_pre.T) + self.W.dot(h_pre) + self.b,0)
                    node.hActs1 = np.tanh(t + self.W.dot(h_pre) + self.b)
                    node.probs = self.softmax(self.Ws.dot(node.hActs1) + self.bs)
                    cost += -np.log(node.probs[node.label])
                #if (node.left != None and node.left.fprop == True and node.right == None):
                #    h_pre = np.hstack((node.left.hActs1,np.zeros(self.wvecDim,)))
                #    node.hActs1 = np.maximum(self.V.dot(h_pre).dot(h_pre.T) + self.W.dot(h_pre) + self.b,0)
                #    node.probs = self.softmax(self.Ws.dot(node.hActs1) + self.bs)
                #    cost += -np.log(node.probs[node.label])
                #if (node.left == None and node.right != None and node.right.fprop == True):
                #    h_pre = np.hstack((np.zeros(self.wvecDim,),node.right.hActs1))
                #    node.hActs1 = np.maximum(self.V.dot(h_pre).dot(h_pre.T) + self.W.dot(h_pre) + self.b,0)
                #    node.probs = self.softmax(self.Ws.dot(node.hActs1) + self.bs)
                #    cost += -np.log(node.probs[node.label])
            ## update the node.fprop
        #if node.isLeaf:
        #    node.hActs = self.L[:,node.word]
        #    node.fprop = True

        #else:
        #    if not node.left.fprop: 
        #        c,tot = self.forwardProp(node.left,correct,guess)
        #        cost += c
                #correct += corr
        #        total += tot
        #    if not node.right.fprop:
        #        c,tot = self.forwardProp(node.right,correct,guess)
        #        cost += c
                #correct += corr
        #        total += tot
            # Affine
        #    lr = np.hstack([node.left.hActs, node.right.hActs])
        #    node.hActs = np.dot(self.W,lr) + self.b
        #    node.hActs += np.tensordot(self.V,np.outer(lr,lr),axes=([1,2],[0,1]))
            # Tanh
        #    node.hActs = np.tanh(node.hActs)

        # Softmax
        #node.probs = np.dot(self.Ws,node.hActs) + self.bs
        #node.probs -= np.max(node.probs)
        #node.probs = np.exp(node.probs)
        #node.probs = node.probs/np.sum(node.probs)

        node.fprop = True
        correct.append(node.label)
        guess.append(np.argmax(node.probs))

        #return cost - np.log(node.probs[node.label]),total + 1
        #    node.fprop = True
        
        
        #
        

        return cost,total + 1

    def softmax(self,x):
        x = x - np.max(x)
        return np.exp(x) / sum(np.exp(x))
    
    def backProp(self,node,error=None):

        # Clear nodes
        node.fprop = False
        ###############################
        if (node.isLeaf == False):
             if(node.left != None or node.right != None):
                mask = np.zeros(self.outputDim,)
                mask[node.label] = 1.0
                delta = node.probs - mask
                self.dbs += delta
                ##
                pre_dh = self.Ws.T.dot(delta)
                if(error == None):
                    error_t = 0.0
                    pre_dh += error_t
                else:
                    pre_dh += error
                ## update the self.dW
                self.dWs += np.outer(delta,node.hActs1)
            
                ## propagate the error downward
                error_down = (1 - np.tanh(node.hActs1)**2) * (pre_dh)
                self.db += error_down
                h_pre = np.hstack((node.left.hActs1,node.right.hActs1))
                T = np.outer(h_pre,h_pre.T)
                for i in xrange(self.wvecDim):
                    self.dV[i,:,:] += error_down[i] * T
                
                S = np.zeros(2 * self.wvecDim,)
                
                for i in xrange(self.wvecDim):
                    S += error_down[i] * (self.V[i,:,:] + self.V[i,:,:].T).dot(h_pre)
                
                if(node.left != None):
                    self.dW[:,0:self.wvecDim] += np.outer(error_down,node.left.hActs1)
                    self.backProp(node.left,self.W[:,0:self.wvecDim].T.dot(error_down) + S[:self.wvecDim])
            
                if(node.right != None):
                    self.dW[:,self.wvecDim:] += np.outer(error_down,node.right.hActs1)
                    self.backProp(node.right,self.W[:,self.wvecDim:].T.dot(error_down) + S[self.wvecDim:])
        else:
            mask = np.zeros(self.outputDim,)
            mask[node.label] = 1.0
            delta = node.probs - mask
            self.dWs += np.outer(delta,node.hActs1)
            self.dbs += delta
            pre_dh = self.Ws.T.dot(delta)
            if(error == None):
                error_t = 0.0
                pre_dh += error_t
            else:
                pre_dh += error
            #self.db += np.sign(node.hActs1) * (pre_dh)
            self.dL[node.word] += pre_dh 
        #    pass
        ##end of backprop!!!
        #deltas = node.probs
        #deltas[node.label] -= 1.0
        #self.dWs += np.outer(deltas,node.hActs)
        #self.dbs += deltas
        #deltas = np.dot(self.Ws.T,deltas)
        
        #if error is not None:
        #    deltas += error

        #deltas *= (1-node.hActs**2)

        # Leaf nodes update word vecs
        #if node.isLeaf:
        #    self.dL[node.word] += deltas
        #    return

        # Hidden grad
        #if not node.isLeaf:
        #    lr = np.hstack([node.left.hActs, node.right.hActs])
        #    outer = np.outer(deltas,lr)
        #    self.dV += (np.outer(lr,lr)[...,None]*deltas).T
        #    self.dW += outer
        #    self.db += deltas
        #    # Error signal to children
        #    deltas = np.dot(self.W.T, deltas) 
        #    deltas += np.tensordot(self.V.transpose((0,2,1))+self.V,
        #                           outer.T,axes=([1,0],[0,1]))
        #    self.backProp(node.left, deltas[:self.wvecDim])
        #    self.backProp(node.right, deltas[self.wvecDim:])
        
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

        print "Checking dW... (might take a while)"
        for W,dW in zip(self.stack[1:],grad[1:]):
            W = W[...,None,None] # add dimension since bias is flat
            dW = dW[...,None,None] 
            for i in xrange(W.shape[0]):
                for j in xrange(W.shape[1]):
                    for k in xrange(W.shape[2]):
                        W[i,j,k] += epsilon
                        costP,_ = self.costAndGrad(data)
                        W[i,j,k] -= epsilon
                        numGrad = (costP - cost)/epsilon
                        err = np.abs(dW[i,j,k] - numGrad)
                        #print "Analytic %.9f, Numerical %.9f, Relative Error %.9f"%(dW[i,j,k],numGrad,err)
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
                #print "Analytic %.9f, Numerical %.9f, Relative Error %.9f"%(dL[j][i],numGrad,err)
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
    outputDim = 5

    nn = RNTN(wvecDim,outputDim,numW,mbSize=4)
    nn.initParams()

    mbData = train[:1]
    #cost, grad = nn.costAndGrad(mbData)

    print "Numerical gradient check..."
    nn.check_grad(mbData)






