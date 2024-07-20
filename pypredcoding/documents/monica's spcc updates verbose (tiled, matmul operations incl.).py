# verbose monica's spcc updates



for i in range(self.iteration):


    # prediction errors
    e0 = (inputs - np.matmul(self.U1, r1[:, :, None]).squeeze())
    e1 = (r1 - self.U2.dot(r2).reshape(r1.shape))
    e2 = (r2 - self.U3.dot(r3))
    e3 = ((np.exp(r3)/np.sum(np.exp(r3))) - label if training else label*0) # softmax cross-entropy loss
    

    # r updates
    self.r1 = self.r1 + (self.k_r/self.sigma_sq0) * np.matmul(np.transpose(self.U1, axes=(0,2,1)), (inputs - np.matmul(self.U1, r1[:, :, None]).squeeze())[:, :, None]).squeeze() \
            + (self.k_r/self.sigma_sq1) * -(r1 - self.U2.dot(r2).reshape(r1.shape)) \
            - self.k_r * self.alpha1 * r1 / self.prior_trans(r1, self.prior)

    self.r2 = self.r2 +  (self.k_r / self.sigma_sq1) * self.U2.T.dot((r1 - self.U2.dot(r2).reshape(r1.shape)).flatten()) \
            + (self.k_r / self.sigma_sq2) * -(r2 - self.U3.dot(r3)) \
            - self.k_r * self.alpha2 * r2 / self.prior_trans(r2, self.prior)

    self.r3 = self.r3 +  (self.k_r / self.sigma_sq2) * self.U3.T.dot((r2 - self.U3.dot(r3))) \
        + (self.k_r / self.sigma_sq3) * -((np.exp(r3)/np.sum(np.exp(r3))) - label if training else label*0) \
            - self.k_r * self.alpha3 * r3 / self.prior_trans(r3, self.prior)

    self.U1 = self.U1 + (self.k_U/self.sigma_sq0) * np.matmul((inputs - np.matmul(self.U1, r1[:, :, None]).squeeze())[:, :, None], r1[:, None, :]) \
            - self.k_U * self.lambda1 * self.U1 / self.prior_trans(self.U1, self.prior)

    self.U2 = self.U2 + (self.k_U / self.sigma_sq1) * np.outer((r1 - self.U2.dot(r2).reshape(r1.shape)).flatten(), r2) \
            - self.k_U * self.lambda2 * self.U2 / self.prior_trans(self.U2, self.prior)

    self.U3 = self.U3 + (self.k_U / self.sigma_sq2) * np.outer((r2 - self.U3.dot(r3)), r3) \
            - self.k_U * self.lambda3 * self.U3 / self.prior_trans(self.U3, self.prior)



# verbose monica's stitched into our code,
# except priors

for iteration in range(0,self.num_rUsimul_iters):
                            
    ### r loop (splitting r loop, U loop mimic's Li architecture)
    ### (i ... n-1)
    for i in range(1, n):
        
        if i == 1:

            # r update
            self.r[i] = self.r[i] + (k_r / self.p.sigma_sq[i]) \
            * np.matmul(np.transpose(self.U[i], axes=(0,2,1)), (self.r[i-1] - np.matmul(self.U[i], self.r[i][:, :, None]).squeeze())[:, :, None]).squeeze()
            # * self.U[i].T.dot(self.f(self.r[i-1] - self.f(self.U[i].dot(self.r[i]))[0])[1]) \
                
            + (k_r/self.p.sigma_sq[i+1]) * -(self.r[i] - self.U[i+1].dot(self.r[i+1]).reshape(self.r[i].shape))
            # + (k_r / self.p.sigma_sq[i+1]) * (self.f(self.U[i+1].dot(self.r[i+1]))[0] - self.r[i]) \
                
            - (k_r / 2) * self.g(self.r[i],self.p.alpha[i])[1]
            
        if i == 2:

            # r update
            self.r[i] = self.r[i] + (k_r / self.p.sigma_sq[i]) \
                
            * self.U[i].T.dot((self.r[i-1] - self.U[i].dot(self.r[i]).reshape(self.r[i-1].shape)).flatten()) \
            # * self.U[i].T.dot(self.f(self.r[i-1] - self.f(self.U[i].dot(self.r[i]))[0])[1]) \
                
            + (k_r / self.p.sigma_sq[i+1]) * -(self.r[i] - self.U[i+1].dot(self.r[i+1]))
            # + (k_r / self.p.sigma_sq[i+1]) * (self.f(self.U[i+1].dot(self.r[i+1]))[0] - self.r[i]) \
                
            - (k_r / 2) * self.g(self.r[i],self.p.alpha[i])[1]
            
        else:
            
            # r update
            self.r[i] = self.r[i] + (k_r / self.p.sigma_sq[i]) \
            * self.U[i].T.dot(self.f(self.r[i-1] - self.f(self.U[i].dot(self.r[i]))[0])[1]) \
            + (k_r / self.p.sigma_sq[i+1]) * (self.f(self.U[i+1].dot(self.r[i+1]))[0] - self.r[i]) \
            - (k_r / 2) * self.g(self.r[i],self.p.alpha[i])[1]

    # final r (Li's "localist") layer update
    self.r[n] = self.r[n] + (k_r / self.p.sigma_sq[n]) \
    * self.U[n].T.dot(self.f(self.r[n-1] - self.f(self.U[n].dot(self.r[n]))[0])[1]) \
    - (k_r / 2) * self.g(self.r[n],self.p.alpha[n])[1] \

    # later: change based on C1, C2 or NC setting
    # C1 for now
    # size eg 212,1 label , 212,1 r[n]
    # only one r learning rate in Li 212.
    + ((k_r) * (label[:,None] - softmax(self.r[n])))

    ### U loop ( i ... n)
    for i in range(1, n+1):
        
        if i == 1:

            # U update
            self.U[i] = self.U[i] + (k_U / self.p.sigma_sq[i]) \
            
            * np.matmul((self.r[i-1] - np.matmul(self.U[i], self.r[i][:, :, None]).squeeze())[:, :, None], self.r[i][:, None, :]) \
            # * self.f(self.r[i-1] - self.f(self.U[i].dot(self.r[i]))[0])[1].dot(self.r[i].T) \
                
            - (k_U / 2) * self.h(self.U[i],self.p.lam[i])[1]
            
        if i == 2:
            
            # U update
            self.U[i] = self.U[i] + (k_U / self.p.sigma_sq[i]) \
                
            * np.outer((self.r[i-1] - self.U[i].dot(self.r[i]).reshape(self.r[i-1].shape)).flatten(), self.r[i]) \ 
            # * self.f(self.r[i-1] - self.f(self.U[i].dot(self.r[i]))[0])[1].dot(self.r[i].T) \
                
            - (k_U / 2) * self.h(self.U[i],self.p.lam[i])[1]
            
        else:
            
            # U update
            self.U[i] = self.U[i] + (k_U / self.p.sigma_sq[i]) \
            * self.f(self.r[i-1] - self.f(self.U[i].dot(self.r[i]))[0])[1].dot(self.r[i].T) \
            - (k_U / 2) * self.h(self.U[i],self.p.lam[i])[1]