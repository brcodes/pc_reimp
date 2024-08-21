import numpy as np

def r_updates_n_1_no_transform(self, label):
    '''
    move to static eventually, as well as update_Component assignment
    '''
        
    kr_1 = self.kr[1]
    ssq_1 = self.ssq[1]
    U_1 = self.U[1]
    r_1 = self.r[1]
    
    #U1 operations
    U1_transpose = np.transpose_U(U_1, self.U1_transpose_dims)
    U1_tdot_r1 = np.tensordot(U_1, r_1, axes=([-1],[0]))
    input_min_U1tdotr1 = self.f(self.r[0] - self.f(U1_tdot_r1)[0])[1]
    
    self.r[1] += (kr_1 / ssq_1) * np.tensordot(U1_transpose, input_min_U1tdotr1, axes=(self.U1T_tdot_dims, self.input_min_U1tdotr1_tdot_dims)) \
                                            + self.rn_topdown_term_dict[self.classif_method](label) \
                                            - (kr_1 / ssq_1) * self.g(r_1, self.alph[1])[1]

def r_updates_n_2_no_transform(self, label):
    
    '''
    two layer model
    '''
    kr_1 = self.kr[1]
    ssq_1 = self.ssq[1]
    U_1 = self.U[1]
    r_1 = self.r[1]
    
    kr_2 = self.kr[2]
    ssq_2 = self.ssq[2]
    U_2 = self.U[2]
    r_2 = self.r[2]
    
    #U1 operations
    U1_transpose = np.transpose_U(U_1, self.U1_transpose_dims)
    U1_tdot_r1 = np.tensordot(U_1, r_1, axes=([-1],[0]))
    input_min_U1tdotr1 = self.f(self.r[0] - self.f(U1_tdot_r1)[0])[1]
    
    self.r[1] += (kr_1 / ssq_1) * np.tensordot(U1_transpose, input_min_U1tdotr1, axes=(self.U1T_tdot_dims, self.input_min_U1tdotr1_tdot_dims)) \
                                        + (kr_2 * ssq_2) * (self.f(U_2.dot(r_2))[0] - r_1) \
                                        - (kr_1 / ssq_1) * self.g(r_1, self.alph[1])[1]
                                        
    self.r[2] += (kr_2 / ssq_2) * (U_2.T.dot(self.f(self.r[1] - self.f(U_2.dot(r_2))[0])[1])) \
                                            + self.rn_topdown_term_dict[self.classif_method](label) \
                                            - (kr_2 / ssq_2) * self.g(r_2, self.alph[2])[1]
                                        
def r_updates_n_gt_eq_3_no_transform(self, label):
    
    n = self.num_layers
                                            
    kr_1 = self.kr[1]
    ssq_1 = self.ssq[1]
    U_1 = self.U[1]
    r_1 = self.r[1]
    
    kr_2 = self.kr[2]
    ssq_2 = self.ssq[2]
    U_2 = self.U[2]
    r_2 = self.r[2]
    
    #U1 operations
    U1_transpose = np.transpose_U(U_1, self.U1_transpose_dims)
    U1_tdot_r1 = np.tensordot(U_1, r_1, axes=([-1],[0]))
    input_min_U1tdotr1 = self.f(self.r[0] - self.f(U1_tdot_r1)[0])[1]
    
    # Layer 1
    self.r[1] += (kr_1 / ssq_1) * np.tensordot(U1_transpose, input_min_U1tdotr1, axes=(self.U1T_tdot_dims, self.input_min_U1tdotr1_tdot_dims)) \
                                        + (kr_2 * ssq_2) * (self.f(U_2.dot(r_2))[0] - r_1) \
                                        - (kr_1 / ssq_1) * self.g(r_1, self.alph[1])[1]
    # Layers 2 to n-1                                    
    for i in range(2,n):
        
        kr_i = self.kr[i]
        ssq_i = self.ssq[i]
        r_i = self.r[i]
        U_i = self.U[i]
        
        self.r[i] += (kr_i / ssq_i) * (U_i.T.dot(self.f(self.r[i-1] - self.f(U_i.dot(r_i))[0])[1])) \
                                            + (self.kr[i+1] * self.ssq[i+1]) * (self.f(self.U[i+1].dot(self.r[i+1]))[0] - r_i) \
                                            - (kr_i / ssq_i) * self.g(r_i, self.alph[i])[1]

    # Layer n
    kr_n = self.kr[n]
    ssq_n = self.ssq[n]
    U_n = self.U[n]
    r_n = self.r[n]

    self.r[n] += (kr_n / ssq_n) * (U_n.T.dot(self.f(self.r[n-1] - self.f(U_n.dot(r_n))[0])[1])) \
                                            + self.rn_topdown_term_dict[self.classif_method](label) \
                                            - (kr_n / ssq_n) * self.g(r_n, self.alph[n])[1]
                                            
def U_updates_n_1_no_transform(self):

    '''u1 will need a tensor dot
    '''
    
    '''
    check if F will work with 3d+ Us
    '''
    
    kU_1 = self.kU[1]
    ssq_1 = self.ssq[1]
    U_1 = self.U[1]
    r_1 = self.r[1]
    
    #U1 operations
    U1_tdot_r1 = np.tensordot(U_1, r_1, axes=([-1],[0]))
    input_min_U1tdotr1 = self.f(self.r[0] - self.f(U1_tdot_r1)[0])[1]
    
    # Layer 1
    self.U[1] += (kU_1 / ssq_1) * np.outer(input_min_U1tdotr1, r_1) \
                    - (kU_1 / ssq_1) * self.h(U_1, self.lam[1])[1]
                        
def U_updates_n_gt_eq_2_no_transform(self):
    
    kU_1 = self.kU[1]
    ssq_1 = self.ssq[1]
    U_1 = self.U[1]
    r_1 = self.r[1]
    
    #U1 operations
    U1_tdot_r1 = np.tensordot(U_1, r_1, axes=([-1],[0]))
    input_min_U1tdotr1 = self.f(self.r[0] - self.f(U1_tdot_r1)[0])[1]
    
    # Layer 1
    self.U[1] += (kU_1 / ssq_1) * np.outer(input_min_U1tdotr1, r_1) \
                    - (kU_1 / ssq_1) * self.h(U_1, self.lam[1])[1]
    
    n = self.num_layers
    
    #i>1 - n will all be the same
    for i in range(1,n+1):
        
        kU_i = self.kU[i]
        ssq_i = self.ssq[i]
        r_i = self.r[i]
        U_i = self.U[i]
        
        #i
        self.U[i] += (kU_i / ssq_i) * np.outer((self.f(self.r[i-1] - self.f(U_i.dot(r_i))[0])[1]), r_i) \
                    - (kU_i / ssq_i) * self.h(U_i, self.lam[i])[1]
                    
                    
def rep_cost_n_1_no_transform(self):
    pass

def rep_cost_n_2_no_transform(self):
    pass

def rep_cost_n_gt_eq_3_no_transform(self):
    pass