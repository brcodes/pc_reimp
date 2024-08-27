import numpy as np

'''
remember to make these self methods within the class first
'''

def r_updates_n_1_no_transform(self, label):
    '''
    move to static eventually, as well as update_Component assignment
    '''
        
    kr_1 = self.kr[1]
    ssq_1 = self.ssq[1]
    U_1 = self.U[1]
    r_1 = self.r[1]
    
    #U1 operations
    U1_transpose = np.transpose(U_1, self.U1_transpose_dims)
    U1_tdot_r1 = np.tensordot(U_1, r_1, axes=([-1],[0]))
    input_min_U1tdotr1 = self.r[0] - self.f(U1_tdot_r1)[0]
    
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
    U1_transpose = np.transpose(U_1, self.U1_transpose_dims)
    U1_tdot_r1 = np.tensordot(U_1, r_1, axes=([-1],[0]))
    input_min_U1tdotr1 = self.r[0] - self.f(U1_tdot_r1)[0]
    
    self.r[1] += (kr_1 / ssq_1) * np.tensordot(U1_transpose, input_min_U1tdotr1, axes=(self.U1T_tdot_dims, self.input_min_U1tdotr1_tdot_dims)) \
                                        + (kr_2 * ssq_2) * (self.f(U_2.dot(r_2))[0] - r_1) \
                                        - (kr_1 / ssq_1) * self.g(r_1, self.alph[1])[1]
                                        
    self.r[2] += (kr_2 / ssq_2) * (U_2.T.dot(self.r[1] - self.f(U_2.dot(r_2))[0])) \
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
    U1_transpose = np.transpose(U_1, self.U1_transpose_dims)
    U1_tdot_r1 = np.tensordot(U_1, r_1, axes=([-1],[0]))
    input_min_U1tdotr1 = self.r[0] - self.f(U1_tdot_r1)[0]
    
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
        
        self.r[i] += (kr_i / ssq_i) * (U_i.T.dot(self.r[i-1] - self.f(U_i.dot(r_i))[0])) \
                                            + (self.kr[i+1] * self.ssq[i+1]) * (self.f(self.U[i+1].dot(self.r[i+1]))[0] - r_i) \
                                            - (kr_i / ssq_i) * self.g(r_i, self.alph[i])[1]

    # Layer n
    kr_n = self.kr[n]
    ssq_n = self.ssq[n]
    U_n = self.U[n]
    r_n = self.r[n]

    self.r[n] += (kr_n / ssq_n) * (U_n.T.dot(self.r[n-1] - self.f(U_n.dot(r_n))[0])) \
                                            + self.rn_topdown_term_dict[self.classif_method](label) \
                                            - (kr_n / ssq_n) * self.g(r_n, self.alph[n])[1]
                                            
def U_updates_n_1_no_transform(self,label):

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
    input_min_U1tdotr1 = self.r[0] - self.f(U1_tdot_r1)[0]
    
    # Layer 1
    self.U[1] += (kU_1 / ssq_1) * np.outer(input_min_U1tdotr1, r_1) \
                    - (kU_1 / ssq_1) * self.h(U_1, self.lam[1])[1]
                        
def U_updates_n_gt_eq_2_no_transform(self,label):
    
    kU_1 = self.kU[1]
    ssq_1 = self.ssq[1]
    U_1 = self.U[1]
    r_1 = self.r[1]
    
    #U1 operations
    U1_tdot_r1 = np.tensordot(U_1, r_1, axes=([-1],[0]))
    input_min_U1tdotr1 = self.r[0] - self.f(U1_tdot_r1)[0]
    
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
        self.U[i] += (kU_i / ssq_i) * np.outer((self.r[i-1] - self.f(U_i.dot(r_i))[0]), r_i) \
                    - (kU_i / ssq_i) * self.h(U_i, self.lam[i])[1]
                    
def Uo_update_no_transform(self, label):
            # Format: Uo += kU_o / ssq_o * (label - softmax(Uo.dot(r_n)))
            '''
            check k/2 vs k/ssqo
            for every top down rn update, U update, V update, (place where a lr is used)
            '''
            o = 'o'
            r_n = self.r[self.num_layers]
            self.U[o] += (self.kU[o]/ self.ssq[o]) * np.outer((label - self.softmax(self.U[o].dot(r_n))), r_n)

def rep_cost_n_1_no_transform(self):
    '''
    move to static eventually, as well as update_Component assignment
    '''
        
    r_0 = self.r[0]
    r_1 = self.r[1]
    U_1 = self.U[1]
    ssq_1 = self.ssq[1]
    
    #U1 operations
    U1_tdot_r1 = np.tensordot(U_1, r_1, axes=([-1],[0]))
    
    # 1st layer axes necessary for dot product (2D or 4D)
    bu_tdot_dims = self.bu_cost_tdot_dims
    
    # Bottom up only
    bu_v = r_0 - self.f(U1_tdot_r1)[0]
    bu_sq = np.tensordot(bu_v, bu_v, axes=(bu_tdot_dims, bu_tdot_dims))
    bu_tot = (1 / ssq_1) * bu_sq
    
    # Priors
    pri_r = self.g(np.squeeze(r_1), self.alph[1])[0]
    pri_U = self.h(U_1, self.lam[1])[0]
    
    return bu_tot + pri_r + pri_U

def rep_cost_n_2_no_transform(self):
    '''
    move to static eventually, as well as update_Component assignment
    '''
        
    r_0 = self.r[0]
    r_1 = self.r[1]
    U_1 = self.U[1]
    r_2 = self.r[2]
    U_2 = self.U[2]
    ssq_1 = self.ssq[1]
    ssq_2 = self.ssq[2]
    
    #U1 operations
    U1_tdot_r1 = np.tensordot(U_1, r_1, axes=([-1],[0]))
    
    # 1st layer axes necessary for dot product (2D or 4D)
    bu_tdot_dims = self.bu_cost_tdot_dims
    
    # Bottom up and td
    bu_v = r_0 - self.f(U1_tdot_r1)[0]
    bu_sq = np.tensordot(bu_v, bu_v, axes=(bu_tdot_dims, bu_tdot_dims))
    bu_tot = (1 / ssq_1) * bu_sq
    
    td_v = r_1 - self.f(U_2.dot(r_2))[0]
    td_sq = td_v.dot(td_v)
    td_tot = (1 / ssq_2) * td_sq
    
    # Priors
    pri_r1 = self.g(np.squeeze(r_1), self.alph[1])[0]
    pri_U1 = self.h(U_1, self.lam[1])[0]
    
    '''
    this will be identical to td layer 1
    another impetus to reduce cost
    '''
    # Bottom up Layer 2
    bu_tot2 = td_tot
    
    pri_r2 = self.g(np.squeeze(r_2), self.alph[2])[0]
    pri_U2 = self.h(U_2, self.lam[2])[0]
    
    return bu_tot + td_tot + bu_tot2 + pri_r1 + pri_U1 + pri_r2 + pri_U2

def rep_cost_n_gt_eq_3_no_transform(self):
    '''
    move to static eventually, as well as update_Component assignment
    '''
        
    r_0 = self.r[0]
    r_1 = self.r[1]
    U_1 = self.U[1]
    r_2 = self.r[2]
    U_2 = self.U[2]
    ssq_1 = self.ssq[1]
    ssq_2 = self.ssq[2]
    
    #U1 operations
    U1_tdot_r1 = np.tensordot(U_1, r_1, axes=([-1],[0]))
    
    # 1st layer axes necessary for dot product (2D or 4D)
    bu_tdot_dims = self.bu_cost_tdot_dims
    
    # Bottom up and td
    bu_v = r_0 - self.f(U1_tdot_r1)[0]
    bu_sq = np.tensordot(bu_v, bu_v, axes=(bu_tdot_dims, bu_tdot_dims))
    bu_tot = (1 / ssq_1) * bu_sq
    
    td_v = r_1 - self.f(U_2.dot(r_2))[0]
    td_sq = td_v.dot(td_v)
    td_tot = (1 / ssq_2) * td_sq
    
    # Priors
    pri_r1 = self.g(np.squeeze(r_1), self.alph[1])[0]
    pri_U1 = self.h(U_1, self.lam[1])[0]
    
    n = self.num_layers
    # For layers 2 to n-1
    bu_i = 0
    td_i = 0
    pri_ri = 0
    pri_Ui = 0
    for i in range(2,n):
        bu_v = self.r[i-1] - self.f(self.U[i].dot(self.r[i]))[0]
        bu_sq = bu_v.dot(bu_v)
        bu_tot += (1 / self.ssq[i]) * bu_sq
        
        td_v = self.r[i] - self.f(self.U[i+1].dot(self.r[i+1]))[0]
        td_sq = td_v.dot(td_v)
        td_tot += (1 / self.ssq[i+1]) * td_sq
    
        pri_r = self.g(np.squeeze(self.r[i]), self.alph[i])[0]
        pri_U = self.h(self.U[i], self.lam[i])[0]
    
        bu_i += bu_tot
        td_i += td_tot
        pri_ri += pri_r
        pri_Ui += pri_U
        
    # Final layer will only have bu term
    bu_vn = self.r[n-1] - self.f(self.U[n].dot(self.r[n]))[0]
    bu_sqn = bu_vn.dot(bu_vn)
    bu_totn = (1 / self.ssq[n]) * bu_sqn
    
    pri_rn = self.g(np.squeeze(self.r[n]), self.alph[n])[0]
    pri_Un = self.h(self.U[n], self.lam[n])[0]
    
    return bu_tot + td_tot + pri_r1 + pri_U1 + bu_i + td_i + pri_ri + pri_Ui + bu_totn + pri_rn + pri_Un