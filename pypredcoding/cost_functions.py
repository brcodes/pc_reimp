import numpy as np

# cost_functions.py
class StaticCostFunction():
    def __init__(self, sPCC):
        # Copy attributes from the model
        self.__dict__.update(sPCC.__dict__)
        
        # Create dictionaries for each unique method prefix
        self.U1r1_dict = self.create_subcomponent_dict("U1r1")
        self.U1T_Idiff_dict = self.create_subcomponent_dict("U1T_Idiff")
        self.U2r2_dict = self.create_subcomponent_dict("U2r2")
        self.U2T_L1diff_dict = self.create_subcomponent_dict("U2T_L1diff")
        self.U3r3_dict = self.create_subcomponent_dict("U3r3")
        self.U3T_L2diff_dict = self.create_subcomponent_dict("U3T_L2diff")
        # U-only components now
        self.Idiff_r1_dict = self.create_subcomponent_dict("Idiff_r1")
        self.L1diff_r2_dict = self.create_subcomponent_dict("L1diff_r2")
        self.L2diff_r3_dict = self.create_subcomponent_dict("L2diff_r3")
        
        # Initialize general methods based on the architecture key
        self.initialize_components(self.architecture)
    
    def create_subcomponent_dict(self, prefix):
        return {
            "flat_hidden_layers": getattr(self, f"{prefix}_fhl"),
            "expand_1st_layer_Li": getattr(self, f"{prefix}_e1l_Li"),
            "expand_1st_layer": getattr(self, f"{prefix}_e1l")
        }
        
    def initialize_components(self, architecture_key):
        
        def U1T_mult_args(self):
            pass
    
        self.U1r1 = self.U1r1_dict[architecture_key]
        self.U1T_Idiff = self.U1T_Idiff_dict[architecture_key]
        self.U2r2 = self.U2r2_dict[architecture_key]
        self.U2T_L1diff = self.U2T_L1diff_dict[architecture_key]
        self.U3r3 = self.U3r3_dict[architecture_key]
        self.U3T_L2diff = self.U3T_L2diff_dict[architecture_key]
        # U-only
        self.Idiff_r1 = self.Idiff_r1_dict[architecture_key]
        self.L1diff_r2 = self.L1diff_r2_dict[architecture_key]
        self.L2diff_r3 = self.L2diff_r3_dict[architecture_key]
        
    # def initialize_components(self, architecture_key):
    #     self.U1mat_mult_r1vecormat = self.U1r1_dict[architecture_key]
    #     self.U1Tmat_mult_Idiffmat = self.U1T_Idiff_dict[architecture_key]
    #     self.U2mat_mult_r2vec = self.U2r2_dict[architecture_key]
    #     self.U2Tmat_mult_L1diffvecormat = self.U2T_L1diff_dict[architecture_key]
    #     self.U_gteq3_mat_mult_r_gteq3_vec = self.U3r3_dict[architecture_key]
    #     self.U3Tmat_mult_L2diffvec = self.U3T_L2diff_dict[architecture_key]
    #     # U-only
    #     self.Idiffmat_mult_r1vecormat = self.Idiff_r1_dict[architecture_key]
    #     self.L1diffvecormat_mult_r2vec = self.L1diff_r2_dict[architecture_key]
    #     self.L_gteq2_diffvec_mult_r_gteq3_vec = self.L2diff_r3_dict[architecture_key]
            
    '''
    cost function subcomponents
    
    used in r or U updates
    dependent on architecture parameter
    '''
    
    def U1r1_fhl(self):
        pass
    
    def U1r1_e1l_Li(self):
        pass
    
    def U1r1_e1l(self):
        pass
    
    def U1T_Idiff_fhl(self):
        pass
    
    def U1T_Idiff_e1l_Li(self):
        pass
    
    def U1T_Idiff_e1l(self):
        pass
    
    def U2r2_fhl(self):
        pass
    
    def U2r2_e1l_Li(self):
        pass
    
    def U2r2_e1l(self):
        pass
    
    def U2T_L1diff_fhl(self):
        pass
    
    def U2T_L1diff_e1l_Li(self):
        pass
    
    def U2T_L1diff_e1l(self):
        pass
    
    def U3r3_fhl(self):
        pass
    
    def U3r3_e1l_Li(self):
        pass
    
    def U3r3_e1l(self):
        pass
    
    def U3T_L2diff_fhl(self):
        pass
    
    def U3T_L2diff_e1l_Li(self):
        pass
    
    def U3T_L2diff_e1l(self):
        pass
    
    '''
    cost function subcomponents
    
    used in U updates only
    dependent on architecture parameter
    '''
    
    def Idiff_r1_fhl(self):
        pass
    
    def Idiff_r1_e1l_Li(self):
        pass
    
    def Idiff_r1_e1l(self):
        pass
    
    def L1diff_r2_fhl(self):
        pass
    
    def L1diff_r2_e1l_Li(self):
        pass
    
    def L1diff_r2_e1l(self):
        pass
    
    def L2diff_r3_fhl(self):
        pass
    
    def L2diff_r3_e1l_Li(self):
        pass
    
    def L2diff_r3_e1l(self):
        pass
    
   
    '''
    actual cost functions
    '''    
        
    def rep_cost_n_1(self):
        r1 = self.r[1]
        U1 = self.U[1]
        bu_tdot_dims = self.bu_error_tdot_dims

        U1r1 = self.U1mat_mult_r1vecormat(U1, r1)
        
        # Bottom-up component of the representation error
        bu_vec = self.r[0] - U1r1
        bu_square = np.tensordot(bu_vec, bu_vec, axes=(bu_tdot_dims, bu_tdot_dims))
        bu_total = (1 / self.ssq[1]) * bu_square
        
        # Priors on that layer
        prior_r = self.g(np.squeeze(r1), self.alph[1])[0]
        prior_U = self.h(U1, self.lam[1])[0]
        
        # Return total
        return bu_total + prior_r + prior_U
        
        pass
    
    def rep_cost_n_2(self):
        r1 = self.r[1]
        r2 = self.r[2]
        U1 = self.U[1]
        U2 = self.U[2]
        bu_tdot_dims = self.bu_error_tdot_dims
        td_tdot_dims = self.td_error_tdot_dims

        # Layer 1
        U1r1 = self.U1mat_mult_r1vecormat(U1, r1)
        
        # Bottom-up component of the representation error
        bu_vec = self.r[0] - U1r1
        bu_square = np.tensordot(bu_vec, bu_vec, axes=(bu_tdot_dims, bu_tdot_dims))
        bu_total = (1 / self.ssq[1]) * bu_square
        
        U2r2 = self.U2mat_mult_r2vec(U2, r2)
        
        # Top-down component of the representation error
        td_vec = r1 - U2r2
        td_square = np.tensordot(td_vec, td_vec, axes=(td_tdot_dims, td_tdot_dims))
        td_total = (1 / self.ssq[2]) * td_square
        
        # Priors on that layer
        prior_r = self.g(np.squeeze(r1), self.alph[1])[0]
        prior_U = self.h(U1, self.lam[1])[0]
        
        # Layer 2
        bu_total2 = td_total
        
        prior_r2 = self.g(np.squeeze(r2), self.alph[2])[0]
        prior_U2 = self.h(U2, self.lam[2])[0]
        
        # Return total
        return bu_total + td_total + prior_r + prior_U + bu_total2 + prior_r2 + prior_U2
    
    def rep_cost_n_gt_eq_3(self):
        
        n = self.num_layers
        r1 = self.r[1]
        r2 = self.r[2]
        U1 = self.U[1]
        U2 = self.U[2]
        bu_tdot_dims = self.bu_error_tdot_dims
        td_tdot_dims = self.td_error_tdot_dims

        # Layer 1
        U1r1 = self.U1mat_mult_r1vecormat(U1, r1)
        
        # Bottom-up component of the representation error
        bu_vec = self.r[0] - U1r1
        bu_square = np.tensordot(bu_vec, bu_vec, axes=(bu_tdot_dims, bu_tdot_dims))
        bu_total = (1 / self.ssq[1]) * bu_square
        
        U2r2 = self.U2mat_mult_r2vec(U2, r2)
        
        # Top-down component of the representation error
        td_vec = r1 - U2r2
        td_square = np.tensordot(td_vec, td_vec, axes=(td_tdot_dims, td_tdot_dims))
        td_total = (1 / self.ssq[2]) * td_square
        
        # Priors on that layer
        prior_r = self.g(np.squeeze(r1), self.alph[1])[0]
        prior_U = self.h(U1, self.lam[1])[0]
        
        # Layer 2
        bu_total2 = td_total
        
        U3r3 = self.U3mat_mult_r3vec(self.U[3], self.r[3])
        
        # Top-down component
        td_vec2 = r2 - U3r3
        td_square2 = np.dot(td_vec2, td_vec2)
        td_total2 = (1 / self.ssq[3]) * td_square2
        
        prior_r2 = self.g(np.squeeze(r2), self.alph[2])[0]
        prior_U2 = self.h(U2, self.lam[2])[0]
        
        # Return total
        return bu_total + td_total + prior_r + prior_U + bu_total2 + prior_r2 + prior_U2
    
    def classif_cost_c1(self, label):
        # Format: -label.dot(np.log(softmax(r_n)))
        return -label.dot(np.log(self.softmax_func(self.r[self.num_layers])))
    
    def classif_cost_c2(self, label):
        # Format: -label.dot(np.log(softmax(Uo.dot(r_n))))
        o = 'o'
        return -label.dot(np.log(self.softmax_func(self.U[o].dot(self.r[self.num_layers])))) + self.h(self.U[o], self.lam[o])[0]
    
    def classif_cost_None(self, label):
        return 0
    
    def rn_topdown_upd_c1(self, label):
        '''
        redo for recurrent =will all be the same except rn_bar'''
        n = self.num_layers
        o = 'o'
        # Format: (k_o / lr_denom) * (label - softmax(r_n))
        c1 = (self.kr[o] / lr_denominator) * (label - self.softmax_func(self.r[n]))
        return c1

    def rn_topdown_upd_c2(self, label):
        # Format: (k_o / 2) * (label - softmax(Uo.dot(r_n)))
        # No "Li" denominator option here, because she never ran a C2 model.
        n = self.num_layers
        o = 'o'
        c2 = (self.kr[o] / 2) * (label - self.softmax_func(self.U[o].dot(self.r[n])))
        return c2
    
    def rn_topdown_upd_None(self, label):
        return 0
    
    def r_updates_n_1(self, label):
        pass
    
    def r_updates_n_2(self, label):
        pass
    
    def r_updates_n_gt_eq_3(self, label):
        pass
    
    def U_updates_n_1(self,label):
        pass
    
    def U_updates_n_gt_eq_2(self,label):
        pass
    
    def Uo_update(self, label):
        # Format: Uo += kU_o / 2 * (label - softmax(Uo.dot(r_n)))
        # No "Li" denominator option here, because she never ran a C2 model.

        o = 'o'
        r_n = self.r[self.num_layers]
        self.U[o] += (self.kU[o] / 2) * np.outer((label - self.softmax_func(self.U[o].dot(r_n))), r_n)

    def classif_guess_c1(self, label):
        guess = np.argmax(self.softmax_func(self.r[self.num_layers]))
        if guess == np.argmax(label):
            return 1
        else:
            return 0
    
    def classif_guess_c2(self, label):
        guess = np.argmax(self.softmax_func(self.U['o'].dot(self.r[self.num_layers])))
        if guess == np.argmax(label):
            return 1
        else:
            return 0
        
    def classif_guess_None(self, label):
        return 0
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    def rep_cost_n_1(self):
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
        bu_tdot_dims = self.bu_error_tdot_dims
        
        # Bottom up only
        bu_v = r_0 - self.f(U1_tdot_r1)[0]
        bu_sq = np.tensordot(bu_v, bu_v, axes=(bu_tdot_dims, bu_tdot_dims))
        bu_tot = (1 / ssq_1) * bu_sq
        
        # Priors
        pri_r = self.g(np.squeeze(r_1), self.alph[1])[0]
        pri_U = self.h(U_1, self.lam[1])[0]
        
        return bu_tot + pri_r + pri_U

    def rep_cost_n_2(self):
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
        bu_tdot_dims = self.bu_error_tdot_dims
        
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

    def rep_cost_n_gt_eq_3(self):
        '''
        move to static eventually, as well as update_Component assignment
        '''
            
        r_0 = self.r[0]
        r_1 = self.r[1]
        U_1 = self.U[1]
        r_2 = self.r[2]
        U_2 = self.U[2]
        ssq_0 = self.ssq[0]
        ssq_1 = self.ssq[1]
        
        '''
        Li style: test
        '''
        
        #U1 operations
        U1_tdot_r1 = np.einsum('ijk,ik->ij', U_1, r_1)
        
        # 1st layer axes necessary for dot product (2D or 4D)
        bu_tdot_dims = self.bu_error_tdot_dims
        
        # Bottom up and tdw
        bu_v = r_0 - self.f(U1_tdot_r1)[0]
        bu_sq = np.tensordot(bu_v, bu_v, axes=(bu_tdot_dims, bu_tdot_dims))
        bu_tot = (1 / ssq_0) * bu_sq
        
        U2_dot_r2 = self.f(U_2.dot(r_2))[0]
        
        td_v = r_1.reshape(U2_dot_r2.shape) - U2_dot_r2
        td_sq = td_v.dot(td_v)
        td_tot = (1 / ssq_1) * td_sq
        
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
            
            if i == 2:
                fUi_dot_ri = self.f(self.U[i].dot(self.r[i]))[0]
                bu_v = self.r[i-1].reshape(fUi_dot_ri.shape) - fUi_dot_ri
            else:
                bu_v = self.r[i-1] - self.f(self.U[i].dot(self.r[i]))[0]
                
            bu_sq = bu_v.dot(bu_v)
            bu_tot += (1 / self.ssq[i-1]) * bu_sq
            
            td_v = self.r[i] - self.f(self.U[i+1].dot(self.r[i+1]))[0]
            td_sq = td_v.dot(td_v)
            td_tot += (1 / self.ssq[i]) * td_sq
        
            pri_r = self.g(np.squeeze(self.r[i]), self.alph[i])[0]
            pri_U = self.h(self.U[i], self.lam[i])[0]
        
            bu_i += bu_tot
            td_i += td_tot
            pri_ri += pri_r
            pri_Ui += pri_U
            
        '''
        test
        '''
            
        # Final layer will only have bu term
        bu_vn = self.r[n-1] - self.f(self.U[n].dot(self.r[n]))[0]
        bu_sqn = bu_vn.dot(bu_vn)
        bu_totn = (1 / self.ssq[n - 1]) * bu_sqn
        
        pri_rn = self.g(np.squeeze(self.r[n]), self.alph[n])[0]
        pri_Un = self.h(self.U[n], self.lam[n])[0]
        
        return bu_tot + td_tot + pri_r1 + pri_U1 + bu_i + td_i + pri_ri + pri_Ui + bu_totn + pri_rn + pri_Un
    
    def classif_cost_c1(self, label):
        # Format: -label.dot(np.log(softmax(r_n)))
        return -label.dot(np.log(self.stable_softmax(self.r[self.num_layers])))
    
    def classif_cost_c2(self, label):
        # Format: -label.dot(np.log(softmax(Uo.dot(r_n))))
        o = 'o'
        return -label.dot(np.log(self.stable_softmax(self.U[o].dot(self.r[self.num_layers])))) + self.h(self.U[o], self.lam[o])[0]
    
    def classif_cost_None(self, label):
        return 0
    
    def rn_topdown_upd_c1(self, label):
        '''
        redo for recurrent =will all be the same except rn_bar'''
        n = self.num_layers
        o = 'o'
        # Format: k_o  * (label - softmax(r_n))
        c1 = (self.kr[o] ) * (label - self.stable_softmax(self.r[n]))
        return c1

    def rn_topdown_upd_c2(self, label):
        # Format: k_o * (label - softmax(Uo.dot(r_n)))
        n = self.num_layers
        o = 'o'
        c2 = (self.kr[o]) * (label - self.stable_softmax(self.U[o].dot(self.r[n])))
        return c2
    
    def rn_topdown_upd_None(self, label):
        return 0
    
    def r_updates_n_1(self, label):
        '''
        move to static eventually, as well as update_Component assignment
        '''
            
        kr_1 = self.kr[1]
        ssq_1 = self.ssq[1]
        U_1 = self.U[1]
        r_1 = self.r[1]
        
        #U1 operations
        U1_transpose = np.transpose(U_1, self.U1T_dims)
        U1_tdot_r1 = np.tensordot(U_1, r_1, axes=([-1],[0]))
        input_min_U1tdotr1 = self.r[0] - self.f(U1_tdot_r1)[0]
        
        self.r[1] += (kr_1 / ssq_1) * np.tensordot(U1_transpose, input_min_U1tdotr1, axes=(self.U1T_tdot_dims, self.bu_error_tdot_dims)) \
                                                + self.rn_topdown_upd_dict[self.classif_method](label) \
                                                - (kr_1 / ssq_1) * self.g(r_1, self.alph[1])[1]

    def r_updates_n_2(self, label):
        
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
        U1_transpose = np.transpose(U_1, self.U1T_dims)
        U1_tdot_r1 = np.tensordot(U_1, r_1, axes=([-1],[0]))
        input_min_U1tdotr1 = self.r[0] - self.f(U1_tdot_r1)[0]
        
        self.r[1] += (kr_1 / ssq_1) * np.tensordot(U1_transpose, input_min_U1tdotr1, axes=(self.U1T_tdot_dims, self.bu_error_tdot_dims)) \
                                            + (kr_2 * ssq_2) * (self.f(U_2.dot(r_2))[0] - r_1) \
                                            - (kr_1 / ssq_1) * self.g(r_1, self.alph[1])[1]
                                            
        self.r[2] += (kr_2 / ssq_2) * (U_2.T.dot(self.r[1] - self.f(U_2.dot(r_2))[0])) \
                                                + self.rn_topdown_upd_dict[self.classif_method](label) \
                                                - (kr_2 / ssq_2) * self.g(r_2, self.alph[2])[1]
                                            
    def r_updates_n_gt_eq_3(self, label):
        
        n = self.num_layers
        
        ssq_0 = self.ssq[0]

        kr_1 = self.kr[1]
        ssq_1 = self.ssq[1]
        U_1 = self.U[1]
        r_1 = self.r[1]

        U_2 = self.U[2]
        r_2 = self.r[2]
        
        #U1 operations
        U1_transpose = np.transpose(U_1, self.U1T_dims)
        
        '''
        Li style: test
        '''
        # Expanded r1
        # Perform einsum operation
        U1_tdot_r1 = np.einsum('ijk,ik->ij', U_1, r_1)
        ## Unexpanded r1
        #U1_tdot_r1 = np.tensordot(U_1, r_1, axes=([-1],[0]))
        
        input_min_U1tdotr1 = self.r[0] - self.f(U1_tdot_r1)[0]
        
        
        # Layer 1
        self.r[1] += (kr_1 / ssq_0) * np.einsum('ijk,jk->ji', U1_transpose, input_min_U1tdotr1) \
                                            + (kr_1 / ssq_1) * (self.f(U_2.dot(r_2))[0].reshape(r_1.shape) - r_1) \
                                            - (kr_1) * self.g(r_1, self.alph[1])[1]
        
        
        # Layers 2 to n-1                                    
        for i in range(2,n):
            
            ssq_imin1 = self.ssq[i-1]
            
            kr_i = self.kr[i]
            ssq_i = self.ssq[i]
            r_i = self.r[i]
            U_i = self.U[i]
            
            
            if i == 2:
                fUi_dot_ri = self.f(U_i.dot(r_i))[0]
                bu_term = self.r[i-1].reshape(fUi_dot_ri.shape) - fUi_dot_ri
            else:
                bu_term = self.r[i-1] - self.f(U_i.dot(r_i))[0]
                
            
            self.r[i] += (kr_i / ssq_imin1) * (U_i.T.dot(bu_term)) \
                                                + (kr_i / ssq_i ) * (self.f(self.U[i+1].dot(self.r[i+1]))[0] - r_i) \
                                                - (kr_i) * self.g(r_i, self.alph[i])[1]

        # Layer n
        ssq_nmin1 = self.ssq[n-1]
        
        kr_n = self.kr[n]
        U_n = self.U[n]
        r_n = self.r[n]

        self.r[n] += (kr_n / ssq_nmin1) * (U_n.T.dot(self.r[n-1] - self.f(U_n.dot(r_n))[0])) \
                                                + self.rn_topdown_upd_dict[self.classif_method](label) \
                                                - (kr_n) * self.g(r_n, self.alph[n])[1]
        
        '''
        test
        '''    

    def U_updates_n_1(self,label):

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
        self.U[1] += (kU_1 / ssq_1) * np.einsum(self.einsum_arg_U1, input_min_U1tdotr1, r_1) \
                        - (kU_1 / ssq_1) * self.h(U_1, self.lam[1])[1]
                            
    def U_updates_n_gt_eq_2(self,label):
        
        ssq_0 = self.ssq[0]
        
        kU_1 = self.kU[1]
        U_1 = self.U[1]
        r_1 = self.r[1]
        
        '''
        Li style: test
        '''
        
        #U1 operations
        U1_tdot_r1 = np.einsum('ijk,ik->ij', U_1, r_1)
        
        input_min_U1tdotr1 = self.r[0] - self.f(U1_tdot_r1)[0]
        
        einsum_arg_U1 = 'ij,ik->ijk'
        
        # Layer 1
        self.U[1] += (kU_1 / ssq_0) * np.einsum(einsum_arg_U1, input_min_U1tdotr1, r_1) \
                        - kU_1 * self.h(U_1, self.lam[1])[1]
        
        n = self.num_layers
        
        #i>1 - n will all be the same
        for i in range(2,n+1):
            
            ssq_imin1 = self.ssq[i-1]
            
            kU_i = self.kU[i]
            r_i = self.r[i]
            U_i = self.U[i]
            
            if i == 2:
                fUi_dot_ri = self.f(U_i.dot(r_i))[0]
                rimin1_min_Uidotri = self.r[i-1].reshape(fUi_dot_ri.shape) - fUi_dot_ri
            else:
                rimin1_min_Uidotri = self.r[i-1] - self.f(U_i.dot(r_i))[0]
            
            #i
            self.U[i] += (kU_i / ssq_imin1) * np.outer(rimin1_min_Uidotri, r_i) \
                        - kU_i * self.h(U_i, self.lam[i])[1]
                        
    def Uo_update(self, label):
        # Format: Uo += kU_o / ssq_o * (label - softmax(Uo.dot(r_n)))
        '''
        check k/2 vs k/ssqo
        for every top down rn update, U update, V update, (place where a lr is used)
        '''
        o = 'o'
        r_n = self.r[self.num_layers]
        self.U[o] += (self.kU[o]/ lr_denominator) * np.outer((label - self.stable_softmax(self.U[o].dot(r_n))), r_n)

    def classif_guess_c1(self, label):
        guess = np.argmax(self.stable_softmax(self.r[self.num_layers]))
        if guess == np.argmax(label):
            return 1
        else:
            return 0
    
    def classif_guess_c2(self, label):
        guess = np.argmax(self.stable_softmax(self.U['o'].dot(self.r[self.num_layers])))
        if guess == np.argmax(label):
            return 1
        else:
            return 0
        
    def classif_guess_None(self, label):
        return 0