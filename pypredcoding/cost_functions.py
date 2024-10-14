import numpy as np

'''
to-do

clean up train
clean up evaluate
implement cost_function classes in train, evaluate
hard-code 10 epoch diagnostic/prints
load checkpoint
save checkpoint
home/log
home/models/checkpoints
home/results/diagnostics/
            /plots/

squeeze into r prior
l activation functions element wise

note in config that e1l Li will only work with tiled, flat case. it is only intended to replicate her dissertation data.
may not even support layer number changes. only hyperparameter changes and specific layer parameter changes (shapes between r1U2 must match up)

make sure it's understood that for our rn topdown update components, we're using a ko
set ko to kn to be Li model
but it's tunable now for the future

iron out pcc
iron out spcc

'''

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
        
        self.rn_topdown_upd_dict = {'c1': self.rn_topdown_upd_c1,
                                    'c2': self.rn_topdown_upd_c2,
                                    None: self.rn_topdown_upd_None}
    
    def create_subcomponent_dict(self, prefix):
        return {
            "flat_hidden_lyrs": getattr(self, f"{prefix}_fhl"),
            "expand_first_lyr_Li": getattr(self, f"{prefix}_e1l_Li"),
            "expand_first_lyr": getattr(self, f"{prefix}_e1l")
        }
        
    def initialize_components(self, architecture_key):
        self.U1mat_mult_r1vecormat = self.U1r1_dict[architecture_key]
        self.U1Tmat_mult_Idiffmat = self.U1T_Idiff_dict[architecture_key]
        self.U2mat_mult_r2vec = self.U2r2_dict[architecture_key]
        self.U2Tmat_mult_L1diffvecormat = self.U2T_L1diff_dict[architecture_key]
        self.U_gteq3_mat_mult_r_gteq3_vec = self.U3r3_dict[architecture_key]
        self.U3Tmat_mult_L2diffvec = self.U3T_L2diff_dict[architecture_key]
        # U-only
        self.Idiffmat_mult_r1vecormat = self.Idiff_r1_dict[architecture_key]
        self.L1diffvecormat_mult_r2vec = self.L1diff_r2_dict[architecture_key]
        self.L_gteq2_diffvec_mult_r_gteq3_vec = self.L2diff_r3_dict[architecture_key]
            
    '''
    cost function subcomponents
    
    used in r or U updates
    dependent on architecture parameter
    '''
    
    def U1r1_fhl(self, U1, r1):
        return np.tensordot(U1, r1, axes=([-1], [0]))    
        
    def U1r1_e1l_Li(self, U1, r1):
        return np.matmul(U1, r1[:, :, None]).squeeze()
    
    def U1r1_e1l(self, U1, r1):
        # np.einsum(self.U1_einsum_arg, U1, r1) 
        # ijk,ik->ij in 3d U1 case, 2d r1 case
        # ijklm,ijm-> ijkl in 5d U1 case, 3d r1 case
        
        return np.einsum(self.U1_einsum_arg, U1, r1)    
    
    def U1T_Idiff_fhl(self, U1T, Idiff):
        # axes=(self.U1T_tdot_dims, self.bu_error_tdot_dims)
        return np.tensordot(U1T, Idiff, axes=(self.U1T_tdot_dims, self.Idiff_tdot_dims))
    
    def U1T_Idiff_e1l_Li(self, U1T, Idiff):
        return np.matmul(U1T, Idiff[:, :, None]).squeeze()
    
    def U1T_Idiff_e1l(self, U1T, Idiff):
        # np.einsum(self.U1T_einsum_arg...
        # ijk,jk->ji in 3d U1 case, 2d r1 case
        # ijklm,jklm->jki in 5d U1 case, 3d r1 case
        return np.einsum(self.U1T_einsum_arg, U1T, Idiff)
    
    def U2r2_fhl(self, U2, r2):
        return np.dot(U2, r2)
    
    def U2r2_e1l_Li(self, U2, r2):
        return np.dot(U2, r2).reshape(self.r[1].shape)
    
    def U2r2_e1l(self, U2, r2):
        return np.tensordot(U2, r2, axes=([-1], [0]))
    
    def U2T_L1diff_fhl(self, U2T, L1diff):
        return np.dot(U2T, L1diff)
    
    def U2T_L1diff_e1l_Li(self, U2T, L1diff):
        return np.dot(U2T, L1diff.flatten())
    
    def U2T_L1diff_e1l(self, U2T, L1diff):
        # example: 16,32,128 3d U1 case, 2d r1 case
        # or: 4,4,32,128 5d U1 case, 3d r1 case
        # U2T = 128,16,32     or:   128,4,4,32
        # L1diff = 16,32       or:   4,4,32
        # arg ijk,jk->i         or:     ijkl,jkl->i
        # In either case set up a self.U2T_args with an identical protocol to self.U1T_args (range, swap last for first.)
        # Then einsums
        return np.einsum(self.U2T_einsum_arg, U2T, L1diff)
    
    '''
    these are all dots
    can shortcut later but kept for clarity
    '''
    
    def U3r3_fhl(self, U3, r3):
        return np.dot(U3, r3)
    
    def U3r3_e1l_Li(self, U3, r3):
        return np.dot(U3, r3)
    
    def U3r3_e1l(self, U3, r3):
        return np.dot(U3, r3)
    
    def U3T_L2diff_fhl(self, U3T, L2diff):
        return np.dot(U3T, L2diff)
    
    def U3T_L2diff_e1l_Li(self, U3T, L2diff):
        return np.dot(U3T, L2diff)
    
    def U3T_L2diff_e1l(self, U3T, L2diff):
        return np.dot(U3T, L2diff)
    
    '''
    cost function subcomponents
    
    used in U updates only
    dependent on architecture parameter
    '''
    
    def Idiff_r1_fhl(self, Idiff, r1):
        # See e1l
        # Looks like einsum_arg_U1 works here
        return np.einsum(self.Idiff_einsum_arg, Idiff, r1)
    
    def Idiff_r1_e1l_Li(self, Idiff, r1):
        return np.matmul(Idiff[:, :, None], r1[:, None, :])
    
    def Idiff_r1_e1l(self, Idiff, r1):
        # Here it'll be
        # 16,864         16,32
        # ij,ik-> ijk 3d U1 case, 2d r1 case
        # if I is 4d (unflat, tiled, 5d U1 case, 3d r1 case)
        # 4,4,24,36      4,4,32         U1: 4,4,24,36,32
        # then ijkl,ijm->ijklm
        return np.einsum(self.Idiff_einsum_arg, Idiff, r1)
    
    def L1diff_r2_fhl(self, L1diff, r2):
        return np.outer(L1diff, r2)
    
    def L1diff_r2_e1l_Li(self, L1diff, r2):
        return np.outer(L1diff.flatten(), r2)
    
    def L1diff_r2_e1l(self, L1diff, r2):
        # 16,32     128     ij,k->ijk
        # 4,4,32    128     ijk,l->ijkl
        return np.einsum(self.L1diff_einsum_arg, L1diff, r2)
    
    def L2diff_r3_fhl(self, L2diff, r3):
        return np.outer(L2diff, r3)
    
    def L2diff_r3_e1l_Li(self, L2diff, r3):
        return np.outer(L2diff, r3)
    
    def L2diff_r3_e1l(self, L2diff, r3):
        return np.outer(L2diff, r3)
    
    
    '''
    actual cost functions
    '''    
        
    def rep_cost_n_1(self):
        r1 = self.r[1]
        U1 = self.U[1]
        bu_tdot_dims = self.Idiff_tdot_dims

        # Bottom-up component of the representation error
        U1r1 = self.U1mat_mult_r1vecormat(U1, r1)
        bu_vec = self.r[0] - U1r1
        bu_square = np.tensordot(bu_vec, bu_vec, axes=(bu_tdot_dims, bu_tdot_dims))
        bu_total = (1 / self.ssq[1]) * bu_square
        
        # Priors on that layer
        prior_r = self.g(np.squeeze(r1), self.alph[1])[0]
        prior_U = self.h(U1, self.lam[1])[0]
        
        # Return total
        return bu_total + prior_r + prior_U
    
    def rep_cost_n_2(self):
        r1 = self.r[1]
        r2 = self.r[2]
        U1 = self.U[1]
        U2 = self.U[2]
        bu_tdot_dims = self.Idiff_tdot_dims
        td_tdot_dims = self.L1diff_tdot_dims

        # Layer 1
        # Bottom-up component of the representation error
        U1r1 = self.U1mat_mult_r1vecormat(U1, r1)
        bu_vec = self.r[0] - U1r1
        bu_square = np.tensordot(bu_vec, bu_vec, axes=(bu_tdot_dims, bu_tdot_dims))
        bu_total = (1 / self.ssq[1]) * bu_square
        
        # Top-down component of the representation error
        U2r2 = self.U2mat_mult_r2vec(U2, r2)
        td_vec = r1 - U2r2
        td_square = np.tensordot(td_vec, td_vec, axes=(td_tdot_dims, td_tdot_dims))
        td_total = (1 / self.ssq[2]) * td_square
        
        # Priors on that layer
        prior_r = self.g(np.squeeze(r1), self.alph[1])[0]
        prior_U = self.h(U1, self.lam[1])[0]
        
        # Layer 2
        # Bottom-up
        bu_total2 = td_total
        
        # Priors on that layer
        prior_r2 = self.g(np.squeeze(r2), self.alph[2])[0]
        prior_U2 = self.h(U2, self.lam[2])[0]
        
        # Return total
        return bu_total + td_total + prior_r + prior_U + bu_total2 + prior_r2 + prior_U2
    
    def rep_cost_n_gt_eq_3(self):
        
        n = self.num_layers
        r1 = self.r[1]
        r2 = self.r[2]
        r3 = self.r[3]
        U1 = self.U[1]
        U2 = self.U[2]
        U3 = self.U[3]
        bu_tdot_dims = self.Idiff_tdot_dims
        td_tdot_dims = self.L1diff_tdot_dims

        # Layer 1
        # Bottom-up component of the representation error
        U1r1 = self.U1mat_mult_r1vecormat(U1, r1)
        bu_vec = self.r[0] - U1r1
        bu_square = np.tensordot(bu_vec, bu_vec, axes=(bu_tdot_dims, bu_tdot_dims))
        bu_total = (1 / self.ssq[0]) * bu_square
        
        # Top-down component of the representation error
        U2r2 = self.U2mat_mult_r2vec(U2, r2)
        td_vec = r1 - U2r2
        td_square = np.tensordot(td_vec, td_vec, axes=(td_tdot_dims, td_tdot_dims))
        td_total = (1 / self.ssq[1]) * td_square
        
        # Priors on that layer
        prior_r = self.g(np.squeeze(r1), self.alph[1])[0]
        prior_U = self.h(U1, self.lam[1])[0]
        
        # Layer 2
        # Bottom-up
        bu_total2 = td_total
        
        # Top-down component
        U3r3 = self.U_gteq3_mat_mult_r_gteq3_vec(U3, r3)
        td_vec2 = r2 - U3r3
        td_square2 = np.dot(td_vec2, td_vec2)
        td_total2 = (1 / self.ssq[2]) * td_square2
        
        # Priors on that layer
        prior_r2 = self.g(np.squeeze(r2), self.alph[2])[0]
        prior_U2 = self.h(U2, self.lam[2])[0]
        
        # Layer 3 - n-1 (skips if 3-layer model)
        bu_totali_all_i = 0
        td_totali_all_i = 0
        prior_ri_all_i = 0
        prior_Ui_all_i = 0
        
        for i in range(3,n):
            
            # If is cheaper than Ui mat mul ri
            if i == 3:
                # Bottom-up
                bu_totali = td_total2
                
                # Priors on that layer
                prior_ri = self.g(np.squeeze(r3), self.alph[i])[0]
                prior_Ui = self.h(self.U3, self.lam[i])[0]
            else:
                # Bottom-up
                Uiri = self.U_gteq3_mat_mult_r_gteq3_vec(self.U[i], self.r[i])
                bu_veci = self.r[i - 1] - Uiri
                bu_squarei = np.dot(bu_veci, bu_veci)
                bu_totali = (1 / self.ssq[i - 1]) * bu_squarei
                
                # Priors on that layer
                prior_ri = self.g(np.squeeze(self.r[i]), self.alph[i])[0]
                prior_Ui = self.h(self.U[i], self.lam[i])[0]
            
            # Top-down component
            Ui1ri1 = self.U_gteq3_mat_mult_r_gteq3_vec(self.U[i + 1], self.r[i + 1])
            td_veci = self.r[i] - Ui1ri1
            td_squarei = np.dot(td_veci, td_veci)
            td_totali = (1 / self.ssq[i]) * td_squarei
            
            bu_totali_all_i += bu_totali
            td_totali_all_i += td_totali
            prior_ri_all_i += prior_ri
            prior_Ui_all_i += prior_Ui
            
        # Layer n
        if n == 3:
            # Bottom-up
            bu_totaln = td_total2
            
            # Priors on that layer
            prior_rn = self.g(np.squeeze(r3), self.alph[3])[0]
            prior_Un = self.h(U3, self.lam[3])[0]

        else:
            # Bottom-up
            bu_totaln = td_totali
            
            # Priors on that layer
            prior_rn = self.g(np.squeeze(self.r[n]), self.alph[n])[0]
            prior_Un = self.h(self.U[n], self.lam[n])[0]
            
        # Return total
        return bu_total + td_total + prior_r + prior_U + bu_total2 + td_total2 + prior_r2 + prior_U2 + \
                bu_totali_all_i + td_totali_all_i + prior_ri_all_i + prior_Ui_all_i + bu_totaln + prior_rn + prior_Un
    
    def classif_cost_c1(self, label):
        # Format: -label.dot(np.log(softmax(r_n)))
        return -label.dot(np.log(self.softmax_func(vector=self.r[self.num_layers])))
    
    def classif_cost_c2(self, label):
        # Format: -label.dot(np.log(softmax(Uo.dot(r_n))))
        o = 'o'
        return -label.dot(np.log(self.softmax_func(vector=self.U[o].dot(self.r[self.num_layers])))) + self.h(self.U[o], self.lam[o])[0]
    
    def classif_cost_None(self, label):
        return 0
    
    def rn_topdown_upd_c1(self, label):
        '''
        redo for recurrent =will all be the same except rn_bar'''
        return label - self.softmax_func(vector=self.r[self.num_layers])

    def rn_topdown_upd_c2(self, label):
        '''
        rn_bar
        '''
        return label - self.softmax_func(vector=self.U['o'].dot(self.r[self.num_layers]))
    
    def rn_topdown_upd_None(self, label):
        return 0
    
    def r_updates_n_1(self, label):
        
        r1 = self.r[1]
        U1 = self.U[1]
        kr1 = self.kr[1]
        
        # Layer 1
        U1T = np.transpose(U1, self.U1T_dims)
        U1r1 = self.U1mat_mult_r1vecormat(U1, r1)
        Idiff = self.r[0] - U1r1
        
        self.r[1] += (kr1 / self.ssq[0]) * self.U1Tmat_mult_Idiffmat(U1T, Idiff) \
                    + (self.kr['o'] / self.lr_rn_td_denominator) * self.rn_topdown_upd_dict[self.classif_method](label) \
                    - (kr1 / self.lr_prior_denominator) * self.g(r1, self.alph[1])[1]
        
    
    def r_updates_n_2(self, label):
        
        r1 = self.r[1]
        r2 = self.r[2]
        U1 = self.U[1]
        U2 = self.U[2]
        kr1 = self.kr[1]
        kr2 = self.kr[2]
        ssq1 = self.ssq[1]
        lr_prior_denominator = self.lr_prior_denominator
        
        # Layer 1
        U1T = np.transpose(U1, self.U1T_dims)
        U1r1 = self.U1mat_mult_r1vecormat(U1, r1)
        U2r2 = self.U2mat_mult_r2vec(U2, r2)
        Idiff = self.r[0] - U1r1
        
        self.r[1] += (kr1 / self.ssq[0]) * self.U1Tmat_mult_Idiffmat(U1T, Idiff) \
                    + (kr1 / ssq1) * (U2r2 - r1) \
                    - (kr1 / lr_prior_denominator) * self.g(r1, self.alph[1])[1]
                    
        # Layer 2
        U2T = np.transpose(U2, self.U2T_dims) 
        L1diff = r1 - U2r2
        
        self.r[2] += (kr2 / ssq1) * self.U2Tmat_mult_L1diffvecormat(U2T, L1diff) \
                    + (self.kr['o'] / self.lr_rn_td_denominator) * self.rn_topdown_upd_dict[self.classif_method](label) \
                    - (kr2 / lr_prior_denominator) * self.g(r2, self.alph[2])[1]
    
    def r_updates_n_gt_eq_3(self, label):
        
        n = self.num_layers
        r1 = self.r[1]
        r2 = self.r[2]
        U1 = self.U[1]
        U2 = self.U[2]
        kr1 = self.kr[1]
        kr2 = self.kr[2]
        ssq1 = self.ssq[1]
        lr_prior_denominator = self.lr_prior_denominator
        
        # Layer 1
        U1T = np.transpose(U1, self.U1T_dims)
        U1r1 = self.U1mat_mult_r1vecormat(U1, r1)
        U2r2 = self.U2mat_mult_r2vec(U2, r2)
        Idiff = self.r[0] - U1r1
        
        self.r[1] += (kr1 / self.ssq[0]) * self.U1Tmat_mult_Idiffmat(U1T, Idiff) \
                    + (kr1 / ssq1) * (U2r2 - r1) \
                    - (kr1 / lr_prior_denominator) * self.g(r1, self.alph[1])[1]
                    
        # Layer 2 - n-1
        for i in range(2, n):
            
            # Layer i == 2
            if i == 2:
                r3 = self.r[3]
                U3 = self.U[3]
                
                U2T = np.transpose(U2, self.U2T_dims) 
                U3r3 = self.U_gteq3_mat_mult_r_gteq3_vec(U3, r3)
                L1diff = r1 - U2r2
                
                self.r[2] += (kr2 / ssq1) * self.U2Tmat_mult_L1diffvecormat(U2T, L1diff) \
                            + (kr2 / self.ssq[2]) * (U3r3 - r2) \
                            - (kr2 / lr_prior_denominator) * self.g(r2, self.alph[2])[1]
            
            # Layer i > 2
            else:
                ri = self.r[i]
                ri1 = self.r[i + 1]
                Ui = self.U[i]
                Ui1 = self.U[i + 1]
                kri = self.kr[i]
                
                UiT = Ui.T
                Uiri = self.U_gteq3_mat_mult_r_gteq3_vec(Ui, ri)
                Ui1ri1 = self.U_gteq3_mat_mult_r_gteq3_vec(Ui1, ri1)
                Limin1diff = self.r[i - 1] - Uiri
                
                self.r[i] += (kri / self.ssq[i - 1]) * self.U3Tmat_mult_L2diffvec(UiT, Limin1diff) \
                            + (kri / self.ssq[i]) * (Ui1ri1 - ri) \
                            - (kri / lr_prior_denominator) * self.g(ri, self.alph[i])[1]
                    
        # Layer n
        rn = self.r[n]
        Un = self.U[n]
        krn = self.kr[n]
        
        UnT = Un.T
        Unrn = self.U_gteq3_mat_mult_r_gteq3_vec(Un, rn)
        Lnmin1diff = self.r[n - 1] - Unrn
        
        self.r[n] += (krn / self.ssq[n - 1]) * self.U3Tmat_mult_L2diffvec(UnT, Lnmin1diff) \
                    + (self.kr['o'] / self.lr_rn_td_denominator) * self.rn_topdown_upd_dict[self.classif_method](label) \
                    - (krn / lr_prior_denominator) * self.g(rn, self.alph[n])[1]
    
    def U_updates_n_1(self,label):
        
        r1 = self.r[1]
        U1 = self.U[1]
        kU1 = self.kU[1]
        
        # Layer 1
        U1r1 = self.U1mat_mult_r1vecormat(U1, r1)
        Idiff = self.r[0] - U1r1
        
        self.U[1] += (kU1 / self.ssq[0]) * self.Idiffmat_mult_r1vecormat(Idiff, r1) \
                    - (kU1 / self.lr_prior_denominator) * self.h(U1, self.lam[1])[1]
    
    def U_updates_n_gt_eq_2(self,label):
        
        n = self.num_layers
        r1 = self.r[1]
        U1 = self.U[1]
        kU1 = self.kU[1]
        lr_prior_denominator = self.lr_prior_denominator
        
        # Layer 1
        U1r1 = self.U1mat_mult_r1vecormat(U1, r1)
        Idiff = self.r[0] - U1r1
        
        self.U[1] += (kU1 / self.ssq[0]) * self.Idiffmat_mult_r1vecormat(Idiff, r1) \
                    - (kU1 / lr_prior_denominator) * self.h(U1, self.lam[1])[1]
                    
        # Layer 2 - n
        for i in range(2,n+1):

            # Layer i == 2
            if i == 2:
                r2 = self.r[2]
                U2 = self.U[2]
                kU2 = self.kU[2]
                
                U2r2 = self.U2mat_mult_r2vec(self.U[2], self.r[2])
                L1diff = r1 - U2r2
                
                self.U[2] += (kU2 / self.ssq[1]) * self.L1diffvecormat_mult_r2vec(L1diff, r2) \
                            + (kU2 / lr_prior_denominator) * self.h(U2, self.lam[2])[1]
            
            # i > 2               
            else:
                ri = self.r[i]
                Ui = self.U[i]
                kUi = self.kU[i]
                
                Uiri = self.U_gteq3_mat_mult_r_gteq3_vec(Ui, ri)
                Limin1diff = self.r[i - 1] - Uiri
                
                #i
                self.U[i] += (kUi / self.ssq[i - 1]) * self.L_gteq2_diffvec_mult_r_gteq3_vec(Limin1diff, ri) \
                            - (kUi / lr_prior_denominator) * self.h(Ui, self.lam[i])[1]
    
    def Uo_update(self, label):
        # No "Li" denominator option here, because she never ran a C2 model.
        o = 'o'
        rn = self.r[self.num_layers]
        self.U[o] += (self.kU[o] / 2) * np.outer((label - self.softmax_func(vector=self.U[o].dot(rn))), rn)

    def classif_guess_c1(self, label):
        guess = np.argmax(self.softmax_func(vector=self.r[self.num_layers]))
        if guess == np.argmax(label):
            return 1
        else:
            return 0
    
    def classif_guess_c2(self, label):
        guess = np.argmax(self.softmax_func(vector=self.U['o'].dot(self.r[self.num_layers])))
        if guess == np.argmax(label):
            return 1
        else:
            return 0
        
    def classif_guess_None(self, label):
        return 0