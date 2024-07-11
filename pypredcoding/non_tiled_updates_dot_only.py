
'''
storage
spc fit for whole images
not yet adjusted for tiles
aka not yet adjusted
for r1 sizes of N (tiles), m (nodes)
eg 3392, 32 in Li 212 case
'''

'''
training
'''

for iteration in range(0,self.num_rUsimul_iters):

    ### r loop (splitting r loop, U loop mimic's Li architecture)
    for i in range(1, n):

        # r update
        self.r[i] = self.r[i] + (k_r / self.p.sigma_sq[i]) \
        * self.U[i].T.dot(self.f(self.U[i].dot(self.r[i]))[1].dot(self.r[i-1] - self.f(self.U[i].dot(self.r[i]))[0])) \
        + (k_r / self.p.sigma_sq[i+1]) * (self.f(self.U[i+1].dot(self.r[i+1]))[0] - self.r[i]) \
        - (k_r / 2) * self.g(self.r[i],self.p.alpha[i])[1]

    # final r (Li's "localist") layer update
    self.r[n] = self.r[n] + (k_r / self.p.sigma_sq[n]) \
    * self.U[n].T.dot(self.f(self.U[n].dot(self.r[n]))[1].dot(self.r[n-1] - self.f(self.U[n].dot(self.r[n]))[0])) \
    - (k_r / 2) * self.g(self.r[n],self.p.alpha[n])[1] \

    # Add classification cost component of final representation update (changes based on C1, C2 or NC setting)
    # NOTE: EDIT WITH REAL MATH
    + ((k_o) * (label[:,None] - softmax(self.r[n])))

    ### U loop
    for i in range(1, n):

        # U update
        self.U[i] = self.U[i] + (k_U / self.p.sigma_sq[i]) \
        * (self.f(self.U[i].dot(self.r[i]))[1].dot(self.r[i-1] - self.f(self.U[i].dot(self.r[i]))[0])).dot(self.r[i].T) \
        - (k_U / 2) * self.h(self.U[i],self.p.lam[i])[1]

    # U[n] update (C1, C2) (identical to U[i], except index numbers)
    self.U[n] = self.U[n] + (k_U / self.p.sigma_sq[n]) \
    * (self.f(self.U[n].dot(self.r[n]))[1].dot(self.r[n-1] - self.f(self.U[n].dot(self.r[n]))[0])).dot(self.r[n].T) \
    - (k_U / 2) * self.h(self.U[n],self.p.lam[n])[1]
    
    
'''
separate e, pe calcs for plotting
'''


def rep_cost(self):
    '''
    Uses current r/U states to compute the least squares portion of the error
    (concerned with accurate reconstruction of the input).
    '''
    E = 0
    # LSQ cost
    PE_list = []
    for i in range(0,len(self.r)-1):
        v = (self.r[i] - self.f(self.U[i+1].dot(self.r[i+1]))[0])
        vTdotv = v.T.dot(v)
        E = E + ((1 / self.p.sigma_sq[i+1]) * vTdotv)[0,0]

        # Also calulate prediction error for each layer
        PE = np.sqrt(vTdotv)
        PE_list.append(PE)

    # priors on r[1],...,r[n]; U[1],...,U[n]
    for i in range(1,len(self.r)):
        E = E + (self.h(self.U[i],self.p.lam[i])[0] + self.g(np.squeeze(self.r[i]),self.p.alpha[i])[0])

    return (E, PE_list)

'''
inits
'''

self.num_tiles_per_img = self.p.num_r1_mods
self.num_training_imgs = int(X.shape[0] / self.num_tiles_per_img)

## Initiate r[0] - r[n] layers, U[1] - U[n] layers; tiled case
# Li uses np.zeros for r inits, np.random.rand (unif) for U inits: may have to switch to those

# Input image TILES layer of size (number of tiles == number of r1 modules, area of a single tile); Li case: 225, 256
self.r[0] = np.random.randn(self.p.num_r1_mods,self.sgl_tile_area)

# Non-input layers (1 - n): hidden layers (1 - n-1) plus output layer (Li's "localist" layer n)
# Ought maybe to switch initialized r, U's from strictly Gaussian (randn) to tunable based on specified model.p.r_prior, U_prior

## Hidden layers 1 & 2 first (only hidden layers directly dependent on num tiles)
# Hidden layer 1
# Li case: r1: 225, 32
#          U1: 225, 256, 32
self.r[1] = np.random.randn(self.p.num_r1_mods, self.p.hidden_sizes[0])
self.U[1] = np.random.randn(self.p.num_r1_mods, self.sgl_tile_area, self.p.hidden_sizes[0])

# Hidden layer 2
# Li case: r2: 128,
#          U2: 7200 (225*32), 128
self.r[2] = np.random.randn(self.p.hidden_sizes[1])
self.U[2] = np.random.randn(self.p.num_r1_mods * self.p.hidden_sizes[0], self.p.hidden_sizes[1])

if self.num_hidden_lyrs > 2:
    # Hidden layer 3 or more
    for layer_num in range(3, self.num_nonin_lyrs + 1):
        self.r[layer_num] = np.random.randn(self.p.hidden_sizes[layer_num-1], 1)
        self.U[layer_num] = np.random.randn(len(self.r[layer_num-1]), len(self.r[layer_num]))

# "Localist" layer (relates size of Y (num classes) to final hidden layer)
self.r[self.num_nonin_lyrs] = np.random.randn(self.p.output_size, 1)
self.U[self.num_nonin_lyrs] = np.random.randn(len(self.r[self.num_nonin_lyrs-1]), len(self.r[self.num_nonin_lyrs]))

else:
    print("Model.num_r1_mods attribute needs to be in [1,n=int<<inf]")
    exit()

# If self.p.class_scheme == 'c2', initiate o and Uo layers
# NOTE: May have to change these sizes to account for Li localist layer (is o redundant in that case?)
if self.p.class_scheme == 'c2':
    # Initialize output layer (Li case: 5, 1)
    self.o = np.random.randn(self.p.output_size,1)
    # And final set of weights to the output (Li case: 5, 128)
    self.U_o = np.random.randn(self.p.output_size, self.p.hidden_sizes[-1])