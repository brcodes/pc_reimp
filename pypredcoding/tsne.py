"""
t-SNE and MNIST error histograms
"""

#for 2021.05.20
#Add to plotting.py later

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import pickle
from kbutil import tsne





"""
Pickle in 100x10 MNIST dataset
"""

# load data
tanh_data_in = open('tanh_100x10.pydb','rb')
X_train, y_train, training_img, non_training_img, scrm_training_img, lena_pw, lena_zoom = pickle.load(tanh_data_in)
tanh_data_in.close()


"""
t-SNE
"""

# # run t-sne
# # e.g. 
# Y = tsne.tsne(X, no_dims, initial_dims = 50, perplexity = 30.0)

# X_train_tsne = tsne.tsne(X_train, no_dims=2, initial_dims=50, perplexity=30.0)

# # split X_train.shape[0](N)x2 vector (1000x2 here) in to 2 lists:
# N*x-coords, N*y-coords for scatter plot


"""
Pickle Out t-SNE result
"""


# # pickle the model (contains self.variables for prediction plotting)
# tsne_out = open('tsne_tanh100x10.pydb','wb')
# pickle.dump(X_train_tsne, tsne_out)
# tsne_out.close()



"""
Pickle In t-SNE result
comment out t-SNE operation above, when complete
"""

tsne_in = open('tsne_tanh100x10.pydb','rb')
tsne_results = pickle.load(tsne_in)
tsne_in.close()
    



"""
Pickle in Evaluated Model
"""

#model size
# model_size = '[32.10]'
# model_size = '[32.32]'
model_size = '[128.32]'


#transformation function
transform_type = 'tanh'
# transform_type = 'linear'

#prior type
prior_type = 'gauss'
# prior_type = 'kurt'

#classification method
class_type = 'NC'
# class_type = 'C1'
# class_type = 'C2'

#trained or untrained
trained = 'T'
# trained = 'nt'

#number of epochs if trained (if not, use -)
# num_epochs = '1000e'
num_epochs = '100e'
# num_epochs = '50e'
# num_epochs = '-'

#dataset trained on if trained (if not, use -)
training_dataset = 'tanh100x10'
# training_dataset = 'tanh10x10'
# training_dataset = '-'

#evaluated or not evaluated with so far
evaluated = 'E'
# evaluated = 'ne'

#images evaluated against, if evaluated (if not, use -)
eval_dataset = 'tanh100x10'
# eval_dataset = 'tanh10x10'
# eval_dataset = '-'

#used or not used for prediction so far
# used_for_pred = 'P'
used_for_pred = 'np'

#images predicted, if used for prediction (if not, use -)
#images 1-5 from April/May 2021 exps
# pred_dataset = '5imgs'
pred_dataset = '-'

#extra identifier for any particular or unique qualities of the model object
# extra_tag = 'randUo'
# extra_tag = 'scaled_ppixel'
# extra_tag = 'pipeline_test'
extra_tag = '-'


# load it
evaluation_in = open('pc.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.pydb'.format(model_size,transform_type,prior_type,class_type,\
  trained,num_epochs,training_dataset, evaluated, eval_dataset, used_for_pred, pred_dataset,extra_tag),'rb')
pcmod,E,C,Classif_success_by_img,Acc = pickle.load(evaluation_in)
evaluation_in.close()



"""
Format and Plot t-SNE Result
"""

dim1_list = []
dim2_list = []

for result in tsne_results:
    dim1 = result[0]
    dim2 = result[1]
    dim1_list.append(dim1)
    dim2_list.append(dim2)
    
print("tsne output shape is {}".format(tsne_results.shape))

#split digits to plot with specific colors or labels

lime_green='#3cb44b'
magenta_red='#e6194b'
sky_blue='#4363d8'
light_orange='#f58231'
bright_purple='#911eb4'
turquoise='#46f0f0'
magenta_pink='#f032e6'
neon_yellow_green='#bcf60c'
pink_skin='#fabebe'
dark_lakefoam_green='#008080'
popcorn_yellow='#fffac8'
carmel_brown='#9a6324'
dark_chocolate='#800000'
keylime_pie='#aaffc3'
camo_green='#808000'
cooked_salmon='#ffd8b1'
navy_blue='#000075'
dark_grey='#808080'
white='#ffffff'
black='#000000'
gold_yellow:'#ffe119'



Emax = max(E)
Emin = min(E)

print(Emax)
print(Emin)

alphas = []

for cost in E:
    # normed_cost = ((cost-Emin) / Emax)
    normed_cost = 1
    alphas.append(normed_cost)
    
print(min(alphas))
    

x0s = dim1_list[0:100]
y0s = dim2_list[0:100]
for i in range(0,1):
    plot = plt.scatter(x0s[i], y0s[i], c=lime_green, marker="$0$", alpha=alphas[i])

x1s = dim1_list[100:200]
y1s = dim2_list[100:200]
for i in range(0,1):
    plot = plt.scatter(x1s[i], y1s[i], c=magenta_red, marker="$1$", alpha=alphas[i])

x2s = dim1_list[200:300]
y2s = dim2_list[200:300]
for i in range(0,1):
    plot = plt.scatter(x2s[i], y2s[i], c=sky_blue,  marker="$2$", alpha=alphas[i])

x3s = dim1_list[300:400]
y3s = dim2_list[300:400]
for i in range(0,1):
    plot = plt.scatter(x3s[i], y3s[i], c=light_orange, marker="$3$", alpha=alphas[i])

x4s = dim1_list[400:500]
y4s = dim2_list[400:500]
for i in range(0,1):
    plot = plt.scatter(x4s[i], y4s[i], c=bright_purple,  marker="$4$",alpha=alphas[i])

x5s = dim1_list[500:600]
y5s = dim2_list[500:600]
for i in range(0,1):
    plot = plt.scatter(x5s[i], y5s[i], c='gold',  marker="$5$",alpha=alphas[i])

x6s = dim1_list[600:700]
y6s = dim2_list[600:700]
for i in range(0,1):
    plot = plt.scatter(x6s[i], y6s[i], c=magenta_pink, marker="$6$", alpha=alphas[i])

x7s = dim1_list[700:800]
y7s = dim2_list[700:800]
for i in range(0,1):
    plot = plt.scatter(x7s[i], y7s[i], c=dark_grey,  marker="$7$",alpha=alphas[i])

x8s = dim1_list[800:900]
y8s = dim2_list[800:900]
for i in range(0,1):
    plot = plt.scatter(x8s[i], y8s[i], c=pink_skin,  marker="$8$",alpha=alphas[i])

x9s = dim1_list[900:1000]
y9s = dim2_list[900:1000]
for i in range(0,1):
    plot = plt.scatter(x9s[i], y9s[i], c=dark_lakefoam_green, marker="$9$", alpha=alphas[i])



# greyscale

# x0s = dim1_list[0:100]
# y0s = dim2_list[0:100]
# for i in range(0,100):
#     plot = plt.scatter(x0s[i], y0s[i], c=black, marker="$0$", alpha=alphas[i])

# x1s = dim1_list[100:200]
# y1s = dim2_list[100:200]
# for i in range(0,100):
#     plot = plt.scatter(x1s[i], y1s[i], c=black, marker="$1$", alpha=alphas[i])

# x2s = dim1_list[200:300]
# y2s = dim2_list[200:300]
# for i in range(0,100):
#     plot = plt.scatter(x2s[i], y2s[i], c=black,  marker="$2$", alpha=alphas[i])

# x3s = dim1_list[300:400]
# y3s = dim2_list[300:400]
# for i in range(0,100):
#     plot = plt.scatter(x3s[i], y3s[i], c=black, marker="$3$", alpha=alphas[i])

# x4s = dim1_list[400:500]
# y4s = dim2_list[400:500]
# for i in range(0,100):
#     plot = plt.scatter(x4s[i], y4s[i], c=black,  marker="$4$",alpha=alphas[i])

# x5s = dim1_list[500:600]
# y5s = dim2_list[500:600]
# for i in range(0,100):
#     plot = plt.scatter(x5s[i], y5s[i], c=black,  marker="$5$",alpha=alphas[i])

# x6s = dim1_list[600:700]
# y6s = dim2_list[600:700]
# for i in range(0,100):
#     plot = plt.scatter(x6s[i], y6s[i], c=black, marker="$6$", alpha=alphas[i])

# x7s = dim1_list[700:800]
# y7s = dim2_list[700:800]
# for i in range(0,100):
#     plot = plt.scatter(x7s[i], y7s[i], c=black,  marker="$7$",alpha=alphas[i])

# x8s = dim1_list[800:900]
# y8s = dim2_list[800:900]
# for i in range(0,100):
#     plot = plt.scatter(x8s[i], y8s[i],c=black,  marker="$8$",alpha=alphas[i])

# x9s = dim1_list[900:1000]
# y9s = dim2_list[900:1000]
# for i in range(0,100):
#     plot = plt.scatter(x9s[i], y9s[i], c=black, marker="$9$", alpha=alphas[i])




# plt.title('t-SNE on 100x10 MNIST image set')
plt.xlabel('dim1')
plt.ylabel('dim2')
ax = plt.gca()
# border thickness
ax.spines["top"].set_linewidth(3)
ax.spines["bottom"].set_linewidth(3)
ax.spines["right"].set_linewidth(3)
ax.spines["left"].set_linewidth(3)

plt.savefig('tsne_tanh100x10_128,32_gauss.png',dpi=1200)

plt.show()




"""
Plot Histogram of E vs Image
"""

# print(len(E))

# num_eval_images = len(E)
# num_eval_images = range(1,num_eval_images+1)

# # set title
# fig, ax = plt.subplots(1)
# fig.suptitle("Loss per image" +'\n' + " {} {} NC model predicting 100x10 dataset".format(model_size,prior_type))
# # set color
# plotE = ax.plot(num_eval_images, E, '#000000', marker='.', markersize=3, linewidth=0,  label="E")

# # set E scale
# ax.set_ylim(0, 600)

# #set image scale
# ax.set_xlim(1, 1000)

# ax.set_xticks([0,100,200,300,400,500,600,700,800,900,1000])

# # set axis names
# ax.set_ylabel("E after 100 r updates")
# ax.set_xlabel("Image")

# # border thickness
# ax.spines["top"].set_linewidth(3)
# ax.spines["bottom"].set_linewidth(3)
# ax.spines["right"].set_linewidth(3)
# ax.spines["left"].set_linewidth(3)

# # show plot
# plt.show()


# # plt.hist(E,1000,[1,1000])
# # plt.suptitle("Error per image - NC model predicting 100x10 dataset")
# # axes = plt.gca()
# # axes.set_ylim(0,15)
# # plt.xlabel('Image')
# # plt.ylabel('E after 100 r updates')
# # plt.show()
















    