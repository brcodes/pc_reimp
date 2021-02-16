import data as data
import parameters as parameters
import model as model



def main():

    # load MNIST data (0.000166 * 60,000 = ~10 images)
    X_train, y_train = data.get_mnist_data(frac_samp=0.000166,return_test=False)

    # flatten training image array to N_patterns x 784
    X_flat = data.flatten_images(X_train)

    # train model
    self.model.train(X_flat, y_train)








if __name__ == '__main__':
    main()
