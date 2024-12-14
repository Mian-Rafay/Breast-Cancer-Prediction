import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.utils import shuffle

class svm_():
    def __init__(self,learning_rate,epoch,C_value,X,Y):

        #initialize the variables
        self.input = X
        self.target = Y
        self.learning_rate =learning_rate
        self.epoch = epoch
        self.C = C_value

        # initialize the weight matrix based on number of features 
        # bias and weights are merged together as one matrix
     
        self.weights = np.zeros(X.shape[1])

    def pre_process(self, x, y):

        # using StandardScaler to normalize the input
        scalar = StandardScaler().fit(x)
        X_ = scalar.transform(x)
        
        scalar = StandardScaler().fit(y)
        Y = scalar.transform(y)
        Y_ = (np.column_stack(Y)).T

        return X_,Y_
    
    # stochastic gradient decent
    def compute_gradient(self,X,Y):
        # organize the array as vector
        X_ = np.array([X])

        # hinge loss
        hinge_distance = 1 - (Y* np.dot(X_,self.weights))

        total_distance = np.zeros(len(self.weights))
        if max(0, hinge_distance[0]) == 0:
            total_distance += self.weights
        else:
            total_distance += self.weights - (self.C * Y[0] * X_[0])

        return total_distance

    def compute_loss(self,X,Y):
        # calculate hinge loss
        loss=0
        # hinge loss implementation- start
        # Part 1
        
        for i in range(X.shape[0]):
            loss += np.maximum(0, 1 - Y[i] * np.dot(X[i], self.weights))

        return (self.C * loss) + (0.5 * np.dot(self.weights, self.weights))

        # hinge loss implementatin - end
   
    
    def stochastic_gradient_descent(self,X, Y, X_test, Y_test, threshold=0.00001):
        epoch_stop = float('inf')
        stop_found = False
        stop_loss = float('inf')
        min_loss = float('inf')
        prev_loss = float('inf')
        best_weights = self.weights.copy()  # Save best weights
        train_losses = []  # To track training losses
        test_losses = []  # To track validation losses
    
        # execute the stochastic gradient descent function for defined epochs
        for epoch in range(self.epoch):

            # shuffle to prevent repeating update cycles
            features, output = shuffle(X, Y)

            for i, feature in enumerate(features):
                gradient = self.compute_gradient(feature, output[i])
                self.weights = self.weights - (self.learning_rate * gradient)

            # print epoch if it is equal to thousand - to minimize number of prints
            if epoch%1000 ==0:
                loss = self.compute_loss(features, output)
                print("Epoch is: {} and Loss is : {}".format(epoch, loss))

            # check for convergence -start
            # Compute training loss
            train_loss = self.compute_loss(features, output)

            # Compute validation loss
            test_loss = self.compute_loss(X_test, Y_test)

            if epoch % (self.epoch // 10) == 0:
                train_losses.append(train_loss)
                test_losses.append(test_loss)
            
            if  stop_found == False and abs(prev_loss - train_loss) < threshold:
                epoch_stop = epoch // 10
                stop_found = True
                stop_loss = prev_loss
                # break # uncomment to utilize early stopping, not all epochs will be printed then
                            
            prev_loss = train_loss
            
            if train_loss < min_loss:
                min_loss = train_loss
                best_weights = self.weights.copy()
                
                
        print(f"Training completed early with {epoch_stop*10} iterations with a loss of {stop_loss}")
        self.weights = best_weights  # update weights
        
        #check for convergence - end
        
        return train_losses, test_losses, epoch_stop
        

    def mini_batch_gradient_descent(self, X, Y, X_test, Y_test, batch_size = 32):

        # mini batch gradient decent implementation - start
        train_losses = []
        test_losses = []
        
        for epoch in range(self.epoch):
            
            # shuffle to prevent repeating update cycles
            features, output = shuffle(X, Y)

            for i in range(0, features.shape[0], batch_size):
                # process the dataset in mini-batches
                batch_X = features[i:i+batch_size]
                batch_Y = output[i:i+batch_size]

                # compute the gradient for the mini-batch
                gradients = np.zeros(len(self.weights))
                for j in range(batch_X.shape[0]):
                    gradient = self.compute_gradient(batch_X[j], batch_Y[j])
                    gradients += gradient

                # update the weights using the average gradient of the mini-batch
                gradients /= batch_X.shape[0]
                self.weights -= self.learning_rate * gradients
                
            # print the loss every 1000 epochs
            if epoch % 1000 == 0:
                loss = self.compute_loss(features, output)
                
                print(f"Epoch: {epoch}, Loss: {loss}")
                
            # compute training loss
            train_loss = self.compute_loss(features, output)

            # compute validation loss
            test_loss = self.compute_loss(X_test, Y_test)

            if epoch % (self.epoch // 10) == 0:
                train_losses.append(train_loss)
                test_losses.append(test_loss)

        print("Training ended...")
        print("weights are: {}".format(self.weights))
        
        # mini batch gradient decent implementation - end
        
        return train_losses, test_losses

    def sampling_strategy(self, X, Y):
        # find the sample with the smallest SVM loss
        min_loss = float('inf')
        best_sample = None
        best_label = None
        best_index = None
    
        for i in range(X.shape[0]):
            sample = X[i]
            label = Y[i]
        
            # compute the hinge loss for the sample
            loss = np.maximum(0, 1 - label * np.dot(sample, self.weights))

            if loss < min_loss:
                min_loss = loss
                best_sample = sample
                best_label = label
                best_index = i
    
        return best_sample, best_label, best_index

    def predict(self,X_test,Y_test):

        # compute predictions on test set
        predicted_values = [np.sign(np.dot(X_test[i], self.weights)) for i in range(X_test.shape[0])]
        
        # compute accuracy
        accuracy= accuracy_score(Y_test, predicted_values)
        print("Accuracy on test dataset: {}".format(accuracy))

        # compute precision - start
        accuracy= precision_score(Y_test, predicted_values)
        print("Precision on test dataset: {}".format(accuracy))
        # compute precision - end

        # compute recall - start
        accuracy= recall_score(Y_test, predicted_values)
        print("Recall on test dataset: {}".format(accuracy))
        # compute recall - end
        
        return accuracy

def early_stopping(X_train, X_test, y_train, y_test):
    # model parameters
    C = 0.001 
    learning_rate = 0.001 
    epoch = 5000

    # instantiate the SVM model
    my_svm = svm_(learning_rate=learning_rate, epoch=epoch, C_value=C, X=X_train, Y=y_train)

    # pre-process data
    X_train_norm, y_train_norm = my_svm.pre_process(X_train, y_train)
    X_test_norm, y_test_norm = my_svm.pre_process(X_test, y_test)

    # train the model with early stopping
    train_losses, test_losses, epoch_stop = my_svm.stochastic_gradient_descent(X_train_norm, y_train_norm, X_test_norm, y_test_norm)
    
    plt.plot(range(len(train_losses)), train_losses, label='Training Loss')
    plt.axvline(x=epoch_stop, color='red', linestyle='--', label="early stop")
    plt.xlabel('Epochs 1/10')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
        
    plt.plot(range(len(test_losses)), test_losses, label='Validation Loss')
    plt.axvline(x=epoch_stop, color='red', linestyle='--', label="early stop")
    plt.xlabel('Epochs 1/10')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    print("Testing model accuracy...")
    my_svm.predict(X_test_norm,y_test)
    
    return my_svm


def mini_batch(X_train, X_test, y_train, y_test):
    # model parameters
    C = 0.0001 
    learning_rate = 0.001 
    epoch = 5000
  
    # intantiate the support vector machine class above
    my_svm = svm_(learning_rate=learning_rate,epoch=epoch,C_value=C,X=X_train,Y=y_train)
    my_svm2 = svm_(learning_rate=learning_rate,epoch=epoch,C_value=C,X=X_train,Y=y_train)
    
    # pre-preocess data
    X_train_norm, y_train_norm = my_svm.pre_process(X_train, y_train)
    X_test_norm, y_test_norm = my_svm.pre_process(X_test, y_test)
    
    # train models
    # part 1
    train_losses_sgd, test_losses_sgd, _ = my_svm.stochastic_gradient_descent(X_train_norm, y_train_norm, X_test_norm, y_test_norm)
    # part 2
    train_losses_mbgd, test_losses_mbgd = my_svm2.mini_batch_gradient_descent(X_train_norm, y_train_norm, X_test_norm, y_test_norm)
    
    print("Testing model accuracy SGD...")
    my_svm.predict(X_test_norm,y_test)
    print("Testing model accuracy MBGD...")
    my_svm2.predict(X_test_norm,y_test)
    
    plt.plot(range(len(train_losses_sgd)), train_losses_sgd, label='Training Loss SGD')
    plt.plot(range(len(train_losses_mbgd)), train_losses_mbgd, label='Training Loss MBGD')
    plt.xlabel('Epochs 1/10')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
        
    plt.plot(range(len(test_losses_sgd)), test_losses_sgd, label='Validation Loss SGD')
    plt.plot(range(len(test_losses_mbgd)), test_losses_mbgd, label='Validation Loss MBGD')
    plt.xlabel('Epochs 1/10')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    
    return my_svm

def smallest_number_of_samples(X_train, y_train, X_valid, y_valid, threshold = 0.001):
    # model parameters
    C = 0.001 
    learning_rate = 0.001 
    epoch = 5000
    n = 5
    
    # shuffle the data
    X_train, y_train = shuffle(X_train, y_train)
    
    # split into small sample for active learning
    x_sample, x_test = X_train[:n], X_train[n:] 
    y_sample, y_test = y_train[:n], y_train[n:] 

    # reshape y_sample and y_test to be 2D arrays
    y_sample = y_sample.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    # instantiate the support vector machine class
    my_svm = svm_(learning_rate=learning_rate, epoch=epoch, C_value=C, X=X_train, Y=y_train)
    
    # Pre-process the data
    X_train_norm, y_train_norm = my_svm.pre_process(x_sample, y_sample)
    X_test_norm, y_test_norm = my_svm.pre_process(x_test, y_test)

    # Train the model with early stopping
    train_losses, test_losses, _ = my_svm.stochastic_gradient_descent(X_train_norm, y_train_norm, X_test_norm, y_test_norm)

    samples = n
    prev_acc = float('inf')
    while True:
        # find the most informative sample (smallest SVM loss)
        best_sample, best_label, best_index = my_svm.sampling_strategy(x_test, y_test)

        # add the best sample to the training set
        x_sample = np.vstack([x_sample, best_sample])
        y_sample = np.append(y_sample, best_label)

        # remove the selected sample from the test set
        x_test = np.delete(x_test, best_index, axis=0)
        y_test = np.delete(y_test, best_index, axis=0)

        # reshape the updated y_sample and y_test to be 2D arrays
        y_sample = y_sample.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)
        
        # pre-process the updated data
        X_train_norm, y_train_norm = my_svm.pre_process(x_sample, y_sample)
        X_test_norm, y_test_norm = my_svm.pre_process(x_test, y_test)

        # train the model again
        train_losses, test_losses, _ = my_svm.stochastic_gradient_descent(X_train_norm, y_train_norm, X_test_norm, y_test_norm)
        
        plt.plot(range(len(train_losses)), train_losses, label='Training Loss')
        plt.xlabel('Epochs 1/10')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
        
        plt.plot(range(len(test_losses)), test_losses, label='Validation Loss')
        plt.xlabel('Epochs 1/10')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
        
        accuracy = my_svm.predict(X_test_norm,y_test)
        # check for stopping criteria based on accuracy
        if abs(prev_acc - accuracy) < threshold:
            print(f"Stopping active learning after {samples} samples: Accuracy within threshold")
            break
        
        prev_acc = accuracy

        samples += 1
    
    print("Testing model accuracy...")
    X_valid_norm, _ = my_svm.pre_process(X_valid, y_valid)
    my_svm.predict(X_valid_norm, y_valid)

    return my_svm



# load datapoints in a pandas dataframe
print("Loading dataset...")
data = pd.read_csv('data.csv')

# drop first and last column 
data.drop(data.columns[[-1, 0]], axis=1, inplace=True)

# segregate inputs and targets

# inputs
X = data.iloc[:, 1:]

# add column for bias
X.insert(loc=len(X.columns),column="bias", value=1)
X_features = X.to_numpy()

# converting categorical variables to integers 
# this is same as using one hot encoding from sklearn
# benign = -1, melignant = 1
category_dict = {'B': -1.0,'M': 1.0}
# transpose to column vector
Y = np.array([(data.loc[:, 'diagnosis']).to_numpy()]).T
Y_target = np.vectorize(category_dict.get)(Y)

# split data into train and test set using sklearn feature set
print("splitting dataset into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X_features, Y_target, test_size=0.2, random_state=42)

my_svm = early_stopping(X_train, X_test, y_train, y_test)

my_svm = mini_batch(X_train, X_test, y_train, y_test)

my_svm = smallest_number_of_samples(X_features, Y_target, X_test, y_test)