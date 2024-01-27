import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
 
def fit_preprocess(data_path):
    # Load from file
    data = pd.read_csv(data_path)
    
    # Obtain preprocess parameters
    Variable_names = ['XMEAS(1)', 'XMEAS(2)', 'XMEAS(3)', 'XMEAS(4)', 'XMEAS(5)', 'XMEAS(6)',
             'XMEAS(7)', 'XMEAS(8)', 'XMEAS(9)', 'XMEAS(10)', 'XMEAS(11)', 'XMEAS(12)',
             'XMEAS(13)', 'XMEAS(14)', 'XMEAS(15)', 'XMEAS(16)', 'XMEAS(17)', 'XMEAS(18)',
             'XMEAS(19)', 'XMEAS(20)', 'XMEAS(21)', 'XMEAS(22)', 'XMEAS(23)', 'XMEAS(24)',
             'XMEAS(25)', 'XMEAS(26)', 'XMEAS(27)', 'XMEAS(28)', 'XMEAS(29)', 'XMEAS(30)',
             'XMEAS(31)', 'XMEAS(32)', 'XMEAS(33)', 'XMEAS(34)', 'XMEAS(35)', 'XMEAS(36)',
             'XMEAS(37)', 'XMEAS(38)', 'XMEAS(39)', 'XMEAS(40)', 'XMEAS(41)',
             'XMV(1)', 'XMV(2)', 'XMV(3)', 'XMV(4)', 'XMV(5)', 'XMV(6)', 'XMV(7)',
             'XMV(8)', 'XMV(9)', 'XMV(11)', 'XMV(10)']
    
    preprocess_params = {}
    for var in Variable_names:
        X = data[var].to_numpy()
        Xmean = X.mean()
        Xstd = X.std()
        preprocess_params[var] = {'mean': Xmean, 'std': Xstd} 
    return preprocess_params
 
 
def load_and_preprocess(data_path, preprocess_params):
    # Load from file
    data = pd.read_csv(data_path)
    
    Variable_names = ['XMEAS(1)', 'XMEAS(2)', 'XMEAS(3)', 'XMEAS(4)', 'XMEAS(5)', 'XMEAS(6)',
             'XMEAS(7)', 'XMEAS(8)', 'XMEAS(9)', 'XMEAS(10)', 'XMEAS(11)', 'XMEAS(12)',
             'XMEAS(13)', 'XMEAS(14)', 'XMEAS(15)', 'XMEAS(16)', 'XMEAS(17)', 'XMEAS(18)',
             'XMEAS(19)', 'XMEAS(20)', 'XMEAS(21)', 'XMEAS(22)', 'XMEAS(23)', 'XMEAS(24)',
             'XMEAS(25)', 'XMEAS(26)', 'XMEAS(27)', 'XMEAS(28)', 'XMEAS(29)', 'XMEAS(30)',
             'XMEAS(31)', 'XMEAS(32)', 'XMEAS(33)', 'XMEAS(34)', 'XMEAS(35)', 'XMEAS(36)',
             'XMEAS(37)', 'XMEAS(38)', 'XMEAS(39)', 'XMEAS(40)', 'XMEAS(41)',
             'XMV(1)', 'XMV(2)', 'XMV(3)', 'XMV(4)', 'XMV(5)', 'XMV(6)', 'XMV(7)',
             'XMV(8)', 'XMV(9)', 'XMV(11)', 'XMV(10)']
    
    X = data.to_numpy()
    # Preprocess data for each variable and concatenate
    for var in Variable_names:
        x_var = data[[var]].to_numpy()
        mean = preprocess_params[var]['mean']
        std = preprocess_params[var]['std']
        x_var = (x_var - mean) / std
        X = np.concatenate((X, x_var), axis=1)#axis 1 means each column gets joined together
 
    y = data['label'].to_numpy()
    return X, y
 
def fit_model(data):
    pca_model = PCA(n_components=20)
    pca_model.fit(data)
    transformed_data = pca_model.transform(data)
    data_reconstructed = pca_model.inverse_transform(transformed_data)
    error = np.mean(np.square(data - data_reconstructed), axis=1)
    pca_model.threshold_value = np.mean(error) + np.std(error)
    return pca_model
 
def predict(new_data, pca_model):
    transformed_new_data = pca_model.transform(new_data)
    new_data_reconstructed = pca_model.inverse_transform(transformed_new_data)
    reconstruction_error_new = np.mean(np.square(new_data - new_data_reconstructed), axis=1)
    predictions = (reconstruction_error_new > pca_model.threshold_value).astype(int)
    return predictions
 
 
def main():
    """
    Main function to test the code when running this script standalone. Not
    mandatory for the deliverable but useful for the grader to check the code
    workflow and for the student to check it works as expected.
    """
    train_path = 'https://raw.githubusercontent.com/iraola/ML4CE-AD/main/coursework/data/data_train.csv'
    test_path = 'https://raw.githubusercontent.com/iraola/ML4CE-AD/main/coursework/data/data_test.csv'
 
    # Set up preprocessing
    preprocess_params = fit_preprocess(train_path)
 
    # Load and preprocess data
    X_train, y_train = load_and_preprocess(train_path, preprocess_params)
    X_test, y_test = load_and_preprocess(test_path, preprocess_params)
 
    # Get the detection model
    threshold = fit_model(X_train)
 
    # Check performance on data
    # (just for us to see it works properly, the grader might use other data)
    y_pred_train = predict(X_train, threshold)
    print('Training accuracy: ', accuracy_score(y_train, y_pred_train))
    y_pred_test = predict(X_test, threshold)
    print('Test accuracy: ', accuracy_score(y_test, y_pred_test))
 
 
if __name__ == '__main__':
    main()