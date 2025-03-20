import numpy as np

# Results of Test:
# Identified bug in generate_traintest as of 3-20-25.
# Bug: getTrainTestIndices should not shuffle indices after merging X_train and X_test into X
# (respectively for Y). Shuffling indices creates unnecessary complication.
                          
class GenerateTrainTest:
    def __init__(self, total_samples=100, test_size=0.2, n_features=20, n_labels=2, shuffle=True, random_state=42):
        self.total_samples = total_samples
        
        self.test_samples = int(test_size*total_samples)
        self.train_samples = self.total_samples - self.test_samples

        self.n_features = n_features
        self.n_labels = n_labels

        self.shuffle = shuffle
        self.random_state = random_state

        
    def generateTrainTest(self, label_distr, train_label_features, test_label_features, train_random_size=0.2, test_random_size=0, test_label_distr=None):
        '''
        Generate X_train, Y_train, X_test, Y_test.
        
        For 'nice' dataset, make label distribution in training and testing sets equal.
        Label distribution allows for over/underrepresentation in training/testing sets.

        Label features determine how much each label depends on each feature.
        Make label_features different for train/testing sets to simulate new data being very
        different from original data. (Test for overfitting)

        Add noise to training/testing data. Default: some noise in training set, no noise in testing set.

        Param:
        label_distr (dict): distribution of labels in dataset. Default: same distribution in train/test.
        test_label_distr (dict, optional): distribution of labels in testing dataset.

        train_label_features (dict): distribution of features contibuting to testing label 
        test_label_features (dict): distribution of features contributing to testing label.

        train_random_size (float): percent of noise in training set
        test_random_size (float): percent of noise in testing set

        Returns:
        self.X_train (np.array): training data (self.train_samples x self.n_features)
        self.X_test (np.array): testing data (self.test_samples x self.n_features)        
        self.Y_train (np.array): training labels (self.train_samples x self.n_labels)
        self.Y_test (np.array): testing labels (self.test_samples x self.n_labels)        

        '''
        if test_label_distr == None: test_label_distr = label_distr

        self.X_train, self.Y_train = self.generateXY(self.train_samples, label_distr, train_label_features, train_random_size)
        self.X_test, self.Y_test = self.generateXY(self.test_samples, test_label_distr,test_label_features,test_random_size)

        assert self.X_train.shape == (self.train_samples, self.n_features)
        assert self.Y_train.shape == (self.train_samples, self.n_labels)
        
        assert self.X_test.shape == (self.test_samples, self.n_features)
        assert self.Y_test.shape == (self.test_samples, self.n_labels)
        
        return self.X_train, self.X_test, self.Y_train, self.Y_test

    def getTrainTestIndices(self):
        '''
        After obtaining X_train and X_test, Y_train and Y_test matrices, merge the train/test
        sets into one (no shuffle), while keeping track of the training set/testing set indices.
        
        
        Returns:
        self.X (np.array): merged X_train and X_test. self.total_samples x self.n_features
        self.Y (np.array): merged Y_train and Y_test self.total_samples x self.n_labels
        self.train_indices (np.array): 1D array of training indices
        self.test_indices (np.array): 1D array of testing indices
        '''
        if self.X_train is None: raise Exception('Call generateTrainTest before getting indices!')

        assert not np.any(np.isnan(self.X_train))
        assert not np.any(np.isnan(self.X_test))
        assert not np.any(np.isnan(self.Y_train))
        assert not np.any(np.isnan(self.Y_test))

        # Merge X_train, X_test while keeping track of indices.
        X_combined = np.vstack((self.X_train, self.X_test))
        Y_combined = np.vstack((self.Y_train, self.Y_test))

        # Create a permutation of indices for shuffling

        # ALTERED: REMOVED SHUFFLING
        # all_indices = np.random.permutation(self.total_samples)
        all_indices = list(range(self.total_samples))

        self.train_indices = all_indices[:self.train_samples]
        self.test_indices = all_indices[self.train_samples:]

        assert not np.any(np.isnan(X_combined))
        assert not np.any(np.isnan(Y_combined))

        # Shuffle indices of X_combined
        self.X = X_combined[all_indices]
        self.Y = Y_combined[all_indices]

        assert self.X.shape == (self.total_samples, self.n_features)
        assert self.Y.shape == (self.total_samples, self.n_labels)

        return self.X, self.Y, self.train_indices, self.test_indices

    def generateXY(self, n_samples, label_distr, label_features, random_size=0.1):
        '''
        Generate data matrix X and label matrix Y.

        Params:
        n_samples (int): number of samples in X and Y.
        label_distr (dict): distribution of labels in dataset {label_idx: percentage}
        label_features (dict): distribution of features contributing to label {label_idx: {feature_idx: weight}}
        random_size (float): percentage of rows with noisy (random) data

        Returns:
        X (np.array): data matrix, n_samples x self.n_features
        Y (np.array): label matrix, n_samples x self.n_labels
        '''
        # label distribution must have same number of labels as n_labels
        assert len(label_distr) == self.n_labels

        # Normalize label distribution 
        label_distr = {k : v/sum(label_distr.values()) for k,v in label_distr.items()}

        # Compute number of samples per label, according to label_distr
        samples_per_label = {k: int(v*n_samples) for k,v in label_distr.items()}

        # Ensure all rows have label
        num_labeled_rows = sum(samples_per_label.values())
        if num_labeled_rows < n_samples:
            most_common_label = max(label_distr, key=label_distr.get)
            samples_per_label[most_common_label] += n_samples - num_labeled_rows

        num_labeled_rows = sum(samples_per_label.values())
        assert num_labeled_rows == n_samples

        X = np.zeros((n_samples, self.n_features))
        Y = np.zeros((n_samples, self.n_labels))

        start_idx = 0

        # Iterate through each label
        for label_idx, n_label_samples in samples_per_label.items():
            
            # Generate samples for this label
            end_idx = start_idx + n_label_samples

            # Iterate through each feature
            for feature_idx in range(self.n_features):
                # check if feature has weight for this label
                weight = label_features.get(label_idx,{}).get(feature_idx, 0)

                if weight > 0:
                    # Fill X from start_idx to end_idx with randomly generated feature values
                    X[start_idx:end_idx, feature_idx] = np.random.normal(
                        loc=weight*10, # mean proportional to weight
                        scale=1.0, # stdev 1
                        size=n_label_samples # fill n_label_sample number of rows
                    )

                # one-hot encode labels
                Y[start_idx:end_idx,label_idx] = 1

            start_idx = end_idx
        
        # Randomly select rows to add random data
        n_random_rows = int(random_size * n_samples)
        random_row_indices = np.random.choice(n_samples, size=n_random_rows, replace=False)
        for row_idx in random_row_indices:
            X[row_idx] = np.random.uniform(low=0, high=10, size=self.n_features)

        # Ensure all values nonnegative
        X = np.clip(X, a_min=0, a_max=None)
       
        # Normalize datapoints in X
        X = X / np.linalg.norm(X, axis=1, keepdims=True)

        # Shuffle
        if self.shuffle:
            rng = np.random.default_rng(seed=self.random_state)
            permutation = rng.permutation(X.shape[0])

            X = X[permutation]
            Y = Y[permutation]
            
        return X, Y
