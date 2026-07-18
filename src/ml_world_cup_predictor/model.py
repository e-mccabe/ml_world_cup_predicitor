import numpy as np

class Node:
    """Class to store each node of the decision tree"""
    def __init__(self,left=None,right=None,split=None,value=None):

        self.left = left
        self.right = right
        self.split = split
        self.value = value
    
    def is_leaf(self):
        return self.value is not None
    


class DecisionTree:
    """
    > Create a Decision Tree fit to training data using .fit()
    > Predict results of a dataset using .predict() 
    
    Initialised using:
        - max_depth             : maximum depth the tree can go to
        - min_samples_split    : the minimum samples in a node that can be investigated to split
    """

    def __init__(self,max_depth = 20,min_samples_split=2,class_weight = None):
        
        # Parameter for max number of steps into the decision tree
        self.max_depth  = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.class_weight = class_weight

    def fit(self,df,target_column):

        sample_length = len(df[target_column])
        class_count = df[target_column].nunique()

        if self.class_weight:
            self.class_weights = (sample_length/(class_count*df[target_column].value_counts()))

        self.root = self._build_tree(df,target_column)
    
    def predict(self,df):

        predictions = [self._traverse_tree(x,self.root) for _,x in df.iterrows()]
        return predictions

    ### ---- Helper Functions ----

    def _is_finished(self,depth,n_samples,n_class_labels):

        condition = (depth >= self.max_depth or n_class_labels == 1 or n_samples < self.min_samples_split)

        return condition

    def _build_tree(self,df,target_column,depth = 0):
        """Build the Decision Tree with recursive node building"""
        

        # Define the numbers of samples and class labels in the current node
        n_samples = len(df)
        n_class_labels = (df[target_column].nunique())

        # Check if any of the end conditions are met
        if self._is_finished(depth,n_samples,n_class_labels):
            outcome = self._leaf_value(df,target_column)
            return Node(value = outcome)

        # Generate Split Dictionary
        optimal_split = self._find_best_split(df,target_column)
        # Split current dataframe
        child_left, child_right = self._generate_split(df,optimal_split)

        if len(child_left) == 0 or len(child_right) == 0:
            outcome = df[target_column].value_counts().idxmax()
            return Node(value = outcome)

        # Recursive node building
        left_node = self._build_tree(child_left,target_column,depth=depth+1)
        right_node = self._build_tree(child_right,target_column,depth=depth+1)
        

        return Node(left = left_node,right = right_node,split = optimal_split)
    
    def _leaf_value(self,df,target_column):

        counts = df[target_column].value_counts()

        if self.class_weights is None:
            return counts.idxmax()
        return (counts * self.class_weights).idxmax()

    def _traverse_tree(self,x,node):
        """Walk down Decision Tree to provide result"""
        if node.is_leaf():
            return node.value
        
        if x[node.split['feature']] < node.split['threshold']:
            return self._traverse_tree(x,node.left)
        return self._traverse_tree(x,node.right)

    def _generate_split(self,df,optimal_split):

        child_left = df[df[optimal_split['feature']] < optimal_split['threshold']]
        child_right = df[df[optimal_split['feature']] >= optimal_split['threshold']]

        return child_left,child_right

    def _find_best_split(self,df,target_column):
        """Iterate over the dataframes, identifying the feature with the largest information gain using parent and child entropy """
        X = df.loc[:,df.columns != target_column]
        y = df[target_column]
        
        # Initialising optimal split dictionary
        optimal ={
            'feature':None,
            'info_gain': -1,
            'threshold':0
        }

        for column in X.columns:
            best_threshold, best_gain = self._find_best_threshold(X[column],y)
            # Update the optimal split if the information gain is greater than previous iterations
            if best_gain > optimal['info_gain']:
                optimal['feature'] = column
                optimal['info_gain'] = best_gain
                optimal['threshold'] = best_threshold

        return optimal

    def _entropy(self,label_counts,total):
        """Calculating the entropy of a node from its label counts"""
        # Not possible to calculate entropy on non-existent data
        if total == 0:
            return 0
        
        # Initialise Entropy Value
        entropy_value = 0

        # Iterate the class count
        for label in label_counts:
            if label == 0:
                continue
            
            probability = label/total
            entropy_value -= probability * np.log2(probability)
        
        return entropy_value
    

    def _find_best_threshold(self,feature,labels):
        """Iterate over feature values to find the threshold with the highest information gain"""

        # Pairing up each feature value with corresponding label and sorting by value
        paired = sorted(zip(feature,labels))

        sample_length = len(labels)

        # The counts when the filter is at the start of the data sequence
        unique_labels, counts = np.unique(labels,return_counts=True)
        right_count = dict(zip(unique_labels,counts))
        left_count = {label: 0 for label in unique_labels}

        best_gain = -1
        best_threshold = None

        parent_entropy = self._entropy(right_count.values(),sample_length)

        for index in range(sample_length-1):

            value_i, label_i = paired[index]

            # Increment the left count of whatever label is at this index as we move the partition to the right of it
            left_count[label_i] += 1
            # Reduce the right count at this index as this label goes to the left of the partition 
            right_count[label_i] -= 1

            value_next,label_next = paired[index+1]

            # If there is no change in class, continue — a threshold between two identical labels can't improve purity
            if label_i == label_next:
                continue
            
            sample_length_left = index + 1
            sample_length_right = sample_length - sample_length_left

            # Calculate the entropy for each child node
            left_entropy = self._entropy(left_count.values(),sample_length_left)
            right_entropy = self._entropy(right_count.values(),sample_length_right)

            information_gain = parent_entropy - (sample_length_left/sample_length)*left_entropy - (sample_length_right/sample_length)*right_entropy

            if information_gain > best_gain:
                best_gain = information_gain
                
                best_threshold = (value_i + value_next)/2
            
        return best_threshold,best_gain

    
    