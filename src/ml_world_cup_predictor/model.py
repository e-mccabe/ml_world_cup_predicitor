import numpy as np

class Node:

    def __init__(self,left=None,right=None,split=None,value=None):

        self.left = left
        self.right = right
        self.split = split
        self.value = value
    
    def is_leaf(self):
        return self.value is not None
    


class DecisionTree:

    def __init__(self,df,y_col,max_depth):

        self.df = df
        self.y_col = y_col
        self.max_depth  = max_depth
        self.root = None

    def fit(self,df,y_col):

        self.root = self._build_tree(df,y_col)
    
    def predict(self,df):

        predictions = [self._traverse_tree(x,self.root) for _,x in df.iterrows()]
        return predictions

    # Helper Functions
    def _build_tree(self,df,y_col,splits = 0):

        if splits >= self.max_depth: 
            outcome = df[y_col].value_counts().idxmax()
            return Node(value = outcome)
        
        optimal_split = self._find_best_split(df,y_col)
        child_left, child_right = self._generate_split(df,optimal_split)

        left_node = self._build_tree(child_left,y_col,splits=splits+1) if len(child_left) > 0 else None
        right_node = self._build_tree(child_right,y_col,splits=splits+1) if len(child_right) > 0 else None
        

        return Node(left = left_node,right = right_node,split = optimal_split)
    
    def _traverse_tree(self,x,node):

        if node.is_leaf():
            return node.value
        
        if x[node.split['feature']] < node.split['threshold']:
            return self._traverse_tree(x,node.left)
        return self._traverse_tree(x,node.right)



    def _generate_split(self,df,optimal_split):

        child_left = df[df[optimal_split['feature']] < optimal_split['threshold']]
        child_right = df[df[optimal_split['feature']] >= optimal_split['threshold']]

        return child_left,child_right

    def _find_best_split(self,df,y_col):

        X = df.loc[:,df.columns != y_col]
        y = df[y_col]
        
        optimal ={
            'feature':None,
            'info_gain': -1,
            'threshold':0
        }
        for column in X.columns:
            
            if column == 'result':
                continue
            
            else:
                best_threshold, best_gain = self._find_best_threshold(X[column],y)
                
                if best_gain > optimal['info_gain']:
                    optimal['feature'] = column
                    optimal['info_gain'] = best_gain
                    optimal['threshold'] = best_threshold

        return optimal

    def _entropy(self,label_counts,total):

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
    
    def _find_best_threshold(self,feature,y):

        # Paring up each feature value with corresponding label and sorting by value
        paired = sorted(zip(feature,y))

        # Full Length of the data
        n = len(y)

        # The counts when the filter is at the furthest left of the data
        labels, counts = np.unique(y,return_counts=True)
        right_count = dict(zip(labels,counts))
        left_count = {label: 0 for label in labels}

        # Initialised before the for loop
        best_gain = -1
        best_threshold = None


        parent_entropy = self._entropy(right_count.values(),n)

        # Iterate over the index of the data
        for i in range(n-1):

            # Get the current data and label
            value_i, label_i = paired[i]

            # Increment the left count of whatever label is at this index as we move the parition to the right of it
            left_count[label_i] += 1
            # Reduce the right count at this index as this label goes to the lift of the partition 
            right_count[label_i] -= 1

            # Identify the next value and label
            value_next,label_next = paired[i+1]

            # If there is no change in class continue because .....
            if label_i == label_next:
                continue
            
            # Calculate the total size of both sides
            n_left = i + 1
            n_right = n - n_left

            # Calculate the entropy for each child node
            left_entropy = self._entropy(left_count.values(),n_left)
            right_entropy = self._entropy(right_count.values(),n_right)

            information_gain = parent_entropy - (n_left/n)*left_entropy - (n_right/n)*right_entropy

            if information_gain > best_gain:
                best_gain = information_gain
                
                best_threshold = (value_i + value_next)/2
            
        return best_threshold,best_gain

    
    