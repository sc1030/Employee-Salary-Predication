from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Creates additional features for the salary prediction model."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        
        # Feature 1: Experience Squared
        X['Experience_Squared'] = X['Years of Experience'] ** 2
        
        # Feature 2: Age/Experience Ratio
        X['Age_Experience_Ratio'] = X['Age'] / (X['Years of Experience'] + 1e-5)

        # Feature 3: Age Group
        X['Age_Group'] = pd.cut(
            X['Age'], 
            bins=[0, 25, 35, 45, 60, 100],
            labels=['<25', '25-35', '35-45', '45-60', '60+']
        )

        # Feature 4: Career Stage (based on years of experience)
        X['Career_Stage'] = pd.cut(
            X['Years of Experience'],
            bins=[-1, 2, 5, 10, 20, 50],
            labels=['Entry', 'Junior', 'Mid', 'Senior', 'Expert']
        )

        return X
