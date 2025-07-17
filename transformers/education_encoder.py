from sklearn.base import BaseEstimator, TransformerMixin

class EducationEncoder(BaseEstimator, TransformerMixin):
    """Encodes education level into an ordinal numeric value."""

    def __init__(self):
        self.education_mapping = {
            'High School': 0,
            "Bachelor's": 1,
            "Master's": 2,
            'PhD': 3,
            'Diploma': 0.5
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['Education_Level_Encoded'] = X['Education Level'].map(self.education_mapping).fillna(0)
        return X
