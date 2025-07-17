from sklearn.base import BaseEstimator, TransformerMixin

class JobGrouper(BaseEstimator, TransformerMixin):
    """Simplifies or groups job titles into high-level job categories."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        title = X['Job Title'].str.lower()

        def group_title(text):
            if 'engineer' in text:
                return 'Engineer'
            elif 'developer' in text:
                return 'Developer'
            elif 'analyst' in text:
                return 'Analyst'
            elif 'manager' in text:
                return 'Manager'
            elif 'consultant' in text:
                return 'Consultant'
            elif 'intern' in text:
                return 'Intern'
            elif 'admin' in text or 'support' in text:
                return 'Admin'
            else:
                return 'Other'

        X['Job_Group'] = title.apply(group_title)
        return X
