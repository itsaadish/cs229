import numpy as np
import util


def main(train_path, valid_path, save_path,plot_path):
    """Problem: Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)
    # *** START CODE HERE ***
    # Train a logistic regression classifier
    # Plot decision boundary on top of validation set set
    # Use np.savetxt to save predictions on eval set to save_path
    clf = LogisticRegression()
    clf.fit(x_train, y_train)
    util.plot(x_valid, y_valid, clf.theta, plot_path)
    np.savetxt(save_path, clf.predict(x_valid))
    # *** END CODE HERE ***


class LogisticRegression:
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=1000000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        self.theta = np.zeros((x.shape[1],1))
        y = y.reshape(y.size,1)
        while True:
            y_predict = self.predict(x)
            y_predict = y_predict.reshape(y_predict.size,1)
            grads = np.dot(x.T,(y_predict - y))/x.shape[0]
            hess = np.dot(x.T, (y_predict * (1 - y_predict) * x)) / x.shape[0]
            print(hess.shape)
            diff = np.dot(np.linalg.inv(hess.T),grads)
            self.theta = self.theta - diff
            if np.abs(diff).sum() < self.eps:
                return self
        # *** END CODE HERE ***

    def predict(self, x):
        """Return predicted probabilities given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***  (n,d) (d,1)
        z =  np.dot(x,self.theta)
        return 1 / (1 + np.exp(-z))
        # *** END CODE HERE ***

if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='logreg_pred_3.txt',plot_path='logreg_3.jpg')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='logreg_pred_4.txt',plot_path='logreg_4.jpg')
