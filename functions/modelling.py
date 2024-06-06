import os 
import logging
import datetime
import sys
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor

def logger(name, log_folder):
    """
    Function to initialize a logger.
    
    Args:
        name (str): Name of the logger.
        log_folder (str): Folder path where log files will be saved.
    
    Returns:
        logger: Logger object.
    """
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    f_name = os.path.join(log_folder, name + "_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".log")
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    handler = logging.FileHandler(filename=f_name, mode='w')
    handler.setFormatter(formatter)
    
    screen_handler = logging.StreamHandler(stream=sys.stdout)
    screen_handler.setFormatter(formatter)
    
    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)
    
    # Clear existing handlers
    if log.hasHandlers():
        log.handlers.clear()
    
    log.addHandler(handler)
    log.addHandler(screen_handler)
    
    return log

def select_features(df, target_column, method='RFE', n_features=None):
    """
    Function to select features from a dataset using either Recursive Feature Elimination (RFE)
    or feature importance from a Random Forest model.

    Args:
        df (DataFrame): Input DataFrame containing the dataset.
        target_column (str): Name of the target column.
        method (str): Method to select features. Options are 'RFE' or 'feature_importance'.
        n_features (int): Number of features to select. If None, all features are considered.

    Returns:
        selected_features (list): List of selected feature names.
        
    Raises:
        ValueError: If an invalid method is provided.
    """
    # Separate the data into features (X) and target (y)
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    if method == 'RFE':
        # Recursive Feature Elimination (RFE)
        estimator = RandomForestRegressor()  # You can use any estimator here
        selector = RFE(estimator, n_features_to_select=n_features)
        selector = selector.fit(X, y)
        selected_features = X.columns[selector.support_].tolist()
        
    elif method == 'feature_importance':
        # Feature Importance
        model = RandomForestRegressor()  # You can use any model here
        model.fit(X, y)
        feature_importances = pd.Series(model.feature_importances_, index=X.columns)
        selected_features = feature_importances.nlargest(n_features).index.tolist()
        
    else:
        raise ValueError("Invalid method. Please choose 'RFE' or 'feature_importance'.")
    
    return selected_features


def train_model(df, target_column, feature_columns, model, kf=3,  
                grid_search=True, param_grid=None, random_search=False,
                n_iter=10):
    """
    Function to train a machine learning model.

    Args:
        df (DataFrame): Input DataFrame containing the dataset.
        target_column (str): Name of the target column.
        feature_columns (list): List of feature columns.
        model: Machine learning model object.
        param_grid (dict): Dictionary containing hyperparameter grid for grid search.
        grid_search (bool): Whether to perform grid search (default True).

    Returns:
        model: Trained machine learning model.
        best_params: Best hyperparameters found during grid search.
        avg_score: Average score of the model evaluation.
    """
    # Initialize logging
    log_folder = os.path.join('C:\\Users\\micha\\OneDrive\\Dokumenty\\Studia\\Development')
    log = logger('RR_Log', log_folder)

    # Step 1: Separating the data into X and y
    X = df[feature_columns]
    y = df[target_column]
    
    log.info("Starting Grid Search for hyperparameter tuning..")
    # Step 2: Grid Searching the Parameter Grid to find the best match
    if grid_search:
        grid_search_cv = GridSearchCV(model, param_grid, cv=kf, scoring='neg_mean_squared_error')
        grid_search_cv.fit(X, y)
        best_params = grid_search_cv.best_params_
        model.set_params(**best_params)
    elif random_search:
        random_search_cv = RandomizedSearchCV(model, param_grid, n_iter=n_iter, cv=kf, scoring='neg_mean_squared_error', random_state=42)
        random_search_cv.fit(X, y)
        best_params = random_search_cv.best_params_
        model.set_params(**best_params)
    else:
        # Use default parameters
        best_params = model.get_params()  # Assuming the default parameters

    log.info("Finished Grid Search for hyperparameter tuning..")

    # Step 3: Model Fitting
    log.info("Starting Model Fitting..")
    model.fit(X, y)
    log.info("Finishing Model Fitting..")
    
    # Step 4: Model Evaluation
    log.info("Evaluating the model..")
    scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')
    avg_score = np.mean(scores)
    
    log.info("Finished Model Evaluation..")
    
    return model, best_params, avg_score

def predict_model(model, df_test, target_column, feature_columns, charts=False):
    """
    Function to make predictions using a trained model and evaluate its performance.

    Args:
        model: Trained machine learning model.
        df_test (DataFrame): Test DataFrame containing the dataset.
        target_column (str): Name of the target column.
        feature_columns (list): List of feature columns.
        charts (bool): Whether to generate and display charts (default False).

    Returns:
        mse: Mean Squared Error of the model's predictions.
        y_pred: Predicted values.
    """
    # Initialize logging
    log_folder = os.path.join('C:\\Users\\micha\\OneDrive\\Dokumenty\\Studia\\Development')
    log_file = os.path.join(log_folder, 'Prediction_Log_' + datetime.now().strftime("%Y%m%d_%H%M%S") + ".log")
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    
    # Step 1: Split test data into features (X) and target (y)
    logging.info("Splitting test data into features and target..")
    X_test = df_test[feature_columns]
    y_test = df_test[target_column]
    
    # Step 2: Make predictions
    logging.info("Making predictions..")
    y_pred = model.predict(X_test)
    
    # Step 3: Evaluate model performance
    logging.info("Evaluating model performance..")
    mse = mean_squared_error(y_test, y_pred)
    logging.info(f"Mean Squared Error: {mse}")
    
    # Step 4: Visualization
    if charts:
        logging.info("Generating charts..")
        # Importance score charts
        feature_importances = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
        sorted_idx = feature_importances.importances_mean.argsort()
        plt.barh(X_test.columns[sorted_idx], feature_importances.importances_mean[sorted_idx])
        plt.xlabel("Permutation Importance")
        plt.title("Feature Importance")
        plt.show()
        
        # Prediction charts
        plt.scatter(y_test, y_pred)
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title("Actual vs. Predicted Values")
        plt.show()
    
    logging.info("Prediction and evaluation completed.")
    return mse, y_pred
