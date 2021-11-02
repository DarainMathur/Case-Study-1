# Case-Study-1: Interest Rates Prediction
![index](https://user-images.githubusercontent.com/28758956/139824545-972d9ecb-659d-4610-a9d1-3df95a787336.jpg)

1. The dataset had 10,000 rows and 55 columns containing data for 10,000 applicants across the country.


2. Certain features were only for joint applications and merging those into a feature that included both joint and individual applicants was the first challenge.


3. In certain columns, there were Nan values, I dealt with the missing values by assuming that the correct value was 0. For ex - the feature 'days_since_last_deliquency' had multiple missing values, so inserting 0 would mean no deliquency had occured yet.


4. After filtering out useful features from the dataset on the basis of correlation with the interest rates, I created the dataset, I would perform feature scaling on.


5. After scaling all numerical features into acceptable ranges, I turned the categorical features into dummy features that would help with the ML models.


6. Then, the test data was fit into the RandomForestRegressor model, which is a ensemble learning technique.

7. The second model used was XGBoost, a gradient boosting framework.

8. Both models performed much better on the training set, showing that models were overfitted.

9. Due to lack of time, I wasn't able to try any other models.

10. One interesting thing was that the models with default hyperparamter setting gave the best result when compared to GridSearchCV fine-tuned models, the code for fine tuning was taken down as it made no improvement to the model.
