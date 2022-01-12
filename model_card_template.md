# Model Card

## Model Details
Trained by: **Ederson Andre de Souza**

Model date: 2022-01-11

The model is a *Sklearn Random Forest Classifier*. 

The parameters are stored in the `./config.yaml` file.

## Intended Use
This model should be used to predict the salary an american based on its demographic data.

Your use is intended for academic and study purposes.

## Training Data
The data source: [Census Bureau data obtained from Udacity](https://github.com/udacity/nd0821-c3-starter-code/blob/master/starter/data/census.csv).

The dataset was already cleaned and it is available in DVC Remote S3 bucket.

The data was splited into train and test sets at a fraction of 80/20.

Categorical features are:
- `workclass`
- `education`
- `marital_status`
- `occupation`
- `relationship`
- `race`
- `sex`
- `native_country`

The remaining columns were set as continuous features.

## Evaluation Data
The data for evaluation is used on the 20% of the data.
No cross validation was used in this experiment.

## Metrics
- `Precision:` 0.782
- `Recall:` 0.566 
- `F_beta:` 0.657

## Ethical Considerations
Census data is publicly available, contained sensitive information about the sex, education, gender and race os people. The output model should not be considered for non-didactic uses, as other important model information was not considered and there was no bias test.

## Caveats and Recommendations
This project was not focused on the search for the best performance of the model, therefore, no deep research was done on hyperparameters or feature engineering. In case the work continues, the suggestion would be to seek more up-to-date and widely available data.