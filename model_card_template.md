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

The data was splited into train and test sets at a fraction of 70/30.

Features used for training are:
- `age`: int
- `workclass`: str
- `education`: str
- `marital-status`: str
- `occupation`: str
- `relationship`: str
- `race`: str
- `sex`: str
- `hours-per-week`: str
- `native-country`: str


## Evaluation Data
The data for evaluation is used on the 20% of the data.
No cross validation was used in this experiment.

## Metrics
- `Precision:` 0.714
- `Recall:` 0.496 
- `F_beta:` 0.585

## Ethical Considerations
Census data is publicly available, contained sensitive information about the sex, education, gender and race os people. The output model should not be considered for non-didactic uses, as other important model information was not considered and there was no bias test.

## Caveats and Recommendations
This model was trained with census data from 1994. Though it provides a useful experiment, no conclusion should be drawn from applying this model to current data, more than 25 years later. It is better to see this model as a way to obtain insights from the home economy landscape of the 90s.

If updated data could be obtained from a similar census, then it would be trivial to implement this model for use nowadays. However, special care should be taken on the categorical features imbalance in order to produce a more balanced data. This is special true with features such as a Native Country, where some classes present less than 10 observations and thus produces very subpar predictions and the consequient bias.