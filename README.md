# NYC Taxi Trip Duration Prediction

Predicting how long a taxi ride will take in New York City, using pickup time, location, and trip details. A regression problem aimed at helping dispatch systems assign drivers more efficiently.

## Problem
Given details about a taxi ride (pickup time, pickup/dropoff coordinates, passenger count), predict the total trip duration in seconds.

## Dataset
~729,000 real NYC taxi trips, including pickup/dropoff timestamps and coordinates. [Dataset source](https://www.kaggle.com/c/nyc-taxi-trip-duration)

## Approach
- **EDA & Cleaning:** Verified data integrity, removed physically impossible values (e.g., 22-day "rides", 1,240km "trips") using domain reasoning, not just statistical rules
- **Feature Engineering:** Extracted `pickup_hour`, `pickup_day_of_week`, `pickup_month` from timestamps; calculated real-world `distance_km` between pickup/dropoff using the Haversine formula
- **Modeling:** Compared 4 algorithms — Linear Regression, KNN, Random Forest, Gradient Boosting
- **Tuning:** Used GridSearchCV with cross-validation to tune Random Forest and Gradient Boosting
- **Ensembling:** Tested a Voting Regressor combining all three tuned models

## Results

| Model | RMSE (seconds) | R² Score |
|---|---|---|
| Linear Regression | 422.12 | 0.5894 |
| KNN (K=5) | 381.80 | 0.6641 |
| Random Forest (default) | 373.02 | 0.6793 |
| KNN (K=30, tuned) | 359.10 | 0.7028 |
| Gradient Boosting (default) | 357.92 | 0.7048 |
| Random Forest (tuned) | 351.95 | 0.7145 |
| **Gradient Boosting (tuned)** | **350.02** | **0.7177** |
| Voting Ensemble | 350.01 | 0.7177 |

**Best model:** Tuned Gradient Boosting — explains ~72% of trip duration variation, with an average error of ~5.8 minutes.

## Key Learnings
- Verified target column integrity early to avoid data leakage (never using `dropoff_datetime` as a feature)
- Distinguished between statistically "unusual" values and physically impossible ones before removing outliers
- Confirmed that untuned complex models can underperform simpler tuned ones, hyperparameter tuning matters as much as algorithm choice
- Found that model ensembling doesn't always help, it depends on whether the combined models fail in different ways

## Tools
Python, pandas, numpy, scikit-learn, matplotlib, seaborn
