# Seoul Bike Demand Classification

INTDSCI Final Project, De La Salle Lipa, Section C2B

The Good The Bad The Lodor: Artvincent Abesamis, Ian Lodor, Kurt Tengonciang

We predict whether an hour in Seoul will have High (505+ rentals) or Low bike-sharing demand from the weather, time of day, and calendar, using the Seoul Bike Sharing Demand dataset from the UCI Machine Learning Repository.

## Results (test set, 1,752 hours)

| Model | Accuracy |
| --- | --- |
| Baseline (most frequent class) | 49.5% |
| Logistic Regression | 86.9% |
| Random Forest | 95.6% |

## How to run

1. Install Python 3 and the libraries: `pip install pandas numpy matplotlib seaborn scikit-learn notebook`
2. Keep `SeoulBikeData.csv` in the same folder as the notebook.
3. Open `seoul_bike_ml_project.ipynb` and run Kernel > Restart & Run All.

## Files

- `seoul_bike_ml_project.ipynb`: the full ML process, from loading to evaluation
- `SeoulBikeData.csv`: the dataset
- `figures/`: every chart from the notebook (Figures 1 to 7)
- `docs/`: the final report
