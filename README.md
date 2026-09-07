# Formula 1 ML fairness analysis

This project looks at whether a regression model performs differently for established Formula 1 teams compared with the rest of the grid.

I used pit stop and race data from 2018 to 2024 and trained a Decision Tree regressor to predict `Lap Time Variation`. The aim was not just to look at one overall accuracy score, but to check whether the error changed depending on the team group.

## The approach

The model uses stint length, air temperature, tyre compound and a team category as inputs. I grouped Mercedes, Ferrari and Red Bull as top-tier teams and compared them with the remaining constructors in the dataset.

I first trained one model on the full training set and measured Mean Absolute Error for each group. I then trained separate models for the two team groups and compared the gap again.

The unified model produced an MAE of about 0.0319 for the top-tier group and 0.0350 for the other teams. With the specialised models, the two group errors were much closer at about 0.0362 and 0.0360.

That reduced the difference between the two group MAEs, although it did not improve the error for both groups. I think that distinction matters because a smaller gap between groups is not automatically the same thing as a better model overall.

## What I used

Python with pandas, NumPy, scikit-learn and matplotlib. The model itself is a `DecisionTreeRegressor`, with a train/test split used for evaluation.

## Repository structure

`src/analysis.py` contains a cleaned version of the analysis pipeline.

`data/README.md` explains the dataset expected by the script. I have not included the raw CSV in this repository yet.

The original notebook was developed in Google Colab as part of my university work. I am cleaning the project up here so the analysis is easier to follow outside the notebook.

## Running it

Install the requirements with:

```bash
pip install -r requirements.txt
```

Place the dataset at `data/f1_pitstops_2018_2024.csv`, then run:

```bash
python src/analysis.py
```

This project was mainly useful for learning how easy it is to hide group-level differences behind one overall model score, and how a mitigation that looks better on one fairness measure can still come with a trade-off elsewhere.
