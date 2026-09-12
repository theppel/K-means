# type: ignore
# flake8: noqa
#
#
#
#
#
#
#
#
#
#| echo: false
#| label: setup

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from kmodes.kprototypes import KPrototypes
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform

mh = pd.read_csv("Mental_Health_and_Lifestyle_Research.csv")

mh.fillna("None", inplace=True)

encoder = OneHotEncoder(sparse_output=False)
encoder.set_output(transform="pandas")
encoder_cols = ["Mental_Health_Status", "Substance_Use", "Physical_Health_Condition"]

# fit and transform relevant columns
encoded_data = encoder.fit_transform(mh[encoder_cols])
mh[encoded_data.columns] = encoded_data

# drop original columns
mh.drop(columns=encoder_cols, inplace=True)
# also drop None columns
mh.drop(columns=["Mental_Health_Status_None", "Substance_Use_None", "Physical_Health_Condition_None"], inplace=True)
# drop person ID
mh.drop(columns=["Person_ID"], inplace=True)

# not necessary to have 2 columns for Gender and Has_Close_Friends - just apply lambda function
mh["Gender"] = mh["Gender"].apply(lambda x: 1 if x == "Male" else 0)
mh["Has_Close_Friends"] = mh["Has_Close_Friends"].apply(lambda x: 1 if x else 0)

SI_map = {
    "Low": 0,
    "Moderate": 1,
    "High": 2
}

DQ_map = {
    "Poor": 0,
    "Fair": 1,
    "Good": 2,
    "Very Good": 3,
    "Excellent": 4
}

mh["Social_Interaction_Freq"] = mh["Social_Interaction_Freq"].map(SI_map)
mh["Diet_Quality"] = mh["Diet_Quality"].map(DQ_map)

continuous_cols = ["Age", "Hours_of_Sleep", "Stress_Level", "Physical_Activity","Work_Hours_per_Day",
                    "Overall_Wellbeing_Score", "Screen_Time_per_Day"]

robust_cols = ["Physical_Activity", "Overall_Wellbeing_Score"]
standard_cols = [col for col in continuous_cols if col not in robust_cols]

standard_scaler = StandardScaler()
standard_scaler.fit(mh[standard_cols])
standard_transformed = standard_scaler.transform(mh[standard_cols])

for i, col in enumerate(standard_cols):
    mh[col] = standard_transformed[:, i]

robust_scaler = RobustScaler()
robust_scaler.fit(mh[robust_cols])
robust_transformed = robust_scaler.transform(mh[robust_cols])

for i, col in enumerate(robust_cols):
    mh[col] = robust_transformed[:, i]

cat_cols = [1] + list(range(10, 22))

mh_matrix = mh.to_numpy()

final_proto = KPrototypes(n_jobs = 1,n_clusters=6, init='Huang', random_state = 0)
labels = final_proto.fit_predict(mh_matrix, categorical=cat_cols)

original = pd.read_csv("Mental_Health_and_Lifestyle_Research.csv")
original.fillna("None", inplace=True)
original.drop("Person_ID", axis=1, inplace=True)

original["Social_Interaction_Freq"] = original["Social_Interaction_Freq"].map(SI_map)
original["Diet_Quality"] = original["Diet_Quality"].map(DQ_map)

# Now add this to the original data frame
original["Cluster"] = labels
original["Cluster"] = original["Cluster"].astype(str)

cat_columns = ["Gender", "Mental_Health_Status", "Substance_Use", "Has_Close_Friends", "Physical_Health_Condition"]
num_columns = [col for col in original.columns if col not in cat_columns]
cat_columns.append("Cluster")

# now group by cluster and get the mode for categorical columns and mean for numerical columns
output = pd.DataFrame()
output = pd.concat([output, original[cat_columns].groupby("Cluster").agg(lambda x: x.mode().iloc[0])], axis=1)
output = pd.concat([output, original[num_columns].groupby("Cluster").mean()], axis=1)

categories = mh.iloc[:, cat_cols]
categories["Cluster"] = labels.astype(str)

output_proportions = pd.DataFrame()
output_proportions = pd.concat([output_proportions, original[num_columns].groupby("Cluster").mean()], axis = 1)
output_proportions = pd.concat([output_proportions, categories.groupby("Cluster").mean()], axis = 1)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| echo: false
#| label: tab-1-1

from great_tables import GT, md, style, loc

num_col_dict = {
    "Hours_of_Sleep": "Hours of Sleep",
    "Stress_Level": "Stress Level",
    "Physical_Activity": "Physical Activity (mins per day)",
    "Work_Hours_per_Day": "Daily Hours Worked",
    "Social_Interaction_Freq": "Social Interaction",
    "Overall_Wellbeing_Score": "Wellbeing Score",
    "Diet_Quality": "Diet Quality",
    "Screen_Time_per_Day": "Daily Screen Time (hrs)"
}

cat_col_dict = {
    "Mental_Health_Status": "MH Diagnosis",
    "Substance_Use": "Substance Use",
    "Has_Close_Friends": "Has Close Friends",
    "Physical_Health_Condition": "Physical Health Condition"
}

numb_cols = output.columns[range(5, 14)].tolist()
cate_cols = output.columns[range(0, 5)].tolist()

num = output[numb_cols]
cat = output[cate_cols]

num["Cluster"] = range(1, 7)
cat["Cluster"] = range(1, 7)

(
    GT(num)
    .fmt_number(columns=numb_cols[:9], decimals=1)
    .cols_label(
        **num_col_dict
    )
    .cols_move_to_start(columns=list(num.columns[::-1]))
    .tab_style(
        style=style.text(font="Computer Modern"),
        locations=[loc.column_labels(), loc.body()]
    )
    .tab_style(
        style=style.text(weight="bold"),
        locations=[loc.column_labels(), loc.body(columns="Cluster")]
    )
    .tab_header("A")
    .tab_style(
        style=style.text(font="Computer Modern", weight="bold", align="left"),
        locations=loc.header()
    )
)
#
#
#
#| echo: false
#| label: tab-1-2

(
    GT(cat)
    .cols_label(
        **cat_col_dict
    )
    .cols_move_to_start(columns=list(num.columns[::-1]))
    .tab_style(
        style=style.text(font="Computer Modern"),
        locations=[loc.column_labels(), loc.body()]
    )
    .tab_style(
        style=style.text(weight="bold"),
        locations=[loc.column_labels(), loc.body(columns="Cluster")]
    )
    .tab_header("B")
    .tab_style(
        style=style.text(font="Computer Modern", weight="bold", align="left"),
        locations=loc.header()
    )
)
#
#
#
#
#
#
#
#| echo: false
#| label: tab-2

categories = mh.iloc[:, cat_cols]
categories["Cluster"] = labels

props = categories.groupby("Cluster").mean()
props = props * 100
props["Cluster"] = range(1, 7)

col_names = [
    "Gender", "Has Close Friends", "Depression", "Mild Anxiety", "Moderate Anxiety", "Severe Anxiety",
    "Drinks Alcohol", "Drinks Alcohol and Smokes", "Smokes", "Diabetes", "Hypertension", "Obesity",
    "Other Chronic Illness"
]

(
    GT(props)
    .fmt_number(columns=list(props.columns), decimals=1)
    .cols_move_to_start(columns="Cluster")
    .cols_label(col_names)
    .fmt_number(columns="Cluster", decimals=0)
)
#
#
#
#
#
