import pandas as pd
import numpy as np

df = pd.read_csv("raw_data.csv")

# Age Preprocessing
df = df.rename(columns={'AGE OF INDIVIDUAL':'Age', 'ID':'Person_ID'})
df["Age"] = df["Age"].replace(0, np.nan)

year_offsets = {2011: 0, 2013: 2, 2015: 4, 2017: 6, 2019: 8, 2021: 10}
df["Year_Offset"] = df["Year"].map(year_offsets)

first_ages = (
    df.dropna(subset=["Age"])
    .sort_values(by=["Person_ID", "Year"])
    .groupby("Person_ID")
    .first()[["Year", "Age"]]
    .rename(columns={"Year": "Base_Year", "Age": "Base_Age"})
)

df = df.merge(first_ages, on="Person_ID", how="left")
df["Age"] = df["Base_Age"] + (df["Year_Offset"] - df["Base_Year"].map(year_offsets))
df["Age"] = df["Age"].where(df["Age"] >= 0)
df["Age"] = df["Age"].round().astype("Int64")
df.drop(columns=["Base_Year", "Base_Age", "Year_Offset"], inplace=True)

# Heads only
df = df[df['RELATION TO HEAD'] == 10]

df = df[(df["Age"] >= 18) & (df["Age"] <= 70)]
df = df.rename(columns={'LABOR INCOME OF HEAD':'Labor_Income',
                        'WTR COVERED BY INSURANCE NOW':'Insurance',
                        'YEARS COMPLETED EDUCATION':'Education',
                        'SEX OF INDIVIDUAL':'Gender',
                        'LAST KNOWN MARITAL STATUS':'Last_Marital',
                        'EMPLOYMENT STATUS':'Employment'})

df["Age_Group"] = pd.cut(df["Age"], bins=[17, 34, 54, 70],
                         labels=["18-34", "35-54", "55-70"])

df["Education"] = df["Education"].map(
    lambda x: 1 if 1 <= x <= 8 else 2 if 9 <= x <= 12 else 3 if 13 <= x <= 17 else None
)

df.dropna(inplace=True)
df["Employment"] = df["Employment"].astype('int')
df = df[~df['Insurance'].isin([0, 8, 9])]
df['Insurance'] = df['Insurance'].replace({5: 0}).astype('int')
df['Education'] = df['Education'].astype('int')
df['Gender'] = df['Gender'].replace({2: 0}).astype('int')

# 2011 is actually the cpi of 2010
cpi_dict = {
    2011: 218.1,
    2013: 229.6,
    2015: 236.7,
    2017: 240.0,
    2019: 251.1,
    2021: 258.8,
}
reference_cpi = cpi_dict[2011]
df["CPI"] = df["Year"].map(cpi_dict)
df["Labor_Income"] = df["Labor_Income"] * (reference_cpi / df["CPI"])

df = df[~df['Last_Marital'].isin([8, 9])]

# Save this data for EDA
df.to_csv("data_heads.csv", index=False)

# Define employment and unemployment categories
df["Employment"] = df["Employment"].map(
    lambda x: 1 if x == 1 else 0 if x in [2, 3] else None
)
df.dropna(inplace=True)

# Save this data for Model 1
df = df[['Person_ID', 'Year', 'Age', 'Labor_Income', 'Insurance',
         'Education', 'Gender', 'Last_Marital', 'Employment']]
df.to_csv("cleaned_data.csv", index=False)

# Add employed_lag
df = df.sort_values(by=['Person_ID', 'Year'])
df['join_year'] = df['Year'] - 2
df_lag = df[['Person_ID', 'Year', 'Employment']].copy()
df_lag.rename(columns={'Year': 'lag_Year', 'Employment': 'employed_lag'}, inplace=True)

df = pd.merge(df, df_lag, left_on=['Person_ID', 'join_year'], 
              right_on=['Person_ID', 'lag_Year'], how='left')
df.drop(columns=['join_year', 'lag_Year'], inplace=True)
df = df.dropna(subset=['employed_lag'])
df['employed_lag'] = df['employed_lag'].astype('int')

# Save this data for Model 2
df = df[['Person_ID', 'Year', 'Age', 'Labor_Income', 'Insurance',
         'Education', 'Gender', 'Last_Marital', 'Employment', 'employed_lag']]
df.to_csv("cleaned_data_model2.csv", index=False)