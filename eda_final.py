import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

######################## Figure 1: Employment status histogram ################################
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'     # Axis label bold
plt.rcParams['axes.titleweight'] = 'bold'
file_path = "./data_heads.csv"
df = pd.read_csv(file_path)
df['Employed'] = df['Employment'].apply(lambda x: 1 if x == 1 else 0)
df = df[~df['Employment'].isin([8, 9])]

# Filter only the employment status codes 1, 2, 3
df_filtered = df[df['Employment'].isin([1, 2, 3])].copy()

# Map employment codes to readable labels
code_to_label = {
    1: '1:Working now',
    2: '2:Temporarily laid off',
    3: '3:Unemployed'
}
df_filtered['Employment_Label'] = df_filtered['Employment'].map(code_to_label)

# Plot the count of selected employment statuses
plt.figure(figsize=(10, 7))
custom_palette = ['#8B0000', '#D2691E', '#F08080'] 
sns.countplot(x='Employment', data=df_filtered, palette=custom_palette)

# Apply custom tick labels
plt.xticks(ticks=[0, 1, 2], labels=[code_to_label[1], code_to_label[2], code_to_label[3]])
plt.title("Selected Employment Status Distribution", fontsize=20, fontweight='bold')
plt.xlabel("Employment Status", fontsize=17, fontweight='bold')
plt.ylabel("Count", fontsize=17, fontweight='bold')
plt.tight_layout()
plt.show()

######################## Figure 2: Insurance Coverage by Employment Status ################################
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman' 
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2

df = pd.read_csv(file_path)

df.columns = df.columns.str.strip()

employment_map = {
    0: "Not Applicable", 1: "Working now", 2: "Temporarily laid off",
    3: "Looking for work", 4: "Retired", 5: "Permanently disabled",
    6: "Keeping house", 7: "Student", 8: "Other", 9: "Unknown"
}
insurance_map = {
    1: "Insured", 5: "Not Insured", 0: "Not Applicable", 9: "Unknown"
}

df["Employment_Status_Label"] = df["Employment"].map(employment_map)
df["Insurance_Coverage_Label"] = df["Insurance"].map(insurance_map)

df_plot = df.dropna(subset=["Employment_Status_Label", "Insurance_Coverage_Label"])

valid_employment = df_plot[
    ~df_plot["Employment_Status_Label"].isin(["Not Applicable", "Unknown"])
]

grouped = (
    valid_employment
    .groupby(["Employment_Status_Label", "Insurance_Coverage_Label"])
    .size()
    .reset_index(name="Count")
)
grouped["Total"] = grouped.groupby("Employment_Status_Label")["Count"].transform("sum")
grouped["Percentage"] = grouped["Count"] / grouped["Total"] * 100

pivot_df = grouped.pivot(index="Employment_Status_Label", columns="Insurance_Coverage_Label", values="Percentage")

for col in ["Insured", "Not Insured", "Not Applicable", "Unknown"]:
    if col not in pivot_df.columns:
        pivot_df[col] = 0
pivot_df = pivot_df[["Insured", "Not Insured", "Not Applicable", "Unknown"]]

custom_palette_swapped = {
    "Insured": "#CD5C5C",
    "Not Insured": "#8B0000",
    "Not Applicable": "#D3D3D3",
    "Unknown": "#A9A9A9"
}

ax = pivot_df.plot(
    kind="bar",
    stacked=True,
    figsize=(14, 6),
    color=[custom_palette_swapped[col] for col in pivot_df.columns]
)

ax.set_ylabel("Percentage", fontsize=17, fontweight='bold')
ax.set_title("Percentage of Insurance Coverage by Employment Status", fontsize=18, fontweight='bold')
ax.set_xlabel("Employment Status", fontsize=17, fontweight='bold')
plt.xticks(rotation=45)
plt.legend(title="Insurance Coverage", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

######################## Figure 3: Median Labor Income by Employment Status ################################
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()

employment_map = {
    0: "Not Applicable", 1: "Working now", 2: "Temporarily laid off",
    3: "Looking for work", 4: "Retired", 5: "Permanently disabled",
    6: "Keeping house", 7: "Student", 8: "Other", 9: "Unknown"
}
df["Employment_Status_Label"] = df["Employment"].map(employment_map)
df = df[df["Employment"] != 9]  

df_income = df.dropna(subset=["Labor_Income", "Employment"])
df_income = df_income[df_income["Employment_Status_Label"].notna()]
income_cap = df_income["Labor_Income"].quantile(0.99)
df_income_capped = df_income[df_income["Labor_Income"] <= income_cap]

median_income = (
    df_income_capped
    .groupby("Employment_Status_Label")["Labor_Income"]
    .median()
    .sort_values(ascending=False)
)

custom_colors_median = [
    "#CD5C5C", "#8B0000", "#A52A2A", "#B22222", "#DC143C",
    "#FF6347", "#E9967A", "#FA8072", "#F08080"
]

plt.rcParams['font.family'] = 'Times New Roman'         
plt.rcParams['font.weight'] = 'bold'                    
plt.rcParams['axes.labelweight'] = 'bold'               
plt.rcParams['axes.titleweight'] = 'bold'               
plt.rcParams['xtick.labelsize'] = 18                    
plt.rcParams['ytick.labelsize'] = 18                    
plt.rcParams['xtick.major.width'] = 1.2                
plt.rcParams['ytick.major.width'] = 1.2

plt.figure(figsize=(14, 6))
sns.barplot(x=median_income.index, y=median_income.values, palette=custom_colors_median)
plt.xlabel("Employment Status", fontsize=17, fontweight='bold')
plt.ylabel("Median Labor Income", fontsize=17, fontweight='bold')
plt.title("Median Labor Income by Employment Status (Outliers Capped)", fontsize=20, fontweight='bold')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

######################## Figure 4: Labor Income by Education Level ################################
# Keep only positive income values
df_income = df[df['Labor_Income'] > 0]

# Calculate IQR and define bounds
Q1 = df_income['Labor_Income'].quantile(0.25)
Q3 = df_income['Labor_Income'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Remove outliers
df_filtered = df_income[(df_income['Labor_Income'] >= lower_bound) & (df_income['Labor_Income'] <= upper_bound)]

# Plot boxplot by education level
plt.figure(figsize=(8, 5))
df_filtered.boxplot(column='Labor_Income', by='Education', grid=False)
plt.title('Labor Income by Education Level (Outliers Removed)', fontsize=20, fontweight='bold')
plt.suptitle('')
plt.xlabel('Education Level', fontsize=17, fontweight='bold')
plt.ylabel('Labor Income', fontsize=17, fontweight='bold')
# Replace x-tick labels
plt.xticks(ticks=[1, 2, 3], labels=['Less than 9 years', 'High School', 'College'])

plt.tight_layout()
plt.show()

######################## Figure 5: Labor Income by Gender ################################
plt.figure(figsize=(8, 5))
df_filtered.boxplot(column='Labor_Income', by='Gender', grid=False)
plt.title('Labor Income by Gender (Outliers Removed)', fontsize=20, fontweight='bold')
plt.suptitle('')
plt.xlabel('Gender', fontsize=17, fontweight='bold')
plt.ylabel('Labor Income', fontsize=17, fontweight='bold')
plt.xticks(ticks=[1, 2], labels=['Female','Male'])
plt.tight_layout()
plt.show()

######################## Figure 6: Years of Education vs Employment Status ################################
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
df = pd.read_csv(file_path)
df['Employed'] = df['Employment'].apply(lambda x: 1 if x == 1 else 0)
df = df[~df['Employment'].isin([8, 9])]
heat_data = pd.crosstab(df['Employed'], df['Education'])

# Optional: rename columns for clarity
heat_data.columns = ['Before High School', 'High School', 'College']

# Plot heatmap
plt.figure(figsize=(8, 5))
sns.heatmap(heat_data, annot=True, fmt='d', cmap='YlGnBu')

# Titles and labels
plt.xticks(rotation=0)
plt.title('Heatmap of Employment Status vs Education Level', fontsize=20, fontweight='bold')
plt.xlabel('Education Level', fontsize=20, fontweight='bold')
plt.ylabel('Employment Status', fontsize=20, fontweight='bold')
plt.tight_layout()
plt.show()

######################## Figure 7: Sex vs Employment Status ################################
# Create counts
counts = df.groupby(['Gender', 'Employment']).size().unstack(fill_value=0)

# Rename gender and employment values for clarity
gender_labels = {0: 'Female', 1: 'Male'}
employment_labels = {
    1: "Employed",
    2: "Temporary laid off",
    3: "unemployed",
    4: "Retired",
    5: "Permanently disabled",
    6: "HouseWife",
    7: "Student"
}

counts.index = counts.index.map(gender_labels)
counts = counts.rename(columns=employment_labels)

# Convert to percentages
counts_percent = counts.div(counts.sum(axis=1), axis=0) * 100

# Plot stacked percentage bar chart
counts_percent.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='tab20')
plt.xticks(rotation=0)
plt.title('Employment Status by Sex (Percentage)', fontsize=20, fontweight='bold')
plt.xlabel('Gender', fontsize=20, fontweight='bold')
plt.ylabel('Percentage', fontsize=20, fontweight='bold')
plt.legend(title='Employment Status', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


######################## Figure 8: Marital status by employment ################################
heat_data = pd.crosstab(df['Last_Marital'], df['Employment'])

# Optional: Rename for readability
marital_labels = {
    1: 'Married',
    2: 'Never Married',
    3: 'Widowed',
    4: 'Divorced',
    5: 'Separated'
}
employment_labels = {
    0: 'Not Working',
    1: 'Working',
    2: 'Temp. Laid Off',
    3: 'Unemployed',
    4: 'Retired',
    5: 'Disabled',
    6: 'Other',
    7: 'Housekeeping'
}

heat_data.index = heat_data.index.map(marital_labels)
heat_data.columns = [employment_labels.get(col, col) for col in heat_data.columns]

# Plot heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(heat_data, annot=True, fmt='d', cmap='YlGnBu')
plt.xticks(rotation=0)
plt.title('Heatmap of Marital Status vs Employment', fontsize=20, fontweight='bold')
plt.xlabel('Employment Status', fontsize=20, fontweight='bold')
plt.ylabel('Marital Status', fontsize=20, fontweight='bold')
plt.tight_layout()
plt.show()


######################## Figure 9: Employment rate by age ################################
df = df[~df['Employment'].isin([4,5,6,7,8,9])]

# Group by age and compute employment rate
age_employment_rate = df.groupby('Age')['Employed'].mean().reset_index()

# Plot line chart
plt.figure(figsize=(10, 6))
plt.plot(age_employment_rate['Age'], age_employment_rate['Employed'], marker='o')
plt.xticks(rotation=0)
plt.title('Employment Rate by Age', fontsize=20, fontweight='bold')
plt.xlabel('Age', fontsize=20, fontweight='bold')
plt.ylabel('Employment Rate', fontsize=20, fontweight='bold')
plt.ylim(0, 1)
plt.grid(True)
plt.tight_layout()
plt.show()
