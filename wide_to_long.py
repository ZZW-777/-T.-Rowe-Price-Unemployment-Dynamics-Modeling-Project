import pandas as pd
import re

codebook = {}
group_to_year = {}
current_year = None 

with open("J345533_labels.txt", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        
        if line.startswith("ER"):
            parts = line.split()
            var_code = parts[0]
            label = " ".join(parts[1:])

            year_match = re.search(r'\b(19|20)\d{2}\b', label)
            if year_match:
                year_found = year_match.group(0)
                if "LABOR INCOME OF HEAD" in label or "LABOR INCOME OF REF PERSON" in label:
                    current_year = str(int(year_found) + 1)
                else:
                    current_year = year_found

            if current_year:
                group_to_year[var_code] = current_year

            base_label = re.sub(r'\s+\d+$', '', label)
            codebook[var_code] = base_label 
# print(group_to_year)

df = pd.read_csv("J345533.csv")

df["ID"] = df["ER30001"] * 1000 + df["ER30002"]

merge_vars = [col for col in df.columns if col in group_to_year and group_to_year[col] != "1968"]

interview_number_vars = [col for col in merge_vars if "INTERVIEW NUMBER" in codebook[col]]

relation_vars = [col for col in merge_vars if "RELATION TO HEAD" in codebook[col] or "RELATION TO REFERENCE PERSON" in codebook[col]]

insurance_vars = [col for col in merge_vars if "H61D WTR COVERED BY INSURANCE NOW" in codebook[col] or "H61D3 WTR COVERED BY INSURANCE NOW" in codebook[col]]

labor_vars = [col for col in merge_vars if "LABOR INCOME OF HEAD" in codebook[col] or "LABOR INCOME OF REF PERSON" in codebook[col]]

long_data_optimized = []

for var_code in merge_vars:
    year = group_to_year[var_code] 
    variable_name = codebook[var_code] 

    if variable_name in [codebook[v] for v in interview_number_vars]:
        variable_name = "INTERVIEW NUMBER"

    if variable_name in [codebook[v] for v in relation_vars]:
        variable_name = "RELATION TO HEAD"
    
    if variable_name in [codebook[v] for v in insurance_vars]:
        variable_name = "WTR COVERED BY INSURANCE NOW"
    
    if variable_name in [codebook[v] for v in labor_vars]:
        variable_name = "LABOR INCOME OF HEAD"

    temp_df = df[['ID']].copy()  
    temp_df['Year'] = year  
    temp_df['Variable'] = variable_name 
    temp_df['Value'] = df[var_code] 

    long_data_optimized.append(temp_df)

long_df_optimized = pd.concat(long_data_optimized, ignore_index=True)

final_df_optimized = long_df_optimized.pivot_table(
    index=['ID', 'Year'], 
    columns='Variable', 
    values='Value', 
    aggfunc='first'
).reset_index()

final_df_optimized = final_df_optimized[final_df_optimized["Year"] != "1968"]
columns_to_remove = ["PERSON NUMBER", "RELEASE NUMBER"]
final_df_optimized = final_df_optimized.drop(columns=[col for col in columns_to_remove if col in final_df_optimized.columns], errors='ignore')

constant_vars = ["ER32000", "ER32006", "ER32022", "ER32049"]

constant_data = df[["ID"] + constant_vars].drop_duplicates(subset=["ID"]).set_index("ID")

final_df_optimized = final_df_optimized.merge(constant_data, on="ID", how="left")

rename_dict = {var: codebook[var] for var in constant_vars}
final_df_optimized.rename(columns=rename_dict, inplace=True)

final_df_optimized = final_df_optimized.convert_dtypes()
final_df_optimized.to_csv("raw_data.csv", index=False)

