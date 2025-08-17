import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib import font_manager

# Set random seed for reproducibility
np.random.seed(42)

df1 = pd.read_csv("cleaned_data.csv")
with open("model1_coefficients.pkl", "rb") as f:
    model1_coefficients = pickle.load(f)

with open("model2_coefficients.pkl", "rb") as f:
    model2_coefficients = pickle.load(f)

df2 = pd.read_csv("df2_saved.csv")

# ==============================
# Simulation Functions
# ==============================
def predict_prob(features, model):
    log_odds = model.get('Intercept', 0)
    for key in ['Education_2[T.True]', 'Education_3[T.True]', 'Gender_1[T.True]',
                'Age', 'Age_Squared', 'Labor_Income']:
        log_odds += model.get(key, 0) * features.get(key, 0)
    for ed in ['Education_2[T.True]', 'Education_3[T.True]']:
        if features.get(ed, 0) == 1 and features.get('Gender_1[T.True]', 0) == 1:
            interaction_key = f"{ed}:Gender_1[T.True]"
            log_odds += model.get(interaction_key, 0)
    log_odds += model.get('Age:Labor_Income', 0) * features['Age'] * features['Labor_Income']
    log_odds += model.get('employed_lag_1[T.True]', 0) * features.get('employed_lag_1[T.True]', 0)
    return 1 / (1 + np.exp(-log_odds))

# def sample_covariates(gender1=None):
#     df_edu2 = df2[df2['Education'] == 2]
#     sigma_squared_edu2 = np.log(1 + (df_edu2['Labor_Income'].std() ** 2) / (df_edu2['Labor_Income'].mean() ** 2))
#     sigma_edu2 = np.sqrt(sigma_squared_edu2)
#     mu_edu2 = np.log(df_edu2['Labor_Income'].mean()) - (sigma_squared_edu2 / 2)
#
#     # For Education 3
#     df_edu3 = df2[df2['Education'] == 3]
#     sigma_squared_edu3 = np.log(1 + (df_edu3['Labor_Income'].std() ** 2) / (df_edu3['Labor_Income'].mean() ** 2))
#     sigma_edu3 = np.sqrt(sigma_squared_edu3)
#     mu_edu3 = np.log(df_edu3['Labor_Income'].mean()) - (sigma_squared_edu3 / 2)
#
#     # For Education 1 (baseline)
#     df_edu1 = df2[df2['Education'] == 1]
#     sigma_squared_edu1 = np.log(1 + (df_edu1['Labor_Income'].std() ** 2) / (df_edu1['Labor_Income'].mean() ** 2))
#     sigma_edu1 = np.sqrt(sigma_squared_edu1)
#     mu_edu1 = np.log(df_edu1['Labor_Income'].mean()) - (sigma_squared_edu1 / 2)
#
#     p_edu2 = (df2['Education'] == 2).mean()
#     p_edu3 = (df2['Education'] == 3).mean()
#     p_edu1 = 1 - p_edu2 - p_edu3
#
#     education_level = np.random.choice(
#         ['edu1', 'edu2', 'edu3'],
#         p=[p_edu1, p_edu2, p_edu3]
#     )
#     edu2 = 1 if education_level == 'edu2' else 0
#     edu3 = 1 if education_level == 'edu3' else 0
#
#     if gender1 is None:
#         gender1 = np.random.binomial(1, (df2['Gender'] == 1).mean())
#
#     if edu2 == 1:
#         labor_income = np.random.lognormal(mean=mu_edu2, sigma=sigma_edu2)
#     elif edu3 == 1:
#         labor_income = np.random.lognormal(mean=mu_edu3, sigma=sigma_edu3)
#     else:
#         labor_income = np.random.lognormal(mean=mu_edu1, sigma=sigma_edu1)
#
#     labor_income = max(labor_income, 0)
#
#     return {
#         'Education_2[T.True]': edu2,
#         'Education_3[T.True]': edu3,
#         'Gender_1[T.True]': gender1,
#         'Labor_Income': labor_income
#     }

def sample_covariates(gender1=None):
    if gender1 is None:
        gender1 = np.random.binomial(1, (df2['Gender'] == 1).mean())

    # Split by gender and education
    def lognorm_params(subset):
        sigma_squared = np.log(1 + (subset['Labor_Income'].std() ** 2) / (subset['Labor_Income'].mean() ** 2))
        sigma = np.sqrt(sigma_squared)
        mu = np.log(subset['Labor_Income'].mean()) - (sigma_squared / 2)
        return mu, sigma

    params = {}

    for edu in [1, 2, 3]:
        for g in [0, 1]:
            subset = df2[(df2['Education'] == edu) & (df2['Gender'] == g)]
            key = f'edu{edu}_g{g}'
            if len(subset) > 0:
                params[key] = lognorm_params(subset)
            else:
                params[key] = (np.log(1), 0.5)  # fallback

    # Choose education level
    p_edu2 = (df2['Education'] == 2).mean()
    p_edu3 = (df2['Education'] == 3).mean()
    p_edu1 = 1 - p_edu2 - p_edu3

    education_level = np.random.choice(
        ['edu1', 'edu2', 'edu3'],
        p=[p_edu1, p_edu2, p_edu3]
    )
    edu2 = 1 if education_level == 'edu2' else 0
    edu3 = 1 if education_level == 'edu3' else 0

    edu_num = 2 if edu2 else (3 if edu3 else 1)
    mu, sigma = params[f'edu{edu_num}_g{gender1}']

    labor_income = np.random.lognormal(mean=mu, sigma=sigma)
    labor_income = max(labor_income, 0)

    return {
        'Education_2[T.True]': edu2,
        'Education_3[T.True]': edu3,
        'Gender_1[T.True]': gender1,
        'Labor_Income': labor_income
    }

def simulate_individual(start_age=24, end_age=65, gender1=None):
    T = end_age - start_age + 1
    cov = sample_covariates(gender1=gender1)
    status = []

    for t in range(T):
        age = start_age + t
        cov['Age'] = age
        cov['Age_Squared'] = age ** 2
        cov['Age:Labor_Income'] = cov['Age'] * cov['Labor_Income']

        if t == 0:
            prob = predict_prob(cov, model1_coefficients)
            prob += np.random.normal(0, 0.01)
            prob = np.clip(prob, 0, 1)
            status.append(1 if np.random.rand() < prob else 0)
        else:
            cov['employed_lag_1[T.True]'] = status[-1]
            prob = predict_prob(cov, model2_coefficients)
            prob += np.random.normal(0, 0.05)
            prob = np.clip(prob, 0, 1)
            status.append(1 if np.random.rand() < prob else 0)
    return status


def simulate_population(n_individuals=10000, gender1=None):
    population = []
    for i in range(n_individuals):
        career = simulate_individual(gender1=gender1)
        population.append(career)
    return np.array(population)

results = simulate_population(n_individuals=10000)
results_male = simulate_population(n_individuals=10000, gender1=1)
results_female = simulate_population(n_individuals=10000, gender1=0)

# Analyze results
ages = list(range(24, 66))
unemployment_rate = 1 - results.mean(axis=0)
p_ue = []  # P(Unemployed | Employed last year)
p_uu = []  # P(Unemployed | Unemployed last year)

for t in range(1, results.shape[1]):
    employed_last = results[:, t-1] == 1
    unemployed_last = results[:, t-1] == 0
    unemployed_this = results[:, t] == 0

    p_ue.append((unemployed_this & employed_last).sum() / employed_last.sum())
    p_uu.append((unemployed_this & unemployed_last).sum() / unemployed_last.sum())

summary_df1 = pd.DataFrame({
    "Age": ages[1:],
    "Unemployment_Rate": unemployment_rate[1:],
    "P(U|E)": p_ue,
    "P(U|U)": p_uu
})
print(summary_df1)

# Compute unemployment spell counts and lengths
def summarize_unemployment(career):
    spells = []
    count = 0
    for s in career:
        if s == 0:
            count += 1
        else:
            if count > 0:
                spells.append(count)
                count = 0
    if count > 0:
        spells.append(count)
    return spells

spell_lengths = [summarize_unemployment(row) for row in results]
unemployment_spell_counts = [len(s) for s in spell_lengths]
average_spell_length = [np.mean(s) if s else 0 for s in spell_lengths]

# Convert to DataFrame for further analysis
summary_df = pd.DataFrame({
    'spell_count': unemployment_spell_counts,
    'avg_spell_length': average_spell_length
})

unemployment_rate_t0 = (results[:, 0] == 0).mean()
print(f"Model1 Simulation Unemployment Rate at Age 24: {unemployment_rate_t0:.4f}")

df1['Employed'] = df1['Employment']  # 1 = employed, 0 = unemployed

simulated_unemployment_by_age = pd.DataFrame({
    'Age': ages,
    'Simulated_Unemployment_Rate': 1 - results.mean(axis=0)
})

real_unemployment_by_age = 1 - df1.groupby('Age')['Employed'].mean()
real_unemployment_by_age = real_unemployment_by_age.reset_index()
real_unemployment_by_age.columns = ['Age', 'Real_Unemployment_Rate']
comparison_df = pd.merge(
    real_unemployment_by_age,
    simulated_unemployment_by_age,
    on='Age',
    how='inner'
)
print(comparison_df)

# Unemployment Spell distribution
all_spell_lengths = [length for sublist in spell_lengths for length in sublist]

plt.rcParams["font.family"] = "Times New Roman"

# -------------------------------
# P1: Distribution of Spell Lengths
# -------------------------------
spell_lengths = [summarize_unemployment(row) for row in results]
all_spell_lengths = [length for sublist in spell_lengths for length in sublist]

plt.figure(figsize=(8, 6))
plt.hist(all_spell_lengths, bins=range(1, 10), edgecolor='black', color='salmon')
plt.title("Distribution of Unemployment Spell Lengths", fontsize=17, fontweight='bold')
plt.xlabel("Length of Spell (Years)", fontsize=14)
plt.ylabel("Frequency", fontsize=14)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.tight_layout()
plt.show()

# -------------------------------
# P2: Real vs Simulated Unemployment and Transition Probabilities
# -------------------------------
summary_with_real = pd.merge(
    summary_df1,
    real_unemployment_by_age,
    on="Age",
    how="inner"
)

plt.figure(figsize=(10, 6))
plt.plot(summary_with_real["Age"].values, summary_with_real["Unemployment_Rate"].values,
         label="Simulated Unemployment Rate", color="orange", linewidth=2)
plt.plot(summary_with_real["Age"].values, summary_with_real["Real_Unemployment_Rate"].values,
         label="Real Unemployment Rate", color="green", linestyle="-.", linewidth=2)
plt.plot(summary_with_real["Age"].values, summary_with_real["P(U|E)"].values,
         label="P(U|E)", color="blue", linestyle="--", linewidth=2)
plt.plot(summary_with_real["Age"].values, summary_with_real["P(U|U)"].values,
         label="P(U|U)", color="red", linestyle=":", linewidth=2)

plt.title("Real vs Simulated Unemployment and Transition Probabilities by Age", fontsize=17, fontweight='bold')
plt.xlabel("Age", fontsize=17)
plt.ylabel("Probability", fontsize=17)
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.ylim(0, 0.3)
plt.grid(True, linestyle='--', alpha=0.6)

legend_font = font_manager.FontProperties(family='Times New Roman', weight='bold', size=12)
plt.legend(prop=legend_font)
plt.tight_layout()
plt.show()

# -------------------------------
# P3: Male & Female Unemployment Rate
# -------------------------------
unemployment_male = 1 - results_male.mean(axis=0)
unemployment_female = 1 - results_female.mean(axis=0)

plt.figure(figsize=(10,6))
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.plot(ages, unemployment_male, label="Male", color="blue")
plt.plot(ages, unemployment_female, label="Female", color="red")
plt.title("Simulated Unemployment Rate by Gender", fontsize=17, fontweight='bold')
plt.xlabel("Age", fontsize=17)
plt.ylabel("Unemployment Rate", fontsize=17)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
