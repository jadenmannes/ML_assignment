from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as ss
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


DATA_PATH = Path("Dropout_Academic Success - Sheet1.csv")
if DATA_PATH is None:
    raise FileNotFoundError(
        "Could not find the dataset. Put 'Dropout_Academic Success - Sheet1.csv' "
        "inside a Data folder next to this script/notebook."
    )

df = pd.read_csv(DATA_PATH)

pd.set_option("display.max_columns", None)
print("Dataset path:", DATA_PATH)
print("Dataset shape:", df.shape)
print(df.head())

# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum().sort_values())
plt.figure(figsize=(12, 8))
sns.heatmap(df.isnull(), cbar=False)
plt.title("Missing Values Heatmap")
plt.tight_layout()
plt.show()

# Check the original class distribution
print("\nOriginal target distribution:")
print(df["Target"].value_counts())
df["Target"].value_counts().plot(kind="bar", figsize=(8, 5))
plt.title("Original Target Distribution")
plt.ylabel("Number of students")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Binary target setup:
# 'Enrolled' students do not have a known final outcome yet, so they are held aside.
# The model is trained on known final outcomes only: Graduate vs Dropout.
TARGET_MAP = {"Graduate": 0, "Dropout": 1}
LABEL_MAP = {value: key for key, value in TARGET_MAP.items()}
TARGET_NAMES = [LABEL_MAP[i] for i in sorted(LABEL_MAP)]

final_outcome_df = df[df["Target"].isin(TARGET_MAP)].copy()
enrolled_students = df[df["Target"].eq("Enrolled")].copy()
final_outcome_df["Target_binary"] = final_outcome_df["Target"].map(TARGET_MAP)

print("\nRows used for supervised binary training:", len(final_outcome_df))
print("Currently enrolled rows held aside for prediction:", len(enrolled_students))
print("\nBinary target distribution:")
print(final_outcome_df["Target"].value_counts())
final_outcome_df["Target"].value_counts().plot(kind="bar")
plt.title("Binary Target Distribution")
plt.ylabel("Number of students")
plt.tight_layout()
plt.show()

# Check unique values
for col in df.columns:
    print(col, df[col].nunique())

# Correlation matrix for the binary modeling data
corr = final_outcome_df.drop(columns=["Target"]).corr(numeric_only=True)
plt.figure(figsize=(16, 14))
sns.heatmap(corr, cmap="coolwarm")
plt.title("Correlation Matrix for Binary Modeling Data")
plt.tight_layout()
plt.show()

# Cramér's V for categorical relationships with the binary target
def cramers_v(x, y):
    table = pd.crosstab(x, y)
    chi2 = ss.chi2_contingency(table)[0]
    n = table.sum().sum()
    phi2 = chi2 / n
    r, k = table.shape
    denominator = min(k - 1, r - 1)
    if denominator == 0:
        return 0
    return (phi2 / denominator) ** 0.5


cramer_results = []

for col in final_outcome_df.columns:
    if col not in ["Target", "Target_binary"]:
        score = cramers_v(final_outcome_df[col], final_outcome_df["Target"])
        cramer_results.append({
            "feature": col,
            "cramers_v": score
        })

cramer_df = pd.DataFrame(cramer_results).sort_values("cramers_v", ascending=False)
print(cramer_df)

plt.figure(figsize=(12, 9))
sns.barplot(data=cramer_df.head(20), x="cramers_v", y="feature")
plt.title("Top 20 Features by Cramér's V with Binary Target")
plt.xlabel("Cramér's V")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# Feature importance on the binary target
feature_columns = [col for col in final_outcome_df.columns if col not in ["Target", "Target_binary"]]
X_binary = final_outcome_df[feature_columns]
y_binary = final_outcome_df["Target_binary"]

feature_model = CatBoostClassifier(
    loss_function="Logloss",
    auto_class_weights="Balanced",
    random_seed=42,
    verbose=0
)
feature_model.fit(X_binary, y_binary)

importance = feature_model.get_feature_importance()

feat_imp = pd.DataFrame({
    "feature": X_binary.columns,
    "importance": importance
}).sort_values("importance", ascending=False)

feat_imp.plot(
    x="feature",
    y="importance",
    kind="barh",
    figsize=(10, 8)
)

# Feature importance plot with fixed spacing for long feature names
plt.figure(figsize=(14, 10))

sns.barplot(
    data=feat_imp.head(20),
    x="importance",
    y="feature"
)

plt.title("Top 20 CatBoost Feature Importance for Dropout vs Graduate")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.subplots_adjust(left=0.42, right=0.95, top=0.92, bottom=0.08)
plt.show()

# Distribution analysis
final_outcome_df[feature_columns].hist(figsize=(15, 10))
plt.tight_layout()
plt.show()

# Target relationship analysis
for col in feature_columns:
    print(f"\n===== {col} =====")

    cross = pd.crosstab(
        final_outcome_df[col],
        final_outcome_df["Target"],
        normalize="index"
    )

    print(cross)

# PCA visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_binary)
pca_df = pd.DataFrame({
    "PC1": X_pca[:, 0],
    "PC2": X_pca[:, 1],
    "Target": final_outcome_df["Target"].values
})

plt.figure(figsize=(9, 6))
sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="Target", alpha=0.7)
plt.title("PCA Visualization: Dropout vs Graduate")
plt.tight_layout()
plt.show()


# Pre-Processing
# Remove duplicates separately for the labeled training data and the enrolled students to be predicted
final_outcome_df = final_outcome_df.drop_duplicates().copy()
enrolled_students = enrolled_students.drop_duplicates().copy()

feature_columns = [col for col in final_outcome_df.columns if col not in ["Target", "Target_binary"]]
X = final_outcome_df[feature_columns].copy()
y = final_outcome_df["Target_binary"].copy()
X_enrolled = enrolled_students[feature_columns].copy()

# Group rare categories using only the supervised training data.
# The same mapping is then applied to the currently enrolled students.
categorical_columns = [
    "Marital status",
    "Application mode",
    "Application order",
    "Course",
    "Daytime/evening attendance",
    "Previous qualification",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "International",
]
categorical_columns = [col for col in categorical_columns if col in feature_columns]

min_count = 20
rare_value_map = {}

for col in categorical_columns:
    counts = X[col].value_counts()
    rare_values = counts[counts < min_count].index
    common_values = counts[counts >= min_count].index

    if len(rare_values) > 0:
        rare_value_map[col] = list(rare_values)
        print(f"\n{col}")
        print(f"Rare categories grouped: {len(rare_values)}")
        print("Rare values:")
        print(list(rare_values)[:10])

        X[col] = X[col].where(~X[col].isin(rare_values), -1)

    # Values in the enrolled group that were not common in training are treated as rare/unknown.
    X_enrolled[col] = X_enrolled[col].where(X_enrolled[col].isin(common_values), -1)

print("\nCategorical columns with rare-category grouping:")
print(list(rare_value_map.keys()))

# Train/validation/test split with stratification to keep dropout/graduate proportions stable
# 80% train, 10% validation, 10% test

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp
)

print("Train shape:", X_train.shape)
print("Validation shape:", X_val.shape)
print("Test shape:", X_test.shape)
print("Enrolled students to predict:", X_enrolled.shape)

print("\nTrain target distribution:")
print(y_train.map(LABEL_MAP).value_counts(normalize=True))

print("\nValidation target distribution:")
print(y_val.map(LABEL_MAP).value_counts(normalize=True))

print("\nTest target distribution:")
print(y_test.map(LABEL_MAP).value_counts(normalize=True))


# Training Model

base_model = CatBoostClassifier(
    loss_function="Logloss",
    eval_metric="F1",
    auto_class_weights="Balanced",
    random_seed=42,
    verbose=0
)

param_grid = {
    "iterations": [300, 500, 800, 1000],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "depth": [4, 5, 6, 7, 8, 10],
    "l2_leaf_reg": [1, 3, 5, 7, 9],
    "border_count": [32, 64, 128],
    "bagging_temperature": [0, 0.5, 1, 2, 5]
}

search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_grid,
    n_iter=25,
    scoring="f1_macro",
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

search.fit(X_train, y_train)

print("Best parameters:")
print(search.best_params_)

print("\nBest CV macro F1:")
print(search.best_score_)


# Train the final validation model with the best parameters.
# This version uses the validation set to monitor overfitting.

best_params = search.best_params_

best_model = CatBoostClassifier(
    **best_params,
    loss_function="Logloss",
    eval_metric="Logloss",
    auto_class_weights="Balanced",
    random_seed=42,
    verbose=100
)

best_model.fit(
    X_train,
    y_train,
    eval_set=(X_val, y_val),
    use_best_model=True,
    early_stopping_rounds=50
)


# Loss Curve

evals_result = best_model.get_evals_result()

train_loss = evals_result["learn"]["Logloss"]
val_loss = evals_result["validation"]["Logloss"]

plt.figure(figsize=(8, 5))
plt.plot(train_loss, label="Train Logloss")
plt.plot(val_loss, label="Validation Logloss")
plt.xlabel("Iteration")
plt.ylabel("Logloss")
plt.title("CatBoost Training vs Validation Loss")
plt.legend()
plt.show()


# Overfitting Check

train_pred = best_model.predict(X_train).astype(int).ravel()
val_pred = best_model.predict(X_val).astype(int).ravel()
y_pred = best_model.predict(X_test).astype(int).ravel()

print("\nOverfitting Check")
print("Train Accuracy:", accuracy_score(y_train, train_pred))
print("Validation Accuracy:", accuracy_score(y_val, val_pred))
print("Test Accuracy:", accuracy_score(y_test, y_pred))

print("\nTrain Macro F1:", f1_score(y_train, train_pred, average="macro"))
print("Validation Macro F1:", f1_score(y_val, val_pred, average="macro"))
print("Test Macro F1:", f1_score(y_test, y_pred, average="macro"))


# Evaluate Model and Predict Enrolled Students

print("\nFinal Test Evaluation")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Macro F1:", f1_score(y_test, y_pred, average="macro"))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=TARGET_NAMES
))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=TARGET_NAMES,
    yticklabels=TARGET_NAMES
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("CatBoost Confusion Matrix: Dropout vs Graduate")
plt.show()


# Use the binary model to predict the likely final outcome for currently enrolled students

if len(X_enrolled) > 0:
    enrolled_predictions = enrolled_students.copy()
    enrolled_pred = best_model.predict(X_enrolled).astype(int).ravel()
    enrolled_dropout_probability = best_model.predict_proba(X_enrolled)[:, 1]

    enrolled_predictions["Predicted final outcome"] = pd.Series(enrolled_pred).map(LABEL_MAP).values
    enrolled_predictions["Dropout probability"] = enrolled_dropout_probability
    enrolled_predictions["Graduate probability"] = 1 - enrolled_dropout_probability

    print("\nPredicted final outcomes for currently enrolled students:")
    print(enrolled_predictions["Predicted final outcome"].value_counts())

    print(enrolled_predictions[
        ["Predicted final outcome", "Dropout probability", "Graduate probability"]
    ].head(10))

    enrolled_predictions.to_csv("enrolled_student_predictions.csv", index=False)
    print("\nSaved enrolled-student predictions to enrolled_student_predictions.csv")

else:
    print("\nNo currently enrolled students were found to predict.")
