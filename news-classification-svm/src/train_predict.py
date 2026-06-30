import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.svm import LinearSVC

# Configuration
DEV_PATH = "development.csv"
EVAL_PATH = "evaluation.csv"

SEED = 42
np.random.seed(SEED)

# Model hyperparameters
C_VAL = 0.10
MAX_ITER = 6000

# Trigram block (3,3) settings
W3 = 0.5
MIN_DF_3 = 12
MAX_FEAT_3 = 12000

# Post-hoc calibration parameters
ALPHA = np.array(
    [0.777385, 1.076283, 0.990557, 1.223782, 1.232677, 0.777111, 1.292146], dtype=float
)
BIAS = np.array(
    [-0.078848, -0.057961, 0.005389, 0.050098, -0.063362, -0.072635, -0.063624],
    dtype=float,
)


# Helpers
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[["title", "article"]] = df[["title", "article"]].fillna("")
    df["text"] = (df["title"] + " " + df["article"]).str.strip()
    df["text_len"] = df["text"].str.split().str.len().astype(np.float32)
    return df


def scale_sparse(X, w: float):
    return X.multiply(w)


# Load

df_train = build_features(pd.read_csv(DEV_PATH))
df_eval = build_features(pd.read_csv(EVAL_PATH))

X_train = df_train[["text", "source", "page_rank", "text_len"]]
y_train = df_train["label"].to_numpy()
X_eval = df_eval[["text", "source", "page_rank", "text_len"]]

# Vectorizers
token_pattern = r"(?u)\b[a-zA-Z][a-zA-Z]+\b"

tfidf_12 = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.8,
    max_features=70_000,
    stop_words="english",
    sublinear_tf=True,
    token_pattern=token_pattern,
)

tfidf_33_weighted = Pipeline(
    steps=[
        (
            "tfidf",
            TfidfVectorizer(
                ngram_range=(3, 3),
                min_df=MIN_DF_3,
                max_df=0.8,
                max_features=MAX_FEAT_3,
                stop_words="english",
                sublinear_tf=True,
                token_pattern=token_pattern,
            ),
        ),
        (
            "weight",
            FunctionTransformer(scale_sparse, kw_args={"w": W3}, accept_sparse=True),
        ),
    ]
)

# Preprocess + model
preprocess = ColumnTransformer(
    transformers=[
        ("tfidf_12", tfidf_12, "text"),
        ("tfidf_33w", tfidf_33_weighted, "text"),
        ("source", OneHotEncoder(handle_unknown="ignore"), ["source"]),
        ("num", "passthrough", ["page_rank", "text_len"]),
    ]
)

model = LinearSVC(
    C=C_VAL,
    class_weight="balanced",
    random_state=SEED,
    max_iter=MAX_ITER,
    tol=1e-4,
)

pipe = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("clf", model),
    ]
)

# Train + predict
pipe.fit(X_train, y_train)

# decision_function
scores = pipe.decision_function(X_eval)
pred = np.argmax(scores * ALPHA[None, :] + BIAS[None, :], axis=1)

# Save submission
submission = pd.DataFrame({"Id": df_eval["Id"], "Predicted": pred})
submission.to_csv("submission.csv", index=False)
print("Saved submission.csv")
