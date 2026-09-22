# Customer Churn Prediction — Comparing Four Classification Models

**A complete, reproducible machine-learning study in a single Jupyter Notebook — written
so you can learn the subject from the ground up, even if you have never done machine
learning before.**

| | |
|---|---|
| **Problem** | Binary classification — will a customer churn (leave)? |
| **Data** | Synthetic dataset generated inside the notebook (2,000 rows, 8 features) |
| **Models compared** | Logistic Regression · Random Forest · K-Nearest Neighbours · Support Vector Machine |
| **Notebook** | [`ML_Model_Comparison.ipynb`](ML_Model_Comparison.ipynb) |
| **Runs end-to-end** | ✅ — verified with `nbconvert --execute` (zero errors, zero warnings) |
| **Language / libraries** | Python 3 · scikit-learn · pandas · numpy · matplotlib |

---

## How to read this document

This README is organised in three parts so it works both as a **textbook-style lesson**
and as a **documentation page for the notebook**:

| Part | What it gives you |
|---|---|
| **Part I — Learning from the ground up** | The absolute core: what machine learning, classification, models, and "comparing models" actually mean, explained from zero, with analogies for every concept. |
| **Part II — The actual study** | What the notebook does at every step, why, and what the numbers say. |
| **Part III — Reference** | Glossary, FAQ, how to run the notebook, file layout, reproducibility notes. |

If you are completely new: read Part I fully, then Part II. If you already know the basics,
jump to Part II and use Part III for reference.

---

# PART I — LEARNING FROM THE GROUND UP

## 1. The whole thing in one paragraph

We have **data about customers** — how long they have been with us, how much they pay,
how often they complain, and so on. Some customers eventually **leave** (they "churn").
We want to build a program that, given those customer facts, **predicts in advance** which
customers are likely to leave, so the business can try to keep them. We cannot know the
ideal prediction rule by hand, so we write code that **learns the rule from past data** —
that is *machine learning*.

There are many possible "learning rules" (each one is called a **model**), and they learn
in very different ways. So we don't just pick one at random — we train **four** of the
most famous models, measure how good each one is using agreed **evaluation metrics**, and
then explain **why** they perform differently. That entire process — data, models,
measurement, reasoning — is this study.

---

## 2. The absolute basics

### 2.1 What is machine learning?

**Machine learning (ML)** is a way of writing software that *improves automatically from
data*, instead of following rules that a human writes by hand.

- *Traditional programming:* a human writes rules, the computer applies them to data.
  `Rules + Data → Output`.
- *Machine learning:* given the input data and the correct answers, the computer **finds
  the rules itself**. `Data + Answers → Rules`.

Think of teaching a child to recognise dogs: you do not give them a dictionary definition
("animal with four legs, a tail, and a wet nose"). You *show* them many pictures of dogs
and non-dogs, and after enough examples they can classify new pictures. That is exactly
what a supervised ML model does:

- the **pictures** are the *input data* (called **features**),
- the labels "dog / not dog" are the **answers** (called the **target** or **label**),
- the model's internal "rule" is what the computer learned,
- applying the rule to a *new* picture is called **prediction**.

### 2.2 What is a classification problem?

**Classification** is the ML task of putting each item into one of a fixed set of
**categories** (classes).

- If there are exactly **two** classes (yes/no, churn/retained, spam/not-spam), it is a
  **binary classification** problem — this is what we have here.
- If there are more than two classes (e.g., Dog / Cat / Bird), it is **multiclass
  classification**.
- The sister task, where the answer is a *number* instead of a category (e.g., predict the
  exact dollar value a customer will spend), is called **regression**.

Our problem is **binary classification**: for each customer we must decide
**Churned (1)** or **Retained (0)**.

### 2.3 What does "training a model" mean?

Every model is built around some **adjustable internal numbers** (called *parameters* —
for example the weights in Logistic Regression) or **example data** (as in k-NN).

**Training (fitting)** is the process of adjusting those numbers so that the model's
predictions match the answers we already know. Concretely:

1. We take a batch of examples where we **know the true answer** (the *training data*).
2. The model makes a prediction for each.
3. We quantify how wrong it is (the **loss**).
4. The training algorithm tweaks the model's numbers to reduce that wrongness.

After training, we hope the model has learned a **general pattern** — something that also
works on customers it has never seen before. Testing whether that is true is the entire
point of the *test set* and the evaluation step.

### 2.4 The vocabulary you need (minimal, but exact)

| Word | What it means | In this study |
|---|---|---|
| **Dataset** | A table of examples | a table with 2,000 rows |
| **Instance / row / sample** | One example in the table | one customer |
| **Feature (column)** | One measurable property of an example | `monthly_charges`, `num_support_tickets`, … |
| **Label / target** | The correct answer for a row | `churn` (0 = retained, 1 = churned) |
| **Class** | A category the label can take | Churned / Retained |
| **Training set** | The rows the model learns from | 80 % of the data |
| **Test set** | The rows used to check the model *after* training | the other 20 % |
| **Model** | A learnable prediction rule | the four classifiers |
| **Hyperparameter** | A knob *we* choose before training | `n_estimators=200`, `n_neighbors=5`, … |
| **Prediction** | The model's guess for an unseen row | "this customer will churn" |

### 2.5 The feature-space picture (one image worth a thousand words)

Imagine every customer drawn as a **dot on a graph**, where the axes are the features. With
two features it is a 2-D picture; our data has eight features, so it is an 8-D space — but
the *idea* is the same. Each dot belongs to one of two groups (churned = red, retained =
blue). The points *tend* to cluster: churners mostly live in one region, retained customers
in another, with some overlap near the middle.

**Classification = drawing a boundary that separates the red region from the blue region.**
A model is a particular *recipe* for drawing that boundary. Different models draw different
*kinds* of boundaries — a straight line, a step-shaped set of boxes, a highly wiggly line,
or a smooth curve. Everything about "why models differ" reduces to **what kind of boundary
each model is able to draw**.

### 2.6 What is "model comparison", and why bother?

No single model is universally best — different problems need different tools. Comparing
models is how we choose the right tool for *this* problem:

- Each model makes different **assumptions** about the data (its *bias*, or the "shapes of
  boundary it can draw").
- Each model can be too simple (misses patterns) or too complex (memorises noise) — the
  **bias–variance trade-off**.
- Each model reacts differently to things like **feature scale**, **class imbalance**,
  **noise**, and **redundant features**.

So we run all four models on the **exact same data**, measure each with the **exact same
metrics**, and let the *evidence* (not fashion or guesswork) decide which one to recommend.

---

## 3. The problem in plain language

### 3.1 The business story

A subscription company notices that some customers end their contracts. In business jargon
each lost customer is **churn**. Losing a customer costs real money: lost revenue now,
and possibly negative word of mouth. The company's wish is:

> **Identify, at any moment, which customers are likely to churn soon — so the retention
> team can act (send an offer, fix a complaint) before the customer leaves.**

The customer's "digital footprint" (tenure, bills, usage, complaints, payment behaviour)
contains signals. The ML task is to learn how those signals combine to predict churn.

### 3.2 Why this is a *good* problem to learn on

- The target is a clean yes/no → classic **binary classification**.
- There are many interpretable features → we can explain *why* a model works.
- The data is slightly **imbalanced** (70/30) → it teaches the crucial lesson that
  *accuracy alone is not enough*.
- The data contains **noise and redundancy** → it teaches how models cope with imperfect,
  realistic data.
- Four famous model *families* fit the problem naturally → perfect setup for comparison.

### 3.3 The features, in plain English

| Feature | What it measures | Why a company would collect it |
|---|---|---|
| `account_tenure_months` | how long the customer has been subscribed | loyalty is a strong churn signal |
| `monthly_charges` | the monthly bill amount | cost pressure can push customers away |
| `service_usage_hours` | monthly active-usage hours | falling engagement precedes leaving |
| `num_support_tickets` | how many support tickets they raised | frustration drives churn |
| `payment_reliability` | a payment-punctuality index | unreliable payers churn more |
| `avg_call_duration_mins` | average call duration | **redundant** (built from other features) |
| `data_transfer_gb` | monthly data consumed | **redundant** (built from other features) |
| `email_optin_rate` | engagement with marketing email | **repeated** (a duplicate of an informative feature) |

The last three rows are a deliberate experiment built into the data: real datasets contain
correlated (redundant) and duplicated (repeated) columns, and models react to that clutter
differently. Comparing how the four models cope is part of the lesson.

### 3.4 Class imbalance, explained from scratch

If 70 % of customers never churn, then a *brain-dead* program that always answers
"Retained" is correct **70 % of the time**. That looks impressive if you only look at
accuracy — but it caught **zero** churners, so it is useless.

In this study that single fact drives three decisions you will see repeatedly:

1. **Report several metrics** (precision, recall, F1, ROC-AUC), not just accuracy.
2. **Tell the models imbalance exists** via `class_weight='balanced'`, so they put extra
   effort into the rare, important class (churn).
3. **Split the data carefully** (`stratify`) so both the training and test sets keep the
   same 70/30 ratio.

---

## 4. The workflow, from first principles

The notebook implements the standard ML pipeline. Each step exists to answer one question:

```
(1) Build the data        → "What am I learning from?"        (synthetic, reproducible)
(2) Explore the data      → "What does my data look like?"    (shape, stats, plots)
(3) Prepare the data      → "Is my data ready for models?"    (scaling, imbalance, split)
(4) Train models          → "Can each model learn the rule?"  (4 families, same data)
(5) Evaluate              → "How good is each model really?"  (metrics, matrices, ROC)
(6) Compare               → "Why do they differ?"             (bias, imbalance, structure)
(7) Conclude              → "Which one should I use, and why?"
```

Why the order matters:

- **We never let the models see the test answers during training.** Otherwise "evaluation"
  would be meaningless — the model could memorise the answers. This is why the split
  happens *before* training, and why the test set is touched only once, at the very end.
- **Preprocessing exists so models get a fair chance.** Some models are disturbed by
  features on wildly different scales; standardising the data removes that unfairness
  *before* they learn.
- **Evaluation gives us a common language.** Because all four models are measured with the
  same metrics on the same test set, their scores are directly comparable.

---

## 5. How four different models look at the *same* problem

This is the heart of the study: **four recipes for drawing the decision boundary**, each
with different strengths, weaknesses, and sensitivities. First an analogy, then each model
explained from core, then a table showing exactly where they differ.

### 5.1 The analogies (remember these — they carry the intuition)

| Model | One-line analogy |
|---|---|
| **Logistic Regression** | A careful human analyst who adds up weighted clues from a checklist and converts the total into a probability. They can only reason in straight-line ("linear") ways. |
| **Random Forest** | A committee of many decision-tree experts; each expert alive on a *different subset of the data and features*, asking a chain of yes/no questions; the final answer is the committee's majority vote. |
| **k-Nearest Neighbours (k-NN)** | Deciding by asking your nearest friends: find the k most similar past customers and go with what *they* did. No learning, just remembering. |
| **Support Vector Machine (SVM)** | A bricklayer who builds the *widest possible straight wall* that separates the two groups; with a "kernel", the wall is first bent into a flexible highway so it can curve around tricky layouts. |

### 5.2 Logistic Regression — the linear baseline

**The idea.** For every input feature, the model learns a **weight** (positive = pushes
towards "churn", negative = pushes towards "retained"). It computes a weighted sum of the
features, then squeezes that number through an **S-shaped logistic curve** into a
probability between 0 and 1:

```
probability(Churn) = 1 / (1 + e^(-(w₁·feature₁ + w₂·feature₂ + … + b)))
```

If the probability is above 0.5 we predict Churn, otherwise Retained.

**What boundary can it draw?** A **straight line** (hyperplane in many dimensions). With
any straight-line classifier, there exist layouts it simply *cannot* separate perfectly.
Our synthetic data is designed to be mostly linear, so it does very well — but the
straight line is a real ceiling.

**Strengths**
- Highly **interpretable**: a positive coefficient tells you the feature pushes *towards*
  churn; the bigger the magnitude, the stronger the push.
- Fast to train; regularised by default (guards against overfitting).
- Gives calibrated **probabilities**, perfect for ROC curves.

**Weaknesses**
- Cannot model non-linear interactions on its own.
- Sensitive to **feature scale** (needs standardisation) because weights are learned via
  gradients.

**Hyperparameters used here** (each one justified — see Part II §9):

| Hyperparameter | Value | Reason |
|---|---|---|
| `penalty` | `l2` (default) | ridge regularisation; left at the default schema-wide |
| `C` | `1.0` | inverse regularisation strength — moderate |
| `solver` | `liblinear` | suits this dataset size and supports class weights |
| `max_iter` | `1000` | guarantees convergence |
| `class_weight` | `balanced` | compensates the 70/30 imbalance |
| `random_state` | `42` | reproducibility |

### 5.3 Random Forest — the ensemble of decision trees

**The idea.** A **decision tree** is a chain of if-then questions: "Is tenure > 24? → then
is monthly_charges > 60? → then … classify." One tree alone is unstable (small changes in
data change the tree). A **Random Forest** fixes that by growing **many slightly different
trees** and averaging their votes:

1. Each tree trains on a **random bootstrap sample** of the data (some rows repeated, some
   omitted).
2. At every split, each tree considers only a **random subset of features**.
3. The randomness makes the trees *diverse*; their average is much more stable and accurate
   than any single tree — this recipe is called **bagging**, and the hoping-for effect is
   that the errors of individual trees cancel out.

**What boundary can it draw?** **Piecewise-constant regions**: boxes in feature space,
because every question is of the form "is feature *x* > threshold?"

**Strengths**
- Captures **non-linear interactions** between features.
- **Scale-invariant** — a tree just compares a feature with a threshold, so standardisation
  is unnecessary. This is why Random Forest is the only model here trained *without* a scaler.
- Robust to **redundant / repeated features** (our dataset deliberately includes both).
- Gives **feature importances** for free (used in the analysis).

**Weaknesses**
- You must bound tree depth, or noisy labels (we have 5 % noise) get memorised → overfitting.
- Less interpretable than Logistic Regression (though importances help).
- Slower than Logistic Regression at training.

**Hyperparameters used here:**

| Hyperparameter | Value | Reason |
|---|---|---|
| `n_estimators` | `200` | enough trees for a stable vote |
| `max_depth` | `8` | limits depth → strong anti-overfitting control |
| `min_samples_split` | `5` | no splitting below 5 samples |
| `min_samples_leaf` | `2` | smoother, more generalised leaves |
| `class_weight` | `balanced` | compensates the imbalance |
| `n_jobs` | `-1` | use all CPU cores |
| `random_state` | `42` | reproducibility |

### 5.4 K-Nearest Neighbours — the instance-based model

**The idea.** This model does **no learning at all in the fitting step** — it literally
*stores the training set*. To classify a new customer it:

1. Computes the **distance** (Euclidean, `p=2`) from the new customer to every training
   customer;
2. Finds the **k nearest** (here `k=5`);
3. Predicts the **majority class** among those five neighbours.

**What boundary can it draw?** Given enough points, *any arbitrary boundary* — but it is
computed **locally**, near each query point, and is extremely sensitive to the arrangement
of the training dots.

**Strengths**
- Conceptually the simplest; adapts to any local structure.
- Trivial to train (just storing data).

**Weaknesses**
- **Distance-based**: without feature scaling, a large-magnitude feature (like
  `monthly_charges`) completely swamps small ones — scaling is *critical*.
- **No native class-weight support**: it cannot reweight the minority class, so the 70/30
  imbalance hits it harder than the other three.
- Predictions are slow (must scan the whole training set) and noisy at the boundary;
  choosing k matters (small k overfits noise, large k oversmooths).

**Hyperparameters used here:**

| Hyperparameter | Value | Reason |
|---|---|---|
| `n_neighbors` | `5` | standard compromise between noise and smoothing |
| `weights` | `uniform` | every neighbour votes equally |
| `metric` / `p` | `minkowski` / `2` | Euclidean distance |
| scaler | `StandardScaler` | without it distances are dominated by scale |

### 5.5 Support Vector Machine — the margin maximiser

**The idea.** An SVM wants the **widest possible separating boundary** ("street") between
the two classes. It keeps only the training points that touch the street — the **support
vectors** — and throws the rest away. Because it maximises the *margin*, it tends to stay
robust to outliers.

**What about non-linearity?** With a **kernel** (here the RBF kernel) the SVM *implicitly*
lifts the data into a higher-dimensional space, where the street can be drawn straight,
then maps back to a **smooth curved boundary** in the original space. The RBF kernel has a
width parameter `gamma`; `C` trades off "draw a wide street" against "do not make training
errors".

**Strengths**
- Excellent on **smooth non-linear boundaries** — and it proves the best *ranking* model
  here (highest ROC-AUC).
- Margin-based robustness; supports `class_weight`.
- Decision scores are fine for ROC curves (AUC is unchanged by a monotone rescale), so we
  avoid the deprecated probability-calibration flag.

**Weaknesses**
- Sensitive to the **`C`/`gamma` choices** and, like k-NN, to **feature scale**.
- Less interpretable; slower to train on large datasets.

**Hyperparameters used here:**

| Hyperparameter | Value | Reason |
|---|---|---|
| `kernel` | `rbf` | flexible non-linear boundary |
| `C` | `1.0` | margin vs. training-error trade-off |
| `gamma` | `scale` | auto-set from feature variance |
| `class_weight` | `balanced` | compensates the imbalance |
| `random_state` | `42` | reproducibility |

### 5.6 Where the four models *fundamentally* differ

| Dimension | Logistic Regression | Random Forest | k-NN | SVM (RBF) |
|---|---|---|---|---|
| **Family** | Linear, parametric | Tree ensemble, bagging | Instance-based, non-parametric | Kernel, margin-based |
| **Shape of boundary** | Straight line | Step-shaped boxes | Arbitrary local boundary | Smooth curved surface |
| **What it "remembers"** | Coefficients (weights) | Hundreds of trained trees | The whole training set | Support vectors only |
| **Needs feature scaling?** | Yes | **No** | Yes (critical) | Yes |
| **Models non-linearity?** | No (by itself) | Yes | Yes | Yes |
| **Shares class weights?** | Yes | Yes | **No** | Yes |
| **Interpretability** | High | Medium | Low–medium | Low |
| **Overfitting risk** | Low (regularised) | Medium (depth must be capped) | High if k too small | Medium (C/gamma) |
| **How it reacts to our data** | Solid because the boundary is mostly linear | Best all-round; ignores scale, tolerant of redundancy | Highest precision but lowest recall; imbalance hurts it | Best ranking (ROC-AUC); smooth boundary |

These differences — not chance — drive the results you will see in Part II.

---

## 6. Evaluation metrics — taught from zero, with worked examples

We need to measure "how good is a model?". Start from the raw ingredients.

### 6.1 The confusion matrix — the counting box

For binary classification only **four** things can happen. We write them in a 2×2 table
(the **confusion matrix**):

```
                    ACTUAL      ACTUAL
                    Churned     Retained
PREDICTED Churned     TP          FP
PREDICTED Retained    FN          TN
```

| Cell | Name | Meaning | Business pain |
|---|---|---|---|
| **TP** True Positive | correctly predicted churn | — (good) |
| **TN** True Negative | correctly predicted retention | — (good) |
| **FP** False Positive | predicted churn, actually retained | false alarm: an offer wasted on a loyal customer |
| **FN** False Negative | predicted retained, actually churned | **missed churner: revenue lost for good** |

**Worked mini-example (10 customers):** suppose 3 truly churn and 7 stay. A model flags 4
customers as churners. Of those, 2 are genuine churners (TP = 2) and 2 are loyal (FP = 2).
Of the 3 real churners, 2 were caught so 1 was missed (FN = 1). Among the 7 loyal
customers, 2 were wrongly flagged, so 5 were correctly kept (TN = 5).

```
                 ACTUAL Churned   ACTUAL Retained
PRED Churned         2 (TP)            2 (FP)
PRED Retained        1 (FN)            5 (TN)
```

Every metric below is just arithmetic on these four numbers.

### 6.2 The four core metrics

**Accuracy** — out of *everything* predicted, how much was right?
`Accuracy = (TP + TN) / (TP + TN + FP + FN) = (2+5)/10 = 0.70`

**Precision** — of the *churn flags we raised*, how many were truly churners?
`Precision = TP / (TP + FP) = 2/4 = 0.50`
*(Precision punishes **false positives** — false alarms.)*

**Recall** — of the *real churners out there*, how many did we catch?
`Recall = TP / (TP + FN) = 2/3 ≈ 0.67`
*(Recall punishes **false negatives** — missed churners. Each missed churner is lost revenue.)*

**F1-score** — the harmonic mean; one number that rewards being *good at both*.
`F1 = 2·Precision·Recall / (Precision + Recall) = 2·0.5·0.67 / (0.5+0.67) ≈ 0.57`

**Why F1 instead of the normal average?** The normal average would let a model hide a
terrible score in one metric (e.g., precision 0.99, recall 0.05 → average 0.52 but a model
that catches almost nothing). F1 refuses: `2·0.99·0.05/(0.99+0.05) ≈ 0.095`.

**Notice in the mini-example**: accuracy is 0.70 but the model is mediocre at churn
detection. This is *exactly* the trap this study is designed to teach — which is
why this study never judges models on accuracy alone.

### 6.3 ROC curve and ROC-AUC — the ranking ability

A model can output a *score* (a probability or a margin value), not just a class. If we
raise the decision threshold, we get fewer, more careful predictions; lower it and we catch
more churners but raise false alarms. The **ROC curve** shows this trade for **every
possible threshold**:

- x-axis: **False Positive Rate** = FP/(FP+TN) — how often loyal customers get flagged;
- y-axis: **True Positive Rate** = Recall — how many churners get caught.

**Area Under the Curve (AUC)** collapses the whole curve into one number:

- `AUC = 0.5` → random guessing (the diagonal line);
- `AUC = 1.0` → perfect ranking of every churner above every loyal customer;
- Interpretation: *"pick a random churner and a random loyal customer — how often does the
  model put the churner's score higher?"*

A model can have high accuracy yet only average AUC; AUC measures **discrimination and
ranking**, which for building a "churn-risk score" is exactly what you want. (For the SVM we
use its raw margin score for the ROC curve — AUC is unaffected by rescaling the score, so no
probability calibration is needed.)

### 6.4 Cross-validation — the overfitting lie detector

**Overfitting** is when a model memorises the *noise* in the training data instead of the
*pattern*, so it looks great on training but fails on new data. **Cross-validation** (CV)
catches this *before* we use the test set:

- split the **training set** into 5 folds;
- train on 4 folds, test on the remaining fold; repeat so every fold is tested once;
- average the 5 scores.

If a model's CV-F1 is much worse than its test-F1, be suspicious (unstable result or subtle
leakage). If training-test gaps are small, the model is generalising. In this study the CV
check strongly supports the final choice.

### 6.5 Which metric should you look at?

| If you care about… | Look at… |
|---|---|
| Overall correctness, when classes are balanced | Accuracy |
| Few **false alarms** (don't annoy loyal customers) | Precision |
| Few **missed churners** (don't lose revenue) | **Recall** |
| A balanced single number between the two | F1-score |
| Building a churn-risk **score** and ranking customers | **ROC-AUC** |

For churn prediction, **recall and F1** are the business-critical ones — a missed churner
is expensive in a way a wasted offer is not.

---

# PART II — THE ACTUAL STUDY

## 7. What the notebook does, step by step

| Stage of the workflow | Notebook section | README |
|---|---|---|
| **A. Synthetic dataset** — defined target, explanation, class distribution, random seed | Section 3 | §8 |
| **B. Exploration & preparation** — shape, samples, statistics, preprocessing, train/test split | Section 4 | §9 |
| **C. Four classifiers** — with stated hyperparameters | Section 5 | §5, §10 |
| **D. Evaluation** — Accuracy, Precision, Recall, F1, confusion matrix, ROC-AUC | Section 6 | §6, §11 |
| **E. Comparison & justification** — beyond accuracy, FP/FN, overfitting, sensitivity; justified conclusion | Sections 7–8 | §12, §13 |
| Notebook structure: problem → imports → generation → EDA → preprocessing → models → evaluation → comparison → conclusion | cells 0→29 in order | — |
| Coding conventions: modular, commented, reusable functions, no repeated code, visible outputs | reusable helpers, clean outputs | §8–13 |
| Runs top-to-bottom without errors | verified with `nbconvert --execute` | §15 |

---

## 8. The dataset in detail

### 8.1 Why synthetic?

The study **generates its own data** instead of loading a file:

- **Known ground truth** — we control the true generator, so we know what "correct" looks like;
- **Design control** — we choose how separable, noisy, imbalanced and redundant the data is;
- **Reproducibility** — a fixed `random_state` produces the identical dataset every run;
- **Zero external files** — the data is created inside the notebook.

### 8.2 The generating recipe

`sklearn.datasets.make_classification` draws Gaussian point clouds per class. Its
parameters were chosen deliberately:

| Parameter | Value | Why |
|---|---|---|
| `n_samples` | 2000 | large enough for stable, comparable results |
| `n_features` | 8 | one column per customer attribute |
| `n_informative` | 5 | genuinely predictive columns |
| `n_redundant` | 2 | correlated with the informative ones |
| `n_repeated` | 1 | duplicated signal (robustness test) |
| `n_clusters_per_class` | 1 | a single density cloud per class |
| `class_sep` | 1.0 | moderate separability → realistic difficulty |
| `flip_y` | 0.05 | 5 % of labels randomly flipped → imperfect reality |
| `weights` | `[0.70, 0.30]` | **70 % retained / 30 % churned** |
| `random_state` | 42 | reproducibility |

```python
X, y = make_classification(
    n_samples=2000, n_features=8, n_informative=5, n_redundant=2,
    n_repeated=1, n_clusters_per_class=1, class_sep=1.0,
    flip_y=0.05, weights=[0.70, 0.30], random_state=42,
)
df = pd.DataFrame(X, columns=feature_names)
df["churn"] = y
```

### 8.3 The class distribution

| Class | Count | Share |
|---|---|---|
| Retained (0) | 1,400 | ≈ 70.0 % |
| Churned (1) | 600 | ≈ 30.0 % |

This 70/30 imbalance is the reason the study *must* go beyond accuracy (ref §3.4).

### 8.4 Reading the EDA

The notebook inspects: shape `(2000, 9)`, sample rows, dtypes (all numeric), missing values
(none), duplicates (none), summary statistics, class-conditional feature means, per-feature
histograms split by class, and a correlation heatmap. The heatmap visibly confirms the
built-in redundancy — the `redundant` columns are plainly correlated with the informative
ones.

---

## 9. Preparation — what is done, and *why*

1. **Missing values / duplicates — nothing to do.** Checks came back clean, so no imputation
   or deduplication is needed. That verdict itself is the justification.
2. **Encoding — not applicable.** All eight features are numeric; one-hot or label encoding
   would add nothing, so it is deliberately skipped.
3. **Feature scaling (`StandardScaler`)** — applied to the models whose decisions depend on
   distances or gradients (**LR, k-NN, SVM**), because a large-magnitude feature would
   otherwise dominate. **Random Forest** is scale-invariant (it only thresholds single
   features), so it is trained **without** a scaler — an explicit, justified asymmetry.
4. **Class imbalance** — handled by `class_weight='balanced'` in LR, RF and SVM so the
   minority churn class contributes proportionally more to the loss. k-NN has no native
   class weights and is kept as a (very informative) comparison point.
5. **Stratified 80/20 split** — `train_test_split(stratify=y)` preserves the 70/30 ratio in
   both partitions, so the test set fairly represents the population and every model is
   scored on the *same* 400 customers.

---

## 10. The four trained models (tie-back to Part I)

| # | Model | Family | Why included | Scaling | Class weights |
|---|---|---|---|---|---|
| 1 | Logistic Regression | linear / probabilistic | required interpretable baseline | ✅ | ✅ |
| 2 | Random Forest | tree ensemble | required; non-linear, scale-free | ❌ | ✅ |
| 3 | k-Nearest Neighbours | instance-based | distance view; no global assumptions | ✅ | ❌ |
| 4 | Support Vector Machine | kernel / margin | smooth non-linear boundary; ranking | ✅ | ✅ |

Hyperparameters are spelled out in each model's section of the notebook and in Part I §5.
Every stochastic model uses `random_state = 42`, so the whole study is reproducible.

---

## 11. Results

All four models are evaluated on the **same** stratified 20 % hold-out (400 customers),
with every metric defined in §6.

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC | 5-fold CV F1 |
|---|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9600 | 0.9291 | 0.9440 | 0.9365 | 0.9634 | 0.9132 ± 0.0134 |
| Random Forest | 0.9775 | 0.9833 | 0.9440 | **0.9633** | 0.9645 | 0.9381 ± 0.0141 |
| K-Nearest Neighbours | 0.9775 | **0.9915** | 0.9360 | 0.9630 | 0.9648 | 0.9334 ± 0.0121 |
| Support Vector Machine | **0.9775** | 0.9833 | 0.9440 | **0.9633** | **0.9676** | **0.9401 ± 0.0123** |

**Best model per metric:**

| Metric | Winner |
|---|---|
| Accuracy | Random Forest (tied by three) |
| Precision | K-Nearest Neighbours |
| Recall | Logistic Regression (tied by three) |
| F1-score | Random Forest, SVM |
| ROC-AUC | Support Vector Machine |
| CV-F1 (train) | Support Vector Machine |

**Business view — mistakes on the 400 test customers:**

| Model | False Positives (wasted offers) | False Negatives (missed churners) |
|---|---:|---:|
| Logistic Regression | 9 | 7 |
| Random Forest | 2 | 7 |
| K-Nearest Neighbours | 1 | 8 |
| Support Vector Machine | 2 | 7 |

The notebook also prints each model's **confusion matrix** and overlays the **ROC curves**.

---

## 12. Comparative analysis — why the numbers look like that

**First, the honest headline:** all four models are very strong (accuracy ≥ 0.96, F1 ≥ 0.93).
That is a property of the *data design* — five informative features, moderate separability,
only 5 % noise leave an achievable ceiling near 0.95 that every model reaches. The *analysis*
therefore lives in the **differences**, and every difference traces back to Part I:

1. **Linear backbone.** Logistic Regression is the only linear model. It finishes last on F1
   (0.9365) with the most false positives (9) — yet it still matches the best recall. A model
   this simple being this competitive tells us the underlying boundary is *mostly straight*;
   the small deficit is from the slightly non-linear, noisy regions.
2. **The ensemble wins overall.** Random Forest reaches top accuracy (0.9775) and recall
   (0.9440) while paying only 2 false positives, matches the top F1, needs no scaling, and —
   crucially — its test F1 (0.9633) is close to its 5-fold CV F1 (0.9381) → **no overfitting**,
   thanks to the capped `max_depth=8`.
3. **k-NN shows the cleanest precision–recall trade-off.** It has the *highest precision*
   (0.9915, only 1 false positive) but the *lowest recall* (0.9360, 8 false negatives). This is
   exactly the profile predicted in §5.4: a local voter with no class weighting is conservative
   — rarely cries wolf, but misses some real churners. It is the only model where the 70/30
   imbalance still visibly bites.
4. **SVM is the best *ranker*.** Highest ROC-AUC (0.9676), top F1, 2 false positives, best
   cross-validated F1 — the RBF margin, plus balanced class weights, models this boundary
   beautifully.

| Cause (from Part I) | Effect seen in the results |
|---|---|
| Model bias: straight vs curved vs local boundaries | LR strong-but-last; flexible models cluster at the top |
| Scaling sensitivity | k-NN & co. need the scaler; RF ignores scale entirely |
| Class imbalance handling | weighted models hold recall at 0.944; k-NN drops to 0.936 |
| Precision–recall trade-off | k-NN maximises precision (1 FP), LR lets FPs grow (9) to protect recall |
| Redundant / repeated features | the duplicated `email_optin_rate` column dominates importances; trees and margins absorb redundancy, k-NN's distances are diluted by collinearity |

---

## 13. Conclusion and recommendation

> **For this customer-churn dataset, Random Forest is the most suitable model.**

The justification is *evidence-based*, not "the highest number wins":

1. Joint **best F1-score** (0.9633) and top accuracy (0.9775);
2. **2 false positives and 7 false negatives** — the most business-friendly mistake profile
   (few false alarms, still catches churners);
3. **No overfitting** — test F1 (0.9633) ≈ CV F1 (0.9381);
4. **Scale-free** — simpler preprocessing (no scaler needed);
5. **Robust to redundancy** — handles the deliberately duplicated/correlated columns.

**Runner-ups:** Support Vector Machine is the best *ranking* model (highest ROC-AUC 0.9676);
if the business wants a churn-risk *score* and will tune the decision threshold later, SVM is
a strong choice. **k-NN** wins on precision alone but misses more real churners and offers no
imbalance control. **Logistic Regression** remains the interpretable baseline that confirms the
data's linear backbone.

**The one-line takeaway:** *evaluate with more than accuracy, read the confusion matrices and
ROC curves, and choose Random Forest for this dataset because it combines the best F1, few
false alarms, and overfitting-free, scale-free behaviour.*

---

# PART III — REFERENCE

## 14. Mini glossary

| Term | Meaning |
|---|---|
| **Accuracy** | correctly predicted ÷ all predictions |
| **AUC / ROC-AUC** | probability a random churner outscores a random loyal customer |
| **Bias–variance trade-off** | simple models underfit; complex models overfit; good training balances both |
| **Binary classification** | predicting one of two classes |
| **Class imbalance** | classes have unequal representation (here 70/30) |
| **Confusion matrix** | the TP/TN/FP/FN counting table |
| **Cross-validation** | re-split the training set several times to test stability |
| **Feature** | an input column (e.g., `monthly_charges`) |
| **Feature scaling** | rescaling columns to comparable magnitude (e.g., `StandardScaler`) |
| **F1-score** | harmonic mean of precision and recall |
| **Fit / training** | learning the model parameters from data |
| **Hyperparameter** | a knob set *before* training (e.g., k in k-NN) |
| **Kernel** | a function that lets an SVM separate non-linearly |
| **Label / target** | the correct answer (`churn`) |
| **Overfitting** | memorising noise; great training, poor test |
| **Pipeline** | chaining preprocessing + model into one unit |
| **Precision** | TP ÷ (TP + FP) — trustworthiness of a "churn" flag |
| **Recall** | TP ÷ (TP + FN) — share of real churners caught |
| **Stratified split** | split that preserves class proportions |
| **Test set** | held-out data used once, at the very end |

## 15. FAQ

**Why synthetic data?** So the ground truth is known, the difficulty is controllable, the run
is reproducible, and no external files are needed.

**Why 2,000 rows and 8 features?** Large enough for stable comparison, small enough to run in
seconds on any laptop. 5 informative + 2 redundant + 1 repeated feature lets us study
redundancy handling.

**Why these four models?** Logistic Regression and Random Forest are the common baselines
every practitioner knows; k-NN and SVM add two *completely different* inductive biases (local
distances vs. kernel margins). Together they span the four classic model families that appear
in almost every real ML project.

**Why scale only three of them?** Scaling is needed only by models whose objective or metric
depends on distances/gradients. Trees pick thresholds per feature, so they are invariant to
monotone rescaling.

**Why `class_weight='balanced'`, and why not k-NN?** The churn class is the important minority;
weighting the loss by inverse class frequency makes models treat it fairly. k-NN has no such
knob, which we *use* as a controlled comparison.

**Why a stratified split?** So the test set mirrors the 70/30 population; otherwise evaluation
could be biased by chance.

**Which metric matters most here?** Recall and F1 — the cost of a missed churner exceeds the
cost of a false alarm. Accuracy alone would hide the difference between a good model and a
pointless one.

**Is a higher numeric metric always better?** No. Each metric favours a different behaviour,
and the "best" model must be judged against the problem's cost structure — exactly why the
conclusion is argued from the confusion matrices, not from a single score.

**Can I change the dataset and re-run?** Yes — change the parameters or the seed in Section 3
of the notebook and re-run the cells; every later step adapts automatically because the whole
pipeline is wired together.

## 16. How to run the notebook

```bash
pip install -r requirements.txt

# Option A — Jupyter (interactive)
jupyter notebook ML_Model_Comparison.ipynb      # then: Kernel > Restart & Run All

# Option B — headless (CI-friendly)
jupyter nbconvert --to notebook --execute --inplace ML_Model_Comparison.ipynb
```

Either way you reproduce the exact numbers in §11, because everything is pinned by the single
global `random_state = 42`.

## 17. Repository layout

```
.
├── ML_Model_Comparison.ipynb  # the complete study (single notebook)
├── README.md                  # this document — a self-contained lesson
├── requirements.txt           # Python packages needed to run the notebook
└── .gitignore                 # ignores checkpoints, caches and virtual envs
```

## 18. Reproducibility & trustworthiness

- **One global seed** (`random_state = 42`) controls data generation, the split, and every
  stochastic model — running twice yields byte-identical results.
- **Same 2,000-row dataset, same 400-row test set, same model configuration** for all four
  models ⇒ differences in results reflect the *algorithms*, not random luck.
- **Honest evaluation** — the test set is only ever used once, at the end; tuning/checks run
  on training data via cross-validation.
- **Verified clean run** — the notebook was executed from scratch with `nbconvert --execute`
  (zero errors, zero warnings) before being committed; every cell carries the visible outputs
  needed to follow the analysis.