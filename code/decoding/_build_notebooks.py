"""
Generator for the teaching-focused decoding notebook series.

Run once to (re)create the five .ipynb files in this folder.
Each notebook is built from small markdown + code cells that follow the
same scaffolding style as the beta_series notebooks: heavy markdown,
small code cells, and TODOs instead of finished solutions.
"""

import json
from pathlib import Path

HERE = Path(__file__).parent


def md(text):
    """Make a markdown cell from a triple-quoted string."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.strip("\n").splitlines(keepends=True),
    }


def code(text):
    """Make a code cell from a triple-quoted string."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip("\n").splitlines(keepends=True),
    }


def write_notebook(filename, cells):
    # nbformat 4.5 requires a unique id on every cell
    for i, c in enumerate(cells):
        c["id"] = f"{Path(filename).stem}-{i:03d}"
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = HERE / filename
    with open(out, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"wrote {out}")


# ======================================================================
# 00_prepare_decoder_inputs.ipynb
# ======================================================================

nb00 = []

nb00.append(md(r"""
# 00 — Prepare Decoder Inputs

**Purpose:** Turn the beta-series catalog into *decoder-ready* tables — one table
per decoding question. This is the bookkeeping notebook for the whole decoding
series. No machine learning happens here yet. The goal is to get the **inputs**
into a clean, well-understood shape so that every later notebook can start from a
single CSV.

This notebook follows the same style as the beta-series notebooks: read the
markdown carefully, then fill in the `TODO`s yourself. The code cells are
deliberately incomplete — the point is for *you* to write the pandas.

---

## Notebook overview

You already produced 64 single-trial beta maps for `sub-097` (32 trials × 2 runs)
and catalogued them in:

```
C:\ManzaRotation\Analysis\outputs\sub-097\decoding\tables\sub-097_catalog.csv
```

That catalog has one row per beta map, with columns:

| column | meaning |
|---|---|
| `subject` | always `sub-097` here |
| `task` | which run the trial came from (`modulate1` or `modulate2`) |
| `condition` | the emotion condition (`pos_val`, `neg_val`, `pos_aro`, `neg_aro`) |
| `trial_num` | the trial index within that condition+run |
| `path` | absolute path to the trial's beta `.nii.gz` |

By the end of this notebook you will have written four new tables, each one
describing a single decoding question:

```
decoder_posval_vs_negval.csv      # positive vs negative VALENCE
decoder_posaro_vs_negaro.csv      # positive vs negative AROUSAL
decoder_valence_vs_arousal.csv    # valence trials vs arousal trials
decoder_4class.csv                # all four conditions at once
```
"""))

nb00.append(md(r"""
## What you are learning

- What a **decoder input table** is and why each row is a single trial.
- The four words that describe *every* supervised-learning problem:
  **samples**, **features**, **labels (`y`)**, and **groups**.
- Why labels are stored *separately* from the images.
- Why **groups** are needed for honest cross-validation.
- The difference between **binary** and **multiclass** decoding.
- How the *scientific question* you ask determines the labels you build.

## Why this matters scientifically

A decoder can only answer the question you encode in its labels. If you label
trials by valence, a high accuracy means "valence is represented in the brain
pattern." If you relabel the *same* images by arousal, you are now asking a
completely different question. Getting the input tables right is therefore not
clerical busywork — **the labels *are* the hypothesis.** Sloppy tables silently
change what your "result" means.
"""))

nb00.append(md(r"""
---
## Section 1 — Configuration and Load the Catalog

As in the beta-series notebooks, define everything from `subject` so the paths
stay general. The catalog lives in the `decoding/tables` output folder.

Reading a CSV with pandas:

```python
catalog = pd.read_csv(catalog_path)
```

**TODO:**
- [ ] Set `subject`.
- [ ] Build `decoding_dir` and `tables_dir` from `project_dir`.
- [ ] Read `sub-097_catalog.csv` into a DataFrame called `catalog`.
- [ ] Print `catalog.shape` (expect `(64, 5)`).
"""))

nb00.append(code(r"""
from pathlib import Path
import pandas as pd

subject = "sub-097"

project_dir  = Path(r"C:\ManzaRotation")
decoding_dir = project_dir / "Analysis" / "outputs" / subject / "decoding"
tables_dir   = decoding_dir / "tables"

catalog_path = tables_dir / f"{subject}_catalog.csv"

# TODO: read the catalog CSV into a DataFrame called `catalog`
# catalog = ...

# TODO: print catalog.shape  -> expect (64, 5)
# TODO: display the first few rows with catalog.head()
"""))

nb00.append(md(r"""
---
## Section 2 — Inspect the Counts

Before building anything, *look* at what you have. Two pandas tools do almost all
of the work when inspecting categorical columns:

- `series.value_counts()` — counts how many rows have each value in one column.
- `df.groupby([...]).size()` — counts rows for each combination of columns.

For example:

```python
catalog["condition"].value_counts()          # how many trials per condition?
catalog.groupby(["task", "condition"]).size() # per run AND condition
```

**Sanity check you are aiming for:** 4 conditions × 8 trials × 2 runs = 64 rows,
so each condition should have 16 trials total (8 per run).

**TODO:**
- [ ] Print `value_counts()` for the `condition` column.
- [ ] Print `value_counts()` for the `task` column.
- [ ] Print the `groupby(["task", "condition"]).size()` table.
- [ ] Confirm the numbers match the sanity check above.
"""))

nb00.append(code(r"""
# TODO: how many trials of each condition?
# catalog["condition"]...

# TODO: how many trials per run?
# catalog["task"]...

# TODO: per run AND condition (should be 8 everywhere)
# catalog.groupby(...)...
"""))

nb00.append(md(r"""
---
## Section 3 — The Four Words: samples, features, labels, groups

Every supervised-learning problem — including fMRI decoding — is described by
four things. Learn these words now; they appear in every later notebook.

**Samples** — the *things you classify*. Here, **one sample = one trial's beta
map**. You have 64 samples. In scikit-learn the samples are the rows of `X`.

**Features** — the *numbers describing each sample*. Here the features are the
**voxel values** inside a brain mask. A beta map has tens of thousands of voxels,
so each sample is a very long vector. `X` has shape `(n_samples, n_features)` —
think *(trials, voxels)*. (Nilearn will build this matrix for you from the
images; you rarely materialise it by hand.)

**Labels (`y`)** — the *answer* for each sample: the category you want the
decoder to predict. `y` is a 1-D array with one entry per sample, e.g.
`["pos_val", "pos_val", "neg_val", ...]`.

**Groups** — a label that says *which trials are not independent of each other*.
Trials from the same scanner run share noise (same head position, same drift,
same day). For honest evaluation you must never train and test on the same run.
The `groups` array (here, the `task` column) lets cross-validation keep whole
runs together. More on this in Section 4.

A useful mental picture of what the decoder eventually sees:

```
            feature_1  feature_2  ...  feature_50000     y          group
trial 1   [  0.21      -0.04     ...     0.88     ]   pos_val    modulate1
trial 2   [ -0.10       0.33     ...     0.05     ]   pos_val    modulate1
...
trial 64  [  0.42       0.19     ...    -0.27     ]   neg_aro    modulate2
```

The images supply the left block (features). Your **table** supplies the right
two columns (`y` and `group`). That is the whole reason this notebook exists.
"""))

nb00.append(md(r"""
---
## Section 4 — Why Labels and Groups Live in a Table, Not in the Images

A beta map is just a brain volume of numbers — it does **not** carry its own
condition label or run identity. Those facts live *outside* the image, in your
catalog. Keeping them in a table (rather than, say, only in the filename) has
three big payoffs:

1. **Separation of concerns.** Images are features; the table is metadata. You
   can relabel the *same* images for a different question (Section 7) without
   touching a single `.nii.gz`.
2. **Alignment.** As long as you build `imgs`, `y`, and `groups` by iterating the
   **same DataFrame in the same order**, sample *i* of `X` always matches entry
   *i* of `y` and entry *i* of `groups`. Misalignment here is the single most
   common decoding bug.
3. **Honest cross-validation.** With only two runs, the natural validation scheme
   is **leave-one-run-out**: train on `modulate1`, test on `modulate2`, then swap.
   The `groups` column is what makes that possible.

**TODO (no code — answer in the markdown cell below):**
- [ ] In your own words, what would go wrong if you shuffled `y` but not `imgs`?
- [ ] Why is it *not* safe to put trials from the same run in both train and test?
"""))

nb00.append(md(r"""
*Your answers:*

> (write here)
"""))

nb00.append(md(r"""
---
## Section 5 — Binary vs Multiclass Decoding

The number of distinct values in `y` decides what *kind* of problem you have.

- **Binary** — exactly two classes. The decoder draws one boundary. Chance
  performance for a balanced binary problem is **50%**.
- **Multiclass** — three or more classes. The decoder must separate several
  categories at once; this is harder, and chance is **1 / (number of classes)**.
  For the 4-class problem here, chance is **25%**.

"Chance level" is the accuracy you would expect from a decoder that has learned
nothing — it is the bar your real accuracy must clear to be meaningful. Always
write it down *before* you look at your score, so you cannot fool yourself.

This series builds three binary tables and one multiclass table:

| table | classes | chance |
|---|---|---|
| `decoder_posval_vs_negval` | pos_val, neg_val | 50% |
| `decoder_posaro_vs_negaro` | pos_aro, neg_aro | 50% |
| `decoder_valence_vs_arousal` | valence, arousal | 50% |
| `decoder_4class` | all four conditions | 25% |
"""))

nb00.append(md(r"""
---
## Section 6 — Build the First Subset: `pos_val` vs `neg_val`

Now the pandas. The first decoding question is *positive vs negative valence*, so
you want only the rows whose `condition` is `pos_val` or `neg_val`.

**Boolean filtering** is the tool. `series.isin([...])` returns a True/False mask
that you use to select rows:

```python
mask   = catalog["condition"].isin(["pos_val", "neg_val"])
subset = catalog[mask].copy()   # .copy() avoids the SettingWithCopyWarning later
```

Then add the two columns the decoder cares about. Storing them explicitly (even
when `label` just duplicates `condition`) keeps every decoder table in the *same
shape*, which makes notebooks 01–04 interchangeable:

```python
subset["label"]  = subset["condition"]   # what we predict (y)
subset["groups"] = subset["task"]        # run identity for CV
```

**TODO:**
- [ ] Build the boolean mask for `pos_val`/`neg_val` and select those rows.
- [ ] Add a `label` column equal to `condition`.
- [ ] Add a `groups` column equal to `task`.
- [ ] Print `subset["label"].value_counts()` — expect 16 and 16.
- [ ] Print `subset.groupby(["groups", "label"]).size()` to confirm balance per run.
"""))

nb00.append(code(r"""
# TODO: filter catalog to pos_val / neg_val rows
# posval_negval = ...

# TODO: add `label` and `groups` columns

# TODO: sanity-check the class balance (overall and per run)
"""))

nb00.append(md(r"""
**Sanity checks to insist on before saving any decoder table:**

- The two classes are **balanced** (16 vs 16). Imbalance inflates accuracy and
  makes 50% the wrong chance level.
- Each class appears in **both** runs. If a class lived in only one run, a
  leave-one-run-out decoder could "decode" it by simply memorising the run.
"""))

nb00.append(md(r"""
---
## Section 7 — Save the Table (and a Reusable Helper)

You will repeat the Section 6 pattern four times, so it is worth wrapping it in a
small helper function. Writing the helper *yourself* is part of the exercise —
the skeleton and the docstring are given, the body is yours.

```python
def make_decoder_table(catalog, conditions, label_from="condition"):
    '''Return a decoder-ready subset of `catalog`.

    Parameters
    ----------
    conditions : list[str]
        Which `condition` values to keep.
    label_from : str
        Name of the column to copy into the new `label` column.
    '''
    # TODO: 1. filter rows whose condition is in `conditions`  (.isin)
    # TODO: 2. .copy() the result
    # TODO: 3. add `label`  = subset[label_from]
    # TODO: 4. add `groups` = subset["task"]
    # TODO: 5. return the subset
    ...
```

Saving a DataFrame to CSV (no index column):

```python
subset.to_csv(tables_dir / "decoder_posval_vs_negval.csv", index=False)
```

**TODO:**
- [ ] Implement `make_decoder_table`.
- [ ] Use it to rebuild the `pos_val` vs `neg_val` table.
- [ ] Save it as `decoder_posval_vs_negval.csv`.
"""))

nb00.append(code(r"""
def make_decoder_table(catalog, conditions, label_from="condition"):
    '''Return a decoder-ready subset of the catalog (see markdown above).'''
    # TODO: implement using .isin(), .copy(), and new label/groups columns
    pass


# TODO: posval_negval = make_decoder_table(catalog, ["pos_val", "neg_val"])
# TODO: posval_negval.to_csv(tables_dir / "decoder_posval_vs_negval.csv", index=False)
"""))

nb00.append(md(r"""
---
## Section 8 — The Arousal Table: `pos_aro` vs `neg_aro`

Exactly the same shape of problem, different conditions. This is the table
notebook 02 will consume.

**TODO:**
- [ ] Call `make_decoder_table` with `["pos_aro", "neg_aro"]`.
- [ ] Sanity-check balance (16 vs 16, present in both runs).
- [ ] Save as `decoder_posaro_vs_negaro.csv`.
"""))

nb00.append(code(r"""
# TODO: build and save the pos_aro vs neg_aro table
"""))

nb00.append(md(r"""
---
## Section 9 — A Genuinely Different Question: Valence vs Arousal

The previous two tables split *within* a dimension (positive vs negative). This
table splits *between* dimensions: every valence trial (`pos_val` + `neg_val`)
against every arousal trial (`pos_aro` + `neg_aro`).

Notice that the **labels no longer match the conditions** — you are *collapsing*
four conditions into two new labels. This is your first taste of the idea that
"the label is the hypothesis": the same 64 images, regrouped, ask whether the
brain distinguishes *which dimension was being modulated* at all.

A clean way to derive the new label is a mapping dictionary plus `Series.map`:

```python
to_dimension = {
    "pos_val": "valence",
    "neg_val": "valence",
    "pos_aro": "arousal",
    "neg_aro": "arousal",
}
catalog["dimension"] = catalog["condition"].map(to_dimension)
```

**TODO:**
- [ ] Build the `to_dimension` mapping and add a `dimension` column to `catalog`.
- [ ] Build a decoder table whose `label` is `dimension` (all 64 rows are kept).
      *Hint:* you can pass `label_from="dimension"` and a `conditions` list of all
      four conditions, or filter differently — your choice.
- [ ] Check balance: 32 `valence` vs 32 `arousal`, present in both runs.
- [ ] Save as `decoder_valence_vs_arousal.csv`.
"""))

nb00.append(code(r"""
# TODO: add a `dimension` column via a mapping dict + .map()

# TODO: build the valence-vs-arousal decoder table (label_from="dimension")

# TODO: sanity-check balance, then save decoder_valence_vs_arousal.csv
"""))

nb00.append(md(r"""
---
## Section 10 — The Multiclass Table: All Four Conditions

The last table keeps every trial and uses the original four conditions as the
label. Notebook 04 will use it for 4-class decoding (chance = 25%).

**TODO:**
- [ ] Build a decoder table containing all four conditions, `label` = `condition`.
- [ ] Confirm `label.value_counts()` shows 16 for each of the four classes.
- [ ] Save as `decoder_4class.csv`.
"""))

nb00.append(code(r"""
# TODO: build and save the 4-class decoder table
"""))

nb00.append(md(r"""
---
## Section 11 — Final Check: List What You Built

End every preparation notebook by confirming the artifacts exist on disk.

**TODO:**
- [ ] Glob `tables_dir` for `decoder_*.csv` and print each filename.
- [ ] For each, re-read it and print its shape + `label.value_counts()`.
- [ ] Confirm: two binary tables of 32 rows, one binary table of 64 rows, one
      4-class table of 64 rows.

**What comes next:** Notebook 01 takes `decoder_posval_vs_negval.csv`, turns the
`path` column into images (`X`), the `label` column into `y`, the `groups` column
into cross-validation folds, and trains your first SVM decoder.
"""))

nb00.append(code(r"""
# TODO: list the decoder_*.csv files and print a one-line summary of each
"""))

write_notebook("00_prepare_decoder_inputs.ipynb", nb00)


# ======================================================================
# 01_binary_decoder.ipynb
# ======================================================================

nb01 = []

nb01.append(md(r"""
# 01 — A First Binary Decoder (pos_val vs neg_val)

**Purpose:** Build your first brain decoder — a support vector machine (SVM) that
tries to tell, from a single-trial beta map, whether the trial was a
**positive-valence** or **negative-valence** emotional experience.

This notebook leans on Nilearn's high-level `Decoder` object, which bundles
masking, feature selection, the classifier, and cross-validation into one tool.
We build it up slowly so you understand *every* piece, rather than treating it as
a black box.

> Reference: this notebook mirrors the structure of Nilearn's decoding tutorials
> (the Haxby "Decoding with ANOVA + SVM" example). If you want a second
> explanation of any step, that example is the canonical source.

---

## Notebook overview

1. Load `decoder_posval_vs_negval.csv`.
2. Turn the table into the three things a decoder needs: `imgs` (the images),
   `y` (the labels), and `groups` (the run identity for cross-validation).
3. Build a Nilearn `Decoder` with a brain mask and ANOVA feature selection.
4. Run leave-one-run-out cross-validation.
5. Read the scores against the correct chance level (50%).

You will **not** fully implement the decoder here. The `TODO`s leave the key
choices — the mask, the screening percentile, the fit call, the score
extraction — for you to write.
"""))

nb01.append(md(r"""
## What you are learning

- What **classification** is, in one sentence.
- How beta maps become **samples** and voxels become **features**.
- Why decoding needs a **mask**.
- What **ANOVA feature selection** does and why it must live *inside*
  cross-validation.
- What **data leakage** is and how it fakes good results.
- What **chance level** is and how to compare against it.
- A working intuition for how a linear **SVM** separates two classes.

## Why this matters scientifically

A univariate contrast (notebook 04 of the beta series) asks "does this *one*
voxel respond differently to positive vs negative valence?" A decoder asks a
richer question: "is there *any pattern across many voxels* that distinguishes
the two?" Above-chance decoding is evidence that valence is represented in the
*multivariate* structure of the BOLD response, even when no single voxel is
individually convincing. That is the core idea of MVPA (multi-voxel pattern
analysis).
"""))

nb01.append(md(r"""
---
## Section 1 — What "Classification" Means

**Classification** = learning a rule that maps an input vector to one of a fixed
set of categories. Training shows the algorithm many `(X_row, y_label)` pairs; it
adjusts an internal boundary so that, for new unseen rows, it predicts the right
label.

In our case:

- input vector `X_row` = one trial's voxel pattern (tens of thousands of numbers)
- category `y_label` = `"pos_val"` or `"neg_val"`

The decoder never sees brain anatomy or your hypothesis — only numbers and the
labels you attached. Everything scientific about the result comes from *how you
set up `X`, `y`, and the validation*. That is why we spend most of this notebook
on setup and only one line on the actual fit.
"""))

nb01.append(md(r"""
---
## Section 2 — Configuration and Load the Table

Load the decoder table you built in notebook 00. From it you will derive three
aligned objects, all by iterating the **same DataFrame in the same order**:

- `imgs`   — a list of image paths (or loaded images) → becomes `X`
- `y`      — the `label` column → the answers
- `groups` — the `groups` column → run identity for cross-validation

Reading the table:

```python
table = pd.read_csv(tables_dir / "decoder_posval_vs_negval.csv")
```

**TODO:**
- [ ] Set `subject` and build `tables_dir`.
- [ ] Read `decoder_posval_vs_negval.csv` into `table`.
- [ ] Print `table.shape` (expect `(32, 7)`) and `table["label"].value_counts()`.
"""))

nb01.append(code(r"""
from pathlib import Path
import pandas as pd

subject = "sub-097"

project_dir  = Path(r"C:\ManzaRotation")
decoding_dir = project_dir / "Analysis" / "outputs" / subject / "decoding"
tables_dir   = decoding_dir / "tables"

# TODO: read decoder_posval_vs_negval.csv into `table`
# table = ...

# TODO: print table.shape and table["label"].value_counts()
"""))

nb01.append(md(r"""
---
## Section 3 — Build `imgs`, `y`, and `groups`

The Nilearn `Decoder` can take a **list of image file paths** directly — it will
load and mask them for you. So `imgs` can be as simple as the `path` column
turned into a list.

```python
imgs   = table["path"].tolist()      # list of 32 file paths -> X
y      = table["label"].tolist()     # list of 32 labels      -> answers
groups = table["groups"].tolist()    # list of 32 run names    -> CV folds
```

**Why `.tolist()`?** Nilearn and scikit-learn happily accept plain Python lists or
NumPy arrays; converting once now avoids pandas-index surprises later. The
critical invariant is **alignment**: `imgs[i]`, `y[i]`, and `groups[i]` must all
describe the same trial. Because they come from the same rows in the same order,
they are aligned by construction — do not sort or filter one without the others.

**TODO:**
- [ ] Build `imgs`, `y`, and `groups` from the three columns.
- [ ] Assert all three have the same length (`len(imgs) == len(y) == len(groups)`).
- [ ] Print the first 3 entries of each and eyeball that row 0 lines up.
"""))

nb01.append(code(r"""
# TODO: imgs   = table["path"]...
# TODO: y      = table["label"]...
# TODO: groups = table["groups"]...

# TODO: assert len(imgs) == len(y) == len(groups)
# TODO: print the first few of each to confirm alignment
"""))

nb01.append(md(r"""
---
## Section 4 — Why We Need a Mask

A beta map is a 3-D box of voxels, but a lot of that box is *outside the brain*
(skull, air, ventricic edges). Those voxels are noise. Two reasons a **mask**
matters:

1. **It defines the features.** The mask selects which voxels become columns of
   `X`. Out-of-brain voxels carry no signal but would add thousands of noisy
   features, making the decoder slower and easier to overfit.
2. **It fixes the feature space.** Every trial is masked with the *same* mask, so
   feature *j* means "the same voxel" across all samples. Without a common mask,
   the columns of `X` would not be comparable.

You already have a brain mask from the beta-series work:

```
C:\ManzaRotation\Derivatives\sub-097\func\
    sub-097_task-modulate1_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz
```

A whole-brain mask is the simplest choice and the right one for a first pass.
Later you might swap in an anatomical ROI (e.g. an emotion-related region) to ask
a more targeted question — but that is a different scientific claim, so start
whole-brain.

**TODO:**
- [ ] Define `mask_path` pointing at the brain mask above.
- [ ] Confirm it exists with `mask_path.exists()`.
"""))

nb01.append(code(r"""
# TODO: point mask_path at the sub-097 brain mask in Derivatives/.../func
# func_deriv_dir = project_dir / "Derivatives" / subject / "func"
# mask_path = func_deriv_dir / f"{subject}_task-modulate1_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz"

# TODO: print mask_path.exists()
"""))

nb01.append(md(r"""
---
## Section 5 — ANOVA Feature Selection (and Why It Must Be *Inside* CV)

Even inside the brain mask you have ~50,000 voxels but only 32 samples. With far
more features than samples, a classifier can find a boundary that *perfectly*
fits the training data by exploiting noise — it **overfits**. Feature selection
fights this by keeping only the voxels that look most informative.

**ANOVA feature selection** scores each voxel by how differently its values
distribute across the two classes (an F-test), then keeps the top *k%* —
controlled by Nilearn's `screening_percentile`. A smaller percentile keeps fewer,
more selective voxels.

Here is the subtle, crucial part — **the rule you must never break:**

> Feature selection has to be re-computed on the **training data of each CV fold
> only**, never on the full dataset.

Why? Choosing voxels using *all* the data — including the trials you will later
test on — lets information from the test set leak into training. The decoder then
looks better than it really is. This is **data leakage**, and it is the most
common way fMRI decoding results turn out to be illusory.

The good news: **Nilearn's `Decoder` does this correctly for you.** When you pass
`screening_percentile`, the ANOVA is refit inside every fold automatically. You
get leak-free feature selection *for free* — as long as you let the `Decoder`
run the cross-validation, rather than selecting features by hand beforehand.

**TODO (no code — answer below):**
- [ ] In one sentence, what is data leakage?
- [ ] Why does selecting voxels on the full dataset count as leakage?
"""))

nb01.append(md(r"""
*Your answers:*

> (write here)
"""))

nb01.append(md(r"""
---
## Section 6 — Chance Level and the SVM Intuition

**Chance level.** This is a *balanced binary* problem (16 vs 16), so a decoder
that guesses randomly scores **50%**. Write that number down now. A result is
only interesting relative to chance — "70% accuracy" means nothing until you know
the bar was 50%.

**SVM intuition.** A linear support vector machine looks for the flat boundary (a
hyperplane) that separates the two classes with the widest possible *margin* — the
biggest gap between the boundary and the nearest trials of each class. The trials
sitting right on the margin are the "support vectors"; they alone define the
boundary. A wide margin tends to generalise better to new trials. For
high-dimensional fMRI patterns, a *linear* SVM is the standard, well-behaved
default — don't reach for fancier kernels first.
"""))

nb01.append(md(r"""
---
## Section 7 — Build the Nilearn `Decoder`

`nilearn.decoding.Decoder` wraps the whole pipeline: masking → ANOVA screening →
SVM → cross-validation. The arguments you care about now:

| argument | meaning |
|---|---|
| `estimator` | which classifier; `"svc"` is a linear SVM |
| `mask` | the brain mask image/path that defines the features |
| `standardize` | z-score each voxel before fitting (recommended) |
| `screening_percentile` | the ANOVA % of voxels to keep (refit per fold) |
| `scoring` | metric to report, e.g. `"accuracy"` |
| `cv` | the cross-validation splitter |

For cross-validation, use **leave-one-run-out** via scikit-learn's
`LeaveOneGroupOut`, which respects the `groups` array:

```python
from sklearn.model_selection import LeaveOneGroupOut
cv = LeaveOneGroupOut()
```

With two runs this yields exactly two folds: train on `modulate1` / test on
`modulate2`, and the reverse. Skeleton:

```python
from nilearn.decoding import Decoder

decoder = Decoder(
    estimator="svc",
    mask=mask_path,
    standardize="zscore_sample",
    screening_percentile=...,   # TODO: pick a starting value (e.g. 5)
    scoring="accuracy",
    cv=cv,
)
```

**TODO:**
- [ ] Import `Decoder` and `LeaveOneGroupOut`.
- [ ] Create `cv = LeaveOneGroupOut()`.
- [ ] Choose a starting `screening_percentile` (5 is a reasonable first guess).
- [ ] Construct the `Decoder` with the mask and `estimator="svc"`.
"""))

nb01.append(code(r"""
from nilearn.decoding import Decoder
from sklearn.model_selection import LeaveOneGroupOut

# TODO: cv = LeaveOneGroupOut()

# TODO: choose screening_percentile (start small, e.g. 5)

# TODO: build the Decoder
# decoder = Decoder(
#     estimator="svc",
#     mask=...,
#     standardize="zscore_sample",
#     screening_percentile=...,
#     scoring="accuracy",
#     cv=cv,
# )
"""))

nb01.append(md(r"""
---
## Section 8 — Fit the Decoder

`decoder.fit(imgs, y, groups=groups)` does a lot at once: it masks every image to
build `X`, then for **each CV fold** it z-scores, runs ANOVA screening on the
*training* trials, fits the SVM, and predicts the held-out run. Passing `groups`
is what tells `LeaveOneGroupOut` where the run boundaries are — forget it and the
split is meaningless.

```python
decoder.fit(imgs, y, groups=groups)
```

This will take a little while (it is loading and masking 32 volumes and fitting
two SVMs). That is expected.

**TODO:**
- [ ] Call `decoder.fit(imgs, y, groups=groups)`.
- [ ] After it returns, print `decoder.cv_scores_` to see per-fold accuracy.
"""))

nb01.append(code(r"""
# TODO: decoder.fit(imgs, y, groups=groups)

# TODO: print decoder.cv_scores_
"""))

nb01.append(md(r"""
---
## Section 9 — Evaluate the Performance

`decoder.cv_scores_` is a dictionary keyed by class label; each value is a list
with one score per CV fold. To summarise, average the per-fold scores.

```python
import numpy as np

# cv_scores_ is keyed by class; for a balanced binary problem the per-fold
# accuracies are shared, so averaging any class's list gives the mean accuracy.
for label, scores in decoder.cv_scores_.items():
    print(label, np.mean(scores), scores)
```

**Interpret carefully:**

- Compare the mean to **chance = 50%**, not to 0.
- Two folds is a *very* small number of estimates — expect the two fold scores to
  bounce around. A single run-pair on one subject is noisy; do not over-read a
  few points above or below 50%.
- "Above chance" here is suggestive, not proof. Real studies repeat this across
  many subjects and test the *group* against chance.

**TODO:**
- [ ] Compute and print the mean cross-validated accuracy.
- [ ] State, in a markdown cell, how it compares to the 50% chance level.
- [ ] Note whether the two folds agree or disagree, and why that matters.
"""))

nb01.append(code(r"""
import numpy as np

# TODO: average decoder.cv_scores_ and print mean accuracy per class / overall
"""))

nb01.append(md(r"""
*Your interpretation (mean accuracy vs 50% chance, fold agreement):*

> (write here)
"""))

nb01.append(md(r"""
---
## Section 10 — (Optional) Look at the Decoder Weight Map

A fitted `Decoder` exposes a weight image per class in `decoder.coef_img_` — a
brain map of how much each voxel contributed to the decision. Plotting it shows
*where* in the brain the discriminating pattern lives.

**A caution that matters scientifically:** SVM weights are **not** the same as a
univariate activation map. A voxel can have a large weight because it helps
*cancel noise* in another voxel, not because it is individually "active." Treat
the weight map as "what the model used," not "where valence is." (Nilearn's
tutorials discuss this; for formal interpretation, look up Haufe et al., 2014.)

**TODO (optional):**
- [ ] Plot `decoder.coef_img_[<class label>]` with `plot_stat_map`.
- [ ] Save it under `decoding/figures/`.

## Summary & what comes next

You built a full leak-free decoding pipeline for one binary question. Notebook 02
reuses *this exact pipeline* with a one-line change — the input table — to ask
the arousal question, so you can compare the two.
"""))

nb01.append(code(r"""
# TODO (optional): plot and save the decoder weight map from decoder.coef_img_
"""))

write_notebook("01_binary_decoder.ipynb", nb01)


# ======================================================================
# 02_binary_decoder_arousal.ipynb
# ======================================================================

nb02 = []

nb02.append(md(r"""
# 02 — Binary Decoder, Arousal (pos_aro vs neg_aro)

**Purpose:** Reuse the *exact* pipeline from notebook 01 to ask a different
question — can we decode **positive vs negative arousal** from single-trial beta
maps? — and then compare the result to the valence decoder.

This notebook is deliberately **short**. The whole point is that once the
pipeline is right, changing the scientific question is mostly changing *which
table you load*. If you find yourself rewriting lots of code here, stop and go
copy the working pieces from notebook 01.

---

## Notebook overview

1. Load `decoder_posaro_vs_negaro.csv` (the only real change from notebook 01).
2. Rebuild `imgs`, `y`, `groups`.
3. Reuse the same `Decoder` + `LeaveOneGroupOut` setup.
4. Fit, score, and **compare to the valence decoder**.

## What you are learning

- That a decoding *pipeline* is reusable across questions — only the **labels**
  (here, via a different input table) change.
- How to compare two decoders' accuracies meaningfully.

## Why this matters scientifically

Valence and arousal are the two classic axes of affect. Decoding each separately,
with an identical pipeline, lets you ask: *is one dimension more strongly
represented in this subject's brain patterns than the other?* Holding the
pipeline fixed is what makes that comparison fair — any accuracy difference is
about the brain, not about analysis choices.
"""))

nb02.append(md(r"""
---
## Section 1 — Load the Arousal Table

Same loading pattern as notebook 01, pointed at the arousal table.

**TODO:**
- [ ] Set `subject`, build `tables_dir`.
- [ ] Read `decoder_posaro_vs_negaro.csv` into `table`.
- [ ] Confirm `table["label"].value_counts()` is 16 / 16 across both runs.
"""))

nb02.append(code(r"""
from pathlib import Path
import pandas as pd
import numpy as np

subject = "sub-097"
project_dir  = Path(r"C:\ManzaRotation")
decoding_dir = project_dir / "Analysis" / "outputs" / subject / "decoding"
tables_dir   = decoding_dir / "tables"

# TODO: read decoder_posaro_vs_negaro.csv into `table`
# TODO: check the class balance
"""))

nb02.append(md(r"""
---
## Section 2 — Rebuild `imgs`, `y`, `groups`

Identical to notebook 01 — the columns have the same names, so the same three
lines work. (If you wrote a helper, this is where reuse pays off.)

**TODO:**
- [ ] Build `imgs`, `y`, `groups` from `table`.
- [ ] Assert equal lengths.
"""))

nb02.append(code(r"""
# TODO: imgs / y / groups from the table columns
# TODO: assert len(imgs) == len(y) == len(groups)
"""))

nb02.append(md(r"""
---
## Section 3 — Reuse the Same Decoder Setup

Nothing about masking, feature selection, the SVM, or cross-validation changes
when the question changes — so reuse the same configuration. Keeping
`screening_percentile`, `estimator`, `mask`, and `cv` **identical** to notebook
01 is what makes the comparison in Section 5 fair.

**TODO:**
- [ ] Point `mask_path` at the same brain mask as notebook 01.
- [ ] Build `cv = LeaveOneGroupOut()`.
- [ ] Build a `Decoder` with the **same** settings you used in notebook 01.
"""))

nb02.append(code(r"""
from nilearn.decoding import Decoder
from sklearn.model_selection import LeaveOneGroupOut

# func_deriv_dir = project_dir / "Derivatives" / subject / "func"
# TODO: mask_path = ...  (same mask as notebook 01)

# TODO: cv = LeaveOneGroupOut()
# TODO: decoder = Decoder(estimator="svc", mask=mask_path, standardize="zscore_sample",
#                         screening_percentile=..., scoring="accuracy", cv=cv)
"""))

nb02.append(md(r"""
---
## Section 4 — Fit and Score

**TODO:**
- [ ] `decoder.fit(imgs, y, groups=groups)`.
- [ ] Compute the mean cross-validated accuracy.
- [ ] Store it in a variable like `arousal_accuracy` so you can compare it next.
"""))

nb02.append(code(r"""
# TODO: fit the decoder
# TODO: arousal_accuracy = mean of decoder.cv_scores_
# TODO: print arousal_accuracy and the per-fold scores
"""))

nb02.append(md(r"""
---
## Section 5 — Compare to the Valence Decoder

Now put the two results side by side. Chance is **50%** for both, so compare each
to 50% and to each other.

**TODO:**
- [ ] Write down (or re-run notebook 01 to get) the valence mean accuracy.
- [ ] Print both accuracies next to the 50% chance line.
- [ ] In the markdown cell below, answer:
  - Which dimension decoded better for `sub-097`?
  - Is the gap large relative to how noisy two-fold estimates are?
  - What would you need (more subjects? more runs? a permutation test?) before
    calling any difference "real"?

**A note on honesty:** with two folds and one subject, almost any difference you
see could be noise. The scientifically correct conclusion at this stage is
usually "suggestive, needs replication," not "arousal beats valence."
"""))

nb02.append(code(r"""
# TODO: print valence vs arousal accuracy against the 50% chance level
"""))

nb02.append(md(r"""
*Your comparison and honest interpretation:*

> (write here)

## Summary & what comes next

You reused one pipeline for a second question and compared results. Notebook 03
does something subtler: it keeps *all* the images but **relabels** them, asking
whether the brain distinguishes the valence *dimension* from the arousal
*dimension* at all.
"""))

write_notebook("02_binary_decoder_arousal.ipynb", nb02)


# ======================================================================
# 03_valence_vs_arousal_decoder.ipynb
# ======================================================================

nb03 = []

nb03.append(md(r"""
# 03 — Decoding the *Dimension*: Valence vs Arousal

**Purpose:** Build a binary decoder whose two classes are not "positive vs
negative," but **valence trials vs arousal trials**. This is where the idea that
"the label *is* the hypothesis" becomes concrete: you keep all 64 images and
simply **relabel** them.

---

## Notebook overview

1. Load `decoder_valence_vs_arousal.csv` (all 64 trials, labels = `valence` /
   `arousal`).
2. Build `imgs`, `y`, `groups`.
3. Reuse the same decoder pipeline.
4. Fit, score against chance (still 50%), and interpret what a high or low score
   would *mean* scientifically.

## What you are learning

- How **relabeling** the same images creates a genuinely different question.
- Why "what does this decoder's accuracy tell me?" depends entirely on the labels.
- That cross-validation grouping (`groups`) still matters no matter the labels.

## Why this matters scientifically

The first two decoders asked "within a dimension, can we tell positive from
negative?" This one asks "can we tell *which dimension was being modulated at
all*?" Those are different claims about how affect is organised in the brain.
Above-chance decoding here would suggest valence-engaging and arousal-engaging
trials evoke distinguishable whole-brain states — independent of their positive
/negative sign.
"""))

nb03.append(md(r"""
---
## Section 1 — Relabeling: the Conceptual Core

In notebooks 01–02, `label` was just a copy of `condition`. Here `label` is a
*derived* category that **collapses** four conditions into two:

| condition | label (`dimension`) |
|---|---|
| `pos_val` | valence |
| `neg_val` | valence |
| `pos_aro` | arousal |
| `neg_aro` | arousal |

You already created this `dimension` label in notebook 00 (Section 9) and saved
it to `decoder_valence_vs_arousal.csv`. The image files are **byte-for-byte the
same** as in the other decoders — only the column you call `y` has changed. That
is the entire difference, and it changes the scientific question completely.

**TODO (no code — answer below):**
- [ ] Two decoders read the *same* beta maps but report very different accuracies.
      How is that possible, given the images are identical?
"""))

nb03.append(md(r"""
*Your answer:*

> (write here)
"""))

nb03.append(md(r"""
---
## Section 2 — Load the Table and Build `imgs`, `y`, `groups`

Same mechanics as before; the table just has 64 rows now and `label` is the
dimension.

**TODO:**
- [ ] Read `decoder_valence_vs_arousal.csv` into `table`.
- [ ] Confirm `table["label"].value_counts()` is 32 `valence` / 32 `arousal`.
- [ ] Build `imgs`, `y`, `groups`; assert equal lengths.
"""))

nb03.append(code(r"""
from pathlib import Path
import pandas as pd
import numpy as np

subject = "sub-097"
project_dir  = Path(r"C:\ManzaRotation")
decoding_dir = project_dir / "Analysis" / "outputs" / subject / "decoding"
tables_dir   = decoding_dir / "tables"

# TODO: read decoder_valence_vs_arousal.csv
# TODO: check 32 / 32 balance
# TODO: build imgs, y, groups
"""))

nb03.append(md(r"""
---
## Section 3 — A Balance Subtlety Worth Noticing

There is a quiet trap in this design. Each *run* contains all four conditions, so
each run is also 50/50 valence/arousal — good. But notice that "valence" pools
`pos_val` **and** `neg_val`. If the positive/negative split were uneven across
runs, the decoder might pick up something other than the dimension itself.

For `sub-097` the design is balanced, so this is fine — but the habit of asking
*"what else could my decoder be latching onto?"* is exactly the skepticism good
MVPA requires. The `groups` / leave-one-run-out scheme guards against the most
obvious confound (run identity); confounds *within* a label are subtler.

**TODO:**
- [ ] Print `table.groupby(["groups", "label"]).size()` and confirm each run is
      16 valence / 16 arousal.
"""))

nb03.append(code(r"""
# TODO: groupby(["groups", "label"]).size() to inspect per-run balance
"""))

nb03.append(md(r"""
---
## Section 4 — Reuse the Pipeline, Fit, and Score

The decoder setup is unchanged from notebook 01. Chance is still **50%** (32 vs
32).

**TODO:**
- [ ] Point `mask_path` at the brain mask.
- [ ] Build `cv = LeaveOneGroupOut()` and the same `Decoder` configuration.
- [ ] `decoder.fit(imgs, y, groups=groups)`.
- [ ] Compute and print the mean cross-validated accuracy vs 50%.
"""))

nb03.append(code(r"""
from nilearn.decoding import Decoder
from sklearn.model_selection import LeaveOneGroupOut

# func_deriv_dir = project_dir / "Derivatives" / subject / "func"
# TODO: mask_path = ...
# TODO: cv = LeaveOneGroupOut()
# TODO: decoder = Decoder(estimator="svc", mask=mask_path, standardize="zscore_sample",
#                         screening_percentile=..., scoring="accuracy", cv=cv)
# TODO: decoder.fit(imgs, y, groups=groups)
# TODO: print mean accuracy vs chance (50%)
"""))

nb03.append(md(r"""
---
## Section 5 — Interpret: What Would a High vs Low Score Mean?

Spell out the science *before* trusting the number:

- **Clearly above 50%:** valence-modulating and arousal-modulating trials evoke
  distinguishable whole-brain patterns — the brain "knows" which dimension was in
  play, beyond the positive/negative sign.
- **Near 50%:** no evidence (in this subject, with this mask and pipeline) that
  the dimension is decodable. That is a real result too, not a failure — but
  remember absence of evidence isn't evidence of absence, especially with two
  folds.

**TODO:**
- [ ] Write your interpretation in the markdown cell below.
- [ ] Note one follow-up analysis that would strengthen whatever you conclude
      (e.g. permutation test for an empirical chance distribution, more subjects,
      an ROI mask).

## Summary & what comes next

You have now seen three binary questions built from the same images by changing
labels. Notebook 04 steps up to **multiclass** decoding — all four conditions at
once — where chance drops to 25% and a **confusion matrix** becomes the key tool
for understanding *which* conditions the decoder confuses.
"""))

nb03.append(md(r"""
*Your interpretation and proposed follow-up:*

> (write here)
"""))

write_notebook("03_valence_vs_arousal_decoder.ipynb", nb03)


# ======================================================================
# 04_multiclass_decoder.ipynb
# ======================================================================

nb04 = []

nb04.append(md(r"""
# 04 — Multiclass Decoding (all four conditions)

**Purpose:** Move from two-class to **four-class** decoding. The decoder must now
assign each trial to one of `pos_val`, `neg_val`, `pos_aro`, `neg_aro`. This is
harder than any binary problem, chance drops to **25%**, and the most useful
output is no longer a single accuracy but a **confusion matrix** that shows
*which* conditions get mixed up.

> Reference: Nilearn's decoding tutorials include multiclass examples; the
> `Decoder` object handles multiclass automatically (one-vs-rest under the hood),
> so most of your binary pipeline carries straight over.

---

## Notebook overview

1. Load `decoder_4class.csv` (all 64 trials, four labels).
2. Build `imgs`, `y`, `groups`.
3. Reuse the decoder pipeline — it goes multiclass with no special handling.
4. Score against chance = 25%.
5. Build and read a **confusion matrix** to interpret the decoder's mistakes.

## What you are learning

- What **multiclass classification** is and why it is harder than binary.
- Why chance = **1 / n_classes** = 25% here.
- How to build, plot, and *read* a **confusion matrix**.
- How to turn "the decoder's mistakes" into a scientific observation.

## Why this matters scientifically

Overall 4-class accuracy is a blunt summary. The *pattern of confusions* is the
interesting part: if the decoder reliably tells valence-conditions from
arousal-conditions but muddles positive vs negative *within* a dimension, that
tells you the dimension is more strongly represented than the sign. A confusion
matrix turns a single number into a structured hypothesis about representational
geometry.
"""))

nb04.append(md(r"""
---
## Section 1 — From Binary to Multiclass

A binary SVM draws one boundary. With four classes, scikit-learn (and Nilearn's
`Decoder`) typically trains **one-vs-rest**: a separate "is it this class or
not?" SVM per class, then predicts whichever is most confident. You do not have
to code this — passing four distinct labels in `y` is enough.

Why is it harder?

- More ways to be wrong: each trial can be misassigned to any of three other
  classes, not just one.
- The data is spread thinner: 16 trials per class, and within each CV fold the
  training set is smaller still.
- **Chance = 1 / number_of_classes = 1 / 4 = 25%.** A decoder scoring 30% is
  doing *something*, even though 30% sounds low — context (the 25% bar) is
  everything.
"""))

nb04.append(md(r"""
---
## Section 2 — Load the 4-Class Table and Build Inputs

**TODO:**
- [ ] Read `decoder_4class.csv` into `table`.
- [ ] Confirm `table["label"].value_counts()` shows 16 for each of the four classes.
- [ ] Build `imgs`, `y`, `groups`; assert equal lengths.
"""))

nb04.append(code(r"""
from pathlib import Path
import pandas as pd
import numpy as np

subject = "sub-097"
project_dir  = Path(r"C:\ManzaRotation")
decoding_dir = project_dir / "Analysis" / "outputs" / subject / "decoding"
tables_dir   = decoding_dir / "tables"

# TODO: read decoder_4class.csv into `table`
# TODO: check 16 per class
# TODO: build imgs, y, groups
"""))

nb04.append(md(r"""
---
## Section 3 — Reuse the Pipeline (Now Multiclass)

The same `Decoder` configuration works. Two small notes:

- `scoring="accuracy"` still makes sense, but with four classes you may also want
  per-class scores — `decoder.cv_scores_` is keyed by class, so you get those for
  free.
- Keep `LeaveOneGroupOut` so each fold still tests on a held-out run.

**TODO:**
- [ ] Point `mask_path` at the brain mask.
- [ ] Build `cv = LeaveOneGroupOut()` and the same-style `Decoder`.
- [ ] `decoder.fit(imgs, y, groups=groups)`.
- [ ] Print `decoder.cv_scores_` (per class) and the overall mean accuracy vs 25%.
"""))

nb04.append(code(r"""
from nilearn.decoding import Decoder
from sklearn.model_selection import LeaveOneGroupOut

# func_deriv_dir = project_dir / "Derivatives" / subject / "func"
# TODO: mask_path = ...
# TODO: cv = LeaveOneGroupOut()
# TODO: decoder = Decoder(estimator="svc", mask=mask_path, standardize="zscore_sample",
#                         screening_percentile=..., scoring="accuracy", cv=cv)
# TODO: decoder.fit(imgs, y, groups=groups)
# TODO: print per-class cv_scores_ and the mean vs 25% chance
"""))

nb04.append(md(r"""
---
## Section 4 — Get Cross-Validated Predictions for a Confusion Matrix

A confusion matrix compares **true** labels to **predicted** labels. You need a
prediction for every trial, made by a model that did *not* train on that trial —
i.e. **cross-validated** predictions. The clean way to get these is
scikit-learn's `cross_val_predict`, which returns one out-of-fold prediction per
sample.

There is a wrinkle: `cross_val_predict` needs an estimator that consumes a
feature *matrix* `X`, but our images are NIfTI files. So first turn the images
into `X` with a masker, then run a plain scikit-learn pipeline. This also makes
the masking step — usually hidden inside `Decoder` — explicit, which is good for
learning.

```python
from nilearn.maskers import NiftiMasker

masker = NiftiMasker(mask_img=mask_path, standardize="zscore_sample")
X = masker.fit_transform(imgs)     # shape (64, n_voxels) -> the feature matrix
print(X.shape)
```

Then a pipeline that mirrors the `Decoder` internals (ANOVA screening + linear
SVM), evaluated with the same leave-one-run-out CV:

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_predict

pipe = Pipeline([
    ("anova", SelectPercentile(f_classif, percentile=...)),   # TODO: match your decoder
    ("svc",   SVC(kernel="linear")),
])

y_pred = cross_val_predict(pipe, X, y, groups=groups, cv=LeaveOneGroupOut())
```

Because `cross_val_predict` refits the whole `pipe` (ANOVA included) inside each
fold, the feature selection stays leak-free — same principle as Section 5 of
notebook 01.

**TODO:**
- [ ] Build `masker` and transform `imgs` into `X`; print `X.shape`.
- [ ] Build the `pipe` with `SelectPercentile` + linear `SVC`.
- [ ] Get `y_pred` via `cross_val_predict` with `groups` and `LeaveOneGroupOut`.
"""))

nb04.append(code(r"""
from nilearn.maskers import NiftiMasker
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_predict

# TODO: masker = NiftiMasker(mask_img=mask_path, standardize="zscore_sample")
# TODO: X = masker.fit_transform(imgs);  print(X.shape)

# TODO: pipe = Pipeline([... ANOVA ..., ... linear SVC ...])
# TODO: y_pred = cross_val_predict(pipe, X, y, groups=groups, cv=LeaveOneGroupOut())
"""))

nb04.append(md(r"""
---
## Section 5 — Build and Plot the Confusion Matrix

A **confusion matrix** is a square table: rows = true class, columns = predicted
class. The diagonal counts correct predictions; off-diagonal cells show *which*
class a true class gets mistaken for. Reading the off-diagonal is the whole point.

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

labels = ["pos_val", "neg_val", "pos_aro", "neg_aro"]   # fix the order explicitly
cm = confusion_matrix(y, y_pred, labels=labels)
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap="Blues", colorbar=True)
```

Fixing the `labels` order matters — otherwise rows/columns come out alphabetical
and your reading of "valence block vs arousal block" gets scrambled.

**TODO:**
- [ ] Choose an explicit `labels` order (group the two valence classes together
      and the two arousal classes together — it makes block structure visible).
- [ ] Compute `cm` and plot it with `ConfusionMatrixDisplay`.
- [ ] Save the figure under `decoding/figures/`.
"""))

nb04.append(code(r"""
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# TODO: labels = [...]   # explicit order; group valence together, arousal together
# TODO: cm = confusion_matrix(y, y_pred, labels=labels)
# TODO: ConfusionMatrixDisplay(...).plot(...)
# TODO: save to decoding/figures/
"""))

nb04.append(md(r"""
---
## Section 6 — Interpreting the Mistakes

Now read the matrix like a scientist, not just a scorekeeper. Ask:

- **Is the diagonal above chance?** With 16 trials per class, pure guessing puts
  ~4 on each diagonal cell. Noticeably more than 4 means that class is decodable.
- **Are confusions structured or random?** If `pos_val` is mostly confused with
  `neg_val` (and `pos_aro` with `neg_aro`), the decoder separates the *dimensions*
  but struggles with *sign within* a dimension — consistent with what notebooks
  01–03 hinted at. If confusions are spread evenly, there is no such structure.
- **Does any one class collapse?** A row that is almost entirely predicted as one
  other class can signal imbalance, a bad fold, or a class that simply isn't
  separable here.

**TODO:**
- [ ] In the markdown cell below, describe the diagonal vs off-diagonal pattern.
- [ ] State whether the confusions look like a "valence block vs arousal block"
      structure, and what that would imply.
- [ ] Connect it back to your binary results: is the multiclass story consistent
      with notebooks 01–03?
"""))

nb04.append(md(r"""
*Your reading of the confusion matrix:*

> (write here)
"""))

nb04.append(md(r"""
---
## Section 7 — Summary of the Whole Series

You have built a complete teaching decoding workflow for `sub-097`:

| notebook | question | classes | chance |
|---|---|---|---|
| 00 | (prepare the input tables) | — | — |
| 01 | positive vs negative **valence** | 2 | 50% |
| 02 | positive vs negative **arousal** | 2 | 50% |
| 03 | **valence** vs **arousal** dimension | 2 | 50% |
| 04 | all four conditions | 4 | 25% |

**The throughline:** the images never changed. Every different "result" came from
**how you labelled and grouped** the same 64 beta maps, and every honest result
depended on **leak-free cross-validation** that respected run boundaries.

**Where a real study goes next:**
- Repeat across **many subjects** and test the *group* accuracy against chance.
- Run a **permutation test** (shuffle `y` many times) to get an *empirical* chance
  distribution instead of assuming exactly 50% / 25%.
- Swap the whole-brain mask for **ROI masks** to localise *where* the information
  lives.
- Inspect **weight maps / confusion structure** to interpret representational
  geometry — carefully, with the Haufe-et-al. caveat in mind.

**TODO:**
- [ ] Write a few sentences summarising what you found across all four decoders
      for `sub-097`, and what you would do next if this were your own study.
"""))

nb04.append(md(r"""
*Your closing summary:*

> (write here)
"""))

write_notebook("04_multiclass_decoder.ipynb", nb04)

print("done")
