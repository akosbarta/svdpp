"""Benchmark Surprise SVD and SVD++ on a split written by make_splits.R.

Usage (from the package root): bench/.venv/bin/python bench/run_surprise.py 1m
"""
import os, sys, time
import pandas as pd
from surprise import BaselineOnly, Dataset, Reader, SVD, SVDpp, accuracy

size = sys.argv[1] if len(sys.argv) > 1 else "100k"
d = os.path.join("bench", "data", size)
tr = pd.read_csv(os.path.join(d, "train.csv"))
te = pd.read_csv(os.path.join(d, "test.csv"))
lo, hi = tr.rating.min(), tr.rating.max()
trainset = Dataset.load_from_df(tr[["user", "item", "rating"]], Reader(rating_scale=(lo, hi))).build_full_trainset()
testset = list(te[["user", "item", "rating"]].itertuples(index=False, name=None))

rows = []
for name, algo in [("surprise BaselineOnly (ALS)", BaselineOnly(bsl_options={"method": "als"}, verbose=False)),
                   ("surprise SVD", SVD(n_factors=20, n_epochs=20, random_state=1)),
                   ("surprise SVD++", SVDpp(n_factors=20, n_epochs=20, random_state=1))]:
    t = time.time(); algo.fit(trainset); el = time.time() - t
    rmse = accuracy.rmse(algo.test(testset), verbose=False)
    rows.append(dict(dataset=size, model=name, settings="defaults, 20 factors", fit_s=round(el, 2), test_rmse=round(rmse, 4)))
    print(rows[-1], flush=True)

os.makedirs(os.path.join("bench", "results"), exist_ok=True)
pd.DataFrame(rows).to_csv(
    os.path.join("bench", "results", f"surprise_{size}.csv"), index=False)
