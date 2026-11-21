# Benchmarks

Compares `svdpp` with [Surprise](https://surpriselib.com) (Python) on identical
MovieLens splits. Run everything from the package root. Data and results are
written to `bench/data/` and `bench/results/` (both git- and build-ignored).

```sh
Rscript bench/make_splits.R 1m          # 100k, 1m or 10m; downloads from GroupLens
uv venv -p 3.12 bench/.venv && VIRTUAL_ENV=bench/.venv uv pip install -r bench/requirements.txt
Rscript bench/run_r.R 1m                # all models; or e.g. run_r.R 1m asvd
bench/.venv/bin/python bench/run_surprise.py 1m
```

Settings: 20 factors, 90/10 random split (seed 1). SGD runs use Surprise's
defaults (20 epochs, lr 0.007, lambda 0.02, init sd 0.1). The L-BFGS lambda is
and the ASVD SGD lr/lambda grid are
chosen on a 10% validation split of the training data, then refitted on the
full training data. Run the R and Python scripts one at a time for timings.

Still to do for publication-quality numbers: 5-fold CV, several seeds, equal
tuning budgets for every method, ML-10M.
