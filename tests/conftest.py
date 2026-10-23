import os

# Must precede native backend import, including during test collection.
os.environ["QULACS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
