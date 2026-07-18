import gc
import torch
import time
from memory_tracker import RSSPeakTracker

def profile_training(fn, *, poll_every_sec=0.05):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    tracker = RSSPeakTracker(poll_every_sec=poll_every_sec)
    tracker.start()
    t0 = time.perf_counter()
    out = fn()
    elapsed = time.perf_counter() - t0
    tracker.stop()
    return out, float(elapsed), float(tracker.peak_rss_mb), float(tracker.start_rss_mb)