# slam/analysis/lc_plot.py
from .optimizers_analysis import OptimizersPlot
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict



class LC_Plotter(OptimizersPlot):
    """Loop-closure specific visualizations."""
    def __init__(self, lc_vals, KF_indices, poses_gt, Rts_abs_all):
        super().__init__(lc_vals, KF_indices, poses_gt,
                         Rts_abs_all,"LC")

    def lc_success_stats(self, stats, aggregate='max', xaxis='frame'):
        """
        Visualize loop-closure matching quality.
        aggregate : 'max' or 'mean'  (when multiple loops for the same current frame)
        xaxis     : 'frame' (image frame id) or 'kf' (keyframe index)
        """
        key = 'curr_frame' if xaxis == 'frame' else 'curr_kf'
        groups = defaultdict(list)
        for d in stats:
            groups[d[key]].append(d)

        xs = sorted(groups.keys())
        if aggregate == 'max':
            agg = lambda arr: float(np.max(arr))
        elif aggregate == 'mean':
            agg = lambda arr: float(np.mean(arr))
        else:
            raise ValueError("aggregate must be 'max' or 'mean'")

        matches_per = [agg([g['num_matches'] for g in groups[x]]) for x in xs]
        inlierpct_per = [agg([g['inlier_pct']  for g in groups[x]]) for x in xs]

        plt.figure(figsize=(12, 6))
        plt.plot(xs, matches_per, marker='o', linewidth=1)
        plt.xlabel('Frame Number' if xaxis == 'frame' else 'Keyframe Index')
        plt.ylabel('# matches')
        plt.title('Matches per successful loop-closure frame')
        plt.grid(True); plt.tight_layout(); plt.show()

        plt.figure(figsize=(12, 6))
        plt.plot(xs, inlierpct_per, marker='o', linewidth=1)
        plt.xlabel('Frame Number' if xaxis == 'frame' else 'Keyframe Index')
        plt.ylabel('Inlier percentage [%]')
        plt.title('Inlier % per successful loop-closure frame')
        plt.grid(True); plt.tight_layout(); plt.show()
