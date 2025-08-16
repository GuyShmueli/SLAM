# SLAM: Final Project

The code is split into classes found in `slam` directory.  

---

## Table of Contents

- [Section 1: Introduction and Overview](#section-1-introduction-and-overview)  
  - [1.1 Background](#11-background)  
  - [1.2 Importance](#12-importance)  
  - [1.3 Algorithms](#13-algorithms)  
    - [1.3.1 Triangulation](#131-triangulation)  
    - [1.3.2 PnP (wrapped with RANSAC)](#132-pnp-wrapped-with-ransac)  
    - [1.3.3 Bundle-Adjustment](#133-bundle-adjustment)  
    - [1.3.4 Loop-Closure](#134-loop-closure)  
- [Section 2: Code](#section-2-code)  
- [Section 3: Performance Analysis](#section-3-performance-analysis)  
  - [3.1 Tracking Statistics](#31-tracking-statistics)  
  - [3.2 Tracking Plots](#32-tracking-plots)  
  - [3.3 PnP, BA and LC Comparison Plots](#33-pnp-ba-and-lc-comparison-plots)  
  - [3.4 BA-specific Plots](#34-ba-specific-plots)  
  - [3.5 LC-specific Plots](#35-lc-specific-plots)  
  - [3.6 The Trajectory](#36-the-trajectory)  
- [Section 4: Extra – SuperPoint and SuperGlue](#section-4-extra--superpoint-and-superglue)

---

## Section 1: Introduction and Overview

### 1.1 Background
SLAM (“Simultaneous Localization and Mapping”) is the core perception task designated to simultaneously estimate the robot’s trajectory (localization) and reconstruct landmarks in its environment (mapping), real-time speaking. In a stereo‑vision setting, the robot receives two synchronized images at each time-step and must:

- Estimate its 6 Degree-of-Freedom (DoF) pose, relative to an arbitrary world frame.
- Reconstruct a 3D representation of the landmarks.

### 1.2 Importance
SLAM is important because, nowadays, robots are involved everywhere. It enables the robot to operate and navigate when a higher precision than achieved by GPS is needed. Some concrete examples:

- **Transportation domain:** Autonomous vehicles require a much lower frame-to‑frame (relative) error than the one introduced by GPS, which can be ~1 meters. SLAM provides ~10^{-2} and even ~10^{-3} meters frame-to-frame error.
- **Medical domain:** Surgical robots employ SLAM inside the human body.
- **Military domain:** Drones can use SLAM in GPS-denied arenas (e.g., indoor).
- **Entertainment domain:** AR headsets embed SLAM pipelines so virtual objects stay rigidly anchored to the physical world.

### 1.3 Algorithms

#### 1.3.1 Triangulation
- **Objective:** Reconstruct a 3D point $X = (p_x, p_y, p_z)^T$ corresponding to a pixel $x = (u, v)^T$ in an image.
- **Challenge:** According to projective geometry, a pixel from a single image can be mapped into a ray, not to a single point.
- **Solution:** We use a rectified stereo-pair, so we can compute $p_z$ while also maintaining geometric inliers by using $v_L = v_R$ validation.
- **Input:** 2 pixels $x_L, x_R$ representing the same landmark in a stereo-camera alongside their camera matrix.

**Algorithmic details:** We use homogenous triangulation, so we work in projective-coordinates:
- $X \rightarrow X_h = (p_x, p_y, p_z, 1)^T$  
- $x \rightarrow x_h = (u, v, 1)^T$

According to the camera-projection equation:
- $\lambda_i x_i^h = P_i X^h$  
- $P_i = K [R_i \mid t_i]$  
Where $i$ represents either the left or right camera $i \in \{1,2\}$, $P_i$ is the corresponding camera matrix, and $\lambda_i$ is the unknown depth-scale introduced by projective geometry.

Hence, the camera-projection equation tells us that, in ideal scenario, $x_i^h$ and $P_i X^h$ vectors are parallel. We can utilize the fact that parallel vectors have a 0-vector cross-product:
- $x_i^h \times P_i X^h = 0$

Which yields us 2 equations for the left camera, and 2 equations for the right one. We can assemble them in a compact $4 \times 4$ linear homogenous system:
```
A_h = (
u1 P13^T − P11^T
v1 P13^T − P12^T
u2 P23^T − P21^T
v2 P23^T − P22^T
)_{4×4}
```
We seek $A_h X_h = 0$, where $X_h$ is exactly what we’re trying to find. The problem is that the right and left camera won’t agree with one another exactly due to real-life noise. We can do least-squares to settle that. It gives us the optimal parameters $\hat{X}_h$:
- $\hat{X}_h = \arg\min_{\lVert X_h \rVert = 1} \lVert A_h X_h \rVert^2$

We can now do an SVD-decomposition to $A_h$: $A_h = U \Sigma V^T$, where $\Sigma = \mathrm{diag}(\sigma_1 \ge \sigma_2 \ge \sigma_3 \ge \sigma_4)$. So $A_h X_h$ is equivalent to:
- $A_h X_h = U \Sigma V^T X_h = U \Sigma (V^T X_h) = U \Sigma y = U (\sigma_1 y_1, \sigma_2 y_2, \sigma_3 y_3, \sigma_4 y_4)^T$

Now we can compute $\lVert A_h X_h \rVert^2$:
- $(A_h X_h)^T (A_h X_h) = (\sigma_1 y_1, \sigma_2 y_2, \sigma_3 y_3, \sigma_4 y_4)^T U^T U (\sigma_1 y_1, \sigma_2 y_2, \sigma_3 y_3, \sigma_4 y_4)^T$
- Since $U$ is orthonormal, $U^T U = I$. Therefore the cost reduces to $\sigma_1^2 y_1^2 + \sigma_2^2 y_2^2 + \sigma_3^2 y_3^2 + \sigma_4^2 y_4^2$.

So our equivalent least-squares to solve is:
- $\hat{X}_h = \arg\min_{\lVert y \rVert = 1} (\sigma \cdot y)$

Note that $\lVert y \rVert = 1$ because $y \equiv V^T X_h$, and $V^T$ can only change $X_h$ direction, not its size. Because $\Sigma$ is ordered such that $\sigma_4$ is the smallest, we should choose $y = (0,0,0,1)^T$, which corresponds to:
- $y = (0,0,0,1)^T = V^T \hat{X}_h$

Left-multiplying by $V$ yields:
- $\hat{X}_h = V y = V (0,0,0,1)^T = V(:,4)$

That is, the optimal solution is the fourth (last) column of $V$.

#### 1.3.2 PnP (wrapped with RANSAC)
- **Objective:** Estimate a 6-DoF pose of a camera. That is, find the extrinsic matrix $[R \mid t]$. In our case, we wish to estimate each relative rotation and translation $R_{i \to i+1}, t_{i \to i+1}$.
- **Challenge:** A full non-linear least-squares fit on all matches would be too slow to run inside the real-time front-end.
- **Solution:** We use P4P solution, wrapped with RANSAC to remove outliers.
- **Input:** The intrinsic matrix $K$, exactly 4 landmarks (3D-points), and their 4 matching pixels in the left-camera (from the triangulation part). Essentially, one camera is enough, but we wrap P4P with RANSAC to remove outliers, so we use the right-camera’s 4 pixels as well.

**Algorithmic details:**

**Minimal hypothesis (AP3P / P4P):**
- Randomly pick 4 correspondences $(X_j, x_j)_{j=1}^4$ from the current track set in the left-image.
- Solve the closed-form algebraic P3P system, then there are at most 4 candidate poses such that only one of them is physically valid.

**Reprojection verification:**
- Once obtained a candidate $[R \mid t]$, we can use it to reproject all the landmarks found in that stereo-pair onto both cameras:  
  $x^{\text{proj}}_L = \pi(K [R \mid t] X), \quad x^{\text{proj}}_R = \pi(K [R' \mid t'] X)$  
  where $[R' \mid t'] = [R \mid t] T_{L \leftarrow R}$.
- A landmark-pixel match is an inlier if both errors $\lVert x^{\text{proj}}_L - x^{\text{meas}}_L \rVert^2$ and $\lVert x^{\text{proj}}_R - x^{\text{meas}}_R \rVert^2$ are smaller than some threshold $\tau$ (say, 2 pixels).
- A mask is created with each landmark-pixel match assigned to ‘True’ or ‘False’.

- If another subset of 4 landmarks yields more inliers (bigger mask sum), the best mask is updated.

**An adaptive approach embedded in the RANSAC-loop:**  
Denoting $w$ as the inlier-ratio $w = \frac{\#\text{inliers}}{\#text{landmarks}}$, $m$ as the sample size ($m=4$) and $N$ as the number of iterations we repeat that loop.

- The probability the whole sample is pure inliers is $P_{\text{good}} = w^m$, hence the probability for it to be bad (at least one outlier) is $P_{\text{bad}} = 1-w^m$.
- The probability that all $N$ iterations yield at least one outlier each time is $P_{\text{all iterations bad}} = (1-w^m)^N$. We want to find $p$ such that  
  $p \ge P_{\text{at least one iteration good}} = 1 - P_{\text{all iterations bad}} \Rightarrow p \ge 1 - (1-w^m)^N$.
- Hence, the minimal number of iterations required to achieve $p$ is:

$$
N_{\text{req}} = \frac{\log(1-p)}{\log(1-w^m)}.
$$



**Pose refinement:**
- Run one Gauss-Newton / `cv2.SOLVEPNP_ITERATIVE` on the final inlier set to minimize true reprojection error in the left image.

#### 1.3.3 Bundle-Adjustment

- **Triangulation** is considered 1-point, multi-view. It tries to answer the following question: Given a multi-view perception, where should the landmark be mapped so reprojecting it will yield the best agreement by its corresponding measured pixels in all frames (in our case 2 frames)? Note that the cameras’ poses are fixed, only the location of the landmark can be moved.
- **PnP** is considered 1-view, multi-point. It tries to answer the following question: Given many points in a single frame and their corresponding landmarks, where should the camera be located so reprojecting all these landmarks will yield the best agreement by their corresponding measured pixels in that frame? Note that the landmarks’ locations are fixed, only the cameras’ poses can be moved.
- **Bundle-Adjustment (BA)** incorporates both ideas simultaneously. Neither landmarks’ locations, nor cameras’ poses are fixed. Both can be moved to satisfy the following non-linear least-squares problem:



$$
{ \hat{R}^L_i, \hat{t}^L_i, \hat{R}^R_i, \hat{t}^R_i }_{i=1}^T,\; \{ \hat{X}_j \}_{j=1}^M = \arg\min_{\{R_i,t_i\},\{X_j\}} \sum_{i=1}^T \sum_{j=1}^{N_i} \left\| x^L_{i,j} - \pi\!\left(K [R^L_i \mid t^L_i] X_j\right)\right\|_2^2 + \left\| x^R_{i,j} - \pi\!\left(K [R^R_i \mid t^R_i] X_j\right)\right\|_2^2.
$$



- **Deterministic vs Probabilistic SLAM:** Deterministic SLAM has a prominent disadvantage. Assuming we have a measurement $z_{ij}$ and 2 projections: $\pi_1, \pi_2$. If these two projections lie, for example, on the same circle around $z_{ij}$, the deterministic approach would say they are similarly probable (as in $L^2$-norm).

If we use a probabilistic approach instead, it will give us extra information. If we know the covariance matrix, then we can see, for example, that $\pi_2$ is within a smaller Mahalanobis distance, hence more probable.

**Probabilistic SLAM cost derivation:**  
We can model the measured pixel $z_{ij}$ (which was detected by SIFT-like algorithms), by reprojecting its corresponding $q_j$ landmark onto camera $c_i$ and add a Gaussian noise $\epsilon_{ij}$:
- $z_{ij} = \pi(c_i, X_j) + \epsilon_{ij}$, $\epsilon_{ij} \sim \mathcal{N}(0, \Sigma_{ij})$  
- Practically, $\Sigma_{ij}$ is usually shared across all measurements $\Sigma_{ij} \equiv \Sigma$.

That is, we know the conditional probability $P(z_{ij} \mid c_i, X_j) \sim \mathcal{N}(\pi(c_i, X_j), \Sigma)$. However, we need the case where we have $z_{ij}$ and wish to model $c_i, X_j$, which is $P(c_i, X_j \mid z_{ij})$. To get that, we can use Bayes rule:


$$
P(c_i, X_j \mid z_{ij}) = \frac{1}{P(z_{ij})} P(z_{ij} \mid c_i, X_j)\, P(c_i, X_j).
$$



The case where we condition our parameters (what we’re trying to model) by the measurements (data) is called ‘posterior’. We wish to find the parameters $c_i, X_j$ that maximize it. That is, we actually perform MAP-estimation.

By noticing that $P(z_{ij})$ does not depend on the parameters, we can drop it and use $\propto$ (proportion) instead of equality:
- $P(c_i, X_j \mid z_{ij}) \propto P(z_{ij} \mid c_i, X_j)\, P(c_i, X_j)$.

We can now either do a Maximum-likelihood estimation, or MAP-estimation. If we do **MLE**, we can drop the prior $P(c_i, X_j)$ and obtain:
- $P(c_i, X_j \mid z_{ij}) \propto P(z_{ij} \mid c_i, X_j)$.

So the objective is:


$$
\{\hat{c}_i\}_{i=1}^T, \{\hat{X}_j\}_{j=1}^M = \arg\max_{\{c_i\},\{X_j\}} \prod_{i,j} P(z_{ij} \mid c_i, X_j).
$$



Writing it as log-likelihood yields:


$$
\{\hat{c}_i\}_{i=1}^T, \{\hat{X}_j\}_{j=1}^M = \arg\min_{\{c_i\},\{X_j\}} \left(-\sum_{i,j} \log P(z_{ij} \mid c_i, X_j)\right) = \arg\min_{\{c_i\},\{X_j\}} \frac{1}{2} \sum_{i,j} \lVert \Delta z_{ij} \rVert_{\Sigma_{ij}}^2,
$$


where $\Delta z_{ij} \equiv z_{ij} - \pi(c_i, X_j) = \epsilon_{ij}$, and $\lVert \Delta z_{ij} \rVert_{\Sigma}^2$ is the squared Mahalanobis-norm $\Delta z_{ij}^T \Sigma^{-1} \Delta z_{ij}$. We can see that the deterministic approach can be achieved by choosing $\Sigma$ to be the $2 \times 2$ identity matrix, hence the probabilistic approach is much more general.

Recall that this was the derivation of the MLE approach. If we want the **MAP** approach we need to look at:
- $P(c_i, X_j \mid z_{ij}) \propto P(z_{ij} \mid c_i, X_j) P(c_i, X_j)$  
and set a prior (anchor) the first pose (usually anchoring $X_0$ is not needed):
- $c_0 \sim \mathcal{N}(\bar{c}_0, \Sigma_{c_0})$, so $-\log P(c_0) = \frac{1}{2} \lVert c_0 - \bar{c}_0 \rVert^2_{\Sigma_{c_0}}$.

Performing the same process as in MLE yields:
- $J_{\mathrm{MAP}} = \frac{1}{2} \sum \lVert \Delta z_{ij} \rVert_\Sigma^2 + \frac{1}{2} \lVert c_0 - \bar{c}_0 \rVert^2_{\Sigma_{c_0}} = J_{\mathrm{MLE}} + \frac{1}{2} \lVert c_0 - \bar{c}_0 \rVert^2_{\Sigma_{c_0}}$.

In order to convert the Mahalanobis-norm into $L^2$-norm, a process called **whitening** is performed:
- $\lVert \Delta z_{ij} \rVert_\Sigma^2 = (\Sigma^{-1/2} \Delta z_{ij})^T (\Sigma^{-1/2} \Delta z_{ij}) \equiv r_{ij}^T r_{ij} = \lVert r_{ij} \rVert_2^2$.

**Optimization:** BA is a non-linear least-squares problem, so we can’t solve it deterministically. Instead, we use an iterative optimization algorithm. Well-known algorithms of this kind include: Gradient-Descent (GD), Gauss-Newton and Levenberg-Marquardt.

Our objective is to approach the cost function’s minimum (preferably global, but could be local). If we look at a small neighbourhood around a minimum, it resembles a smiling parabola. If we look even at a smaller neighbourhood, the curve flattens out and looks almost like a horizontal line.

- In **GD** we move in small steps towards the opposite direction of the gradient, to iteratively converge to a minimum: $x^{(t+1)} = x^{(t)} - \alpha \nabla f(x^{(t)})$. The advantage of GD is that it involves only first derivatives, so it is faster to compute. However, it converges only when it arrives at a point where it is flat enough, which may take a long time.

- In **Gauss-Newton** we linearize the residuals: $r(x^{(t+1)}) = r(x^{(t)} + \Delta x) \approx r(x^{(t)}) + \frac{\partial r}{\partial x}\big|_{(t)} \Delta x$. Where $J \equiv \frac{\partial r}{\partial x}\big|_{(t)}$ is the (2 × 9) Jacobian of $r$. We’ll first plug it into the cost $F(\Delta x) = \frac{1}{2} \lVert r(x^{(t+1)}) \rVert^2_\Sigma$:



$$
F(\Delta x) = \tfrac{1}{2} (r(x^{(t)}) + J \Delta x)^T \Sigma^{-1} (r(x^{(t)}) + J \Delta x)
= \tfrac{1}{2} r^T \Sigma^{-1} r + \tfrac{1}{2} r^T \Sigma^{-1} J \Delta x + \tfrac{1}{2} \Delta x^T J^T \Sigma^{-1} r + \tfrac{1}{2} \Delta x^T J^T \Sigma^{-1} J \Delta x.
$$



Now, we’ll denote the **gradient** as:
- $g \equiv - J^T \Sigma_{\text{meas}}^{-1} r(x^{(t)})$

And the **Hessian** (approximated information-matrix of the entire state-vector $x$) as:


$$
H =
\begin{pmatrix}
H_{pp}^{6\times 6} & H_{pl}^{6\times 3} \\
H_{lp}^{3\times 6} & H_{ll}^{3\times 3}
\end{pmatrix}
\equiv J^T \Sigma_{\text{meas}}^{-1} J.
$$



We want to find its minimum with respect to $\Delta x$, so: $\nabla_{\Delta x} F = H \Delta x - g = 0$. Hence, the **normal equations** are:
- $H^{(t)} \Delta x^{(t)} = g^{(t)}$  
- $H^{(t)} = J^{(t)T} \Sigma_{\text{meas}}^{-1} J^{(t)}$  
- $g^{(t)} = - J^{(t)T} \Sigma_{\text{meas}}^{-1} r(x^{(t)})$

More precisely, in order to avoid storing huge, sparse $H$ and $g$, **GTSAM** utilizes their sparsity for a more efficient storage by performing **Cholesky decomposition** over $H$ and solving the corresponding equations.

The advantage of **Gauss-Newton** is that, due to its parabolic characterization, it converges faster to a minimum. Its disadvantage is that it requires second derivatives, so it is more time-consuming.

**Levenberg-Marquardt** tries to integrate both approaches. Far from the minimum, it acts like GD. When getting closer to it, it acts like Gauss-Newton. Hence, this is the approach I used.

- **Gauge freedom:** Due to the fact we have only relative poses, we’re missing the global characterization of the trajectory. If we don’t set a prior (with high confidence), we will probably end up with an ill-posed problem.
- **Global vs Local Bundle-Adjustment:** Global means to optimize all poses and landmarks in one shot, while local BA limits the optimization to a sliding window of, say, 10–20 frames. The notable advantage is in the online part: when adding new frames, we will not have to re-optimize poses and landmarks that have already been optimized. Only the new frames will be optimized, which makes it more appropriate for real-time operation.

#### 1.3.4 Loop-Closure
- **Pose-Graph:** Because local BA refines only a sliding window of keyframes, older poses accumulate drift and lose global consistency. To re-align the entire trajectory in real-time, we treat each keyframe pose as a node and the relative-pose constraints (odometry, local-BA priors, loop-closure edges) as a sparse pose graph. Running a fast non-linear optimisation on that graph periodically (or after every new key-frame) updates all poses at low cost, restoring global coherence while keeping the heavy global BA off the real-time path.
- **Loop-closure** is the moment the system recognizes that the current camera view corresponds to a place it has already visited before. The front-end confirms this by matching features (place recognition + RANSAC-PnP) and creates a loop-closure edge (a 6-DoF relative-pose constraint with its uncertainty). Passing that edge to the back-end lets the pose-graph optimizer redistribute accumulated drift over the whole trajectory, “closing the loop” and greatly reducing global error without needing global BA.

---

## Section 2: Code

The code is split into classes, found in sub-directories. *(PDF p.9)*

| Stage                         | File                                               | Lines      |
|------------------------------|----------------------------------------------------|------------|
| Triangulation                | `slam.frontend.geometry.triangulation`             | —          |
| RanSAC                       | `slam.frontend.geometry.pnp`                       | 102        |
| PnP                          | `slam.frontend.geometry.pnp`                       | —          |
| Database definition          | `slam.backend.tracking.database`                   | —          |
| Adding a frame to the database | `slam.backend.tracking.fill_database`            | 85         |
| Single Bundle creation       | `slam.backend.optimizers.bundle_adjustment`        | 126        |
| The relative transformation  | `slam.backend.optimizers.loop_closure`             | 266, 272   |
| PoseGraph building           | `slam.backend.optimizers.loop_closure`             | 252        |
| Loop closure detection       | `slam.backend.optimizers.loop_closure`             | 252        |

---

## Section 3: Performance Analysis

> **Note:** I have implemented the code for all the plots, but the guidelines asked for substantial subset of them and restricted the project to 20 pages, so I didn’t present all the plots in the project. *(PDF p.10)*

### 3.1 Tracking Statistics
- Total number of tracks: **204,432**.  
- Number of frames: **2,600**.  
- Mean track length (frames): **5.48**.  
- Mean number of frame links: **430.65**.

### 3.2 Tracking Plots

#### Track Length Histogram *(Figure 3.2.1 — PDF p.10)*
Longevity of feature tracks (detection + matching + linking) across the whole sequence. `slam.backend.tracking.database` line **552**.

**What it shows:** To present the data in a meaningful way, the y-axis is logarithmic scaled due to the large ratio between \#short tracks / \#long tracks.

**Reading the plot:** Most tracks are short with a long, thin tail out to ~50 frames. A handful of very long tracks exist, but the majority die within a few frames.

**What it demonstrates:**  
- **Positive:** The system consistently generates many usable tracks. The presence of tracks of length 30–50 indicates pockets of highly stable features.  
- **Negative:** The dominance of short lifetimes implies frequent track breakage (viewpoint change, or aggressive filtering), which limits how many constraints survive into BA.

#### Per-Frame Connectivity *(Figure 3.2.2 — PDF p.11)*
A connectivity graph presenting the number of tracks on the frame with links also in the next frame. The horizontal line is the mean. `slam.backend.tracking.database` line **568**.

**What it shows:** For each frame $t$, the number of tracks that continue to frame $t+1$. This measures tracker stability and database linking quality frame-to-frame.

**Reading the plot:** Values at around ~300–400 for long stretches, but with large fluctuations: dips near ~150–250 indicate hard segments (low texture, blur, large rotations), while peaks ~500–700 reflect very track-rich frames.

**What it demonstrates:**  
- **Positive:** Sustained mid-high connectivity across most of the run, which yields enough constraints to propagate downstream to PnP/BA.  
- **Negative:** There are a few dips which demonstrate weaker constraints and more drift risk (or even ill-posed problem).

#### Number of matches per Frame *(Figure 3.2.3 — PDF p.11)*
Number of frame-to-frame matches before applying any geometrical outlier filtering. `slam.backend.tracking.fill_database` line **262**.

**Reading the plot:** Mean around ~550 (red baseline). Wide spread (~200–1000). There is a sustained drop (~150–350) that lines up with the connectivity dip in Figure 3.2.2, followed by recovery to high-match regions later in the sequence.

**What it demonstrates:**  
- **Positive:** The system usually produces a high volume of matches, supporting stable RANSAC-PnP.  
- **Negative:** The slumps are bottlenecks, fewer reliable matches lead to fewer verified inliers, which lead to shakier poses.

#### Percentage of PnP Inliers per Frame *(Figure 3.2.4 — PDF p.12)*
PnP inlier percentage per frame. `slam.backend.tracking.fill_database` line **262**.

**What it shows:** Geometric verification quality based on two tests:  
I. Stereo condition (approximately the same image y-coordinate).  
II. RANSAC-PnP (as explained in 1.3.2).

**Reading the plot:** The red baseline is ~65%. Most frames sit between 60–80%. The weakest region (~50%) aligns with the low-matches/low-connectivity segment.

**What it demonstrates:**  
- **Positive:** A stable, high inlier rate over thousands of frames indicates good outlier rejection, and broadly consistent scene geometry.  
- **Negative:** Dips show where PnP gets less reliable, as in previous figures.

#### PnP Reprojection Error vs. Distance from Reference *(Figure 3.2.5 — PDF p.13)*
PnP reprojection error (L2 pixels) as a function of temporal distance from a reference frame (an arbitrary track of length 40 was chosen), plotted for the left/right cameras separately. `slam.analysis.pnp_plot` line **72**.

**Reading the plot:** Error grows almost monotonically all along the track, for both cameras.

**What it demonstrates:** Left/right cameras are consistent, and the monotonically increasing behavior is what I was expecting before applying any global optimizers.

### 3.3 PnP, BA and LC Comparison Plots

#### PnP, BA and LC Absolute Position Error (with respect to KITTI’s ground-truth) *(Figure 3.3.1 — PDF p.13–14)*
Per-frame absolute position error (in meters) after PnP (top), BA (middle) and LC (bottom), decomposed into x, y, z and Euclidean norm. The PnP function is in `slam.analysis.pnp_plot` line **91**, while the BA/LC in `slam.analysis.optimizers_analysis` line **31**.

**What’s plotted & how it’s computed:**  
For each frame $i$, compare the estimated camera center $C^{\text{est}}_i$ to the KITTI ground-truth center $C^{\text{gt}}_i$ in the world frame and report the component-wise absolute translation errors $\lVert e_{x,i} \rVert, \lVert e_{y,i} \rVert, \lVert e_{z,i} \rVert$ and the Euclidean norm $\lVert e_i \rVert$, where $e_i = C^{\text{est}}_i - C^{\text{gt}}_i$ (meters).

**Observations & interpretation:**
- **PnP (top):** Errors grow with traveled distance as drift accumulates. The total position error norm peaks at ~60 m around frames ~2,000, dominated mainly by y-error.
- **BA (middle):** Windowed joint optimization reduces drift and smooths the per-axis errors. The peak norm drops to ~15 m (4× reduction vs. PnP), dominated mainly by z-error. The reason BA did manage to decrease dramatically the y-error, while decreasing more modestly the z-error is:
  

$$
u = \frac{f_x X}{Z} + c_x, \quad v = \frac{f_y Y}{Z} + c_y.
$$


  Small camera translations $\delta t = (\delta X, \delta Y, \delta Z)$ induce pixel shifts with such components in the Jacobian:  
  $\frac{\partial u}{\partial \delta X}, \frac{\partial v}{\partial \delta Y} \propto \frac{1}{Z}$, and $\left|\frac{\partial u}{\partial \delta Z}\right|, \left|\frac{\partial v}{\partial \delta Z}\right| \propto \frac{1}{Z^2}$.  
  That’s why changing z-coordinate reduces the cost much less efficiently than the same-sized change in x or y.
- **LC (bottom):** Adding loop-closure constraints anchors the trajectory globally and removes most accumulated drift. The error norm stays within a 0–5 meters range across the entire run (~10× lower than PnP peaks, ~3× lower than BA peaks). Per-axis errors mostly lie below 2 m, with small oscillations after each closure as the pose-graph re-distributes corrections.

#### PnP, BA and LC Absolute Rotation Error *(Figure 3.3.2 — PDF p.15)*
Per-frame absolute orientation error after PnP, BA and LC. `slam.analysis.optimizers_analysis` line **90**.

The orientation error is defined as the minimal rotation that aligns $R^{\text{est}}_k$ to $R^{\text{gt}}_k$, and calculated as


$$
\phi_k = \arccos\!\left(\frac{\mathrm{tr}\!\left((R^{\text{gt}}_k)^T R^{\text{est}}_k\right)-1}{2}\right)\, [\deg].
$$


This metric is axis-agnostic (not yaw/pitch/roll separately). The behavior is similar to the Euclidean absolute position error.

#### PnP, BA and LC Relative Position Error *(Figure 3.3.3 — PDF p.15)*
Error of the relative translation between consecutive keyframes. `slam.analysis.optimizers_analysis` line **46**.

**What is plotted & how it’s computed:** For each adjacent keyframe pair $(k \to k+1)$, compare the estimated relative translation $t^{\text{est}}_{k \to k+1}$ with the ground-truth relative translation $t^{\text{gt}}_{k \to k+1}$ and plot:


$$
e^{\text{rel pos}}_k = \left\| t^{\text{est}}_{k \to k+1} - t^{\text{gt}}_{k \to k+1} \right\| \; [\mathrm{m}].
$$



**What it shows:** All three curves (PnP/BA/LC) closely overlap: most errors are ~0.1–0.3 m, with a few spikes up to ~0.6 m. BA and LC make only small changes to this local metric.

**What it demonstrates:** Relative inter-KF motion is already well constrained by stereo + PnP conditions over the distance of a single window. BA and LC mainly correct global drift rather than these local edges.

#### PnP, BA and LC Relative Orientation Error *(Figure 3.3.4 — PDF p.16)*
Error of the relative orientation between consecutive keyframes. `slam.analysis.optimizers_analysis` line **68**.

The behavior is similar to the relative translation error, there is only a minimal improvement after performing BA due to the reasons mentioned above.

### 3.4 BA-specific Plots

#### Optimization Error (Mean & Median) *(Figure 3.4.1 — PDF p.16–17)*
For each sliding BA window, the average (top) and median (bottom) factor error of the graph before (blue) and after (orange) optimization. `slam.analysis.ba_plot` lines **101** (mean) & **113** (median).

**What’s plotted & how it’s computed:**  
For a factor graph $G$, with factors $f_i$ and current values $\theta$ we define the total graph error:


$$
E(\theta) \equiv \sum_{i \in G} \tfrac{1}{2} \left\| W_i\, (h_i(\theta) - z_i) \right\|^2,
$$


where $W_i \equiv \Sigma^{-1/2}$ is a whitening matrix from the factor’s noise model, $z_i$ is the measurement, and $h_i(\theta)$ is the factor prediction.

The **mean factor error** is $\bar{E}(\theta) = E(\theta) / \lVert G \rVert$. The **median factor error** is the median across all factors in $G$. The median is the 50th percentile (half the factors have cost which is lower than it). It’s a robust “typical residual” which is insensitive to outliers (unlike the mean).

As expected, in both graphs the results after BA optimization are better than before. Note the lower typical values for the median, due to its robustness to outliers (insensitive to extreme values).

### 3.5 LC-specific Plots

#### Number of matches & Inlier percentage per successful loop-closure frame *(Figure 3.5.1 — PDF p.17)*
For every current frame that produced a successful loop, present the \#matches (top), and the \%inliers (bottom) passed the RANSAC-PnP test. `slam.analysis.lc_plot` line **15**.

#### LC Uncertainties Plots *(Figure 3.5.2 — PDF p.18)*
**Location (top) & Angle (bottom) Uncertainties vs. Keyframe.** For each keyframe we plot the 1σ size of the translation uncertainty ellipsoid from the pose marginal (the 3×3 translation/rotation block of the 6×6 Pose3 covariance), **without LC** (orange) and **with LC** (blue). `slam.analysis.optimizers_analysis` line **133** (location) & **155** (angle).

I chose the metric $\det(\Sigma)^{1/6}$ because it converts the covariance determinant into an average standard-deviation scale ($\sigma_1^2 \sigma_2^2 \sigma_3^2$ has units of m$^6$ when talking about translation, or deg$^6$ when talking about rotation). Without loops, both translation and rotation uncertainties grow steadily along the trajectory. With loops, the angular curve stays almost flat, and the translation curve stays consistently lower and shows sharp drops right after loop events.

#### Per-frame uncertainty score *(Figure 3.5.3 — PDF p.19)*
Per-frame uncertainty score, plotted on a $\log_{10}$ scale, to present trends more clearly. The BA curve drifts upward over time (uncertainty accumulates), while the LC curve stays lower and shows distinct drops right after loop events. `slam.analysis.optimizers_analysis` line **178**.

### 3.6 The trajectory

#### Bird eye view of PnP, BA and LC estimation, including the GT for comparison *(Figure 3.6.1 — PDF p.19)*
Top-down path (X lateral vs Z forward) of the estimated trajectory at three stages: **PnP (red)**, **after local BA (blue)**, and **after LC (green)** overlaid on ground-truth (black dashed). `slam.utils.utils` line **144**.

**Reading the plot:**
- **PnP (RMSE 27.00 m):** noticeable forward-axis drift, and the closing segments don’t meet the start (classic accumulated error).
- **After BA (RMSE 8.60 m):** the path tightens and shape distortion shrinks, corners align better but there’s still residual forward drift across long straights.
- **After LC, 23 loops (RMSE 2.39 m):** the green track almost agrees with ground‑truth, loops close cleanly and both large loops line up in scale and heading.

**What it demonstrates:**  
Clear, staged improvement: **PnP → BA** removes local inconsistency, **BA → LC** removes global drift. The large RMSE drop (27.00 → 8.60 → 2.39 m) quantifies this. Multiple runs were conducted to ensure reproducibility.

---

## Section 4: Extra – SuperPoint and SuperGlue *(PDF p.20)*

I replaced the feature extraction **AKAZE** algorithm with **SuperPoint (“SP”)** neural‑network, and the matcher with **SuperGlue (“SG”)**. Here are some key results:

*(Figure 4.1 — PDF p.20):* Presenting the corresponding figures when SP and SG were used instead of the classical extractor and matcher.

The consistently higher PnP inlier fraction indicates fewer erroneous matches and better geometric consistency frame-to-frame. That gives PnP a stronger motion prior and helps stabilise tracking through low-texture or viewpoint-change segments.

The big win from SP & SG is global consistency, not just local PnP. There is a moderate PnP improvement (**23.30 m vs 27.00 m**), but the final gain after LC is large (**1.55 m vs 2.39 m** which is ~35% lower RMSE). This pays off: more reliable closures leads to better graph conditioning which leads to tighter alignment to GT.

---

*This README is a complete Markdown conversion of the original project document, with all sections, formulas, captions, and notes retained.*