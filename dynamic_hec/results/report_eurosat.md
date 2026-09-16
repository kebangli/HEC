# Dynamic Head Routing HEC — results

| dataset | seed | ways | static_hec20 | static_hec8 | static_cf20 | pool_uniform64 | knn8 | class_cond8 | router8 | router_thresh | router_soft | hybrid_static_router | router_soft_p20 | oracle8 | oracle_corr8 | oracle_soft |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| eurosat | 123 | 10 | 78.0 | 78.0 | 76.0 | 68.0 | 68.0 | 70.0 | 74.0 | 74.0 | 76.0 | 78.0 | 78.0 | 72.0 | 72.0 | 84.0 |
| eurosat | 148 | 10 | 82.0 | 84.0 | 82.0 | 78.0 | 74.0 | 72.0 | 72.0 | 72.0 | 84.0 | 84.0 | 82.0 | 72.0 | 72.0 | 88.0 |
| eurosat | 173 | 10 | 82.0 | 86.0 | 84.0 | 62.0 | 60.0 | 60.0 | 70.0 | 70.0 | 86.0 | 84.0 | 84.0 | 66.0 | 66.0 | 92.0 |

## Mean over seeds
| dataset | static_hec20 | static_hec8 | static_cf20 | pool_uniform64 | knn8 | class_cond8 | router8 | router_thresh | router_soft | hybrid_static_router | router_soft_p20 | oracle8 | oracle_corr8 | oracle_soft |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| eurosat | 80.7 | 82.7 | 80.7 | 69.3 | 67.3 | 67.3 | 72.0 | 72.0 | 82.0 | 82.0 | 81.3 | 70.0 | 70.0 | 88.0 |

## Head usage distribution (top-10 heads per dynamic method)
- **eurosat / usage_knn8_global**: 185 (L6.H17), 631 (L22.H15), 124 (L4.H12), 88 (L3.H4), 197 (L7.H1), 59 (L2.H3), 652 (L23.H8), 408 (L14.H16), 47 (L1.H19), 692 (L24.H20)
- **eurosat / usage_class_cond8_global**: 631 (L22.H15), 644 (L23.H0), 745 (L26.H17), 41 (L1.H13), 185 (L6.H17), 263 (L9.H11), 408 (L14.H16), 617 (L22.H1), 124 (L4.H12), 88 (L3.H4)
- **eurosat / usage_router8_global**: 692 (L24.H20), 644 (L23.H0), 652 (L23.H8), 631 (L22.H15), 222 (L7.H26), 124 (L4.H12), 91 (L3.H7), 745 (L26.H17)
- **eurosat / router8 query-set diversity** (mean pairwise jaccard) = 1.00 vs support-static-8 0.08 (lower = more input-adaptive)
- **eurosat / class_cond8 set sizes**: mean 4.1, min 4, max 5 (adaptive budget <= 8)