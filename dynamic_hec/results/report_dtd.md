# Dynamic Head Routing HEC — results

| dataset | seed | ways | static_hec20 | static_hec8 | static_cf20 | pool_uniform64 | knn8 | class_cond8 | router8 | router_thresh | router_soft | hybrid_static_router | router_soft_p20 | oracle8 | oracle_corr8 | oracle_soft |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dtd | 123 | 20 | 90.0 | 91.0 | 90.0 | 66.0 | 67.0 | 61.0 | 71.0 | 71.0 | 91.0 | 91.0 | 91.0 | 63.0 | 63.0 | 98.0 |
| dtd | 148 | 20 | 81.0 | 82.0 | 80.0 | 64.0 | 62.0 | 59.0 | 60.0 | 60.0 | 80.0 | 81.0 | 80.0 | 62.0 | 64.0 | 86.0 |
| dtd | 173 | 20 | 78.0 | 77.0 | 76.0 | 57.0 | 57.0 | 62.0 | 65.0 | 65.0 | 77.0 | 78.0 | 78.0 | 65.0 | 65.0 | 82.0 |

## Mean over seeds
| dataset | static_hec20 | static_hec8 | static_cf20 | pool_uniform64 | knn8 | class_cond8 | router8 | router_thresh | router_soft | hybrid_static_router | router_soft_p20 | oracle8 | oracle_corr8 | oracle_soft |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dtd | 83.0 | 83.3 | 82.0 | 62.3 | 62.0 | 60.7 | 65.3 | 65.3 | 82.7 | 83.3 | 83.0 | 63.3 | 64.0 | 88.7 |

## Head usage distribution (top-10 heads per dynamic method)
- **dtd / usage_knn8_global**: 693 (L24.H21), 698 (L24.H26), 646 (L23.H2), 750 (L26.H22), 772 (L27.H16), 645 (L23.H1), 752 (L26.H24), 633 (L22.H17), 644 (L23.H0), 689 (L24.H17)
- **dtd / usage_class_cond8_global**: 698 (L24.H26), 693 (L24.H21), 772 (L27.H16), 691 (L24.H19), 697 (L24.H25), 780 (L27.H24), 729 (L26.H1), 728 (L26.H0), 645 (L23.H1), 628 (L22.H12)
- **dtd / usage_router8_global**: 780 (L27.H24), 725 (L25.H25), 693 (L24.H21), 697 (L24.H25), 698 (L24.H26), 646 (L23.H2), 633 (L22.H17), 750 (L26.H22)
- **dtd / router8 query-set diversity** (mean pairwise jaccard) = 1.00 vs support-static-8 0.15 (lower = more input-adaptive)
- **dtd / class_cond8 set sizes**: mean 4.0, min 4, max 4 (adaptive budget <= 8)