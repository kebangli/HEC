# Dynamic Head Routing HEC — results

| dataset | seed | ways | static_hec20 | static_hec8 | static_cf20 | pool_uniform64 | knn8 | class_cond8 | router8 | router_thresh | router_soft | hybrid_static_router | router_soft_p20 | oracle8 | oracle_corr8 | oracle_soft |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| oxford_pets | 123 | 20 | 95.0 | 95.0 | 95.0 | 83.0 | 85.0 | 81.0 | 84.0 | 84.0 | 95.0 | 95.0 | 94.0 | 83.0 | 83.0 | 98.0 |
| oxford_pets | 148 | 20 | 98.0 | 98.0 | 99.0 | 84.0 | 79.0 | 73.0 | 80.0 | 80.0 | 98.0 | 98.0 | 99.0 | 80.0 | 80.0 | 100.0 |
| oxford_pets | 173 | 20 | 97.0 | 97.0 | 96.0 | 74.0 | 77.0 | 76.0 | 79.0 | 79.0 | 96.0 | 96.0 | 96.0 | 74.0 | 73.0 | 98.0 |

## Mean over seeds
| dataset | static_hec20 | static_hec8 | static_cf20 | pool_uniform64 | knn8 | class_cond8 | router8 | router_thresh | router_soft | hybrid_static_router | router_soft_p20 | oracle8 | oracle_corr8 | oracle_soft |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| oxford_pets | 96.7 | 96.7 | 96.7 | 80.3 | 80.3 | 76.7 | 81.0 | 81.0 | 96.3 | 96.3 | 96.3 | 79.0 | 78.7 | 98.7 |

## Head usage distribution (top-10 heads per dynamic method)
- **oxford_pets / usage_knn8_global**: 775 (L27.H19), 713 (L25.H13), 772 (L27.H16), 633 (L22.H17), 698 (L24.H26), 647 (L23.H3), 728 (L26.H0), 729 (L26.H1), 651 (L23.H7), 747 (L26.H19)
- **oxford_pets / usage_class_cond8_global**: 651 (L23.H7), 633 (L22.H17), 713 (L25.H13), 698 (L24.H26), 647 (L23.H3), 695 (L24.H23), 772 (L27.H16), 775 (L27.H19), 646 (L23.H2), 661 (L23.H17)
- **oxford_pets / usage_router8_global**: 775 (L27.H19), 772 (L27.H16), 713 (L25.H13), 695 (L24.H23), 698 (L24.H26), 647 (L23.H3), 651 (L23.H7), 633 (L22.H17)
- **oxford_pets / router8 query-set diversity** (mean pairwise jaccard) = 1.00 vs support-static-8 0.46 (lower = more input-adaptive)
- **oxford_pets / class_cond8 set sizes**: mean 4.0, min 4, max 4 (adaptive budget <= 8)