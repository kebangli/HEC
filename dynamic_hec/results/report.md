# Dynamic Head Routing HEC — results

| dataset | seed | ways | static_hec20 | static_hec8 | static_cf20 | knn8 | class_cond8 | router8 | router_soft | oracle8 |
|---|---|---|---|---|---|---|---|---|---|---|
| oxford_pets | 123 | 10 | 96.0 | 96.0 | 96.0 | 76.0 | 72.0 | 82.0 | 96.0 | 78.0 |

## Mean over seeds
| dataset | static_hec20 | static_hec8 | static_cf20 | knn8 | class_cond8 | router8 | router_soft | oracle8 |
|---|---|---|---|---|---|---|---|---|
| oxford_pets | 96.0 | 96.0 | 96.0 | 76.0 | 72.0 | 82.0 | 96.0 | 78.0 |

## Head usage distribution (top-10 heads per dynamic method)
- **oxford_pets / usage_knn8_global**: 612 (L21.H24), 613 (L21.H25), 744 (L26.H16), 449 (L16.H1), 550 (L19.H18), 602 (L21.H14), 772 (L27.H16), 763 (L27.H7), 769 (L27.H13), 768 (L27.H12)
- **oxford_pets / usage_class_cond8_global**: 752 (L26.H24), 612 (L21.H24), 754 (L26.H26), 750 (L26.H22), 613 (L21.H25), 772 (L27.H16), 628 (L22.H12), 744 (L26.H16), 645 (L23.H1), 646 (L23.H2)
- **oxford_pets / usage_router8_global**: 772 (L27.H16), 612 (L21.H24), 744 (L26.H16), 613 (L21.H25), 750 (L26.H22), 752 (L26.H24), 754 (L26.H26), 645 (L23.H1), 646 (L23.H2)
- **oxford_pets / router8 query-set diversity** (mean pairwise jaccard) = 0.90 vs support-static-8 0.28 (lower = more input-adaptive)
- **oxford_pets / class_cond8 set sizes**: mean 4.0, min 4, max 4 (adaptive budget <= 8)