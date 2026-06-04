# fisher_test.py
from scipy.stats import fisher_exact

# Defense boundary experiment: 4 scenarios × 20 trials = 80 total per condition
direct_success = round(0.038 * 80)   # 3
direct_fail = 80 - direct_success     # 77
tl_success = round(0.238 * 80)        # 19
tl_fail = 80 - tl_success             # 61

table = [[direct_success, direct_fail],
         [tl_success, tl_fail]]

odds_ratio, p_value = fisher_exact(table, alternative='less')
print(f"Direct sanitized: {direct_success}/80 = {direct_success/80:.1%}")
print(f"Trust laundered:  {tl_success}/80 = {tl_success/80:.1%}")
print(f"Fisher exact p = {p_value:.6f}")
print(f"Odds ratio = {odds_ratio:.3f}")