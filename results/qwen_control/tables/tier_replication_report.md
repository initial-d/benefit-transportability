# Tier-routing replication report

## Qwen3.5 35B -> 397B

- Campaign: `qwen35b_122b_397b_thinking_on8192_verify_all_full`
- Complete paired items: 1900
- Overall small accuracy: 91.3%
- Overall large accuracy: 95.8%
- Overall marginal benefit: 4.5 pp, 95% bootstrap CI [3.3, 5.8] pp
- Rescues / regressions: 126 / 40
- Historical policy: `{"assignment": "large", "optimization": "large", "probability": "small", "routing": "large", "temporal": "large"}`
- Target-audit policy: `{"assignment": "large", "optimization": "large", "probability": "small", "routing": "large", "temporal": "small"}`

| Env | n | lower acc | upper acc | benefit pp | 95% CI | rescues | regressions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 200 | 87.0 | 91.5 | 4.5 | [0.5, 8.5] | 13 | 4 |
| B | 300 | 97.3 | 99.0 | 1.7 | [-0.3, 4.0] | 8 | 3 |
| C | 300 | 90.0 | 96.7 | 6.7 | [3.3, 10.3] | 25 | 5 |
| D | 500 | 94.0 | 97.0 | 3.0 | [0.6, 5.6] | 28 | 13 |
| E_audit | 100 | 86.0 | 92.0 | 6.0 | [0.0, 13.0] | 8 | 2 |
| E_test | 500 | 88.6 | 94.8 | 6.2 | [3.2, 9.2] | 44 | 13 |

| Family | historical ABC benefit pp | E_test benefit pp | gap pp |
| --- | ---: | ---: | ---: |
| assignment | 1.9 | -3.0 | -4.9 |
| optimization | 8.1 | 22.0 | 13.9 |
| probability | 0.0 | 0.0 | 0.0 |
| routing | 3.1 | 12.0 | 8.9 |
| temporal | 8.1 | 0.0 | -8.1 |

| E_test policy | acc | mean total tok | mean reasoning tok | tier counts |
| --- | ---: | ---: | ---: | --- |
| uniform_lowest | 88.6 | 12823.5 | 7927.7 | `{"small": 500}` |
| uniform_highest | 94.8 | 10760.9 | 7099.3 | `{"large": 500, "small": 0}` |
| historical_ABC | 94.8 | 11092.6 | 7297.4 | `{"large": 400, "small": 100}` |
| target_audit | 94.8 | 11793.5 | 7729.2 | `{"large": 300, "small": 200}` |

## GPT-5.6 Luna -> Sol

- Campaign: `gpt56_luna_terra_sol_medium_answeronly_hard20_full`
- Complete paired items: 600
- Overall luna accuracy: 94.5%
- Overall sol accuracy: 95.2%
- Overall marginal benefit: 0.7 pp, 95% bootstrap CI [-1.2, 2.7] pp
- Rescues / regressions: 20 / 16
- Historical policy: `{"assignment": "luna", "optimization": "sol", "probability": "terra", "routing": "luna", "temporal": "luna"}`
- Target-audit policy: `{"assignment": "terra", "optimization": "sol", "probability": "luna", "routing": "terra", "temporal": "luna"}`

| Env | n | lower acc | upper acc | benefit pp | 95% CI | rescues | regressions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 100 | 88.0 | 86.0 | -2.0 | [-7.0, 3.0] | 2 | 4 |
| B | 100 | 96.0 | 93.0 | -3.0 | [-9.0, 3.0] | 3 | 6 |
| C | 100 | 93.0 | 96.0 | 3.0 | [-3.0, 9.0] | 6 | 3 |
| D | 100 | 94.0 | 100.0 | 6.0 | [2.0, 11.0] | 6 | 0 |
| E_audit | 100 | 97.0 | 97.0 | 0.0 | [-4.0, 4.0] | 2 | 2 |
| E_test | 100 | 99.0 | 99.0 | 0.0 | [-3.0, 3.0] | 1 | 1 |

| Family | historical ABC benefit pp | E_test benefit pp | gap pp |
| --- | ---: | ---: | ---: |
| assignment | -1.7 | 0.0 | 1.7 |
| optimization | 6.7 | 0.0 | -6.7 |
| probability | -5.0 | 0.0 | 5.0 |
| routing | -1.7 | 0.0 | 1.7 |
| temporal | -1.7 | 0.0 | 1.7 |

| E_test policy | acc | mean total tok | mean reasoning tok | tier counts |
| --- | ---: | ---: | ---: | --- |
| uniform_lowest | 99.0 | 1086.7 | 537.6 | `{"luna": 100}` |
| uniform_highest | 99.0 | 836.7 | 290.7 | `{"luna": 0, "sol": 100}` |
| historical_ABC | 99.0 | 998.5 | 450.6 | `{"luna": 60, "sol": 20, "terra": 20}` |
| target_audit | 96.0 | 897.3 | 350.4 | `{"luna": 40, "sol": 20, "terra": 40}` |
