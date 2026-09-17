# Stroke classifier evaluation: 2025-01-10_wingfield

- session 2025-01-10_wingfield: classifier rule-1, left-handed
- 144 labels and 194 classified swings in range; 84 matched pairs scored (60 labels and 110 swings unmatched)
- accuracy 0.583 (49/84), macro F1 0.397 (does not meet the 90% target)
- 19 of the classified swings were two-handed

| truth \ predicted | serve | forehand | backhand | volley | support | recall |
|---|---|---|---|---|---|---|
| serve | 7 | 12 | 7 | 0 | 26 | 0.269 |
| forehand | 0 | 33 | 4 | 0 | 37 | 0.892 |
| backhand | 0 | 10 | 9 | 0 | 19 | 0.474 |
| volley | 0 | 2 | 0 | 0 | 2 | 0.000 |
| precision | 1.000 | 0.579 | 0.450 | 0.000 | 84 | 0.583 |
