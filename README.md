# kalman_info_consistency

A minimal project to validate the consistency between covariance-form and information-form updates of the Kalman filter.

## Structure

```
kalman_info_consistency/
├─ src/
│  ├─ __init__.py
│  ├─ kalman.py
│  ├─ covariance.py
│  ├─ info.py
│  └─ linalg_utils.py
├─ tests/
│  └─ test_consistency.py
├─ requirements.txt
└─ README.md
```

## Model Settings

- State `[x, v]^T`
- Transition `A = [[1, 1], [0, 1]]`
- Process noise `Q = diag(1e-5, 1e-5)`
- Measurement noise `R_t = [[1e-3]]`
- Initial covariance `P0 = diag(1, 1)`
- All computations use NumPy float64
- Time-varying observation matrix: even `t` observe position, odd `t` observe velocity

## Run Tests

Install dependencies and run pytest:

```
pip install -r kalman_info_consistency/requirements.txt
pytest -q
```
