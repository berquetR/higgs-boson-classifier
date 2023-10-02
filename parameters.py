# Optimal parameters for every implementation

# Contains the gammas for Gradient Descent and Stochastic Gradient Descent to be used for x grouped by jet number
ls_gd_params = [0.075, 0.085, 0.061, 0.053]
ls_sgd_params = [0.008, 0.007, 0.003, 0.001]

# Contains the degree expansion for Least Squares to be used for x grouped by jet number
ls_params = [1, 1, 1, 1]

# Contains the couples (lambda, degree) for Ridge Regression to be used for x grouped by jet number
ridge_params = [(0.1, 6), (1.e-6, 5), (1.e-6, 6), (1.e-5, 6)]

# Contains the couples (lambda, degree, gamma) for Penalised Ridge Regression to be used for x grouped by jet number
pen_log_reg_params = [
    (0., 11, 0.2),
    (0., 10, 0.2),
    (0., 12, 0.2),
    (0., 12, 0.06)
]

