import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error


#Data generation
np.random.seed(42)

X_train = np.linspace(-1, 1, 50).reshape(-1, 1)

y_target = np.sinc(3 * X_train).ravel()

noise = np.random.normal(0, 0.1, size=y_target.shape)
y_train = y_target + noise

X_plot = np.linspace(-1, 1, 500).reshape(-1, 1)

print("Starting SVR Grid Search...")


#Parameters for the SVR cross validation
svr_param_grid = {
    'C': [1, 10, 100, 1000],
    'gamma': [1, 5, 10, 20, 'scale'],
    'epsilon': [0.01, 0.05, 0.1, 0.2]
}

svr = SVR(kernel='rbf')
cv = KFold(n_splits=5, shuffle=True, random_state=42)


#cross validation for SVR
grid_svr = GridSearchCV(estimator=svr,
                        param_grid=svr_param_grid,
                        cv=cv,
                        scoring='neg_mean_squared_error',
                        n_jobs=-1)

grid_svr.fit(X_train, y_train)

best_svr_params = grid_svr.best_params_
best_svr_model = grid_svr.best_estimator_
best_svr_score = -grid_svr.best_score_

y_pred_svr = best_svr_model.predict(X_plot)
svr_mse_train = mean_squared_error(y_train, best_svr_model.predict(X_train))

print(f"Best SVR Params: {best_svr_params}")
print(f"SVR CV MSE: {best_svr_score:.6f}")
print(f"SVR Final MSE: {svr_mse_train:.6f}")

print("\nStarting Ridge Polynomial Grid Search...")

#LINEAR REGRESSION

pipeline = make_pipeline(PolynomialFeatures(), Ridge())

ridge_param_grid = {
    'polynomialfeatures__degree': np.arange(1, 16),
    'ridge__alpha': np.logspace(-4, 3, 20)
}

#cross validation for Ridge regression
grid_ridge = GridSearchCV(estimator=pipeline,
                          param_grid=ridge_param_grid,
                          cv=cv,
                          scoring='neg_mean_squared_error',
                          n_jobs=-1)

grid_ridge.fit(X_train, y_train)

best_ridge_params = grid_ridge.best_params_
best_ridge_model = grid_ridge.best_estimator_
best_ridge_score = -grid_ridge.best_score_

y_pred_ridge = best_ridge_model.predict(X_plot)
ridge_mse_train = mean_squared_error(y_train, best_ridge_model.predict(X_train))

print(f"Best Ridge Params: {best_ridge_params}")
print(f"Ridge CV MSE: {best_ridge_score:.6f}")
print(f"Ridge Final MSE: {ridge_mse_train:.6f}")

plt.rcParams['figure.figsize'] = (20, 8)
plt.rcParams['font.size'] = 11
plt.style.use('seaborn-v0_8-whitegrid')

fig, (ax1, ax2) = plt.subplots(1, 2)

# Plot SVR
ax1.scatter(X_train, y_train, color='gray', s=30, alpha=0.6, label='Raw Data')
ax1.plot(X_plot, y_pred_svr, color='#d62728', linewidth=2.5, label='Best SVR Model')

eps = best_svr_params['epsilon']
ax1.plot(X_plot, y_pred_svr + eps, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax1.plot(X_plot, y_pred_svr - eps, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax1.fill_between(X_plot.ravel(), y_pred_svr - eps, y_pred_svr + eps, color='red', alpha=0.1, label=f'Tube $\epsilon={eps}$')

sv_indices = best_svr_model.support_
ax1.scatter(X_train[sv_indices], y_train[sv_indices],
            s=80, facecolors='none', edgecolors='black', linewidth=1.5,
            zorder=10, label=f'Support Vectors ({len(sv_indices)})')

title_svr = (f"SVR Optimal: C={best_svr_params['C']}, $\gamma$={best_svr_params['gamma']}, $\epsilon$={best_svr_params['epsilon']}\n"
             f"MSE: {svr_mse_train:.5f}")
ax1.set_title(title_svr)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.legend(loc='upper right', frameon=True)
ax1.set_ylim(-0.8, 1.2)

# Plot Ridge
ax2.scatter(X_train, y_train, color='gray', s=30, alpha=0.6, label='Raw Data')
ax2.plot(X_plot, y_pred_ridge, color='blue', linewidth=2.5, label='Best Ridge Poly Model')

degree = best_ridge_params['polynomialfeatures__degree']
alpha = best_ridge_params['ridge__alpha']
title_ridge = (f"Ridge Optimal: Degree={degree}, $\\alpha$={alpha:.4f}\n"
               f"MSE: {ridge_mse_train:.5f}")
ax2.set_title(title_ridge)
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.legend(loc='upper right', frameon=True)
ax2.set_ylim(-0.8, 1.2)

plt.tight_layout()

plt.show()
