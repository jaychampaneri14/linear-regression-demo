"""
Linear Regression Demo
Comprehensive tutorial covering simple, multiple, polynomial, Ridge, and Lasso regression.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')


# ─── 1. SIMPLE LINEAR REGRESSION ─────────────────────────────────────────────
def demo_simple_lr():
    print("\n" + "="*50)
    print("1. SIMPLE LINEAR REGRESSION")
    print("="*50)
    np.random.seed(42)
    X = np.linspace(0, 10, 100).reshape(-1, 1)
    y = 3.5 * X.ravel() + 7 + np.random.normal(0, 3, 100)

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    r2   = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    print(f"  Slope: {model.coef_[0]:.4f}  (true: 3.5)")
    print(f"  Intercept: {model.intercept_:.4f}  (true: 7.0)")
    print(f"  R²={r2:.4f}, RMSE={rmse:.4f}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(X, y, alpha=0.6, s=20, label='Data')
    ax.plot(X, y_pred, 'r-', lw=2, label=f'y = {model.coef_[0]:.2f}x + {model.intercept_:.2f}')
    ax.set_title('Simple Linear Regression')
    ax.set_xlabel('X'); ax.set_ylabel('y')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('simple_lr.png', dpi=150)
    plt.close()
    return model


# ─── 2. MULTIPLE LINEAR REGRESSION ────────────────────────────────────────────
def demo_multiple_lr():
    print("\n" + "="*50)
    print("2. MULTIPLE LINEAR REGRESSION")
    print("="*50)
    np.random.seed(42)
    n = 500
    X1   = np.random.uniform(0, 10, n)
    X2   = np.random.uniform(0, 5, n)
    X3   = np.random.uniform(0, 3, n)
    X4   = np.random.uniform(-2, 2, n)          # noise feature
    y    = 2*X1 + 3*X2 - 1.5*X3 + 5 + np.random.normal(0, 2, n)
    X    = np.column_stack([X1, X2, X3, X4])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"  Coefficients: {dict(zip(['X1','X2','X3','X4'], model.coef_.round(3)))}")
    print(f"  True coefs:   X1=2.0, X2=3.0, X3=-1.5, X4=0.0")
    print(f"  R²={r2:.4f}")

    plt.figure(figsize=(6, 5))
    plt.scatter(y_test, y_pred, alpha=0.4, s=15)
    mn, mx = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    plt.plot([mn, mx], [mn, mx], 'r--', lw=2)
    plt.xlabel('Actual'); plt.ylabel('Predicted')
    plt.title('Multiple Linear Regression')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('multiple_lr.png', dpi=150)
    plt.close()
    return model


# ─── 3. POLYNOMIAL REGRESSION ─────────────────────────────────────────────────
def demo_polynomial_lr():
    print("\n" + "="*50)
    print("3. POLYNOMIAL REGRESSION & OVERFITTING")
    print("="*50)
    np.random.seed(42)
    X = np.linspace(-3, 3, 80).reshape(-1, 1)
    y = 0.5*X.ravel()**3 - 2*X.ravel()**2 + X.ravel() + 3 + np.random.normal(0, 2, 80)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    degrees = [1, 2, 3, 5, 9]
    X_plot  = np.linspace(-3, 3, 300).reshape(-1, 1)
    colors  = ['blue', 'green', 'red', 'purple', 'orange']

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(X_train, y_train, s=20, alpha=0.7, label='Train')
    plt.scatter(X_test, y_test, s=20, alpha=0.7, marker='^', label='Test')
    for deg, color in zip(degrees, colors):
        pipe = Pipeline([('poly', PolynomialFeatures(deg)), ('lr', LinearRegression())])
        pipe.fit(X_train, y_train)
        y_plot = pipe.predict(X_plot)
        tr_r2 = r2_score(y_train, pipe.predict(X_train))
        te_r2 = r2_score(y_test, pipe.predict(X_test))
        print(f"  Degree {deg}: Train R²={tr_r2:.3f}, Test R²={te_r2:.3f}")
        plt.plot(X_plot, y_plot.clip(-50, 50), color=color, lw=1.5, label=f'deg={deg}')
    plt.ylim(-20, 20); plt.legend(fontsize=7); plt.title('Polynomial Degrees')
    plt.grid(True, alpha=0.3)

    # Best degree
    plt.subplot(1, 2, 2)
    pipe3 = Pipeline([('poly', PolynomialFeatures(3)), ('lr', LinearRegression())])
    pipe3.fit(X_train, y_train)
    plt.scatter(X, y, s=15, alpha=0.5)
    plt.plot(X_plot, pipe3.predict(X_plot), 'r-', lw=2, label='Degree 3 (best)')
    plt.title('Best Polynomial Fit (degree=3)'); plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('polynomial_lr.png', dpi=150)
    plt.close()


# ─── 4. REGULARIZATION ────────────────────────────────────────────────────────
def demo_regularization():
    print("\n" + "="*50)
    print("4. REGULARIZATION: RIDGE, LASSO, ELASTICNET")
    print("="*50)
    np.random.seed(42)
    n, p = 200, 50
    # Only first 5 features are relevant
    X = np.random.randn(n, p)
    true_coef = np.zeros(p)
    true_coef[:5] = [3, -2, 1.5, -1, 0.8]
    y = X @ true_coef + np.random.normal(0, 1, n)

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_s, y, test_size=0.2, random_state=42)

    models = {
        'OLS':        LinearRegression(),
        'Ridge(α=1)': Ridge(alpha=1.0),
        'Ridge(α=10)':Ridge(alpha=10.0),
        'Lasso(α=0.1)':Lasso(alpha=0.1),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2   = r2_score(y_test, y_pred)
        coef_sparsity = (np.abs(model.coef_) < 0.01).mean()
        results[name] = {'r2': r2, 'sparsity': coef_sparsity, 'model': model}
        print(f"  {name:18s}: R²={r2:.4f}, Sparsity={coef_sparsity:.1%}")

    # Plot coefficient paths
    alphas = np.logspace(-3, 3, 50)
    ridge_coefs = [Ridge(alpha=a).fit(X_train, y_train).coef_ for a in alphas]
    lasso_coefs = [Lasso(alpha=a, max_iter=5000).fit(X_train, y_train).coef_ for a in alphas]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for i in range(p):
        ax1.plot(np.log10(alphas), [c[i] for c in ridge_coefs],
                 color='red' if i < 5 else 'lightblue', alpha=0.4 if i >= 5 else 1.0)
        ax2.plot(np.log10(alphas), [c[i] for c in lasso_coefs],
                 color='red' if i < 5 else 'lightblue', alpha=0.4 if i >= 5 else 1.0)
    ax1.set_title('Ridge Coefficient Paths'); ax1.set_xlabel('log10(alpha)'); ax1.set_ylabel('Coefficient')
    ax2.set_title('Lasso Coefficient Paths'); ax2.set_xlabel('log10(alpha)')
    plt.suptitle('Regularization Paths (red = true features)')
    plt.tight_layout()
    plt.savefig('regularization_paths.png', dpi=150)
    plt.close()
    print("  Regularization path plots saved.")


# ─── 5. LEARNING CURVES ────────────────────────────────────────────────────────
def demo_learning_curves():
    print("\n" + "="*50)
    print("5. LEARNING CURVES")
    print("="*50)
    np.random.seed(42)
    X = np.random.randn(500, 5)
    y = 2 * X[:, 0] + 3 * X[:, 1] - X[:, 2] + np.random.normal(0, 1, 500)

    model = Ridge(alpha=1.0)
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=5, scoring='r2',
        train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
    )
    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, train_scores.mean(axis=1), 'o-', color='blue', label='Train R²')
    plt.fill_between(train_sizes,
                     train_scores.mean(axis=1) - train_scores.std(axis=1),
                     train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.2, color='blue')
    plt.plot(train_sizes, val_scores.mean(axis=1), 'o-', color='red', label='Validation R²')
    plt.fill_between(train_sizes,
                     val_scores.mean(axis=1) - val_scores.std(axis=1),
                     val_scores.mean(axis=1) + val_scores.std(axis=1), alpha=0.2, color='red')
    plt.xlabel('Training Set Size')
    plt.ylabel('R² Score')
    plt.title('Learning Curves — Ridge Regression')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('learning_curves.png', dpi=150)
    plt.close()
    print("  Learning curves saved.")


def main():
    print("=" * 60)
    print("LINEAR REGRESSION COMPREHENSIVE DEMO")
    print("=" * 60)

    demo_simple_lr()
    demo_multiple_lr()
    demo_polynomial_lr()
    demo_regularization()
    demo_learning_curves()

    print("\n--- Summary of Output Files ---")
    for f in ['simple_lr.png', 'multiple_lr.png', 'polynomial_lr.png',
              'regularization_paths.png', 'learning_curves.png']:
        print(f"  {f}")

    print("\n✓ Linear Regression Demo complete!")


if __name__ == '__main__':
    main()
