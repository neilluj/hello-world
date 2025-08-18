import pandas as pd
import numpy as np
import statsmodels.api as sm

# ---- 1. Cargar datos ----
df = pd.read_csv('datos.csv')

# ---- 2. Definir variables ----
X = df[['x1', 'x2']]
X = sm.add_constant(X)
y = df['y']

# ---- 3. Ajustar modelo logístico ----
model = sm.Logit(y, X)
result = model.fit()
print(result.summary())

# ---- 4. Función bootstrap para coeficientes ----
def bootstrap_summary(model, X, y, n_boot=1000, random_state=42):
    np.random.seed(random_state)
    coefs = []
    n = len(y)
    
    for _ in range(n_boot):
        idx = np.random.choice(np.arange(n), size=n, replace=True)
        X_boot = X.iloc[idx]
        y_boot = y.iloc[idx]
        model_boot = sm.Logit(y_boot, X_boot)
        try:
            res_boot = model_boot.fit(disp=0)
            coefs.append(res_boot.params.values)
        except:
            continue
    
    coefs = np.array(coefs)
    
    # Errores estándar bootstrap
    se_boot = pd.Series(coefs.std(axis=0), index=X.columns)
    
    # Intervalos de confianza 2.5% y 97.5%
    ci_lower = pd.Series(np.percentile(coefs, 2.5, axis=0), index=X.columns)
    ci_upper = pd.Series(np.percentile(coefs, 97.5, axis=0), index=X.columns)
    
    # Z-values y p-values aproximados
    z_boot = result.params / se_boot
    p_boot = 2 * (1 - np.abs(sm.distributions.norm.cdf(z_boot)))
    
    # Odds ratios y sus intervalos
    odds_ratio = np.exp(result.params)
    ci_lower_or = np.exp(ci_lower)
    ci_upper_or = np.exp(ci_upper)
    
    # Crear DataFrame consolidado
    df_combined = pd.DataFrame({
        'coef_original': result.params,
        'se_original': result.bse,
        'se_bootstrap': se_boot,
        'ci_lower_2.5%': ci_lower,
        'ci_upper_97.5%': ci_upper,
        'z_boot': z_boot,
        'p_boot': p_boot,
        'odds_ratio': odds_ratio,
        'ci_lower_OR': ci_lower_or,
        'ci_upper_OR': ci_upper_or
    })
    
    return df_combined

# ---- 5. Calcular y mostrar DataFrame final ----
df_final = bootstrap_summary(model, X, y, n_boot=1000)
print("\nResumen completo consolidado:\n", df_final)