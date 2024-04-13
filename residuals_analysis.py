import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy import stats
import seaborn as sns

# サンプルデータ(アンスコムの数値例)
X1 = np.array([10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]) 
Y1 = np.array([8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68])
Y2 = np.array([9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74])
Y3 = np.array([7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73])
X2 = np.array([8, 8, 8, 8, 8, 8, 8, 19, 8 ,8 ,8])
Y4 = np.array([6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.5, 5.56, 7.91, 6.89])

def residuals_analysis_plots(X, Y, Xlabel, Ylabel, filename):
    X = X.reshape(-1, 1)
    # 線形回帰モデルの構築とフィッティング
    model = LinearRegression()
    model.fit(X, Y)
    # 回帰直線の描画
    plt.scatter(X, Y, color='blue')
    plt.plot(X, model.predict(X.reshape(-1, 1)), color='red')
    plt.xlabel(Xlabel)
    plt.ylabel(Ylabel)
    # x軸とy軸の範囲と間隔を指定
    plt.xticks(np.arange(0, 21, 5), fontsize=12)
    plt.yticks(np.arange(0, 16, 5), fontsize=12)
    #plt.title('Linear Regression for Y1')
    # 回帰係数、切片、決定係数の値を取得
    coefficients = model.coef_
    intercept = model.intercept_
    r_squared = model.score(X.reshape(-1, 1), Y)
    # 回帰係数、切片、決定係数をグラフ内に書き込む
    plt.text(3, 13, f'Regression Coefficients: {coefficients[0]:.2f}', fontsize=10, color='black')
    plt.text(3, 12, f'Intercept: {intercept:.2f}', fontsize=10, color='black')
    plt.text(3, 11, f'R-squared: {r_squared:.2f}', fontsize=10, color='black')
    # グラフをPNGファイルとして保存
    plt.savefig(rf'./linear_regression_plot_{filename}.png', facecolor='white', dpi=150)
    plt.show()
    # Yの推定値
    Y_pred = model.predict(X)
    # 残差
    residuals = Y - Y_pred
    # 標準化された残差
    standardized_residuals = residuals / np.std(residuals, ddof=1)
    # レバレッジ（てこ比）の計算
    H = X @ np.linalg.inv(X.T @ X) @ X.T
    leverage = np.diag(H)
    # クックの距離の計算
    cook_distance = (standardized_residuals ** 2) * leverage / (1 - leverage) / X.shape[0]
    # クックの距離が0.5と1になる標準化残差を逆算
    standardized_residual_05 = np.sqrt(0.5 * (1 - leverage)/ leverage * (X.shape[1] + 1))
    standardized_residual_1 = np.sqrt(1 * (1 - leverage)/ leverage * (X.shape[1] + 1))
    # leverageとstandardized_residual_05を結合して2次元のnumpy配列を作成
    data = np.column_stack((leverage, standardized_residual_05, -standardized_residual_05))
    # xに関してソート
    sorted_data = data[data[:,0].argsort()]
    # ダービン・ワトソン統計量の計算
    residuals_diff = np.diff(residuals)
    dw_statistic = np.sum(residuals_diff**2) / np.sum(residuals**2)
    # グラフの作成
    plt.figure(figsize=(18, 12))
    # グラフ1: サンプルNo.を横軸として、残差eをプロット
    plt.subplot(2, 3, 1)
    plt.plot(np.arange(1, 12), residuals, 'o-')
    plt.xlabel('Sample No.')
    plt.ylabel('Residuals')
    plt.text(1, 1.3, f'Durbin-Watson: {dw_statistic:.2f}', fontsize=10, color='black')
    plt.title('Residuals vs. Sample No.')
    # LOWESS平滑化の線を追加
    lowess_results = sm.nonparametric.lowess(residuals, np.arange(1, 12))
    plt.plot(lowess_results[:,0], lowess_results[:, 1], color='orange', linewidth=2)
    # グラフ2: Yの推定値を横軸として、残差eをプロット
    plt.subplot(2, 3, 2)
    plt.plot(Y_pred, residuals, 'o')
    plt.xlabel(f'Predicted {Ylabel}')
    plt.ylabel('Residuals')
    plt.title(f'Residuals vs. Predicted {Ylabel}')
    # LOWESS平滑化の線を追加
    lowess_results = sm.nonparametric.lowess(residuals, Y_pred)
    lowess_results_sorted = lowess_results[lowess_results[:, 0].argsort()]
    plt.plot(lowess_results_sorted[:, 0], lowess_results_sorted[:, 1], color='orange', linewidth=2)
    # グラフ3: Xを横軸として、残差eをプロット
    plt.subplot(2, 3, 3)
    plt.plot(X, residuals, 'o', label=Xlabel)
    plt.xlabel(Xlabel)
    plt.ylabel('Residuals')
    plt.title(f'Residuals vs. {Xlabel}')
    plt.legend()
    # LOWESS平滑化の線を追加
    lowess_results = sm.nonparametric.lowess(residuals, X[:,0])
    lowess_results_sorted = lowess_results[lowess_results[:, 0].argsort()]
    plt.plot(lowess_results_sorted[:,0], lowess_results_sorted[:, 1], color='orange', linewidth=2)
    # グラフ4: 残差eの正規確率プロット
    plt.subplot(2, 3, 4)
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Normal Probability Plot')
    # グラフ5: 標準化された残差の絶対値の平方根をy軸とし、Yの推定値をx軸とするグラフ
    plt.subplot(2, 3, 5)
    plt.plot(Y_pred, np.sqrt(np.abs(standardized_residuals)), 'o')
    plt.xlabel(f'Predicted {Ylabel}')
    plt.ylabel('Root Absolute Standardized Residuals')
    plt.title(f'Root Absolute Standardized Residuals vs. Predicted {Ylabel}')
    # LOWESS平滑化の線を追加
    lowess_results = sm.nonparametric.lowess(np.sqrt(np.abs(standardized_residuals)), Y_pred)
    lowess_results_sorted = lowess_results[lowess_results[:, 0].argsort()]
    plt.plot(lowess_results_sorted[:, 0], lowess_results_sorted[:, 1], color='orange', linewidth=2)
    # グラフ6: レバレッジをx軸に、標準化された残差をy軸とするグラフ
    plt.subplot(2, 3, 6)
    plt.plot(leverage, standardized_residuals, 'o')
    plt.xlabel('Leverage')
    plt.ylabel('Standardized Residuals')
    plt.title('Standardized Residuals vs. Leverage')
    # LOWESS平滑化の線を追加
    lowess_results = sm.nonparametric.lowess(standardized_residuals, leverage)
    lowess_results_sorted = lowess_results[lowess_results[:, 0].argsort()]
    plt.plot(lowess_results_sorted[:, 0], lowess_results_sorted[:, 1], color='orange', linewidth=2)
    # クックの距離を点線で表示
    plt.axhline(y=0, color='gray', linestyle='-')
    plt.plot(sorted_data[:, 0], sorted_data[:, 1], '--',color='black', label="cook's distance= 0.5")
    plt.plot(sorted_data[:, 0], sorted_data[:, 2], '--', color='black')
    plt.ylim([-8, 8])
    plt.legend()
    plt.tight_layout()
    # グラフをPNGファイルとして保存
    plt.savefig(rf'./residuals_analysis_plot_{filename}.png', facecolor='white', dpi=150)
    plt.show()

#残差分析を実行
residuals_analysis_plots(X1, Y1, Xlabel='X1', Ylabel='Y1', filename='X1_Y1')
residuals_analysis_plots(X1, Y2, Xlabel='X1', Ylabel='Y2', filename='X1_Y2')
residuals_analysis_plots(X1, Y3, Xlabel='X1', Ylabel='Y3', filename='X1_Y3')
residuals_analysis_plots(X2, Y4, Xlabel='X2', Ylabel='Y4', filename='X2_Y4')