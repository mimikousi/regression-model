import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn import metrics
from sklearn.model_selection import train_test_split, KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor

def load_data():
    df = pd.read_csv('debutanizer_data.csv', header=0)
    # 時系列データなので、実務データを想定しindexに時刻を割り当てる
    start_datetime = '2024-01-01 00:00:00'
    n = len(df)
    date_index = pd.date_range(start=start_datetime, periods=n, freq='min') 
    df.index = date_index
    # 目的変数の測定時間を考慮（5分間）
    df['y'] = df['y'].shift(5)
    df = df.dropna()
    return df

def evaluate_performance_std(X, y, y_train, model, save_to_csv=False, filename='performance.csv'):
    """
    モデルの性能を評価し、結果をプロットします。
    オプションで予測結果をCSVファイルに保存します。
    
    Parameters:
    X : pd.DataFrame
        説明変数のデータセット
    y : pd.Series
        目的変数の値
    y_train : pd.Series
        トレーニングデータの目的変数（標準化解除に使用）
    model : sklearn model
        評価する訓練済みのモデル
    save_to_csv : bool, optional
        結果をCSVファイルに保存するかどうかを指定 (default is False)
    filename : str, optional
        出力するCSVファイルの名前 (default is 'performance.csv')

    Returns:
    r2 : 決定係数R^2
    rmse : RMSE
    mae : MAE
        
    """
    # ｙの推定値を算出
    y_pred = model.predict(X) * y_train.std() + y_train.mean() 

    # 性能指標の算出
    r2 = metrics.r2_score(y, y_pred)
    rmse = np.sqrt(metrics.mean_squared_error(y, y_pred))
    mae = metrics.mean_absolute_error(y, y_pred)
    print(f"R^2: {r2:.3f}, RMSE: {rmse:.3f}, MAE: {mae:.3f}")
    
    # プロットの作成
    plt.figure(figsize=(6, 6))
    plt.scatter(y, y_pred, color='blue')
    y_max = max(y.max(), y_pred.max())
    y_min = min(y.min(), y_pred.min()) 
    
     # 取得した最小値-5%から最大値+5%まで、対角線を作成
    plt.plot([y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)],
             [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)], 'k-') 
    plt.ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
    plt.xlim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)) 
    plt.xlabel('Actual y')
    plt.ylabel('Predicted y')
    # テキストボックスを右下に配置
    plt.text(0.98, 0.02, f'R^2: {r2:.3f}\nRMSE: {rmse:.3f}\nMAE: {mae:.3f}', transform=plt.gca().transAxes,
             fontsize=9, verticalalignment='bottom', horizontalalignment='right', 
             bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white', alpha=0.5))
    plt.title(filename[:-4])
    plt.grid(False)
    plt.axis('equal')
    plt.show()

    # 時系列のプロット
    plt.figure(figsize=(10, 5))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))  # 日時形式を設定
    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=5))  # 4時間ごとに目盛りを設定
    plt.plot(y.index, y, label='Actual y', color='blue', marker='', linestyle='-')
    plt.plot(y.index, y_pred, label='Predicted y', color='red', marker='', linestyle='-')
    #plt.gcf().autofmt_xdate()  # x軸のラベルを自動で斜めにして読みやすくする
    plt.ylabel('y')
    plt.title(filename[:-4])
    plt.legend()
    plt.show()
    
    # データの保存
    if save_to_csv:
        results_df = pd.DataFrame({
            'Actual': y,
            'Predicted': y_pred.flatten(),
            'Error': y - y_pred.flatten()
        })
        results_df.to_csv(filename, index=False)
        print(f"Data saved to {filename}")
    
    return r2, rmse, mae

def evaluate_model_std(X_train_scaled, y_train, X_test_scaled, y_test, model, save_csv=False, train_filename='train_performance.csv', test_filename='test_performance.csv'):
    """
    トレーニングデータとテストデータの両方でモデルの性能を評価します。
    
    Parameters:
    X_train_scaled : pd.DataFrame
        標準化されたトレーニングデータの説明変数
    y_train : pd.Series
        トレーニングデータの目的変数
    X_test_scaled : pd.DataFrame
        標準化されたテストデータの説明変数
    y_test : pd.Series
        テストデータの目的変数
    model : sklearn model
        評価する訓練済みのモデル
    save_csv : bool, optional
        結果をCSVファイルに保存するかどうかを指定 (default is False)
    train_filename : str, optional
        トレーニングデータの予測結果を保存するCSVファイルの名前 (default is 'train_performance.csv')
    test_filename : str, optional
        テストデータの予測結果を保存するCSVファイルの名前 (default is 'test_performance.csv')
    """
    print("Evaluating Training Data")
    evaluate_performance_std(X_train_scaled, y_train, y_train, model, save_to_csv=save_csv, filename=train_filename)
    print("Evaluating Test Data")
    evaluate_performance_std(X_test_scaled, y_test, y_train, model, save_to_csv=save_csv, filename=test_filename)

def perform_pls_regression(X, y, fold_number=5, test_size=0.4, shuffle=False):
    """
    PLS回帰モデルを構築し、クロスバリデーションを用いて最適な主成分数を決定します。
    その後、最適な主成分数でモデルを再構築し、訓練データとテストデータの性能を評価します。

    Parameters:
    X : pd.DataFrame
        説明変数のデータセット
    y : pd.Series
        目的変数のデータ
    fold_number : int, optional
        クロスバリデーションで使用するフォールドの数 (default is 5)。
    test_size : float, optional
        データを分割する際のテストデータの割合 (default is 0.4)。
    shuffle : bool, optional
        データ分割時にデータをシャッフルするかどうか (時系列データを想定して、デフォルトはFalse)。

    Returns:
    best_components : 最適な主成分数
    standard_regression_coefficients : 標準回帰係数
    beta_train : 回帰係数
    intercept_train : 切片
    """
    # データの分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=shuffle, random_state=0)

    # データの標準化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    y_train_scaled = scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()

    # 主成分数を決定するためのクロスバリデーション
    max_components = X_train.shape[1]
    r2_scores = []
    for n_components in range(1, max_components + 1):
        pls = PLSRegression(n_components=n_components)
        y_cv = cross_val_predict(pls, X_train_scaled, y_train_scaled, cv=fold_number)
        r2 = metrics.r2_score(y_train_scaled, y_cv)
        r2_scores.append((n_components, r2))

    # r2_scoresをグラフで表示
    components = [item[0] for item in r2_scores]
    r2_values = [item[1] for item in r2_scores]
    plt.figure(figsize=(8, 5))
    plt.plot(components, r2_values, marker='o', linestyle='-', color='b')
    plt.xlabel('Number of Components')
    plt.ylabel('R2 Score (CV)')
    plt.grid(True)
    plt.show()

    # 最適な主成分数の決定
    best_components = max(r2_scores, key=lambda item: item[1])[0]

    # 最適な主成分数で再度PLSモデルを作成
    pls_final = PLSRegression(n_components=best_components)
    pls_final.fit(X_train_scaled, y_train_scaled)

    # 標準回帰係数の取得
    standard_regression_coefficients = pd.DataFrame(pls_final.coef_, index=X_train.columns,
                                                    columns=['Standard Regression Coefficients'])

    # 回帰係数、切片の算出
    std_X_train = X_train.std()
    std_y_train = y_train.std()
    beta_train = std_y_train * standard_regression_coefficients.iloc[:, 0] / std_X_train
    intercept_train = y_train.mean() - X_train.mean() @ beta_train

    evaluate_model_std(X_train_scaled, y_train, X_test_scaled, y_test, model=pls_final)

    return best_components, standard_regression_coefficients, beta_train, intercept_train

def evaluate_performance(X, y, model, save_to_csv=False, filename='performance.csv'):
    """
    モデルの性能を評価し、結果をプロットします。
    オプションで予測結果をCSVファイルに保存します。
    
    Parameters:
    X : pd.DataFrame
        説明変数のデータセット
    y : pd.Series
        目的変数の値
    model : sklearn model
        評価する訓練済みのモデル
    save_to_csv : bool, optional
        結果をCSVファイルに保存するかどうかを指定 (default is False)
    filename : str, optional
        出力するCSVファイルの名前 (default is 'performance.csv')

    Returns:
    r2 : 決定係数R^2
    rmse : RMSE
    mae : MAE
        
    """
    # ｙの推定値を算出
    y_pred = model.predict(X)

    # 性能指標の算出
    r2 = metrics.r2_score(y, y_pred)
    rmse = np.sqrt(metrics.mean_squared_error(y, y_pred))
    mae = metrics.mean_absolute_error(y, y_pred)
    print(f"R^2: {r2:.3f}, RMSE: {rmse:.3f}, MAE: {mae:.3f}")
    
    # プロットの作成
    plt.figure(figsize=(6, 6))
    plt.scatter(y, y_pred, color='blue')
    y_max = max(y.max(), y_pred.max())
    y_min = min(y.min(), y_pred.min()) 
    
     # 取得した最小値-5%から最大値+5%まで、対角線を作成
    plt.plot([y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)],
             [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)], 'k-') 
    plt.ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
    plt.xlim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)) 
    plt.xlabel('Actual y')
    plt.ylabel('Predicted y')
    # テキストボックスを右下に配置
    plt.text(0.98, 0.02, f'R^2: {r2:.3f}\nRMSE: {rmse:.3f}\nMAE: {mae:.3f}', transform=plt.gca().transAxes,
             fontsize=9, verticalalignment='bottom', horizontalalignment='right', 
             bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white', alpha=0.5))
    plt.title(filename[:-4])
    plt.grid(False)
    plt.axis('equal')
    plt.show()

    # 時系列のプロット
    plt.figure(figsize=(10, 5))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))  # 日時形式を設定
    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=5))  # 5時間ごとに目盛りを設定
    plt.plot(y.index, y, label='Actual y', color='blue', marker='', linestyle='-')
    plt.plot(y.index, y_pred, label='Predicted y', color='red', marker='', linestyle='-')
    #plt.gcf().autofmt_xdate()  # x軸のラベルを自動で斜めにして読みやすくする
    plt.ylabel('y')
    plt.title(filename[:-4])
    plt.legend()
    plt.show()
    
    # データの保存
    if save_to_csv:
        results_df = pd.DataFrame({
            'Actual': y,
            'Predicted': y_pred.flatten(),
            'Error': y - y_pred.flatten()
        })
        results_df.to_csv(filename, index=False)
        print(f"Data saved to {filename}")
    
    return r2, rmse, mae

def evaluate_model(X_train, y_train, X_test, y_test, model, save_csv=False, train_filename='train_performance.csv', test_filename='test_performance.csv'):
    """
    トレーニングデータとテストデータの両方でモデルの性能を評価します。
    
    Parameters:
    X_train : pd.DataFrame
        トレーニングデータの説明変数
    y_train : pd.Series
        トレーニングデータの目的変数
    X_test : pd.DataFrame
        テストデータの説明変数
    y_test : pd.Series
        テストデータの目的変数
    model : sklearn model
        評価する訓練済みのモデル
    save_csv : bool, optional
        結果をCSVファイルに保存するかどうかを指定 (default is False)
    train_filename : str, optional
        トレーニングデータの予測結果を保存するCSVファイルの名前 (default is 'train_performance.csv')
    test_filename : str, optional
        テストデータの予測結果を保存するCSVファイルの名前 (default is 'test_performance.csv')
    """
    print("Evaluating Training Data")
    evaluate_performance(X_train, y_train, model, save_to_csv=save_csv, filename=train_filename)
    print("Evaluating Test Data")
    evaluate_performance(X_test, y_test, model, save_to_csv=save_csv, filename=test_filename)

def perform_dt_regression(X, y, fold_number=5, test_size=0.4, min_samples_leaf=3, shuffle=False):
    """
    決定木（DT）モデルを構築し、クロスバリデーションを用いて木の深さを最適化します。
    その後、最適な木の深さでモデルを再構築し、トレーニングデータとテストデータの性能を評価します。

    Parameters:
    X : pd.DataFrame
        説明変数のデータセット
    y : pd.Series
        目的変数のデータ
    fold_number : int, optional
        クロスバリデーションで使用するフォールドの数 (default is 5)。
    test_size : float, optional
        データを分割する際のテストデータの割合 (default is 0.4)。
    min_samplts_leaf : int, optional
        葉ノードごとのサンプル数の最小値 (default is 3)。
    shuffle : bool, optional
        データ分割時にデータをシャッフルするかどうか (時系列データを想定し、デフォルトはFalse)。

    Returns:
    optimal_max_depth : 最適な木の深さ
    dt_final : 最終的な決定木モデル

    """
    # データの分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=shuffle, random_state=0)

    # 木の深さを決定するためのクロスバリデーション
    cross_validation = KFold(n_splits=fold_number, shuffle=False)
    max_depths = np.arange(1, 31)
    r2_scores = []
    for max_depth in max_depths:
        dt = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=0)
        y_cv = cross_val_predict(dt, X_train, y_train, cv=cross_validation)
        r2 = metrics.r2_score(y_train, y_cv)
        r2_scores.append((max_depth, r2))

    # r2_scoresをグラフで表示
    number_of_max_depth = [item[0] for item in r2_scores]
    r2_values = [item[1] for item in r2_scores]
    plt.figure(figsize=(8, 5))
    plt.plot(number_of_max_depth , r2_values, marker='o', linestyle='-', color='b')
    plt.xlabel('Number of max_depth')
    plt.ylabel('R2 Score (CV)')
    plt.grid(True)
    plt.show()

    # 最適な木の深さの決定
    optimal_max_depth = max(r2_scores, key=lambda item: item[1])[0]
    print(f"Optimal number of max_depth: {optimal_max_depth}")

    # 最適な木の深さで決定木モデルを作成
    dt_final = DecisionTreeRegressor(max_depth=optimal_max_depth, min_samples_leaf=min_samples_leaf, random_state=0)
    dt_final.fit(X_train, y_train) 

    evaluate_model(X_train, y_train, X_test, y_test, model=dt_final)

    return optimal_max_depth, dt_final

def perform_rf_regression(X, y, number_of_trees=300, test_size=0.4, shuffle=False):
    """
    ランダムフォレスト（RF）モデルを構築し、OOBを用いて説明変数の数の割合を最適化します。
    その後、最適な説明変数の数の割合でモデルを再構築し、トレーニングデータとテストデータの性能を評価します。

    Parameters:
    X : pd.DataFrame
        説明変数のデータセット
    y : pd.Series
        目的変数のデータ
    number_of_trees : int, optional
        ランダムフォレストのサブデータセット(=決定木モデル)の数 (default is 300)。
    test_size : float, optional
        データを分割する際のテストデータの割合 (default is 0.4)。
    min_samplts_leaf : int, optional
        葉ノードごとのサンプル数の最小値
    shuffle : bool, optional
        データ分割時にデータをシャッフルするかどうか (時系列データを想定し、デフォルトはFalse)。

    Returns:
    optimal_x_variables_rate : 最適なXの数の割合
    rf_final : 最終的な決定木モデル

    """
    # データの分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=shuffle, random_state=0)

    # OOBを用いた説明変数の割合の最適化
    x_variables_rates = np.arange(1, 11, dtype=float) / 10  # 決定木における X の数の割合
    r2_oob = []
    for x_variables_rate in x_variables_rates:
        # ランダムフォレストモデルの作成
        rf = RandomForestRegressor(n_estimators=number_of_trees,
                                    max_features=int(math.ceil(X_train.shape[1] * x_variables_rate)),
                                    oob_score=True)
        # モデルの訓練
        rf.fit(X_train, y_train)
        r2 = metrics.r2_score(y_train, rf.oob_prediction_)
        r2_oob.append((x_variables_rate, r2))

    # r2_scoresをグラフで表示
    components = [item[0] for item in r2_oob]
    r2_values = [item[1] for item in r2_oob]
    plt.figure(figsize=(8, 5))
    plt.plot(components, r2_values, marker='o', linestyle='-', color='b')
    plt.xlabel('rate of X-variables')
    plt.ylabel('R2 Score (OOB)')
    plt.grid(True)
    plt.show()

    # 最適なXの数の割合の決定
    optimal_x_variables_rate = max(r2_oob, key=lambda item: item[1])[0]
    print(f"Optimal rate of X-variables: {optimal_x_variables_rate}")

    # 最適なXの数の割合でRFモデルを作成
    rf_final = RandomForestRegressor(n_estimators=number_of_trees, max_depth=3,
                              max_features=int(math.ceil(X_train.shape[1] * optimal_x_variables_rate)),
                              oob_score=True) # RF モデルの宣言
    rf_final.fit(X_train, y_train) 

    evaluate_model(X_train, y_train, X_test, y_test, model=rf_final)

    return optimal_x_variables_rate, rf_final

def evaluate_performance_gpr(X, y, y_train, model, save_to_csv=False, filename='performance.csv'):
    """
    モデルの性能を評価し、結果をプロットします。
    オプションで予測結果をCSVファイルに保存します。
    
    Parameters:
    X : pd.DataFrame
        説明変数のデータセット
    y : pd.Series
        目的変数の値
    y_train : pd.Series
        トレーニングデータの目的変数（標準化解除に使用）
    model : sklearn model
        評価する訓練済みのモデル
    save_to_csv : bool, optional
        結果をCSVファイルに保存するかどうかを指定 (default is False)
    filename : str, optional
        出力するCSVファイルの名前 (default is 'performance.csv')

    Returns:
    r2 : 決定係数R^2
    rmse : RMSE
    mae : MAE
        
    """
    # ｙの推定値を算出
    estimated_y_train, estimated_y_train_std = model.predict(X, return_std=True)
    y_pred = estimated_y_train * y_train.std() + y_train.mean() 
    y_pred_std = estimated_y_train_std* y_train.std()
    
    # 性能指標の算出
    r2 = metrics.r2_score(y, y_pred)
    rmse = np.sqrt(metrics.mean_squared_error(y, y_pred))
    mae = metrics.mean_absolute_error(y, y_pred)
    print(f"R^2: {r2:.3f}, RMSE: {rmse:.3f}, MAE: {mae:.3f}")
    
    # プロットの作成
    plt.figure(figsize=(6, 6))
    plt.scatter(y, y_pred, color='blue')
    y_max = max(y.max(), y_pred.max())
    y_min = min(y.min(), y_pred.min()) 
    
     # 取得した最小値-5%から最大値+5%まで、対角線を作成
    plt.plot([y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)],
             [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)], 'k-') 
    plt.ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
    plt.xlim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)) 
    plt.xlabel('Actual y')
    plt.ylabel('Predicted y')
    # テキストボックスを右下に配置
    plt.text(0.98, 0.02, f'R^2: {r2:.3f}\nRMSE: {rmse:.3f}\nMAE: {mae:.3f}', transform=plt.gca().transAxes,
             fontsize=9, verticalalignment='bottom', horizontalalignment='right', 
             bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white', alpha=0.5))
    plt.title(filename[:-4])
    plt.grid(False)
    plt.axis('equal')
    plt.show()

    # 時系列のプロット
    plt.figure(figsize=(10, 5))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))  # 日時形式を設定
    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=5))  # 4時間ごとに目盛りを設定
    plt.plot(y.index, y, label='Actual y', color='blue', marker='', linestyle='-')
    plt.plot(y.index, y_pred, label='Predicted y', color='red', marker='', linestyle='-')
    plt.fill_between(y.index, y_pred - 1.96*y_pred_std, y_pred + 1.96*y_pred_std, alpha=0.2, color='b', label='95% confidence interval')
    #plt.gcf().autofmt_xdate()  # x軸のラベルを自動で斜めにして読みやすくする
    plt.ylabel('y')
    plt.title(filename[:-4])
    plt.legend()
    plt.show()
    
    # データの保存
    if save_to_csv:
        results_df = pd.DataFrame({
            'Actual': y,
            'Predicted': y_pred.flatten(),
            'Error': y - y_pred.flatten()
        })
        results_df.to_csv(filename, index=False)
        print(f"Data saved to {filename}")
    
    return r2, rmse, mae

def evaluate_model_gpr(X_train_scaled, y_train, X_test_scaled, y_test, model, save_csv=False, train_filename='train_performance.csv', test_filename='test_performance.csv'):
    """
    トレーニングデータとテストデータの両方でモデルの性能を評価します。
    
    Parameters:
    X_train_scaled : pd.DataFrame
        標準化されたトレーニングデータの説明変数
    y_train : pd.Series
        トレーニングデータの目的変数
    X_test_scaled : pd.DataFrame
        標準化されたテストデータの説明変数
    y_test : pd.Series
        テストデータの目的変数
    model : sklearn model
        評価する訓練済みのモデル
    save_csv : bool, optional
        結果をCSVファイルに保存するかどうかを指定 (default is False)
    train_filename : str, optional
        トレーニングデータの予測結果を保存するCSVファイルの名前 (default is 'train_performance.csv')
    test_filename : str, optional
        テストデータの予測結果を保存するCSVファイルの名前 (default is 'test_performance.csv')
    """
    print("Evaluating Training Data")
    evaluate_performance_gpr(X_train_scaled, y_train, y_train, model, save_to_csv=save_csv, filename=train_filename)
    print("Evaluating Test Data")
    evaluate_performance_gpr(X_test_scaled, y_test, y_train, model, save_to_csv=save_csv, filename=test_filename)

