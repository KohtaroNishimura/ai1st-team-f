
from flask import Flask, request, jsonify
from flask_cors import CORS  # 追加
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os

app = Flask(__name__)
CORS(app)  # CORSを有効にする

# ルートパスの定義
@app.route("/")
def home():
    return "Hello Flask!"

# データファイルのパス
data_file_path = 'jisseki6dayskakoucolor.csv'

# データを読み込む
data = pd.read_csv(data_file_path)

# 上限条件を設定
machine_limits = {'SMP-1': {'Type of printing': 'convex intermittent',
  'number of colors': 5.0,
  'laminate': 1,
  'paper width': 280,
  'sending': 200,
  'glue killer': 1},
 'SMP-2': {'Type of printing': 'convex intermittent',
  'number of colors': 6.0,
  'laminate': 1,
  'paper width': 280,
  'sending': 200,
  'glue killer': 0},
 'SMP-3': {'Type of printing': 'convex intermittent',
  'number of colors': 4.0,
  'laminate': 1,
  'paper width': 280,
  'sending': 200,
  'glue killer': 1},
 'SMP-4auto': {'Type of printing': 'convex intermittent',
  'number of colors': 5.0,
  'laminate': 1,
  'paper width': 225,
  'sending': 180,
  'glue killer': 0},
 'CS-1': {'Type of printing': 'convex intermittent',
  'number of colors': 3.0,
  'laminate': 1,
  'paper width': 180,
  'sending': 120,
  'glue killer': 0},
 'JNAS-2': {'Type of printing': 'convex intermittent',
  'number of colors': 4.0,
  'laminate': 0,
  'paper width': 240,
  'sending': 200,
  'glue killer': 0},
 'SL-4': {'Type of printing': 'convex intermittent',
  'number of colors': 4.0,
  'laminate': 0,
  'paper width': 210,
  'sending': 170,
  'glue killer': 0},
 'SL-5': {'Type of printing': 'convex intermittent',
  'number of colors': 3.0,
  'laminate': 0,
  'paper width': 210,
  'sending': 170,
  'glue killer': 0},
 'SL-6': {'Type of printing': 'convex intermittent',
  'number of colors': 2.0,
  'laminate': 0,
  'paper width': 210,
  'sending': 170,
  'glue killer': 0},
 'FX-3': {'Type of printing': 'intermittent off',
  'number of colors': 5.0,
  'laminate': 0,
  'paper width': 270,
  'sending': 260,
  'glue killer': 0}}

def filter_machines(machine_limits, job_data):
    filtered_machines = []
    for machine, limits in machine_limits.items():
        match = all(job_data[key] <= limits[key] for key in job_data if key in limits)
        if match:
            filtered_machines.append(machine)
    return filtered_machines

@app.route('/predict', methods=['POST'])
def predict():
    # ジョブデータを受け取る
    job_data = request.json
    
    # 上限条件を満たす機種を抽出
    valid_machines = filter_machines(machine_limits, job_data)
    filtered_data = data[data['Machine'].isin(valid_machines)]
    
    # 特徴量とラベルを設定
    X = filtered_data[['number of colors', 'Type of printing', 'laminate', 'glue killer', 'paper width', 'sending' , 'color']]
    y = filtered_data['Machine']
    
    # カテゴリ変数を数値に変換
    X = pd.get_dummies(X)
    
    # 訓練データとテストデータに分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # ランダムフォレストモデルを構築
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # ジョブデータをデータフレーム化
    job_df = pd.DataFrame([job_data])
    job_df = pd.get_dummies(job_df)
    job_df = job_df.reindex(columns=X.columns, fill_value=0)
    
    # 確率分布を計算
    probabilities = model.predict_proba(job_df)[0]
    
    # 機種ごとの確率分布を整理
    machine_probs = pd.DataFrame({'Machine': model.classes_, 'Probability': probabilities})
    machine_probs = machine_probs.sort_values(by='Probability', ascending=False)
    
    # 結果を返す
    return jsonify(machine_probs.to_dict(orient='records'))

# CSVファイルからジョブデータを読み込み、複数行の予測結果を返すエンドポイント
@app.route('/predict_csv', methods=['POST'])
def predict_csv():
    # CSVファイルを受け取る
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    
    # CSVファイルを読み込む
    job_data_df = pd.read_csv(file)
    
    # 上限条件を満たす機種を抽出し、予測結果を保存するためのリストを用意
    results = []
    
    for _, job_data in job_data_df.iterrows():
        valid_machines = filter_machines(machine_limits, job_data)
        filtered_data = data[data['Machine'].isin(valid_machines)]
        
        # 特徴量とラベルを設定
        X = filtered_data[['number of colors', 'Type of printing', 'laminate', 'glue killer', 'paper width', 'sending' , 'color']]
        y = filtered_data['Machine']
        
        # カテゴリ変数を数値に変換
        X = pd.get_dummies(X)
        
        # 訓練データとテストデータに分割
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # ランダムフォレストモデルを構築
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        
        # ジョブデータをデータフレーム化
        job_df = pd.DataFrame([job_data])
        job_df = pd.get_dummies(job_df)
        job_df = job_df.reindex(columns=X.columns, fill_value=0)
        
        # 確率分布を計算
        probabilities = model.predict_proba(job_df)[0]
        
        # 機種ごとの確率分布を整理
        machine_probs = pd.DataFrame({'Machine': model.classes_, 'Probability': probabilities})
        machine_probs = machine_probs.sort_values(by='Probability', ascending=False)
        
        # 結果をリストに追加
        results.append(machine_probs.to_dict(orient='records'))
    
    # 結果をJSONで返す
    return jsonify(results)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # 環境変数 PORT からポート番号を取得
    app.run(host="0.0.0.0", port=port)  # 0.0.0.0 にバインド
