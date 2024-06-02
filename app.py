import boto3
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dash import Dash, dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from io import BytesIO
import base64
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from threading import Thread
import plotly.express as px
import plotly.graph_objs as go
from scipy.sparse import csr_matrix
import warnings

warnings.filterwarnings("ignore")

# 初始化S3客户端
s3 = boto3.client('s3')
bucket_name = 'xwynews' # 设置为我的S3桶名

def read_file_from_s3(file_key, file_type):
    response = s3.get_object(Bucket=bucket_name, Key=file_key)
    file_content = response['Body'].read()
    if file_type == 'csv':
        return pd.read_csv(BytesIO(file_content), on_bad_lines='skip')
    elif file_type == 'parquet':
        return pd.read_parquet(BytesIO(file_content))
    else:
        raise ValueError("Unsupported file type")

# 更新文件路径，现在使用S3的键值
file_names = [
    "data/interventions_bxl.parquet.gzip",
    "data/interventions_bxl2.parquet.gzip",
    "data/interventions1.parquet.gzip",
    "data/interventions2.parquet.gzip",
    "data/interventions3.parquet.gzip"
]
dfs = [read_file_from_s3(file_name, 'parquet') for file_name in file_names]

# CSV文件读取
file_list = [
    'data/cad9.parquet.csv',
    'data/interventions1.parquet.csv',
    'data/interventions2.parquet.csv',
    'data/interventions3.parquet.csv',
    'data/interventions_bxl.parquet.csv'
]
columns_list = [
    ['T0', 'EventType32Trip', 'EventLevel32Trip'],
    ['T0', 'EventType32Firstcall', 'EventLevel32Trip'],
    ['T0', 'EventType32Firstcall', 'EventLevel32Trip'],
    ['T0', 'EventType32Firstcall', 'EventLevel32Trip'],
    ['T0', 'Eventtype_firstcall', 'Eventlevel_trip'],
]

df_loc = pd.DataFrame()
for i in range(len(file_list)):
    df = read_file_from_s3(file_list[i], 'csv')
    df_tmp = df.loc[:,columns_list[i]]
    df_tmp = df_tmp.dropna(axis=0)
    df_tmp["T0"] = df_tmp["T0"].map(lambda x: str(x))
    df_tmp.columns = ['T0', 'Firstcall', 'EventLevel']
    if len(df_tmp['T0'][1]) < 23:
        df_tmp['T0'] = pd.to_datetime(df_tmp['T0'], format='%d%b%y:%H:%M:%S')
        df_tmp['T0'] = df_tmp['T0'].dt.strftime('%Y-%m-%d %H:%M:%S.000')
    df_loc = pd.concat([df_loc, df_tmp], axis=0, sort=False)

# 处理第二个DataFrame的 'EventType and EventLevel' 列
df2 = dfs[1]
df2['event type trip'] = df2['EventType and EventLevel'].apply(lambda x: x.split()[0] if isinstance(x, str) else None)
df2['event level trip'] = df2['EventType and EventLevel'].apply(lambda x: x.split()[1] if isinstance(x, str) else None)

# 转换 'T0' 和 'T3' 列为日期时间格式并计算等待时间
date_format = "%d%b%y:%H:%M:%S"
df2['T0'] = pd.to_datetime(df2['T0'], format=date_format, errors='coerce')
df2['T3'] = pd.to_datetime(df2['T3'], format=date_format, errors='coerce')
df2['waiting time'] = (df2['T3'] - df2['T0']).dt.total_seconds() // 60

# 标准化列名
column_mapping = [
    {'abandon_reason': 'abandon reason', 'waiting_time': 'waiting time', 'Eventlevel_trip': 'event level trip', 'vector_type': 'vector type', 'eventtype_trip': 'event type trip'},
    {'Abandon reason NL': 'abandon reason', 'Vector type NL': 'vector type'},
    {'Abandon reason': 'abandon reason', 'Waiting time': 'waiting time', 'EventType Trip': 'event type trip', 'EventLevel Trip': 'event level trip', 'Vector type': 'vector type'},
    {'Abandon reason': 'abandon reason', 'Waiting time': 'waiting time', 'EventType Trip': 'event type trip', 'EventLevel Trip': 'event level trip', 'Vector type': 'vector type'},
    {'Abandon reason': 'abandon reason', 'Waiting time': 'waiting time', 'EventType Trip': 'event type trip', 'EventLevel Trip': 'event level trip', 'Vector type': 'vector type'}
]

for i, df in enumerate(dfs):
    df.rename(columns=column_mapping[i], inplace=True)

# 处理 'abandon reason' 列: 删除空值并编码
for i, df in enumerate(dfs):
    df.dropna(subset=['abandon reason'], inplace=True)
    if i == 1:
        df['abandon reason'] = df['abandon reason'].apply(lambda x: 1 if x == 'Dood Ter Plaatse' else 0)
    else:
        df['abandon reason'] = df['abandon reason'].apply(lambda x: 1 if x == 'Overleden' else 0)

# 提取 'event type trip' 列以 'P039' 或 'P003' 开头的行
filtered_dfs = [df[df['event type trip'].str.startswith(('P039', 'P003'), na=False)] for df in dfs]

# 提取四列并合并
selected_columns = ['waiting time', 'event level trip', 'vector type', 'abandon reason']
merged_df = pd.concat([df[selected_columns] for df in filtered_dfs if all(col in df.columns for col in selected_columns)], ignore_index=True)

# 删除 'waiting time' 列中的空值
merged_df.dropna(subset=['waiting time'], inplace=True)

# 特征预处理
categorical_features = ['event level trip', 'vector type']
numeric_features = ['waiting time']
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_features),
        ('num', StandardScaler(), numeric_features)
    ]
)

# 数据平衡处理
X = merged_df.drop(columns=['abandon reason'])
y = merged_df['abandon reason']
X_preprocessed = preprocessor.fit_transform(X)

# 将稀疏矩阵转换为密集数组
X_preprocessed = X_preprocessed.toarray() if isinstance(X_preprocessed, csr_matrix) else X_preprocessed

smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_preprocessed, y)

# 保存处理后的数据
np.save('X_res.npy', X_res, allow_pickle=True)
np.save('y_res.npy', y_res, allow_pickle=True)

df = df_loc
df['Cardiac Call'] = df["Firstcall"].map(lambda x: 1 if 'cardiac arrest' in x.lower() else 0)
df = df[df["Cardiac Call"]==1]
df['T0'] = df['T0'].map(lambda x: x[:23])
df['datetime'] = pd.to_datetime(df['T0'])
df['hour'] = df['datetime'].dt.hour

cnt_hour_level = df.groupby(['hour', 'EventLevel']).size().reset_index(name='count')

# 心脏骤停数量趋势图的数据
df_calc = df.loc[:, ["datetime", "Cardiac Call"]]
df_calc.set_index('datetime', inplace=True)
df_resampled = df_calc.resample('48H').sum()
df_resampled = df_resampled.rolling(window=10).mean()
df_resampled.dropna(inplace=True)

# 事件等级按小时统计的图表
def plot_event_level_per_hour():
    try:
        pivot_table = cnt_hour_level.pivot(index='hour', columns='EventLevel', values='count').fillna(0)
        fig = go.Figure()
        for column in pivot_table.columns:
            fig.add_trace(go.Scatter(x=pivot_table.index, y=pivot_table[column], mode='lines+markers', name=column))
        fig.update_layout(title='Count of EventLevel per Hour', xaxis_title='Hour', yaxis_title='Count', legend_title='Type',height=550)
        return fig
    except Exception as e:
        print(f"Error in plot_event_level_per_hour: {e}")
        return None

# 心脏骤停数量趋势图
def plot_cardiac_arrest_trend():
    try:
        fig = px.line(df_resampled, y='Cardiac Call', title='Cardiac Arrest Sum Every 72 Hours')
        fig.update_layout(xaxis_title='Time', yaxis_title='Sum of Cardiac Arrest',height=550)
        return fig
    except Exception as e:
        print(f"Error in plot_cardiac_arrest_trend: {e}")
        return None

# LSTM模型预测图
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, num_layers=1):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# 在启动时缓存结果
lstm_forecast_img = None

def create_lstm_forecast_plot():
    try:
        global lstm_forecast_img
        if lstm_forecast_img is not None:
            return lstm_forecast_img

        data = df_resampled.values
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)

        def create_dataset(data, time_step=1):
            X, y = [], []
            for i in range(len(data)-time_step-1):
                X.append(data[i:(i+time_step), 0])
                y.append(data[i + time_step, 0])
            return np.array(X), np.array(y)

        time_step = 20
        X, y = create_dataset(scaled_data, time_step)
        X = X.reshape(X.shape[0], X.shape[1], 1)

        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        train_size = int(len(X) * 0.9)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        batch_size = 128
        train_data = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

        model = LSTMModel(input_dim=1, hidden_dim=50, output_dim=1, num_layers=2)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        num_epochs = 300  # 减少Epoch次数
        for epoch in range(num_epochs):
            model.train()
            for batch_X, batch_y in train_loader:
                outputs = model(batch_X)
                optimizer.zero_grad()
                loss = criterion(outputs, batch_y.unsqueeze(1))
                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

        model.eval()
        with torch.no_grad():
            forecast = model(X_test)

        forecast = scaler.inverse_transform(forecast.numpy())
        y_test = scaler.inverse_transform(y_test.numpy().reshape(-1, 1))

        time = df_resampled.index[21:]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=time[:len(y_train)], y=scaler.inverse_transform(y_train.numpy().reshape(-1, 1)).flatten(), mode='lines', name='Train'))
        fig.add_trace(go.Scatter(x=time[len(y_train):], y=y_test.flatten(), mode='lines', name='Test'))
        fig.add_trace(go.Scatter(x=time[len(y_train):], y=forecast.flatten(), mode='lines', name='Forecast'))
        fig.update_layout(title='LSTM Model Forecast', xaxis_title='Time', yaxis_title='Value',height=550)
        lstm_forecast_img = fig
        return fig
    except Exception as e:
        print(f"Error in create_lstm_forecast_plot: {e}")
        return None
    
# 图表生成函数
def plot_roc_curve():
    try:
        X_res = np.load('X_res.npy', allow_pickle=True)
        y_res = np.load('y_res.npy', allow_pickle=True)
        
        X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=42)

        # Random Forest
        model_rf = RandomForestClassifier(random_state=42)
        model_rf.fit(X_train, y_train)
        y_test_proba_rf = model_rf.predict_proba(X_test)[:, 1]
        fpr_rf, tpr_rf, _ = roc_curve(y_test, y_test_proba_rf)
        roc_auc_rf = auc(fpr_rf, tpr_rf)

        # XGBoost
        model_xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        model_xgb.fit(X_train, y_train)
        y_test_proba_xgb = model_xgb.predict_proba(X_test)[:, 1]
        fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_test_proba_xgb)
        roc_auc_xgb = auc(fpr_xgb, tpr_xgb)

        # CatBoost
        model_cb = CatBoostClassifier(silent=True)
        model_cb.fit(X_train, y_train)
        y_test_proba_cb = model_cb.predict_proba(X_test)[:, 1]
        fpr_cb, tpr_cb, _ = roc_curve(y_test, y_test_proba_cb)
        roc_auc_cb = auc(fpr_cb, tpr_cb)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr_rf, y=tpr_rf, mode='lines', name=f'Random Forest (AUC = {roc_auc_rf:.2f})'))
        fig.add_trace(go.Scatter(x=fpr_xgb, y=tpr_xgb, mode='lines', name=f'XGBoost (AUC = {roc_auc_xgb:.2f})'))
        fig.add_trace(go.Scatter(x=fpr_cb, y=tpr_cb, mode='lines', name=f'CatBoost (AUC = {roc_auc_cb:.2f})'))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), name='Random Guess'))
        fig.update_layout(title='Receiver Operating Characteristic (ROC) Curve', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate',height=550)
        return fig
    except Exception as e:
        print(f"Error in plot_roc_curve: {e}")
        return None

def plot_evaluation_metrics():
    try:
        X_res = np.load('X_res.npy', allow_pickle=True)
        y_res = np.load('y_res.npy', allow_pickle=True)
        
        X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=42)

        # 模型训练和评估
        models = {
            'Random Forest': RandomForestClassifier(random_state=42),
            'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
            'CatBoost': CatBoostClassifier(silent=True)
        }

        metrics = {'Model': [], 'Accuracy': [], 'Precision': [], 'Recall': [], 'F1 Score': [], 'ROC-AUC': []}

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_test_pred = model.predict(X_test)
            y_test_proba = model.predict_proba(X_test)[:, 1]

            metrics['Model'].append(name)
            metrics['Accuracy'].append(accuracy_score(y_test, y_test_pred))
            metrics['Precision'].append(precision_score(y_test, y_test_pred))
            metrics['Recall'].append(recall_score(y_test, y_test_pred))
            metrics['F1 Score'].append(f1_score(y_test, y_test_pred))
            metrics['ROC-AUC'].append(roc_auc_score(y_test, y_test_proba))

        metrics_df = pd.DataFrame(metrics)

        fig = px.bar(metrics_df, x='Model', y=['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC'], barmode='group')
        fig.update_layout(title='Comparison of Evaluation Metrics', xaxis_title='Model', yaxis_title='Score', legend_title='Metrics',height = 550)
        return fig
    except Exception as e:
        print(f"Error in plot_evaluation_metrics: {e}")
        return None

# 创建Dash应用
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H1("Data Analysis Results"), className="mb-4")),
    dbc.Row([
        dbc.Col(dcc.Dropdown(
            id='chart-dropdown',
            options=[
                {'label': 'Event Level per Hour', 'value': 'event_level_per_hour'},
                {'label': 'Cardiac Arrest Trend', 'value': 'cardiac_arrest_trend'},
                {'label': 'LSTM Model Forecast', 'value': 'lstm_forecast'},
                {'label': 'ROC Curve', 'value': 'roc_curve'},
                {'label': 'Evaluation Metrics', 'value': 'evaluation_metrics'}
            ],
            value='event_level_per_hour'
        ), width=6)
    ]),
    dbc.Row([
        dbc.Col(html.Div(id='chart-container'), width=12),
        dbc.Col(html.Div(id='description-container'), width=12)
    ])
], fluid=True)

@app.callback(
    [Output('chart-container', 'children'),
     Output('description-container', 'children')],
    [Input('chart-dropdown', 'value')]
)
def update_content(chart_type):
    descriptions = {
        'event_level_per_hour': 'Each line representing a different level of severity, ranging from N0 to N7B. N0, the blue line, represents the most severe level of cardiac arrest events, and severity decreases as the numbers increase. The blue line,notably peaks in the late morning and early afternoon, aligning with our observation that these times experience higher counts of severe incidents. Although the most severe events (N0) generally occur less frequently than less severe events (such as N1, represented by the red line), their fluctuation patterns over time are similar.',
        'cardiac_arrest_trend': 'The visualization reveals that a time series plot, smoothed with a rolling average, identifies seasonal trends in cardiac arrest incidents, particularly noting significant peaks from mid-December to February. This data suggests that the cold winter months pose a higher risk for cardiac arrests, emphasizing the need for increased awareness and preventive measures during this period.',
        'lstm_forecast': 'The models predictions (green line) closely align with the test data (orange line), indicating a high degree of fit. It successfully captures the overall trends and patterns in the data, particularly during notable fluctuations such as the observed increase in cardiac arrest events between April and May 2023. Additionally, the model performs well on the training data (blue line), closely mirroring actual data fluctuations and demonstrating its robust learning capability.',
        'roc_curve': 'From the image above, we can see that all the curves are close to the upper left corner, and all models have relatively high ROC AUC scores and perform well. Although the CatBoost model has a slightly higher AUC of 0.83, the difference is minimal, with both Random Forest and XGBoost scoring 0.82.',
        'evaluation_metrics': 'The five evaluation metrics are quite similar, yet XGBoost slightly outperforms the other models. Additionally, this model is very fast to train, saving considerable time. Therefore, we choose XGBoost as the optimal model for predicting patient mortality due to cardiac events.'
    }
    description_text = descriptions.get(chart_type, "No description available.")
    styled_description = html.Div(description_text, className='description-text')

    # 根据chart_type调用相应的绘图函数
    fig = None
    if chart_type == 'event_level_per_hour':
        fig = plot_event_level_per_hour()
    elif chart_type == 'cardiac_arrest_trend':
        fig = plot_cardiac_arrest_trend()
    elif chart_type == 'lstm_forecast':
        fig = create_lstm_forecast_plot()
    elif chart_type == 'roc_curve':
        fig = plot_roc_curve()
    elif chart_type == 'evaluation_metrics':
        fig = plot_evaluation_metrics()
    
    return dcc.Graph(figure=fig, config={
    'displaylogo': False,
    'modeBarButtonsToRemove': ['lasso2d']
}), styled_description

if __name__ == '__main__':
    app.run_server(debug=False, dev_tools_ui=False, dev_tools_props_check=False)