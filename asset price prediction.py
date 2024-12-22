import yfinance as yf
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np  

# Ambil data AAPL dari Yahoo Finance
crypto = yf.Ticker("AAPL")
data = crypto.history(period='1y')

# Menghapus nilai NaN dan memperbaiki data
data = data.dropna()
data = data.ffill()
data.index = pd.to_datetime(data.index)

# Hitung SMA-50 dan SMA-200
data['SMA_50'] = data['Close'].rolling(window=50).mean()
data['SMA_200'] = data['Close'].rolling(window=200).mean()

# Menghapus baris yang memiliki NaN pada SMA-50 atau SMA-200
data = data.dropna(subset=['SMA_50', 'SMA_200'])

# Normalisasi data
scaler = MinMaxScaler()
data[['Close', 'SMA_50', 'SMA_200']] = scaler.fit_transform(data[['Close', 'SMA_50', 'SMA_200']])

# Mengambil data yang tidak terlalu ekstrim (menghapus nilai lebih dari quantile 0.95)
data = data[data['Close'] < data['Close'].quantile(0.95)]

# Membatasi data antara tahun 2023 hingga 2024
data = data.loc['2023-12-18':'2024-12-18']

# Menyiapkan fitur (X) dan target (y)
x = data[['SMA_50', 'SMA_200']].values

# Target variabel adalah harga penutupan BTC besok
data['Close_Tomorrow'] = data['Close'].shift(-1)
y = data['Close_Tomorrow'].values

# Pastikan tidak ada NaN dalam y
x = x[~np.isnan(y)]  
y = y[~np.isnan(y)]

# Membagi data menjadi data training dan testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=78)

# Mengonversi data ke tensor PyTorch
x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Ubah input data untuk LSTM (dimensi [batch_size, time_step, input_size])
x_train_tensor = x_train_tensor.unsqueeze(1)
x_test_tensor = x_test_tensor.unsqueeze(1)

# Definisikan model LSTM
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, (hn, cn) = self.lstm(x)
        out = self.fc(hn[-1])
        return out  

# Inisialisasi model
input_size = 2
hidden_size = 120
output_size = 1

model = LSTMModel(input_size, hidden_size, output_size)

# Loss function dan optimizer
criterion = nn.MSELoss()  
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training model
for epoch in range(10000):
    model.train()
    y_pred = model(x_train_tensor)
    loss = criterion(y_pred.squeeze(), y_train_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/10000], Loss: {loss.item():.4f}")

# Ambil data terbaru untuk prediksi
latest_data = data[['SMA_50', 'SMA_200']].iloc[-1].values
latest_data_tensor = torch.tensor(latest_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

# Evaluasi model dan prediksi harga penutupan besok
model.eval()
with torch.no_grad():
    y_test_pred = model(x_test_tensor)

y_test_pred = y_test_pred.squeeze().numpy()

# Prediksi harga besok
model.eval()
with torch.no_grad():
    next_day_pred = model(latest_data_tensor).item()

# Transformasi harga kembali ke skala asli
next_day_pred = scaler.inverse_transform([[next_day_pred, latest_data[0], latest_data[1]]])[0][0]

# Menampilkan hasil prediksi harga besok
print(f"Prediksi harga AAPL untuk besok: ${next_day_pred:.2f}")

# Visualisasi Candlestick dengan Plotly
fig = go.Figure(data=[go.Candlestick(x=data.index,
                                     open=data['Open'],
                                     high=data['High'],
                                     low=data['Low'],
                                     close=data['Close'],
                                     name="Candlestick"),
                      go.Scatter(x=data.index, 
                                 y=data['SMA_50'], 
                                 mode='lines', 
                                 name='SMA-50',
                                 line=dict(color='blue', width=2)),
                      go.Scatter(x=data.index, 
                                 y=data['SMA_200'], 
                                 mode='lines', 
                                 name='SMA-200',
                                 line=dict(color='red', width=2))
                    ])

# Update layout dan tampilkan grafik
fig.update_layout(
    title="AAPL Candlestick Chart with SMA-50 and SMA-200",
    xaxis_title="Data",
    yaxis_title="Price (USD)",
    xaxis_rangeslider_visible=False,  
    template="plotly_dark"  
)

fig.show()
