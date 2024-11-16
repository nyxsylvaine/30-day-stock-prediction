import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os
from pathlib import Path
from prophet import Prophet
import plotly.graph_objs as go

# Popüler hisse senetleri listesi
TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'BRK-B', 'NVDA',
    'JPM', 'JNJ', 'V', 'PG', 'UNH', 'HD', 'DIS', 'VZ', 'NFLX',
    'PYPL', 'INTC', 'CMCSA'
]

# Bugünün tarihi ve 4 yıl önceki tarih
END_DATE = datetime.now()
START_DATE = END_DATE - timedelta(days=4 * 365)

# Çalışma dizini
BASE_DIRECTORY = Path.cwd()
SAVE_DIRECTORY = BASE_DIRECTORY / "Veriler"
GRAPH_DIRECTORY = BASE_DIRECTORY / "Grafikler"

# Verileri indir
def fetch_data(tickers, start_date, end_date):
    print("Veriler indiriliyor...")
    data = yf.download(tickers, start=start_date, end=end_date, group_by="ticker", progress=False)
    all_data = []

    for ticker in tickers:
        try:
            ticker_data = data[ticker]
            ticker_data['Ticker'] = ticker
            ticker_data.reset_index(inplace=True)
            ticker_data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Ticker']
            all_data.append(ticker_data)
        except KeyError:
            print(f"{ticker} için veri bulunamadı.")

    if not all_data:
        raise ValueError("Hiçbir veri çekilemedi.")
    return pd.concat(all_data, ignore_index=True)

# Verileri ön işleme
def preprocess_data(data):
    print("Veriler işleniyor...")
    data = data[['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']]
    data = data.infer_objects()
    data.interpolate(method='linear', inplace=True)  # Doğrusal doldurma
    data.ffill(inplace=True)  # İleri doldurma
    data.bfill(inplace=True)  # Geri doldurma
    return data

# Prophet modeli ile tahmin
def forecast_data(data, tickers):
    print("Tahmin işlemleri başlatılıyor...")
    predictions = []

    for ticker in tickers:
        ticker_data = data[data['Ticker'] == ticker][['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
        ticker_data.dropna(inplace=True)
        ticker_data['ds'] = ticker_data['ds'].dt.tz_localize(None)

        # Prophet modelini oluştur ve eğit
        model = Prophet(
            daily_seasonality=True,
            yearly_seasonality=True,
            weekly_seasonality=True,
            seasonality_prior_scale=10.0,
            changepoint_prior_scale=0.5
        )
        model.fit(ticker_data)

        # Gelecek 30 gün için tahmin yap
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)

        # Tahmin edilen ve gerçek fiyatları birleştir
        forecast['Real Price'] = ticker_data['y'].tolist() + [None] * 30
        forecast['Ticker'] = ticker
        predictions.append(forecast[['ds', 'Ticker', 'yhat', 'Real Price']])

    return pd.concat(predictions, ignore_index=True)

# Grafik oluştur ve kaydet
def save_graphs(predictions, tickers, graph_directory):
    print("Grafikler oluşturuluyor...")
    for ticker in tickers:
        ticker_data = predictions[predictions['Ticker'] == ticker]

        # Grafik oluştur
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ticker_data['ds'], y=ticker_data['Real Price'], mode='lines', name='Gerçek Fiyat'))
        fig.add_trace(go.Scatter(x=ticker_data['ds'], y=ticker_data['yhat'], mode='lines', name='Tahmin Edilen Fiyat', line=dict(dash='dot')))
        fig.update_layout(
            title=f"{ticker} Fiyat Tahmini",
            xaxis_title="Tarih",
            yaxis_title="Fiyat",
            xaxis_rangeslider_visible=True
        )

        # Grafiği HTML olarak kaydet
        graph_path = graph_directory / f"{ticker}_fiyat_tahmini.html"
        fig.write_html(graph_path)
    print("Grafikler kaydedildi.")

# Ana işlem akışı
def main():
    try:
        # Verileri indir ve işleme al
        all_data = fetch_data(TICKERS, START_DATE, END_DATE)
        all_data = preprocess_data(all_data)

        # Tarih etiketli klasörler oluştur
        current_date = datetime.now().strftime("%Y%m%d_%H%M")
        data_directory = SAVE_DIRECTORY / f"{current_date}_veriler"
        graph_directory = GRAPH_DIRECTORY / f"{current_date}_grafikler"
        os.makedirs(data_directory, exist_ok=True)
        os.makedirs(graph_directory, exist_ok=True)

        # Verileri kaydet
        file_path = data_directory / f"{current_date}_hisse_verileri.csv"
        all_data.to_csv(file_path, index=False, sep=';')
        print(f"Veriler kaydedildi: {file_path}")

        # Tahmin yap
        predictions = forecast_data(all_data, TICKERS)

        # Grafik oluştur ve kaydet
        save_graphs(predictions, TICKERS, graph_directory)

        # Verilerin önizlemesini göster
        print("Tablo:")
        print(all_data.head())

    except Exception as e:
        print(f"Hata oluştu: {e}")

if __name__ == "__main__":
    main()
