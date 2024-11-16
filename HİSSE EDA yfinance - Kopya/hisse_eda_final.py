import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os
from pathlib import Path
from prophet import Prophet
import plotly.graph_objs as go

# Bugünün tarihi
end_date = datetime.now()
# 4 yıl önceki tarih
start_date = end_date - timedelta(days=4*365)

# Popüler hisse senetleri listesi
tickers = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'BRK-B', 'NVDA', 
    'JPM', 'JNJ', 'V', 'PG', 'UNH', 'HD', 'DIS', 'VZ', 'NFLX', 
    'PYPL', 'INTC', 'CMCSA'
]

# Çalışma dizinini al
base_directory = Path(__file__).parent

# Veriler ve grafikler için göreceli dizinler oluştur
save_directory = base_directory / "Veriler"
graph_directory = base_directory / "Grafikler"

# Hisse verilerini saklamak için boş bir liste oluştur
all_data = []

# Her bir hisse senedi için verileri çek
for ticker in tickers:
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if data.empty:
        print(f"{ticker} için veri bulunamadı.")
    else:
        data['Ticker'] = ticker  # Ticker bilgisini ekle
        data.reset_index(inplace=True)  # Tarih sütununu indeks yerine sütun olarak ekle
        
        # Sütun adlarını düzelt
        data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Ticker']
        
        all_data.append(data)  # DataFrame'i listeye ekle

# Eğer hiç veri yoksa hata ver
if not all_data:
    print("Hiçbir veri çekilemedi.")
else:
    # Tüm verileri birleştirirken index'i sıfırla
    all_data = pd.concat(all_data, ignore_index=True)

    # Gerekli sütunları seç
    all_data = all_data[['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']]

    # Eksik verileri doldurma işlemleri
    all_data = all_data.infer_objects()
    all_data.interpolate(method='linear', inplace=True)  # Doğrusal olarak eksik verileri doldur
    all_data.ffill(inplace=True)  # İleri doldurma
    all_data.bfill(inplace=True)  # Geri doldurma

    # Geçerli tarihi al
    current_date = datetime.now().strftime("%Y%m%d_%H%M")

    # Verileri kaydetmek için dizin oluştur
    data_directory = save_directory / f"{current_date}_veriler"
    os.makedirs(data_directory, exist_ok=True)

    # CSV dosyasını kaydetme
    try:
        file_path = data_directory / f"{current_date}_hisse_verileri.csv"
        all_data.to_csv(file_path, index=False, sep=';')
        print(f"Veriler başarıyla kaydedildi: {file_path}")
    except Exception as e:
        print(f"Dosya kaydedilirken hata oluştu: {e}")

    # Tahminleri ve gerçek veriyi depolamak için boş bir DataFrame oluştur
    predictions = []

    for ticker in tickers:
        # Prophet'e uygun veri çerçevesi oluştur
        ticker_data = all_data[all_data['Ticker'] == ticker][['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})

        # Eksik değerleri kaldır
        ticker_data = ticker_data.dropna()

        # Zaman dilimini kaldır
        ticker_data['ds'] = ticker_data['ds'].dt.tz_localize(None)

        # Prophet modeli
        model = Prophet(
            daily_seasonality=True,
            yearly_seasonality=True,
            weekly_seasonality=True,
            seasonality_prior_scale=10.0,  # Sezonluk bileşenlerin etkisini artır
            changepoint_prior_scale=0.5  # Değişim noktalarının esnekliğini artır
        )
        model.fit(ticker_data)

        # Geçmiş veriler ve ileriye dönük 30 gün için tahmin
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)

        # Tahmin edilen fiyatları ve gerçek fiyatları birleştir
        forecast['Real Price'] = ticker_data['y'].tolist() + [None] * 30
        forecast['Ticker'] = ticker
        predictions.append(forecast[['ds', 'Ticker', 'yhat', 'Real Price']])

    # Tüm tahminleri birleştir
    predictions = pd.concat(predictions, ignore_index=True)

    # Grafikler için dizin oluştur
    graph_folder = graph_directory / f"{current_date}_grafikler"
    os.makedirs(graph_folder, exist_ok=True)

    # Grafik oluşturma ve kaydetme
    for ticker in tickers:
        ticker_data = predictions[predictions['Ticker'] == ticker]

        # Grafik oluştur
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ticker_data['ds'], y=ticker_data['Real Price'], mode='lines', name='Gerçek Fiyat'))
        fig.add_trace(go.Scatter(x=ticker_data['ds'], y=ticker_data['yhat'], mode='lines', name='Tahmin Edilen Fiyat', line=dict(dash='dot')))

        # Grafik başlığı ve etiketler
        fig.update_layout(
            title=f"{ticker} Fiyat Tahmini",
            xaxis_title="Tarih",
            yaxis_title="Fiyat",
            xaxis_rangeslider_visible=True
        )

        # Grafiği HTML dosyası olarak kaydet
        graph_path = graph_folder / f"{ticker}_fiyat_tahmini_{current_date}.html"
        fig.write_html(graph_path)

    # CSV dosyasını oku ve tabloyu göster
    try:
        data = pd.read_csv(file_path, sep=';')
        print("Tablo:")
        print(data)
    except Exception as e:
        print(f"Dosya okunurken hata oluştu: {e}")

    print(f"Tahminler başarıyla kaydedildi: {graph_folder}")
