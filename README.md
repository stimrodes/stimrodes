import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def get_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

if __name__ == '__main__':
    ticker = 'ODAS.IS'  # Örnek olarak Apple Inc. (AAPL) hisse senedi sembolü kullanıyoruz.
    start_date = '2023-01-01'
    end_date = '2023-07-20'

    stock_data = get_stock_data(ticker, start_date, end_date)

    # Basit Lineer Regresyon için verileri hazırlama
    X = pd.DataFrame(stock_data['Open'])  # Açılış fiyatını kullanıyoruz.
    y = stock_data['Close']  # Kapanış fiyatını tahmin etmek istiyoruz.

    # Eğitim ve test verilerine ayırma
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Lineer Regresyon modeli oluşturma ve eğitim
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Test verileri üzerinde tahmin yapma
    y_pred = model.predict(X_test)

    # Sonuçları görselleştirme
    plt.figure(figsize=(12, 6))
    plt.scatter(X_test, y_test, color='blue', label='Gerçek Değerler')
    plt.plot(X_test, y_pred, color='red', linewidth=2, label='Tahminler')
    plt.xlabel('Açılış Fiyatı')
    plt.ylabel('Kapanış Fiyatı')
    plt.title('Lineer Regresyon ile Hisse Senedi Fiyat Tahmini')
    plt.legend()
    plt.grid()
    plt.show()
