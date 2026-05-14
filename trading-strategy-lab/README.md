# Trading Strategy Lab

這個資料夾包含交易策略庫網站的 GitHub Pages 入口頁與 Streamlit app。

## GitHub Pages 入口

公開入口頁：

https://shirley-weg.github.io/trading-strategy-lab/

## Streamlit Cloud 部署

GitHub Pages 不能直接執行 Python、yfinance 或 TAIFEX 抓資料。若要讓網站可以即時跑回測，請在 Streamlit Cloud 建立 app：

```text
Repository: shirley-weg/shirley-weg.github.io
Branch: main
Main file path: trading-strategy-lab/streamlit_app.py
```

需要的 Python 套件已放在 repo root 的 `requirements.txt`，也在本資料夾保留一份 `requirements.txt`。

## 目前功能

- Notebook preset pair 與自訂 pair
- Yahoo Finance 股票資料下載
- Rolling OLS spread / z-score
- t 日 Close 產生訊號，t+1 日 Open 進出場
- 手續費與賣出稅率
- OLS hedge ratio 或 rolling max-Sharpe grid 權重
- 權益曲線、benchmark 比較、drawdown、spread signal、trade P&L、entry weights、交易明細

## 研究提醒

本工具僅供研究與教學使用，不構成投資建議。回測結果會受到資料品質、交易成本、滑價、流動性、參數過度最佳化與市場制度變化影響。
