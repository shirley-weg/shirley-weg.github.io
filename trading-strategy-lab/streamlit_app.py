
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Literal
import itertools
import logging
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import statsmodels.api as sm
import yfinance as yf
from statsmodels.tsa.stattools import coint

logging.getLogger("yfinance").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore")

single_stock_futures = [{'code': '1101', 'name': '台泥', 'industry': '水泥工業'},
 {'code': '1102', 'name': '亞泥', 'industry': '水泥工業'},
 {'code': '1210', 'name': '大成', 'industry': '食品工業'},
 {'code': '1216', 'name': '統一', 'industry': '食品工業'},
 {'code': '1301', 'name': '台塑', 'industry': '塑膠工業'},
 {'code': '1303', 'name': '南亞', 'industry': '塑膠工業'},
 {'code': '1312', 'name': '國喬', 'industry': '塑膠工業'},
 {'code': '1314', 'name': '中石化', 'industry': '塑膠工業'},
 {'code': '1319', 'name': '東陽', 'industry': '汽車工業'},
 {'code': '1326', 'name': '台化', 'industry': '塑膠工業'},
 {'code': '1402', 'name': '遠東新', 'industry': '紡織纖維'},
 {'code': '1440', 'name': '南紡', 'industry': '紡織纖維'},
 {'code': '1476', 'name': '儒鴻', 'industry': '紡織纖維'},
 {'code': '1477', 'name': '聚陽', 'industry': '紡織纖維'},
 {'code': '1503', 'name': '士電', 'industry': '電機機械'},
 {'code': '1504', 'name': '東元', 'industry': '電機機械'},
 {'code': '1513', 'name': '中興電', 'industry': '電機機械'},
 {'code': '1536', 'name': '和大', 'industry': '汽車工業'},
 {'code': '1560', 'name': '中砂', 'industry': '電機機械'},
 {'code': '1565', 'name': '精華', 'industry': '生技醫療業'},
 {'code': '1590', 'name': '亞德客-KY', 'industry': '電機機械'},
 {'code': '1605', 'name': '華新', 'industry': '電器電纜'},
 {'code': '1608', 'name': '華榮', 'industry': '電器電纜'},
 {'code': '1609', 'name': '大亞', 'industry': '電器電纜'},
 {'code': '1717', 'name': '長興', 'industry': '化學工業'},
 {'code': '1718', 'name': '中纖', 'industry': '化學工業'},
 {'code': '1722', 'name': '台肥', 'industry': '化學工業'},
 {'code': '1795', 'name': '美時', 'industry': '生技醫療業'},
 {'code': '1802', 'name': '台玻', 'industry': '玻璃陶瓷'},
 {'code': '1904', 'name': '正隆', 'industry': '造紙工業'},
 {'code': '1905', 'name': '華紙', 'industry': '造紙工業'},
 {'code': '1907', 'name': '永豐餘', 'industry': '造紙工業'},
 {'code': '1909', 'name': '榮成', 'industry': '造紙工業'},
 {'code': '2002', 'name': '中鋼', 'industry': '鋼鐵工業'},
 {'code': '2006', 'name': '東和鋼鐵', 'industry': '鋼鐵工業'},
 {'code': '2014', 'name': '中鴻', 'industry': '鋼鐵工業'},
 {'code': '2027', 'name': '大成鋼', 'industry': '鋼鐵工業'},
 {'code': '2049', 'name': '上銀', 'industry': '電機機械'},
 {'code': '2059', 'name': '川湖', 'industry': '電子零組件業'},
 {'code': '2105', 'name': '正新', 'industry': '橡膠工業'},
 {'code': '2201', 'name': '裕隆', 'industry': '汽車工業'},
 {'code': '2231', 'name': '為升', 'industry': '汽車工業'},
 {'code': '2301', 'name': '光寶科', 'industry': '光電業'},
 {'code': '2303', 'name': '聯電', 'industry': '半導體業'},
 {'code': '2308', 'name': '台達電', 'industry': '電子零組件業'},
 {'code': '2312', 'name': '金寶', 'industry': '其他電子業'},
 {'code': '2313', 'name': '華通', 'industry': '電子零組件業'},
 {'code': '2317', 'name': '鴻海', 'industry': '其他電子業'},
 {'code': '2323', 'name': '中環', 'industry': '光電業'},
 {'code': '2324', 'name': '仁寶', 'industry': '電腦及週邊設備業'},
 {'code': '2327', 'name': '國巨', 'industry': '電子零組件業'},
 {'code': '2328', 'name': '廣宇', 'industry': '電子零組件業'},
 {'code': '2329', 'name': '華泰', 'industry': '半導體業'},
 {'code': '2330', 'name': '台積電', 'industry': '半導體業'},
 {'code': '2331', 'name': '精英', 'industry': '電腦及週邊設備業'},
 {'code': '2332', 'name': '友訊', 'industry': '通信網路業'},
 {'code': '2337', 'name': '旺宏', 'industry': '半導體業'},
 {'code': '2338', 'name': '光罩', 'industry': '半導體業'},
 {'code': '2340', 'name': '台亞', 'industry': '半導體業'},
 {'code': '2344', 'name': '華邦電', 'industry': '半導體業'},
 {'code': '2345', 'name': '智邦', 'industry': '通信網路業'},
 {'code': '2347', 'name': '聯強', 'industry': '電子通路業'},
 {'code': '2352', 'name': '佳世達', 'industry': '電腦及週邊設備業'},
 {'code': '2353', 'name': '宏碁', 'industry': '電腦及週邊設備業'},
 {'code': '2354', 'name': '鴻準', 'industry': '其他電子業'},
 {'code': '2355', 'name': '敬鵬', 'industry': '電子零組件業'},
 {'code': '2356', 'name': '英業達', 'industry': '電腦及週邊設備業'},
 {'code': '2357', 'name': '華碩', 'industry': '電腦及週邊設備業'},
 {'code': '2360', 'name': '致茂', 'industry': '其他電子業'},
 {'code': '2367', 'name': '燿華', 'industry': '電子零組件業'},
 {'code': '2368', 'name': '金像電', 'industry': '電子零組件業'},
 {'code': '2371', 'name': '大同', 'industry': '電機機械'},
 {'code': '2376', 'name': '技嘉', 'industry': '電腦及週邊設備業'},
 {'code': '2377', 'name': '微星', 'industry': '電腦及週邊設備業'},
 {'code': '2379', 'name': '瑞昱', 'industry': '半導體業'},
 {'code': '2382', 'name': '廣達', 'industry': '電腦及週邊設備業'},
 {'code': '2383', 'name': '台光電', 'industry': '電子零組件業'},
 {'code': '2385', 'name': '群光', 'industry': '電子零組件業'},
 {'code': '2388', 'name': '威盛', 'industry': '半導體業'},
 {'code': '2392', 'name': '正崴', 'industry': '電子零組件業'},
 {'code': '2393', 'name': '億光', 'industry': '光電業'},
 {'code': '2395', 'name': '研華', 'industry': '電腦及週邊設備業'},
 {'code': '2401', 'name': '凌陽', 'industry': '半導體業'},
 {'code': '2404', 'name': '漢唐', 'industry': '其他電子業'},
 {'code': '2408', 'name': '南亞科', 'industry': '半導體業'},
 {'code': '2409', 'name': '友達', 'industry': '光電業'},
 {'code': '2412', 'name': '中華電', 'industry': '通信網路業'},
 {'code': '2421', 'name': '建準', 'industry': '電子零組件業'},
 {'code': '2439', 'name': '美律', 'industry': '通信網路業'},
 {'code': '2441', 'name': '超豐', 'industry': '半導體業'},
 {'code': '2449', 'name': '京元電子', 'industry': '半導體業'},
 {'code': '2454', 'name': '聯發科', 'industry': '半導體業'},
 {'code': '2455', 'name': '全新', 'industry': '通信網路業'},
 {'code': '2457', 'name': '飛宏', 'industry': '電子零組件業'},
 {'code': '2458', 'name': '義隆', 'industry': '半導體業'},
 {'code': '2474', 'name': '可成', 'industry': '其他電子業'},
 {'code': '2481', 'name': '強茂', 'industry': '半導體業'},
 {'code': '2485', 'name': '兆赫', 'industry': '通信網路業'},
 {'code': '2486', 'name': '一詮', 'industry': '光電業'},
 {'code': '2489', 'name': '瑞軒', 'industry': '光電業'},
 {'code': '2492', 'name': '華新科', 'industry': '電子零組件業'},
 {'code': '2498', 'name': '宏達電', 'industry': '通信網路業'},
 {'code': '2515', 'name': '中工', 'industry': '建材營造業'},
 {'code': '2520', 'name': '冠德', 'industry': '建材營造業'},
 {'code': '2542', 'name': '興富發', 'industry': '建材營造業'},
 {'code': '2548', 'name': '華固', 'industry': '建材營造業'},
 {'code': '2603', 'name': '長榮', 'industry': '航運業'},
 {'code': '2605', 'name': '新興', 'industry': '航運業'},
 {'code': '2606', 'name': '裕民', 'industry': '航運業'},
 {'code': '2609', 'name': '陽明', 'industry': '航運業'},
 {'code': '2610', 'name': '華航', 'industry': '航運業'},
 {'code': '2615', 'name': '萬海', 'industry': '航運業'},
 {'code': '2618', 'name': '長榮航', 'industry': '航運業'},
 {'code': '2633', 'name': '台灣高鐵', 'industry': '航運業'},
 {'code': '2634', 'name': '漢翔', 'industry': '航運業'},
 {'code': '2801', 'name': '彰銀', 'industry': '金融保險業'},
 {'code': '2834', 'name': '臺企銀', 'industry': '金融保險業'},
 {'code': '2880', 'name': '華南金', 'industry': '金融保險業'},
 {'code': '2881', 'name': '富邦金', 'industry': '金融保險業'},
 {'code': '2882', 'name': '國泰金', 'industry': '金融保險業'},
 {'code': '2883', 'name': '凱基金', 'industry': '金融保險業'},
 {'code': '2884', 'name': '玉山金', 'industry': '金融保險業'},
 {'code': '2885', 'name': '元大金', 'industry': '金融保險業'},
 {'code': '2886', 'name': '兆豐金', 'industry': '金融保險業'},
 {'code': '2887', 'name': '台新新光金', 'industry': '金融保險業'},
 {'code': '2890', 'name': '永豐金', 'industry': '金融保險業'},
 {'code': '2891', 'name': '中信金', 'industry': '金融保險業'},
 {'code': '2892', 'name': '第一金', 'industry': '金融保險業'},
 {'code': '2913', 'name': '農林', 'industry': '貿易百貨業'},
 {'code': '2915', 'name': '潤泰全', 'industry': '其他業'},
 {'code': '3005', 'name': '神基', 'industry': '電腦及週邊設備業'},
 {'code': '3006', 'name': '晶豪科', 'industry': '半導體業'},
 {'code': '3008', 'name': '大立光', 'industry': '光電業'},
 {'code': '3017', 'name': '奇鋐', 'industry': '電腦及週邊設備業'},
 {'code': '3019', 'name': '亞光', 'industry': '光電業'},
 {'code': '3034', 'name': '聯詠', 'industry': '半導體業'},
 {'code': '3035', 'name': '智原', 'industry': '半導體業'},
 {'code': '3036', 'name': '文曄', 'industry': '電子通路業'},
 {'code': '3037', 'name': '欣興', 'industry': '電子零組件業'},
 {'code': '3042', 'name': '晶技', 'industry': '電子零組件業'},
 {'code': '3044', 'name': '健鼎', 'industry': '電子零組件業'},
 {'code': '3045', 'name': '台灣大', 'industry': '通信網路業'},
 {'code': '3078', 'name': '僑威', 'industry': '電子零組件業'},
 {'code': '3081', 'name': '聯亞', 'industry': '通信網路業'},
 {'code': '3105', 'name': '穩懋', 'industry': '半導體業'},
 {'code': '3152', 'name': '璟德', 'industry': '通信網路業'},
 {'code': '3189', 'name': '景碩', 'industry': '半導體業'},
 {'code': '3211', 'name': '順達', 'industry': '電腦及週邊設備業'},
 {'code': '3227', 'name': '原相', 'industry': '半導體業'},
 {'code': '3231', 'name': '緯創', 'industry': '電腦及週邊設備業'},
 {'code': '3260', 'name': '威剛', 'industry': '半導體業'},
 {'code': '3264', 'name': '欣銓', 'industry': '半導體業'},
 {'code': '3293', 'name': '鈊象', 'industry': '文化創意業'},
 {'code': '3324', 'name': '雙鴻', 'industry': '電腦及週邊設備業'},
 {'code': '3374', 'name': '精材', 'industry': '半導體業'},
 {'code': '3376', 'name': '新日興', 'industry': '電子零組件業'},
 {'code': '3380', 'name': '明泰', 'industry': '通信網路業'},
 {'code': '3406', 'name': '玉晶光', 'industry': '光電業'},
 {'code': '3443', 'name': '創意', 'industry': '半導體業'},
 {'code': '3481', 'name': '群創', 'industry': '光電業'},
 {'code': '3529', 'name': '力旺', 'industry': '半導體業'},
 {'code': '3532', 'name': '台勝科', 'industry': '半導體業'},
 {'code': '3533', 'name': '嘉澤', 'industry': '電子零組件業'},
 {'code': '3552', 'name': '同致', 'industry': '汽車工業'},
 {'code': '3653', 'name': '健策', 'industry': '電子零組件業'},
 {'code': '3661', 'name': '世芯-KY', 'industry': '半導體業'},
 {'code': '3665', 'name': '貿聯-KY', 'industry': '其他電子業'},
 {'code': '3673', 'name': 'TPK-KY', 'industry': '光電業'},
 {'code': '3680', 'name': '家登', 'industry': '其他電子業'},
 {'code': '3691', 'name': '碩禾', 'industry': '光電業'},
 {'code': '3702', 'name': '大聯大', 'industry': '電子通路業'},
 {'code': '3706', 'name': '神達', 'industry': '電腦及週邊設備業'},
 {'code': '3711', 'name': '日月光投控', 'industry': '半導體業'},
 {'code': '3714', 'name': '富采', 'industry': '光電業'},
 {'code': '4123', 'name': '晟德', 'industry': '生技醫療業'},
 {'code': '4128', 'name': '中天', 'industry': '生技醫療業'},
 {'code': '4162', 'name': '智擎', 'industry': '生技醫療業'},
 {'code': '4736', 'name': '泰博', 'industry': '生技醫療業'},
 {'code': '4743', 'name': '合一', 'industry': '生技醫療業'},
 {'code': '4904', 'name': '遠傳', 'industry': '通信網路業'},
 {'code': '4919', 'name': '新唐', 'industry': '半導體業'},
 {'code': '4938', 'name': '和碩', 'industry': '電腦及週邊設備業'},
 {'code': '4958', 'name': '臻鼎-KY', 'industry': '電子零組件業'},
 {'code': '5009', 'name': '榮剛', 'industry': '鋼鐵工業'},
 {'code': '5269', 'name': '祥碩', 'industry': '半導體業'},
 {'code': '5274', 'name': '信驊', 'industry': '半導體業'},
 {'code': '5347', 'name': '世界', 'industry': '半導體業'},
 {'code': '5371', 'name': '中光電', 'industry': '光電業'},
 {'code': '5388', 'name': '中磊', 'industry': '通信網路業'},
 {'code': '5425', 'name': '台半', 'industry': '半導體業'},
 {'code': '5457', 'name': '宣德', 'industry': '電子零組件業'},
 {'code': '5483', 'name': '中美晶', 'industry': '半導體業'},
 {'code': '5534', 'name': '長虹', 'industry': '建材營造業'},
 {'code': '5871', 'name': '中租-KY', 'industry': '其他業'},
 {'code': '5876', 'name': '上海商銀', 'industry': '金融保險業'},
 {'code': '5880', 'name': '合庫金', 'industry': '金融保險業'},
 {'code': '5904', 'name': '寶雅', 'industry': '貿易百貨業'},
 {'code': '6005', 'name': '群益證', 'industry': '金融保險業'},
 {'code': '6116', 'name': '彩晶', 'industry': '光電業'},
 {'code': '6121', 'name': '新普', 'industry': '電腦及週邊設備業'},
 {'code': '6139', 'name': '亞翔', 'industry': '其他電子業'},
 {'code': '6147', 'name': '頎邦', 'industry': '半導體業'},
 {'code': '6153', 'name': '嘉聯益', 'industry': '電子零組件業'},
 {'code': '6173', 'name': '信昌電', 'industry': '電子零組件業'},
 {'code': '6176', 'name': '瑞儀', 'industry': '光電業'},
 {'code': '6182', 'name': '合晶', 'industry': '半導體業'},
 {'code': '6188', 'name': '廣明', 'industry': '光電業'},
 {'code': '6213', 'name': '聯茂', 'industry': '電子零組件業'},
 {'code': '6223', 'name': '旺矽', 'industry': '半導體業'},
 {'code': '6239', 'name': '力成', 'industry': '半導體業'},
 {'code': '6245', 'name': '立端', 'industry': '電腦及週邊設備業'},
 {'code': '6257', 'name': '矽格', 'industry': '半導體業'},
 {'code': '6269', 'name': '台郡', 'industry': '電子零組件業'},
 {'code': '6271', 'name': '同欣電', 'industry': '半導體業'},
 {'code': '6274', 'name': '台燿', 'industry': '電子零組件業'},
 {'code': '6278', 'name': '台表科', 'industry': '電子零組件業'},
 {'code': '6279', 'name': '胡連', 'industry': '電子零組件業'},
 {'code': '6282', 'name': '康舒', 'industry': '電子零組件業'},
 {'code': '6285', 'name': '啟碁', 'industry': '通信網路業'},
 {'code': '6290', 'name': '良維', 'industry': '電子零組件業'},
 {'code': '6414', 'name': '樺漢', 'industry': '電腦及週邊設備業'},
 {'code': '6443', 'name': '元晶', 'industry': '光電業'},
 {'code': '6472', 'name': '保瑞', 'industry': '生技醫療業'},
 {'code': '6488', 'name': '環球晶', 'industry': '半導體業'},
 {'code': '6505', 'name': '台塑化', 'industry': '油電燃氣業'},
 {'code': '6510', 'name': '精測', 'industry': '半導體業'},
 {'code': '6526', 'name': '達發', 'industry': '半導體業'},
 {'code': '6547', 'name': '高端疫苗', 'industry': '生技醫療業'},
 {'code': '6669', 'name': '緯穎', 'industry': '電腦及週邊設備業'},
 {'code': '6757', 'name': '台灣虎航', 'industry': '航運業'},
 {'code': '6770', 'name': '力積電', 'industry': '半導體業'},
 {'code': '8039', 'name': '台虹', 'industry': '電子零組件業'},
 {'code': '8044', 'name': '網家', 'industry': '數位雲端'},
 {'code': '8046', 'name': '南電', 'industry': '電子零組件業'},
 {'code': '8069', 'name': '元太', 'industry': '光電業'},
 {'code': '8086', 'name': '宏捷科', 'industry': '半導體業'},
 {'code': '8112', 'name': '至上', 'industry': '電子通路業'},
 {'code': '8150', 'name': '南茂', 'industry': '半導體業'},
 {'code': '8163', 'name': '達方', 'industry': '電子零組件業'},
 {'code': '8299', 'name': '群聯', 'industry': '半導體業'},
 {'code': '8358', 'name': '金居', 'industry': '電子零組件業'},
 {'code': '8436', 'name': '大江', 'industry': '生技醫療業'},
 {'code': '9904', 'name': '寶成', 'industry': '運動休閒'},
 {'code': '9914', 'name': '美利達', 'industry': '運動休閒'},
 {'code': '9938', 'name': '百和', 'industry': '其他業'},
 {'code': '9939', 'name': '宏全', 'industry': '其他業'},
 {'code': '9945', 'name': '潤泰新', 'industry': '建材營造業'},
 {'code': '9958', 'name': '世紀鋼', 'industry': '鋼鐵工業'}]


# ============================================================
# Streamlit App Settings
# ============================================================

st.set_page_config(page_title="Trading Strategy Lab", layout="wide")
st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; }
    div[data-testid="stMetric"] {
      background: #fff; border: 1px solid #e6e8ef; border-radius: 8px;
      padding: 14px 16px; box-shadow: 0 1px 2px rgba(20, 29, 47, 0.04);
    }
    .strategy-card {
      background: #ffffff;
      border: 1px solid #e6e8ef;
      border-radius: 12px;
      padding: 24px 26px;
      box-shadow: 0 1px 2px rgba(20, 29, 47, 0.04);
      margin-top: 16px;
      margin-bottom: 12px;
      min-height: 190px;
    }
    .strategy-card h3 { margin-top: 0; margin-bottom: 8px; }
    .strategy-card p { color: #607080; margin-bottom: 0; line-height: 1.7; }
    .small-note { color:#607080; font-size:0.92rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

StrategyName = Literal["distance", "cointegration", "ff5_alpha"]


@dataclass(frozen=True)
class AppSettings:
    strategy: StrategyName
    industry: str
    data_start: pd.Timestamp
    trading_start: pd.Timestamp
    trading_end: pd.Timestamp | None
    market_ticker: str
    lookback_days: int
    weekly_freq: str
    k_final: int
    preselect_multiplier: int
    min_obs: int
    pvalue_threshold: float
    beta_diff_threshold: float
    entry_z: float
    exit_z: float
    stop_z: float
    max_holding_days: int
    reentry_reset_z: float
    fee_rate: float
    initial_capital: float
    auto_adjust: bool
    top_n_display: int

    @property
    def preselect_n(self) -> int:
        return int(self.k_final * self.preselect_multiplier)


# ============================================================
# Basic Universe Helpers
# ============================================================

def clean_stock_id(s: pd.Series) -> pd.Series:
    """
    將股票代號統一成 4 碼字串。
    例如：2330、2330.0、"2330 台積電" 都會轉成 "2330"。
    """
    return (
        s.astype(str)
         .str.extract(r"(\d+)")[0]
         .str.zfill(4)
    )


def stock_pool_df() -> pd.DataFrame:
    df = pd.DataFrame(single_stock_futures).copy()
    df["code"] = df["code"].astype(str)
    df["ticker_yf"] = df["code"].map(to_yf_ticker)
    return df


def to_yf_ticker(code_or_ticker: str) -> str:
    value = str(code_or_ticker).strip().upper()
    if not value:
        return value
    if value.startswith("^"):
        return value
    if value.endswith(".TW") or value.endswith(".TWO"):
        return value
    if value.isdigit():
        return f"{value}.TW"
    return value


def code_from_ticker(ticker: str) -> str:
    ticker = str(ticker)
    return ticker.replace(".TW", "").replace(".TWO", "")


def resolve_stock_name(code_or_ticker: str) -> str:
    code = code_from_ticker(code_or_ticker)
    df = stock_pool_df()
    row = df[df["code"] == code]
    if row.empty:
        return ""
    return str(row.iloc[0]["name"])


def resolve_stock_industry(code_or_ticker: str) -> str:
    code = code_from_ticker(code_or_ticker)
    df = stock_pool_df()
    row = df[df["code"] == code]
    if row.empty:
        return ""
    return str(row.iloc[0]["industry"])


def format_stock_label(code_or_ticker: str) -> str:
    code = code_from_ticker(code_or_ticker)
    name = resolve_stock_name(code)
    return f"{code} {name}" if name else code


def get_industry_options() -> list[str]:
    df = stock_pool_df()
    industries = sorted(df["industry"].dropna().unique().tolist())
    return ["全部"] + industries


def get_universe_by_industry(industry: str) -> pd.DataFrame:
    df = stock_pool_df()
    if industry != "全部":
        df = df[df["industry"] == industry].copy()
    return df.reset_index(drop=True)


def canonical_pair_name(ticker_x: str, ticker_y: str) -> str:
    a, b = sorted([str(ticker_x), str(ticker_y)])
    return f"{a}_{b}"


def display_pair_name(ticker_x: str, ticker_y: str) -> str:
    return f"{ticker_x}_{ticker_y}"


# ============================================================
# Data Download
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def download_ohlc_data_cached(
    tickers_tuple: tuple[str, ...],
    start: str,
    end: str | None,
    auto_adjust: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tickers = list(dict.fromkeys(tickers_tuple))
    if len(tickers) == 0:
        return pd.DataFrame(), pd.DataFrame()

    yf_end = None
    if end:
        yf_end = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    data = yf.download(
        tickers=tickers,
        start=start,
        end=yf_end,
        auto_adjust=auto_adjust,
        progress=False,
        group_by="column",
        threads=True,
    )

    if data.empty:
        return pd.DataFrame(), pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        open_df = data["Open"].copy() if "Open" in data.columns.get_level_values(0) else pd.DataFrame()
        close_df = data["Close"].copy() if "Close" in data.columns.get_level_values(0) else pd.DataFrame()
    else:
        ticker = tickers[0]
        open_df = data[["Open"]].copy()
        close_df = data[["Close"]].copy()
        open_df.columns = [ticker]
        close_df.columns = [ticker]

    open_df = open_df.sort_index().dropna(axis=1, how="all")
    close_df = close_df.sort_index().dropna(axis=1, how="all")

    # yfinance sometimes returns columns in a different order; keep only requested tickers.
    open_df = open_df[[c for c in tickers if c in open_df.columns]]
    close_df = close_df[[c for c in tickers if c in close_df.columns]]

    return open_df, close_df


@st.cache_data(ttl=3600, show_spinner=False)
def download_market_price_cached(
    market_ticker: str,
    start: str,
    end: str | None,
    auto_adjust: bool = True,
) -> pd.Series:
    yf_end = None
    if end:
        yf_end = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    data = yf.download(
        market_ticker,
        start=start,
        end=yf_end,
        auto_adjust=auto_adjust,
        progress=False,
    )
    if data.empty:
        return pd.Series(dtype=float, name="market")
    if isinstance(data.columns, pd.MultiIndex):
        price = data["Close"].iloc[:, 0].copy()
    else:
        price = data["Close"].copy()
    price = price.sort_index()
    price.name = "market"
    return price


def clean_ohlc_data(open_df: pd.DataFrame, close_df: pd.DataFrame, min_non_na_ratio: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    common_cols = open_df.columns.intersection(close_df.columns)
    open_df = open_df[common_cols].copy()
    close_df = close_df[common_cols].copy()

    if len(close_df) == 0 or len(common_cols) == 0:
        return open_df, close_df

    min_obs_required = int(len(close_df) * min_non_na_ratio)
    valid_cols = close_df.dropna(axis=1, thresh=min_obs_required).columns

    open_clean = open_df[valid_cols].ffill()
    close_clean = close_df[valid_cols].ffill()

    common_index = open_clean.index.intersection(close_clean.index)
    open_clean = open_clean.loc[common_index]
    close_clean = close_clean.loc[common_index]

    return open_clean, close_clean


def prepare_download_for_universe(settings: AppSettings, extra_tickers: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
    universe = get_universe_by_industry(settings.industry)
    tickers = universe["ticker_yf"].tolist()
    if extra_tickers:
        tickers = list(dict.fromkeys(tickers + [to_yf_ticker(t) for t in extra_tickers]))

    with st.spinner(f"下載 {len(tickers)} 檔股票價格資料..."):
        open_df, close_df = download_ohlc_data_cached(
            tuple(tickers),
            settings.data_start.strftime("%Y-%m-%d"),
            settings.trading_end.strftime("%Y-%m-%d") if settings.trading_end is not None else None,
            settings.auto_adjust,
        )
        open_df, close_df = clean_ohlc_data(open_df, close_df, min_non_na_ratio=0.8)

    with st.spinner("下載大盤資料以估計 beta..."):
        market_price = download_market_price_cached(
            settings.market_ticker,
            settings.data_start.strftime("%Y-%m-%d"),
            settings.trading_end.strftime("%Y-%m-%d") if settings.trading_end is not None else None,
            settings.auto_adjust,
        )

    return open_df, close_df, market_price, universe


# ============================================================
# Pair Selection Helpers
# ============================================================

def estimate_beta_from_price(stock_price: pd.Series, market_price: pd.Series, min_obs: int = 200) -> float:
    data = pd.concat([stock_price, market_price], axis=1).dropna()
    data.columns = ["stock", "market"]
    returns = np.log(data).diff().dropna()
    if len(returns) < min_obs:
        return np.nan
    market_var = np.var(returns["market"].values, ddof=0)
    if market_var == 0:
        return np.nan
    beta = np.cov(returns["stock"].values, returns["market"].values, ddof=0)[0, 1] / market_var
    return float(beta)


def build_beta_table_for_candidate_pairs(
    candidate_pairs: pd.DataFrame,
    formation_price_df: pd.DataFrame,
    stock_info_df: pd.DataFrame,
    market_price_full: pd.Series,
    min_obs: int = 200,
) -> pd.DataFrame:
    if candidate_pairs is None or len(candidate_pairs) == 0:
        return pd.DataFrame(columns=["ticker_yf", "code", "name", "industry", "beta"])

    used_tickers = sorted(set(candidate_pairs["ticker_x"]).union(set(candidate_pairs["ticker_y"])))
    formation_start = formation_price_df.index.min()
    formation_end = formation_price_df.index.max()
    market_price = market_price_full.loc[formation_start:formation_end].copy()
    info = stock_info_df.set_index("ticker_yf").copy()
    records: list[dict[str, object]] = []

    for ticker in used_tickers:
        if ticker not in formation_price_df.columns:
            beta = np.nan
        else:
            beta = estimate_beta_from_price(formation_price_df[ticker], market_price, min_obs=min_obs)
        records.append({
            "ticker_yf": ticker,
            "code": info.loc[ticker, "code"] if ticker in info.index else code_from_ticker(ticker),
            "name": info.loc[ticker, "name"] if ticker in info.index else resolve_stock_name(ticker),
            "industry": info.loc[ticker, "industry"] if ticker in info.index else resolve_stock_industry(ticker),
            "beta": beta,
        })
    return pd.DataFrame(records)


def apply_beta_and_industry_filter(
    candidate_pairs: pd.DataFrame,
    beta_df: pd.DataFrame,
    beta_diff_threshold: float = 0.2,
    final_k: int = 10,
    sort_by: dict[str, list] | None = None,
) -> pd.DataFrame:
    if candidate_pairs is None or len(candidate_pairs) == 0 or beta_df is None or len(beta_df) == 0:
        return pd.DataFrame()
    beta_map = beta_df.set_index("ticker_yf")["beta"].to_dict()
    filtered = candidate_pairs.copy()
    filtered["beta_x"] = filtered["ticker_x"].map(beta_map)
    filtered["beta_y"] = filtered["ticker_y"].map(beta_map)
    filtered["beta_diff"] = (filtered["beta_x"] - filtered["beta_y"]).abs()
    filtered["same_industry"] = filtered["industry_x"] == filtered["industry_y"]
    filtered = filtered.dropna(subset=["beta_x", "beta_y"])
    filtered = filtered[(filtered["same_industry"]) & (filtered["beta_diff"] < beta_diff_threshold)].copy()
    if len(filtered) == 0:
        return filtered
    if sort_by is not None:
        filtered = filtered.sort_values(by=sort_by["by"], ascending=sort_by["ascending"]).reset_index(drop=True)
    else:
        filtered = filtered.reset_index(drop=True)
    return filtered.head(final_k).copy()


def select_pairs_distance_preselect(formation_prices: pd.DataFrame, stock_info_df: pd.DataFrame, preselect_n: int = 100, min_obs: int = 200) -> tuple[pd.DataFrame, pd.DataFrame]:
    info = stock_info_df.set_index("ticker_yf").copy()
    prices = formation_prices.copy().dropna(axis=1, how="all").ffill()
    log_prices = np.log(prices)
    normalized = (log_prices - log_prices.mean()) / log_prices.std(ddof=0)
    tickers = [ticker for ticker in normalized.columns if ticker in info.index]
    results: list[dict[str, object]] = []

    for ticker_x, ticker_y in itertools.combinations(tickers, 2):
        pair_data = normalized[[ticker_x, ticker_y]].dropna()
        if len(pair_data) < min_obs:
            continue
        distance = float(np.sum((pair_data[ticker_x].values - pair_data[ticker_y].values) ** 2))
        results.append({
            "ticker_x": ticker_x, "code_x": info.loc[ticker_x, "code"], "name_x": info.loc[ticker_x, "name"], "industry_x": info.loc[ticker_x, "industry"],
            "ticker_y": ticker_y, "code_y": info.loc[ticker_y, "code"], "name_y": info.loc[ticker_y, "name"], "industry_y": info.loc[ticker_y, "industry"],
            "n_obs": len(pair_data), "distance": distance,
        })
    all_results = pd.DataFrame(results)
    if len(all_results) == 0:
        return all_results, all_results
    all_results = all_results.sort_values("distance", ascending=True).reset_index(drop=True)
    return all_results, all_results.head(preselect_n).copy()


def coint_test_bidirectional(x: pd.Series, y: pd.Series, trend: str = "c", maxlag: int | None = None, autolag: str = "aic") -> dict[str, object]:
    coint_t_xy, p_xy, crit_xy = coint(x, y, trend=trend, maxlag=maxlag, autolag=autolag)
    coint_t_yx, p_yx, crit_yx = coint(y, x, trend=trend, maxlag=maxlag, autolag=autolag)
    if p_xy <= p_yx:
        return {"coint_t": coint_t_xy, "p_value": p_xy, "crit_1pct": crit_xy[0], "crit_5pct": crit_xy[1], "crit_10pct": crit_xy[2], "direction": "x_on_y", "coint_t_xy": coint_t_xy, "p_value_xy": p_xy, "coint_t_yx": coint_t_yx, "p_value_yx": p_yx}
    return {"coint_t": coint_t_yx, "p_value": p_yx, "crit_1pct": crit_yx[0], "crit_5pct": crit_yx[1], "crit_10pct": crit_yx[2], "direction": "y_on_x", "coint_t_xy": coint_t_xy, "p_value_xy": p_xy, "coint_t_yx": coint_t_yx, "p_value_yx": p_yx}


def select_pairs_cointegration_preselect(
    formation_prices: pd.DataFrame,
    stock_info_df: pd.DataFrame,
    pvalue_threshold: float = 0.05,
    preselect_n: int = 100,
    min_obs: int = 200,
    trend: str = "c",
    maxlag: int | None = None,
    autolag: str = "aic",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    info = stock_info_df.set_index("ticker_yf").copy()
    prices = formation_prices.copy().dropna(axis=1, how="all").ffill()
    log_prices = np.log(prices)
    tickers = [ticker for ticker in log_prices.columns if ticker in info.index]
    results: list[dict[str, object]] = []

    for ticker_x, ticker_y in itertools.combinations(tickers, 2):
        pair_data = log_prices[[ticker_x, ticker_y]].dropna()
        if len(pair_data) < min_obs:
            continue
        try:
            test_result = coint_test_bidirectional(pair_data[ticker_x], pair_data[ticker_y], trend=trend, maxlag=maxlag, autolag=autolag)
            row = {
                "ticker_x": ticker_x, "code_x": info.loc[ticker_x, "code"], "name_x": info.loc[ticker_x, "name"], "industry_x": info.loc[ticker_x, "industry"],
                "ticker_y": ticker_y, "code_y": info.loc[ticker_y, "code"], "name_y": info.loc[ticker_y, "name"], "industry_y": info.loc[ticker_y, "industry"],
                "n_obs": len(pair_data),
                "coint_t": test_result["coint_t"], "p_value": test_result["p_value"], "crit_1pct": test_result["crit_1pct"], "crit_5pct": test_result["crit_5pct"], "crit_10pct": test_result["crit_10pct"],
                "direction": test_result["direction"], "coint_t_xy": test_result["coint_t_xy"], "p_value_xy": test_result["p_value_xy"], "coint_t_yx": test_result["coint_t_yx"], "p_value_yx": test_result["p_value_yx"],
                "is_cointegrated": float(test_result["p_value"]) <= pvalue_threshold,
            }
        except Exception as exc:
            row = {
                "ticker_x": ticker_x, "code_x": info.loc[ticker_x, "code"], "name_x": info.loc[ticker_x, "name"], "industry_x": info.loc[ticker_x, "industry"],
                "ticker_y": ticker_y, "code_y": info.loc[ticker_y, "code"], "name_y": info.loc[ticker_y, "name"], "industry_y": info.loc[ticker_y, "industry"],
                "n_obs": len(pair_data), "coint_t": np.nan, "p_value": np.nan, "crit_1pct": np.nan, "crit_5pct": np.nan, "crit_10pct": np.nan,
                "direction": None, "coint_t_xy": np.nan, "p_value_xy": np.nan, "coint_t_yx": np.nan, "p_value_yx": np.nan, "is_cointegrated": False, "error": str(exc),
            }
        results.append(row)

    all_results = pd.DataFrame(results)
    if len(all_results) == 0:
        return all_results, all_results
    candidates = all_results[all_results["is_cointegrated"]].copy()
    if len(candidates) == 0:
        return all_results, candidates
    candidates = candidates.sort_values(by=["coint_t", "p_value"], ascending=[True, True]).reset_index(drop=True)
    return all_results, candidates.head(preselect_n).copy()


def select_pairs_with_beta_filter(
    method: StrategyName,
    formation_prices: pd.DataFrame,
    stock_info_df: pd.DataFrame,
    market_price_full: pd.Series,
    settings: AppSettings,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if method == "distance":
        all_raw, preselected = select_pairs_distance_preselect(formation_prices, stock_info_df, settings.preselect_n, settings.min_obs)
        sort_by = {"by": ["distance"], "ascending": [True]}
    else:
        all_raw, preselected = select_pairs_cointegration_preselect(formation_prices, stock_info_df, settings.pvalue_threshold, settings.preselect_n, settings.min_obs)
        sort_by = {"by": ["coint_t", "p_value"], "ascending": [True, True]}

    beta_df = build_beta_table_for_candidate_pairs(preselected, formation_prices, stock_info_df, market_price_full, settings.min_obs)
    final_candidates = apply_beta_and_industry_filter(preselected, beta_df, settings.beta_diff_threshold, settings.k_final, sort_by=sort_by)
    return all_raw, preselected, final_candidates


# ============================================================
# Spread Parameter Helpers
# ============================================================

def fit_distance_spread_params(pair_row: pd.Series | dict, formation_prices: pd.DataFrame) -> dict[str, object]:
    ticker_x = str(pair_row["ticker_x"])
    ticker_y = str(pair_row["ticker_y"])
    pair_prices = formation_prices[[ticker_x, ticker_y]].dropna()
    log_prices = np.log(pair_prices)
    mu_x = float(log_prices[ticker_x].mean())
    std_x = float(log_prices[ticker_x].std(ddof=0))
    mu_y = float(log_prices[ticker_y].mean())
    std_y = float(log_prices[ticker_y].std(ddof=0))
    z_x = (log_prices[ticker_x] - mu_x) / std_x
    z_y = (log_prices[ticker_y] - mu_y) / std_y
    spread = z_x - z_y
    return {"method": "distance", "ticker_x": ticker_x, "ticker_y": ticker_y, "mu_x": mu_x, "std_x": std_x, "mu_y": mu_y, "std_y": std_y, "spread_mean": float(spread.mean()), "spread_std": float(spread.std(ddof=0)), "hedge_ratio": 1.0}


def fit_coint_spread_params(pair_row: pd.Series | dict, formation_prices: pd.DataFrame) -> dict[str, object]:
    ticker_x = str(pair_row["ticker_x"])
    ticker_y = str(pair_row["ticker_y"])
    pair_prices = formation_prices[[ticker_x, ticker_y]].dropna()
    log_prices = np.log(pair_prices)
    x = log_prices[ticker_x]
    y = log_prices[ticker_y]
    model = sm.OLS(x, sm.add_constant(y)).fit()
    alpha = float(model.params["const"])
    beta = float(model.params[ticker_y])
    spread = x - alpha - beta * y
    return {"method": "cointegration", "ticker_x": ticker_x, "ticker_y": ticker_y, "alpha": alpha, "hedge_ratio": beta, "spread_mean": float(spread.mean()), "spread_std": float(spread.std(ddof=0))}


def fit_spread_params(method: StrategyName, pair_row: pd.Series | dict, formation_prices: pd.DataFrame) -> dict[str, object]:
    return fit_distance_spread_params(pair_row, formation_prices) if method == "distance" else fit_coint_spread_params(pair_row, formation_prices)


def compute_z_at_date(params: dict[str, object], close_df: pd.DataFrame, date: pd.Timestamp) -> float:
    ticker_x = str(params["ticker_x"])
    ticker_y = str(params["ticker_y"])
    if ticker_x not in close_df.columns or ticker_y not in close_df.columns or date not in close_df.index:
        return np.nan
    price_x = close_df.loc[date, ticker_x]
    price_y = close_df.loc[date, ticker_y]
    if pd.isna(price_x) or pd.isna(price_y):
        return np.nan
    log_x = np.log(price_x)
    log_y = np.log(price_y)
    method = str(params["method"])
    if method == "distance":
        z_x = (log_x - float(params["mu_x"])) / float(params["std_x"])
        z_y = (log_y - float(params["mu_y"])) / float(params["std_y"])
        spread = z_x - z_y
    else:
        spread = log_x - float(params["alpha"]) - float(params["hedge_ratio"]) * log_y
    std = float(params["spread_std"])
    if std == 0 or np.isnan(std):
        return np.nan
    return float((spread - float(params["spread_mean"])) / std)


def calculate_pair_weights(direction: int, hedge_ratio: float) -> tuple[float, float]:
    raw_x = 1.0
    raw_y = -float(hedge_ratio)
    gross = abs(raw_x) + abs(raw_y)
    return direction * raw_x / gross, direction * raw_y / gross


# ============================================================
# Backtest: Fixed Pair, Weekly Rolling Formation, No Force Close
# ============================================================

def get_week_start_dates(price_df: pd.DataFrame, trading_start: pd.Timestamp, trading_end: pd.Timestamp | None, weekly_freq: str = "W-FRI") -> list[pd.Timestamp]:
    prices = price_df.loc[trading_start:trading_end].copy()
    trading_dates = prices.index
    if len(trading_dates) == 0:
        return []
    week_start_dates = (
        pd.Series(trading_dates, index=trading_dates)
        .groupby(trading_dates.to_period(weekly_freq))
        .first()
        .dropna()
        .tolist()
    )
    return [pd.Timestamp(d) for d in week_start_dates]


def evaluate_equity_curve(equity_curve: pd.Series, trades_df: pd.DataFrame, annualization: int = 252) -> dict[str, float]:
    if equity_curve is None or len(equity_curve) < 2:
        return {"final_equity": np.nan, "total_return": np.nan, "sharpe": np.nan, "max_drawdown": np.nan, "n_trades": 0, "win_rate": np.nan, "avg_trade_pnl": np.nan}
    daily_return = equity_curve.pct_change().dropna()
    sharpe = np.nan
    if len(daily_return) > 0 and daily_return.std(ddof=0) != 0:
        sharpe = float((daily_return.mean() / daily_return.std(ddof=0)) * np.sqrt(annualization))
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1
    total_return = float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1)
    if trades_df is None or len(trades_df) == 0:
        n_trades = 0
        win_rate = np.nan
        avg_trade_pnl = np.nan
    else:
        n_trades = int(len(trades_df))
        win_rate = float((trades_df["pnl"] > 0).mean())
        avg_trade_pnl = float(trades_df["pnl"].mean())
    return {"final_equity": float(equity_curve.iloc[-1]), "total_return": total_return, "sharpe": sharpe, "max_drawdown": float(drawdown.min()), "n_trades": n_trades, "win_rate": win_rate, "avg_trade_pnl": avg_trade_pnl}


def run_single_pair_rolling_backtest(
    method: StrategyName,
    ticker_x: str,
    ticker_y: str,
    open_df: pd.DataFrame,
    close_df: pd.DataFrame,
    stock_info_df: pd.DataFrame,
    market_price_full: pd.Series,
    settings: AppSettings,
) -> dict[str, object]:
    ticker_x = to_yf_ticker(ticker_x)
    ticker_y = to_yf_ticker(ticker_y)
    pair_key = canonical_pair_name(ticker_x, ticker_y)

    all_dates = close_df.loc[settings.trading_start:settings.trading_end].index
    if len(all_dates) < 2:
        raise ValueError("回測期間資料不足。")

    week_start_set = set(get_week_start_dates(close_df, settings.trading_start, settings.trading_end, settings.weekly_freq))

    cash = float(settings.initial_capital)
    active = False
    pending_order: dict[str, object] | None = None
    blocked_after_stop_loss = False

    shares_x = 0.0
    shares_y = 0.0
    entry_capital = np.nan
    entry_date: pd.Timestamp | None = None
    entry_signal_z = np.nan
    entry_price_x = np.nan
    entry_price_y = np.nan
    entry_direction = 0
    holding_days = 0
    active_params: dict[str, object] | None = None

    current_params: dict[str, object] | None = None
    current_eligible = False
    weekly_records: list[dict[str, object]] = []
    trade_records: list[dict[str, object]] = []
    equity_records: list[dict[str, object]] = []
    z_records: list[dict[str, object]] = []
    selected_pair_snapshots: list[pd.DataFrame] = []

    for i, date in enumerate(all_dates):
        date = pd.Timestamp(date)

        # Weekly re-selection. Existing position keeps its original params.
        if date in week_start_set:
            current_params = None
            current_eligible = False
            pos = close_df.index.get_loc(date)
            if pos >= settings.lookback_days:
                formation_prices = close_df.iloc[pos - settings.lookback_days:pos].copy()
                if len(formation_prices) >= settings.min_obs:
                    all_raw, preselected, final_candidates = select_pairs_with_beta_filter(method, formation_prices, stock_info_df, market_price_full, settings)
                    if len(final_candidates) > 0:
                        snapshot = final_candidates.copy()
                        snapshot["rebalance_date"] = date
                        snapshot["formation_start"] = formation_prices.index.min()
                        snapshot["formation_end"] = formation_prices.index.max()
                        selected_pair_snapshots.append(snapshot)

                        for _, row in final_candidates.iterrows():
                            if canonical_pair_name(row["ticker_x"], row["ticker_y"]) == pair_key:
                                current_eligible = True
                                row_for_params = dict(row)
                                # Use user's input direction for spread definition.
                                row_for_params["ticker_x"] = ticker_x
                                row_for_params["ticker_y"] = ticker_y
                                current_params = fit_spread_params(method, row_for_params, formation_prices)
                                break

                    weekly_records.append({
                        "method": method,
                        "rebalance_date": date,
                        "formation_start": formation_prices.index.min(),
                        "formation_end": formation_prices.index.max(),
                        "n_candidates": len(final_candidates),
                        "pair_eligible": current_eligible,
                        "active_position": active,
                    })

        # Execute pending orders at today's open.
        if pending_order is not None:
            action = str(pending_order["action"])
            open_x = open_df.loc[date, ticker_x]
            open_y = open_df.loc[date, ticker_y]
            close_x = close_df.loc[date, ticker_x]
            close_y = close_df.loc[date, ticker_y]
            if not (pd.isna(open_x) or pd.isna(open_y)):
                if action == "enter":
                    params = pending_order["spread_params"]  # type: ignore[assignment]
                    direction = int(pending_order["direction"])
                    weight_x, weight_y = calculate_pair_weights(direction, float(params["hedge_ratio"]))  # type: ignore[index]
                    dollar_x = cash * weight_x
                    dollar_y = cash * weight_y
                    shares_x = dollar_x / open_x
                    shares_y = dollar_y / open_y
                    traded_notional = abs(shares_x * open_x) + abs(shares_y * open_y)
                    cost = traded_notional * settings.fee_rate
                    entry_capital = cash
                    cash -= shares_x * open_x
                    cash -= shares_y * open_y
                    cash -= cost
                    active = True
                    active_params = params  # type: ignore[assignment]
                    entry_date = date
                    entry_signal_z = float(pending_order["signal_z"])
                    entry_price_x = float(open_x)
                    entry_price_y = float(open_y)
                    entry_direction = direction
                    holding_days = 0
                elif action == "exit" and active:
                    traded_notional = abs(shares_x * open_x) + abs(shares_y * open_y)
                    cost = traded_notional * settings.fee_rate
                    cash += shares_x * open_x
                    cash += shares_y * open_y
                    cash -= cost
                    pnl = cash - entry_capital
                    trade_records.append({
                        "method": method,
                        "pair": display_pair_name(ticker_x, ticker_y),
                        "ticker_x": ticker_x,
                        "ticker_y": ticker_y,
                        "entry_date": entry_date,
                        "exit_date": date,
                        "direction": entry_direction,
                        "entry_z": entry_signal_z,
                        "exit_signal_z": pending_order.get("exit_signal_z", np.nan),
                        "entry_price_x": entry_price_x,
                        "entry_price_y": entry_price_y,
                        "exit_price_x": float(open_x),
                        "exit_price_y": float(open_y),
                        "holding_days": holding_days,
                        "exit_reason": pending_order.get("exit_reason", "exit"),
                        "pnl": float(pnl),
                        "return": float(pnl / entry_capital) if entry_capital else np.nan,
                    })
                    if pending_order.get("exit_reason") == "stop_loss":
                        blocked_after_stop_loss = True
                    active = False
                    shares_x = shares_y = 0.0
                    active_params = None
                    entry_date = None
                    entry_signal_z = np.nan
                    entry_direction = 0
                    holding_days = 0
            pending_order = None

        # Mark to market and z-score record at today's close.
        close_x = close_df.loc[date, ticker_x]
        close_y = close_df.loc[date, ticker_y]
        equity = cash + shares_x * close_x + shares_y * close_y if active and not (pd.isna(close_x) or pd.isna(close_y)) else cash
        z_active = compute_z_at_date(active_params, close_df, date) if active and active_params is not None else np.nan
        z_current = compute_z_at_date(current_params, close_df, date) if current_params is not None else np.nan
        equity_records.append({"date": date, "equity": float(equity), "active": active, "cash": float(cash)})
        z_records.append({"date": date, "z_current_candidate": z_current, "z_active_position": z_active, "pair_eligible": current_eligible, "blocked_after_stop_loss": blocked_after_stop_loss})

        if i >= len(all_dates) - 1:
            continue

        # Exit signal for active position, using entry-time params.
        if active and pending_order is None and active_params is not None:
            z_today = compute_z_at_date(active_params, close_df, date)
            if not pd.isna(z_today):
                holding_days += 1
                exit_reason = None
                if abs(z_today) < settings.exit_z:
                    exit_reason = "mean_reversion"
                elif abs(z_today) > settings.stop_z:
                    exit_reason = "stop_loss"
                elif holding_days >= settings.max_holding_days:
                    exit_reason = "max_holding_days"
                if exit_reason is not None:
                    pending_order = {"action": "exit", "exit_signal_z": z_today, "exit_reason": exit_reason}
            continue

        # Entry signal only when current weekly selection says this pair is eligible.
        if (not active) and pending_order is None and current_params is not None and current_eligible:
            z_today = compute_z_at_date(current_params, close_df, date)
            prev_date = all_dates[i - 1] if i > 0 else None
            z_prev = compute_z_at_date(current_params, close_df, prev_date) if prev_date is not None else np.nan
            if pd.isna(z_today) or pd.isna(z_prev):
                continue

            if blocked_after_stop_loss:
                if abs(z_today) < settings.reentry_reset_z:
                    blocked_after_stop_loss = False
                continue

            direction = None
            if (z_prev <= settings.entry_z) and (settings.entry_z < z_today < settings.stop_z):
                direction = -1
            elif (z_prev >= -settings.entry_z) and (-settings.stop_z < z_today < -settings.entry_z):
                direction = 1

            if direction is not None:
                pending_order = {"action": "enter", "spread_params": current_params, "direction": direction, "signal_z": z_today}

    # Final liquidation at backtest end.
    final_date = pd.Timestamp(all_dates[-1])
    if active and active_params is not None:
        close_x = close_df.loc[final_date, ticker_x]
        close_y = close_df.loc[final_date, ticker_y]
        traded_notional = abs(shares_x * close_x) + abs(shares_y * close_y)
        cost = traded_notional * settings.fee_rate
        cash += shares_x * close_x + shares_y * close_y - cost
        pnl = cash - entry_capital
        z_final = compute_z_at_date(active_params, close_df, final_date)
        trade_records.append({
            "method": method, "pair": display_pair_name(ticker_x, ticker_y), "ticker_x": ticker_x, "ticker_y": ticker_y,
            "entry_date": entry_date, "exit_date": final_date, "direction": entry_direction,
            "entry_z": entry_signal_z, "exit_signal_z": z_final,
            "entry_price_x": entry_price_x, "entry_price_y": entry_price_y,
            "exit_price_x": float(close_x), "exit_price_y": float(close_y), "holding_days": holding_days,
            "exit_reason": "backtest_end", "pnl": float(pnl), "return": float(pnl / entry_capital) if entry_capital else np.nan,
        })
        equity_records.append({"date": final_date, "equity": float(cash), "active": False, "cash": float(cash)})

    equity_df = pd.DataFrame(equity_records)
    equity_curve = equity_df.drop_duplicates("date", keep="last").set_index("date")["equity"] if len(equity_df) else pd.Series(dtype=float)
    trades_df = pd.DataFrame(trade_records)
    z_df = pd.DataFrame(z_records).drop_duplicates("date", keep="last").set_index("date") if len(z_records) else pd.DataFrame()
    weekly_df = pd.DataFrame(weekly_records)
    selected_pairs_history = pd.concat(selected_pair_snapshots, axis=0).reset_index(drop=True) if selected_pair_snapshots else pd.DataFrame()
    metrics = evaluate_equity_curve(equity_curve, trades_df)
    summary = pd.DataFrame([{**metrics, "method": method, "ticker_x": ticker_x, "ticker_y": ticker_y, "initial_capital": settings.initial_capital}])
    return {"summary": summary, "equity_curve": equity_curve, "trades": trades_df, "zscore": z_df, "weekly_summary": weekly_df, "selected_pairs_history": selected_pairs_history}


# ============================================================
# Charts and Output Helpers
# ============================================================

def plot_price_chart(close_df: pd.DataFrame, ticker_x: str, ticker_y: str) -> go.Figure:
    fig = go.Figure()
    pair = close_df[[ticker_x, ticker_y]].dropna()
    for ticker in [ticker_x, ticker_y]:
        fig.add_trace(go.Scatter(x=pair.index, y=pair[ticker], mode="lines", name=f"{ticker} Close"))
    fig.update_layout(title="Historical Close Price", template="plotly_white", height=460, hovermode="x unified")
    return fig


def plot_normalized_price_chart(close_df: pd.DataFrame, ticker_x: str, ticker_y: str) -> go.Figure:
    fig = go.Figure()
    pair = close_df[[ticker_x, ticker_y]].dropna()
    norm = pair / pair.iloc[0]
    for ticker in [ticker_x, ticker_y]:
        fig.add_trace(go.Scatter(x=norm.index, y=norm[ticker], mode="lines", name=f"{ticker} normalized"))
    fig.update_layout(title="Normalized Close Price", template="plotly_white", height=420, hovermode="x unified", yaxis_tickformat=".2f")
    return fig


def plot_equity_curve(equity_curve: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity_curve.index, y=equity_curve.values, mode="lines", name="Equity"))
    fig.update_layout(title="Equity Curve", template="plotly_white", height=460, hovermode="x unified")
    return fig


def plot_drawdown(equity_curve: pd.Series) -> go.Figure:
    running_max = equity_curve.cummax()
    dd = equity_curve / running_max - 1
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dd.index, y=dd.values, fill="tozeroy", mode="lines", name="Drawdown"))
    fig.update_layout(title="Drawdown", template="plotly_white", height=350, yaxis_tickformat=".1%", hovermode="x unified")
    return fig


def plot_zscore(z_df: pd.DataFrame, settings: AppSettings, trades_df: pd.DataFrame | None = None) -> go.Figure:
    fig = go.Figure()
    if z_df is not None and len(z_df) > 0:
        if "z_current_candidate" in z_df.columns:
            fig.add_trace(go.Scatter(x=z_df.index, y=z_df["z_current_candidate"], mode="lines", name="Current weekly z"))
        if "z_active_position" in z_df.columns:
            fig.add_trace(go.Scatter(x=z_df.index, y=z_df["z_active_position"], mode="lines", name="Active position z"))
    for y, name in [(settings.entry_z, "+Entry"), (-settings.entry_z, "-Entry"), (settings.exit_z, "+Exit"), (-settings.exit_z, "-Exit"), (settings.stop_z, "+Stop"), (-settings.stop_z, "-Stop")]:
        fig.add_hline(y=y, line_dash="dash", annotation_text=name, annotation_position="top left")
    if trades_df is not None and len(trades_df) > 0:
        fig.add_trace(go.Scatter(x=trades_df["entry_date"], y=trades_df["entry_z"], mode="markers", name="Entry", marker=dict(size=9, symbol="triangle-up")))
        fig.add_trace(go.Scatter(x=trades_df["exit_date"], y=trades_df["exit_signal_z"], mode="markers", name="Exit", marker=dict(size=9, symbol="x")))
    fig.update_layout(title="Spread Z-score Signal", template="plotly_white", height=430, hovermode="x unified")
    return fig


def plot_trade_pnl(trades_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if trades_df is not None and len(trades_df) > 0:
        labels = [f"{pd.Timestamp(d).date()}" for d in trades_df["exit_date"]]
        fig.add_trace(go.Bar(x=labels, y=trades_df["pnl"], name="Trade P/L"))
    fig.update_layout(title="Trade P/L", template="plotly_white", height=330, hovermode="x unified")
    return fig


def show_summary_metrics(summary_df: pd.DataFrame) -> None:
    if summary_df is None or summary_df.empty:
        st.warning("沒有績效摘要可顯示。")
        return
    row = summary_df.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Return", f"{float(row['total_return']):.2%}" if pd.notna(row["total_return"]) else "NA")
    c2.metric("Sharpe", f"{float(row['sharpe']):.2f}" if pd.notna(row["sharpe"]) else "NA")
    c3.metric("Max Drawdown", f"{float(row['max_drawdown']):.2%}" if pd.notna(row["max_drawdown"]) else "NA")
    c4.metric("Trades", f"{int(row['n_trades'])}" if pd.notna(row["n_trades"]) else "0")
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Win Rate", f"{float(row['win_rate']):.2%}" if pd.notna(row["win_rate"]) else "NA")
    c6.metric("Avg Trade P/L", f"{float(row['avg_trade_pnl']):,.0f}" if pd.notna(row["avg_trade_pnl"]) else "NA")
    c7.metric("Final Equity", f"{float(row['final_equity']):,.0f}" if pd.notna(row["final_equity"]) else "NA")
    c8.metric("Initial Capital", f"{float(row['initial_capital']):,.0f}" if "initial_capital" in row and pd.notna(row["initial_capital"]) else "NA")


def df_to_excel_bytes(sheets: dict[str, pd.DataFrame | pd.Series]) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for name, obj in sheets.items():
            safe_name = str(name)[:31]
            if isinstance(obj, pd.Series):
                obj.to_frame(obj.name or "value").to_excel(writer, sheet_name=safe_name)
            elif isinstance(obj, pd.DataFrame):
                obj.to_excel(writer, sheet_name=safe_name, index=True)
    output.seek(0)
    return output.getvalue()


# ============================================================
# Streamlit Rendering
# ============================================================

def render_strategy_selector() -> None:
    st.title("Trading Strategy Lab")
    st.caption("請選擇要使用的交易策略。")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="strategy-card">
              <h3>配對策略1(距離法)</h3>
              <p>
                依細產業股票池，使用標準化 log price 計算 SSD 距離，
                篩選距離最小的 pair，再用 rolling formation 與 z-score 進行回測。
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("進入 配對策略1(距離法)", type="primary", use_container_width=True):
            st.session_state["selected_strategy"] = "distance"
            st.rerun()

    with col2:
        st.markdown(
            """
            <div class="strategy-card">
              <h3>配對策略2(共整合法)</h3>
              <p>
                依細產業股票池，使用 Engle-Granger 共整合檢定，
                篩選 p-value 通過且 coint_t 較負的 pair，再執行 rolling 回測。
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("進入 配對策略2(共整合法)", use_container_width=True):
            st.session_state["selected_strategy"] = "cointegration"
            st.rerun()

    with col3:
        st.markdown(
            """
            <div class="strategy-card">
              <h3>策略3(台股五因子 Alpha 月調倉策略)</h3>
              <p>
                使用已建構好的台股五因子 rolling regression alpha，
                每月從市值前 150 股票池中選出 Alpha Top N，
                以 t+1 收盤價調倉並與 0050 benchmark 比較。
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("進入 策略3(五因子 Alpha)", use_container_width=True):
            st.session_state["selected_strategy"] = "ff5_alpha"
            st.rerun()

    st.info(
        "策略1/2：先在「選股 / 選 Pair」產生候選清單，再進行單一 pair rolling 回測。"
        "策略3：使用 Colab 產出的 alpha_scores 與 AD PRICE CSV/XLSX，直接產生 Top N 持股與回測。"
    )

def sidebar_settings(strategy: StrategyName) -> AppSettings:
    if st.sidebar.button("返回策略選擇"):
        st.session_state["selected_strategy"] = None
        st.rerun()

    st.sidebar.caption(f"目前策略：{'配對策略1(距離法)' if strategy == 'distance' else '配對策略2(共整合法)'}")
    st.sidebar.divider()

    industries = get_industry_options()
    default_ind = "金融保險業" if "金融保險業" in industries else industries[0]
    industry = st.sidebar.selectbox("細產業股票池", industries, index=industries.index(default_ind), help="可選全部，但全部股票池執行共整合篩選會較慢。")

    st.sidebar.header("資料與回測期間")
    data_start = st.sidebar.date_input("資料下載起日", value=pd.Timestamp("2022-01-01").date())
    trading_start = st.sidebar.date_input("正式回測起日", value=pd.Timestamp("2024-01-01").date())
    use_today = st.sidebar.checkbox("回測到最新資料", value=True)
    if use_today:
        trading_end = None
        st.sidebar.caption("回測結束日：yfinance 最新可取得資料")
    else:
        trading_end = pd.Timestamp(st.sidebar.date_input("正式回測結束日", value=pd.Timestamp.today().date()))
    market_ticker = st.sidebar.text_input("Beta benchmark", value="^TWII")
    auto_adjust = st.sidebar.toggle("使用 yfinance auto_adjust", value=True)

    st.sidebar.divider()
    st.sidebar.header("選股 / Pair 篩選")
    lookback_days = st.sidebar.number_input("Rolling formation lookback days", min_value=60, max_value=1000, value=252, step=10)
    min_obs = st.sidebar.number_input("Formation 最少共同資料筆數", min_value=30, max_value=900, value=200, step=10)
    k_final = st.sidebar.slider("每週候選 pair 數量 final_k", 1, 30, 10, 1)
    preselect_multiplier = st.sidebar.slider("初選倍數 multiplier", 1, 30, 10, 1)
    beta_diff_threshold = st.sidebar.number_input("Beta 差距門檻", min_value=0.0, value=0.2, step=0.05, format="%.4f")
    pvalue_threshold = 0.05
    if strategy == "cointegration":
        pvalue_threshold = st.sidebar.selectbox("共整合 p-value 門檻", [0.01, 0.05, 0.10], index=1)
    top_n_display = st.sidebar.slider("候選 pair 表格顯示筆數", 1, 50, 10, 1)

    st.sidebar.divider()
    st.sidebar.header("交易規則")
    entry_z = st.sidebar.slider("Entry z-score", 0.5, 4.0, 2.0, 0.1)
    exit_z = st.sidebar.slider("Exit z-score", 0.0, 2.0, 0.5, 0.1)
    stop_z = st.sidebar.slider("Stop z-score", 1.0, 6.0, 3.0, 0.1)
    max_holding_days = st.sidebar.number_input("Max holding days", min_value=1, max_value=1000, value=60, step=5)
    reentry_reset_z = st.sidebar.slider("Stop loss 後重新允許進場 abs(z) <", 0.0, 2.0, 1.0, 0.1)

    st.sidebar.divider()
    st.sidebar.header("成本與資金")
    fee_rate = st.sidebar.number_input("買賣手續費率", min_value=0.0, value=0.001425, step=0.0001, format="%.6f")
    initial_capital = st.sidebar.number_input("單一 pair 初始資金", min_value=10_000, value=1_000_000, step=100_000)

    return AppSettings(
        strategy=strategy,
        industry=industry,
        data_start=pd.Timestamp(data_start),
        trading_start=pd.Timestamp(trading_start),
        trading_end=trading_end,
        market_ticker=market_ticker,
        lookback_days=int(lookback_days),
        weekly_freq="W-FRI",
        k_final=int(k_final),
        preselect_multiplier=int(preselect_multiplier),
        min_obs=int(min_obs),
        pvalue_threshold=float(pvalue_threshold),
        beta_diff_threshold=float(beta_diff_threshold),
        entry_z=float(entry_z),
        exit_z=float(exit_z),
        stop_z=float(stop_z),
        max_holding_days=int(max_holding_days),
        reentry_reset_z=float(reentry_reset_z),
        fee_rate=float(fee_rate),
        initial_capital=float(initial_capital),
        auto_adjust=bool(auto_adjust),
        top_n_display=int(top_n_display),
    )


def render_pair_selection_tab(settings: AppSettings) -> None:
    st.subheader("① 選股 / 選 Pair")
    universe = get_universe_by_industry(settings.industry)
    st.write(f"目前股票池：**{settings.industry}**，共 **{len(universe)}** 檔。")
    st.dataframe(universe[["code", "name", "industry", "ticker_yf"]], use_container_width=True, hide_index=True)

    ref_date = st.date_input("選 Pair 參考日期：使用此日期前的 rolling formation 選 pair", value=settings.trading_start.date())
    run_screen = st.button("執行選股 / 選 Pair", type="primary", use_container_width=True)

    key = f"screen_{settings.strategy}"
    if run_screen:
        try:
            open_df, close_df, market_price, universe = prepare_download_for_universe(settings)
            ref_ts = pd.Timestamp(ref_date)
            all_dates = close_df.index
            if ref_ts not in all_dates:
                earlier = all_dates[all_dates <= ref_ts]
                if len(earlier) == 0:
                    st.error("參考日期之前沒有可用價格資料。")
                    return
                ref_ts = pd.Timestamp(earlier[-1])
            pos = all_dates.get_loc(ref_ts)
            if pos < settings.lookback_days:
                st.error("參考日期前資料不足，請提前資料下載起日或縮短 lookback days。")
                return
            formation_prices = close_df.iloc[pos - settings.lookback_days:pos].copy()
            with st.spinner("計算候選 pair..."):
                all_raw, preselected, final_candidates = select_pairs_with_beta_filter(settings.strategy, formation_prices, universe, market_price, settings)
            st.session_state[key] = {
                "ref_date": ref_ts,
                "formation_start": formation_prices.index.min(),
                "formation_end": formation_prices.index.max(),
                "all_raw": all_raw,
                "preselected": preselected,
                "final_candidates": final_candidates,
            }
        except Exception as exc:
            st.error(f"選 pair 失敗：{exc}")
            return

    result = st.session_state.get(key)
    if not result:
        st.info("按下「執行選股 / 選 Pair」後，這裡會顯示候選 pair 清單。")
        return

    final_candidates = result["final_candidates"]
    st.caption(f"Formation period：{pd.Timestamp(result['formation_start']).date()} ～ {pd.Timestamp(result['formation_end']).date()}；參考日期：{pd.Timestamp(result['ref_date']).date()}")

    if final_candidates is None or len(final_candidates) == 0:
        st.warning("目前條件下沒有通過 beta 差距與同產業篩選的候選 pair。")
        return

    show = final_candidates.copy().head(settings.top_n_display)
    metric_cols = ["distance"] if settings.strategy == "distance" else ["coint_t", "p_value", "direction"]
    display_cols = ["code_x", "name_x", "code_y", "name_y", "industry_x", "industry_y", "n_obs", "beta_x", "beta_y", "beta_diff"] + metric_cols
    display_cols = [c for c in display_cols if c in show.columns]
    for col in ["distance", "coint_t", "p_value", "beta_x", "beta_y", "beta_diff"]:
        if col in show.columns:
            show[col] = pd.to_numeric(show[col], errors="coerce").round(6)
    st.dataframe(show[display_cols], use_container_width=True, hide_index=True)

    labels = []
    for idx, row in final_candidates.iterrows():
        score = f"distance={row['distance']:.3f}" if settings.strategy == "distance" else f"p={row['p_value']:.4f}, t={row['coint_t']:.3f}"
        labels.append(f"{idx+1}. {row['code_x']} {row['name_x']} / {row['code_y']} {row['name_y']} | {score} | beta_diff={row['beta_diff']:.3f}")

    selected_label = st.selectbox("套用候選 pair 到回測", labels)
    selected_idx = labels.index(selected_label)
    selected = final_candidates.iloc[selected_idx]
    if st.button("套用選取 pair 到回測輸入框", use_container_width=True):
        st.session_state[f"{settings.strategy}_code_x"] = str(selected["code_x"])
        st.session_state[f"{settings.strategy}_code_y"] = str(selected["code_y"])
        st.success(f"已套用：{selected['code_x']} / {selected['code_y']}。請切到「單一 Pair 回測」執行。")

    excel_bytes = df_to_excel_bytes({"final_candidates": final_candidates, "preselected": result["preselected"], "all_raw": result["all_raw"]})
    st.download_button("下載候選 pair Excel", data=excel_bytes, file_name=f"{settings.strategy}_pair_candidates.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


def render_backtest_tab(settings: AppSettings) -> None:
    st.subheader("② 單一 Pair Rolling 回測")
    st.write("此回測會每週重新選 pair；只有當輸入的 pair 在該週候選清單中時，才允許用空閒資金開新倉。舊部位不因週度 re-selection 強制平倉，會繼續用進場時的 spread 參數管理。")

    col_a, col_b, col_c = st.columns([0.25, 0.25, 0.50])
    with col_a:
        code_x = st.text_input("股票 X 代號", key=f"{settings.strategy}_code_x", placeholder="例如 2892").strip().upper()
    with col_b:
        code_y = st.text_input("股票 Y 代號", key=f"{settings.strategy}_code_y", placeholder="例如 5880").strip().upper()
    with col_c:
        if code_x or code_y:
            st.caption(f"辨識結果：{format_stock_label(code_x)} / {format_stock_label(code_y)}")
            st.caption(f"產業：{resolve_stock_industry(code_x) or 'NA'} / {resolve_stock_industry(code_y) or 'NA'}")

    run = st.button("Run rolling backtest", type="primary", use_container_width=True)
    if not run:
        st.info("請輸入股票代號，或先到「選股 / 選 Pair」套用候選 pair。")
        return
    if not code_x or not code_y:
        st.error("請輸入兩檔股票代號。")
        return
    if code_x == code_y:
        st.error("兩檔股票代號不能相同。")
        return

    ticker_x = to_yf_ticker(code_x)
    ticker_y = to_yf_ticker(code_y)

    try:
        open_df, close_df, market_price, universe = prepare_download_for_universe(settings, extra_tickers=[ticker_x, ticker_y])
        if ticker_x not in close_df.columns or ticker_y not in close_df.columns:
            st.error("下載資料中找不到輸入股票。請確認 yfinance 是否支援該代號。")
            return
        with st.spinner("執行 rolling formation no-force-close 回測..."):
            result = run_single_pair_rolling_backtest(settings.strategy, ticker_x, ticker_y, open_df, close_df, universe, market_price, settings)
    except Exception as exc:
        st.error(f"回測失敗：{exc}")
        return

    st.markdown(f"### 回測結果：{format_stock_label(code_x)} / {format_stock_label(code_y)}")
    show_summary_metrics(result["summary"])

    equity_curve = result["equity_curve"]
    trades_df = result["trades"]
    z_df = result["zscore"]
    trading_close = close_df.loc[settings.trading_start:settings.trading_end]

    chart_tab1, chart_tab2, chart_tab3, table_tab = st.tabs(["績效圖表", "價格與訊號", "交易損益", "資料表"])
    with chart_tab1:
        if len(equity_curve) > 0:
            st.plotly_chart(plot_equity_curve(equity_curve), use_container_width=True)
            st.plotly_chart(plot_drawdown(equity_curve), use_container_width=True)
        else:
            st.warning("沒有 equity curve 可顯示。")
    with chart_tab2:
        st.plotly_chart(plot_price_chart(trading_close, ticker_x, ticker_y), use_container_width=True)
        st.plotly_chart(plot_normalized_price_chart(trading_close, ticker_x, ticker_y), use_container_width=True)
        st.plotly_chart(plot_zscore(z_df, settings, trades_df), use_container_width=True)
    with chart_tab3:
        st.plotly_chart(plot_trade_pnl(trades_df), use_container_width=True)
    with table_tab:
        st.markdown("#### 交易紀錄")
        st.dataframe(trades_df, use_container_width=True, hide_index=True)
        st.markdown("#### 每週選股狀態")
        st.dataframe(result["weekly_summary"], use_container_width=True, hide_index=True)
        st.markdown("#### 歷史候選 pair")
        st.dataframe(result["selected_pairs_history"], use_container_width=True, hide_index=True)
        st.markdown("#### Z-score / eligible 狀態")
        st.dataframe(z_df, use_container_width=True)

    excel_bytes = df_to_excel_bytes({
        "summary": result["summary"],
        "equity_curve": equity_curve,
        "trades": trades_df,
        "weekly_summary": result["weekly_summary"],
        "selected_pairs": result["selected_pairs_history"],
        "zscore": z_df,
    })
    st.download_button("下載完整回測結果 Excel", data=excel_bytes, file_name=f"{settings.strategy}_{code_x}_{code_y}_rolling_backtest.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


def render_method_notes(settings: AppSettings) -> None:
    st.subheader("③ 方法說明")
    if settings.strategy == "distance":
        st.markdown(
            """
            ### 配對策略1(距離法)
            1. 每週用過去 `lookback_days` 個交易日作為 formation period。
            2. 對每檔股票取 log price 後標準化。
            3. 對所有兩兩 pair 計算 `SSD = sum((Z_X - Z_Y)^2)`。
            4. 先取 SSD 最小的初選 pair，再做 `beta_diff < threshold` 與同產業篩選。
            5. 單一 pair 回測時，只有當該 pair 在當週候選清單中，才允許開新倉。
            """
        )
    else:
        st.markdown(
            """
            ### 配對策略2(共整合法)
            1. 每週用過去 `lookback_days` 個交易日作為 formation period。
            2. 對所有兩兩 pair 的 log price 做 Engle-Granger 共整合檢定。
            3. 保留 `p_value <= threshold`，再依 `coint_t` 最負排序。
            4. 對初選 pair 做 `beta_diff < threshold` 與同產業篩選。
            5. 單一 pair 回測時，spread 使用進場當下 formation period 估出的 OLS `alpha` 與 `hedge_ratio`。
            """
        )
    st.markdown(
        """
        ### 共同交易邏輯
        - 今日收盤產生訊號，明日開盤成交。
        - 不在 stop zone 進場：只允許 `entry_z < abs(z) < stop_z`。
        - 只在 crossing signal 進場，避免 z-score 長期在極端區間時反覆追單。
        - stop loss 後必須等 `abs(z) < reentry_reset_z` 才允許重新進場。
        - 不在週末強制平倉；舊部位繼續用進場時的 spread 參數管理。
        - 僅在 `exit_z`、`stop_z`、`max_holding_days` 或回測結束時出場。
        """
    )


def render_strategy_page(strategy: StrategyName) -> None:
    title = "配對策略1(距離法)" if strategy == "distance" else "配對策略2(共整合法)"
    st.title(title)
    st.caption("細產業選股 / 候選 pair 清單 / 手動輸入 pair rolling 回測")
    settings = sidebar_settings(strategy)

    tabs = st.tabs(["① 選股 / 選 Pair", "② 單一 Pair 回測", "③ 方法說明"])
    with tabs[0]:
        render_pair_selection_tab(settings)
    with tabs[1]:
        render_backtest_tab(settings)
    with tabs[2]:
        render_method_notes(settings)



# ============================================================
# Strategy 3: Taiwan FF5 Alpha Top N Monthly Strategy
# ============================================================

@dataclass(frozen=True)
class FF5AlphaSettings:
    alpha_path: str
    price_path: str
    top_n: int
    initial_capital: float
    commission_rate: float
    sell_tax_rate: float
    benchmark_ticker: str
    benchmark_auto_adjust: bool


def strategy3_sidebar_settings() -> FF5AlphaSettings:
    if st.sidebar.button("返回策略選擇"):
        st.session_state["selected_strategy"] = None
        st.rerun()

    st.sidebar.caption("目前策略：策略3(台股五因子 Alpha 月調倉策略)")
    st.sidebar.divider()

    st.sidebar.header("資料檔案")
    st.sidebar.caption("建議把資料放在 GitHub repo 的 data/strategy3/ 資料夾。支援 CSV、XLSX，AD PRICE 可填單一檔案或資料夾。")
    alpha_path = st.sidebar.text_input(
        "Alpha scores CSV 路徑",
        value="data/strategy3/alpha_scores_ff5_top150.csv",
        help="需要包含 formation_date、stock_id、stock_name、alpha。"
    )
    price_path = st.sidebar.text_input(
        "AD PRICE CSV 或資料夾路徑",
        value="data/strategy3/ad_price",
        help="可填單一 CSV/XLSX，或填資料夾路徑。欄位可用英文 date/stock_id/stock_name/adj_close，或 CMoney 中文欄位：日期、股票代號、股票名稱、收盤價。"
    )

    st.sidebar.divider()
    st.sidebar.header("策略參數")
    top_n = st.sidebar.slider("Alpha Top N 持股檔數", min_value=1, max_value=50, value=10, step=1)
    initial_capital = st.sidebar.number_input("初始資金", min_value=10_000, value=1_000_000, step=100_000)
    commission_rate = st.sidebar.number_input("買賣手續費率", min_value=0.0, value=0.001425, step=0.0001, format="%.6f")
    sell_tax_rate = st.sidebar.number_input("賣出交易稅率", min_value=0.0, value=0.0, step=0.0005, format="%.6f")
    st.sidebar.caption("目前執行價固定為：t+1 收盤價。")

    st.sidebar.divider()
    st.sidebar.header("Benchmark")
    benchmark_ticker = st.sidebar.text_input("Benchmark ticker", value="0050.TW")
    benchmark_auto_adjust = st.sidebar.toggle("Benchmark 使用 yfinance auto_adjust", value=True)

    return FF5AlphaSettings(
        alpha_path=alpha_path,
        price_path=price_path,
        top_n=int(top_n),
        initial_capital=float(initial_capital),
        commission_rate=float(commission_rate),
        sell_tax_rate=float(sell_tax_rate),
        benchmark_ticker=str(benchmark_ticker).strip(),
        benchmark_auto_adjust=bool(benchmark_auto_adjust),
    )



def strategy3_force_datetime_ns(s: pd.Series) -> pd.Series:
    """Normalize dates and force pandas datetime64[ns] for merge_asof compatibility."""
    out = pd.to_datetime(s, errors="coerce")
    try:
        if getattr(out.dt, "tz", None) is not None:
            out = out.dt.tz_localize(None)
    except Exception:
        pass
    return out.dt.normalize().astype("datetime64[ns]")

def parse_strategy3_date_series(s: pd.Series) -> pd.Series:
    """
    支援三種日期格式：
    1. 一般日期字串 / pandas datetime
    2. Excel serial number，例如 45123
    3. YYYYMMDD 數字或字串，例如 20240131
    """
    raw = s.copy()
    raw_str = raw.astype(str).str.strip()

    is_yyyymmdd = raw_str.str.fullmatch(r"\d{8}")
    out = pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns]")

    if is_yyyymmdd.any():
        out.loc[is_yyyymmdd] = pd.to_datetime(
            raw_str.loc[is_yyyymmdd],
            format="%Y%m%d",
            errors="coerce",
        )

    remain = ~is_yyyymmdd
    out.loc[remain] = pd.to_datetime(raw.loc[remain], errors="coerce")

    num = pd.to_numeric(raw, errors="coerce")
    is_excel_serial = (
        num.between(20000, 60000)
        & raw_str.str.fullmatch(r"\d+(\.0)?")
    )

    if is_excel_serial.any():
        out.loc[is_excel_serial] = pd.to_datetime(
            num.loc[is_excel_serial],
            unit="D",
            origin="1899-12-30",
            errors="coerce",
        )

    return strategy3_force_datetime_ns(out)


def strategy3_to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("--", "", regex=False)
        .str.replace("- -", "", regex=False)
        .str.strip(),
        errors="coerce",
    )


def resolve_strategy3_path(path_str: str) -> Path:
    """
    讓策略3資料路徑同時支援：
    - data/strategy3/...
    - trading-strategy-lab/data/strategy3/...

    會優先嘗試以 streamlit_app.py 所在資料夾為基準，
    也會嘗試目前 working directory 與 repo root 常見結構。
    """
    raw = str(path_str).strip().strip('"').strip("'")
    if not raw:
        raise FileNotFoundError("資料路徑是空的。")

    p = Path(raw)
    if p.is_absolute():
        if p.exists():
            return p
        raise FileNotFoundError(f"找不到檔案或資料夾：{p}")

    app_dir = Path(__file__).resolve().parent
    cwd = Path.cwd().resolve()

    candidates = [
        app_dir / p,
        cwd / p,
        cwd / "trading-strategy-lab" / p,
        app_dir.parent / p,
    ]

    seen = set()
    unique_candidates = []
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate not in seen:
            unique_candidates.append(candidate)
            seen.add(candidate)

    for candidate in unique_candidates:
        if candidate.exists():
            return candidate

    checked = "\n".join(str(c) for c in unique_candidates)
    raise FileNotFoundError(
        f"找不到檔案或資料夾：{path_str}\n"
        f"已嘗試以下位置：\n{checked}"
    )


@st.cache_data(show_spinner=False)
def strategy3_read_csv_file(path_str: str) -> pd.DataFrame:
    """
    讀取策略3資料，支援：
    - 單一 CSV
    - 單一 XLSX / XLS
    - 資料夾：自動合併其中所有 CSV / XLSX / XLS

    注意：資料夾內會自動忽略 .gitkeep、暫存檔與非資料檔。
    """
    path = resolve_strategy3_path(path_str)

    def _read_csv(p: Path) -> pd.DataFrame:
        last_error = None
        for enc in ["utf-8-sig", "utf-8", "big5", "cp950"]:
            try:
                return pd.read_csv(p, encoding=enc)
            except Exception as exc:
                last_error = exc
        raise last_error

    def _read_one(p: Path) -> pd.DataFrame:
        suffix = p.suffix.lower()
        if suffix == ".csv":
            return _read_csv(p)
        if suffix in [".xlsx", ".xls"]:
            return pd.read_excel(p)
        raise ValueError(f"不支援的檔案格式：{p.name}")

    if path.is_dir():
        files: list[Path] = []
        for pattern in ["*.csv", "*.xlsx", "*.xls"]:
            files.extend(sorted(path.glob(pattern)))

        files = [f for f in files if not f.name.startswith("~$")]

        if len(files) == 0:
            raise FileNotFoundError(f"資料夾中沒有 CSV/XLSX/XLS 檔案：{path}")

        dfs = []
        for f in files:
            df = _read_one(f)
            df["source_file"] = f.name
            dfs.append(df)

        return pd.concat(dfs, ignore_index=True)

    return _read_one(path)


def strategy3_normalize_alpha(alpha_raw: pd.DataFrame) -> pd.DataFrame:
    df = alpha_raw.copy()
    df.columns = [str(c).strip() for c in df.columns]

    rename_map = {
        "形成日": "formation_date",
        "調倉日": "formation_date",
        "股票代號": "stock_id",
        "股票名稱": "stock_name",
        "年化alpha": "alpha_annualized",
        "年化Alpha": "alpha_annualized",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    required = ["formation_date", "stock_id", "alpha"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"alpha scores 缺少必要欄位：{missing}")

    df["formation_date"] = parse_strategy3_date_series(df["formation_date"])
    if "holding_start" in df.columns:
        df["holding_start"] = parse_strategy3_date_series(df["holding_start"])
    if "holding_end" in df.columns:
        df["holding_end"] = parse_strategy3_date_series(df["holding_end"])

    df["stock_id"] = clean_stock_id(df["stock_id"])
    for c in [
        "alpha", "alpha_annualized", "pvalue_alpha", "alpha_rank",
        "rank_in_top150", "beta_mkt", "beta_smb", "beta_hml", "beta_rmw", "beta_cma",
        "r2", "adj_r2", "n_obs"
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["formation_date", "stock_id", "alpha"])
    df = df.drop_duplicates(["formation_date", "stock_id"], keep="last")
    return df.sort_values(["formation_date", "alpha"], ascending=[True, False]).reset_index(drop=True)


def strategy3_normalize_price(price_raw: pd.DataFrame) -> pd.DataFrame:
    df = price_raw.copy()
    df.columns = [str(c).strip() for c in df.columns]

    rename_map = {
        "日期": "date",
        "股票代號": "stock_id",
        "股票名稱": "stock_name",
        "收盤價": "adj_close",
        "還原收盤價": "adj_close",
        "成交量": "volume",
        "成交金額(千)": "trading_value_thousand",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    required = ["date", "stock_id", "adj_close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"AD PRICE 缺少必要欄位：{missing}")

    if "stock_name" not in df.columns:
        df["stock_name"] = ""

    df["date"] = parse_strategy3_date_series(df["date"])
    df["stock_id"] = clean_stock_id(df["stock_id"])
    df["adj_close"] = strategy3_to_num(df["adj_close"])

    df = df.dropna(subset=["date", "stock_id", "adj_close"])
    df = df.drop_duplicates(["date", "stock_id"], keep="last")
    return df.sort_values(["date", "stock_id"]).reset_index(drop=True)


def strategy3_next_trading_date(trading_dates: pd.DatetimeIndex, current_date: pd.Timestamp) -> pd.Timestamp | pd.NaT:
    future = trading_dates[trading_dates > current_date]
    if len(future) == 0:
        return pd.NaT
    return pd.Timestamp(future[0])


def strategy3_create_positions(alpha: pd.DataFrame, price_df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    trading_dates = pd.DatetimeIndex(sorted(price_df["date"].dropna().unique()))
    rows: list[pd.DataFrame] = []

    for fdate, g in alpha.groupby("formation_date"):
        g = g.sort_values("alpha", ascending=False).head(top_n).copy()
        if len(g) == 0:
            continue

        if "holding_start" not in g.columns or g["holding_start"].isna().all():
            holding_start = strategy3_next_trading_date(trading_dates, pd.Timestamp(fdate))
            g["holding_start"] = holding_start

        g = g.dropna(subset=["holding_start"])
        if len(g) == 0:
            continue

        n_selected = len(g)
        g["position_rank"] = np.arange(1, n_selected + 1)
        g["target_weight"] = 1.0 / n_selected
        g["top_n_setting"] = int(top_n)
        g["n_alpha_available"] = int(len(alpha[alpha["formation_date"] == fdate]))
        rows.append(g)

    if len(rows) == 0:
        return pd.DataFrame()

    positions = pd.concat(rows, ignore_index=True)

    # holding_end：下一個 holding_start
    unique_starts = sorted(pd.to_datetime(positions["holding_start"].dropna().unique()))
    start_to_end = {
        unique_starts[i]: unique_starts[i + 1] if i + 1 < len(unique_starts) else pd.NaT
        for i in range(len(unique_starts))
    }
    positions["holding_end"] = positions["holding_start"].map(start_to_end)

    positions["target_weight"] = (
        positions["target_weight"] /
        positions.groupby("holding_start")["target_weight"].transform("sum")
    )

    return positions.sort_values(["formation_date", "position_rank"]).reset_index(drop=True)


def strategy3_run_backtest(
    positions: pd.DataFrame,
    price_df: pd.DataFrame,
    initial_capital: float,
    commission_rate: float,
    sell_tax_rate: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    price_lookup = price_df.set_index(["date", "stock_id"])["adj_close"].sort_index()
    name_lookup = price_df.drop_duplicates("stock_id").set_index("stock_id")["stock_name"].to_dict()
    trading_dates = pd.DatetimeIndex(sorted(price_df["date"].dropna().unique()))

    rebalance_dates = pd.DatetimeIndex(sorted(positions["holding_start"].dropna().unique()))
    rebalance_set = set(rebalance_dates)

    if len(rebalance_dates) == 0:
        raise ValueError("沒有可用 holding_start，無法回測。")

    backtest_dates = trading_dates[trading_dates >= rebalance_dates[0]]
    if len(backtest_dates) == 0:
        raise ValueError("AD PRICE 中沒有回測期間價格。")

    cash = float(initial_capital)
    holdings: dict[str, float] = {}
    last_price: dict[str, float] = {}

    nav_rows: list[dict[str, object]] = []
    trade_rows: list[dict[str, object]] = []
    rebalance_rows: list[dict[str, object]] = []

    prev_nav = float(initial_capital)

    for current_date in backtest_dates:
        current_date = pd.Timestamp(current_date)

        today_prices = price_df[price_df["date"] == current_date]
        for _, r in today_prices.iterrows():
            last_price[r["stock_id"]] = r["adj_close"]

        pre_trade_stock_value = 0.0
        missing_price_count = 0
        current_values: dict[str, float] = {}

        for sid, shares in holdings.items():
            px = price_lookup.get((current_date, sid), np.nan)
            if pd.isna(px):
                px = last_price.get(sid, np.nan)

            if pd.isna(px) or px <= 0:
                missing_price_count += 1
                value = 0.0
            else:
                value = float(shares * px)

            current_values[sid] = value
            pre_trade_stock_value += value

        pre_trade_nav = cash + pre_trade_stock_value

        total_fee_today = 0.0
        turnover_today = 0.0
        buy_value_today = 0.0
        sell_value_today = 0.0

        if current_date in rebalance_set:
            target_df = positions[positions["holding_start"] == current_date].copy()

            valid_list = []
            for _, r in target_df.iterrows():
                sid = r["stock_id"]
                px = price_lookup.get((current_date, sid), np.nan)
                valid_list.append(pd.notna(px) and px > 0)

            target_df = target_df.loc[valid_list].copy()

            if len(target_df) > 0:
                target_df["target_weight"] = target_df["target_weight"] / target_df["target_weight"].sum()

                target_weights = dict(zip(target_df["stock_id"], target_df["target_weight"]))
                all_sids = sorted(set(list(holdings.keys()) + list(target_weights.keys())))

                target_values = {
                    sid: pre_trade_nav * target_weights.get(sid, 0.0)
                    for sid in all_sids
                }

                close_prices = {}
                for sid in all_sids:
                    px = price_lookup.get((current_date, sid), np.nan)
                    if pd.isna(px):
                        px = last_price.get(sid, np.nan)
                    close_prices[sid] = px

                diff_values = {}
                for sid in all_sids:
                    current_value = current_values.get(sid, 0.0)
                    target_value = target_values.get(sid, 0.0)
                    diff_values[sid] = target_value - current_value

                # Sell first
                for sid, diff_value in diff_values.items():
                    if diff_value >= 0:
                        continue

                    shares_old = holdings.get(sid, 0.0)
                    px = close_prices.get(sid, np.nan)

                    if shares_old <= 0 or pd.isna(px) or px <= 0:
                        continue

                    sell_value = min(-diff_value, shares_old * px)
                    sell_shares = sell_value / px

                    commission = sell_value * commission_rate
                    tax = sell_value * sell_tax_rate
                    fee = commission + tax

                    cash += sell_value - fee
                    holdings[sid] = shares_old - sell_shares

                    if abs(holdings[sid]) < 1e-10:
                        holdings.pop(sid, None)

                    sell_value_today += sell_value
                    turnover_today += sell_value
                    total_fee_today += fee

                    trade_rows.append({
                        "date": current_date,
                        "stock_id": sid,
                        "stock_name": name_lookup.get(sid, None),
                        "action": "SELL",
                        "price_type": "t+1 close",
                        "price": px,
                        "shares": sell_shares,
                        "trade_value": sell_value,
                        "commission": commission,
                        "tax": tax,
                        "total_fee": fee,
                        "cash_after_trade": cash,
                        "pre_trade_nav": pre_trade_nav,
                    })

                # Buy after selling
                buy_candidates = []
                for sid, diff_value in diff_values.items():
                    if diff_value <= 0:
                        continue

                    px = close_prices.get(sid, np.nan)
                    if pd.isna(px) or px <= 0:
                        continue

                    buy_candidates.append((sid, diff_value, px))

                desired_buy_value = sum(x[1] for x in buy_candidates)
                if desired_buy_value > 0:
                    max_affordable_buy_value = cash / (1 + commission_rate)
                    buy_scale = min(1.0, max_affordable_buy_value / desired_buy_value)
                else:
                    buy_scale = 0.0

                for sid, desired_value, px in buy_candidates:
                    buy_value = desired_value * buy_scale
                    if buy_value <= 0:
                        continue

                    buy_shares = buy_value / px
                    if buy_shares <= 0:
                        continue

                    commission = buy_value * commission_rate
                    fee = commission
                    total_cost = buy_value + fee

                    if total_cost > cash + 1e-8:
                        continue

                    cash -= total_cost
                    holdings[sid] = holdings.get(sid, 0.0) + buy_shares

                    buy_value_today += buy_value
                    turnover_today += buy_value
                    total_fee_today += fee

                    trade_rows.append({
                        "date": current_date,
                        "stock_id": sid,
                        "stock_name": name_lookup.get(sid, None),
                        "action": "BUY",
                        "price_type": "t+1 close",
                        "price": px,
                        "shares": buy_shares,
                        "trade_value": buy_value,
                        "commission": commission,
                        "tax": 0.0,
                        "total_fee": fee,
                        "cash_after_trade": cash,
                        "pre_trade_nav": pre_trade_nav,
                    })

                rebalance_rows.append({
                    "date": current_date,
                    "pre_trade_nav": pre_trade_nav,
                    "n_target_stocks": len(target_df),
                    "n_holdings_after": len(holdings),
                    "buy_value": buy_value_today,
                    "sell_value": sell_value_today,
                    "turnover_value": turnover_today,
                    "turnover_rate": turnover_today / pre_trade_nav if pre_trade_nav > 0 else np.nan,
                    "total_fee": total_fee_today,
                    "cash_after_rebalance": cash,
                })

        post_trade_stock_value = 0.0
        for sid, shares in holdings.items():
            px = price_lookup.get((current_date, sid), np.nan)
            if pd.isna(px):
                px = last_price.get(sid, np.nan)
            if pd.notna(px) and px > 0:
                post_trade_stock_value += shares * px

        post_trade_nav = cash + post_trade_stock_value
        daily_return = post_trade_nav / prev_nav - 1 if prev_nav > 0 else np.nan

        nav_rows.append({
            "date": current_date,
            "cash": cash,
            "stock_value": post_trade_stock_value,
            "portfolio_nav": post_trade_nav,
            "daily_return": daily_return,
            "cum_return": post_trade_nav / initial_capital - 1,
            "n_holdings": len(holdings),
            "is_rebalance_day": current_date in rebalance_set,
            "fee_paid_today": total_fee_today,
            "turnover_today": turnover_today,
            "missing_price_count": missing_price_count,
        })

        prev_nav = post_trade_nav

    nav_df = pd.DataFrame(nav_rows)
    trades_df = pd.DataFrame(trade_rows)
    rebalance_df = pd.DataFrame(rebalance_rows)

    nav_df["running_max"] = nav_df["portfolio_nav"].cummax()
    nav_df["drawdown"] = nav_df["portfolio_nav"] / nav_df["running_max"] - 1

    return nav_df, trades_df, rebalance_df


@st.cache_data(ttl=3600, show_spinner=False)
def strategy3_download_benchmark(
    ticker: str,
    start: str,
    end: str,
    auto_adjust: bool,
    initial_capital: float,
) -> pd.DataFrame:
    yf_end = (pd.Timestamp(end) + pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    raw = yf.download(
        ticker,
        start=start,
        end=yf_end,
        auto_adjust=auto_adjust,
        progress=False,
    )

    if raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    bench = raw.reset_index().rename(columns={"Date": "date", "Close": "benchmark_close"})
    bench["date"] = strategy3_force_datetime_ns(bench["date"])
    bench = bench[["date", "benchmark_close"]].dropna().sort_values("date")
    bench["benchmark_daily_return"] = bench["benchmark_close"].pct_change()
    bench.loc[bench.index[0], "benchmark_daily_return"] = 0.0
    bench["benchmark_nav"] = initial_capital * (1 + bench["benchmark_daily_return"]).cumprod()
    bench["benchmark_cum_return"] = bench["benchmark_nav"] / initial_capital - 1
    bench["benchmark_running_max"] = bench["benchmark_nav"].cummax()
    bench["benchmark_drawdown"] = bench["benchmark_nav"] / bench["benchmark_running_max"] - 1
    return bench


def strategy3_attach_benchmark(nav_df: pd.DataFrame, settings: FF5AlphaSettings) -> pd.DataFrame:
    """
    下載 benchmark，對齊策略日期，並將 benchmark NAV rebased 到策略第一天。

    修正重點：
    - yfinance 的 date 有時是 datetime64[s]，策略 NAV 是 datetime64[ns]。
      pd.merge_asof 需要左右 key 完全同型別，所以這裡強制轉成 datetime64[ns]。
    """
    out = nav_df.copy()
    out["date"] = strategy3_force_datetime_ns(out["date"])
    out = out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # 多抓幾天，避免策略第一天不是 benchmark 交易日導致缺值
    start = (out["date"].min() - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    end = out["date"].max().strftime("%Y-%m-%d")

    bench = strategy3_download_benchmark(
        settings.benchmark_ticker,
        start,
        end,
        settings.benchmark_auto_adjust,
        settings.initial_capital,
    )

    if bench.empty:
        for c in [
            "benchmark_close", "benchmark_daily_return", "benchmark_nav",
            "benchmark_cum_return", "benchmark_drawdown",
            "active_daily_return", "active_cum_return", "relative_nav"
        ]:
            out[c] = np.nan
        return out

    bench = bench.copy()
    bench["date"] = strategy3_force_datetime_ns(bench["date"])
    bench = bench.dropna(subset=["date", "benchmark_close"]).sort_values("date").reset_index(drop=True)

    left = out[["date"]].copy().sort_values("date").reset_index(drop=True)
    left["date"] = strategy3_force_datetime_ns(left["date"])

    right = bench[["date", "benchmark_close"]].copy().sort_values("date").reset_index(drop=True)
    right["date"] = strategy3_force_datetime_ns(right["date"])

    aligned = pd.merge_asof(
        left,
        right,
        on="date",
        direction="backward",
    )

    aligned["benchmark_daily_return"] = aligned["benchmark_close"].pct_change()
    aligned.loc[aligned.index[0], "benchmark_daily_return"] = 0.0

    aligned["benchmark_nav"] = settings.initial_capital * (
        1 + aligned["benchmark_daily_return"].fillna(0)
    ).cumprod()

    aligned["benchmark_cum_return"] = aligned["benchmark_nav"] / settings.initial_capital - 1
    aligned["benchmark_running_max"] = aligned["benchmark_nav"].cummax()
    aligned["benchmark_drawdown"] = aligned["benchmark_nav"] / aligned["benchmark_running_max"] - 1

    aligned["date"] = strategy3_force_datetime_ns(aligned["date"])

    out = pd.merge(
        out,
        aligned[[
            "date", "benchmark_close", "benchmark_daily_return",
            "benchmark_nav", "benchmark_cum_return", "benchmark_drawdown"
        ]],
        on="date",
        how="left",
    )

    out["active_daily_return"] = out["daily_return"] - out["benchmark_daily_return"]
    out["active_cum_return"] = out["cum_return"] - out["benchmark_cum_return"]
    out["relative_nav"] = out["portfolio_nav"] / out["benchmark_nav"]
    return out

def strategy3_calc_performance(nav_df: pd.DataFrame, trades_df: pd.DataFrame, rebalance_df: pd.DataFrame, settings: FF5AlphaSettings) -> pd.DataFrame:
    def _one(nav_col: str, ret_col: str, label: str) -> dict[str, object]:
        daily_ret = nav_df[ret_col].dropna()
        final_nav = nav_df[nav_col].iloc[-1]
        total_return = final_nav / settings.initial_capital - 1
        n_days = len(nav_df)
        years = n_days / 252
        ann_return = (final_nav / settings.initial_capital) ** (1 / years) - 1 if years > 0 else np.nan
        ann_vol = daily_ret.std() * np.sqrt(252)
        sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(252) if daily_ret.std() != 0 else np.nan
        running_max = nav_df[nav_col].cummax()
        max_drawdown = (nav_df[nav_col] / running_max - 1).min()
        win_rate = (daily_ret > 0).mean()
        return {
            "name": label,
            "initial_capital": settings.initial_capital,
            "final_nav": final_nav,
            "total_return": total_return,
            "annualized_return": ann_return,
            "annualized_volatility": ann_vol,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "n_trading_days": n_days,
        }

    rows = [_one("portfolio_nav", "daily_return", f"FF5 Alpha Top {settings.top_n}")]
    if "benchmark_nav" in nav_df.columns and nav_df["benchmark_nav"].notna().any():
        rows.append(_one("benchmark_nav", "benchmark_daily_return", settings.benchmark_ticker))

    perf = pd.DataFrame(rows)

    total_fee_paid = trades_df["total_fee"].sum() if len(trades_df) > 0 else 0.0
    total_trade_value = trades_df["trade_value"].sum() if len(trades_df) > 0 else 0.0
    avg_turnover = rebalance_df["turnover_rate"].mean() if len(rebalance_df) > 0 else np.nan

    perf["commission_rate"] = np.nan
    perf["sell_tax_rate"] = np.nan
    perf["n_rebalances"] = np.nan
    perf["total_fee_paid"] = np.nan
    perf["total_trade_value"] = np.nan
    perf["avg_turnover_per_rebalance"] = np.nan
    # keep this column as object dtype; assigning a string into a float column can fail on Streamlit/Pandas
    perf["execution_price"] = ""
    perf["benchmark"] = settings.benchmark_ticker

    strategy_name = f"FF5 Alpha Top {settings.top_n}"
    perf.loc[perf["name"] == strategy_name, "commission_rate"] = settings.commission_rate
    perf.loc[perf["name"] == strategy_name, "sell_tax_rate"] = settings.sell_tax_rate
    perf.loc[perf["name"] == strategy_name, "n_rebalances"] = len(rebalance_df)
    perf.loc[perf["name"] == strategy_name, "total_fee_paid"] = total_fee_paid
    perf.loc[perf["name"] == strategy_name, "total_trade_value"] = total_trade_value
    perf.loc[perf["name"] == strategy_name, "avg_turnover_per_rebalance"] = avg_turnover
    perf.loc[perf["name"] == strategy_name, "execution_price"] = "t+1 close"

    if len(perf) >= 2:
        strat_total = perf.loc[perf["name"] == strategy_name, "total_return"].iloc[0]
        bench_total = perf.loc[perf["name"] == settings.benchmark_ticker, "total_return"].iloc[0]
        strat_ann = perf.loc[perf["name"] == strategy_name, "annualized_return"].iloc[0]
        bench_ann = perf.loc[perf["name"] == settings.benchmark_ticker, "annualized_return"].iloc[0]

        active_ret = nav_df["active_daily_return"].dropna() if "active_daily_return" in nav_df.columns else pd.Series(dtype=float)
        tracking_error = active_ret.std() * np.sqrt(252) if len(active_ret) else np.nan
        information_ratio = active_ret.mean() / active_ret.std() * np.sqrt(252) if len(active_ret) and active_ret.std() != 0 else np.nan

        perf["strategy_minus_benchmark_total_return"] = np.nan
        perf["strategy_minus_benchmark_annualized_return"] = np.nan
        perf["tracking_error"] = np.nan
        perf["information_ratio"] = np.nan

        perf.loc[perf["name"] == strategy_name, "strategy_minus_benchmark_total_return"] = strat_total - bench_total
        perf.loc[perf["name"] == strategy_name, "strategy_minus_benchmark_annualized_return"] = strat_ann - bench_ann
        perf.loc[perf["name"] == strategy_name, "tracking_error"] = tracking_error
        perf.loc[perf["name"] == strategy_name, "information_ratio"] = information_ratio

    return perf


def strategy3_plot_nav(nav_df: pd.DataFrame, top_n: int, benchmark_ticker: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=nav_df["date"], y=nav_df["portfolio_nav"], mode="lines", name=f"FF5 Alpha Top {top_n}"))
    if "benchmark_nav" in nav_df.columns and nav_df["benchmark_nav"].notna().any():
        fig.add_trace(go.Scatter(x=nav_df["date"], y=nav_df["benchmark_nav"], mode="lines", name=f"{benchmark_ticker} Benchmark"))
    fig.update_layout(title="Portfolio NAV vs Benchmark", template="plotly_white", height=460, hovermode="x unified")
    return fig


def strategy3_plot_cum_return(nav_df: pd.DataFrame, top_n: int, benchmark_ticker: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=nav_df["date"], y=nav_df["cum_return"], mode="lines", name=f"FF5 Alpha Top {top_n}"))
    if "benchmark_cum_return" in nav_df.columns and nav_df["benchmark_cum_return"].notna().any():
        fig.add_trace(go.Scatter(x=nav_df["date"], y=nav_df["benchmark_cum_return"], mode="lines", name=f"{benchmark_ticker} Benchmark"))
    fig.update_layout(title="Cumulative Return", template="plotly_white", height=420, yaxis_tickformat=".1%", hovermode="x unified")
    return fig


def strategy3_plot_drawdown(nav_df: pd.DataFrame, top_n: int, benchmark_ticker: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=nav_df["date"], y=nav_df["drawdown"], fill="tozeroy", mode="lines", name=f"FF5 Alpha Top {top_n}"))
    if "benchmark_drawdown" in nav_df.columns and nav_df["benchmark_drawdown"].notna().any():
        fig.add_trace(go.Scatter(x=nav_df["date"], y=nav_df["benchmark_drawdown"], mode="lines", name=f"{benchmark_ticker} Benchmark"))
    fig.update_layout(title="Drawdown", template="plotly_white", height=350, yaxis_tickformat=".1%", hovermode="x unified")
    return fig


def render_strategy3_notes() -> None:
    st.subheader("③ 方法說明 / GitHub 資料準備")

    st.markdown(
        """
        ### 策略3：台股五因子 Alpha 月調倉策略

        **策略邏輯**
        1. 先在 Colab 建構台股五因子：`MKT_RF`, `SMB`, `HML`, `RMW`, `CMA`。
        2. 對市值前 150 股票做 rolling five-factor regression。
        3. 每個 `formation_date` 取得每檔股票的五因子 alpha。
        4. 網站端依 alpha 由高到低排序，選出 Alpha Top N。
        5. t 日收到訊號，t+1 收盤價調倉。
        6. 買賣皆扣手續費，並與 0050 或自訂 benchmark 比較。

        **網站必要 CSV**
        - `alpha_scores_ff5_top150.csv`
        - `cmoney_daily_adjusted_price.csv`，或將多個 `.xlsx` / `.csv` 放在 `ad_price/` 資料夾

        **alpha_scores_ff5_top150.csv 必要欄位**
        - `formation_date`
        - `stock_id`
        - `stock_name`
        - `alpha`

        **AD PRICE 必要欄位**
        - 英文欄位：`date`, `stock_id`, `stock_name`, `adj_close`
        - 或 CMoney 中文欄位：`日期`, `股票代號`, `股票名稱`, `收盤價`
        """
    )

    st.markdown(
        """
        ### GitHub 上傳步驟

        1. 在專案根目錄建立資料夾：
           ```text
           data/strategy3/
           ```

        2. 把 Colab 產出的 alpha 檔放進去：
           ```text
           data/strategy3/alpha_scores_ff5_top150.csv
           ```

        3. 把 AD PRICE 還原收盤價資料整理成一個 CSV 後放進去：
           ```text
           data/strategy3/cmoney_daily_adjusted_price.csv
           ```

        4. 你的 GitHub repo 建議結構：
           ```text
           your-repo/
           ├── app.py
           ├── requirements.txt
           └── data/
               └── strategy3/
                   ├── alpha_scores_ff5_top150.csv
                   └── cmoney_daily_adjusted_price.csv
           ```

        5. `requirements.txt` 至少需要：
           ```text
           streamlit
           pandas
           numpy
           plotly
           statsmodels
           yfinance
           openpyxl
           ```

        6. 如果 CSV 超過 GitHub 單檔 100MB 限制，建議：
           - 使用 Git LFS；
           - 或把 AD PRICE 分年放在 `data/strategy3/ad_price/` 資料夾，網站路徑填資料夾；支援 `.csv`、`.xlsx`、`.xls`；
           - 或改用 Streamlit Cloud secrets / 外部雲端資料源。
        """
    )


def render_strategy3_page() -> None:
    st.title("策略3(台股五因子 Alpha 月調倉策略)")
    st.caption("使用已建構之五因子 rolling alpha，每月選 Alpha Top N，並以 t+1 收盤價調倉。")

    settings = strategy3_sidebar_settings()

    tabs = st.tabs(["① 回測結果", "② 持股 / 交易資料", "③ 方法說明 / GitHub 資料"])

    with tabs[0]:
        st.subheader("① 回測結果")
        st.write(
            f"目前設定：Alpha Top **{settings.top_n}**，初始資金 **{settings.initial_capital:,.0f}**，"
            f"手續費率 **{settings.commission_rate:.4%}**，Benchmark：**{settings.benchmark_ticker}**。"
        )

        run = st.button("執行策略3回測", type="primary", use_container_width=True)

        if run:
            try:
                with st.spinner("讀取 alpha scores..."):
                    alpha_raw = strategy3_read_csv_file(settings.alpha_path)
                    alpha = strategy3_normalize_alpha(alpha_raw)

                with st.spinner("讀取 AD PRICE..."):
                    price_raw = strategy3_read_csv_file(settings.price_path)
                    price_df = strategy3_normalize_price(price_raw)

                with st.spinner("產生 Top N 每月持股..."):
                    positions = strategy3_create_positions(alpha, price_df, settings.top_n)
                    if positions.empty:
                        st.error("沒有成功產生持股清單，請檢查 alpha scores 與 price data。")
                        return

                with st.spinner("執行 t+1 收盤價回測..."):
                    nav_df, trades_df, rebalance_df = strategy3_run_backtest(
                        positions=positions,
                        price_df=price_df,
                        initial_capital=settings.initial_capital,
                        commission_rate=settings.commission_rate,
                        sell_tax_rate=settings.sell_tax_rate,
                    )

                with st.spinner(f"下載 benchmark：{settings.benchmark_ticker}..."):
                    nav_df = strategy3_attach_benchmark(nav_df, settings)

                perf = strategy3_calc_performance(nav_df, trades_df, rebalance_df, settings)

                st.session_state["strategy3_result"] = {
                    "alpha": alpha,
                    "price_df": price_df,
                    "positions": positions,
                    "nav_df": nav_df,
                    "trades_df": trades_df,
                    "rebalance_df": rebalance_df,
                    "perf": perf,
                    "settings": settings,
                }

            except Exception as exc:
                st.error(f"策略3回測失敗：{exc}")
                return

        result = st.session_state.get("strategy3_result")
        if not result:
            st.info("請先確認左側參數與 CSV 路徑，然後按「執行策略3回測」。")
            return

        nav_df = result["nav_df"]
        trades_df = result["trades_df"]
        rebalance_df = result["rebalance_df"]
        perf = result["perf"]
        positions = result["positions"]
        settings_used = result["settings"]

        strategy_row = perf.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Return", f"{strategy_row['total_return']:.2%}")
        c2.metric("Sharpe", f"{strategy_row['sharpe_ratio']:.2f}" if pd.notna(strategy_row["sharpe_ratio"]) else "NA")
        c3.metric("Max Drawdown", f"{strategy_row['max_drawdown']:.2%}")
        c4.metric("Final NAV", f"{strategy_row['final_nav']:,.0f}")

        st.plotly_chart(strategy3_plot_nav(nav_df, settings_used.top_n, settings_used.benchmark_ticker), use_container_width=True)
        st.plotly_chart(strategy3_plot_cum_return(nav_df, settings_used.top_n, settings_used.benchmark_ticker), use_container_width=True)
        st.plotly_chart(strategy3_plot_drawdown(nav_df, settings_used.top_n, settings_used.benchmark_ticker), use_container_width=True)

        st.markdown("#### 績效比較表")
        st.dataframe(perf, use_container_width=True, hide_index=True)

        excel_bytes = df_to_excel_bytes({
            "performance": perf,
            "nav": nav_df,
            "positions": positions,
            "trades": trades_df,
            "rebalance": rebalance_df,
        })
        st.download_button(
            "下載策略3完整回測結果 Excel",
            data=excel_bytes,
            file_name=f"strategy3_ff5_alpha_top{settings_used.top_n}_backtest.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    with tabs[1]:
        st.subheader("② 持股 / 交易資料")
        result = st.session_state.get("strategy3_result")
        if not result:
            st.info("請先到「回測結果」執行策略3回測。")
            return

        positions = result["positions"]
        trades_df = result["trades_df"]
        rebalance_df = result["rebalance_df"]
        nav_df = result["nav_df"]

        st.markdown("#### 每月 Alpha Top N 成分股")
        st.dataframe(positions, use_container_width=True, hide_index=True)

        st.markdown("#### 交易紀錄")
        st.dataframe(trades_df, use_container_width=True, hide_index=True)

        st.markdown("#### 調倉摘要")
        st.dataframe(rebalance_df, use_container_width=True, hide_index=True)

        st.markdown("#### NAV 明細")
        st.dataframe(nav_df, use_container_width=True, hide_index=True)

    with tabs[2]:
        render_strategy3_notes()

def main() -> None:
    if "selected_strategy" not in st.session_state:
        st.session_state["selected_strategy"] = None

    if st.session_state["selected_strategy"] is None:
        render_strategy_selector()
        return

    if st.session_state["selected_strategy"] == "distance":
        render_strategy_page("distance")
    elif st.session_state["selected_strategy"] == "cointegration":
        render_strategy_page("cointegration")
    elif st.session_state["selected_strategy"] == "ff5_alpha":
        render_strategy3_page()
    else:
        st.session_state["selected_strategy"] = None
        st.rerun()


if __name__ == "__main__":
    main()
