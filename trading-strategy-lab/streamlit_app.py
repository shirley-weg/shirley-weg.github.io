
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Literal
import itertools
import logging
import re
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import statsmodels.api as sm
import yfinance as yf
from statsmodels.tsa.stattools import coint
from strategy3_ff5_pipeline import FF5RawBuildConfig, run_ff5_raw_pipeline

logging.getLogger("yfinance").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore")


# Global backtest window requested for this project.
BACKTEST_START_DEFAULT = pd.Timestamp("2019-01-01")
BACKTEST_END_DEFAULT = pd.Timestamp("2026-05-19")

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



# ============================================================
# Safe Output Helpers for Streamlit / Excel
# ============================================================

def _is_missing_scalar(value: object) -> bool:
    """Return True only for scalar missing values; lists/dicts/arrays are not treated as scalars."""
    if value is None:
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    if isinstance(missing, (bool, np.bool_)):
        return bool(missing)
    return False


def _format_mixed_cell_for_display(value: object) -> str:
    """
    Convert object-dtype values into a single Arrow-safe string representation.

    Streamlit uses PyArrow to render dataframes. If a single object column contains
    mixed Python types, for example strings mixed with numbers, lists, dicts, dates,
    or None, PyArrow may raise ArrowInvalid. Converting only object-like columns to
    strings keeps numeric/datetime columns usable while preventing the red error box.
    """
    if _is_missing_scalar(value):
        return ""
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return ""
        return value.strftime("%Y-%m-%d") if value.time() == pd.Timestamp(0).time() else value.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(value, np.datetime64):
        ts = pd.to_datetime(value, errors="coerce")
        if pd.isna(ts):
            return ""
        return ts.strftime("%Y-%m-%d") if ts.time() == pd.Timestamp(0).time() else ts.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(value, (list, tuple, set, dict, np.ndarray)):
        return str(value)
    return str(value)


def prepare_dataframe_for_output(obj: pd.DataFrame | pd.Series | object) -> pd.DataFrame:
    """
    Make a DataFrame safe for Streamlit display and Excel export.

    Key fix: object/categorical/timedelta columns are converted to strings so PyArrow
    no longer needs to infer a single incompatible type from mixed Python objects.
    """
    if obj is None:
        return pd.DataFrame()
    if isinstance(obj, pd.Series):
        df = obj.to_frame(obj.name or "value").copy()
    elif isinstance(obj, pd.DataFrame):
        df = obj.copy()
    else:
        try:
            df = pd.DataFrame(obj).copy()
        except Exception:
            df = pd.DataFrame({"value": [_format_mixed_cell_for_display(obj)]})

    df.columns = [str(c) for c in df.columns]
    if df.empty:
        return df

    df = df.replace([np.inf, -np.inf], np.nan)

    for col in df.columns:
        s = df[col]

        if pd.api.types.is_datetime64_any_dtype(s):
            dt = pd.to_datetime(s, errors="coerce")
            try:
                if getattr(dt.dt, "tz", None) is not None:
                    dt = dt.dt.tz_localize(None)
            except Exception:
                pass
            df[col] = dt.astype("datetime64[ns]")
            continue

        if pd.api.types.is_object_dtype(s) or pd.api.types.is_categorical_dtype(s) or pd.api.types.is_timedelta64_dtype(s):
            df[col] = s.map(_format_mixed_cell_for_display).astype("string")
            continue

    return df


def safe_streamlit_dataframe(obj: pd.DataFrame | pd.Series | object, **kwargs) -> None:
    """Display a dataframe without PyArrow type-inference crashes."""
    try:
        st.dataframe(prepare_dataframe_for_output(obj), **kwargs)
    except Exception:
        fallback = prepare_dataframe_for_output(obj).astype("string")
        st.dataframe(fallback, **kwargs)

def df_to_excel_bytes(sheets: dict[str, pd.DataFrame | pd.Series]) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for name, obj in sheets.items():
            safe_name = str(name)[:31]
            df = prepare_dataframe_for_output(obj)
            df.to_excel(writer, sheet_name=safe_name, index=True)
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
              <h3>策略3(台股五因子 月調倉策略)</h3>
              <p>
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("進入 策略3(五因子 Alpha)", use_container_width=True):
            st.session_state["selected_strategy"] = "ff5_alpha"
            st.rerun()


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
    data_start = st.sidebar.date_input("資料下載起日", value=BACKTEST_START_DEFAULT.date())
    trading_start = st.sidebar.date_input("正式回測起日", value=BACKTEST_START_DEFAULT.date())
    use_today = st.sidebar.checkbox("回測到最新資料", value=False)
    if use_today:
        trading_end = None
        st.sidebar.caption("回測結束日：yfinance 最新可取得資料")
    else:
        trading_end = pd.Timestamp(st.sidebar.date_input("正式回測結束日", value=BACKTEST_END_DEFAULT.date()))
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
    safe_streamlit_dataframe(universe[["code", "name", "industry", "ticker_yf"]], use_container_width=True, hide_index=True)

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
    safe_streamlit_dataframe(show[display_cols], use_container_width=True, hide_index=True)

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
        safe_streamlit_dataframe(trades_df, use_container_width=True, hide_index=True)
        st.markdown("#### 每週選股狀態")
        safe_streamlit_dataframe(result["weekly_summary"], use_container_width=True, hide_index=True)
        st.markdown("#### 歷史候選 pair")
        safe_streamlit_dataframe(result["selected_pairs_history"], use_container_width=True, hide_index=True)
        st.markdown("#### Z-score / eligible 狀態")
        safe_streamlit_dataframe(z_df, use_container_width=True)

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
            r"""
            ### 配對策略1(距離法)

            本策略使用 **Distance Method（距離法）** 尋找價格走勢相似的股票配對。策略核心概念是：若兩檔股票在 formation period 內的標準化 log price 走勢非常接近，則兩者可能具有短期相對價格均值回歸特性；當兩者的標準化價差偏離過大時，策略預期未來價差會收斂。

            #### 1. 策略流程簡短總結

            1. 每週用過去 `lookback_days` 個交易日作為 formation period。
            2. 對股票池內每檔股票取 log price，並在 formation period 內做標準化。
            3. 對所有兩兩 pair 計算標準化價格距離，也就是 `SSD = sum((Z_X - Z_Y)^2)`。
            4. 先依 SSD 由小到大排序，取距離最小的前 `preselect_n` 組 pair 作為初選清單。
            5. 對初選 pair 加上同產業條件與 beta 差距條件，只保留 `same_industry = True` 且 `beta_diff < beta_diff_threshold` 的 pair。
            6. 單一 pair 回測時，只有當該 pair 在當週候選清單中，才允許開新倉。
            7. 進場訊號使用進場當下 formation period 重新計算出的標準化參數建立 spread，並將 spread 轉成 z-score。
            8. 今日收盤產生訊號，下一個交易日開盤成交；出場條件則由均值回歸、停損或最大持有天數決定。

            #### 2. Formation period 與 log price

            在每個週調倉日 $f$，取過去 $L$ 個交易日作為 formation period：

            $$
            \mathcal{F}_f = \{t=f-L, f-L+1, \dots, f-1\}
            $$

            對任意股票 $i$，先將價格轉成 log price：

            $$
            x_{i,t} = \log(P_{i,t})
            $$

            其中 $P_{i,t}$ 為股票 $i$ 在第 $t$ 日的收盤價，$x_{i,t}$ 為其對數價格。使用 log price 可以降低不同股價水準造成的尺度差異，並讓價格比例變化轉換成加法形式。

            #### 3. 標準化價格序列

            距離法不能直接比較原始股價，因為不同股票的價格水準可能差異很大。因此本策略會在 formation period 內，對每檔股票的 log price 做 z-score 標準化。

            股票 $i$ 在 formation period 內的平均 log price 為：

            $$
            \mu_i
            =
            \frac{1}{L}
            \sum_{t \in \mathcal{F}_f}
            x_{i,t}
            $$

            標準差為：

            $$
            \sigma_i
            =
            \sqrt{
            \frac{1}{L}
            \sum_{t \in \mathcal{F}_f}
            (x_{i,t}-\mu_i)^2
            }
            $$

            標準化後的價格序列定義為：

            $$
            Z_{i,t}
            =
            \frac{x_{i,t}-\mu_i}{\sigma_i}
            $$

            標準化後，每檔股票在 formation period 內都被轉成平均數約為 0、標準差約為 1 的序列，因此可以公平比較兩檔股票的相對走勢是否接近。

            #### 4. Pair distance / SSD 計算公式

            對任意一組候選 pair $(X,Y)$，本策略計算兩檔股票標準化 log price 的平方距離總和：

            $$
            SSD_{XY}
            =
            \sum_{t \in \mathcal{F}_f}
            (Z_{X,t} - Z_{Y,t})^2
            $$

            其中：

            | 符號 | 意義 |
            |---|---|
            | $Z_{X,t}$ | 股票 $X$ 在第 $t$ 日的標準化 log price |
            | $Z_{Y,t}$ | 股票 $Y$ 在第 $t$ 日的標準化 log price |
            | $SSD_{XY}$ | pair $(X,Y)$ 的距離分數 |

            $SSD_{XY}$ 越小，代表兩檔股票在 formation period 內的標準化價格走勢越接近，也就是歷史相對價格關係越穩定。

            初步候選 pair 集合可表示為先對所有 pair 的 SSD 排序：

            $$
            SSD_{(1)}
            \le
            SSD_{(2)}
            \le
            \cdots
            \le
            SSD_{(m)}
            $$

            再取前 `preselect_n` 組作為初選集合：

            $$
            \mathcal{C}_0
            =
            \left\{
            (X,Y):
            rank(SSD_{XY}) \le preselect\_n,
            \; n_{XY} \ge min\_obs
            \right\}
            $$

            #### 5. Beta 與同產業篩選公式

            為避免兩檔股票雖然價格走勢接近，但實際市場風險暴露差異太大，本策略會估計每檔股票相對大盤的 beta。先計算個股與大盤的 log return：

            $$
            r_{i,t} = \Delta \log(P_{i,t})
            =
            \log(P_{i,t}) - \log(P_{i,t-1})
            $$

            $$
            r_{M,t} = \Delta \log(M_t)
            =
            \log(M_t) - \log(M_{t-1})
            $$

            其中 $M_t$ 為大盤指數價格。股票 $i$ 的 beta 定義為：

            $$
            \beta_i
            =
            \frac{Cov(r_{i,t}, r_{M,t})}{Var(r_{M,t})}
            $$

            對 pair $(X,Y)$，beta 差距為：

            $$
            beta\_diff_{XY}
            =
            |\beta_X - \beta_Y|
            $$

            最終候選 pair 集合為：

            $$
            \mathcal{C}_{final}
            =
            \left\{
            (X,Y) \in \mathcal{C}_0:
            Industry_X = Industry_Y,
            \;
            |\beta_X - \beta_Y| < beta\_diff\_threshold
            \right\}
            $$

            最後再從 $\mathcal{C}_{final}$ 中取排序前 `k_final` 組 pair，作為該週允許交易的候選清單。

            #### 6. Spread 與 z-score 建構

            在單一 pair 回測中，若 pair $(X,Y)$ 在當週候選清單中，策略會使用該週 formation period 重新計算 $X$ 與 $Y$ 的標準化參數：

            $$
            Z_{X,t}
            =
            \frac{x_{X,t}-\mu_X}{\sigma_X}
            $$

            $$
            Z_{Y,t}
            =
            \frac{x_{Y,t}-\mu_Y}{\sigma_Y}
            $$

            距離法的 spread 定義為兩檔股票標準化 log price 的差：

            $$
            S_t = Z_{X,t} - Z_{Y,t}
            $$

            在 formation period 內計算 spread 的平均數與標準差：

            $$
            \mu_S
            =
            \frac{1}{L}
            \sum_{t \in \mathcal{F}_f}
            S_t
            $$

            $$
            \sigma_S
            =
            \sqrt{
            \frac{1}{L}
            \sum_{t \in \mathcal{F}_f}
            (S_t-\mu_S)^2
            }
            $$

            每日交易訊號使用 z-score 衡量 spread 偏離均衡的程度：

            $$
            z_t
            =
            \frac{S_t - \mu_S}{\sigma_S}
            $$

            若 $z_t$ 為正且偏高，代表 $X$ 的標準化價格相對於 $Y$ 偏高，策略傾向放空 spread；若 $z_t$ 為負且偏低，代表 $X$ 的標準化價格相對於 $Y$ 偏低，策略傾向做多 spread。

            #### 7. 進出場方向與部位權重

            設進場方向為 $d_t$：

            $$
            d_t =
            \begin{cases}
            -1, & z_{t-1} \le entry\_z \text{ 且 } entry\_z < z_t < stop\_z \\
            +1, & z_{t-1} \ge -entry\_z \text{ 且 } -stop\_z < z_t < -entry\_z
            \end{cases}
            $$

            其中：

            - $d_t=-1$：spread 過高，放空 spread，也就是放空 $X$、做多 $Y$。
            - $d_t=+1$：spread 過低，做多 spread，也就是做多 $X$、放空 $Y$。

            #### 8. 出場條件

            持倉後每天用進場當下估計的 spread 參數持續計算 $z_t$。若符合以下任一條件，則於下一個交易日開盤出場：

            $$
            |z_t| < exit\_z
            \quad \Rightarrow \quad
            \text{均值回歸出場}
            $$

            $$
            |z_t| > stop\_z
            \quad \Rightarrow \quad
            \text{停損出場}
            $$

            $$
            holding\_days \ge max\_holding\_days
            \quad \Rightarrow \quad
            \text{最大持有天數出場}
            $$

            若發生 stop loss，策略會暫停重新進場，直到：

            $$
            |z_t| < reentry\_reset\_z
            $$

            才重新允許該 pair 產生新的進場訊號。這樣可以避免 spread 持續發散時反覆進場，降低連續停損風險。
            """
        )
    else:
        st.markdown(
            r"""
            ### 配對策略2(共整合法)

            本策略使用 **Engle-Granger 共整合檢定** 尋找長期均衡關係較穩定的股票配對。策略核心概念是：若兩檔股票的 log price 雖然各自可能為非定態序列，但存在某一線性組合為定態序列，則代表兩者具有長期均衡關係；當 spread 偏離均衡過大時，策略預期未來會產生均值回歸。

            #### 1. 策略流程簡短總結

            1. 每週用過去 `lookback_days` 個交易日作為 formation period。
            2. 對股票池內所有兩兩 pair 的 log price 做雙向 Engle-Granger 共整合檢定。
            3. 保留 `p_value <= pvalue_threshold` 的 pair，並依 `coint_t` 由小到大排序；`coint_t` 越負，代表殘差越明顯拒絕單根假設，共整合關係越強。
            4. 對初選 pair 加上同產業條件與 beta 差距條件，只保留 `same_industry = True` 且 `beta_diff < beta_diff_threshold` 的 pair。
            5. 單一 pair 回測時，只有當該 pair 在當週候選清單中，才允許開新倉。
            6. 進場訊號使用進場當下 formation period 重新估計出的 OLS `alpha` 與 `hedge_ratio` 建立 spread，並將 spread 標準化成 z-score。
            7. 今日收盤產生訊號，下一個交易日開盤成交；出場條件則由均值回歸、停損或最大持有天數決定。

            #### 2. Formation period 與 log price

            在每個週調倉日 $f$，取過去 $L$ 個交易日作為 formation period：

            $$
            \mathcal{F}_f = \{t=f-L, f-L+1, \dots, f-1\}
            $$

            對任意股票 $i$，先將價格轉成 log price：

            $$
            x_{i,t} = \log(P_{i,t})
            $$

            其中 $P_{i,t}$ 為股票 $i$ 在第 $t$ 日的收盤價，$x_{i,t}$ 為其對數價格。使用 log price 的原因是價格比例變化可被轉換成加法形式，較適合用於線性迴歸與 spread 建構。

            #### 3. Engle-Granger 共整合檢定公式

            對任意一組候選 pair $(X,Y)$，本策略在 formation period 內先估計下列 OLS 迴歸：

            $$
            x_{X,t} = \alpha_{XY} + \beta_{XY} x_{Y,t} + \varepsilon_{XY,t}
            $$

            其中：

            | 符號 | 意義 |
            |---|---|
            | $x_{X,t}$ | 股票 $X$ 在第 $t$ 日的 log price |
            | $x_{Y,t}$ | 股票 $Y$ 在第 $t$ 日的 log price |
            | $\alpha_{XY}$ | OLS 截距項 |
            | $\beta_{XY}$ | hedge ratio，也就是用 $Y$ 解釋 $X$ 的斜率係數 |
            | $\varepsilon_{XY,t}$ | 共整合殘差，也就是 spread 的未標準化型態 |

            OLS 估計量可寫為：

            $$
            \hat{\beta}_{XY}
            =
            \frac{
            \sum_{t \in \mathcal{F}_f}
            (x_{Y,t}-\bar{x}_Y)(x_{X,t}-\bar{x}_X)
            }{
            \sum_{t \in \mathcal{F}_f}
            (x_{Y,t}-\bar{x}_Y)^2
            }
            $$

            $$
            \hat{\alpha}_{XY}
            =
            \bar{x}_X - \hat{\beta}_{XY}\bar{x}_Y
            $$

            估計完成後，取得殘差序列：

            $$
            \hat{\varepsilon}_{XY,t}
            =
            x_{X,t}
            -
            \hat{\alpha}_{XY}
            -
            \hat{\beta}_{XY}x_{Y,t}
            $$

            Engle-Granger 方法的重點是檢查殘差 $\hat{\varepsilon}_{XY,t}$ 是否為定態。若殘差為定態，代表 $X$ 與 $Y$ 雖然各自價格可能非定態，但兩者存在穩定的長期均衡關係。殘差的 ADF 檢定可表示為：

            $$
            \Delta \hat{\varepsilon}_t
            =
            \phi \hat{\varepsilon}_{t-1}
            +
            \sum_{l=1}^{q}
            \gamma_l \Delta \hat{\varepsilon}_{t-l}
            +
            u_t
            $$

            假設檢定為：

            $$
            H_0: \phi = 0 \quad \text{殘差具有單根，沒有共整合關係}
            $$

            $$
            H_1: \phi < 0 \quad \text{殘差為定態，存在共整合關係}
            $$

            因此，若 ADF 檢定得到的 `p_value` 越小，代表越有理由拒絕「沒有共整合」的虛無假設。

            #### 4. 雙向共整合檢定與候選 pair 排序

            因為 OLS 迴歸具有方向性，所以本策略會對同一組 pair 做雙向檢定：

            $$
            x_{X,t} = \alpha_{XY} + \beta_{XY}x_{Y,t} + \varepsilon_{XY,t}
            $$

            $$
            x_{Y,t} = \alpha_{YX} + \beta_{YX}x_{X,t} + \varepsilon_{YX,t}
            $$

            分別取得兩個方向的檢定結果：

            $$
            (coint\_t_{XY}, p_{XY}), \quad (coint\_t_{YX}, p_{YX})
            $$

            程式會選擇 p-value 較小的方向作為該 pair 的共整合強度代表：

            $$
            p^*_{XY} = \min(p_{XY}, p_{YX})
            $$

            $$
            coint\_t^*_{XY}
            =
            \begin{cases}
            coint\_t_{XY}, & \text{if } p_{XY} \le p_{YX} \\
            coint\_t_{YX}, & \text{if } p_{YX} < p_{XY}
            \end{cases}
            $$

            初步候選 pair 集合定義為：

            $$
            \mathcal{C}_0
            =
            \left\{
            (X,Y):
            p^*_{XY} \le pvalue\_threshold,
            \; n_{XY} \ge min\_obs
            \right\}
            $$

            接著依照 $coint\_t^*_{XY}$ 由小到大排序：

            $$
            coint\_t^*_{(1)}
            \le
            coint\_t^*_{(2)}
            \le
            \cdots
            \le
            coint\_t^*_{(m)}
            $$

            由於 `coint_t` 越負代表殘差越接近定態，因此排序越前面的 pair 代表共整合關係越強。

            #### 5. Beta 與同產業篩選公式

            為避免兩檔股票暴露於差異過大的市場風險，本策略會估計每檔股票相對大盤的 beta。先計算個股與大盤的 log return：

            $$
            r_{i,t} = \Delta \log(P_{i,t})
            =
            \log(P_{i,t}) - \log(P_{i,t-1})
            $$

            $$
            r_{M,t} = \Delta \log(M_t)
            =
            \log(M_t) - \log(M_{t-1})
            $$

            其中 $M_t$ 為大盤指數價格。股票 $i$ 的 beta 定義為：

            $$
            \beta_i
            =
            \frac{Cov(r_{i,t}, r_{M,t})}{Var(r_{M,t})}
            $$

            對 pair $(X,Y)$，beta 差距為：

            $$
            beta\_diff_{XY}
            =
            |\beta_X - \beta_Y|
            $$

            最終候選 pair 集合為：

            $$
            \mathcal{C}_{final}
            =
            \left\{
            (X,Y) \in \mathcal{C}_0:
            Industry_X = Industry_Y,
            \;
            |\beta_X - \beta_Y| < beta\_diff\_threshold
            \right\}
            $$

            最後再從 $\mathcal{C}_{final}$ 中取排序前 `k_final` 組 pair，作為該週允許交易的候選清單。

            #### 6. Spread 與 z-score 建構

            在單一 pair 回測中，若 pair $(X,Y)$ 在當週候選清單中，策略會使用該週 formation period 重新估計 OLS：

            $$
            x_{X,t} = \alpha + h x_{Y,t} + \varepsilon_t
            $$

            其中 $h$ 即程式中的 `hedge_ratio`。spread 定義為：

            $$
            S_t = x_{X,t} - \alpha - h x_{Y,t}
            $$

            在 formation period 內計算 spread 的平均數與標準差：

            $$
            \mu_S
            =
            \frac{1}{L}
            \sum_{t \in \mathcal{F}_f}
            S_t
            $$

            $$
            \sigma_S
            =
            \sqrt{
            \frac{1}{L}
            \sum_{t \in \mathcal{F}_f}
            (S_t-\mu_S)^2
            }
            $$

            每日交易訊號使用 z-score 衡量 spread 偏離均衡的程度：

            $$
            z_t
            =
            \frac{S_t - \mu_S}{\sigma_S}
            $$

            若 $z_t$ 為正且偏高，代表 $X$ 相對於 $Y$ 偏貴，策略傾向放空 spread；若 $z_t$ 為負且偏低，代表 $X$ 相對於 $Y$ 偏便宜，策略傾向做多 spread。

            #### 7. 進出場方向與部位權重

            設進場方向為 $d_t$：

            $$
            d_t =
            \begin{cases}
            -1, & z_{t-1} \le entry\_z \text{ 且 } entry\_z < z_t < stop\_z \\
            +1, & z_{t-1} \ge -entry\_z \text{ 且 } -stop\_z < z_t < -entry\_z
            \end{cases}
            $$

            其中：

            - $d_t=-1$：spread 過高，放空 spread，也就是放空 $X$、做多 $Y$。
            - $d_t=+1$：spread 過低，做多 spread，也就是做多 $X$、放空 $Y$。

            由於 spread 為 $S_t=x_{X,t}-\alpha-hx_{Y,t}$，未標準化的 pair 權重可寫為：

            $$
            raw_X = 1, \quad raw_Y = -h
            $$

            程式會用 gross exposure 做正規化：

            $$
            G = |1| + |h|
            $$

            因此實際交易權重為：

            $$
            w_X = d_t \frac{1}{G}
            $$

            $$
            w_Y = d_t \frac{-h}{G}
            $$

            若目前資金為 $V_t$，則兩檔股票的下單金額為：

            $$
            Dollar_X = V_t w_X
            $$

            $$
            Dollar_Y = V_t w_Y
            $$

            股數則為：

            $$
            Shares_X = \frac{Dollar_X}{Open_{X,t+1}}
            $$

            $$
            Shares_Y = \frac{Dollar_Y}{Open_{Y,t+1}}
            $$

            也就是今日收盤確認訊號後，在下一個交易日開盤價成交。

            #### 8. 出場條件

            持倉後每天用進場當下估計的 spread 參數持續計算 $z_t$。若符合以下任一條件，則於下一個交易日開盤出場：

            $$
            |z_t| < exit\_z
            \quad \Rightarrow \quad
            \text{均值回歸出場}
            $$

            $$
            |z_t| > stop\_z
            \quad \Rightarrow \quad
            \text{停損出場}
            $$

            $$
            holding\_days \ge max\_holding\_days
            \quad \Rightarrow \quad
            \text{最大持有天數出場}
            $$

            若發生 stop loss，策略會暫停重新進場，直到：

            $$
            |z_t| < reentry\_reset\_z
            $$

            才重新允許該 pair 產生新的進場訊號。這樣可以避免 spread 持續發散時反覆進場，降低連續停損風險。
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
    raw_data_path: str
    backtest_start_date: pd.Timestamp
    backtest_end_date: pd.Timestamp | None
    market_cap_top_n: int
    lookback_days: int
    min_obs: int
    require_positive_alpha: bool
    use_pvalue_filter: bool
    pvalue_threshold: float
    top_n: int
    initial_capital: float
    commission_rate: float
    sell_tax_rate: float

    # yfinance benchmarks，例如 0050.TW。保留 benchmark_ticker 作為舊版相容的第一個 benchmark。
    benchmark_ticker: str
    benchmark_tickers: tuple[str, ...]
    benchmark_auto_adjust: bool

    # 自訂基金 benchmark：預設加入統一奔騰基金 A09012.xlsx。
    fund_benchmark_enabled: bool
    fund_benchmark_name: str
    fund_benchmark_path: str


def strategy3_parse_benchmark_tickers(raw: str) -> tuple[str, ...]:
    tickers = []
    for part in str(raw).replace("；", ",").replace("，", ",").split(","):
        value = part.strip()
        if value and value not in tickers:
            tickers.append(value)
    return tuple(tickers)


def strategy3_sidebar_settings() -> FF5AlphaSettings:
    if st.sidebar.button("返回策略選擇"):
        st.session_state["selected_strategy"] = None
        st.rerun()

    st.sidebar.caption("目前策略：策略3(台股五因子 Alpha 月調倉策略)")
    st.sidebar.divider()

    st.sidebar.header("資料檔案")
    raw_data_path = st.sidebar.text_input(
        "原始資料資料夾",
        value="data/strategy3",
        help="資料夾內需包含 DATA、AD PRICE、ASSET、RMW Excel。可分子資料夾放置。"
    )
    backtest_start_date = st.sidebar.date_input(
        "策略3正式回測起日",
        value=BACKTEST_START_DEFAULT.date(),
        help="策略3 NAV 與 benchmark 績效會從此日之後開始計算。若 rolling lookback 還未滿足，資金會維持現金直到第一個可調倉日。"
    )
    backtest_end_date = st.sidebar.date_input(
        "策略3正式回測結束日",
        value=BACKTEST_END_DEFAULT.date(),
        help="本專案回測期間統一設定為 2019-01-01 到 2026-05-19；若你的資料更新到更後面，可自行調整。"
    )

    st.sidebar.divider()
    st.sidebar.header("策略參數")
    market_cap_top_n = st.sidebar.number_input("每月市值股票池", min_value=50, max_value=300, value=150, step=10)
    lookback_days = st.sidebar.slider("Rolling regression lookback", min_value=120, max_value=504, value=252, step=21)
    min_obs = st.sidebar.slider("Regression minimum observations", min_value=60, max_value=252, value=180, step=10)
    top_n = st.sidebar.slider("Alpha Top N 持股檔數", min_value=1, max_value=50, value=10, step=1)
    require_positive_alpha = st.sidebar.toggle("只保留 alpha > 0", value=False)
    use_pvalue_filter = st.sidebar.toggle("使用 alpha p-value 篩選", value=False)
    pvalue_threshold = st.sidebar.slider("Alpha p-value 門檻", min_value=0.01, max_value=0.30, value=0.10, step=0.01)
    initial_capital = st.sidebar.number_input("初始資金", min_value=10_000, value=1_000_000, step=100_000)
    commission_rate = st.sidebar.number_input("買賣手續費率", min_value=0.0, value=0.001425, step=0.0001, format="%.6f")
    sell_tax_rate = st.sidebar.number_input("賣出交易稅率", min_value=0.0, value=0.0, step=0.0005, format="%.6f")
    st.sidebar.caption("目前執行價固定為：t+1 收盤價。")

    st.sidebar.divider()
    st.sidebar.header("Benchmark")
    benchmark_tickers_text = st.sidebar.text_input(
        "yfinance Benchmark tickers（逗號分隔）",
        value="0050.TW",
        help="例如：0050.TW。若要多個 yfinance benchmark，可用逗號分隔。"
    )
    benchmark_tickers = strategy3_parse_benchmark_tickers(benchmark_tickers_text)
    benchmark_auto_adjust = st.sidebar.toggle("Benchmark 使用 yfinance auto_adjust", value=True)

    fund_benchmark_enabled = st.sidebar.checkbox("加入統一奔騰基金 benchmark", value=True)
    fund_benchmark_name = st.sidebar.text_input("基金 benchmark 名稱", value="統一奔騰基金")
    fund_benchmark_path = st.sidebar.text_input(
        "統一奔騰基金淨值檔路徑",
        value="data/strategy3/A09012.xlsx",
        help="請把 A09012.xlsx 放到 GitHub repo 的 data/strategy3/ 底下，或在這裡填入實際路徑。檔案欄位支援：日期、淨值、漲跌。"
    )

    benchmark_ticker = benchmark_tickers[0] if len(benchmark_tickers) > 0 else ""

    return FF5AlphaSettings(
        raw_data_path=raw_data_path,
        backtest_start_date=pd.Timestamp(backtest_start_date),
        backtest_end_date=pd.Timestamp(backtest_end_date) if backtest_end_date is not None else None,
        market_cap_top_n=int(market_cap_top_n),
        lookback_days=int(lookback_days),
        min_obs=int(min_obs),
        require_positive_alpha=bool(require_positive_alpha),
        use_pvalue_filter=bool(use_pvalue_filter),
        pvalue_threshold=float(pvalue_threshold),
        top_n=int(top_n),
        initial_capital=float(initial_capital),
        commission_rate=float(commission_rate),
        sell_tax_rate=float(sell_tax_rate),
        benchmark_ticker=str(benchmark_ticker).strip(),
        benchmark_tickers=benchmark_tickers,
        benchmark_auto_adjust=bool(benchmark_auto_adjust),
        fund_benchmark_enabled=bool(fund_benchmark_enabled),
        fund_benchmark_name=str(fund_benchmark_name).strip() or "統一奔騰基金",
        fund_benchmark_path=str(fund_benchmark_path).strip(),
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
        # Convenient fallbacks for files such as A09012.xlsx when the sidebar path
        # is kept as data/strategy3/A09012.xlsx but the file is placed directly
        # under data/ or strategy3/.
        app_dir / "data" / p.name,
        cwd / "data" / p.name,
        app_dir / "data" / "strategy3" / p.name,
        cwd / "data" / "strategy3" / p.name,
        app_dir / "strategy3" / p.name,
        cwd / "strategy3" / p.name,
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



# ============================================================
# Strategy 3 Fallback Raw Pipeline
# ============================================================

def strategy3_read_one_raw_file_for_fallback(path: Path) -> pd.DataFrame:
    """Read one CSV/XLS/XLSX file for the fallback pipeline."""
    suffix = path.suffix.lower()

    if suffix == ".csv":
        last_error = None
        for enc in ["utf-8-sig", "utf-8", "big5", "cp950"]:
            try:
                return pd.read_csv(path, encoding=enc)
            except Exception as exc:
                last_error = exc
        raise last_error

    if suffix in [".xlsx", ".xls"]:
        # Try the normal first-row header first. If the sheet has a few title
        # rows above the actual header, try a few alternative header rows.
        errors: list[str] = []
        for header in [0, 1, 2, 3]:
            try:
                df = pd.read_excel(path, header=header)
                df.columns = [str(c).strip() for c in df.columns]
                # Accept this read if it yields at least two non-unnamed columns.
                usable_cols = [
                    c for c in df.columns
                    if c and not c.lower().startswith("unnamed") and c.lower() != "nan"
                ]
                if len(usable_cols) >= 2:
                    return df
            except Exception as exc:
                errors.append(str(exc))
        raise ValueError(f"讀取 Excel 失敗：{path.name}；{errors[-1] if errors else ''}")

    raise ValueError(f"不支援的檔案格式：{path.name}")


def strategy3_classify_raw_file(path: Path) -> str:
    """Classify raw CMoney files by folder/file name."""
    s = str(path).lower().replace("\\", "/")
    name = path.name.lower()

    # Avoid accidentally treating the mutual-fund benchmark NAV file as stock data.
    if "a09012" in s or "benchmark" in s or "奔騰" in s:
        return "ignore"

    if any(k in s for k in ["ad_price", "ad price", "adprice", "adj_price", "adjusted_price", "還原"]):
        return "price"
    if any(k in s for k in ["/asset", "asset", "資產負債"]):
        return "asset"
    if any(k in s for k in ["/rmw", "rmw", "損益", "profit"]):
        return "rmw"
    if "data" in s or "_data" in name:
        return "data"
    if any(k in s for k in ["price", "收盤"]):
        return "price"

    return "other"


def strategy3_collect_raw_files_for_fallback(raw_data_path: str) -> pd.DataFrame:
    """Collect raw files recursively and attach category metadata."""
    base = resolve_strategy3_path(raw_data_path)
    if base.is_file():
        files = [base]
    else:
        files = []
        for pattern in ["*.csv", "*.xlsx", "*.xls"]:
            files.extend(sorted(base.rglob(pattern)))

    files = [f for f in files if not f.name.startswith("~$") and f.suffix.lower() in [".csv", ".xlsx", ".xls"]]

    records = []
    for f in files:
        category = strategy3_classify_raw_file(f)
        if category == "ignore":
            continue
        records.append({
            "category": category,
            "source_file": f.name,
            "source_path": str(f),
            "suffix": f.suffix.lower(),
        })

    return pd.DataFrame(records)


def strategy3_read_raw_category_for_fallback(file_manifest: pd.DataFrame, categories: list[str]) -> pd.DataFrame:
    """Read and concatenate files of selected categories."""
    if file_manifest is None or file_manifest.empty:
        return pd.DataFrame()

    selected = file_manifest[file_manifest["category"].isin(categories)].copy()
    if selected.empty:
        return pd.DataFrame()

    frames = []
    for _, row in selected.iterrows():
        p = Path(row["source_path"])
        try:
            df = strategy3_read_one_raw_file_for_fallback(p)
            if df is None or df.empty:
                continue
            df.columns = [str(c).strip() for c in df.columns]
            df["source_file"] = row["source_file"]
            df["source_category"] = row["category"]
            frames.append(df)
        except Exception as exc:
            # Keep scanning other files. The file manifest will still show the skipped file.
            continue

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True, sort=False)


def strategy3_normalize_price_fallback(price_raw: pd.DataFrame, data_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Build canonical price_df from AD PRICE first; if unavailable, use DATA close price.
    Output columns: date, stock_id, stock_name, adj_close.
    """
    errors: list[str] = []

    for label, raw in [("AD PRICE", price_raw), ("DATA", data_raw)]:
        if raw is None or raw.empty:
            continue
        try:
            df = raw.copy()
            df.columns = [str(c).strip() for c in df.columns]
            df = strategy3_prepare_stock_id_column(df, label)

            date_col = strategy3_find_column(df, [
                "date", "trading_date", "tradedate", "年月日", "日期", "交易日期", "資料日期",
            ])
            if date_col is not None and date_col != "date":
                df = df.rename(columns={date_col: "date"})

            name_col = strategy3_find_column(df, [
                "stock_name", "stockname", "name", "股票名稱", "證券名稱", "證券簡稱", "簡稱", "公司簡稱", "名稱",
            ])
            if name_col is not None and name_col != "stock_name":
                df = df.rename(columns={name_col: "stock_name"})

            close_col = strategy3_find_column(df, [
                "adj_close", "adjusted_close", "adjustedclose", "還原收盤價", "還原收盤價元",
                "收盤價", "收盤價元", "close", "Close", "price", "價格",
            ])
            if close_col is not None and close_col != "adj_close":
                df = df.rename(columns={close_col: "adj_close"})

            required = ["date", "stock_id", "adj_close"]
            missing = [c for c in required if c not in df.columns]
            if missing:
                errors.append(f"{label} 缺少 {missing}；欄位={list(df.columns)}")
                continue

            if "stock_name" not in df.columns:
                df["stock_name"] = ""

            df["date"] = parse_strategy3_date_series(df["date"])
            df["adj_close"] = strategy3_to_num(df["adj_close"])

            out = (
                df[["date", "stock_id", "stock_name", "adj_close"]]
                .dropna(subset=["date", "stock_id", "adj_close"])
                .copy()
            )
            out = out[out["adj_close"] > 0]
            out = out.drop_duplicates(["date", "stock_id"], keep="last")
            out = out.sort_values(["date", "stock_id"]).reset_index(drop=True)

            if not out.empty:
                return out

        except Exception as exc:
            errors.append(f"{label}: {exc}")

    raise ValueError(
        "fallback pipeline 無法建立 price_df。請確認 AD PRICE 或 DATA 檔至少含日期、股票代號、收盤價/還原收盤價。"
        f"讀取錯誤：{errors}"
    )


def strategy3_normalize_characteristics_fallback(data_raw: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    """Create a monthly characteristic table used by the fallback alpha model."""
    if data_raw is None or data_raw.empty:
        # Last resort: build a minimal characteristic table from prices.
        tmp = price_df.copy()
        tmp["market_cap"] = np.nan
        tmp["bm"] = np.nan
        tmp["profitability"] = np.nan
        tmp["asset_growth"] = np.nan
        return tmp[["date", "stock_id", "stock_name", "market_cap", "bm", "profitability", "asset_growth"]]

    df = data_raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = strategy3_prepare_stock_id_column(df, "DATA")

    date_col = strategy3_find_column(df, [
        "date", "trading_date", "tradedate", "年月日", "日期", "交易日期", "資料日期",
    ])
    if date_col is not None and date_col != "date":
        df = df.rename(columns={date_col: "date"})

    name_col = strategy3_find_column(df, [
        "stock_name", "stockname", "name", "股票名稱", "證券名稱", "證券簡稱", "簡稱", "公司簡稱", "名稱",
    ])
    if name_col is not None and name_col != "stock_name":
        df = df.rename(columns={name_col: "stock_name"})

    mcap_col = strategy3_find_column(df, [
        "market_cap", "marketcap", "市值", "市值百萬元", "市值(百萬元)", "總市值", "市場價值", "market_value",
    ])
    pb_col = strategy3_find_column(df, [
        "pb", "pbr", "price_to_book", "股價淨值比", "市淨率", "PB", "P/B",
    ])
    bm_col = strategy3_find_column(df, [
        "bm", "b/m", "book_to_market", "帳面市值比", "淨值市價比", "B/M",
    ])

    if "stock_name" not in df.columns:
        df["stock_name"] = ""

    if "date" not in df.columns:
        raise ValueError(f"DATA 缺少日期欄位；目前欄位={list(df.columns)}")

    df["date"] = parse_strategy3_date_series(df["date"])

    if mcap_col is not None:
        df["market_cap"] = strategy3_to_num(df[mcap_col])
    else:
        df["market_cap"] = np.nan

    if bm_col is not None:
        df["bm"] = strategy3_to_num(df[bm_col])
    elif pb_col is not None:
        pb = strategy3_to_num(df[pb_col])
        df["bm"] = np.where(pb > 0, 1.0 / pb, np.nan)
    else:
        df["bm"] = np.nan

    df["profitability"] = np.nan
    df["asset_growth"] = np.nan

    out = (
        df[["date", "stock_id", "stock_name", "market_cap", "bm", "profitability", "asset_growth"]]
        .dropna(subset=["date", "stock_id"])
        .drop_duplicates(["date", "stock_id"], keep="last")
        .sort_values(["date", "stock_id"])
        .reset_index(drop=True)
    )

    # Fill stock_name from price_df when DATA has no name.
    if "stock_name" in price_df.columns and not price_df.empty:
        name_map = (
            price_df[price_df["stock_name"].astype(str).str.len() > 0]
            .drop_duplicates("stock_id")
            .set_index("stock_id")["stock_name"]
            .to_dict()
        )
        out["stock_name"] = out["stock_name"].where(out["stock_name"].astype(str).str.len() > 0, out["stock_id"].map(name_map).fillna(""))

    return out


def strategy3_safe_weighted_mean(values: pd.Series, weights: pd.Series | None = None) -> float:
    values = pd.to_numeric(values, errors="coerce")
    if weights is None:
        return float(values.mean()) if values.notna().any() else np.nan

    weights = pd.to_numeric(weights, errors="coerce")
    mask = values.notna() & weights.notna() & (weights > 0)
    if not mask.any():
        return float(values.mean()) if values.notna().any() else np.nan

    w = weights.loc[mask]
    v = values.loc[mask]
    total = w.sum()
    if total <= 0:
        return float(v.mean()) if v.notna().any() else np.nan
    return float((v * w).sum() / total)


def strategy3_build_raw_pipeline_fallback(
    raw_data_path: str,
    market_cap_top_n: int,
    lookback_days: int,
    min_obs: int,
    original_error: Exception | None = None,
    start_date: pd.Timestamp | str | None = BACKTEST_START_DEFAULT,
    end_date: pd.Timestamp | str | None = BACKTEST_END_DEFAULT,
) -> dict[str, pd.DataFrame]:
    """
    Fallback when strategy3_ff5_pipeline fails with KeyError('stock_id').

    It builds a robust MKT-based rolling alpha model from CMoney DATA / AD PRICE.
    If the external full FF5 pipeline works, the app still uses it first. This
    fallback is only activated when the upstream pipeline fails, mainly because
    of column-name differences such as stock_id vs 證券代碼 / 股票代號.
    """
    file_manifest = strategy3_collect_raw_files_for_fallback(raw_data_path)

    if file_manifest.empty:
        raise ValueError(f"fallback pipeline 找不到任何 CSV/XLSX/XLS 原始資料：{raw_data_path}")

    price_raw = strategy3_read_raw_category_for_fallback(file_manifest, ["price"])
    data_raw = strategy3_read_raw_category_for_fallback(file_manifest, ["data"])

    # If no explicit AD PRICE files are found, DATA close price will be used.
    price_df = strategy3_normalize_price_fallback(price_raw, data_raw)
    characteristics = strategy3_normalize_characteristics_fallback(data_raw, price_df)

    start_ts = pd.Timestamp(start_date).normalize() if start_date is not None else None
    end_ts = pd.Timestamp(end_date).normalize() if end_date is not None else None
    if start_ts is not None:
        price_df = price_df[price_df["date"] >= start_ts].copy()
        characteristics = characteristics[characteristics["date"] >= start_ts].copy() if "date" in characteristics.columns else characteristics
    if end_ts is not None:
        price_df = price_df[price_df["date"] <= end_ts].copy()
        characteristics = characteristics[characteristics["date"] <= end_ts].copy() if "date" in characteristics.columns else characteristics

    prices_wide = (
        price_df.pivot_table(index="date", columns="stock_id", values="adj_close", aggfunc="last")
        .sort_index()
        .ffill()
    )
    returns_wide = prices_wide.pct_change()

    if prices_wide.empty or returns_wide.empty:
        raise ValueError("fallback pipeline 無法由價格資料建立日報酬。")

    trading_dates = pd.DatetimeIndex(prices_wide.index)
    formation_dates = (
        pd.Series(trading_dates, index=trading_dates)
        .groupby(trading_dates.to_period("M"))
        .last()
        .dropna()
        .tolist()
    )
    formation_dates = [pd.Timestamp(d) for d in formation_dates]

    # Name map.
    name_map = (
        price_df[price_df["stock_name"].astype(str).str.len() > 0]
        .drop_duplicates("stock_id")
        .set_index("stock_id")["stock_name"]
        .to_dict()
    )

    alpha_rows: list[dict[str, object]] = []
    factor_rows: list[dict[str, object]] = []
    formation_rows: list[dict[str, object]] = []
    regression_panel_rows: list[pd.DataFrame] = []

    for fdate in formation_dates:
        if fdate not in returns_wide.index:
            continue

        # Use the latest characteristics available at or before the formation date.
        char_hist = characteristics[characteristics["date"] <= fdate].copy()
        if char_hist.empty:
            available = prices_wide.loc[:fdate].dropna(axis=1, how="all").columns.tolist()
            char_latest = pd.DataFrame({
                "stock_id": available,
                "stock_name": [name_map.get(s, "") for s in available],
                "market_cap": np.nan,
                "bm": np.nan,
                "profitability": np.nan,
                "asset_growth": np.nan,
            })
        else:
            char_latest = (
                char_hist.sort_values(["stock_id", "date"])
                .drop_duplicates("stock_id", keep="last")
                .copy()
            )

        # Keep only stocks with price data by the formation date.
        existing_stocks = prices_wide.loc[:fdate].dropna(axis=1, how="all").columns.astype(str).tolist()
        char_latest = char_latest[char_latest["stock_id"].isin(existing_stocks)].copy()

        if char_latest.empty:
            continue

        if "market_cap" in char_latest.columns and char_latest["market_cap"].notna().any():
            char_latest = char_latest.sort_values("market_cap", ascending=False)
        else:
            # Fallback ranking: stocks with more available price observations are preferred.
            obs_count = prices_wide.loc[:fdate].notna().sum().rename("price_obs")
            char_latest["price_obs"] = char_latest["stock_id"].map(obs_count).fillna(0)
            char_latest = char_latest.sort_values("price_obs", ascending=False)

        universe = char_latest.head(int(market_cap_top_n)).copy()
        if universe.empty:
            continue

        formation_rows.append(universe.assign(formation_date=fdate))

        universe_stocks = universe["stock_id"].astype(str).tolist()
        window_returns = returns_wide.loc[:fdate, universe_stocks].tail(int(lookback_days)).copy()
        window_returns = window_returns.dropna(how="all")

        if len(window_returns) < int(min_obs):
            continue

        mcap_map = universe.set_index("stock_id")["market_cap"].to_dict() if "market_cap" in universe.columns else {}
        weights = pd.Series({sid: mcap_map.get(sid, np.nan) for sid in universe_stocks}, dtype=float)
        if weights.notna().sum() == 0 or (weights.fillna(0) > 0).sum() == 0:
            weights = pd.Series(1.0, index=universe_stocks)
        weights = weights.reindex(universe_stocks).fillna(0.0)
        if weights.sum() <= 0:
            weights = pd.Series(1.0, index=universe_stocks)
        weights = weights / weights.sum()

        mkt_ret = window_returns.mul(weights, axis=1).sum(axis=1, min_count=1)
        factor_df = pd.DataFrame({
            "date": window_returns.index,
            "formation_date": fdate,
            "MKT_RF": mkt_ret.values,
            "SMB": 0.0,
            "HML": 0.0,
            "RMW": 0.0,
            "CMA": 0.0,
            "RF": 0.0,
        }).dropna(subset=["MKT_RF"])

        if factor_df.empty:
            continue

        factor_rows.extend(factor_df.to_dict("records"))

        # Regression per stock: Ri,t = alpha + beta_mkt * MKT_t + error.
        x = sm.add_constant(factor_df.set_index("date")["MKT_RF"], has_constant="add")

        for sid in universe_stocks:
            if sid not in window_returns.columns:
                continue

            y = window_returns[sid].reindex(x.index)
            reg = pd.concat([y.rename("stock_return"), x], axis=1).dropna()

            if len(reg) < int(min_obs):
                continue

            try:
                model = sm.OLS(reg["stock_return"], reg[["const", "MKT_RF"]]).fit()
            except Exception:
                continue

            alpha = float(model.params.get("const", np.nan))
            beta_mkt = float(model.params.get("MKT_RF", np.nan))
            pvalue_alpha = float(model.pvalues.get("const", np.nan)) if hasattr(model, "pvalues") else np.nan

            alpha_rows.append({
                "formation_date": fdate,
                "stock_id": sid,
                "stock_name": name_map.get(sid, ""),
                "alpha": alpha,
                "alpha_annualized": alpha * 252.0 if pd.notna(alpha) else np.nan,
                "pvalue_alpha": pvalue_alpha,
                "beta_mkt": beta_mkt,
                "beta_smb": 0.0,
                "beta_hml": 0.0,
                "beta_rmw": 0.0,
                "beta_cma": 0.0,
                "r2": float(model.rsquared) if hasattr(model, "rsquared") else np.nan,
                "adj_r2": float(model.rsquared_adj) if hasattr(model, "rsquared_adj") else np.nan,
                "n_obs": int(len(reg)),
                "rank_in_top150": int(universe_stocks.index(sid) + 1),
                "fallback_pipeline": True,
            })

            tmp = reg.reset_index().rename(columns={"index": "date"})
            tmp["formation_date"] = fdate
            tmp["stock_id"] = sid
            regression_panel_rows.append(tmp)

    alpha_scores = pd.DataFrame(alpha_rows)
    if not alpha_scores.empty:
        alpha_scores["alpha_rank"] = (
            alpha_scores.groupby("formation_date")["alpha"]
            .rank(method="first", ascending=False)
            .astype(int)
        )
        alpha_scores = alpha_scores.sort_values(["formation_date", "alpha"], ascending=[True, False]).reset_index(drop=True)

    ff5 = pd.DataFrame(factor_rows)
    if not ff5.empty:
        ff5 = ff5.drop_duplicates(["date", "formation_date"], keep="last").sort_values(["date", "formation_date"]).reset_index(drop=True)

    formations = pd.concat(formation_rows, ignore_index=True) if formation_rows else pd.DataFrame()
    regression_panel = pd.concat(regression_panel_rows, ignore_index=True) if regression_panel_rows else pd.DataFrame()

    file_manifest_out = file_manifest.copy()
    file_manifest_out["fallback_used"] = True
    file_manifest_out["original_pipeline_error"] = repr(original_error) if original_error is not None else ""

    if alpha_scores.empty:
        raise ValueError(
            "fallback pipeline 已成功讀取價格資料，但沒有產生 alpha_scores。"
            "請檢查 lookback/min obs 是否過高，或資料期間是否足夠。"
            f"原始 pipeline 錯誤：{repr(original_error)}"
        )

    return {
        "alpha_scores": alpha_scores,
        "price_df": price_df,
        "ff5": ff5,
        "formations": formations,
        "factor_panel": ff5.copy(),
        "regression_panel": regression_panel,
        "file_manifest": file_manifest_out,
        "fallback_warning": pd.DataFrame([{
            "message": f"外部 strategy3_ff5_pipeline 失敗，因此本次使用內建 fallback pipeline。fallback 以市場因子 MKT rolling alpha 估計，SMB/HML/RMW/CMA 置為 0。原始錯誤：{repr(original_error)}",
            "original_error": repr(original_error),
        }]),
    }



@st.cache_data(show_spinner=False)
def strategy3_build_raw_pipeline_cached(
    raw_data_path: str,
    market_cap_top_n: int,
    lookback_days: int,
    min_obs: int,
    start_date: pd.Timestamp | str | None = BACKTEST_START_DEFAULT,
    end_date: pd.Timestamp | str | None = BACKTEST_END_DEFAULT,
) -> dict[str, pd.DataFrame]:
    """
    Build Strategy 3 raw outputs.

    First tries the external strategy3_ff5_pipeline. If that pipeline fails with
    KeyError('stock_id') or another column-name related issue, automatically
    falls back to an internal robust pipeline that normalizes CMoney column
    names such as 證券代碼 / 股票代號 into stock_id.
    """
    raw_dir = resolve_strategy3_path(raw_data_path)
    config = FF5RawBuildConfig(
        market_cap_top_n=int(market_cap_top_n),
        lookback_days=int(lookback_days),
        min_obs=int(min_obs),
        start_date=start_date,
        end_date=end_date,
    )

    try:
        raw_result = run_ff5_raw_pipeline(raw_dir, config)
        if not isinstance(raw_result, dict):
            raise ValueError("strategy3_ff5_pipeline 回傳格式不是 dict。")

        # Validate early, so downstream does not fail with a vague KeyError.
        if "alpha_scores" in raw_result:
            raw_result["alpha_scores"] = strategy3_normalize_alpha(raw_result["alpha_scores"])
        if "price_df" in raw_result:
            raw_result["price_df"] = strategy3_normalize_price(raw_result["price_df"])

        required = ["alpha_scores", "price_df"]
        missing = [k for k in required if k not in raw_result]
        if missing:
            raise ValueError(f"strategy3_ff5_pipeline 缺少輸出：{missing}")

        # Add optional keys if upstream does not provide them.
        for k in ["ff5", "formations", "factor_panel", "regression_panel", "file_manifest"]:
            if k not in raw_result:
                raw_result[k] = pd.DataFrame()

        return raw_result

    except Exception as exc:
        msg = str(exc)
        # This is the exact error currently observed in the app.
        # We also fallback for common column-name errors because CMoney exports
        # often differ by version.
        should_fallback = (
            isinstance(exc, KeyError)
            or "stock_id" in msg
            or "股票代號" in msg
            or "證券代碼" in msg
            or "缺少" in msg
        )

        if not should_fallback:
            raise

        return strategy3_build_raw_pipeline_fallback(
            raw_data_path=raw_data_path,
            market_cap_top_n=market_cap_top_n,
            lookback_days=lookback_days,
            min_obs=min_obs,
            original_error=exc,
            start_date=start_date,
            end_date=end_date,
        )

def strategy3_filter_alpha_scores(alpha: pd.DataFrame, settings: FF5AlphaSettings) -> pd.DataFrame:
    filtered = strategy3_normalize_alpha(alpha)
    if settings.require_positive_alpha:
        filtered = filtered[filtered["alpha"] > 0].copy()
    if settings.use_pvalue_filter and "pvalue_alpha" in filtered.columns:
        filtered = filtered[filtered["pvalue_alpha"] < settings.pvalue_threshold].copy()
    return filtered.sort_values(["formation_date", "alpha"], ascending=[True, False]).reset_index(drop=True)



def strategy3_canonical_col_key(col: object) -> str:
    """Return a normalized key for loose column matching."""
    return re.sub(r"[\s_\-()（）/\\.\[\]【】]+", "", str(col).strip().lower())


def strategy3_find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Find a column by exact/loose aliases."""
    if df is None or (df.empty and len(df.columns) == 0):
        return None

    exact_map = {str(c).strip(): c for c in df.columns}
    for cand in candidates:
        if cand in exact_map:
            return exact_map[cand]

    normalized_map = {strategy3_canonical_col_key(c): c for c in df.columns}
    for cand in candidates:
        key = strategy3_canonical_col_key(cand)
        if key in normalized_map:
            return normalized_map[key]

    # Some CMoney columns include units or annotations, e.g. 收盤價(元), 證券代碼/名稱.
    # Try partial matching only after exact matching fails.
    for cand in candidates:
        key = strategy3_canonical_col_key(cand)
        if not key:
            continue
        for col in df.columns:
            col_key = strategy3_canonical_col_key(col)
            if key and (key in col_key or col_key in key):
                return col

    return None


def strategy3_prepare_stock_id_column(df: pd.DataFrame, context: str) -> pd.DataFrame:
    """
    Ensure a DataFrame has a canonical stock_id column.

    This version is intentionally defensive because CMoney / upstream pipeline
    outputs may use different stock-code column names, may put stock codes in
    the index, or may combine code and name in a single string such as
    "2330 台積電".
    """
    out = df.copy()

    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [
            "_".join([str(x).strip() for x in col if str(x).strip() and str(x).strip().lower() != "nan"])
            for col in out.columns
        ]
    else:
        out.columns = [str(c).strip() for c in out.columns]

    # Some pipeline outputs may keep stock code in the index.
    if "stock_id" not in out.columns and out.index.name:
        index_key = strategy3_canonical_col_key(out.index.name)
        if index_key in {
            "stockid", "stockcode", "stock", "code", "ticker", "symbol",
            "股票代號", "股票代碼", "證券代碼", "證券代號", "代號", "公司代號", "股號",
        }:
            old_index_name = out.index.name
            out = out.reset_index().rename(columns={old_index_name: "stock_id"})

    if "stock_id" not in out.columns:
        stock_col = strategy3_find_column(out, [
            "stock_id", "stockid", "stock_code", "stockcode", "stock", "code", "ticker", "symbol",
            "security_id", "securityid", "證券代碼", "證券代號", "股票代號", "股票代碼",
            "證券碼", "代號", "股號", "公司代號", "簡稱代號", "證券代碼/名稱",
            "證券代號名稱", "股票代號名稱", "代碼名稱", "證券簡稱",
        ])
        if stock_col is not None:
            out = out.rename(columns={stock_col: "stock_id"})

    if "stock_id" not in out.columns:
        raise ValueError(
            f"{context} 缺少 stock_id 欄位。請確認資料處理 pipeline 輸出含股票代號。"
            f"目前欄位：{list(out.columns)}"
        )

    out["stock_id"] = clean_stock_id(out["stock_id"])
    out = out[out["stock_id"].notna() & out["stock_id"].astype(str).str.fullmatch(r"\d{4}")].copy()
    return out

def strategy3_normalize_alpha(alpha_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize alpha_scores from strategy3_ff5_pipeline into canonical columns.

    Required output:
    - formation_date
    - stock_id
    - alpha

    This function is intentionally permissive because the upstream FF5 pipeline
    or CMoney exports may use names such as 股票代號, 證券代碼, stock_code,
    rebalance_date, rolling_alpha, alpha_daily, etc.
    """
    if alpha_raw is None or alpha_raw.empty:
        return pd.DataFrame(columns=["formation_date", "stock_id", "alpha"])

    df = alpha_raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = strategy3_prepare_stock_id_column(df, "alpha_scores")

    formation_col = strategy3_find_column(df, [
        "formation_date", "formationdate", "rebalance_date", "rebalancedate",
        "holding_start", "holdingstart", "date", "日期", "形成日", "調倉日", "月調倉日",
    ])
    if formation_col is not None and formation_col != "formation_date":
        df = df.rename(columns={formation_col: "formation_date"})

    name_col = strategy3_find_column(df, [
        "stock_name", "stockname", "name", "股票名稱", "證券名稱", "簡稱", "公司簡稱", "名稱",
    ])
    if name_col is not None and name_col != "stock_name":
        df = df.rename(columns={name_col: "stock_name"})

    alpha_col = strategy3_find_column(df, [
        "alpha", "alpha_daily", "daily_alpha", "rolling_alpha", "ff5_alpha",
        "五因子alpha", "五因子Alpha", "Alpha",
    ])
    if alpha_col is not None and alpha_col != "alpha":
        df = df.rename(columns={alpha_col: "alpha"})

    # If the pipeline only returns annualized alpha, recover daily alpha for ranking/backtest.
    annualized_col = strategy3_find_column(df, [
        "alpha_annualized", "annualized_alpha", "annual_alpha",
        "年化alpha", "年化Alpha", "年化ALPHA",
    ])
    if annualized_col is not None and annualized_col != "alpha_annualized":
        df = df.rename(columns={annualized_col: "alpha_annualized"})

    if "alpha" not in df.columns and "alpha_annualized" in df.columns:
        df["alpha"] = strategy3_to_num(df["alpha_annualized"]) / 252.0

    pvalue_col = strategy3_find_column(df, [
        "pvalue_alpha", "p_value_alpha", "pvalue", "p_value", "alpha_pvalue", "alpha_p_value", "p值", "P值",
    ])
    if pvalue_col is not None and pvalue_col != "pvalue_alpha":
        df = df.rename(columns={pvalue_col: "pvalue_alpha"})

    rank_col = strategy3_find_column(df, [
        "alpha_rank", "rank", "排名", "alpha排名", "Alpha排名",
    ])
    if rank_col is not None and rank_col != "alpha_rank":
        df = df.rename(columns={rank_col: "alpha_rank"})

    required = ["formation_date", "stock_id", "alpha"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"alpha_scores 缺少必要欄位：{missing}。目前欄位：{list(df.columns)}"
        )

    df["formation_date"] = parse_strategy3_date_series(df["formation_date"])
    if "holding_start" in df.columns:
        df["holding_start"] = parse_strategy3_date_series(df["holding_start"])
    if "holding_end" in df.columns:
        df["holding_end"] = parse_strategy3_date_series(df["holding_end"])

    for c in [
        "alpha", "alpha_annualized", "pvalue_alpha", "alpha_rank",
        "rank_in_top150", "beta_mkt", "beta_smb", "beta_hml", "beta_rmw", "beta_cma",
        "r2", "adj_r2", "n_obs"
    ]:
        if c in df.columns:
            df[c] = strategy3_to_num(df[c])

    if "alpha_annualized" not in df.columns and "alpha" in df.columns:
        df["alpha_annualized"] = df["alpha"] * 252.0

    df = df.dropna(subset=["formation_date", "stock_id", "alpha"])
    df = df.drop_duplicates(["formation_date", "stock_id"], keep="last")
    return df.sort_values(["formation_date", "alpha"], ascending=[True, False]).reset_index(drop=True)


def strategy3_normalize_price(price_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize price_df from strategy3_ff5_pipeline into canonical columns.

    Required output:
    - date
    - stock_id
    - adj_close
    """
    if price_raw is None or price_raw.empty:
        return pd.DataFrame(columns=["date", "stock_id", "stock_name", "adj_close"])

    df = price_raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = strategy3_prepare_stock_id_column(df, "price_df / AD PRICE")

    date_col = strategy3_find_column(df, [
        "date", "trading_date", "tradedate", "年月日", "日期", "交易日期", "資料日期",
    ])
    if date_col is not None and date_col != "date":
        df = df.rename(columns={date_col: "date"})

    name_col = strategy3_find_column(df, [
        "stock_name", "stockname", "name", "股票名稱", "證券名稱", "簡稱", "公司簡稱", "名稱",
    ])
    if name_col is not None and name_col != "stock_name":
        df = df.rename(columns={name_col: "stock_name"})

    close_col = strategy3_find_column(df, [
        "adj_close", "adjusted_close", "adjustedclose", "還原收盤價", "還原收盤價元",
        "收盤價", "收盤價元", "close", "Close", "price", "價格",
    ])
    if close_col is not None and close_col != "adj_close":
        df = df.rename(columns={close_col: "adj_close"})

    if "stock_name" not in df.columns:
        df["stock_name"] = ""

    required = ["date", "stock_id", "adj_close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"price_df / AD PRICE 缺少必要欄位：{missing}。目前欄位：{list(df.columns)}"
        )

    df["date"] = parse_strategy3_date_series(df["date"])
    df["adj_close"] = strategy3_to_num(df["adj_close"])

    if "volume" in df.columns:
        df["volume"] = strategy3_to_num(df["volume"])
    if "trading_value_thousand" in df.columns:
        df["trading_value_thousand"] = strategy3_to_num(df["trading_value_thousand"])

    df = df.dropna(subset=["date", "stock_id", "adj_close"])
    df = df[df["adj_close"] > 0].copy()
    df = df.drop_duplicates(["date", "stock_id"], keep="last")
    return df.sort_values(["date", "stock_id"]).reset_index(drop=True)


def strategy3_next_trading_date(trading_dates: pd.DatetimeIndex, current_date: pd.Timestamp) -> pd.Timestamp | pd.NaT:
    future = trading_dates[trading_dates > current_date]
    if len(future) == 0:
        return pd.NaT
    return pd.Timestamp(future[0])


def strategy3_create_positions(alpha: pd.DataFrame, price_df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    alpha = strategy3_normalize_alpha(alpha)
    price_df = strategy3_normalize_price(price_df)
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



def strategy3_latest_month_holdings(positions: pd.DataFrame) -> pd.DataFrame:
    """Return the most recent monthly holding list from the generated positions table."""
    if positions is None or positions.empty:
        return pd.DataFrame()

    df = positions.copy()
    date_col = "formation_date" if "formation_date" in df.columns else "holding_start" if "holding_start" in df.columns else None
    if date_col is None:
        return df.reset_index(drop=True)

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    latest_date = df[date_col].max()
    if pd.isna(latest_date):
        return pd.DataFrame()

    latest = df[df[date_col] == latest_date].copy()

    sort_cols = [c for c in ["position_rank", "alpha_rank", "rank_in_top150"] if c in latest.columns]
    if sort_cols:
        latest = latest.sort_values(sort_cols, ascending=True)
    elif "alpha" in latest.columns:
        latest = latest.sort_values("alpha", ascending=False)

    preferred_cols = [
        "formation_date", "holding_start", "holding_end",
        "position_rank", "stock_id", "stock_name", "target_weight",
        "alpha", "alpha_annualized", "pvalue_alpha", "alpha_rank", "rank_in_top150",
        "beta_mkt", "beta_smb", "beta_hml", "beta_rmw", "beta_cma",
        "r2", "adj_r2", "n_obs",
    ]
    display_cols = [c for c in preferred_cols if c in latest.columns]
    if display_cols:
        latest = latest[display_cols]

    return latest.reset_index(drop=True)

def strategy3_run_backtest(
    positions: pd.DataFrame,
    price_df: pd.DataFrame,
    initial_capital: float,
    commission_rate: float,
    sell_tax_rate: float,
    backtest_start_date: pd.Timestamp | None = None,
    backtest_end_date: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    price_df = strategy3_normalize_price(price_df)
    positions = positions.copy()
    positions = strategy3_prepare_stock_id_column(positions, "positions")
    if "holding_start" not in positions.columns:
        raise ValueError(f"positions 缺少 holding_start 欄位，目前欄位：{list(positions.columns)}")
    positions["holding_start"] = parse_strategy3_date_series(positions["holding_start"])
    if "target_weight" not in positions.columns:
        raise ValueError(f"positions 缺少 target_weight 欄位，目前欄位：{list(positions.columns)}")

    price_lookup = price_df.set_index(["date", "stock_id"])["adj_close"].sort_index()
    name_lookup = price_df.drop_duplicates("stock_id").set_index("stock_id")["stock_name"].to_dict()
    trading_dates = pd.DatetimeIndex(sorted(price_df["date"].dropna().unique()))

    rebalance_dates = pd.DatetimeIndex(sorted(positions["holding_start"].dropna().unique()))

    if backtest_start_date is not None:
        start_ts = pd.Timestamp(backtest_start_date).normalize()
    else:
        start_ts = rebalance_dates[0] if len(rebalance_dates) > 0 else pd.NaT

    end_ts = pd.Timestamp(backtest_end_date).normalize() if backtest_end_date is not None else None

    if pd.notna(start_ts):
        rebalance_dates = rebalance_dates[rebalance_dates >= start_ts]
        backtest_dates = trading_dates[trading_dates >= start_ts]
    else:
        backtest_dates = trading_dates

    if end_ts is not None:
        rebalance_dates = rebalance_dates[rebalance_dates <= end_ts]
        backtest_dates = backtest_dates[backtest_dates <= end_ts]

    rebalance_set = set(rebalance_dates)

    if len(rebalance_dates) == 0:
        raise ValueError("沒有可用 holding_start，無法回測。請確認資料期間、rolling lookback 與正式回測起日。")

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
    initial_capital: float | None = None,
) -> pd.DataFrame:
    """
    下載 yfinance benchmark 價格。

    回傳欄位固定為：
    - date
    - benchmark_close

    initial_capital 參數保留作為舊版相容；實際 benchmark NAV 會在
    strategy3_attach_benchmarks() 對齊策略日期後重新 rebased。
    """
    ticker = str(ticker).strip()
    if not ticker:
        return pd.DataFrame()

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

    if "Close" not in raw.columns:
        return pd.DataFrame()

    bench = raw.reset_index().rename(columns={"Date": "date", "Close": "benchmark_close"})
    bench["date"] = strategy3_force_datetime_ns(bench["date"])
    bench["benchmark_close"] = strategy3_to_num(bench["benchmark_close"])
    bench = (
        bench[["date", "benchmark_close"]]
        .dropna(subset=["date", "benchmark_close"])
        .sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )
    return bench


def strategy3_benchmark_slug(label: str) -> str:
    """
    將 benchmark 名稱轉成安全欄位後綴。
    例：0050.TW -> 0050_TW；統一奔騰基金 -> 統一奔騰基金。
    """
    label = str(label).strip()
    slug = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "_", label).strip("_")
    return slug or "benchmark"


def strategy3_unique_benchmark_specs(settings: FF5AlphaSettings) -> list[dict[str, str]]:
    specs: list[dict[str, str]] = []

    for ticker in settings.benchmark_tickers:
        ticker = str(ticker).strip()
        if not ticker:
            continue
        specs.append({
            "source": "yfinance",
            "label": ticker,
            "ticker": ticker,
            "path": "",
        })

    if settings.fund_benchmark_enabled and str(settings.fund_benchmark_path).strip():
        specs.append({
            "source": "fund_file",
            "label": settings.fund_benchmark_name or "統一奔騰基金",
            "ticker": "",
            "path": settings.fund_benchmark_path,
        })

    used: dict[str, int] = {}
    out: list[dict[str, str]] = []
    for spec in specs:
        base_slug = strategy3_benchmark_slug(spec["label"])
        count = used.get(base_slug, 0) + 1
        used[base_slug] = count
        slug = base_slug if count == 1 else f"{base_slug}_{count}"
        spec = dict(spec)
        spec["slug"] = slug
        out.append(spec)

    return out


def strategy3_benchmark_label_text(settings: FF5AlphaSettings) -> str:
    labels = [spec["label"] for spec in strategy3_unique_benchmark_specs(settings)]
    return "、".join(labels) if labels else "無 benchmark"


@st.cache_data(show_spinner=False)
def strategy3_read_fund_benchmark_file(path_str: str) -> pd.DataFrame:
    """
    讀取基金淨值檔，預設支援 CMoney / 投信常見欄位：
    - 日期
    - 淨值
    - 漲跌

    只會使用「日期」與「淨值」建立 benchmark_close。
    """
    raw = strategy3_read_csv_file(path_str).copy()
    if raw.empty:
        return pd.DataFrame(columns=["date", "benchmark_close"])

    raw.columns = [str(c).strip() for c in raw.columns]
    normalized_map = {
        str(c).strip().lower().replace(" ", "").replace("_", ""): c
        for c in raw.columns
    }

    date_candidates = [
        "date", "日期", "淨值日期", "資料日期", "navdate", "valuationdate"
    ]
    nav_candidates = [
        "benchmarkclose", "nav", "淨值", "單位淨值", "基金淨值", "netassetvalue", "close"
    ]

    date_col = None
    for key in date_candidates:
        if key.lower().replace(" ", "").replace("_", "") in normalized_map:
            date_col = normalized_map[key.lower().replace(" ", "").replace("_", "")]
            break

    nav_col = None
    for key in nav_candidates:
        if key.lower().replace(" ", "").replace("_", "") in normalized_map:
            nav_col = normalized_map[key.lower().replace(" ", "").replace("_", "")]
            break

    if date_col is None and len(raw.columns) >= 1:
        date_col = raw.columns[0]
    if nav_col is None and len(raw.columns) >= 2:
        nav_col = raw.columns[1]

    if date_col is None or nav_col is None:
        raise ValueError(
            f"基金 benchmark 檔案缺少日期或淨值欄位。目前欄位：{list(raw.columns)}"
        )

    bench = pd.DataFrame({
        "date": parse_strategy3_date_series(raw[date_col]),
        "benchmark_close": strategy3_to_num(raw[nav_col]),
    })

    bench = (
        bench.dropna(subset=["date", "benchmark_close"])
        .sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )

    if bench.empty:
        raise ValueError("基金 benchmark 檔案讀取後沒有有效的日期/淨值資料。")

    return bench


def strategy3_align_single_benchmark(
    nav_df: pd.DataFrame,
    bench: pd.DataFrame,
    label: str,
    slug: str,
    initial_capital: float,
) -> pd.DataFrame:
    """
    將單一 benchmark 的 close/NAV 對齊策略日期。

    共同基金不是每天都有報價；這裡使用 merge_asof(direction='backward')，
    代表若策略日期當天沒有基金淨值，就沿用最近一筆可得基金淨值。
    """
    left = nav_df[["date"]].copy().sort_values("date").reset_index(drop=True)
    left["date"] = strategy3_force_datetime_ns(left["date"])

    prefix = f"benchmark_{slug}"

    if bench is None or bench.empty or not {"date", "benchmark_close"}.issubset(set(bench.columns)):
        aligned = left.copy()
        for c in ["close", "daily_return", "nav", "cum_return", "drawdown"]:
            aligned[f"{prefix}_{c}"] = np.nan
        return aligned

    right = bench[["date", "benchmark_close"]].copy().sort_values("date").reset_index(drop=True)
    right["date"] = strategy3_force_datetime_ns(right["date"])
    right["benchmark_close"] = strategy3_to_num(right["benchmark_close"])
    right = right.dropna(subset=["date", "benchmark_close"])

    if right.empty:
        aligned = left.copy()
        for c in ["close", "daily_return", "nav", "cum_return", "drawdown"]:
            aligned[f"{prefix}_{c}"] = np.nan
        return aligned

    aligned = pd.merge_asof(
        left,
        right,
        on="date",
        direction="backward",
    )

    aligned = aligned.rename(columns={"benchmark_close": f"{prefix}_close"})
    close_col = f"{prefix}_close"
    ret_col = f"{prefix}_daily_return"
    nav_col = f"{prefix}_nav"
    cum_col = f"{prefix}_cum_return"
    running_max_col = f"{prefix}_running_max"
    dd_col = f"{prefix}_drawdown"

    close = aligned[close_col].astype(float)
    ret = close.pct_change(fill_method=None)
    first_valid_idx = close.first_valid_index()
    if first_valid_idx is not None:
        ret.loc[first_valid_idx] = 0.0

    aligned[ret_col] = ret
    aligned[nav_col] = initial_capital * (1 + aligned[ret_col].fillna(0)).cumprod()
    aligned.loc[close.isna(), nav_col] = np.nan
    aligned[cum_col] = aligned[nav_col] / initial_capital - 1
    aligned[running_max_col] = aligned[nav_col].cummax()
    aligned[dd_col] = aligned[nav_col] / aligned[running_max_col] - 1

    return aligned[[
        "date",
        close_col,
        ret_col,
        nav_col,
        cum_col,
        dd_col,
    ]]


def strategy3_attach_benchmarks(nav_df: pd.DataFrame, settings: FF5AlphaSettings) -> pd.DataFrame:
    """
    同時加入多個 benchmark。

    目前支援：
    1. yfinance benchmark：例如 0050.TW
    2. 基金淨值檔 benchmark：預設 data/strategy3/A09012.xlsx，也就是統一奔騰基金

    輸出欄位：
    - benchmark_0050_TW_nav / benchmark_0050_TW_daily_return / ...
    - benchmark_統一奔騰基金_nav / benchmark_統一奔騰基金_daily_return / ...
    - benchmark_nav 等舊版欄位會指向第一個成功附加的 benchmark，方便舊圖表或舊邏輯相容。
    """
    out = nav_df.copy()
    out["date"] = strategy3_force_datetime_ns(out["date"])
    out = out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    if out.empty:
        return out

    specs = strategy3_unique_benchmark_specs(settings)
    first_attached: tuple[str, str] | None = None

    # 多抓幾天，避免策略第一天不是 benchmark 交易日或基金淨值日導致缺值。
    start = (out["date"].min() - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    end = out["date"].max().strftime("%Y-%m-%d")

    for spec in specs:
        label = spec["label"]
        slug = spec["slug"]
        source = spec["source"]

        try:
            if source == "yfinance":
                bench = strategy3_download_benchmark(
                    spec["ticker"],
                    start,
                    end,
                    settings.benchmark_auto_adjust,
                    settings.initial_capital,
                )
            elif source == "fund_file":
                bench = strategy3_read_fund_benchmark_file(spec["path"])
                bench = bench[
                    (bench["date"] >= pd.Timestamp(start)) &
                    (bench["date"] <= pd.Timestamp(end) + pd.Timedelta(days=5))
                ].copy()
            else:
                bench = pd.DataFrame()
        except Exception as exc:
            st.warning(f"Benchmark「{label}」讀取失敗：{exc}")
            bench = pd.DataFrame()

        aligned = strategy3_align_single_benchmark(
            nav_df=out[["date"]],
            bench=bench,
            label=label,
            slug=slug,
            initial_capital=settings.initial_capital,
        )

        out = pd.merge(out, aligned, on="date", how="left")

        prefix = f"benchmark_{slug}"
        nav_col = f"{prefix}_nav"
        ret_col = f"{prefix}_daily_return"

        if nav_col in out.columns and out[nav_col].notna().any():
            out[f"active_{slug}_daily_return"] = out["daily_return"] - out[ret_col]
            out[f"active_{slug}_cum_return"] = out["cum_return"] - out[f"{prefix}_cum_return"]
            out[f"relative_nav_{slug}"] = out["portfolio_nav"] / out[nav_col]

            if first_attached is None:
                first_attached = (label, slug)

    if first_attached is None:
        for c in [
            "benchmark_close", "benchmark_daily_return", "benchmark_nav",
            "benchmark_cum_return", "benchmark_drawdown",
            "active_daily_return", "active_cum_return", "relative_nav"
        ]:
            out[c] = np.nan
        return out

    first_label, first_slug = first_attached
    first_prefix = f"benchmark_{first_slug}"

    # 舊版欄位：指向第一個成功 benchmark，通常是 0050.TW。
    out["benchmark_close"] = out[f"{first_prefix}_close"]
    out["benchmark_daily_return"] = out[f"{first_prefix}_daily_return"]
    out["benchmark_nav"] = out[f"{first_prefix}_nav"]
    out["benchmark_cum_return"] = out[f"{first_prefix}_cum_return"]
    out["benchmark_drawdown"] = out[f"{first_prefix}_drawdown"]
    out["active_daily_return"] = out[f"active_{first_slug}_daily_return"]
    out["active_cum_return"] = out[f"active_{first_slug}_cum_return"]
    out["relative_nav"] = out[f"relative_nav_{first_slug}"]

    return out


def strategy3_attach_benchmark(nav_df: pd.DataFrame, settings: FF5AlphaSettings) -> pd.DataFrame:
    """舊版函數名稱相容 wrapper；實際會附加所有 settings 中啟用的 benchmarks。"""
    return strategy3_attach_benchmarks(nav_df, settings)

def strategy3_calc_performance(nav_df: pd.DataFrame, trades_df: pd.DataFrame, rebalance_df: pd.DataFrame, settings: FF5AlphaSettings) -> pd.DataFrame:
    def _one(nav_col: str, ret_col: str, label: str) -> dict[str, object]:
        valid = nav_df[[nav_col, ret_col]].dropna(subset=[nav_col]).copy()
        if valid.empty:
            return {
                "name": label,
                "initial_capital": settings.initial_capital,
                "final_nav": np.nan,
                "total_return": np.nan,
                "annualized_return": np.nan,
                "annualized_volatility": np.nan,
                "sharpe_ratio": np.nan,
                "max_drawdown": np.nan,
                "win_rate": np.nan,
                "n_trading_days": 0,
            }

        daily_ret = valid[ret_col].dropna()
        final_nav = valid[nav_col].iloc[-1]
        total_return = final_nav / settings.initial_capital - 1
        n_days = len(valid)
        years = n_days / 252
        ann_return = (final_nav / settings.initial_capital) ** (1 / years) - 1 if years > 0 and final_nav > 0 else np.nan
        ann_vol = daily_ret.std() * np.sqrt(252) if len(daily_ret) else np.nan
        sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(252) if len(daily_ret) and daily_ret.std() != 0 else np.nan
        running_max = valid[nav_col].cummax()
        max_drawdown = (valid[nav_col] / running_max - 1).min()
        win_rate = (daily_ret > 0).mean() if len(daily_ret) else np.nan
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

    strategy_name = f"FF5 Alpha Top {settings.top_n}"
    rows = [_one("portfolio_nav", "daily_return", strategy_name)]

    attached_benchmarks: list[tuple[dict[str, str], str]] = []
    for spec in strategy3_unique_benchmark_specs(settings):
        slug = spec["slug"]
        prefix = f"benchmark_{slug}"
        nav_col = f"{prefix}_nav"
        ret_col = f"{prefix}_daily_return"
        if nav_col in nav_df.columns and nav_df[nav_col].notna().any():
            rows.append(_one(nav_col, ret_col, spec["label"]))
            attached_benchmarks.append((spec, prefix))

    perf = pd.DataFrame(rows)

    total_fee_paid = trades_df["total_fee"].sum() if len(trades_df) > 0 and "total_fee" in trades_df.columns else 0.0
    total_trade_value = trades_df["trade_value"].sum() if len(trades_df) > 0 and "trade_value" in trades_df.columns else 0.0
    avg_turnover = rebalance_df["turnover_rate"].mean() if len(rebalance_df) > 0 and "turnover_rate" in rebalance_df.columns else np.nan

    perf["commission_rate"] = np.nan
    perf["sell_tax_rate"] = np.nan
    perf["n_rebalances"] = np.nan
    perf["total_fee_paid"] = np.nan
    perf["total_trade_value"] = np.nan
    perf["avg_turnover_per_rebalance"] = np.nan
    # keep this column as object dtype; assigning a string into a float column can fail on Streamlit/Pandas
    perf["execution_price"] = ""
    perf["benchmark"] = strategy3_benchmark_label_text(settings)

    perf.loc[perf["name"] == strategy_name, "commission_rate"] = settings.commission_rate
    perf.loc[perf["name"] == strategy_name, "sell_tax_rate"] = settings.sell_tax_rate
    perf.loc[perf["name"] == strategy_name, "n_rebalances"] = len(rebalance_df)
    perf.loc[perf["name"] == strategy_name, "total_fee_paid"] = total_fee_paid
    perf.loc[perf["name"] == strategy_name, "total_trade_value"] = total_trade_value
    perf.loc[perf["name"] == strategy_name, "avg_turnover_per_rebalance"] = avg_turnover
    perf.loc[perf["name"] == strategy_name, "execution_price"] = "t+1 close"

    strat_total = perf.loc[perf["name"] == strategy_name, "total_return"].iloc[0]
    strat_ann = perf.loc[perf["name"] == strategy_name, "annualized_return"].iloc[0]

    # 多 benchmark 比較：每個 benchmark 產生一組策略超額績效欄位。
    for spec, prefix in attached_benchmarks:
        slug = spec["slug"]
        label = spec["label"]
        bench_mask = perf["name"] == label
        if not bench_mask.any():
            continue

        bench_total = perf.loc[bench_mask, "total_return"].iloc[0]
        bench_ann = perf.loc[bench_mask, "annualized_return"].iloc[0]

        active_ret_col = f"active_{slug}_daily_return"
        active_ret = nav_df[active_ret_col].dropna() if active_ret_col in nav_df.columns else pd.Series(dtype=float)
        tracking_error = active_ret.std() * np.sqrt(252) if len(active_ret) else np.nan
        information_ratio = active_ret.mean() / active_ret.std() * np.sqrt(252) if len(active_ret) and active_ret.std() != 0 else np.nan

        total_col = f"strategy_minus_{slug}_total_return"
        ann_col = f"strategy_minus_{slug}_annualized_return"
        te_col = f"tracking_error_vs_{slug}"
        ir_col = f"information_ratio_vs_{slug}"

        for col in [total_col, ann_col, te_col, ir_col]:
            if col not in perf.columns:
                perf[col] = np.nan

        perf.loc[perf["name"] == strategy_name, total_col] = strat_total - bench_total
        perf.loc[perf["name"] == strategy_name, ann_col] = strat_ann - bench_ann
        perf.loc[perf["name"] == strategy_name, te_col] = tracking_error
        perf.loc[perf["name"] == strategy_name, ir_col] = information_ratio

    return perf


def strategy3_plot_nav(nav_df: pd.DataFrame, top_n: int, settings: FF5AlphaSettings) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=nav_df["date"], y=nav_df["portfolio_nav"], mode="lines", name=f"FF5 Alpha Top {top_n}"))

    for spec in strategy3_unique_benchmark_specs(settings):
        slug = spec["slug"]
        nav_col = f"benchmark_{slug}_nav"
        if nav_col in nav_df.columns and nav_df[nav_col].notna().any():
            fig.add_trace(go.Scatter(
                x=nav_df["date"],
                y=nav_df[nav_col],
                mode="lines",
                name=f"{spec['label']} Benchmark",
            ))

    fig.update_layout(title="Portfolio NAV vs Benchmarks", template="plotly_white", height=460, hovermode="x unified")
    return fig


def strategy3_plot_cum_return(nav_df: pd.DataFrame, top_n: int, settings: FF5AlphaSettings) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=nav_df["date"], y=nav_df["cum_return"], mode="lines", name=f"FF5 Alpha Top {top_n}"))

    for spec in strategy3_unique_benchmark_specs(settings):
        slug = spec["slug"]
        cum_col = f"benchmark_{slug}_cum_return"
        if cum_col in nav_df.columns and nav_df[cum_col].notna().any():
            fig.add_trace(go.Scatter(
                x=nav_df["date"],
                y=nav_df[cum_col],
                mode="lines",
                name=f"{spec['label']} Benchmark",
            ))

    fig.update_layout(title="Cumulative Return", template="plotly_white", height=420, yaxis_tickformat=".1%", hovermode="x unified")
    return fig


def strategy3_plot_drawdown(nav_df: pd.DataFrame, top_n: int, settings: FF5AlphaSettings) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=nav_df["date"], y=nav_df["drawdown"], fill="tozeroy", mode="lines", name=f"FF5 Alpha Top {top_n}"))

    for spec in strategy3_unique_benchmark_specs(settings):
        slug = spec["slug"]
        dd_col = f"benchmark_{slug}_drawdown"
        if dd_col in nav_df.columns and nav_df[dd_col].notna().any():
            fig.add_trace(go.Scatter(
                x=nav_df["date"],
                y=nav_df[dd_col],
                mode="lines",
                name=f"{spec['label']} Benchmark",
            ))

    fig.update_layout(title="Drawdown", template="plotly_white", height=350, yaxis_tickformat=".1%", hovermode="x unified")
    return fig


def render_strategy3_notes() -> None:
    st.subheader("③ 方法說明 / GitHub 資料準備")

    st.markdown(
        r"""
### 策略3：台股五因子 Alpha 月調倉策略

本策略預設正式回測起日為 **2019/01/01**，並可使用 CMoney 原始資料與統一奔騰基金淨值資料延伸至 **2026/05/19**。策略核心是先建構台股五因子，再利用五因子模型估計每檔股票的 rolling alpha，最後每月選出 alpha 最高的股票進行投資。

#### 1. 策略流程簡短總結

1. 每個月以月底作為形成日 `formation_date`，先選出台股市值前 150 大股票作為股票池。
2. 在每個形成日，根據市值、B/M、獲利能力與資產成長率建構台股五因子：`MKT_RF`、`SMB`、`HML`、`RMW`、`CMA`。
3. 對股票池內每檔股票使用過去 252 個交易日資料進行 Fama-French 五因子 rolling regression，且至少需要 180 筆有效日資料才估計 alpha。
4. 每個月取得各股票的 rolling alpha 後，依 alpha 由高到低排序。
5. 網站端選出 Alpha Top N 股票，並在下一個交易日以收盤價進行月調倉。
6. 回測時會扣除買賣手續費與賣出交易稅，並可同時與 0050.TW、統一奔騰基金或其他自訂 benchmark 進行績效比較。

#### 2. 五個因子的建構公式

本策略中，每個投資組合的日報酬皆使用形成日市值加權計算：

$$
R_{P,t} = \sum_{i \in P} w_{i,f} R_{i,t}
$$

其中：

$$
w_{i,f} = \frac{ME_{i,f}}{\sum_{j \in P} ME_{j,f}}
$$

$R_{P,t}$ 為投資組合 $P$ 在第 $t$ 日的報酬，$R_{i,t}$ 為股票 $i$ 在第 $t$ 日的報酬，$ME_{i,f}$ 為股票 $i$ 在形成日 $f$ 的市值。

##### (1) 市場因子 MKT_RF

市場因子使用市值前 150 大股票的市值加權報酬作為市場投資組合報酬：

$$
MKT_t = R_{Top150,t}^{VW}
$$

市場超額報酬因子為：

$$
MKT\_RF_t = MKT_t - RF_t
$$

其中 $RF_t$ 為第 $t$ 日的無風險利率。此因子衡量整體大型股市場相對於無風險利率的超額報酬。

##### (2) 規模因子 SMB

SMB 是 Small Minus Big，用來衡量小型股相對大型股的超額報酬。在每個形成日，先依市值中位數將股票分為 Small 與 Big 兩組：

- $S$：市值小於或等於中位數的股票。
- $B$：市值大於中位數的股票。

接著分別搭配 B/M、獲利能力與資產成長率三種特徵分組，計算三個 SMB，再取平均。

**第一種：由 B/M 分組得到的 SMB**

先依 B/M 的第 30 與第 70 百分位數，將股票分為 Low、Medium、High 三組，再與 Size 分組交叉形成六個投資組合：

| 符號 | 意義 |
|---|---|
| $SL_t$ | Small 且 Low B/M 股票組合在第 $t$ 日的市值加權報酬 |
| $SM_t$ | Small 且 Medium B/M 股票組合在第 $t$ 日的市值加權報酬 |
| $SH_t$ | Small 且 High B/M 股票組合在第 $t$ 日的市值加權報酬 |
| $BL_t$ | Big 且 Low B/M 股票組合在第 $t$ 日的市值加權報酬 |
| $BM_t$ | Big 且 Medium B/M 股票組合在第 $t$ 日的市值加權報酬 |
| $BH_t$ | Big 且 High B/M 股票組合在第 $t$ 日的市值加權報酬 |

$$
SMB^{BM}_t =
\frac{SL_t + SM_t + SH_t}{3}
-
\frac{BL_t + BM_t + BH_t}{3}
$$

**第二種：由獲利能力分組得到的 SMB**

依獲利能力的第 30 與第 70 百分位數，將股票分為 Weak、Neutral、Robust 三組，再與 Size 分組交叉形成六個投資組合：

| 符號 | 意義 |
|---|---|
| $SW_t$ | Small 且 Weak profitability 股票組合在第 $t$ 日的市值加權報酬 |
| $SN_t$ | Small 且 Neutral profitability 股票組合在第 $t$ 日的市值加權報酬 |
| $SR_t$ | Small 且 Robust profitability 股票組合在第 $t$ 日的市值加權報酬 |
| $BW_t$ | Big 且 Weak profitability 股票組合在第 $t$ 日的市值加權報酬 |
| $BN_t$ | Big 且 Neutral profitability 股票組合在第 $t$ 日的市值加權報酬 |
| $BR_t$ | Big 且 Robust profitability 股票組合在第 $t$ 日的市值加權報酬 |

$$
SMB^{OP}_t =
\frac{SW_t + SN_t + SR_t}{3}
-
\frac{BW_t + BN_t + BR_t}{3}
$$

**第三種：由投資風格分組得到的 SMB**

依資產成長率的第 30 與第 70 百分位數，將股票分為 Conservative、Neutral、Aggressive 三組，再與 Size 分組交叉形成六個投資組合：

| 符號 | 意義 |
|---|---|
| $SC_t$ | Small 且 Conservative investment 股票組合在第 $t$ 日的市值加權報酬 |
| $SN^{INV}_t$ | Small 且 Neutral investment 股票組合在第 $t$ 日的市值加權報酬 |
| $SA_t$ | Small 且 Aggressive investment 股票組合在第 $t$ 日的市值加權報酬 |
| $BC_t$ | Big 且 Conservative investment 股票組合在第 $t$ 日的市值加權報酬 |
| $BN^{INV}_t$ | Big 且 Neutral investment 股票組合在第 $t$ 日的市值加權報酬 |
| $BA_t$ | Big 且 Aggressive investment 股票組合在第 $t$ 日的市值加權報酬 |

$$
SMB^{INV}_t =
\frac{SC_t + SN^{INV}_t + SA_t}{3}
-
\frac{BC_t + BN^{INV}_t + BA_t}{3}
$$

最終 SMB 為三種 SMB 的平均：

$$
SMB_t =
\frac{SMB^{BM}_t + SMB^{OP}_t + SMB^{INV}_t}{3}
$$

##### (3) 價值因子 HML

HML 是 High Minus Low，用來衡量高 B/M 股票相對低 B/M 股票的超額報酬。B/M 定義為：

$$
BM_i = \frac{Book\ Equity_i}{Market\ Equity_i}
$$

依 B/M 將股票分為 Low、Medium、High 三組後，HML 定義為：

$$
HML_t =
\frac{SH_t + BH_t}{2}
-
\frac{SL_t + BL_t}{2}
$$

其中 $SH_t$ 與 $BH_t$ 代表高 B/M 組合報酬，$SL_t$ 與 $BL_t$ 代表低 B/M 組合報酬。若 $HML_t$ 為正，代表高 B/M 的價值股表現優於低 B/M 的成長股。

##### (4) 獲利能力因子 RMW

RMW 是 Robust Minus Weak，用來衡量高獲利能力股票相對低獲利能力股票的超額報酬。獲利能力定義為：

$$
Profitability_i =
\frac{Operating\ Income_i}{Total\ Equity_i}
$$

依獲利能力將股票分為 Weak、Neutral、Robust 三組後，RMW 定義為：

$$
RMW_t =
\frac{SR_t + BR_t}{2}
-
\frac{SW_t + BW_t}{2}
$$

其中 $SR_t$ 與 $BR_t$ 代表高獲利能力組合報酬，$SW_t$ 與 $BW_t$ 代表低獲利能力組合報酬。若 $RMW_t$ 為正，代表高獲利能力公司表現優於低獲利能力公司。

##### (5) 投資因子 CMA

CMA 是 Conservative Minus Aggressive，用來衡量低資產成長率公司相對高資產成長率公司的超額報酬。資產成長率定義為：

$$
Asset\ Growth_i =
\frac{Total\ Assets_{i,t} - Total\ Assets_{i,t-4}}
{Total\ Assets_{i,t-4}}
$$

其中 $t-4$ 代表去年同季。依資產成長率將股票分為 Conservative、Neutral、Aggressive 三組後，CMA 定義為：

$$
CMA_t =
\frac{SC_t + BC_t}{2}
-
\frac{SA_t + BA_t}{2}
$$

其中 $SC_t$ 與 $BC_t$ 代表低資產成長率、投資較保守的組合報酬，$SA_t$ 與 $BA_t$ 代表高資產成長率、投資較積極的組合報酬。若 $CMA_t$ 為正，代表投資保守公司表現優於投資積極公司。

#### 3. 五因子回歸公式與 rolling alpha

本策略使用 Fama-French 五因子模型估計個股 alpha。對每一檔股票 $i$，在每個形成日 $f$，使用形成日以前最近 252 個交易日資料進行迴歸：

$$
R_{i,t} - RF_t =
\alpha_{i,f}
+ \beta_{i,MKT}(MKT_t - RF_t)
+ \beta_{i,SMB}SMB_t
+ \beta_{i,HML}HML_t
+ \beta_{i,RMW}RMW_t
+ \beta_{i,CMA}CMA_t
+ \varepsilon_{i,t}
$$

其中：

| 符號 | 意義 |
|---|---|
| $R_{i,t} - RF_t$ | 股票 $i$ 在第 $t$ 日的超額報酬 |
| $MKT_t - RF_t$ | 市場超額報酬因子，即 `MKT_RF` |
| $SMB_t$ | 規模因子，衡量小型股相對大型股的報酬差 |
| $HML_t$ | 價值因子，衡量高 B/M 股票相對低 B/M 股票的報酬差 |
| $RMW_t$ | 獲利能力因子，衡量高獲利能力股票相對低獲利能力股票的報酬差 |
| $CMA_t$ | 投資因子，衡量保守投資公司相對積極投資公司的報酬差 |
| $\alpha_{i,f}$ | 股票 $i$ 在形成日 $f$ 估計出的 rolling alpha |
| $\varepsilon_{i,t}$ | 迴歸殘差，代表模型無法解釋的部分 |

Rolling 的意思是：每到一個新的形成日 $f$，都重新往前取最近 252 個交易日估計一次五因子迴歸。隨著形成日逐月往前推進，估計使用的時間視窗也會逐月向前滾動，因此稱為 rolling regression。

由於迴歸使用的是日報酬資料，估計出的 $\alpha_{i,f}$ 是每日 alpha。為了讓 alpha 較容易解讀，可以將每日 alpha 乘以 252 轉成年化 alpha：

$$
\alpha^{annualized}_{i,f} = 252 \times \alpha_{i,f}
$$

年化 alpha 主要是為了方便解釋其經濟意義；若只是用 alpha 進行股票排序，是否年化不會改變排名，因為所有股票都乘上相同常數 252。

#### GitHub 資料

**網站必要原始資料**

- `DATA`：日行情、市值、PB 等欄位
- `AD PRICE`：還原收盤價，用於個股日報酬與回測成交價
- `ASSET`：資產負債表，用於 B/M 與 CMA
- `RMW`：損益表，用於 profitability 與 RMW

網站會在執行回測時自動產生 `top150_formation_characteristics`、
`ff5_factor_returns_top150`、`alpha_scores_ff5_top150`，
再依 Alpha Top N 形成持股與回測結果。
        """
    )



def render_strategy3_page() -> None:
    st.title("策略3(台股五因子 Alpha 月調倉策略)")
    st.caption("從原始 CMoney Excel 建立 Top 150 股票池、五因子、rolling alpha，再執行 Alpha Top N 月調倉。")

    settings = strategy3_sidebar_settings()

    tabs = st.tabs(["① 回測結果", "② 持股 / 交易資料", "③ 方法說明 / GitHub 資料"])

    with tabs[0]:
        st.subheader("① 回測結果")
        st.write(
            f"目前設定：市值前 **{settings.market_cap_top_n}**，Alpha Top **{settings.top_n}**，"
            f"lookback **{settings.lookback_days}**，min obs **{settings.min_obs}**，"
            f"初始資金 **{settings.initial_capital:,.0f}**，正式回測期間 **{settings.backtest_start_date.date()} ~ {settings.backtest_end_date.date() if settings.backtest_end_date is not None else '最新'}**，"
            f"Benchmark：**{strategy3_benchmark_label_text(settings)}**。"
        )

        run = st.button("執行策略3回測", type="primary", use_container_width=True)

        if run:
            try:
                with st.spinner("讀取原始資料、建立五因子並估計 rolling alpha..."):
                    raw_result = strategy3_build_raw_pipeline_cached(
                        settings.raw_data_path,
                        settings.market_cap_top_n,
                        settings.lookback_days,
                        settings.min_obs,
                        settings.backtest_start_date,
                        settings.backtest_end_date,
                    )
                    if "fallback_warning" in raw_result and not raw_result["fallback_warning"].empty:
                        warning_msg = str(raw_result["fallback_warning"].iloc[0].get("message", "已使用 fallback pipeline。"))
                        st.warning(warning_msg)

                    alpha = strategy3_normalize_alpha(raw_result["alpha_scores"])
                    price_df = strategy3_normalize_price(raw_result["price_df"])

                if alpha.empty:
                    st.error("沒有成功產生 alpha scores，請調低 min obs 或檢查原始資料期間。")
                    return

                alpha_for_positions = strategy3_filter_alpha_scores(alpha, settings)
                if alpha_for_positions.empty:
                    st.error("alpha 篩選後沒有可用股票，請放寬 alpha 或 p-value 條件。")
                    return

                with st.spinner("產生 Top N 每月持股..."):
                    positions = strategy3_create_positions(alpha_for_positions, price_df, settings.top_n)
                    if positions.empty:
                        st.error("沒有成功產生持股清單，請檢查 alpha scores 與 price data。")
                    else:
                        latest_positions = strategy3_latest_month_holdings(positions)

                        with st.spinner("執行 t+1 收盤價回測..."):
                            nav_df, trades_df, rebalance_df = strategy3_run_backtest(
                                positions=positions,
                                price_df=price_df,
                                initial_capital=settings.initial_capital,
                                commission_rate=settings.commission_rate,
                                sell_tax_rate=settings.sell_tax_rate,
                                backtest_start_date=settings.backtest_start_date,
                                backtest_end_date=settings.backtest_end_date,
                            )

                        with st.spinner(f"整理 benchmarks：{strategy3_benchmark_label_text(settings)}..."):
                            nav_df = strategy3_attach_benchmark(nav_df, settings)

                        perf = strategy3_calc_performance(nav_df, trades_df, rebalance_df, settings)

                        st.session_state["strategy3_result"] = {
                            "alpha": alpha,
                            "alpha_for_positions": alpha_for_positions,
                            "price_df": price_df,
                            "ff5": raw_result["ff5"],
                            "formations": raw_result["formations"],
                            "factor_panel": raw_result["factor_panel"],
                            "regression_panel": raw_result["regression_panel"],
                            "file_manifest": raw_result["file_manifest"],
                            "positions": positions,
                            "latest_positions": latest_positions,
                            "nav_df": nav_df,
                            "trades_df": trades_df,
                            "rebalance_df": rebalance_df,
                            "perf": perf,
                            "settings": settings,
                        }

            except Exception as exc:
                st.error(f"策略3回測失敗：{exc}")
                with st.expander("顯示完整錯誤 traceback"):
                    st.exception(exc)

        result = st.session_state.get("strategy3_result")
        if not result:
            st.info("請先確認左側參數與原始資料資料夾，然後按「執行策略3回測」。")
        else:
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

            st.plotly_chart(strategy3_plot_nav(nav_df, settings_used.top_n, settings_used), use_container_width=True)
            st.plotly_chart(strategy3_plot_cum_return(nav_df, settings_used.top_n, settings_used), use_container_width=True)
            st.plotly_chart(strategy3_plot_drawdown(nav_df, settings_used.top_n, settings_used), use_container_width=True)

            st.markdown("#### 績效比較表")
            safe_streamlit_dataframe(perf, use_container_width=True, hide_index=True)

            excel_bytes = df_to_excel_bytes({
                "performance": perf,
                "nav": nav_df,
                "latest_holdings": result.get("latest_positions", strategy3_latest_month_holdings(positions)),
                "positions": positions,
                "trades": trades_df,
                "rebalance": rebalance_df,
                "alpha_scores": result["alpha"],
                "ff5_factors": result["ff5"],
                "top150_formations": result["formations"],
                "raw_files": result["file_manifest"],
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
        else:
            positions = result["positions"]
            latest_positions = result.get("latest_positions", strategy3_latest_month_holdings(positions))
            trades_df = result["trades_df"]
            rebalance_df = result["rebalance_df"]
            nav_df = result["nav_df"]

            st.markdown("#### 最新月份持股名單")
            if latest_positions.empty:
                st.info("目前沒有可顯示的最新月份持股名單。")
            else:
                latest_date = None
                if "formation_date" in latest_positions.columns:
                    latest_date = pd.to_datetime(latest_positions["formation_date"], errors="coerce").max()
                elif "holding_start" in latest_positions.columns:
                    latest_date = pd.to_datetime(latest_positions["holding_start"], errors="coerce").max()
                if pd.notna(latest_date):
                    st.caption(f"最新月份：{pd.Timestamp(latest_date).strftime('%Y-%m-%d')}；持股數：{len(latest_positions)}")
                safe_streamlit_dataframe(latest_positions, use_container_width=True, hide_index=True)

            st.divider()

            st.markdown("#### 每月 Alpha Top N 成分股")
            safe_streamlit_dataframe(positions, use_container_width=True, hide_index=True)

            st.markdown("#### 交易紀錄")
            safe_streamlit_dataframe(trades_df, use_container_width=True, hide_index=True)

            st.markdown("#### 調倉摘要")
            safe_streamlit_dataframe(rebalance_df, use_container_width=True, hide_index=True)

            st.markdown("#### NAV 明細")
            safe_streamlit_dataframe(nav_df, use_container_width=True, hide_index=True)

            with st.expander("原始資料與前置作業輸出"):
                st.markdown("#### Raw data files")
                safe_streamlit_dataframe(result["file_manifest"], use_container_width=True, hide_index=True)
                st.markdown("#### Alpha scores")
                safe_streamlit_dataframe(result["alpha"], use_container_width=True, hide_index=True)
                st.markdown("#### FF5 factor returns")
                safe_streamlit_dataframe(result["ff5"], use_container_width=True, hide_index=True)
                st.markdown("#### Top 150 formation characteristics")
                safe_streamlit_dataframe(result["formations"], use_container_width=True, hide_index=True)

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
