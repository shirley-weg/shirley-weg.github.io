
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Literal
import itertools
import logging
import warnings

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

StrategyName = Literal["distance", "cointegration"]
APP_VERSION = "2026-05-18-v2-sidebar-settings-fixed"


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




def sidebar_settings(strategy: StrategyName) -> AppSettings:
    """Render left sidebar controls and return an AppSettings object.

    This function is intentionally shared by both strategies.  Strategy-specific
    controls are kept minimal: cointegration uses the p-value threshold, while
    distance simply carries the value without using it.
    """

    default_data_start = pd.Timestamp("2022-01-01").date()
    default_trading_start = pd.Timestamp("2024-01-01").date()
    default_trading_end = pd.Timestamp.today().date()

    universe_all = stock_pool_df()
    industries = sorted(universe_all["industry"].dropna().unique().tolist())
    industry_options = ["全部"] + industries
    default_industry_index = industry_options.index("金融保險業") if "金融保險業" in industry_options else 0

    with st.sidebar:
        st.header("參數設定")
        st.caption(f"版本：{APP_VERSION}")

        if st.button("← 返回策略選單", use_container_width=True, key=f"back_to_selector_{strategy}"):
            st.session_state["selected_strategy"] = None
            st.rerun()

        st.divider()
        st.subheader("股票池")
        industry = st.selectbox(
            "細產業股票池",
            options=industry_options,
            index=default_industry_index,
            key=f"industry_{strategy}",
            help="選擇要進行配對篩選的細產業。選『全部』會使用完整股票池，速度會較慢。",
        )

        st.subheader("資料與回測期間")
        data_start = st.date_input(
            "資料下載起日",
            value=default_data_start,
            key=f"data_start_{strategy}",
            help="必須早於正式回測起日，因為 rolling formation 需要過去歷史資料。",
        )
        trading_start = st.date_input(
            "正式回測起日",
            value=default_trading_start,
            key=f"trading_start_{strategy}",
        )
        use_latest = st.checkbox(
            "回測到最新資料",
            value=True,
            key=f"use_latest_{strategy}",
        )
        if use_latest:
            trading_end = None
        else:
            trading_end = st.date_input(
                "正式回測結束日",
                value=default_trading_end,
                key=f"trading_end_{strategy}",
            )
        market_ticker = st.text_input(
            "Beta benchmark",
            value="^TWII",
            key=f"market_ticker_{strategy}",
        ).strip()
        auto_adjust = st.checkbox(
            "使用 yfinance auto_adjust",
            value=True,
            key=f"auto_adjust_{strategy}",
        )

        st.subheader("Pair 篩選參數")
        lookback_days = int(st.number_input(
            "Rolling formation lookback days",
            min_value=60,
            max_value=1000,
            value=252,
            step=21,
            key=f"lookback_days_{strategy}",
        ))
        weekly_freq = st.selectbox(
            "每週重選頻率標記",
            options=["W-FRI", "W-THU", "W-WED", "W-TUE", "W-MON"],
            index=0,
            key=f"weekly_freq_{strategy}",
            help="以該週期分組後取每組第一個實際交易日作為重新篩選日。",
        )
        min_obs = int(st.number_input(
            "Formation 最少共同資料筆數",
            min_value=30,
            max_value=1000,
            value=200,
            step=10,
            key=f"min_obs_{strategy}",
        ))
        k_final = int(st.number_input(
            "候選 pair 數量 final_k",
            min_value=1,
            max_value=50,
            value=10,
            step=1,
            key=f"k_final_{strategy}",
            help="模式一顯示的最終候選數；模式二內部仍先取 Top 5，再交易最佳 n 組。",
        ))
        preselect_multiplier = int(st.number_input(
            "初選倍數 multiplier",
            min_value=1,
            max_value=50,
            value=10,
            step=1,
            key=f"preselect_multiplier_{strategy}",
        ))
        beta_diff_threshold = float(st.number_input(
            "Beta 差距門檻",
            min_value=0.0,
            max_value=5.0,
            value=0.2,
            step=0.05,
            format="%.3f",
            key=f"beta_diff_threshold_{strategy}",
        ))
        top_n_display = int(st.number_input(
            "候選 pair 表格顯示筆數",
            min_value=1,
            max_value=100,
            value=20,
            step=1,
            key=f"top_n_display_{strategy}",
        ))

        if strategy == "cointegration":
            pvalue_threshold = float(st.number_input(
                "共整合 p-value 門檻",
                min_value=0.001,
                max_value=0.5,
                value=0.05,
                step=0.01,
                format="%.3f",
                key=f"pvalue_threshold_{strategy}",
            ))
        else:
            pvalue_threshold = 0.05

        st.subheader("交易規則")
        entry_z = float(st.number_input(
            "Entry z-score",
            min_value=0.1,
            max_value=10.0,
            value=2.0,
            step=0.1,
            format="%.2f",
            key=f"entry_z_{strategy}",
        ))
        exit_z = float(st.number_input(
            "Exit z-score",
            min_value=0.0,
            max_value=10.0,
            value=0.5,
            step=0.1,
            format="%.2f",
            key=f"exit_z_{strategy}",
        ))
        stop_z = float(st.number_input(
            "Stop z-score",
            min_value=0.1,
            max_value=20.0,
            value=3.0,
            step=0.1,
            format="%.2f",
            key=f"stop_z_{strategy}",
        ))
        max_holding_days = int(st.number_input(
            "Max holding days",
            min_value=1,
            max_value=2000,
            value=60,
            step=5,
            key=f"max_holding_days_{strategy}",
        ))
        reentry_reset_z = float(st.number_input(
            "Stop loss 後重新允許進場 abs(z) <",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=0.1,
            format="%.2f",
            key=f"reentry_reset_z_{strategy}",
        ))

        st.subheader("成本與資金")
        fee_rate = float(st.number_input(
            "買賣手續費率",
            min_value=0.0,
            max_value=0.05,
            value=0.001425,
            step=0.0001,
            format="%.6f",
            key=f"fee_rate_{strategy}",
        ))
        initial_capital = float(st.number_input(
            "單一 pair 初始資金",
            min_value=10_000.0,
            max_value=1_000_000_000.0,
            value=1_000_000.0,
            step=100_000.0,
            format="%.0f",
            key=f"initial_capital_{strategy}",
        ))

        st.caption("目前版本：模式二每週重新篩選 Top 5，但已開倉部位不會因下一週重新篩選而強制平倉。")

    return AppSettings(
        strategy=strategy,
        industry=industry,
        data_start=pd.Timestamp(data_start),
        trading_start=pd.Timestamp(trading_start),
        trading_end=None if trading_end is None else pd.Timestamp(trading_end),
        market_ticker=market_ticker or "^TWII",
        lookback_days=lookback_days,
        weekly_freq=weekly_freq,
        k_final=k_final,
        preselect_multiplier=preselect_multiplier,
        min_obs=min_obs,
        pvalue_threshold=pvalue_threshold,
        beta_diff_threshold=beta_diff_threshold,
        entry_z=entry_z,
        exit_z=exit_z,
        stop_z=stop_z,
        max_holding_days=max_holding_days,
        reentry_reset_z=reentry_reset_z,
        fee_rate=fee_rate,
        initial_capital=initial_capital,
        auto_adjust=auto_adjust,
        top_n_display=top_n_display,
    )


# ============================================================
# Basic Universe Helpers
# ============================================================

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
# Streamlit Rendering - Revised Two-Mode Version
# ============================================================

def df_to_excel_bytes(sheets: dict[str, pd.DataFrame | pd.Series]) -> bytes:
    """Excel export with CSV ZIP fallback when openpyxl is unavailable."""
    output = BytesIO()
    try:
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            for name, obj in sheets.items():
                safe_name = str(name)[:31]
                if isinstance(obj, pd.Series):
                    obj.to_frame(obj.name or "value").to_excel(writer, sheet_name=safe_name)
                elif isinstance(obj, pd.DataFrame):
                    obj.to_excel(writer, sheet_name=safe_name, index=True)
        output.seek(0)
        return output.getvalue()
    except ModuleNotFoundError:
        import zipfile
        output = BytesIO()
        with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for name, obj in sheets.items():
                if isinstance(obj, pd.Series):
                    df = obj.to_frame(obj.name or "value")
                elif isinstance(obj, pd.DataFrame):
                    df = obj
                else:
                    continue
                zf.writestr(f"{name}.csv", df.to_csv(index=True).encode("utf-8-sig"))
        output.seek(0)
        return output.getvalue()


def get_formation_window(close_df: pd.DataFrame, reference_date: pd.Timestamp, lookback_days: int) -> pd.DataFrame:
    reference_date = pd.Timestamp(reference_date)
    prior_dates = close_df.index[close_df.index < reference_date]
    if len(prior_dates) < lookback_days:
        return pd.DataFrame()
    end_pos = close_df.index.get_loc(prior_dates[-1])
    start_pos = end_pos - lookback_days + 1
    if start_pos < 0:
        return pd.DataFrame()
    return close_df.iloc[start_pos:end_pos + 1].copy()


def select_pairs_with_beta_filter_top_k(
    method: StrategyName,
    formation_prices: pd.DataFrame,
    stock_info_df: pd.DataFrame,
    market_price_full: pd.Series,
    settings: AppSettings,
    top_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    preselect_n = max(int(top_k) * int(settings.preselect_multiplier), int(top_k))
    if method == "distance":
        all_raw, preselected = select_pairs_distance_preselect(formation_prices, stock_info_df, preselect_n, settings.min_obs)
        sort_by = {"by": ["distance"], "ascending": [True]}
    else:
        all_raw, preselected = select_pairs_cointegration_preselect(
            formation_prices,
            stock_info_df,
            settings.pvalue_threshold,
            preselect_n,
            settings.min_obs,
        )
        sort_by = {"by": ["coint_t", "p_value"], "ascending": [True, True]}
    beta_df = build_beta_table_for_candidate_pairs(preselected, formation_prices, stock_info_df, market_price_full, settings.min_obs)
    final_candidates = apply_beta_and_industry_filter(preselected, beta_df, settings.beta_diff_threshold, top_k, sort_by=sort_by)
    return all_raw, preselected, final_candidates


def compute_z_series_from_params(params: dict[str, object], close_df: pd.DataFrame) -> pd.Series:
    tx = str(params["ticker_x"])
    ty = str(params["ticker_y"])
    dates = close_df[[tx, ty]].dropna().index
    values = [compute_z_at_date(params, close_df, pd.Timestamp(d)) for d in dates]
    return pd.Series(values, index=dates, name="z_current_candidate")


def run_fixed_pair_backtest(
    method: StrategyName,
    ticker_x: str,
    ticker_y: str,
    open_df: pd.DataFrame,
    close_df: pd.DataFrame,
    settings: AppSettings,
) -> dict[str, object]:
    ticker_x = to_yf_ticker(ticker_x)
    ticker_y = to_yf_ticker(ticker_y)

    all_dates = close_df.loc[settings.trading_start:settings.trading_end].index
    if len(all_dates) < 2:
        raise ValueError("回測期間資料不足。")

    formation_prices = get_formation_window(close_df, settings.trading_start, settings.lookback_days)
    if len(formation_prices) < settings.min_obs:
        raise ValueError("正式回測起日前 formation 資料不足，請提前資料下載起日或縮短 lookback days。")

    pair_row = {"ticker_x": ticker_x, "ticker_y": ticker_y}
    params = fit_spread_params(method, pair_row, formation_prices)

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

    trade_records: list[dict[str, object]] = []
    equity_records: list[dict[str, object]] = []
    z_records: list[dict[str, object]] = []

    for i, date in enumerate(all_dates):
        date = pd.Timestamp(date)
        if ticker_x not in open_df.columns or ticker_y not in open_df.columns:
            raise ValueError("輸入股票不在下載資料欄位中。")
        if date not in open_df.index:
            continue
        open_x = open_df.loc[date, ticker_x]
        open_y = open_df.loc[date, ticker_y]
        close_x = close_df.loc[date, ticker_x]
        close_y = close_df.loc[date, ticker_y]
        z_today = compute_z_at_date(params, close_df, date)

        # Execute yesterday's order at today's open.
        if pending_order is not None:
            action = str(pending_order["action"])
            if pd.isna(open_x) or pd.isna(open_y):
                pending_order = None
            elif action == "enter":
                direction = int(pending_order["direction"])
                weight_x, weight_y = calculate_pair_weights(direction, float(params["hedge_ratio"]))
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
                holding_days = 0
                entry_date = date
                entry_signal_z = float(pending_order["signal_z"])
                entry_price_x = float(open_x)
                entry_price_y = float(open_y)
                entry_direction = direction
                pending_order = None
            elif action == "exit":
                traded_notional = abs(shares_x * open_x) + abs(shares_y * open_y)
                cost = traded_notional * settings.fee_rate
                cash += shares_x * open_x
                cash += shares_y * open_y
                cash -= cost
                pnl = cash - float(entry_capital)
                trade_records.append({
                    "method": method,
                    "pair": display_pair_name(ticker_x, ticker_y),
                    "ticker_x": ticker_x,
                    "ticker_y": ticker_y,
                    "entry_date": entry_date,
                    "exit_date": date,
                    "direction": entry_direction,
                    "entry_z": entry_signal_z,
                    "exit_signal_z": float(pending_order["exit_signal_z"]),
                    "entry_price_x": entry_price_x,
                    "entry_price_y": entry_price_y,
                    "exit_price_x": float(open_x),
                    "exit_price_y": float(open_y),
                    "holding_days": holding_days,
                    "exit_reason": pending_order["exit_reason"],
                    "pnl": pnl,
                    "return": pnl / float(entry_capital),
                })
                if pending_order["exit_reason"] == "stop_loss":
                    blocked_after_stop_loss = True
                shares_x = shares_y = 0.0
                active = False
                holding_days = 0
                pending_order = None

        equity = cash + shares_x * close_x + shares_y * close_y if active else cash
        equity_records.append({"date": date, "equity": equity, "active": active})
        z_records.append({
            "date": date,
            "z_current_candidate": z_today,
            "z_active_position": z_today if active else np.nan,
            "eligible": True,
        })

        if i >= len(all_dates) - 1 or pd.isna(z_today):
            continue
        prev_date = pd.Timestamp(all_dates[i - 1]) if i > 0 else None
        if prev_date is None:
            continue
        z_prev = compute_z_at_date(params, close_df, prev_date)
        if pd.isna(z_prev):
            continue

        if not active:
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
                pending_order = {"action": "enter", "direction": direction, "signal_z": z_today}
        else:
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

    # Liquidate at backtest end only.
    final_date = pd.Timestamp(all_dates[-1])
    if active:
        close_x = close_df.loc[final_date, ticker_x]
        close_y = close_df.loc[final_date, ticker_y]
        traded_notional = abs(shares_x * close_x) + abs(shares_y * close_y)
        cost = traded_notional * settings.fee_rate
        cash += shares_x * close_x
        cash += shares_y * close_y
        cash -= cost
        z_final = compute_z_at_date(params, close_df, final_date)
        pnl = cash - float(entry_capital)
        trade_records.append({
            "method": method,
            "pair": display_pair_name(ticker_x, ticker_y),
            "ticker_x": ticker_x,
            "ticker_y": ticker_y,
            "entry_date": entry_date,
            "exit_date": final_date,
            "direction": entry_direction,
            "entry_z": entry_signal_z,
            "exit_signal_z": z_final,
            "entry_price_x": entry_price_x,
            "entry_price_y": entry_price_y,
            "exit_price_x": float(close_x),
            "exit_price_y": float(close_y),
            "holding_days": holding_days,
            "exit_reason": "backtest_end",
            "pnl": pnl,
            "return": pnl / float(entry_capital),
        })
        equity_records.append({"date": final_date, "equity": cash, "active": False})

    equity_curve = pd.DataFrame(equity_records).drop_duplicates("date", keep="last").set_index("date")["equity"]
    trades_df = pd.DataFrame(trade_records)
    z_df = pd.DataFrame(z_records).drop_duplicates("date", keep="last").set_index("date")
    summary = pd.DataFrame([{**evaluate_equity_curve(equity_curve, trades_df), "initial_capital": settings.initial_capital}])
    return {
        "summary": summary,
        "equity_curve": equity_curve,
        "trades": trades_df,
        "zscore": z_df,
        "formation_start": formation_prices.index.min(),
        "formation_end": formation_prices.index.max(),
    }


def run_formal_topn_strategy_backtest(
    method: StrategyName,
    open_df: pd.DataFrame,
    close_df: pd.DataFrame,
    stock_info_df: pd.DataFrame,
    market_price_full: pd.Series,
    settings: AppSettings,
    n_pairs: int,
    weekly_top_k: int = 5,
) -> dict[str, object]:
    all_dates = close_df.loc[settings.trading_start:settings.trading_end].index
    if len(all_dates) < 2:
        raise ValueError("回測期間資料不足。")
    week_start_set = set(get_week_start_dates(close_df, settings.trading_start, settings.trading_end, settings.weekly_freq))

    slots: list[dict[str, object]] = [
        {"slot_id": i, "active": False, "cash": float(settings.initial_capital), "pending_order": None}
        for i in range(int(n_pairs))
    ]
    blocked_pairs: set[str] = set()
    current_candidates = pd.DataFrame()
    current_candidate_params: dict[str, dict[str, object]] = {}
    selected_pair_snapshots: list[pd.DataFrame] = []
    weekly_records: list[dict[str, object]] = []
    trade_records: list[dict[str, object]] = []
    equity_records: list[dict[str, object]] = []

    for i, date in enumerate(all_dates):
        date = pd.Timestamp(date)

        # Weekly selection: screen TOP5 and use best n pairs only for free slots.
        if date in week_start_set:
            current_candidates = pd.DataFrame()
            current_candidate_params = {}
            formation_prices = get_formation_window(close_df, date, settings.lookback_days)
            if len(formation_prices) >= settings.min_obs:
                all_raw, preselected, top5_candidates = select_pairs_with_beta_filter_top_k(
                    method, formation_prices, stock_info_df, market_price_full, settings, top_k=weekly_top_k
                )
                current_candidates = top5_candidates.head(int(n_pairs)).copy()
                if len(current_candidates) > 0:
                    snap = current_candidates.copy()
                    snap["rebalance_date"] = date
                    snap["formation_start"] = formation_prices.index.min()
                    snap["formation_end"] = formation_prices.index.max()
                    selected_pair_snapshots.append(snap)
                    for _, row in current_candidates.iterrows():
                        pname = display_pair_name(row["ticker_x"], row["ticker_y"])
                        current_candidate_params[pname] = fit_spread_params(method, row, formation_prices)
                weekly_records.append({
                    "method": method,
                    "rebalance_date": date,
                    "formation_start": formation_prices.index.min(),
                    "formation_end": formation_prices.index.max(),
                    "top5_count": len(top5_candidates),
                    "selected_n_count": len(current_candidates),
                    "active_before_trade": sum(bool(s.get("active")) for s in slots),
                    "free_slots": sum(not bool(s.get("active")) for s in slots),
                })

        # Execute pending orders at today's open.
        for slot in slots:
            pending_order = slot.get("pending_order")
            if pending_order is None:
                continue
            action = str(pending_order["action"])
            if action == "enter":
                params = pending_order["spread_params"]
                ticker_x = str(params["ticker_x"])
                ticker_y = str(params["ticker_y"])
                open_x = open_df.loc[date, ticker_x]
                open_y = open_df.loc[date, ticker_y]
                if pd.isna(open_x) or pd.isna(open_y):
                    slot["pending_order"] = None
                    continue
                capital = float(slot["cash"])
                direction = int(pending_order["direction"])
                weight_x, weight_y = calculate_pair_weights(direction, float(params["hedge_ratio"]))
                shares_x = (capital * weight_x) / open_x
                shares_y = (capital * weight_y) / open_y
                traded_notional = abs(shares_x * open_x) + abs(shares_y * open_y)
                cost = traded_notional * settings.fee_rate
                cash = capital - shares_x * open_x - shares_y * open_y - cost
                slot.update({
                    "active": True,
                    "cash": cash,
                    "shares_x": shares_x,
                    "shares_y": shares_y,
                    "ticker_x": ticker_x,
                    "ticker_y": ticker_y,
                    "pair": display_pair_name(ticker_x, ticker_y),
                    "spread_params": params,
                    "direction": direction,
                    "entry_date": date,
                    "entry_signal_z": float(pending_order["signal_z"]),
                    "entry_capital": capital,
                    "entry_price_x": float(open_x),
                    "entry_price_y": float(open_y),
                    "holding_days": 0,
                    "pending_order": None,
                })
            elif action == "exit":
                ticker_x = str(slot["ticker_x"])
                ticker_y = str(slot["ticker_y"])
                open_x = open_df.loc[date, ticker_x]
                open_y = open_df.loc[date, ticker_y]
                if pd.isna(open_x) or pd.isna(open_y):
                    slot["pending_order"] = None
                    continue
                shares_x = float(slot["shares_x"])
                shares_y = float(slot["shares_y"])
                traded_notional = abs(shares_x * open_x) + abs(shares_y * open_y)
                cost = traded_notional * settings.fee_rate
                cash = float(slot["cash"]) + shares_x * open_x + shares_y * open_y - cost
                pnl = cash - float(slot["entry_capital"])
                trade_records.append({
                    "method": method,
                    "slot_id": slot["slot_id"],
                    "pair": slot["pair"],
                    "ticker_x": ticker_x,
                    "ticker_y": ticker_y,
                    "entry_date": slot["entry_date"],
                    "exit_date": date,
                    "direction": slot["direction"],
                    "entry_z": slot["entry_signal_z"],
                    "exit_signal_z": float(pending_order["exit_signal_z"]),
                    "entry_price_x": slot["entry_price_x"],
                    "entry_price_y": slot["entry_price_y"],
                    "exit_price_x": float(open_x),
                    "exit_price_y": float(open_y),
                    "holding_days": slot["holding_days"],
                    "exit_reason": pending_order["exit_reason"],
                    "pnl": pnl,
                    "return": pnl / float(slot["entry_capital"]),
                })
                if pending_order["exit_reason"] == "stop_loss":
                    blocked_pairs.add(str(slot["pair"]))
                old_id = slot["slot_id"]
                slot.clear()
                slot.update({"slot_id": old_id, "active": False, "cash": cash, "pending_order": None})

        # Mark to market.
        portfolio_equity = 0.0
        for slot in slots:
            if slot.get("active"):
                tx = str(slot["ticker_x"])
                ty = str(slot["ticker_y"])
                cx = close_df.loc[date, tx]
                cy = close_df.loc[date, ty]
                if pd.isna(cx) or pd.isna(cy):
                    portfolio_equity += float(slot["cash"])
                else:
                    portfolio_equity += float(slot["cash"]) + float(slot["shares_x"]) * cx + float(slot["shares_y"]) * cy
            else:
                portfolio_equity += float(slot["cash"])
        equity_records.append({
            "date": date,
            "equity": portfolio_equity,
            "active_slots": sum(bool(s.get("active")) for s in slots),
            "free_slots": sum(not bool(s.get("active")) for s in slots),
        })

        if i >= len(all_dates) - 1:
            continue

        # Exit signals for active positions.
        for slot in slots:
            if not slot.get("active") or slot.get("pending_order") is not None:
                continue
            z_today = compute_z_at_date(slot["spread_params"], close_df, date)
            if pd.isna(z_today):
                continue
            slot["holding_days"] = int(slot["holding_days"]) + 1
            exit_reason = None
            if abs(z_today) < settings.exit_z:
                exit_reason = "mean_reversion"
            elif abs(z_today) > settings.stop_z:
                exit_reason = "stop_loss"
            elif int(slot["holding_days"]) >= settings.max_holding_days:
                exit_reason = "max_holding_days"
            if exit_reason is not None:
                slot["pending_order"] = {"action": "exit", "exit_signal_z": z_today, "exit_reason": exit_reason}

        # Entry only for free slots and current weekly best n pairs.
        free_slots = [slot for slot in slots if (not slot.get("active")) and slot.get("pending_order") is None]
        if len(free_slots) == 0 or current_candidates is None or len(current_candidates) == 0:
            continue
        active_pairs = {str(slot.get("pair")) for slot in slots if slot.get("active")}
        used_pairs = set()
        prev_date = pd.Timestamp(all_dates[i - 1]) if i > 0 else None
        if prev_date is None:
            continue
        for _, row in current_candidates.iterrows():
            if len(free_slots) == 0:
                break
            pname = display_pair_name(row["ticker_x"], row["ticker_y"])
            if pname in active_pairs or pname in used_pairs or pname not in current_candidate_params:
                continue
            params = current_candidate_params[pname]
            z_today = compute_z_at_date(params, close_df, date)
            z_prev = compute_z_at_date(params, close_df, prev_date)
            if pd.isna(z_today) or pd.isna(z_prev):
                continue
            if pname in blocked_pairs:
                if abs(z_today) < settings.reentry_reset_z:
                    blocked_pairs.remove(pname)
                continue
            direction = None
            if (z_prev <= settings.entry_z) and (settings.entry_z < z_today < settings.stop_z):
                direction = -1
            elif (z_prev >= -settings.entry_z) and (-settings.stop_z < z_today < -settings.entry_z):
                direction = 1
            if direction is None:
                continue
            slot = free_slots.pop(0)
            slot["pending_order"] = {"action": "enter", "spread_params": params, "direction": direction, "signal_z": z_today}
            used_pairs.add(pname)

    # Final liquidation.
    final_date = pd.Timestamp(all_dates[-1])
    for slot in slots:
        if not slot.get("active"):
            continue
        tx = str(slot["ticker_x"])
        ty = str(slot["ticker_y"])
        cx = close_df.loc[final_date, tx]
        cy = close_df.loc[final_date, ty]
        shares_x = float(slot["shares_x"])
        shares_y = float(slot["shares_y"])
        traded_notional = abs(shares_x * cx) + abs(shares_y * cy)
        cost = traded_notional * settings.fee_rate
        cash = float(slot["cash"]) + shares_x * cx + shares_y * cy - cost
        pnl = cash - float(slot["entry_capital"])
        z_final = compute_z_at_date(slot["spread_params"], close_df, final_date)
        trade_records.append({
            "method": method,
            "slot_id": slot["slot_id"],
            "pair": slot["pair"],
            "ticker_x": tx,
            "ticker_y": ty,
            "entry_date": slot["entry_date"],
            "exit_date": final_date,
            "direction": slot["direction"],
            "entry_z": slot["entry_signal_z"],
            "exit_signal_z": z_final,
            "entry_price_x": slot["entry_price_x"],
            "entry_price_y": slot["entry_price_y"],
            "exit_price_x": float(cx),
            "exit_price_y": float(cy),
            "holding_days": slot["holding_days"],
            "exit_reason": "backtest_end",
            "pnl": pnl,
            "return": pnl / float(slot["entry_capital"]),
        })

    equity_curve = pd.DataFrame(equity_records).drop_duplicates("date", keep="last").set_index("date")["equity"]
    trades_df = pd.DataFrame(trade_records)
    selected_pairs_history = pd.concat(selected_pair_snapshots, axis=0).reset_index(drop=True) if selected_pair_snapshots else pd.DataFrame()
    weekly_summary = pd.DataFrame(weekly_records)
    summary = pd.DataFrame([{**evaluate_equity_curve(equity_curve, trades_df), "initial_capital": float(settings.initial_capital) * int(n_pairs), "n_pairs": int(n_pairs), "weekly_top_k": int(weekly_top_k)}])
    return {
        "summary": summary,
        "equity_curve": equity_curve,
        "trades": trades_df,
        "weekly_summary": weekly_summary,
        "selected_pairs_history": selected_pairs_history,
    }


def render_strategy_selector() -> None:
    st.title("Trading Strategy Lab")
    st.caption("請先選擇要使用的交易策略。")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            <div class="strategy-card">
              <h3>配對策略1(距離法)</h3>
              <p>使用標準化 log price 的 SSD 距離選 pair。策略內含手動固定 pair 回測與正式週度 Top-N 交易模式。</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("進入配對策略1(距離法)", type="primary", use_container_width=True):
            st.session_state["selected_strategy"] = "distance"
            st.rerun()
    with col2:
        st.markdown(
            """
            <div class="strategy-card">
              <h3>配對策略2(共整合法)</h3>
              <p>使用 Engle-Granger 共整合檢定選 pair，並以 OLS residual 作為 spread 交易訊號。</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("進入配對策略2(共整合法)", type="primary", use_container_width=True):
            st.session_state["selected_strategy"] = "cointegration"
            st.rerun()


def render_mode1_tab(settings: AppSettings) -> None:
    st.subheader("模式一：先篩選最佳 Pair，再手動選一組固定回測")
    st.write("此模式先用參考日前的 formation window 產生候選 pair。回測時不每週重新篩選 pair，而是固定使用回測起日前的 formation window 估計參數，直接回測到結束日 / 最新資料。")

    run_screen = st.button("執行選股 / 選 Pair", type="primary", use_container_width=True, key=f"mode1_screen_{settings.strategy}")
    ref_col, info_col = st.columns([0.35, 0.65])
    with ref_col:
        ref_date = st.date_input("選 Pair 參考日期", value=settings.trading_start.date(), key=f"mode1_ref_{settings.strategy}")
    with info_col:
        st.info(f"股票池：{settings.industry}；顯示 Top {settings.top_n_display}；正式回測起日：{settings.trading_start.date()}")

    key = f"mode1_screen_result_{settings.strategy}"
    if run_screen:
        try:
            open_df, close_df, market_price, universe = prepare_download_for_universe(settings)
            ref_ts = pd.Timestamp(ref_date)
            formation_prices = get_formation_window(close_df, ref_ts, settings.lookback_days)
            if formation_prices.empty:
                st.error("參考日前資料不足，請提前資料下載起日或縮短 lookback days。")
                return
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
    if result:
        final_candidates = result["final_candidates"]
        st.caption(f"Pair selection formation：{pd.Timestamp(result['formation_start']).date()} ～ {pd.Timestamp(result['formation_end']).date()}")
        if final_candidates is None or len(final_candidates) == 0:
            st.warning("目前條件下沒有通過篩選的候選 pair。")
        else:
            metric_cols = ["distance"] if settings.strategy == "distance" else ["coint_t", "p_value", "direction"]
            display_cols = ["code_x", "name_x", "code_y", "name_y", "industry_x", "industry_y", "n_obs", "beta_x", "beta_y", "beta_diff"] + metric_cols
            display_cols = [c for c in display_cols if c in final_candidates.columns]
            st.dataframe(final_candidates.head(settings.top_n_display)[display_cols], use_container_width=True, hide_index=True)
            labels = []
            for idx, row in final_candidates.iterrows():
                score = f"distance={row['distance']:.3f}" if settings.strategy == "distance" else f"p={row['p_value']:.4f}, t={row['coint_t']:.3f}"
                labels.append(f"{idx+1}. {row['code_x']} {row['name_x']} / {row['code_y']} {row['name_y']} | {score} | beta_diff={row['beta_diff']:.3f}")
            selected_label = st.selectbox("套用候選 pair 到回測", labels, key=f"mode1_pair_select_{settings.strategy}")
            selected_idx = labels.index(selected_label)
            selected = final_candidates.iloc[selected_idx]
            if st.button("套用選取 pair 到回測輸入框", use_container_width=True, key=f"mode1_apply_{settings.strategy}"):
                st.session_state[f"{settings.strategy}_fixed_code_x"] = str(selected["code_x"])
                st.session_state[f"{settings.strategy}_fixed_code_y"] = str(selected["code_y"])
                st.success(f"已套用：{selected['code_x']} / {selected['code_y']}")
            st.download_button(
                "下載候選 pair Excel/ZIP",
                data=df_to_excel_bytes({"final_candidates": final_candidates, "preselected": result["preselected"], "all_raw": result["all_raw"]}),
                file_name=f"{settings.strategy}_mode1_candidates.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    st.divider()
    st.markdown("### 固定 Pair 回測")
    col_x, col_y = st.columns(2)
    with col_x:
        code_x = st.text_input("股票 X 代號", key=f"{settings.strategy}_fixed_code_x", placeholder="例如 2892").strip().upper()
    with col_y:
        code_y = st.text_input("股票 Y 代號", key=f"{settings.strategy}_fixed_code_y", placeholder="例如 5880").strip().upper()

    if st.button("Run fixed-pair backtest", type="primary", use_container_width=True, key=f"mode1_backtest_{settings.strategy}"):
        if not code_x or not code_y:
            st.error("請輸入兩檔股票代號。")
            return
        try:
            ticker_x = to_yf_ticker(code_x)
            ticker_y = to_yf_ticker(code_y)
            open_df, close_df, market_price, universe = prepare_download_for_universe(settings, extra_tickers=[ticker_x, ticker_y])
            if ticker_x not in close_df.columns or ticker_y not in close_df.columns:
                st.error("下載資料中找不到輸入股票。請確認 yfinance 是否支援該代號。")
                return
            with st.spinner("執行固定 pair 回測..."):
                result = run_fixed_pair_backtest(settings.strategy, ticker_x, ticker_y, open_df, close_df, settings)
            st.session_state[f"mode1_backtest_result_{settings.strategy}"] = result
            st.session_state[f"mode1_backtest_pair_{settings.strategy}"] = (ticker_x, ticker_y, close_df)
        except Exception as exc:
            st.error(f"回測失敗：{exc}")
            return

    bt = st.session_state.get(f"mode1_backtest_result_{settings.strategy}")
    pair_state = st.session_state.get(f"mode1_backtest_pair_{settings.strategy}")
    if bt and pair_state:
        ticker_x, ticker_y, close_df = pair_state
        st.markdown(f"### 回測結果：{format_stock_label(ticker_x)} / {format_stock_label(ticker_y)}")
        show_summary_metrics(bt["summary"])
        equity_curve = bt["equity_curve"]
        trades_df = bt["trades"]
        z_df = bt["zscore"]
        charts = st.tabs(["績效", "價格與訊號", "交易紀錄"])
        with charts[0]:
            st.plotly_chart(plot_equity_curve(equity_curve), use_container_width=True)
            st.plotly_chart(plot_drawdown(equity_curve), use_container_width=True)
        with charts[1]:
            trading_close = close_df.loc[settings.trading_start:settings.trading_end]
            st.plotly_chart(plot_price_chart(trading_close, ticker_x, ticker_y), use_container_width=True)
            st.plotly_chart(plot_normalized_price_chart(trading_close, ticker_x, ticker_y), use_container_width=True)
            st.plotly_chart(plot_zscore(z_df, settings, trades_df), use_container_width=True)
        with charts[2]:
            st.plotly_chart(plot_trade_pnl(trades_df), use_container_width=True)
            st.dataframe(trades_df, use_container_width=True, hide_index=True)
        st.download_button(
            "下載固定 Pair 回測結果 Excel/ZIP",
            data=df_to_excel_bytes({"summary": bt["summary"], "equity_curve": equity_curve, "trades": trades_df, "zscore": z_df}),
            file_name=f"{settings.strategy}_{code_x}_{code_y}_fixed_backtest.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


def render_mode2_formal_strategy(settings: AppSettings) -> None:
    st.subheader("模式二：正式交易策略模式")
    st.write("每週重新篩選 Top 5 組 pair，再取最好的 n 組進行交易；使用者不手動指定 pair。已開倉部位不會因下一週重新篩選而強制平倉，會一路持有到觸發出場條件。")
    n_pairs = st.number_input("每週從 Top 5 中交易最佳 n 組 pair", min_value=1, max_value=5, value=1, step=1, key=f"formal_n_{settings.strategy}")
    st.caption("如果 Top 5 不滿 5 組，就以實際通過篩選的組數為準。若已有持倉，新的 pair 只會用空閒 slot 進場。")

    if st.button("Run formal Top-N strategy", type="primary", use_container_width=True, key=f"formal_run_{settings.strategy}"):
        try:
            open_df, close_df, market_price, universe = prepare_download_for_universe(settings)
            with st.spinner("執行正式交易策略回測..."):
                result = run_formal_topn_strategy_backtest(settings.strategy, open_df, close_df, universe, market_price, settings, int(n_pairs), weekly_top_k=5)
            st.session_state[f"formal_result_{settings.strategy}"] = result
        except Exception as exc:
            st.error(f"正式策略回測失敗：{exc}")
            return

    result = st.session_state.get(f"formal_result_{settings.strategy}")
    if result:
        st.markdown("### 正式策略回測結果")
        show_summary_metrics(result["summary"])
        equity_curve = result["equity_curve"]
        trades_df = result["trades"]
        chart_tab, table_tab = st.tabs(["績效圖表", "資料表"])
        with chart_tab:
            st.plotly_chart(plot_equity_curve(equity_curve), use_container_width=True)
            st.plotly_chart(plot_drawdown(equity_curve), use_container_width=True)
            st.plotly_chart(plot_trade_pnl(trades_df), use_container_width=True)
        with table_tab:
            st.markdown("#### 交易紀錄")
            st.dataframe(trades_df, use_container_width=True, hide_index=True)
            st.markdown("#### 每週選出的交易候選 pair")
            st.dataframe(result["selected_pairs_history"], use_container_width=True, hide_index=True)
            st.markdown("#### 每週摘要")
            st.dataframe(result["weekly_summary"], use_container_width=True, hide_index=True)
        st.download_button(
            "下載正式策略結果 Excel/ZIP",
            data=df_to_excel_bytes({
                "summary": result["summary"],
                "equity_curve": equity_curve,
                "trades": trades_df,
                "selected_pairs": result["selected_pairs_history"],
                "weekly_summary": result["weekly_summary"],
            }),
            file_name=f"{settings.strategy}_formal_topn_backtest.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


def render_method_notes(settings: AppSettings) -> None:
    st.subheader("方法說明")
    if settings.strategy == "distance":
        st.markdown(
            """
            ### 配對策略1(距離法)
            - 每次 formation period 先對價格取 log price。
            - 對每檔股票做標準化。
            - 對每組 pair 計算 `SSD = sum((Z_X - Z_Y)^2)`。
            - SSD 越小代表 formation period 中兩檔走勢越接近。
            """
        )
    else:
        st.markdown(
            """
            ### 配對策略2(共整合法)
            - 每次 formation period 對所有 pair 做 Engle-Granger 共整合檢定。
            - 保留 `p_value <= threshold`，再依 `coint_t` 越負越優先排序。
            - Spread 使用 `log_X = alpha + beta * log_Y + residual` 的 residual。
            """
        )
    st.markdown(
        """
        ### 兩種模式
        1. **模式一：固定 Pair 回測**  
           先篩選候選 pair，再由使用者自行選一組 pair。回測時不每週重新選 pair，而是固定使用回測起日前的 formation 參數一路回測。

        2. **模式二：正式交易策略**  
           每週自動篩選 Top 5 pair，再交易最佳 n 組。使用者不手動指定 pair。舊持倉不因下一週重選而強制平倉，會持有到觸發出場條件。

        ### 共同交易邏輯
        - 今日收盤產生訊號，明日開盤成交。
        - 只在 crossing signal 進場。
        - 不在 stop zone 進場。
        - stop loss 後必須等 `abs(z) < reentry_reset_z` 才允許重新進場。
        - 出場條件：`abs(z)<exit_z`、`abs(z)>stop_z`、`max_holding_days`、或回測結束。
        """
    )


def render_strategy_page(strategy: StrategyName) -> None:
    title = "配對策略1(距離法)" if strategy == "distance" else "配對策略2(共整合法)"
    st.title(title)
    st.caption(f"版本：{APP_VERSION}｜請先在本策略內選擇模式：模式一為手動固定 Pair 回測；模式二為每週自動 Top-N 正式策略回測。")
    settings = sidebar_settings(strategy)

    st.markdown("### 選擇本策略執行模式")
    mode = st.radio(
        "執行模式",
        options=["模式一：篩選最佳 Pair 後手動選一組固定回測", "模式二：正式交易策略，每週 Top 5 自動交易最佳 n 組", "方法說明"],
        index=0,
        horizontal=False,
        key=f"strategy_mode_{strategy}",
    )

    st.divider()

    if mode.startswith("模式一"):
        st.info("目前模式：先在參考日篩選最佳 pair，再由使用者選一組固定 pair 回測；回測期間不會每週重新篩選 pair。")
        render_mode1_tab(settings)
    elif mode.startswith("模式二"):
        st.info("目前模式：正式交易策略。每週篩選 Top 5 pair，交易其中最佳 n 組；已開倉部位不因下一週重新篩選而強制平倉。")
        render_mode2_formal_strategy(settings)
    else:
        render_method_notes(settings)


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
    else:
        st.session_state["selected_strategy"] = None
        st.rerun()


if __name__ == "__main__":
    main()
