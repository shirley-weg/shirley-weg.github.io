from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import html
import re
import contextlib
import io
import logging
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller


logging.getLogger("yfinance").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore")


# Taiwan single-stock futures underlyings (ordinary stocks only; ETF futures excluded)
# Fields: code, name, industry
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

KNOWN_NAMES = {item["code"]: item["name"] for item in single_stock_futures}

PERIOD = "3y"
INTERVAL = "1d"
MIN_OBS = 60
CORR_THRESHOLD = 0.9

ADF_P_THRESHOLD = 0.05
MIN_HALF_LIFE = 2
MAX_HALF_LIFE = 60
MAX_TREND_STRENGTH = 1.0
MIN_CROSSINGS_PER_YEAR = 4
TOP_N = 10
STOP_Z = 3.5
MAX_OPEN_POSITIONS_PER_DIRECTION = 3
MA_WINDOW = 5


@dataclass(frozen=True)
class Config:
    lookback: int
    entry_z: float
    exit_z: float
    capital: float
    broker_fee: float
    sell_tax: float
    integer_shares: bool


st.set_page_config(page_title="Trading Strategy Lab", layout="wide")
st.markdown(
    """
    <style>
    .block-container { padding-top: 1.7rem; }
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
    }
    .strategy-card h3 {
      margin-top: 0;
      margin-bottom: 8px;
    }
    .strategy-card p {
      color: #607080;
      margin-bottom: 0;
      line-height: 1.7;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def main() -> None:
    if "selected_strategy" not in st.session_state:
        st.session_state["selected_strategy"] = None
    if "a_code_input" not in st.session_state:
        st.session_state["a_code_input"] = ""
    if "b_code_input" not in st.session_state:
        st.session_state["b_code_input"] = ""

    if st.session_state["selected_strategy"] is None:
        render_strategy_selector()
        return

    st.title("Trading Strategy Lab")
    st.caption("Pair trading backtest with rolling OLS hedge ratio.")
    settings = sidebar_settings()
    render_backtest(settings)


def render_strategy_selector() -> None:
    st.title("Trading Strategy Lab")
    st.caption("請先選擇要使用的交易策略。")

    st.markdown(
        """
        <div class="strategy-card">
          <h3>策略1：Pair Trading 回測</h3>
          <p>
            使用 rolling OLS 估計 spread 與 z-score，並使用 OLS beta
            作為雙邊部位權重。
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("進入策略1", type="primary", use_container_width=True):
            st.session_state["selected_strategy"] = "strategy_1"
            st.rerun()

    st.info("目前已開放策略1；後續可以在這個入口頁繼續新增策略2、策略3。")


def sidebar_settings() -> dict[str, object]:
    if st.sidebar.button("返回策略選擇"):
        st.session_state["selected_strategy"] = None
        st.rerun()

    st.sidebar.caption("目前策略：策略1 Pair Trading")
    st.sidebar.divider()

    st.sidebar.header("回測設定")
    col_a, col_b = st.sidebar.columns(2)
    with col_a:
        a_code = st.text_input("A code", key="a_code_input").strip().upper()
    with col_b:
        b_code = st.text_input("B code", key="b_code_input").strip().upper()

    today = pd.Timestamp.today().date()
    start_default = (pd.Timestamp(today) - pd.DateOffset(years=2)).date()
    start = st.sidebar.date_input("Backtest start", value=start_default)
    end = st.sidebar.date_input("Backtest end", value=today)
    benchmark = st.sidebar.text_input("Benchmark", value="^TWII")

    a_name = resolve_stock_name(a_code) if a_code else ""
    b_name = resolve_stock_name(b_code) if b_code else ""
    if a_code or b_code:
        st.sidebar.caption(
            f"辨識結果：{format_stock_label(a_code, a_name)} / {format_stock_label(b_code, b_name)}"
        )

    st.sidebar.divider()
    st.sidebar.header("策略參數")
    lookback = st.sidebar.slider("Rolling OLS lookback", 40, 260, 120, 10)
    entry_z = st.sidebar.slider("Entry z-score", 0.5, 4.0, 2.0, 0.1)
    exit_z = st.sidebar.slider("Exit z-score", -1.0, 1.0, 0.0, 0.1)
    capital = st.sidebar.number_input("Capital per pair", min_value=10_000, value=100_000, step=10_000)
    integer_shares = st.sidebar.toggle("整股交易", value=True)

    st.sidebar.divider()
    st.sidebar.header("交易成本")
    broker_fee = st.sidebar.number_input("Broker fee", min_value=0.0, value=0.001425, step=0.0001, format="%.6f")
    sell_tax = st.sidebar.number_input("Sell tax", min_value=0.0, value=0.001425, step=0.0001, format="%.6f")

    return {
        "a_code": a_code,
        "a_name": a_name,
        "b_code": b_code,
        "b_name": b_name,
        "start": pd.Timestamp(start),
        "end": pd.Timestamp(end),
        "benchmark": benchmark,
        "config": Config(
            lookback=lookback,
            entry_z=entry_z,
            exit_z=exit_z,
            capital=float(capital),
            broker_fee=float(broker_fee),
            sell_tax=float(sell_tax),
            integer_shares=integer_shares,
        ),
    }


def render_backtest(settings: dict[str, object]) -> None:
    a_code = str(settings["a_code"])
    b_code = str(settings["b_code"])
    a_name = str(settings["a_name"])
    b_name = str(settings["b_name"])
    start = pd.Timestamp(settings["start"])
    end = pd.Timestamp(settings["end"])
    config: Config = settings["config"]  # type: ignore[assignment]

    left, right = st.columns([0.52, 0.48])

    with left:
        st.subheader("回測標的")
        if a_code and b_code:
            st.markdown(f"### {format_stock_label(a_code, a_name)} / {format_stock_label(b_code, b_name)}")
            st.write("訊號使用 t 日 Close；交易使用 t+1 日 Open。")
        else:
            st.info("請先在左側輸入 A code / B code，或從右側推薦 pair 套用到回測。")
        run = st.button("Run backtest", type="primary", use_container_width=True)

    with right:
        render_pair_screening_panel()

    if not run:
        st.info("調整左側設定後按 Run backtest。")
        return
    if not a_code or not b_code:
        st.error("請先輸入 A code 和 B code，或從推薦 pair 套用到回測。")
        return
    if a_code == b_code:
        st.error("A code 和 B code 不能相同。")
        return
    if start >= end:
        st.error("Backtest start 必須早於 end。")
        return

    try:
        with st.spinner("下載價格並執行回測..."):
            warmup = max(config.lookback * 3, 365)
            download_start = start - pd.DateOffset(days=warmup)
            open_df, close_df = download_ohlc([a_code, b_code], download_start, end)
            benchmark = download_benchmark(str(settings["benchmark"]), download_start, end)
            result = run_backtest(open_df, close_df, benchmark, a_code, b_code, start, end, config)
    except Exception as exc:
        st.error(f"回測失敗：{exc}")
        return

    show_summary(result["summary"])
    show_latest_trading_signal(result, a_code, b_code, a_name, b_name, config)
    show_charts(result, a_code, b_code, str(settings["benchmark"]), config)
    show_tables(result)


def render_pair_screening_panel() -> None:
    st.subheader("細產業 Pair 推薦")
    industry_groups = build_industry_groups()
    industry_options = sorted(industry_groups.keys())

    industry = st.selectbox("選擇細產業", industry_options, index=industry_options.index("半導體業") if "半導體業" in industry_options else 0)
    stocks = industry_groups[industry]
    st.caption(
        f"此細產業共有 {len(stocks)} 檔可篩選股票；"
        "篩選邏輯使用高相關性、ADF、half-life、trend strength 與 crossings。"
    )

    run_screen = st.button("篩選最佳 10 組 Pair", use_container_width=True)

    if run_screen:
        with st.spinner("下載價格並篩選 pair..."):
            try:
                best_pairs = screen_best_pairs_by_industry(industry, tuple(stocks))
            except Exception as exc:
                st.error(f"篩選失敗：{exc}")
                return
        st.session_state["screened_industry"] = industry
        st.session_state["screened_pairs"] = best_pairs.to_dict("records")

    records = st.session_state.get("screened_pairs", [])
    screened_industry = st.session_state.get("screened_industry")

    if not records or screened_industry != industry:
        st.info("請選擇細產業後按「篩選最佳 10 組 Pair」。")
        return

    best_pairs_df = pd.DataFrame(records)
    if best_pairs_df.empty:
        st.warning("這個細產業目前沒有符合篩選條件的 pair。")
        return

    display_cols = [
        "rank",
        "stock_A_code", "stock_A_name",
        "stock_B_code", "stock_B_name",
        "correlation",
        "beta_hedge_ratio",
        "adf_pvalue",
        "half_life",
        "trend_strength",
        "crossings_per_year",
        "score",
        "suitable",
    ]
    show_df = best_pairs_df[[c for c in display_cols if c in best_pairs_df.columns]].copy()
    for c in ["correlation", "beta_hedge_ratio", "adf_pvalue", "half_life", "trend_strength", "crossings_per_year"]:
        if c in show_df.columns:
            show_df[c] = show_df[c].astype(float).round(4)
    st.dataframe(show_df, use_container_width=True, hide_index=True)

    pair_labels = [
        f"{int(row['rank'])}. {row['stock_A_code']} {row['stock_A_name']} / "
        f"{row['stock_B_code']} {row['stock_B_name']} | "
        f"corr={float(row['correlation']):.3f} | score={int(row['score'])}"
        for _, row in best_pairs_df.iterrows()
    ]
    selected_label = st.selectbox("套用推薦 pair 到回測", pair_labels)
    selected_idx = pair_labels.index(selected_label)
    selected = best_pairs_df.iloc[selected_idx]

    if st.button("套用選取 pair", type="primary", use_container_width=True):
        st.session_state["a_code_input"] = str(selected["stock_A_code"])
        st.session_state["b_code_input"] = str(selected["stock_B_code"])
        st.rerun()

def build_industry_groups() -> dict[str, list[tuple[str, str]]]:
    groups: dict[str, list[tuple[str, str]]] = {}
    for item in single_stock_futures:
        industry = str(item["industry"])
        code = str(item["code"])
        name = str(item["name"])
        groups.setdefault(industry, []).append((code, name))
    return groups



def candidate_yahoo_tickers(code: str) -> list[str]:
    code = str(code).strip().upper()
    if not code:
        return []
    if "." in code or code.startswith("^") or "-" in code or "=" in code:
        return [code]
    return [f"{code}.TW", f"{code}.TWO"]


def extract_close_series_from_single_download(raw: pd.DataFrame) -> pd.Series:
    if raw is None or raw.empty:
        return pd.Series(dtype=float)

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            close = raw["Close"]
            if isinstance(close, pd.DataFrame):
                return close.iloc[:, 0]
            return close

        if "Close" in raw.columns.get_level_values(1):
            for col in raw.columns:
                if col[1] == "Close":
                    return raw[col]

        return pd.Series(dtype=float)

    if "Close" not in raw.columns:
        return pd.Series(dtype=float)

    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        return close.iloc[:, 0]
    return close


def extract_ohlc_from_single_download(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        data: dict[str, pd.Series] = {}
        for field in ["Open", "Close"]:
            if field in raw.columns.get_level_values(0):
                obj = raw[field]
                data[field] = obj.iloc[:, 0] if isinstance(obj, pd.DataFrame) else obj
            elif field in raw.columns.get_level_values(1):
                for col in raw.columns:
                    if col[1] == field:
                        data[field] = raw[col]
                        break
        if "Open" in data and "Close" in data:
            return pd.DataFrame(data)
        return pd.DataFrame()

    if "Open" not in raw.columns or "Close" not in raw.columns:
        return pd.DataFrame()

    return raw[["Open", "Close"]].copy()



@st.cache_data(ttl=3600, show_spinner=False)
def screen_best_pairs_by_industry(industry: str, stocks: tuple[tuple[str, str], ...]) -> pd.DataFrame:
    if len(stocks) <= 1:
        return pd.DataFrame()

    price_df = download_screening_prices(stocks, PERIOD, INTERVAL)
    group_price = get_group_price(price_df, stocks)

    if group_price.shape[1] <= 1:
        return pd.DataFrame()

    log_price = np.log(group_price)
    corr_pairs = extract_high_corr_pairs(industry, log_price, CORR_THRESHOLD)

    if corr_pairs.empty:
        return pd.DataFrame()

    screening_records: list[dict[str, object]] = []

    for _, row in corr_pairs.iterrows():
        col_a = f"{row['stock_A_code']}_{row['stock_A_name']}"
        col_b = f"{row['stock_B_code']}_{row['stock_B_name']}"

        if col_a not in price_df.columns or col_b not in price_df.columns:
            continue

        pair_price = price_df[[col_a, col_b]].dropna().copy()
        if len(pair_price) < MIN_OBS:
            continue

        pair_log = np.log(pair_price)
        log_a = pair_log[col_a]
        log_b = pair_log[col_b]

        alpha, beta, fitted, spread = ols_spread(log_a, log_b)
        spread_series = pd.Series(spread, index=pair_log.index)
        spread_mean = float(spread_series.mean())
        spread_std = float(spread_series.std())

        adf_stat, adf_pvalue = adf_test(spread_series)
        half_life = estimate_half_life(spread_series)
        trend_slope, trend_pvalue, trend_strength = trend_strength_test(spread_series)
        crossings, crossings_per_year = count_crossings(spread_series)

        record = row.to_dict()
        record.update({
            "alpha": float(alpha),
            "beta_hedge_ratio": float(beta),
            "spread_method": "ols_regression_residual",
            "spread_mean": spread_mean,
            "spread_std": spread_std,
            "spread_min": float(spread_series.min()),
            "spread_max": float(spread_series.max()),
            "adf_stat": float(adf_stat),
            "adf_pvalue": float(adf_pvalue),
            "adf_pass_5pct": bool(adf_pvalue < ADF_P_THRESHOLD),
            "half_life": float(half_life) if pd.notna(half_life) else np.nan,
            "half_life_reasonable": bool(pd.notna(half_life) and MIN_HALF_LIFE <= half_life <= MAX_HALF_LIFE),
            "trend_slope": float(trend_slope),
            "trend_pvalue": float(trend_pvalue),
            "trend_strength": float(trend_strength),
            "no_obvious_trend": bool(trend_strength < MAX_TREND_STRENGTH),
            "mean_crossings": int(crossings),
            "crossings_per_year": float(crossings_per_year),
            "enough_crossings": bool(crossings_per_year >= MIN_CROSSINGS_PER_YEAR),
            "start_date": spread_series.index.min(),
            "end_date": spread_series.index.max(),
            "observations": int(len(spread_series)),
        })
        screening_records.append(record)

    screening_df = pd.DataFrame(screening_records)
    if screening_df.empty:
        return pd.DataFrame()

    screening_df["score"] = screening_df.apply(score_pair, axis=1)
    screening_df["suitable"] = (
        screening_df["adf_pass_5pct"]
        & screening_df["half_life_reasonable"]
        & screening_df["no_obvious_trend"]
        & screening_df["enough_crossings"]
    )

    screening_df = screening_df.sort_values(
        ["suitable", "score", "adf_pvalue", "trend_strength"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)

    best_pairs_df = screening_df[screening_df["suitable"]].copy()
    if best_pairs_df.empty:
        best_pairs_df = screening_df.head(TOP_N).copy()
    else:
        best_pairs_df = best_pairs_df.head(TOP_N).copy()

    best_pairs_df = best_pairs_df.reset_index(drop=True)
    best_pairs_df.insert(0, "rank", np.arange(1, len(best_pairs_df) + 1))
    return best_pairs_df

@st.cache_data(ttl=3600, show_spinner=False)
def download_screening_prices(stocks: tuple[tuple[str, str], ...], period: str, interval: str) -> pd.DataFrame:
    import yfinance as yf

    cache_dir = Path(__file__).resolve().parent / ".cache" / "yfinance"
    cache_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(yf, "set_tz_cache_location"):
        yf.set_tz_cache_location(str(cache_dir))

    price_dict: dict[str, pd.Series] = {}

    for code, name in stocks:
        series = pd.Series(dtype=float)

        for ticker in candidate_yahoo_tickers(code):
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    with contextlib.redirect_stderr(io.StringIO()):
                        raw = yf.download(
                            tickers=ticker,
                            period=period,
                            interval=interval,
                            auto_adjust=True,
                            progress=False,
                            threads=False,
                        )

                series = extract_close_series_from_single_download(raw)
                series = pd.to_numeric(series, errors="coerce").dropna()
                series = series[series > 0]

                if len(series) >= MIN_OBS:
                    break
            except Exception:
                series = pd.Series(dtype=float)

        if len(series) >= MIN_OBS:
            price_dict[f"{code}_{name}"] = series

    price_df = pd.DataFrame(price_dict).sort_index()
    price_df = price_df.ffill().dropna(how="all")
    return price_df


def extract_close_series(raw: pd.DataFrame, ticker: str) -> pd.Series:
    if isinstance(raw.columns, pd.MultiIndex):
        level0 = raw.columns.get_level_values(0)
        level1 = raw.columns.get_level_values(1)

        if ticker in level0 and "Close" in raw[ticker].columns:
            return raw[ticker]["Close"]

        if "Close" in level0 and ticker in raw["Close"].columns:
            return raw["Close"][ticker]

        if ticker in level1 and "Close" in level0:
            return raw[("Close", ticker)]

        raise KeyError(f"Cannot find Close for {ticker}")

    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        if ticker in close.columns:
            return close[ticker]
        return close.iloc[:, 0]
    return close


def get_group_price(price_df: pd.DataFrame, stock_list: tuple[tuple[str, str], ...]) -> pd.DataFrame:
    cols = []
    for code, name in stock_list:
        col = f"{code}_{name}"
        if col in price_df.columns:
            cols.append(col)

    if not cols:
        return pd.DataFrame()

    group_price = price_df[cols].copy()
    group_price = group_price.dropna(axis=1, how="all")
    group_price = group_price.ffill().dropna()
    group_price = group_price.loc[:, (group_price > 0).all(axis=0)]

    return group_price


def extract_high_corr_pairs(group: str, log_price_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    corr = log_price_df.corr()
    cols = corr.columns.tolist()

    records = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c = corr.iloc[i, j]
            if pd.notna(c) and c > threshold:
                code_a, name_a = cols[i].split("_", 1)
                code_b, name_b = cols[j].split("_", 1)
                records.append({
                    "group": group,
                    "stock_A_code": code_a,
                    "stock_A_name": name_a,
                    "stock_B_code": code_b,
                    "stock_B_name": name_b,
                    "correlation": float(c),
                })

    return pd.DataFrame(records)


def ols_spread(log_a: pd.Series, log_b: pd.Series) -> tuple[float, float, np.ndarray, np.ndarray]:
    y = np.asarray(log_a, dtype=float)
    x = np.asarray(log_b, dtype=float)

    X = np.column_stack([np.ones(len(x)), x])
    alpha, beta = np.linalg.lstsq(X, y, rcond=None)[0]

    fitted = alpha + beta * x
    spread = y - fitted

    return float(alpha), float(beta), fitted, spread


def adf_test(spread: pd.Series) -> tuple[float, float]:
    s = pd.Series(spread).dropna()
    try:
        result = adfuller(s, autolag="AIC")
        return float(result[0]), float(result[1])
    except Exception:
        return float("nan"), float("nan")


def estimate_half_life(spread: pd.Series) -> float:
    s = pd.Series(spread).dropna()
    lagged = s.shift(1).dropna()
    delta = s.diff().dropna()

    idx = lagged.index.intersection(delta.index)
    y = delta.loc[idx]
    x = lagged.loc[idx]

    if len(y) < 2:
        return float("nan")

    try:
        X = sm.add_constant(x)
        model = sm.OLS(y, X).fit()
        beta = model.params.iloc[1]
    except Exception:
        return float("nan")

    if beta >= 0:
        return float("nan")

    return float(-np.log(2) / beta)


def trend_strength_test(spread: pd.Series) -> tuple[float, float, float]:
    s = pd.Series(spread).dropna()
    if len(s) < 2:
        return float("nan"), float("nan"), float("inf")

    t = np.arange(len(s))
    try:
        X = sm.add_constant(t)
        model = sm.OLS(s.values, X).fit()
        slope = model.params[1]
        pvalue = model.pvalues[1]
    except Exception:
        return float("nan"), float("nan"), float("inf")

    total_change = abs(slope) * len(s)
    std = s.std()
    strength = np.inf if std == 0 else total_change / std

    return float(slope), float(pvalue), float(strength)


def count_crossings(spread: pd.Series) -> tuple[int, float]:
    s = pd.Series(spread).dropna()
    centered = s - s.mean()
    signs = np.sign(centered)

    crossings = int(np.sum(signs.shift(1) * signs < 0))
    years = len(s) / 252

    return crossings, float(crossings / years if years > 0 else np.nan)


def score_pair(row: pd.Series) -> int:
    score = 0

    adf_pvalue = row["adf_pvalue"]
    if pd.notna(adf_pvalue):
        if adf_pvalue < 0.01:
            score += 35
        elif adf_pvalue < 0.05:
            score += 25
        elif adf_pvalue < 0.10:
            score += 10

    hl = row["half_life"]
    if pd.notna(hl) and MIN_HALF_LIFE <= hl <= MAX_HALF_LIFE:
        score += 25
        if 5 <= hl <= 30:
            score += 10

    trend_strength = row["trend_strength"]
    if pd.notna(trend_strength):
        if trend_strength < 0.5:
            score += 20
        elif trend_strength < MAX_TREND_STRENGTH:
            score += 10

    crossings_per_year = row["crossings_per_year"]
    if pd.notna(crossings_per_year):
        if crossings_per_year >= 12:
            score += 10
        elif crossings_per_year >= MIN_CROSSINGS_PER_YEAR:
            score += 5

    correlation = row["correlation"]
    if pd.notna(correlation):
        if correlation >= 0.95:
            score += 10
        elif correlation >= 0.90:
            score += 7
        elif correlation >= 0.80:
            score += 3

    return score


def format_stock_label(code: str, name: str) -> str:
    code = str(code).strip()
    name = str(name).strip()
    if code and name:
        return f"{code} {name}"
    return code or "尚未選擇"


def resolve_stock_name(code: str) -> str:
    code = str(code).strip().upper()
    if not code:
        return ""

    if code in KNOWN_NAMES:
        return KNOWN_NAMES[code]

    twse_name = fetch_twse_stock_name(code)
    if twse_name:
        return twse_name

    yahoo_name = fetch_yahoo_tw_stock_name(code)
    if yahoo_name:
        return yahoo_name

    return ""


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_twse_name_map() -> dict[str, str]:
    url = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"

    try:
        response = requests.get(url, timeout=8)
        response.raise_for_status()
        data = response.json()
    except Exception:
        return {}

    name_map: dict[str, str] = {}
    for row in data:
        code = str(row.get("Code", "")).strip()
        name = str(row.get("Name", "")).strip()
        if code and name:
            name_map[code] = name

    return name_map


def fetch_twse_stock_name(code: str) -> str:
    name_map = fetch_twse_name_map()
    return name_map.get(code, "")


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_yahoo_tw_stock_name(code: str) -> str:
    for ticker in candidate_yahoo_tickers(code):
        url = f"https://tw.stock.yahoo.com/quote/{ticker}"

        try:
            response = requests.get(
                url,
                timeout=8,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            response.raise_for_status()
        except Exception:
            continue

        text = response.text
        match = re.search(r"<title>(.*?)</title>", text, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            continue

        title = html.unescape(match.group(1))
        title = re.sub(r"\s+", " ", title).strip()

        # Yahoo Taiwan title usually looks like:
        # 兆豐金 (2886.TW) 走勢圖 - Yahoo奇摩股市
        if " (" in title:
            candidate = title.split(" (", 1)[0].strip()
            if candidate and candidate != code:
                return candidate

    return ""


@st.cache_data(ttl=3600, show_spinner=False)
def download_ohlc(codes: list[str], start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    import yfinance as yf

    cache_dir = Path(__file__).resolve().parent / ".cache" / "yfinance"
    cache_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(yf, "set_tz_cache_location"):
        yf.set_tz_cache_location(str(cache_dir))

    clean_codes = sorted({c.strip().upper() for c in codes if c.strip()})
    open_series: dict[str, pd.Series] = {}
    close_series: dict[str, pd.Series] = {}

    for code in clean_codes:
        downloaded = pd.DataFrame()

        for ticker in candidate_yahoo_tickers(code):
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    with contextlib.redirect_stderr(io.StringIO()):
                        raw = yf.download(
                            tickers=ticker,
                            start=start.strftime("%Y-%m-%d"),
                            end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                            interval="1d",
                            auto_adjust=True,
                            progress=False,
                            threads=False,
                        )

                downloaded = extract_ohlc_from_single_download(raw)
                downloaded = downloaded.apply(pd.to_numeric, errors="coerce").dropna()
                downloaded = downloaded[(downloaded["Open"] > 0) & (downloaded["Close"] > 0)]

                if not downloaded.empty:
                    break
            except Exception:
                downloaded = pd.DataFrame()

        if downloaded.empty:
            raise ValueError(
                f"Yahoo Finance returned no price data for {code}. "
                f"Tried {', '.join(candidate_yahoo_tickers(code))}."
            )

        downloaded.index = pd.to_datetime(downloaded.index).tz_localize(None).normalize()
        open_series[code] = downloaded["Open"]
        close_series[code] = downloaded["Close"]

    open_df = pd.DataFrame(open_series).sort_index()
    close_df = pd.DataFrame(close_series).sort_index()

    valid = open_df.notna().all(axis=1) & close_df.notna().all(axis=1)
    open_df = normalize_index(open_df.loc[valid].sort_index())
    close_df = normalize_index(close_df.loc[valid].sort_index())

    if open_df.empty or close_df.empty:
        raise ValueError("Downloaded data has no complete Open/Close rows.")

    return open_df, close_df


@st.cache_data(ttl=3600, show_spinner=False)
def download_benchmark(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    import yfinance as yf

    raw = yf.download(
        tickers=ticker,
        start=start.strftime("%Y-%m-%d"),
        end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=True,
        progress=False,
    )
    if raw.empty:
        return pd.Series(dtype=float, name=ticker)
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close.name = ticker
    return normalize_index(close.dropna())


def to_yahoo_ticker(code: str, suffix: str = ".TW") -> str:
    if "." in code or code.startswith("^") or "-" in code or "=" in code:
        return code
    return f"{code}{suffix}"


def extract_price_field(raw: pd.DataFrame, field: str, codes: list[str], suffix: str) -> pd.DataFrame:
    if isinstance(raw.columns, pd.MultiIndex):
        if field in raw.columns.get_level_values(0):
            out = raw[field].copy()
        else:
            out = pd.concat(
                {ticker: raw[(ticker, field)] for ticker in raw.columns.get_level_values(0).unique() if (ticker, field) in raw.columns},
                axis=1,
            )
    else:
        out = raw[[field]].copy()
        out.columns = [codes[0]]

    out = out.rename(columns={to_yahoo_ticker(code, suffix): code for code in codes})
    missing = [code for code in codes if code not in out.columns]
    if missing:
        raise ValueError(f"Missing price fields for: {', '.join(missing)}")
    return out[codes].apply(pd.to_numeric, errors="coerce")


def normalize_index(obj: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    out = obj.copy()
    out.index = pd.to_datetime(out.index).tz_localize(None).normalize()
    return out


def rolling_spread_signal(close_df: pd.DataFrame, a_code: str, b_code: str, lookback: int) -> pd.DataFrame:
    logp = np.log(close_df[[a_code, b_code]].dropna())
    rows: list[dict[str, object]] = []
    for i in range(lookback, len(logp) - 1):
        signal_date = logp.index[i]
        exec_date = logp.index[i + 1]
        train = logp.iloc[i - lookback : i]
        alpha, beta = ols(train[a_code].to_numpy(), train[b_code].to_numpy())
        train_spread = train[a_code].to_numpy() - (alpha + beta * train[b_code].to_numpy())
        spread_mean = float(train_spread.mean())
        spread_std = float(train_spread.std(ddof=1))
        if not np.isfinite(spread_std) or spread_std == 0:
            continue
        spread = float(logp.iloc[i][a_code] - alpha - beta * logp.iloc[i][b_code])
        zscore = (spread - spread_mean) / spread_std
        rows.append({"signal_date": signal_date, "exec_date": exec_date, "alpha": alpha, "beta": beta, "spread": spread, "spread_mean": spread_mean, "spread_std": spread_std, "zscore": zscore})
    if not rows:
        return pd.DataFrame(columns=["exec_date", "alpha", "beta", "spread", "spread_mean", "spread_std", "zscore", "ma5"]).rename_axis("signal_date")
    out = pd.DataFrame(rows).set_index("signal_date")
    out["ma5"] = out["spread"].rolling(MA_WINDOW).mean()
    return out

def latest_spread_signal(close_df: pd.DataFrame, a_code: str, b_code: str, lookback: int) -> dict[str, object]:
    logp = np.log(close_df[[a_code, b_code]].dropna())
    if len(logp) <= lookback:
        return {}

    rows: list[dict[str, object]] = []

    # Include the latest available close, even though it has no next-day exec_date yet.
    for i in range(lookback, len(logp)):
        signal_date = logp.index[i]
        train = logp.iloc[i - lookback : i]
        if len(train) < lookback:
            continue

        alpha, beta = ols(train[a_code].to_numpy(), train[b_code].to_numpy())
        train_spread = train[a_code].to_numpy() - (alpha + beta * train[b_code].to_numpy())
        spread_mean = float(train_spread.mean())
        spread_std = float(train_spread.std(ddof=1))
        if not np.isfinite(spread_std) or spread_std == 0:
            continue

        spread = float(logp.iloc[i][a_code] - alpha - beta * logp.iloc[i][b_code])
        zscore = (spread - spread_mean) / spread_std
        rows.append({
            "signal_date": signal_date,
            "alpha": float(alpha),
            "beta": float(beta),
            "spread": spread,
            "spread_mean": spread_mean,
            "spread_std": spread_std,
            "zscore": float(zscore),
        })

    if not rows:
        return {}

    latest_df = pd.DataFrame(rows).set_index("signal_date")
    latest_df["ma5"] = latest_df["spread"].rolling(MA_WINDOW).mean()

    latest_date = latest_df.index[-1]
    row = latest_df.iloc[-1]

    return {
        "signal_date": latest_date,
        "alpha": float(row["alpha"]),
        "beta": float(row["beta"]),
        "spread": float(row["spread"]),
        "spread_mean": float(row["spread_mean"]),
        "spread_std": float(row["spread_std"]),
        "ma5": float(row["ma5"]) if pd.notna(row["ma5"]) else np.nan,
        "zscore": float(row["zscore"]),
        "a_close": float(close_df.loc[latest_date, a_code]),
        "b_close": float(close_df.loc[latest_date, b_code]),
    }

def run_backtest(open_df: pd.DataFrame, close_df: pd.DataFrame, benchmark: pd.Series, a_code: str, b_code: str, start: pd.Timestamp, end: pd.Timestamp, config: Config) -> dict[str, object]:
    full_close = close_df.loc[close_df.index <= end].copy()
    full_open = open_df.loc[open_df.index <= end].copy()
    if len(full_close) <= config.lookback + 5:
        raise ValueError("The selected date range is too short for this lookback.")
    signals = rolling_spread_signal(full_close, a_code, b_code, config.lookback)
    latest_signal = latest_spread_signal(full_close, a_code, b_code, config.lookback)
    signals = signals.loc[(signals.index >= start) & (signals.index <= end)].copy()
    trades = backtest_pair(signals, full_open, full_close, a_code, b_code, config)
    equity = build_equity(trades, full_close, start, config.capital)
    equity["daily_pnl"] = equity["cumulative_pnl"].diff().fillna(equity["cumulative_pnl"])
    equity["daily_return"] = equity["daily_pnl"] / config.capital
    equity["drawdown"] = equity["equity"] / equity["equity"].cummax() - 1
    comparison = return_comparison(equity, benchmark, config.capital)
    summary = summarize(equity, trades, config.capital)
    price_history = full_close.loc[(full_close.index >= start) & (full_close.index <= end), [a_code, b_code]]
    weight_cols = ["entry_date", "a_weight", "b_weight", "a_sign", "b_sign", "direction", "weight_method"]
    weights = trades[weight_cols].copy() if not trades.empty else pd.DataFrame(columns=weight_cols)
    return {"signals": signals, "latest_signal": latest_signal, "trades": trades, "equity": equity, "comparison": comparison, "summary": summary, "price_history": price_history, "weights": weights}


def backtest_pair(signals: pd.DataFrame, open_df: pd.DataFrame, close_df: pd.DataFrame, a_code: str, b_code: str, config: Config) -> pd.DataFrame:
    trades: list[dict[str, object]] = []
    open_positions: list[dict[str, object]] = []

    for signal_date, row in signals.iterrows():
        exec_date = row["exec_date"]
        if exec_date not in open_df.index:
            continue

        a_price = float(open_df.loc[exec_date, a_code])
        b_price = float(open_df.loc[exec_date, b_code])
        z = float(row["zscore"])
        spread = float(row["spread"])
        ma5 = float(row["ma5"]) if "ma5" in row and pd.notna(row["ma5"]) else np.nan
        beta = float(row["beta"])

        if not np.isfinite(beta) or beta <= 0:
            continue

        still_open: list[dict[str, object]] = []
        for position in open_positions:
            stop_loss = should_stop_loss(str(position["direction"]), z)
            normal_exit = should_exit(str(position["direction"]), z, config.exit_z)

            if stop_loss or normal_exit:
                exit_cost = transaction_cost(
                    float(position["a_shares"]) * a_price,
                    "sell" if position["a_sign"] == 1 else "buy",
                    config,
                ) + transaction_cost(
                    float(position["b_shares"]) * b_price,
                    "sell" if position["b_sign"] == 1 else "buy",
                    config,
                )
                pnl = trade_pnl(position, a_price, b_price) - exit_cost
                position.update({
                    "exit_signal_date": signal_date,
                    "exit_date": exec_date,
                    "exit_zscore": z,
                    "exit_spread": spread,
                    "exit_ma5": ma5,
                    "a_exit_price": a_price,
                    "b_exit_price": b_price,
                    "exit_cost": exit_cost,
                    "total_cost": float(position["entry_cost"]) + exit_cost,
                    "holding_days": (pd.Timestamp(exec_date) - pd.Timestamp(position["entry_date"])).days,
                    "pnl": pnl,
                    "return_on_gross_exposure": pnl / float(position["gross_exposure"]),
                    "exit_reason": "z_stop_loss" if stop_loss else "normal_exit",
                    "status": "closed",
                })
                trades.append(position)
            else:
                still_open.append(position)

        open_positions = still_open

        direction, a_sign, b_sign = entry_direction(z, spread, ma5, config.entry_z)
        if direction is None:
            continue

        if is_stop_zone(z):
            continue

        same_direction_open_count = sum(
            1 for position in open_positions
            if str(position["direction"]) == direction
        )
        if same_direction_open_count >= MAX_OPEN_POSITIONS_PER_DIRECTION:
            continue

        weights = choose_weights(beta)
        a_notional = config.capital * float(weights["a_weight"])
        b_notional = config.capital * float(weights["b_weight"])
        a_shares = int(a_notional // a_price) if config.integer_shares else a_notional / a_price
        b_shares = int(b_notional // b_price) if config.integer_shares else b_notional / b_price

        if a_shares <= 0 or b_shares <= 0:
            continue

        a_entry_notional = float(a_shares * a_price)
        b_entry_notional = float(b_shares * b_price)
        entry_cost = transaction_cost(
            a_entry_notional,
            "buy" if a_sign == 1 else "sell",
            config,
        ) + transaction_cost(
            b_entry_notional,
            "buy" if b_sign == 1 else "sell",
            config,
        )

        open_positions.append({
            "a_code": a_code,
            "b_code": b_code,
            "direction": direction,
            "entry_signal_date": signal_date,
            "entry_date": exec_date,
            "entry_zscore": z,
            "entry_spread": spread,
            "entry_ma5": ma5,
            "entry_beta": beta,
            "a_sign": a_sign,
            "b_sign": b_sign,
            "a_entry_price": a_price,
            "b_entry_price": b_price,
            "a_shares": a_shares,
            "b_shares": b_shares,
            "a_entry_notional": a_entry_notional,
            "b_entry_notional": b_entry_notional,
            "gross_exposure": a_entry_notional + b_entry_notional,
            "entry_cost": entry_cost,
            "entry_layer": same_direction_open_count + 1,
            **weights,
        })

    if open_positions:
        last_date = close_df.index[-1]
        for position in open_positions:
            pnl = trade_pnl(
                position,
                float(close_df.loc[last_date, a_code]),
                float(close_df.loc[last_date, b_code]),
            )
            position.update({
                "exit_signal_date": pd.NaT,
                "exit_date": pd.NaT,
                "exit_zscore": np.nan,
                "a_exit_price": np.nan,
                "b_exit_price": np.nan,
                "exit_cost": 0.0,
                "total_cost": float(position["entry_cost"]),
                "holding_days": (last_date - pd.Timestamp(position["entry_date"])).days,
                "pnl": pnl,
                "return_on_gross_exposure": pnl / float(position["gross_exposure"]),
                "exit_reason": "still_open",
                "status": "open",
            })
            trades.append(position)

    return pd.DataFrame(trades)

def choose_weights(beta: float) -> dict[str, object]:
    return {"a_weight": 0.5, "b_weight": 0.5, "weight_method": "fixed_50_50"}


def build_equity(trades: pd.DataFrame, close_df: pd.DataFrame, start: pd.Timestamp, capital: float) -> pd.DataFrame:
    dates = close_df.loc[close_df.index >= start].index
    pnl_curve = pd.Series(0.0, index=dates, name="cumulative_pnl")
    if trades.empty:
        return pd.DataFrame({"cumulative_pnl": pnl_curve, "equity": capital + pnl_curve})
    for _, tr in trades.iterrows():
        entry_date = pd.Timestamp(tr["entry_date"])
        a_code = str(tr["a_code"])
        b_code = str(tr["b_code"])
        if tr["status"] == "closed":
            exit_date = pd.Timestamp(tr["exit_date"])
            active = dates[(dates >= entry_date) & (dates < exit_date)]
            for day in active:
                pnl_curve.loc[day] += trade_pnl(tr, float(close_df.loc[day, a_code]), float(close_df.loc[day, b_code]))
            pnl_curve.loc[dates >= exit_date] += float(tr["pnl"])
        else:
            active = dates[dates >= entry_date]
            for day in active:
                pnl_curve.loc[day] += trade_pnl(tr, float(close_df.loc[day, a_code]), float(close_df.loc[day, b_code]))
    return pd.DataFrame({"cumulative_pnl": pnl_curve, "equity": capital + pnl_curve})


def return_comparison(equity: pd.DataFrame, benchmark: pd.Series, capital: float) -> pd.DataFrame:
    strategy = equity["equity"] / capital - 1
    if benchmark.empty:
        return pd.DataFrame({"strategy_cumulative_return": strategy})
    bench = benchmark.reindex(strategy.index).ffill().dropna()
    strategy = strategy.reindex(bench.index).ffill().fillna(0)
    return pd.DataFrame({"strategy_cumulative_return": strategy, "benchmark_cumulative_return": bench / bench.iloc[0] - 1})


def summarize(equity: pd.DataFrame, trades: pd.DataFrame, capital: float) -> dict[str, float]:
    returns = equity["daily_return"].replace([np.inf, -np.inf], np.nan).dropna()
    ending = float(equity["equity"].iloc[-1])
    closed = trades[trades["status"].eq("closed")] if not trades.empty else trades
    wins = closed[closed["pnl"] > 0] if not closed.empty else closed
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1 / 365.25)
    return {"total_return": ending / capital - 1, "ending_equity": ending, "cagr": (ending / capital) ** (1 / years) - 1, "volatility": returns.std(ddof=1) * np.sqrt(252) if len(returns) else np.nan, "sharpe": sharpe(returns), "max_drawdown": float(equity["drawdown"].min()), "trade_count": float(len(trades)), "win_rate": float(len(wins) / len(closed)) if len(closed) else np.nan, "total_pnl": ending - capital}


def show_summary(summary: dict[str, float]) -> None:
    metrics = [("Total Return", pct(summary["total_return"])), ("Sharpe", num(summary["sharpe"])), ("Max DD", pct(summary["max_drawdown"])), ("CAGR", pct(summary["cagr"])), ("Win Rate", pct(summary["win_rate"])), ("Trades", f"{int(summary['trade_count'])}")]
    for i in range(0, len(metrics), 3):
        cols = st.columns(3)
        for col, (label, value) in zip(cols, metrics[i : i + 3]):
            col.metric(label, value)


def show_latest_trading_signal(result: dict[str, object], a_code: str, b_code: str, a_name: str, b_name: str, config: Config) -> None:
    latest_signal: dict[str, object] = result.get("latest_signal", {})  # type: ignore[assignment]
    trades: pd.DataFrame = result["trades"]  # type: ignore[assignment]

    st.subheader("Latest Trading Signal")

    if not latest_signal:
        st.info("目前資料不足，無法產生最新交易訊號。")
        return

    latest_date = pd.Timestamp(latest_signal["signal_date"])
    latest_z = float(latest_signal["zscore"])
    latest_spread = float(latest_signal["spread"])
    latest_ma5 = float(latest_signal["ma5"]) if "ma5" in latest_signal and pd.notna(latest_signal["ma5"]) else np.nan
    latest_beta = float(latest_signal["beta"])
    a_close = float(latest_signal["a_close"])
    b_close = float(latest_signal["b_close"])

    open_positions = trades[trades["status"].eq("open")].copy() if not trades.empty and "status" in trades.columns else pd.DataFrame()
    direction, a_sign, b_sign = entry_direction(latest_z, latest_spread, latest_ma5, config.entry_z)
    in_stop_zone = is_stop_zone(latest_z)

    if direction is None:
        signal_text = "無新進場訊號"
        layer_text = "無"
        a_action = "無"
        b_action = "無"
        a_signed_weight = np.nan
        b_signed_weight = np.nan
        a_suggested_shares = 0
        b_suggested_shares = 0
    elif in_stop_zone:
        signal_text = f"有 {direction} 訊號，但 z-score 已進入停損區，不開新倉"
        layer_text = "不開倉"
        a_action = "無"
        b_action = "無"
        a_signed_weight = np.nan
        b_signed_weight = np.nan
        a_suggested_shares = 0
        b_suggested_shares = 0
    else:
        same_direction_open_count = 0
        if not open_positions.empty:
            same_direction_open_count = int((open_positions["direction"].astype(str) == direction).sum())

        if same_direction_open_count >= MAX_OPEN_POSITIONS_PER_DIRECTION:
            signal_text = f"有 {direction} 訊號，但同方向已滿 {MAX_OPEN_POSITIONS_PER_DIRECTION} 層，不再加碼"
            layer_text = f"已滿 {MAX_OPEN_POSITIONS_PER_DIRECTION} 層"
            a_action = "無"
            b_action = "無"
            a_signed_weight = np.nan
            b_signed_weight = np.nan
            a_suggested_shares = 0
            b_suggested_shares = 0
        else:
            weights = choose_weights(latest_beta)
            a_signed_weight = float(weights["a_weight"]) * a_sign
            b_signed_weight = float(weights["b_weight"]) * b_sign
            a_notional = config.capital * float(weights["a_weight"])
            b_notional = config.capital * float(weights["b_weight"])
            a_suggested_shares = int(a_notional // a_close) if config.integer_shares else a_notional / a_close
            b_suggested_shares = int(b_notional // b_close) if config.integer_shares else b_notional / b_close
            layer_text = f"第 {same_direction_open_count + 1} 層 / 最多 {MAX_OPEN_POSITIONS_PER_DIRECTION} 層"
            signal_text = f"新進場訊號：{direction}"
            a_action = action_text(a_sign)
            b_action = action_text(b_sign)

    if open_positions.empty:
        stop_loss_text = "無持倉"
        exit_text = "無持倉"
    else:
        stop_loss_count = int(open_positions["direction"].astype(str).apply(lambda d: should_stop_loss(d, latest_z)).sum())
        exit_count = int(open_positions["direction"].astype(str).apply(lambda d: should_exit(d, latest_z, config.exit_z)).sum())
        stop_loss_text = "是" if stop_loss_count > 0 else "否"
        exit_text = "是" if exit_count > 0 else "否"

    render_latest_cards([
        ("最新日期", latest_date.strftime("%Y-%m-%d")),
        ("最新 z-score", f"{latest_z:.2f}"),
        ("是否停損", stop_loss_text),
        ("是否出場", exit_text),
    ])

    signal_df = pd.DataFrame([{
        "今日訊號": signal_text,
        "A 標的": format_stock_label(a_code, a_name),
        "A 買賣方向": a_action,
        "A signed weight": signed_pct(a_signed_weight),
        "A 建議股數": format_shares(a_suggested_shares),
        "B 標的": format_stock_label(b_code, b_name),
        "B 買賣方向": b_action,
        "B signed weight": signed_pct(b_signed_weight),
        "B 建議股數": format_shares(b_suggested_shares),
        "目前第幾層": layer_text,
        "估算價格基準": "最新 Close；實際交易仍以隔天 Open 為準",
    }])
    st.dataframe(signal_df, use_container_width=True, hide_index=True)

    st.markdown("#### 目前理論持倉狀況")
    if open_positions.empty:
        st.info("截至最新資料，目前沒有未平倉部位。")
        return

    open_positions["a_signed_shares"] = open_positions["a_shares"].astype(float) * open_positions["a_sign"].astype(float)
    open_positions["b_signed_shares"] = open_positions["b_shares"].astype(float) * open_positions["b_sign"].astype(float)
    open_positions["a_signed_weight"] = open_positions["a_weight"].astype(float) * open_positions["a_sign"].astype(float)
    open_positions["b_signed_weight"] = open_positions["b_weight"].astype(float) * open_positions["b_sign"].astype(float)
    open_positions["current_zscore"] = latest_z
    open_positions["current_action"] = open_positions["direction"].astype(str).apply(
        lambda d: "停損出場" if should_stop_loss(d, latest_z) else ("正常出場" if should_exit(d, latest_z, config.exit_z) else "續抱")
    )

    total_a_signed_shares = float(open_positions["a_signed_shares"].sum())
    total_b_signed_shares = float(open_positions["b_signed_shares"].sum())
    total_open_pnl = float(open_positions["pnl"].sum()) if "pnl" in open_positions.columns else np.nan

    render_latest_cards([
        ("未平倉筆數", f"{len(open_positions)}"),
        (f"{a_code} 淨股數", format_shares(total_a_signed_shares)),
        (f"{b_code} 淨股數", format_shares(total_b_signed_shares)),
        ("未實現 P&L", num(total_open_pnl)),
    ])

    display_cols = [
        "direction", "entry_layer", "entry_date", "entry_zscore", "current_zscore",
        "a_signed_shares", "b_signed_shares", "a_signed_weight", "b_signed_weight",
        "pnl", "holding_days", "current_action",
    ]
    position_df = open_positions[[c for c in display_cols if c in open_positions.columns]].copy()
    for col in ["entry_zscore", "current_zscore", "a_signed_weight", "b_signed_weight", "pnl"]:
        if col in position_df.columns:
            position_df[col] = position_df[col].astype(float).round(4)
    st.dataframe(position_df, use_container_width=True, hide_index=True)


def action_text(sign: int | float) -> str:
    if sign > 0:
        return "買進 / 做多"
    if sign < 0:
        return "賣出 / 放空"
    return "無"


def signed_pct(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "無"
    return f"{value:+.2%}"


def format_shares(value: float | int) -> str:
    if value is None or not np.isfinite(float(value)):
        return "無"
    if abs(float(value) - int(float(value))) < 1e-9:
        return f"{int(value):,}"
    return f"{float(value):,.2f}"


def render_latest_cards(items: list[tuple[str, str]]) -> None:
    cards = ""
    for label, value in items:
        safe_label = html.escape(str(label))
        safe_value = html.escape(str(value))
        cards += (
            '<div class="latest-signal-card">'
            f'<div class="latest-signal-label">{safe_label}</div>'
            f'<div class="latest-signal-value">{safe_value}</div>'
            '</div>'
        )

    st.markdown(
        f"""
        <style>
        .latest-signal-grid {{
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
          gap: 12px;
          margin: 0.4rem 0 1rem 0;
        }}
        .latest-signal-card {{
          background: #ffffff;
          border: 1px solid #e6e8ef;
          border-radius: 8px;
          padding: 14px 16px;
          box-shadow: 0 1px 2px rgba(20, 29, 47, 0.04);
          min-height: 92px;
        }}
        .latest-signal-label {{
          color: #344054;
          font-size: 0.95rem;
          font-weight: 600;
          line-height: 1.35;
          margin-bottom: 8px;
        }}
        .latest-signal-value {{
          color: #17202a;
          font-size: clamp(1.25rem, 2.4vw, 2.05rem);
          font-weight: 500;
          line-height: 1.15;
          white-space: normal;
          overflow-wrap: anywhere;
          word-break: break-word;
        }}
        </style>
        <div class="latest-signal-grid">
          {cards}
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_charts(result: dict[str, object], a_code: str, b_code: str, benchmark_name: str, config: Config) -> None:
    price_history: pd.DataFrame = result["price_history"]  # type: ignore[assignment]
    signals: pd.DataFrame = result["signals"]  # type: ignore[assignment]
    comparison: pd.DataFrame = result["comparison"]  # type: ignore[assignment]
    equity: pd.DataFrame = result["equity"]  # type: ignore[assignment]
    trades: pd.DataFrame = result["trades"]  # type: ignore[assignment]
    weights: pd.DataFrame = result["weights"]  # type: ignore[assignment]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=price_history.index, y=price_history[a_code], name=a_code, mode="lines"))
    fig.add_trace(go.Scatter(x=price_history.index, y=price_history[b_code], name=b_code, mode="lines"))
    fig.update_layout(title="Historical Close Price", template="plotly_white", height=520, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    fig = spread_chart(signals, config)
    fig.update_layout(height=520)
    st.plotly_chart(fig, use_container_width=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=comparison.index, y=comparison["strategy_cumulative_return"], name="Strategy", mode="lines"))
    if "benchmark_cumulative_return" in comparison:
        fig.add_trace(go.Scatter(x=comparison.index, y=comparison["benchmark_cumulative_return"], name=benchmark_name, mode="lines"))
    fig.update_layout(title="Cumulative Return vs Benchmark", template="plotly_white", height=520, yaxis_tickformat=".1%", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity.index, y=equity["drawdown"], name="Drawdown", mode="lines", fill="tozeroy"))
    fig.update_layout(title="Drawdown", template="plotly_white", height=520, yaxis_tickformat=".1%", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    fig = trade_chart(trades)
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    fig = weight_chart(weights)
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

def spread_chart(signals: pd.DataFrame, config: Config) -> go.Figure:
    fig = go.Figure()
    if signals.empty:
        fig.update_layout(title="Spread Signal", template="plotly_white", height=370)
        return fig

    upper = signals["spread_mean"] + config.entry_z * signals["spread_std"]
    lower = signals["spread_mean"] - config.entry_z * signals["spread_std"]
    ma5 = signals["ma5"] if "ma5" in signals.columns else pd.Series(index=signals.index, dtype=float)
    long_signals = signals[(signals["zscore"] <= -config.entry_z) & (signals["spread"] < ma5)]
    short_signals = signals[(signals["zscore"] >= config.entry_z) & (signals["spread"] > ma5)]

    fig.add_trace(go.Scatter(x=signals.index, y=signals["spread"], name="spread", mode="lines"))
    fig.add_trace(go.Scatter(x=signals.index, y=signals["spread_mean"], name="rolling mean", mode="lines"))
    if "ma5" in signals.columns:
        fig.add_trace(go.Scatter(x=signals.index, y=signals["ma5"], name="5MA", mode="lines"))
    fig.add_trace(go.Scatter(x=signals.index, y=upper, name="+entry band", mode="lines", line=dict(dash="dash")))
    fig.add_trace(go.Scatter(x=signals.index, y=lower, name="-entry band", mode="lines", line=dict(dash="dash")))

    if not short_signals.empty:
        fig.add_trace(go.Scatter(
            x=short_signals.index,
            y=short_signals["spread"],
            name="short signal",
            mode="markers",
            marker=dict(symbol="triangle-down", size=11),
            customdata=short_signals[["zscore", "ma5"]] if "ma5" in short_signals.columns else short_signals[["zscore"]],
            hovertemplate="short signal<br>%{x}<br>spread=%{y:.4f}<br>z=%{customdata[0]:.2f}<br>MA5=%{customdata[1]:.4f}<extra></extra>" if "ma5" in short_signals.columns else "short signal<br>%{x}<br>spread=%{y:.4f}<br>z=%{customdata[0]:.2f}<extra></extra>",
        ))

    if not long_signals.empty:
        fig.add_trace(go.Scatter(
            x=long_signals.index,
            y=long_signals["spread"],
            name="long signal",
            mode="markers",
            marker=dict(symbol="triangle-up", size=11),
            customdata=long_signals[["zscore", "ma5"]] if "ma5" in long_signals.columns else long_signals[["zscore"]],
            hovertemplate="long signal<br>%{x}<br>spread=%{y:.4f}<br>z=%{customdata[0]:.2f}<br>MA5=%{customdata[1]:.4f}<extra></extra>" if "ma5" in long_signals.columns else "long signal<br>%{x}<br>spread=%{y:.4f}<br>z=%{customdata[0]:.2f}<extra></extra>",
        ))

    fig.update_layout(title="Spread Signal", template="plotly_white", height=370, hovermode="x unified")
    return fig

def trade_chart(trades: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if trades.empty:
        fig.update_layout(title="Trade P&L", template="plotly_white", height=320)
        return fig
    labels = pd.to_datetime(trades["entry_date"]).dt.strftime("%Y-%m-%d") + " " + trades["direction"].astype(str)
    colors = np.where(trades["pnl"] >= 0, "#0f766e", "#c2410c")
    fig.add_trace(go.Bar(x=labels, y=trades["pnl"], marker_color=colors, name="P&L"))
    fig.update_layout(title="Trade P&L", template="plotly_white", height=320)
    return fig


def weight_chart(weights: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if weights.empty:
        fig.update_layout(title="Signed Entry Weights", template="plotly_white", height=320)
        return fig

    signed_weights = weights.copy()
    signed_weights["a_signed_weight"] = signed_weights["a_weight"] * signed_weights["a_sign"]
    signed_weights["b_signed_weight"] = signed_weights["b_weight"] * signed_weights["b_sign"]

    fig.add_trace(go.Scatter(
        x=signed_weights["entry_date"],
        y=signed_weights["a_signed_weight"],
        name="A signed weight",
        mode="lines+markers",
        customdata=signed_weights[["direction", "a_weight"]],
        hovertemplate=(
            "Date: %{x}<br>"
            "Direction: %{customdata[0]}<br>"
            "A signed weight: %{y:.0%}<br>"
            "A gross weight: %{customdata[1]:.0%}<extra></extra>"
        ),
    ))
    fig.add_trace(go.Scatter(
        x=signed_weights["entry_date"],
        y=signed_weights["b_signed_weight"],
        name="B signed weight",
        mode="lines+markers",
        customdata=signed_weights[["direction", "b_weight"]],
        hovertemplate=(
            "Date: %{x}<br>"
            "Direction: %{customdata[0]}<br>"
            "B signed weight: %{y:.0%}<br>"
            "B gross weight: %{customdata[1]:.0%}<extra></extra>"
        ),
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="#6b7280")
    fig.update_layout(title="Signed Entry Weights", template="plotly_white", height=320, yaxis_tickformat=".0%", hovermode="x unified")
    return fig


def show_tables(result: dict[str, object]) -> None:
    trades: pd.DataFrame = result["trades"]  # type: ignore[assignment]
    signals: pd.DataFrame = result["signals"]  # type: ignore[assignment]
    st.subheader("交易明細")
    if trades.empty:
        st.warning("這段期間沒有觸發交易。")
    else:
        cols = ["direction", "entry_layer", "entry_date", "exit_date", "entry_zscore", "exit_zscore", "a_weight", "b_weight", "gross_exposure", "pnl", "return_on_gross_exposure", "holding_days", "exit_reason", "status"]
        st.dataframe(trades[[c for c in cols if c in trades.columns]], use_container_width=True)
        st.download_button("Download trades CSV", trades.to_csv(index=False).encode("utf-8-sig"), "pair_trades.csv", "text/csv")
    with st.expander("Signal data"):
        st.dataframe(signals.tail(300), use_container_width=True)


def ols(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    X = np.column_stack([np.ones(len(x)), x])
    alpha, beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return float(alpha), float(beta)


def entry_direction(z: float, spread: float, ma5: float, entry_z: float) -> tuple[str | None, int, int]:
    if not np.isfinite(ma5):
        return None, 0, 0
    if z <= -entry_z and spread < ma5:
        return "long_spread", 1, -1
    if z >= entry_z and spread > ma5:
        return "short_spread", -1, 1
    return None, 0, 0

def should_exit(direction: str, z: float, exit_z: float) -> bool:
    return (direction == "long_spread" and z >= 0) or (direction == "short_spread" and z <= 0)

def should_stop_loss(direction: str, z: float) -> bool:
    return (direction == "long_spread" and z <= -STOP_Z) or (direction == "short_spread" and z >= STOP_Z)


def is_stop_zone(z: float) -> bool:
    return abs(z) >= STOP_Z


def transaction_cost(notional: float, side: str, config: Config) -> float:
    cost = abs(notional) * config.broker_fee
    if side == "sell":
        cost += abs(notional) * config.sell_tax
    return float(cost)


def trade_pnl(trade: pd.Series | dict[str, object], a_price: float, b_price: float) -> float:
    pnl_a = float(trade["a_sign"]) * float(trade["a_shares"]) * (a_price - float(trade["a_entry_price"]))
    pnl_b = float(trade["b_sign"]) * float(trade["b_shares"]) * (b_price - float(trade["b_entry_price"]))
    return float(pnl_a + pnl_b - float(trade["entry_cost"]))


def sharpe(returns: pd.Series) -> float:
    clean = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return float("nan")
    std = clean.std(ddof=1)
    if not np.isfinite(std) or std == 0:
        return float("nan")
    return float(np.sqrt(252) * clean.mean() / std)


def pct(value: float) -> str:
    return "NA" if value is None or not np.isfinite(value) else f"{value:.2%}"


def num(value: float) -> str:
    return "NA" if value is None or not np.isfinite(value) else f"{value:.2f}"


if __name__ == "__main__":
    main()
