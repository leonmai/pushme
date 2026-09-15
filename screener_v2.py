"""
A 股 15min 量价异动 v2 (T+1 + 5日出场 + 排除 bar)
=================================================

策略规则 (用户定义):
  1. 过去 20 个交易日单日最大涨幅 < 3%  (避免追高)
  2. 15 分钟线层面:
     (a) 当根 15min 涨幅 0% ~ 2%             (温和上涨)
     (b) 当根成交量 > 前一根 15min × 2        (放量)
     (c) 最近两个交易日 15min 总成交量: 今日 >= 昨日 × 0.9  (震荡/缩量后启动)

回测规则 (V2):
  - 时间排除: 11:15 / 13:00 / 14:45 整点三根 bar 不参与
  - 入场: 当日任一 valid bar 触发信号, 在该 bar 的 close 买入
  - 资金: 1 万元/只, 整百股
  - A 股 T+1: 当天不能卖, 次日起每日检查日 K
  - 出场: 未来 5 个交易日内, 日 K close >= buy × 1.01 → 当日 close 卖出
           5 日内都未触发 → 第 5 天 close 强平

数据源:
  - 实时: stock_zh_a_spot (新浪)
  - 日 K: stock_zh_a_hist (东财, 前复权)
  - 15min K: stock_zh_a_minute (新浪, 前复权)

用法:
  python screener_v2.py --mode=screen
  python screener_v2.py --mode=backtest --year=2026 --month=8
  python screener_v2.py --mode=backtest --date=2026-08-04 --sample=10

"""
import argparse
import json
import os
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, date
from pathlib import Path

import numpy as np
import pandas as pd
import threading

# ---- 软依赖: 缺包时降级, 不阻断主流程 (主路径已改为 baostock + 纯 requests) ----
try:
    import baostock as bs
except Exception:
    bs = None

try:
    import akshare as ak
except Exception:
    ak = None

try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else []


# 本地缓存目录: 二次复用日 K/15min K, 跨次/跨月提速
CACHE_DIR = Path(__file__).parent / '.cache'
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_TTL_DAYS = 7  # 缓存有效期 7 天

# ============ 配置 ============
POOL_MIN_TURNOVER = 5_000_000
POOL_EXCLUDE_PREFIX = ('bj',)
EXCLUDE_STAR_MARKET = True    # 科创板(688xxx)剔除: 用户要求全部选股/回测不含科创板

INTRADAY_PCT_MIN = 0.0
INTRADAY_PCT_MAX = 2.0
VOL_MULT = 2.0
YESTERDAY_VOL_RATIO = 0.9

# 大前提 (用户原话 "过去 2-5 天是下跌")
# 取 target_dt 之前最近的 `DECLINE_WINDOW` 个交易日, 收盘价累计跌幅 >= MIN_DECLINE_PCT
DECLINE_WINDOW = 5                       # 看几个交易日
MIN_DECLINE_PCT = -0.5                   # 累计跌幅至少 -0.5% 才算"下跌" (避免噪音)
RELAX_MIN_DECLINE_PCT = 0.0              # --relax 时只看 close[t-1] < close[t-N], 不要求跌幅门槛

# 连续暴跌剔除 (用户 2026-09-15 新增规则):
# 最近 CRASH_WINDOW 个交易日内, 若出现 CRASH_CONSEC 个连续交易日, 且每个单日跌幅都 > CRASH_DROP_PCT%,
# 视为处于连续暴跌/崩盘中, 风险过高, 从候选池剔除 (避免接飞刀).
CRASH_WINDOW = 5                       # 观察窗口 (交易日)
CRASH_CONSEC = 3                       # 连续下跌天数
CRASH_DROP_PCT = 5.0                   # 单日跌幅阈值 (%)

TP_PCT = 1.0                       # 未来 N 天日线涨幅达到 1% 即出场
HOLD_DAYS = 5                      # 持有期上限 5 个交易日
CAPITAL_PER_STOCK = 10_000
LOT = 100

# 排除的 15min bar (新浪 stock_zh_a_minute 用 close time 标记)
#   11:15-11:30 那根  -> 标记 11:30
#   13:00-13:15 那根  -> 标记 13:15
#   14:45-15:00 那根  -> 标记 15:00
EXCLUDED_BAR_TIMES = {
    (11, 30),   # 上午最后 15min (close time 标记)
    (13, 15),   # 下午首 15min
    (15, 0),    # 收盘前最后 15min
}

OUTPUT_DIR = Path("results_v2")
CACHE_DIR = Path(__file__).parent / '.cache_v2'
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ============ utils ============
def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def code_format(code: str) -> str:
    s = str(code).strip()
    if s.startswith(('sh', 'sz', 'bj')):
        return s[2:]
    return s


def market_prefix(code: str) -> str:
    s = str(code).strip()
    if s.startswith(('sh', 'sz', 'bj')):
        return s
    if s.startswith(('60', '68', '9')):
        return f"sh{s}"
    if s.startswith(('00', '30', '20')):
        return f"sz{s}"
    return s


def is_star_market(code: str) -> bool:
    """科创板(688xxx)判定 (纯6位代码或带 sh/sz/bj 前缀均可)"""
    return code_format(code).startswith('688')


def is_valid_pool_code(code: str) -> bool:
    s = str(code)
    if s.startswith(POOL_EXCLUDE_PREFIX):
        return False
    if EXCLUDE_STAR_MARKET and is_star_market(s):
        return False
    return True


def is_excluded_bar(t) -> bool:
    return (t.hour, t.minute) in EXCLUDED_BAR_TIMES


# ============ 数据拉取 ============
_pool_cache = None


def get_stock_pool() -> pd.DataFrame:
    global _pool_cache
    if _pool_cache is not None:
        return _pool_cache
    log("拉取全市场快照…")
    df = ak.stock_zh_a_spot()
    df = df.rename(columns={'代码': 'code', '名称': 'name',
                            '最新价': 'price', '涨跌幅': 'pct',
                            '成交额': 'turnover', '今开': 'open',
                            '昨收': 'prev_close'})
    df = df[~df['code'].str.startswith(POOL_EXCLUDE_PREFIX)]
    df = df[df['turnover'] >= POOL_MIN_TURNOVER]
    df = df[~df['name'].str.contains('ST|退', na=False)]
    df = df[df['price'] >= 1.0]
    # 科创板(688xxx)剔除: 用户要求全部选股/回测不含科创板
    if EXCLUDE_STAR_MARKET:
        n_star = int(df['code'].astype(str).str.startswith('688').sum())
        df = df[~df['code'].astype(str).str.startswith('688')]
        log(f"剔除科创板(688xxx): {n_star} 只")
    df = df.reset_index(drop=True)
    log(f"股票池: {len(df)} 只")
    _pool_cache = df
    return df


def _cache_path(key: str) -> Path:
    safe = key.replace('/', '_').replace('\\', '_').replace(':', '_')
    return CACHE_DIR / f"{safe}.pkl"


def _cache_load(key: str) -> pd.DataFrame | None:
    """加载缓存. 超过 TTL 或损坏则视为 None."""
    p = _cache_path(key)
    if not p.exists():
        return None
    age_days = (time.time() - p.stat().st_mtime) / 86400
    if age_days > CACHE_TTL_DAYS:
        return None
    try:
        with open(p, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None


def _cache_save(key: str, df: pd.DataFrame) -> None:
    try:
        with open(_cache_path(key), 'wb') as f:
            pickle.dump(df, f, protocol=4)
    except Exception as e:
        log(f"缓存写入失败 {key}: {e}")


def _daily_eng_to_zh(df) -> pd.DataFrame:
    """把新浪/腾讯的英文列日K转成东财中文列格式 (日期/收盘至少要有)"""
    if df is None or df.empty:
        return pd.DataFrame()
    try:
        out = pd.DataFrame({
            '日期': pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d'),
            '开盘': df['open'].astype(float),
            '收盘': df['close'].astype(float),
            '最高': df['high'].astype(float),
            '最低': df['low'].astype(float),
            '成交量': df['volume'].astype(float),
        })
        if 'amount' in df.columns:
            out['成交额'] = df['amount'].astype(float)
    except Exception:
        return pd.DataFrame()
    out['涨跌幅'] = out['收盘'].pct_change() * 100.0
    out = out.dropna(subset=['收盘']).reset_index(drop=True)
    return out


_em_down = False  # 东财熔断: 连续拉取失败后本进程内不再尝试东财


def fetch_daily(code: str, start: str, end: str, retries: int = 2) -> pd.DataFrame:
    global _em_down
    key = f"daily_{code}_{start}_{end}"
    cached = _cache_load(key)
    if cached is not None:
        return cached
    code6 = code_format(code)
    s_date = start.replace('-', '')
    e_date = end.replace('-', '')
    # 源1: 东财 (中文列). 失败自动熔断, 避免每只都等满重试.
    if not _em_down:
        em_ok = False
        for i in range(retries):
            try:
                df = ak.stock_zh_a_hist(symbol=code6, period='daily',
                                        start_date=s_date, end_date=e_date,
                                        adjust='qfq')
                if df is not None and not df.empty:
                    _cache_save(key, df)
                    return df
                em_ok = True  # 有响应但空数据(非网络错误), 也给新浪源机会
                break
            except Exception:
                em_ok = False
            time.sleep(0.8)
        if not em_ok:
            _em_down = True
            log("东财日K连续失败, 本进程内熔断, 转备用源")
    # 源2: 新浪 (英文列, qfq)
    try:
        sym = market_prefix(code)
        df = ak.stock_zh_a_daily(symbol=sym, start_date=s_date,
                                 end_date=e_date, adjust='qfq')
        out = _daily_eng_to_zh(df)
        if not out.empty:
            _cache_save(key, out)
            return out
    except Exception:
        pass
    # 源3: 腾讯 (英文列, qfq)
    try:
        sym = market_prefix(code)
        df = ak.stock_zh_a_hist_tx(symbol=sym, start_date=s_date,
                                   end_date=e_date, adjust='qfq')
        out = _daily_eng_to_zh(df)
        if not out.empty:
            _cache_save(key, out)
            return out
    except Exception:
        pass
    return pd.DataFrame()


def fetch_15min(code: str, retries: int = 3) -> pd.DataFrame:
    key = f"15min_{code}"
    cached = _cache_load(key)
    if cached is not None:
        return cached
    symbol = f"{market_prefix(code)}"
    for i in range(retries):
        try:
            df = ak.stock_zh_a_minute(symbol=symbol, period='15', adjust='qfq')
            if df is not None and not df.empty:
                df['day'] = pd.to_datetime(df['day'])
                _cache_save(key, df)
                return df
        except Exception:
            if i == retries - 1:
                return pd.DataFrame()
            time.sleep(2.0)
    return pd.DataFrame()


# ============ baostock 历史 15min (回测用, 覆盖任意历史月份) ============
_bs_lock = threading.Lock()


def _bs_query(bs_code: str, start: str, end: str, retries: int = 2) -> list:
    """baostock 单线程串行查询 (session 非线程安全, 全局锁保护)"""
    with _bs_lock:
        for i in range(retries):
            try:
                lg = bs.login()
                try:
                    rs = bs.query_history_k_data_plus(
                        bs_code, 'date,time,open,high,low,close,volume',
                        start_date=start, end_date=end,
                        frequency='15', adjustflag='2')  # 2 = 前复权
                    rows = []
                    # baostock 0.9.30: while rs.next() 游标迭代会死循环(pandas 3.0 下已验证),
                    # 必须一次性 get_data() 取出再转回行列表。
                    if rs.error_code == '0':
                        _d = rs.get_data()
                        if _d is not None and len(_d):
                            rows = _d.values.tolist()
                    return rows
                finally:
                    bs.logout()
            except Exception:
                if i == retries - 1:
                    return []
                time.sleep(1.0)
    return []


def fetch_15min_bs(code: str, year: int, month: int) -> pd.DataFrame:
    """baostock 拉取某年某月整月 15min K, 列统一为 day/open/close/high/low/volume (qfq)
    注意: 新浪 15min 仅覆盖近 6 个月, 历史月份必须用本函数."""
    mp = market_prefix(code)
    code6 = code_format(code)
    bs_code = f"{mp[:2]}.{code6}"
    if month == 12:
        end = f"{year + 1}-01-01"
    else:
        end = f"{year}-{month + 1:02d}-01"
    start = f"{year}-{month:02d}-01"
    key = f"15min_bs2_{bs_code}_{year}{month:02d}"
    cached = _cache_load(key)
    if cached is not None:
        return cached
    rows = _bs_query(bs_code, start, end)
    if not rows:
        return pd.DataFrame()
    try:
        df = pd.DataFrame(rows, columns=['date', 'time', 'open', 'high', 'low', 'close', 'volume'])
        for c in ['open', 'high', 'low', 'close', 'volume']:
            df[c] = pd.to_numeric(df[c])
        df['day'] = pd.to_datetime(df['time'].str[:14], format='%Y%m%d%H%M%S')
        df = df[['day', 'open', 'close', 'high', 'low', 'volume']].sort_values('day').reset_index(drop=True)
        # 排除时间: 与新浪源一致 (close time 标法: 11:30 / 13:15 / 15:00)
        # 2026-09-08 修正: 不再物理删除, 改由 filter_valid_bars 打 `_ex` 标记,
        # 保证「前一根 / 前3根」比较不跨午休、不跨日. (key 加 bs2 以避开旧缓存)
        _cache_save(key, df)
        return df
    except Exception:
        return pd.DataFrame()


# ============ 信号 ============
def filter_valid_bars(df15: pd.DataFrame) -> pd.DataFrame:
    """标记 11:30 / 13:15 / 15:00 三根 bar (close time 标法)

    2026-09-08 修正: 不再物理删除, 而是加 `_ex` 标记列.
    被排除的 bar 只**不作为信号根**, 但必须保留为「前一根 / 前3根」的比较基准,
    否则 13:30 的 bar 会跨午休与 11:15 比较, 量比严重失真
    (实盘案例: 张江高科 2026-09-08 13:30 量比 30.40× → 真实 1.97×, 非信号)."""
    df = df15.copy().reset_index(drop=True)
    if '_ex' not in df.columns:
        df['_ex'] = df['day'].dt.time.map(lambda t: is_excluded_bar(t))
    return df


def find_signal_bar(df15_valid: pd.DataFrame) -> dict | None:
    """
    在当日所有 bar 中, 自前向后扫, 返回最早触发信号的那根.
    跳过 `_ex` 标记 bar 作为信号根, 但比较基准用相邻真实 bar.
    返回: dict 含 trigger_idx, change_pct, vol_ratio, ytd_ratio, signal_time, close
    """
    if df15_valid.empty or len(df15_valid) < 2:
        return None
    df = df15_valid.reset_index(drop=True)

    # 准备 "今日/昨日量" - 按交易日聚合
    # 这里 df15_valid 已经是过滤后的, 但 groupby 取 day.date 会拿到原始交易日
    days = df['day'].dt.date.unique()
    if len(days) < 2:
        return None
    target_day = days[-1]
    prev_day = sorted(days)[-2]

    daily_vol = df.groupby(df['day'].dt.date)['volume'].sum()
    yesterday_vol = float(daily_vol.get(prev_day, 0))
    today_vol = float(daily_vol.get(target_day, 0))
    if yesterday_vol <= 0:
        return None
    ytd_ratio = today_vol / yesterday_vol

    today_bars = df[df['day'].dt.date == target_day].reset_index(drop=True)
    if len(today_bars) < 2:
        return None

    # 顺序扫描每一根, 只要有任意一根独立满足 (a)/(b) 且 (c) 全局满足即触发
    for i in range(1, len(today_bars)):
        cur = today_bars.iloc[i]
        if bool(cur.get('_ex', False)):
            continue        # 被排除的 bar 不发信号, 但仍作为下面各 bar 的比较基准
        prev = today_bars.iloc[i - 1]
        try:
            open_p = float(cur['open'])
            close_p = float(cur['close'])
            cur_vol = float(cur['volume'])
            prev_vol = float(prev['volume'])
        except Exception:
            continue
        if open_p <= 0 or prev_vol <= 0:
            continue

        chg = (close_p - open_p) / open_p * 100.0
        volr = cur_vol / prev_vol

        # (a) 涨幅
        if not (INTRADAY_PCT_MIN <= chg <= INTRADAY_PCT_MAX):
            continue
        # (b) 量比
        if volr < VOL_MULT:
            continue
        # (c) 整体今/昨量比
        if ytd_ratio < YESTERDAY_VOL_RATIO:
            continue

        return {
            'trigger_idx': i,
            'signal_time': cur['day'],
            'open': open_p,
            'close': close_p,
            'change_pct': chg,
            'vol_ratio': volr,
            'ytd_vol_ratio': ytd_ratio,
            'total_bars': len(today_bars),
            'signal_bar_vol': cur_vol,   # 当根成交量绝对值 (用于"每日取绝对量最大 N 只"排序)
            'prev_bar_vol': prev_vol,
        }
    return None


def check_recent_decline(ddf: pd.DataFrame, target_dt: date,
                          window: int = DECLINE_WINDOW,
                          min_pct: float = MIN_DECLINE_PCT) -> tuple[bool, dict | None]:
    """
    用日 K 判断近 `window` 个交易日是否处于下跌趋势.
    - 取 target_dt 之前最近的 `window` 个交易日
    - 起点 = 第 1 天收盘, 终点 = 第 N 天 (即 target_dt 前一个交易日) 收盘
    - 累计涨跌幅 = (end - start) / start * 100
    - 通过条件: 累计涨跌幅 <= min_pct (即下跌或微跌, 默认要求至少 -0.5%)
    - 返回: (是否通过, 诊断 dict 含 start/end 收盘 + 累计%)
    """
    if ddf is None or ddf.empty:
        return False, None
    date_col = pd.to_datetime(ddf['日期']).dt.date
    ddf_pre = ddf[date_col < target_dt]
    if len(ddf_pre) < window:
        return False, None
    last_n = ddf_pre.tail(window)
    if '收盘' not in last_n.columns:
        return False, None
    start_close = float(last_n.iloc[0]['收盘'])
    end_close = float(last_n.iloc[-1]['收盘'])
    if start_close <= 0:
        return False, None
    cum_pct = (end_close - start_close) / start_close * 100
    diag = {
        'window': window,
        'start_close': start_close,
        'end_close': end_close,
        'cum_pct': cum_pct,
    }
    return cum_pct <= min_pct, diag


def has_recent_crash(ddf: pd.DataFrame, target_dt: date,
                     window: int = CRASH_WINDOW,
                     consec: int = CRASH_CONSEC,
                     drop_pct: float = CRASH_DROP_PCT) -> bool:
    """
    最近 `window` 个交易日内, 是否出现 `consec` 个连续交易日, 每个单日跌幅都 > `drop_pct`%.
    命中返回 True (该个股应被剔除, 处于连续暴跌中).
    说明: 用日 K 的「涨跌幅」列 (单日 close-to-close 收益率) 判定; 若缺失则退而由「收盘」现算.
    """
    if ddf is None or ddf.empty:
        return False
    if '涨跌幅' not in ddf.columns and '收盘' not in ddf.columns:
        return False
    date_col = pd.to_datetime(ddf['日期']).dt.date
    ddf_pre = ddf[date_col < target_dt]
    if len(ddf_pre) < window:
        return False
    last_n = ddf_pre.tail(window).copy()
    if '涨跌幅' in last_n.columns:
        rets = pd.to_numeric(last_n['涨跌幅'], errors='coerce').dropna().values
    else:
        rets = (last_n['收盘'].pct_change() * 100.0).dropna().values
    run = 0
    for r in rets:
        if r < -drop_pct:
            run += 1
            if run >= consec:
                return True
        else:
            run = 0
    return False


# ============ 实时筛选 ============
def screen_one(row, target_dt: date):
    code = row['code']
    name = row['name']
    target_str = target_dt.strftime('%Y-%m-%d')

    ddf = fetch_daily(code, (target_dt - timedelta(days=80)).strftime('%Y-%m-%d'),
                      (target_dt - timedelta(days=1)).strftime('%Y-%m-%d'))
    if not check_recent_decline(ddf, target_dt)[0]:
        return None
    if has_recent_crash(ddf, target_dt,
                        window=CRASH_WINDOW, consec=CRASH_CONSEC,
                        drop_pct=CRASH_DROP_PCT):
        return None

    df15 = fetch_15min(code)
    if df15.empty:
        return None

    valid = filter_valid_bars(df15)
    today_valid = valid[valid['day'].dt.date == target_dt].reset_index(drop=True)
    if today_valid.empty:
        return None

    sig = find_signal_bar(pd.concat([
        valid[valid['day'].dt.date < target_dt].tail(60),  # 历史 bar (含昨日)
        today_valid
    ], ignore_index=True))
    if sig is None:
        return None

    return {
        'code': code, 'name': name,
        'date': target_str,
        'bar_time': str(sig['signal_time']),
        'bar_close': round(sig['close'], 3),
        'bar_change_pct': round(sig['change_pct'], 2),
        'bar_vol_ratio': round(sig['vol_ratio'], 2),
        'today_yesterday_vol_ratio': round(sig['ytd_vol_ratio'], 2),
    }


def run_screen(target_date: str | None = None, max_workers: int = 8,
               sample_size: int | None = None) -> pd.DataFrame:
    pool = get_stock_pool()
    if target_date is None:
        target_dt = datetime.now().date()
    else:
        target_dt = datetime.strptime(target_date, '%Y-%m-%d').date()
    if sample_size and sample_size < len(pool):
        pool = pool.sample(n=sample_size, random_state=42).reset_index(drop=True)
        log(f"抽样股票池: {len(pool)} 只")

    log(f"实时筛选: date={target_dt}, 池子={len(pool)} 只")
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(screen_one, row, target_dt): row
                   for _, row in pool.iterrows()}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="筛选进度"):
            try:
                r = fut.result(timeout=60)
            except Exception:
                r = None
            if r is not None:
                results.append(r)
    df = pd.DataFrame(results)
    if df.empty:
        log("无标的触发。")
    else:
        log(f"命中 {len(df)} 只")
    return df


# ============ 回测 (T+1) ============
def backtest_one_symbol(code: str, name: str, target_dt: date,
                        df15_valid: pd.DataFrame,
                        ddf: pd.DataFrame,
                        trade_dates: list[date],
                        capital: float = CAPITAL_PER_STOCK,
                        min_decline_pct: float = MIN_DECLINE_PCT,
                        hold_days: int = HOLD_DAYS) -> dict | None:
    """
    T+1 hold_days 日出场回测 (不加止损)
      - 入场: target_dt 触发信号那根 bar 的 close
      - 出场: trade_dates 中 target_dt 之后的交易日到 +hold_days 之内,
              日 K close >= buy × (1 + TP_PCT%) 即按当日 close 卖出
      - hold_days 个交易日内都未触发, 第 hold_days 个交易日 (或最末一个可交易日) close 强平
      - 持有期内不设止损
    """
    # 1) 近 5 天累计下跌 >= min_decline_pct%
    decline_ok, decline_diag = check_recent_decline(ddf, target_dt, min_pct=min_decline_pct)
    if not decline_ok:
        return None

    # 2) 找当日信号
    today_valid = df15_valid[df15_valid['day'].dt.date == target_dt].reset_index(drop=True)
    hist_valid = df15_valid[df15_valid['day'].dt.date < target_dt]
    if today_valid.empty:
        return None
    sig = find_signal_bar(pd.concat([hist_valid.tail(60), today_valid], ignore_index=True))
    if sig is None:
        return None

    buy_price = sig['close']
    if buy_price <= 0:
        return None
    shares = int(capital / buy_price / LOT) * LOT
    if shares <= 0:
        return None
    actual_capital = shares * buy_price
    target_price = buy_price * (1 + TP_PCT / 100.0)

    # 3) 后续 hold_days 个交易日的日 K
    date_col = pd.to_datetime(ddf['日期']).dt.date
    # trade_dates 是回测范围内的全量交易日, 转 set 便于查找
    td_set = set(trade_dates)
    future_dates = [d for d in trade_dates if d > target_dt][:hold_days]
    if not future_dates:
        return None

    # 用日 K 数据建立 date→row 索引
    ddf_indexed = ddf.copy()
    ddf_indexed['_date'] = date_col
    row_by_date = {row['_date']: row for _, row in ddf_indexed.iterrows()}

    exit_price = None
    exit_date = None
    exit_reason = None
    day_offset = None
    for offset, d in enumerate(future_dates, start=1):
        row = row_by_date.get(d)
        if row is None:
            continue
        try:
            close_p = float(row['收盘'])
        except Exception:
            continue
        if pd.isna(row.get('收盘')):
            continue
        if close_p >= target_price:
            exit_price = close_p
            exit_date = d
            exit_reason = f'T+{offset}止盈(>=1%)'
            day_offset = offset
            break

    if exit_price is None:
        # hold_days 日内未触发, 取最近一个有效日的 close 强平
        last_valid = None
        for d in reversed(future_dates):
            row = row_by_date.get(d)
            if row is not None and not pd.isna(row.get('收盘')):
                last_valid = row
                exit_date = d
                break
        if last_valid is None:
            return None
        exit_price = float(last_valid['收盘'])
        exit_reason = f'{hold_days}日未触发-强平'
        day_offset = hold_days

    pnl = (exit_price - buy_price) * shares
    pnl_pct = (exit_price - buy_price) / buy_price * 100.0
    return {
        'code': code, 'name': name,
        'date': target_dt.strftime('%Y-%m-%d'),
        'signal_time': str(sig['signal_time']),
        'bar_change_pct': round(sig['change_pct'], 2),
        'bar_vol_ratio': round(sig['vol_ratio'], 2),
        'signal_bar_vol': int(sig.get('signal_bar_vol', 0)),
        'prev_bar_vol': int(sig.get('prev_bar_vol', 0)),
        'today_yesterday_vol_ratio': round(sig['ytd_vol_ratio'], 2),
        'decline_window': decline_diag['window'] if decline_diag else None,
        'decline_cum_pct': round(decline_diag['cum_pct'], 2) if decline_diag else None,
        'buy_price': round(buy_price, 3),
        'exit_date': exit_date.strftime('%Y-%m-%d') if exit_date else '-',
        'exit_price': round(exit_price, 3),
        'hold_days': day_offset,
        'shares': shares,
        'capital_used': round(actual_capital, 2),
        'pnl': round(pnl, 2),
        'pnl_pct': round(pnl_pct, 2),
        'exit_reason': exit_reason,
    }


def run_backtest(year: int, month: int, max_workers: int = 8,
                 sample_size: int = 10, target_date: str | None = None,
                 min_decline_pct: float = MIN_DECLINE_PCT,
                 top_by_turnover: bool = False,
                 hold_days: int = HOLD_DAYS,
                 max_positions: int = 0) -> pd.DataFrame:
    # max_positions: 每日最多买入 N 只 (0 = 不限)。若当日命中超过 N,
    # 按信号强度排序取前 N: 量比大 > 当根涨幅高 > 前期跌得深, 再按代码兜底保证确定性
    pool = get_stock_pool()
    if sample_size and sample_size < len(pool):
        if top_by_turnover:
            pool = pool.sort_values('turnover', ascending=False).head(sample_size).reset_index(drop=True)
            log(f"Top-{sample_size} 股票池 (按成交额排序)")
        else:
            pool = pool.sample(n=sample_size, random_state=42).reset_index(drop=True)
            log(f"随机抽样股票池: {len(pool)} 只")

    if month == 12:
        month_last = datetime(year + 1, 1, 1) - timedelta(days=1)
    else:
        month_last = datetime(year, month + 1, 1) - timedelta(days=1)

    first_dt = datetime(year, month, 1).date()
    last_dt = month_last.date()
    # 交易日历扩展到下月 (供月末入场 + T+N 跨月出场)
    cal_end = (month_last + timedelta(days=20)).date()
    all_dates = [d.date() for d in pd.date_range(first_dt, cal_end)]

    # 交易日历 (用上证指数日 K)
    try:
        idx_df = ak.stock_zh_index_daily(symbol='sh000001')
        idx_df['date'] = pd.to_datetime(idx_df['date']).dt.date
        trade_dates = sorted([d for d in all_dates if d in set(idx_df['date'].tolist())])
    except Exception:
        trade_dates = all_dates
    log(f"交易日历: {len(trade_dates)} 天 ({trade_dates[0]} ~ {trade_dates[-1]})")

    if target_date:
        td_set = set(trade_dates)
        td = datetime.strptime(target_date, '%Y-%m-%d').date()
        if td not in td_set:
            log(f"{td} 不是交易日或不在月份范围内")
            return pd.DataFrame()
        run_dates = [td]
    else:
        # 只回测当月交易日 (出场日可落在次月)
        run_dates = [d for d in trade_dates if first_dt <= d <= last_dt]
    log(f"回测日: {len(run_dates)} 天")

    # 缓存
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    log(f"预拉 15min K 线 (baostock {year}-{month:02d}, 已排除无效时段)…")
    cache_15 = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(fetch_15min_bs, row['code'], year, month): row['code']
                   for _, row in pool.iterrows()}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="预拉15min"):
            code = futures[fut]
            try:
                df = fut.result(timeout=180)
                if df is not None and not df.empty:
                    cache_15[code] = df  # 已在 baostock 层排除无效时段
            except Exception:
                pass
    log(f"15min 缓存: {len(cache_15)} 只")

    log("预拉日 K (向前 60 天 + 向后 20 天覆盖跨月出场)…")
    daily_start = (first_dt - timedelta(days=60)).strftime('%Y-%m-%d')
    daily_end = (month_last + timedelta(days=20)).strftime('%Y-%m-%d')
    cache_daily = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(fetch_daily, row['code'], daily_start, daily_end): row['code']
                   for _, row in pool.iterrows()}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="预拉日K"):
            code = futures[fut]
            try:
                df = fut.result(timeout=60)
                if df is not None and not df.empty:
                    cache_daily[code] = df
            except Exception:
                pass
    log(f"日K 缓存: {len(cache_daily)} 只")

    # 回测
    all_trades = []
    pool_dict = {row['code']: row['name'] for _, row in pool.iterrows()}
    for td in run_dates:
        day_results = []
        for code in cache_15.keys():
            if code not in cache_daily or code not in pool_dict:
                continue
            r = backtest_one_symbol(code, pool_dict[code], td,
                                    cache_15[code], cache_daily[code],
                                    trade_dates,
                                    min_decline_pct=min_decline_pct,
                                    hold_days=hold_days)
            if r is not None:
                day_results.append(r)
        # 当日去重: 同一 code 只保留最早触发的一笔
        seen = set()
        uniq = []
        for r in day_results:
            if r['code'] in seen:
                continue
            seen.add(r['code'])
            uniq.append(r)
        # 每日持仓上限: 命中超过 max_positions 时按"信号根成交量绝对值"取前 N
        # (用户: 取 15min 量 > 前一根 15min 量的脉冲, 按当根绝对量排序取最大前 N)
        if max_positions and len(uniq) > max_positions:
            uniq.sort(key=lambda r: (r.get('signal_bar_vol', 0), r['code']), reverse=True)
            uniq = uniq[:max_positions]
            log(f"  {td}: 命中超过每日上限 {max_positions}, 已截取当根成交量最大的 {max_positions} 只")
        for i, r in enumerate(uniq, 1):
            r['day_rank'] = i
        log(f"  {td}: 命中 {len(uniq)} 笔 (raw {len(day_results)})")
        all_trades.extend(uniq)

    return pd.DataFrame(all_trades)


# ============ 报告输出 ============
def print_trade_table(df: pd.DataFrame, title: str):
    print("\n" + "=" * 110)
    print(f"  {title}")
    print("=" * 110)
    if df.empty:
        print("  (无数据)")
        return
    cols = ['date', 'code', 'name', 'buy_price', 'exit_date',
            'exit_price', 'hold_days', 'pnl_pct', 'pnl', 'exit_reason']
    show = df[cols].copy()
    show.columns = ['买入日', '代码', '名称', '买入价', '卖出日', '卖出价',
                    '持有日', '盈亏%', '盈亏额', '结果']
    print(show.to_string(index=False))


def summarize(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    wins = df[df['pnl'] > 0]
    losses = df[df['pnl'] < 0]
    flat = df[df['pnl'] == 0]
    total_pnl = df['pnl'].sum()
    win_rate = len(wins) / len(df) * 100 if len(df) else 0
    pf = (wins['pnl'].sum() / abs(losses['pnl'].sum())) if len(losses) else float('inf')
    by_reason = df.groupby('exit_reason')['pnl'].agg(['count', 'sum', 'mean']).round(2)
    by_hold = df.groupby('hold_days')['pnl'].agg(['count', 'sum']).round(2)
    daily = df.groupby('date')['pnl'].agg(['count', 'sum']).round(2)
    return {
        'trades': len(df),
        'wins': len(wins),
        'losses': len(losses),
        'flat': len(flat),
        'win_rate': round(win_rate, 2),
        'total_pnl': round(total_pnl, 2),
        'profit_factor': round(pf, 2) if pf != float('inf') else 'inf',
        'by_reason': by_reason,
        'by_hold_days': by_hold,
        'daily': daily,
    }


def print_summary(s: dict):
    if not s:
        print("\n无回测结果。")
        return
    print("\n" + "=" * 60)
    print("  回测汇总")
    print("=" * 60)
    print(f"  总交易笔数   : {s['trades']}")
    print(f"  盈利/亏损/平 : {s['wins']} / {s['losses']} / {s['flat']}")
    print(f"  胜率         : {s['win_rate']}%")
    print(f"  总盈亏       : {s['total_pnl']:>10} 元")
    print(f"  盈亏比(PF)   : {s['profit_factor']}")
    print("\n  按退出原因分组：")
    print(s['by_reason'].to_string())
    print("\n  按持有天数分组：")
    print(s['by_hold_days'].to_string())
    print("\n  每日明细:")
    print(s['daily'].to_string())


# ============ main ============
def main():
    parser = argparse.ArgumentParser(description='A 股 15min 异动 v2 (T+1)')
    parser.add_argument('--mode', choices=['screen', 'backtest'], required=True)
    parser.add_argument('--date', type=str, default=None, help='回测单日 (YYYY-MM-DD)')
    parser.add_argument('--year', type=int, default=2026)
    parser.add_argument('--month', type=int, default=8)
    parser.add_argument('--months', type=str, default=None,
                        help='多月份回测, 逗号分隔 (如 2024-08,2025-08,2026-08); 提供时忽略 --year/--month')
    parser.add_argument('--hold-days', type=int, default=HOLD_DAYS,
                        help='持有期上限 (交易日数, 默认 5)')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--sample', type=int, default=10, help='股票池抽样数 (默认 10)')
    parser.add_argument('--min-decline-pct', type=float, default=None,
                        help='大前提阈值: 近 5 日累计跌幅上限 (默认 -0.5, 即至少跌 0.5%)')
    parser.add_argument('--relax', action='store_true',
                        help='放宽大前提阈值: 改为只要 close[t-1] < close[t-5] 即视为下跌')
    parser.add_argument('--top', action='store_true',
                        help='按成交额排序取 top-N (替代随机抽样)')
    parser.add_argument('--max-positions', type=int, default=0,
                        help='每日最多买入 N 只 (0=不限; 命中超 N 时按信号强度取前 N)')
    parser.add_argument('--output', type=str, default='results_v2')
    args = parser.parse_args()

    if args.relax:
        min_decline_pct = RELAX_MIN_DECLINE_PCT
    elif args.min_decline_pct is not None:
        min_decline_pct = args.min_decline_pct
    else:
        min_decline_pct = MIN_DECLINE_PCT
    log(f"大前提阈值: 近 {DECLINE_WINDOW} 日累计涨跌幅 <= {min_decline_pct}% (即至少下跌 {abs(min_decline_pct)}%)")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    if args.mode == 'screen':
        df = run_screen(target_date=args.date, max_workers=args.workers,
                        sample_size=args.sample)
        print_trade_table(df, "实时筛选命中")
        if not df.empty:
            f = out / f"screen_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(f, index=False, encoding='utf-8-sig')
            log(f"已保存: {f}")
    else:
        if args.months:
            months = [m.strip() for m in args.months.split(',') if m.strip()]
        else:
            months = [f"{args.year}-{args.month:02d}"]
        all_trades = []
        for ym in months:
            y, m = int(ym.split('-')[0]), int(ym.split('-')[1])
            log(f"\n===== 回测 {ym} (持有期 {args.hold_days} 日) =====")
            df = run_backtest(y, m, max_workers=args.workers,
                              sample_size=args.sample, target_date=args.date,
                              min_decline_pct=min_decline_pct,
                              top_by_turnover=args.top,
                              hold_days=args.hold_days,
                              max_positions=args.max_positions)
            if df.empty:
                log(f"{ym} 无任何成交。")
                continue
            all_trades.append(df.assign(month=ym))
            print_trade_table(df, f"{ym} 回测明细")
            s = summarize(df)
            print_summary(s)
            f_csv = out / f"backtest_{ym.replace('-', '')}_trades.csv"
            df.to_csv(f_csv, index=False, encoding='utf-8-sig')
            log(f"明细已保存: {f_csv}")
            f_sum = out / f"backtest_{ym.replace('-', '')}_summary.json"
            ss = {k: (v.to_dict() if hasattr(v, 'to_dict') else v) for k, v in s.items()}
            with open(f_sum, 'w', encoding='utf-8') as fp:
                json.dump(ss, fp, ensure_ascii=False, indent=2, default=str)
            log(f"汇总已保存: {f_sum}")

        if len(all_trades) > 1:
            merged = pd.concat(all_trades, ignore_index=True)
            tag = f"hold{args.hold_days}_" + "_".join(m.replace('-', '') for m in months)
            f_merge = out / f"backtest_{tag}_trades.csv"
            merged.to_csv(f_merge, index=False, encoding='utf-8-sig')
            log(f"合并明细已保存: {f_merge} ({len(merged)} 笔)")
            # 分月对照汇总
            ms = summarize(merged)
            print("\n" + "=" * 60)
            print(f"  合并回测汇总 (共 {len(merged)} 笔)")
            print("=" * 60)
            print_summary(ms)
            by_month = merged.groupby('month')['pnl'].agg(['count', 'sum', 'mean']).round(2)
            print("\n  分月盈亏:")
            print(by_month.to_string())
            # 保存合并汇总
            f_sum_all = out / f"backtest_{tag}_summary.json"
            ss_all = {k: (v.to_dict() if hasattr(v, 'to_dict') else v) for k, v in ms.items()}
            ss_all['by_month'] = by_month.to_dict()
            with open(f_sum_all, 'w', encoding='utf-8') as fp:
                json.dump(ss_all, fp, ensure_ascii=False, indent=2, default=str)
            log(f"合并汇总已保存: {f_sum_all}")
            return merged
        elif all_trades:
            return all_trades[0]


if __name__ == '__main__':
    main()
