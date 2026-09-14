"""
多月份有效性验证 (monthly_validate)
===================================
目的: 验证 v7 策略在**不同月份 / 不同市况**下是否稳定有效 (样本外检验)。
     之前只测了 2024-08 / 2025-08 / 2026-08 三个月, 结论不可外推。

规则 (严格对齐 live_scout v7):
  入场:
    1. 大前提: 近 5 个交易日累计跌幅 <= -1%   (A级 <= -3%, B级 -3%~-1%)
    2. 15min: 涨幅 0~2%, 量 >= 前一根 x2
    3. 今/昨 15min 总量比 >= 0.9
    4. 排除 11:30 / 13:15 / 15:00 三根作为信号根 (但保留为比较基准)
  出场:
    T+1 起 5 个交易日内, 日K收盘 >= 买入价 x 1.01 → 当日收盘止盈
    5 日内未触发 → 第 5 日收盘强平, 不止损
  组合:
    每日最多 5 只, 按量比降序取 Top5, 每只 1 万元等权

大盘择时 (上证 > MA20) 作为**可切换维度**同时统计, 便于判断其必要性。

产出:
  results_mv/signals_<yyyymm>.csv   每月全量信号
  results_mv/monthly_summary.csv    按月汇总
  results_mv/report.html            可视化报告

用法:
  python monthly_validate.py --months=2026-07,2026-08 --sample=300
  python monthly_validate.py --months=2025-09,2025-10,...,2026-08 --sample=300
"""
import argparse
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent))
import screener_v2 as S

OUT = Path(__file__).parent / 'results_mv'
OUT.mkdir(exist_ok=True)

# ---- v7 参数 (与 live_scout 一致) ----
DECLINE_A = -3.0
DECLINE_B = -1.0
VOL_MULT = 2.0
YTD_MIN = 0.9
PCT_MIN, PCT_MAX = 0.0, 2.0
TOP_N = 5
TP_PCT = 1.0
HOLD_DAYS = 5
CAPITAL = 10_000

HDR = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
       'Referer': 'http://vip.stock.finance.sina.com.cn/'}
# 2026-09-09: web.ifzq.gtimg.cn 会返回 501 (风控重定向), 必须用 ifzq.gtimg.cn
TX_KLINE = 'https://ifzq.gtimg.cn/appstock/app/fqkline/get'
SINA_MKT = ('http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/'
            'Market_Center.getHQNodeData')
SINA_HQ = 'http://hq.sinajs.cn/list='

_daily_cache: dict = {}
_dc_lock = threading.Lock()


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ============ 数据层 (纯 requests, 不碰 akshare) ============
def fetch_daily_qfq(code: str, start: str, end: str) -> pd.DataFrame:
    """日K 前复权, 中文列 (日期/开盘/收盘/最高/最低/成交量)"""
    key = f'{code}_{start}_{end}'
    with _dc_lock:
        if key in _daily_cache:
            return _daily_cache[key]
    sym = S.market_prefix(code)
    df = pd.DataFrame()
    for q in ('qfq', ''):
        try:
            r = requests.get(TX_KLINE, timeout=20, headers=HDR,
                             params={'param': f'{sym},day,{start},{end},2000,{q}'})
            j = r.json()['data'][sym]
            arr = j.get(f'qfqday' if q == 'qfq' else 'day') or []
            if not arr:
                continue
            df = pd.DataFrame([a[:6] for a in arr],
                              columns=['日期', '开盘', '收盘', '最高', '最低', '成交量'])
            break
        except Exception:
            continue
    if not df.empty:
        for c in ('开盘', '收盘', '最高', '最低', '成交量'):
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df['日期'] = pd.to_datetime(df['日期']).dt.date
        df = df.dropna(subset=['收盘']).reset_index(drop=True)
    with _dc_lock:
        _daily_cache[key] = df
    return df


POOL_FILE = OUT / 'pool.csv'


def sina_all_codes() -> pd.DataFrame:
    """新浪全量 A 股清单 (分页). 盘前 amount=0, 但 symbol/name/settlement 有效"""
    import json
    rows = []
    for page in range(1, 60):
        try:
            r = requests.get(SINA_MKT, timeout=20, headers=HDR, params={
                'page': page, 'num': 100, 'sort': 'symbol', 'asc': 1,
                'node': 'hs_a', 'symbol': '', '_s_r_a': 'page'})
            if r.status_code != 200 or r.text.strip() in ('', 'null'):
                break
            got = json.loads(r.text)
            if not got:
                break
            rows.extend(got)
        except Exception:
            break
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df['code'] = df['symbol'].astype(str).apply(
        lambda s: s[2:] if s.startswith(('sh', 'sz', 'bj')) else s)
    df = df.rename(columns={'trade': 'price', 'settlement': 'prev_close'})
    return df[['code', 'name']].drop_duplicates('code').reset_index(drop=True)


def bs_pool_amount(codes: list, start: str, end: str) -> dict:
    """一次 login 会话串行查所有股票的日均成交额"""
    import baostock as bs
    out = {}
    with _bs_lock:
        try:
            bs.login()
        except Exception as ex:
            log(f'baostock login 失败: {ex}')
            return out
        try:
            for i, code in enumerate(codes, 1):
                bs_code = f'{S.market_prefix(code)[:2]}.{S.code_format(code)}'
                try:
                    rs = bs.query_history_k_data_plus(
                        bs_code, 'date,amount', start_date=start, end_date=end,
                        frequency='d', adjustflag='3')
                    amts = []
                    # 同上: rs.next() 会死循环, 改用 get_data() 整表取第 2 列(amount)
                    if rs.error_code == '0':
                        _d = rs.get_data()
                        if _d is not None and len(_d):
                            for v in _d.iloc[:, 1]:
                                try:
                                    amts.append(float(v))
                                except Exception:
                                    pass
                    if amts:
                        out[code] = sum(amts) / len(amts)
                except Exception:
                    continue
                if i % 500 == 0:
                    log(f'  成交额 {i}/{len(codes)}')
        finally:
            try:
                bs.logout()
            except Exception:
                pass
    return out


def get_pool(sample: int, base_month: str = '2026-08') -> pd.DataFrame:
    """股票池: 用基准月的日均成交额降序取 Top sample (baostock 真实历史, 不依赖盘前快照)
    注: 科创板(688xxx)已被剔除 (用户要求全部选股/回测不含科创板)"""
    if POOL_FILE.exists():
        df = pd.read_csv(POOL_FILE, dtype={'code': str})
        df['code'] = df['code'].astype(str).str.zfill(6)
        df = df[~df['code'].astype(str).str.startswith('688')]   # 科创板剔除 (防旧缓存含688)
        if len(df) >= sample:
            df = df.head(sample)
            log(f'股票池 {len(df)} 只 (读缓存, 已剔除科创板)')
            return df[['code', 'name']]
    log('构建股票池: 新浪全量清单 + baostock 日均成交额…')
    uni = sina_all_codes()
    if uni.empty:
        return uni
    uni = uni[~uni['code'].str.startswith(('4', '8', '9'))]
    uni = uni[~uni['code'].astype(str).str.startswith('688')]   # 科创板剔除
    uni = uni[~uni['name'].astype(str).str.contains('ST|退', na=False)]
    log(f'  候选 {len(uni)} 只')
    y, m = int(base_month[:4]), int(base_month[5:7])
    end = f'{y + 1}-01-01' if m == 12 else f'{y}-{m + 1:02d}-01'
    amt = bs_pool_amount(uni['code'].tolist(), f'{y}-{m:02d}-01', end)
    log(f'  拿到成交额 {len(amt)} 只')
    uni['turnover'] = uni['code'].map(amt)
    uni = uni.dropna(subset=['turnover'])
    uni = uni.sort_values('turnover', ascending=False).head(sample).reset_index(drop=True)
    uni[['code', 'name']].to_csv(POOL_FILE, index=False, encoding='utf-8-sig')
    log(f'股票池 {len(uni)} 只 (日均成交额 Top{len(uni)})')
    return uni[['code', 'name']]


# ============ baostock 15min (批量会话, 一次 login 查多只) ============
_bs_lock = threading.Lock()


def bs_fetch_month(codes: list, year: int, month: int) -> dict:
    """一次 login 会话串行查完本月所有股票 15min K, 返回 {code: df}"""
    import baostock as bs
    if month == 12:
        s, e = f'{year}-12-01', f'{year + 1}-01-01'
    else:
        s, e = f'{year}-{month:02d}-01', f'{year}-{month + 1:02d}-01'
    out = {}
    with _bs_lock:
        try:
            bs.login()
        except Exception as ex:
            log(f'baostock login 失败: {ex}')
            return out
        try:
            for i, code in enumerate(codes, 1):
                bs_code = f'{S.market_prefix(code)[:2]}.{S.code_format(code)}'
                cached = S._cache_load(f'15min_bs2_{bs_code}_{year}{month:02d}')
                if cached is not None:
                    out[code] = cached
                    continue
                try:
                    rs = bs.query_history_k_data_plus(
                        bs_code, 'date,time,open,high,low,close,volume',
                        start_date=s, end_date=e, frequency='15', adjustflag='2')
                    rows = []
                    # baostock 0.9.30: while rs.next() 游标迭代会死循环(pandas 3.0 下已验证),
                    # 必须一次性 get_data() 取出再转回行列表。
                    if rs.error_code == '0':
                        _d = rs.get_data()
                        if _d is not None and len(_d):
                            rows = _d.values.tolist()
                    if rows:
                        df = pd.DataFrame(rows, columns=['date', 'time', 'open', 'high', 'low', 'close', 'volume'])
                        for c in ('open', 'high', 'low', 'close', 'volume'):
                            df[c] = pd.to_numeric(df[c], errors='coerce')
                        df['day'] = pd.to_datetime(df['time'].str[:14], format='%Y%m%d%H%M%S')
                        df = df[['day', 'open', 'close', 'high', 'low', 'volume']]
                        df = df.sort_values('day').reset_index(drop=True)
                        S._cache_save(f'15min_bs2_{bs_code}_{year}{month:02d}', df)
                        out[code] = df
                except Exception:
                    continue
                if i % 100 == 0:
                    log(f'  15min {i}/{len(codes)}')
        finally:
            try:
                bs.logout()
            except Exception:
                pass
    return out


def bs_daily(syms: list, start: str, end: str, adjust: str = '2',
             tag: str = '') -> dict:
    """一次 login 会话串行拉多只标的日K (开/高/低/收/量/额)

    2026-09-09: 腾讯 fqkline 在批量请求下会返回 501 风控页, 日K统一改用 baostock。
    adjust: '2'=前复权(个股)  '3'=不复权(指数)
    """
    import baostock as bs
    out = {}
    with _bs_lock:
        try:
            bs.login()
        except Exception as ex:
            log(f'baostock login 失败: {ex}')
            return out
        try:
            for i, sym in enumerate(syms, 1):
                key = f'dailybs_{sym}_{start}_{end}_{adjust}'
                cached = S._cache_load(key)
                if cached is not None:
                    out[sym] = cached
                    continue
                try:
                    rs = bs.query_history_k_data_plus(
                        sym, 'date,open,high,low,close,volume,amount',
                        start_date=start, end_date=end,
                        frequency='d', adjustflag=adjust)
                    rows = []
                    # baostock 0.9.30: while rs.next() 游标迭代会死循环(pandas 3.0 下已验证),
                    # 必须一次性 get_data() 取出再转回行列表。
                    if rs.error_code == '0':
                        _d = rs.get_data()
                        if _d is not None and len(_d):
                            rows = _d.values.tolist()
                    if rows:
                        df = pd.DataFrame(rows, columns=[
                            '日期', '开盘', '最高', '最低', '收盘', '成交量', '成交额'])
                        for c in ('开盘', '最高', '最低', '收盘', '成交量', '成交额'):
                            df[c] = pd.to_numeric(df[c], errors='coerce')
                        df['日期'] = pd.to_datetime(df['日期']).dt.date
                        df = df.dropna(subset=['收盘']).sort_values('日期').reset_index(drop=True)
                        S._cache_save(key, df)
                        out[sym] = df
                except Exception:
                    continue
                if i % 200 == 0:
                    log(f'  {tag}日K {i}/{len(syms)}')
        finally:
            try:
                bs.logout()
            except Exception:
                pass
    return out


# ============ 信号与出场 ============
def find_signals(df15: pd.DataFrame, target_day) -> list:
    """修复版: 排除 bar 只不作为信号根, 但仍作为「前一根」基准"""
    if df15 is None or df15.empty:
        return []
    df = df15.copy()
    if '_ex' not in df.columns:
        df['_ex'] = df['day'].dt.time.map(S.is_excluded_bar)
    df['d'] = df['day'].dt.date
    days = sorted(df['d'].unique())
    if target_day not in days or len(days) < 2:
        return []
    idx = days.index(target_day)
    if idx == 0:
        return []
    prev_day = days[idx - 1]
    dv = df.groupby('d')['volume'].sum()
    yv, tv = float(dv.get(prev_day, 0)), float(dv.get(target_day, 0))
    if yv <= 0:
        return []
    ytd = tv / yv
    if ytd < YTD_MIN:
        return []

    bars = df[df['d'] == target_day].reset_index(drop=True)
    out = []
    for i in range(1, len(bars)):
        cur = bars.iloc[i]
        if bool(cur.get('_ex', False)):
            continue
        prev = bars.iloc[i - 1]
        try:
            op, cl = float(cur['open']), float(cur['close'])
            cv, pv = float(cur['volume']), float(prev['volume'])
        except Exception:
            continue
        if op <= 0 or pv <= 0:
            continue
        chg = (cl - op) / op * 100
        vr = cv / pv
        if not (PCT_MIN <= chg <= PCT_MAX) or vr < VOL_MULT:
            continue
        out.append({'bar_time': cur['day'], 'buy_price': cl,
                    'bar_change_pct': chg, 'vol_ratio': vr,
                    'ytd_vol_ratio': ytd, 'signal_bar_vol': cv})
    return out


def simulate_exit(ddf: pd.DataFrame, buy_dt, buy_price: float) -> dict:
    fut = ddf[ddf['日期'] > buy_dt].head(HOLD_DAYS)
    if fut.empty:
        return {}
    for i, (_, r) in enumerate(fut.iterrows(), 1):
        c = float(r['收盘'])
        if c >= buy_price * (1 + TP_PCT / 100):
            return {'exit_date': str(r['日期']), 'exit_price': c, 'hold_days': i,
                    'pnl_pct': (c / buy_price - 1) * 100, 'exit_reason': '止盈'}
    last = fut.iloc[-1]
    c = float(last['收盘'])
    return {'exit_date': str(last['日期']), 'exit_price': c, 'hold_days': len(fut),
            'pnl_pct': (c / buy_price - 1) * 100, 'exit_reason': '强平'}


def market_ma20(idx: pd.DataFrame) -> dict:
    """上证每日 收盘 vs MA20 → {date: bool}"""
    df = idx
    if df is None or df.empty:
        return {}
    df = df.sort_values('日期').reset_index(drop=True)
    df['ma20'] = df['收盘'].rolling(20).mean()
    return {d: bool(c > m) if pd.notna(m) else False
            for d, c, m in zip(df['日期'], df['收盘'], df['ma20'])}


def scan_month(year: int, month: int, pool: pd.DataFrame, mall: dict,
               dailies: dict, days: list) -> pd.DataFrame:
    t0 = time.time()
    log(f'=== {year}-{month:02d} : 拉 15min (baostock) ===')
    codes = pool['code'].tolist()
    m15 = bs_fetch_month(codes, year, month)
    log(f'  15min 拿到 {len(m15)}/{len(codes)} 只, 用时 {time.time() - t0:.0f}s')
    if not days:
        log('  日历为空')
        return pd.DataFrame()
    log(f'  交易日 {len(days)} 天')

    rows = []
    for code, name in zip(pool['code'], pool['name']):
        ddf = dailies.get(code)
        df15 = m15.get(code)
        if ddf is None or df15 is None or ddf.empty or df15.empty:
            continue
        for d in days:
            ok, diag = S.check_recent_decline(ddf, d, window=5, min_pct=DECLINE_B)
            if not diag or diag['cum_pct'] > DECLINE_B:
                continue
            for f in find_signals(df15, d):
                ex = simulate_exit(ddf, d, f['buy_price'])
                if not ex:
                    continue
                rows.append({'code': code, 'name': name, 'date': str(d),
                             'decline_5d_pct': diag['cum_pct'],
                             'grade': 'A' if diag['cum_pct'] <= DECLINE_A else 'B',
                             'market_ok': mall.get(d, False), **f, **ex})
    df = pd.DataFrame(rows)
    mk = sum(1 for d in days if mall.get(d, False))
    log(f'=== {year}-{month:02d} 完成: {len(df)} 条信号, 大盘可交易 {mk}/{len(days)} 天, '
        f'用时 {time.time() - t0:.0f}s ===')
    return df


def stats(df: pd.DataFrame, label: str) -> dict:
    if df.empty:
        return {'分组': label, '交易数': 0, '胜率%': np.nan, '止盈率%': np.nan,
                '平均收益%': np.nan, '中位收益%': np.nan, 'PF': np.nan,
                '盈亏比': np.nan, '累计收益%': np.nan,
                '最差单笔%': np.nan, '最好单笔%': np.nan, '最大连亏': 0}
    p = df['pnl_pct']
    win = p[p >= 0]
    loss = p[p < 0]
    pf = (win.sum() / abs(loss.sum())) if len(loss) and abs(loss.sum()) > 0 else np.inf
    avg_w = win.mean() if len(win) else 0.0
    avg_l = abs(loss.mean()) if len(loss) else 0.0
    # 最大连亏: 按出场日期排序后 pnl<0 的最长连续段
    try:
        seq = df.sort_values('exit_date')['pnl_pct'].tolist()
        mx = cur = 0
        for v in seq:
            cur = cur + 1 if v < 0 else 0
            mx = max(mx, cur)
    except Exception:
        mx = 0
    return {'分组': label, '交易数': len(df),
            '胜率%': round((p >= 0).mean() * 100, 1),
            '止盈率%': round((df['exit_reason'] == '止盈').mean() * 100, 1),
            '平均收益%': round(p.mean(), 3),
            '中位收益%': round(p.median(), 3),
            'PF': round(pf, 2) if np.isfinite(pf) else 999,
            '盈亏比': round(avg_w / avg_l, 2) if avg_l > 0 else 999,
            '累计收益%': round(p.sum(), 1),
            '最差单笔%': round(p.min(), 1),
            '最好单笔%': round(p.max(), 1),
            '最大连亏': mx}


def top5_by_day(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    d = df.sort_values(['date', 'vol_ratio'], ascending=[True, False])
    return d.groupby('date', group_keys=False).head(TOP_N).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--months', default='2026-07')
    ap.add_argument('--sample', type=int, default=300)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--no-cache', action='store_true')
    args = ap.parse_args()

    months = [tuple(int(x) for x in m.split('-')) for m in args.months.split(',')]
    pool = get_pool(args.sample)
    if pool.empty:
        log('股票池为空')
        return

    y0, m0 = months[0]
    y1, m1 = months[-1]
    daily_start = (pd.Timestamp(f'{y0}-{m0:02d}-01') - timedelta(days=150)).strftime('%Y-%m-%d')
    daily_end = (pd.Timestamp(f'{y1}-{m1:02d}-01') + pd.offsets.MonthEnd(0)
                 + timedelta(days=60)).strftime('%Y-%m-%d')
    log(f'日K区间 {daily_start} ~ {daily_end}')

    # 1) 上证指数日K (不复权) → MA20 择时 + 交易日历
    log('拉上证指数日K…')
    idx = bs_daily(['sh.000001'], '2023-06-01', daily_end,
                   adjust='3', tag='指数').get('sh.000001')
    mall = market_ma20(idx)
    days_map = {}
    if idx is not None and not idx.empty:
        for d in idx['日期']:
            days_map.setdefault((d.year, d.month), []).append(d)
    log(f'上证 MA20 判定覆盖 {len(mall)} 天, 日历覆盖 {len(days_map)} 个月')

    # 2) 个股日K (前复权), 全区间一次拉完, 磁盘缓存复用
    log(f'拉个股日K ({len(pool)} 只, baostock 串行)…')
    syms = [f'{S.market_prefix(c)[:2]}.{S.code_format(c)}' for c in pool['code']]
    dmap = bs_daily(syms, daily_start, daily_end, adjust='2', tag='个股')
    dailies = {s.split('.')[1]: d for s, d in dmap.items()}
    log(f'个股日K 拿到 {len(dailies)}/{len(syms)} 只')

    all_sig = []
    for y, m in months:
        tag = f'{y}{m:02d}'
        f = OUT / f'signals_{tag}.csv'
        df = pd.DataFrame()
        if f.exists() and not args.no_cache:
            try:
                tmp = pd.read_csv(f, dtype={'code': str})
                if not tmp.empty and 'market_ok' in tmp.columns:
                    df = tmp
                    log(f'{tag} 读缓存 {len(df)} 条')
            except Exception:
                df = pd.DataFrame()
        if df.empty:
            df = scan_month(y, m, pool, mall, dailies, days_map.get((y, m), []))
            if not df.empty:
                df.to_csv(f, index=False, encoding='utf-8-sig')
        if not df.empty:
            df['yyyymm'] = tag
            all_sig.append(df)

    if not all_sig:
        log('无信号')
        return
    A = pd.concat(all_sig, ignore_index=True)
    A.to_csv(OUT / 'all_signals.csv', index=False, encoding='utf-8-sig')

    # 按月汇总
    rows = []
    for tag, g in A.groupby('yyyymm'):
        t5 = top5_by_day(g)
        rows.append({'月份': tag, **stats(g, '全信号')})
        r2 = stats(t5, '日均Top5')
        rows.append({'月份': tag, **{k: v for k, v in r2.items() if k != '分组'},
                     '分组': 'Top5'})
        gk = g[g['market_ok']]
        if not gk.empty:
            r3 = stats(top5_by_day(gk), 'Top5+大盘')
            rows.append({'月份': tag, **{k: v for k, v in r3.items() if k != '分组'},
                         '分组': 'Top5+大盘'})
    ms = pd.DataFrame(rows)
    ms.to_csv(OUT / 'monthly_summary.csv', index=False, encoding='utf-8-sig')

    print('\n' + '=' * 100)
    print('按月汇总')
    print('=' * 100)
    print(ms.to_string(index=False))

    # 总体
    print('\n' + '=' * 100)
    print('全样本总体')
    print('=' * 100)
    tot = []
    tot.append(stats(A, '全信号'))
    tot.append(stats(top5_by_day(A), '每日Top5'))
    tot.append(stats(top5_by_day(A[A['market_ok']]), 'Top5 + 大盘MA20'))
    tot.append(stats(A[A['grade'] == 'A'], '仅A级(跌>3%)'))
    tot.append(stats(top5_by_day(A[A['grade'] == 'A']), 'A级 Top5'))
    tot.append(stats(top5_by_day(A[A['grade'] == 'B']), 'B级 Top5'))
    td = pd.DataFrame(tot)
    td.to_csv(OUT / 'total_summary.csv', index=False, encoding='utf-8-sig')
    print(td.to_string(index=False))


if __name__ == '__main__':
    main()
