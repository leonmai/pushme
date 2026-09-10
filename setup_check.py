#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
环境自检 —— 迁移到新机器后，先跑这个。

用法:
    python setup_check.py

检查三项:
  1. 依赖包是否齐全
  2. 目录结构是否完整
  3. 数据源连通性（新浪实时 / 腾讯日K / baostock）
"""
import sys
import os
from pathlib import Path

HERE = Path(__file__).parent
OK, WARN, FAIL = '[OK]  ', '[WARN]', '[FAIL]'


def head(t):
    print('\n' + '=' * 60)
    print('  ' + t)
    print('=' * 60)


def check_pkgs():
    head('1. 依赖包检查')
    need = [('pandas', 'pandas'), ('numpy', 'numpy'),
            ('requests', 'requests'), ('baostock', 'baostock')]
    opt = [('akshare', 'akshare'), ('tqdm', 'tqdm')]
    bad = []
    for mod, name in need:
        try:
            m = __import__(mod)
            print(f'{OK} {name:<12} {getattr(m, "__version__", "")}')
        except Exception as e:
            print(f'{FAIL} {name:<12} 缺失 -> {e}')
            bad.append(name)
    for mod, name in opt:
        try:
            m = __import__(mod)
            print(f'{OK} {name:<12} {getattr(m, "__version__", "")} (可选)')
        except Exception:
            print(f'{WARN} {name:<12} 缺失 (可选，不影响运行)')
    return bad


def check_files():
    head('2. 文件结构检查')
    need = ['live_scout.py', 'track_positions.py', 'screener_v2.py',
            'monthly_validate.py', 'stop_analysis.py', 'report_mv.py']
    miss = []
    for f in need:
        p = HERE / f
        print(f'{OK if p.exists() else FAIL} {f}')
        if not p.exists():
            miss.append(f)
    d = HERE / 'results_live'
    print(f'{"[OK]" if d.exists() else "[..]"}  results_live/  (持仓状态，会自动创建)')
    return miss


def check_net():
    head('3. 数据源连通性')
    import requests
    tests = [
        ('新浪实时行情', 'https://hq.sinajs.cn/list=sh000001',
         {'Referer': 'https://finance.sina.com.cn'}),
        ('腾讯日K', 'https://ifzq.gtimg.cn/appstock/app/fqkline/get'
                    '?param=sh000001,day,,,10,', None),
    ]
    for name, url, hdr in tests:
        try:
            r = requests.get(url, timeout=15,
                             headers=hdr or {'User-Agent': 'Mozilla/5.0'})
            print(f'{OK} {name:<12} HTTP {r.status_code}, {len(r.content)} bytes')
        except Exception as e:
            print(f'{FAIL} {name:<12} {e}')
    # baostock 的 socket 没有超时保护: 网络不通或服务端挂起时会永久阻塞
    # (实测 2026-09-09 卡在 login 成功后的查询上, 整整 38 分钟无响应)。
    # 这里放进 daemon 线程 + join 超时, 超时按失败处理而不是把整个自检吊死。
    import threading
    from datetime import datetime, timedelta
    res = {}

    def _probe():
        try:
            import baostock as bs
            lg = bs.login()
            if lg.error_code != '0':
                res['err'] = f'登录失败: {lg.error_msg}'
                return
            end = datetime.now()
            start = end - timedelta(days=30)
            rs = bs.query_history_k_data_plus(
                'sh.000001', 'date,close',
                start_date=start.strftime('%Y-%m-%d'),
                end_date=end.strftime('%Y-%m-%d'),
                frequency='d', adjustflag='3')
            # 不用 while rs.next(): 该游标在 baostock 0.9.30 下会死循环
            n = len(rs.get_data())
            bs.logout()
            res['n'] = n
        except Exception as e:
            res['err'] = str(e)

    t = threading.Thread(target=_probe, daemon=True)
    t.start()
    t.join(timeout=25)
    if t.is_alive():
        print(f'{FAIL} baostock     查询超时 (>25s 无响应) —— 网络被拦或服务端繁忙')
        print('               不影响盘中选股(用新浪/腾讯)，但**回测无法进行**')
    elif 'err' in res:
        print(f'{FAIL} baostock     {res["err"]}')
    elif res.get('n', 0) > 0:
        print(f'{OK} baostock     登录成功，取到 {res["n"]} 条日K')
    else:
        print(f'{WARN} baostock     登录成功但无数据（可能是非交易日/日期超出范围）')


def check_market():
    head('4. 大盘 MA20 状态（策略择时参考，当前仅提示不过滤）')
    import threading
    res = {}

    def _probe():
        # 直连腾讯取上证日K。不走 screener_v2.fetch_daily —— 那条链路针对个股,
        # 查指数 sh000001 会返回乱数据(实测给出 11.7 而非 3953)。
        # day 数组格式: [日期, 开盘, 收盘, 最高, 最低, 成交量]
        try:
            import requests
            r = requests.get(
                'https://ifzq.gtimg.cn/appstock/app/fqkline/get',
                params={'param': 'sh000001,day,,,250,'},
                headers={'User-Agent': 'Mozilla/5.0'}, timeout=20)
            arr = (r.json().get('data', {}).get('sh000001', {}).get('day')
                   or r.json().get('data', {}).get('sh000001', {}).get('qfqday')
                   or [])
            if len(arr) < 21:
                res['empty'] = True
                return
            res['date'] = arr[-1][0]
            res['close'] = float(arr[-1][2])
            res['ma20'] = sum(float(x[2]) for x in arr[-20:]) / 20
        except Exception as e:
            res['err'] = str(e)

    t = threading.Thread(target=_probe, daemon=True)
    t.start()
    t.join(timeout=30)
    if t.is_alive():
        print(f'{WARN} 取日K超时 (>30s)，跳过（多为数据源限流，稍后重试即可）')
        return
    if res.get('empty'):
        print(f'{WARN} 取不到上证日K，跳过')
        return
    if 'err' in res:
        print(f'{WARN} {res["err"]}')
        return
    ok = res['close'] > res['ma20']
    print(f'{OK} 上证 {res["date"]} 收盘 {res["close"]:.1f}, '
          f'MA20 {res["ma20"]:.1f} -> '
          f'{"在 MA20 上方（偏多）" if ok else "在 MA20 下方（偏空）"}')


def main():
    print('A股量价策略 v8 —— 环境自检')
    print(f'目录: {HERE}')
    print(f'Python: {sys.version.split()[0]}  ({sys.executable})')
    bad = check_pkgs()
    miss = check_files()
    check_net()
    check_market()

    head('结论')
    if bad or miss:
        print('存在问题，按上面 [FAIL] 项处理：')
        if bad:
            print(f'  缺包 -> pip install {" ".join(bad)}')
        if miss:
            print(f'  缺文件 -> {" ".join(miss)}（从原机器重新拷贝）')
        return 1
    print('环境就绪。下一步：')
    print(f'  cd /d "{HERE}"')
    print('  python live_scout.py --pool=400 --workers=12')
    return 0


if __name__ == '__main__':
    sys.exit(main())
