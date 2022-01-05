import re
import requests
import datetime
from datetime import timedelta
import lxml.html
import pandas as pd
import numpy as np
from bcb import sgs



def _discount_curve_builder(_curve, dib):
    def _(t):
        t = np.array(t)
        r = _curve(t)
        f = (1 + r) ** (t/dib)
        return 1/f
    return _


def ff_discount_curve(terms, rates, dib=252):
    _curve = ff_curve(terms, rates, dib)
    return _discount_curve_builder(_curve, dib)


def nss_discount_curve(par, dib=252):
    _curve = nss_curve(par, dib)
    return _discount_curve_builder(_curve, dib)


def ff_curve(terms, rates, dib=252):
    log_pu = np.log((1 + rates)**(terms/dib))
    def _(t):
        t = np.array(t)
        pu = np.exp(np.interp(t, terms, log_pu))
        return pu ** (252 / t) - 1
    return _


def nss(t, b1, b2, b3, b4, l1, l2):
    v = b1 + \
        b2 * (1 - np.exp(-l1*t)) / (l1*t) + \
        b3 * ((1 - np.exp(-l1*t)) / (l1*t) - np.exp(-l1*t)) + \
        b4 * ((1 - np.exp(-l2*t)) / (l2*t) - np.exp(-l2*t))
    return v


def nss_curve(par, dib=252):
    ts = lambda t: nss(t, par[0], par[1], par[2], par[3], par[4], par[5])
    def _(t):
        t = t/dib
        r = ts(t)
        return r
    return _


def bizdayse(cal, refdate, dc):
    return cal.bizdays(refdate, refdate + timedelta(dc))


def to_numeric(elm):
    s = elm.text
    s = s.strip()
    s = s.replace(',', '.')
    return float(s)


def get_curve(refdate, ticker, cal):
    # refdate = '2020-12-14'
    # ticker = 'PRE'
    url = "http://www2.bmf.com.br/pages/portal/bmfbovespa/lumis/lum-taxas-referenciais-bmf-ptBR.asp"
    url = f"{url}?Data={format(refdate, '%d/%m/%Y')}&Data1={format(refdate, '%Y%m%d')}&slcTaxa={ticker}"
    doc = lxml.html.parse(url).getroot()

    xs = doc.xpath("//table[contains(@id, 'tb_principal1')]")
    x = [to_numeric(elm) for elm in xs[0].findall('td')]
    dc = x[::3]
    tx_252 = x[1::3]

    terms = np.array([bizdayse(cal, refdate, d) for d in dc])
    rates = np.array([r/100 for r in x[1::3]])

    log_pu = np.log((1 + rates)**(terms/252))

    def interp_ff(term):
        term = np.array(term)
        pu = np.exp(np.interp(term, terms, log_pu))
        return (pu ** (252 / term) - 1) * 100

    return interp_ff


def get_contracts(refdate):
    def _cleanup(x):
        if x is None:
            return ''
        x = x.strip()\
             .replace('.', '')\
             .replace(',', '.')
        return x
    url = 'https://www2.bmf.com.br/pages/portal/bmfbovespa/lumis/lum-ajustes-do-pregao-ptBR.asp'
    res = requests.post(url, data=dict(dData1=refdate.strftime('%d/%m/%Y')), verify=False)
    root = lxml.html.fromstring(res.text)

    rx = re.compile(r'Atualizado em: (\d\d/\d\d/\d\d\d\d)')
    mx = rx.search(res.text)
    if mx is None:
        return None
    
    refdate = datetime.datetime.strptime(mx.group(1), '%d/%m/%Y')
    table = root.xpath("//table[contains(@id, 'tblDadosAjustes')]")
    if len(table) == 0:
        return None
    data = [_cleanup(td.text) for td in table[0].xpath('//td')]
    df = pd.DataFrame({
        'DataRef': refdate,
        'Mercadoria': flatten_names(recycle(data, 0, 6)),
        'CDVencimento': recycle(data, 1, 6),
        'PUAnterior': recycle(data, 2, 6),
        'PUAtual': recycle(data, 3, 6),
        'Variacao': recycle(data, 4, 6)
    })
    df['Vencimento'] = df['CDVencimento'].map(contract_to_maturity)
    df['PUAnterior'] = df['PUAnterior'].astype('float64')
    df['PUAtual'] = df['PUAtual'].astype('float64')
    df['Variacao'] = df['Variacao'].astype('float64')
    return df
   
def flatten_names(nx):
    for ix in range(len(nx)):
        if (nx[ix] != ""):
            last_name = nx[ix]
        nx[ix] = last_name
    x = [x[:3] for x in nx]
    return x


def recycle(s, i, m):
    assert len(s) % m == 0
    assert i < m
    assert i >= 0
    l = len(s)
    idx = list(range(i, l, m))
    return [s[i] for i in idx]


def contract_to_maturity(x):
    maturity_code = x[-3:]

    year = int(maturity_code[-2:]) + 2000

    m_ = dict(F = 1, G = 2, H = 3, J = 4, K = 5, M = 6,
              N = 7, Q = 8, U = 9, V = 10, X = 11, Z = 12)
    month_code = maturity_code[0]
    month = int(m_[month_code])

    return datetime.datetime(year, month, 1)


def build_curve(code, df, **kwargs):
    if code == 'DI1':
        fut = build_di1_futures(df, **kwargs)
        ts = build_di1_termstructure(fut, **kwargs)
    elif code == 'DAP':
        fut = build_dap_futures(df, **kwargs)
        ts = build_dap_termstructure(fut, **kwargs)
    elif code == 'IND':
        fut = build_ind_futures(df, **kwargs)
        ts = build_ind_termstructure(fut, **kwargs)
    elif code == 'BGI':
        fut = build_bgi_futures(df, **kwargs)
        ts = build_bgi_termstructure(fut, **kwargs)
    elif code == 'CCM':
        fut = build_ccm_futures(df, **kwargs)
        ts = build_ccm_termstructure(fut, **kwargs)
    return build_ffinterp_curve(ts, kwargs['cal'])


def build_ind_futures(df, **kwargs):
    cal = kwargs['cal']
    refdate = kwargs['refdate']
    spot = kwargs['spot']

    ctr = df[(df['Mercadoria'] == 'IND') &
             (df['DataRef'] ==  refdate)].reset_index(drop=True)
    
    ctr['Dia15'] = ctr['Vencimento'].map(lambda dt: dt.replace(day=15))
    ctr['Mes'] = ctr['Vencimento'].dt.month
    ctr['Ano'] = ctr['Vencimento'].dt.year
    
    ctr['WedBefore'] = ctr.apply(lambda df: cal.getdate('first wed before 15th day', df['Ano'], df['Mes']), axis=1)
    ctr['WedAfter'] = ctr.apply(lambda df: cal.getdate('first wed after 15th day', df['Ano'], df['Mes']), axis=1)
    
    ctr['DifBefore'] = (ctr['Dia15'] - pd.to_datetime(ctr['WedBefore'])).dt.days
    ctr['DifAfter'] = (pd.to_datetime(ctr['WedAfter']) - ctr['Dia15']).dt.days
    
    is_wed = ctr['Dia15'].dt.weekday == 2
    is_before = ctr['DifBefore'] < ctr['DifAfter']
    is_after = ~is_before

    ctr.loc[is_before, 'Maturity'] = ctr.loc[is_before, 'WedBefore']
    ctr.loc[is_after, 'Maturity'] = ctr.loc[is_after, 'WedAfter']
    ctr.loc[is_wed, 'Maturity'] = ctr.loc[is_wed, 'Dia15']
    ctr['Maturity'] = ctr['Maturity'].map(cal.following)

    ctr['DU'] = list(cal.vec.bizdays(ctr['DataRef'], ctr['Maturity']))
    ctr['DC'] = (ctr['Maturity'] - ctr['DataRef']).dt.days
    ctr = ctr[ctr['DU'] > 0].reset_index(drop=True)
    
    ctr = ctr.rename(columns={'PUAtual': 'PU'})
    crv = ctr[['DataRef', 'Maturity', 'DU', 'DC', 'PU']].copy()

    if spot is None:
        return crv
    
    first_term = pd.DataFrame({
        'DataRef': refdate,
        'Maturity': [cal.offset(refdate, 1)],
        'DU': [1],
        'DC': [(cal.offset(refdate, 1) - refdate.date()).days],
        'PU': [spot]
    })
    
    df = pd.concat([first_term, crv], axis=0).reset_index(drop=True)
    df['DataRef'] = pd.to_datetime(df['DataRef'])
    df['Maturity'] = pd.to_datetime(df['Maturity'])
    
    return df


def build_ind_termstructure(fut, **kwargs):
    rf_curve = kwargs['rf_curve']
    spot = kwargs['spot']
    
    riskfree = rf_curve(fut['DU'])
    f_riskfree = (1 + riskfree['Rate']) ** (riskfree['DU']/252)
    
    fut['Rate'] = (f_riskfree * spot/fut['PU']) ** (252/fut['DU']) - 1
    return fut[['DataRef', 'Maturity', 'DU', 'Rate']].copy()


def build_dap_futures(df, **kwargs):
    cal = kwargs['cal']
    refdate = kwargs['refdate']
    
    ctr = df[(df['Mercadoria'] == 'DAP') &
             (df['DataRef'] == refdate)].reset_index(drop=True)
    
    ctr['Vencimento'] = ctr['Vencimento'].map(lambda dt: dt.replace(day=15))
    ctr['Maturity'] = ctr['Vencimento'].map(cal.following)
    ctr['DU'] = list(cal.vec.bizdays(ctr['DataRef'], ctr['Maturity']))
    ctr = ctr[ctr['DU'] > 0].reset_index(drop=True)
    
    ctr = ctr.rename(columns={'PUAtual': 'PU'})
    
    return ctr[['DataRef', 'Maturity', 'DU', 'PU']].copy()


def build_dap_termstructure(fut, **kwargs):
    notional = kwargs['notional']
    
    fut['Rate'] = (notional / fut['PU']) ** (252/fut['DU']) - 1
    return fut[['DataRef', 'Maturity', 'DU', 'Rate']].copy()


def build_di1_futures(df, **kwargs):
    refdate = kwargs['refdate']
    cal = kwargs['cal']
    
    ctr = df[(df['Mercadoria'] == 'DI1') &
             (df['DataRef'] == refdate)].reset_index(drop=True)
    ctr['Maturity'] = list(cal.vec.adjust_next(ctr['Vencimento']))
    ctr['DU'] = list(cal.vec.bizdays(ctr['DataRef'], ctr['Maturity']))
    ctr['DC'] = (ctr['Maturity'] - ctr['DataRef']).dt.days
    ctr = ctr[ctr['DU'] > 0]
    
    ctr = ctr.rename(columns={'PUAtual': 'PU'})
    return ctr[['DataRef', 'Maturity', 'DU', 'DC', 'PU']]


def build_di1_termstructure(fut, **kwargs):
    refdate = kwargs['refdate']
    cal = kwargs['cal']
    notional = kwargs['notional']
    
    fut['Rate'] = (notional / fut['PU'])**(252 / fut['DU']) - 1
    fut_curve = fut[['DataRef', 'Maturity', 'DU', 'Rate']]
    
    cdi = sgs.get(('CDI', 4389), start_date=refdate, end_date=refdate)
    first_term = pd.DataFrame({
        'DataRef': refdate,
        'Maturity': [cal.offset(refdate, 1)],
        'DU': [1],
        'Rate': [cdi.iloc[0, 0]/100]
    })
    
    df = pd.concat([first_term, fut_curve], axis=0).reset_index(drop=True)
    df['DataRef'] = pd.to_datetime(df['DataRef'])
    df['Maturity'] = pd.to_datetime(df['Maturity'])
    
    return df


def build_bgi_futures(df, **kwargs):
    cal = kwargs['cal']
    refdate = kwargs['refdate']
    spot = kwargs['spot']
    
    ctr = df[(df['Mercadoria'] == 'BGI') &
             (df['DataRef'] == refdate)].reset_index(drop=True)
    
    ctr['Mes'] = ctr['Vencimento'].dt.month
    ctr['Ano'] = ctr['Vencimento'].dt.year
    ctr['Maturity'] = ctr.apply(lambda df: cal.getdate('last bizday', df['Ano'], df['Mes']), axis=1)

    ctr['DU'] = list(cal.vec.bizdays(ctr['DataRef'], ctr['Maturity']))
    ctr['DC'] = (ctr['Maturity'] - ctr['DataRef']).dt.days
    ctr = ctr[ctr['DU'] > 0].reset_index(drop=True)
    
    ctr = ctr.rename(columns={'PUAtual': 'PU'})
    crv = ctr[['DataRef', 'Maturity', 'DU', 'DC', 'PU']].copy()

    if spot is None:
        return crv

    first_term = pd.DataFrame({
        'DataRef': refdate,
        'Maturity': [cal.offset(refdate, 1)],
        'DU': [1],
        'DC': [(cal.offset(refdate, 1) - refdate.date()).days],
        'PU': [spot]
    })

    df = pd.concat([first_term, crv], axis=0).reset_index(drop=True)
    df['DataRef'] = pd.to_datetime(df['DataRef'])
    df['Maturity'] = pd.to_datetime(df['Maturity'])

    return df


def build_bgi_termstructure(fut, **kwargs):
    rf_curve = kwargs['rf_curve']
    spot = kwargs['spot']
    use_spot = kwargs.get('use_spot', False)
    
    riskfree = rf_curve(fut['DU'])
    f_riskfree = (1 + riskfree['Rate']) ** (riskfree['DU']/252)
    
    fut['Rate'] = (f_riskfree * spot/fut['PU']) ** (252/fut['DU']) - 1
    if use_spot:
        return fut[['DataRef', 'Maturity', 'DU', 'Rate']].copy()
    else:
        return fut[['DataRef', 'Maturity', 'DU', 'Rate']].copy().drop([0])


def build_ccm_futures(df, **kwargs):
    cal = kwargs['cal']
    refdate = kwargs['refdate']
    spot = kwargs.get('spot')
    
    ctr = df[(df['Mercadoria'] == 'CCM') &
             (df['DataRef'] == refdate)].reset_index(drop=True)
    
    ctr['Mes'] = ctr['Vencimento'].dt.month
    ctr['Ano'] = ctr['Vencimento'].dt.year
    ctr['Maturity'] = ctr.apply(lambda df: cal.getdate('15th day', df['Ano'], df['Mes']), axis=1)

    ctr['DU'] = list(cal.vec.bizdays(ctr['DataRef'], ctr['Maturity']))
    ctr['DC'] = (ctr['Maturity'] - ctr['DataRef']).dt.days
    ctr = ctr[ctr['DU'] > 0].reset_index(drop=True)
    
    ctr = ctr.rename(columns={'PUAtual': 'PU'})
    crv = ctr[['DataRef', 'Maturity', 'DU', 'DC', 'PU']].copy()

    if spot is None:
        return crv
        
    first_term = pd.DataFrame({
        'DataRef': refdate,
        'Maturity': [cal.offset(refdate, 1)],
        'DU': [1],
        'DC': [(cal.offset(refdate, 1) - refdate.date()).days],
        'PU': [spot]
    })

    df = pd.concat([first_term, crv], axis=0).reset_index(drop=True)
    df['DataRef'] = pd.to_datetime(df['DataRef'])
    df['Maturity'] = pd.to_datetime(df['Maturity'])

    return df


def build_ccm_termstructure(fut, **kwargs):
    rf_curve = kwargs['rf_curve']
    spot = kwargs['spot']
    use_spot = kwargs.get('use_spot', False)
    
    riskfree = rf_curve(fut['DU'])
    f_riskfree = (1 + riskfree['Rate']) ** (riskfree['DU']/252)
    
    fut['Rate'] = (f_riskfree * spot/fut['PU']) ** (252/fut['DU']) - 1
    if use_spot:
        return fut[['DataRef', 'Maturity', 'DU', 'Rate']].copy()
    else:
        return fut[['DataRef', 'Maturity', 'DU', 'Rate']].copy().drop([0])


def build_loglininterp_curve(ts, cal):
    refdate = ts['DataRef'].iloc[0]
    log_pu = np.log(ts['PU'])
    terms = ts['DU']
    def _curve(i_terms=None):
        if i_terms is None:
            return ts
        i_terms = np.array(i_terms)
        pu = np.exp(np.interp(i_terms, terms, log_pu))
        return pd.DataFrame({
            'DataRef': refdate,
            'DU': i_terms,
            'Maturity': cal.vec.offset(refdate, i_terms),
            'PU': pu
        })
    return _curve


def build_ffinterp_curve(ts, cal):
    refdate = ts['DataRef'].iloc[0]
    log_pu = np.log((1 + ts['Rate'])**(ts['DU']/252))
    terms = ts['DU']
    def _curve(i_terms=None):
        if i_terms is None:
            return ts
        i_terms = np.array(i_terms)
        pu = np.exp(np.interp(i_terms, terms, log_pu))
        return pd.DataFrame({
            'DataRef': refdate,
            'DU': i_terms,
            'Maturity': cal.vec.offset(refdate, i_terms),
            'Rate': pu ** (252 / i_terms) - 1
        })
    return _curve


def bizdiff(dates, cal):
    return [cal.bizdays(d1, d2) for d1, d2 in zip(dates[:-1], dates[1:])]