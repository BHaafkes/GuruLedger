import pandas as pd
import numpy as np
import requests
from datetime import date
from math import ceil
import time
import os
import json
from flask import Flask

# Firebase Admin SDK voor Firestore
import firebase_admin
from firebase_admin import credentials, firestore

# --- Flask App Initialisatie & Robuuste Firestore Initialisatie ---
app = Flask(__name__)
try:
    if not firebase_admin._apps:
        print("➡️ Initialiseren van Firebase App...")
        creds_json_string = os.environ.get('FIRESTORE_CREDENTIALS')
        if not creds_json_string:
            raise ValueError("FIRESTORE_CREDENTIALS secret niet gevonden.")
        creds_dict = json.loads(creds_json_string)
        cred = credentials.Certificate(creds_dict)
        firebase_admin.initialize_app(cred)
        print("✅ Firebase App succesvol geïnitialiseerd.")
except Exception as e:
    print(f"❌ ERNSTIGE FOUT tijdens initialisatie van Firebase: {e}")

# --- Configuratie ---
SEC_CIK_TICKER_URL = "https://www.sec.gov/files/company_tickers.json"
MARKETCAP_URL = "https://companiesmarketcap.com/?download=csv"
HEADERS = {"User-Agent": "GuruLedger Automated Screener youremail@example.com"}

FACTS_TO_FETCH = [
    # (Lijst blijft ongewijzigd)
    {"column_name": "Assets_LastQuarter-2", "tag": "Assets", "unit": "USD", "timeframe_offset": -2, "type": "balance_sheet"},
    {"column_name": "Assets_LastQuarter-1", "tag": "Assets", "unit": "USD", "timeframe_offset": -1, "type": "balance_sheet"},
    {"column_name": "Assets_LastQuarter", "tag": "Assets", "unit": "USD", "timeframe_offset": 0, "type": "balance_sheet"},
    {"column_name": "CurrentAssets_LastQuarter-2", "tag": "AssetsCurrent", "unit": "USD", "timeframe_offset": -2, "type": "balance_sheet"},
    {"column_name": "CurrentAssets_LastQuarter-1", "tag": "AssetsCurrent", "unit": "USD", "timeframe_offset": -1, "type": "balance_sheet"},
    {"column_name": "CurrentAssets_LastQuarter", "tag": "AssetsCurrent", "unit": "USD", "timeframe_offset": 0, "type": "balance_sheet"},
    {"column_name": "CurrentLiabilities_LastQuarter-2", "tag": "LiabilitiesCurrent", "unit": "USD", "timeframe_offset": -2, "type": "balance_sheet"},
    {"column_name": "CurrentLiabilities_LastQuarter-1", "tag": "LiabilitiesCurrent", "unit": "USD", "timeframe_offset": -1, "type": "balance_sheet"},
    {"column_name": "CurrentLiabilities_LastQuarter", "tag": "LiabilitiesCurrent", "unit": "USD", "timeframe_offset": 0, "type": "balance_sheet"},
    {"column_name": "Liabilities_LastQuarter-2", "tag": "Liabilities", "unit": "USD", "timeframe_offset": -2, "type": "balance_sheet"},
    {"column_name": "Liabilities_LastQuarter-1", "tag": "Liabilities", "unit": "USD", "timeframe_offset": -1, "type": "balance_sheet"},
    {"column_name": "Liabilities_LastQuarter", "tag": "Liabilities", "unit": "USD", "timeframe_offset": 0, "type": "balance_sheet"},
    {"column_name": "NetFixedAssets_LastQuarter-2", "tag": "PropertyPlantAndEquipmentNet", "unit": "USD", "timeframe_offset": -2, "type": "balance_sheet"},
    {"column_name": "NetFixedAssets_LastQuarter-1", "tag": "PropertyPlantAndEquipmentNet", "unit": "USD", "timeframe_offset": -1, "type": "balance_sheet"},
    {"column_name": "NetFixedAssets_LastQuarter", "tag": "PropertyPlantAndEquipmentNet", "unit": "USD", "timeframe_offset": 0, "type": "balance_sheet"},
    {"column_name": "StockholdersEquity_LastQuarter-2", "tag": "StockholdersEquity", "unit": "USD", "timeframe_offset": -2, "type": "balance_sheet"},
    {"column_name": "StockholdersEquity_LastQuarter-1", "tag": "StockholdersEquity", "unit": "USD", "timeframe_offset": -1, "type": "balance_sheet"},
    {"column_name": "StockholdersEquity_LastQuarter", "tag": "StockholdersEquity", "unit": "USD", "timeframe_offset": 0, "type": "balance_sheet"},
    {"column_name": "NetIncome_LastQuarter-2", "tag": "NetIncomeLoss", "unit": "USD", "timeframe_offset": -2, "type": "income_statement"},
    {"column_name": "NetIncome_LastQuarter-1", "tag": "NetIncomeLoss", "unit": "USD", "timeframe_offset": -1, "type": "income_statement"},
    {"column_name": "NetIncome_LastQuarter", "tag": "NetIncomeLoss", "unit": "USD", "timeframe_offset": 0, "type": "income_statement"},
    {"column_name": "EPS_LastQuarter-2", "tag": "EarningsPerShareBasic", "unit": "USD-per-shares", "timeframe_offset": -2, "type": "income_statement"},
    {"column_name": "EPS_LastQuarter-1", "tag": "EarningsPerShareBasic", "unit": "USD-per-shares", "timeframe_offset": -1, "type": "income_statement"},
    {"column_name": "EPS_LastQuarter", "tag": "EarningsPerShareBasic", "unit": "USD-per-shares", "timeframe_offset": 0, "type": "income_statement"},
    {"column_name": "EBIT_LastQuarter-5", "tag": "OperatingIncomeLoss", "unit": "USD", "timeframe_offset": -5, "type": "income_statement"},
    {"column_name": "EBIT_LastQuarter-4", "tag": "OperatingIncomeLoss", "unit": "USD", "timeframe_offset": -4, "type": "income_statement"},
    {"column_name": "EBIT_LastQuarter-3", "tag": "OperatingIncomeLoss", "unit": "USD", "timeframe_offset": -3, "type": "income_statement"},
    {"column_name": "EBIT_LastQuarter-2", "tag": "OperatingIncomeLoss", "unit": "USD", "timeframe_offset": -2, "type": "income_statement"},
    {"column_name": "EBIT_LastQuarter-1", "tag": "OperatingIncomeLoss", "unit": "USD", "timeframe_offset": -1, "type": "income_statement"},
    {"column_name": "EBIT_LastQuarter", "tag": "OperatingIncomeLoss", "unit": "USD", "timeframe_offset": 0, "type": "income_statement"},
]

# --- **NIEUWE, EFFICIËNTE** Firestore Helper Functie ---
def write_df_to_firestore(db_client, df, collection_name):
    """Verwijdert een collectie en schrijft een DataFrame weg naar Firestore met Batched Writes."""
    collection_ref = db_client.collection(collection_name)
    
    # 1. Efficiënt verwijderen in batches
    docs = collection_ref.limit(500).stream()
    deleted = 0
    for doc in docs:
        batch = db_client.batch()
        while True:
            doc_list = list(docs)
            if not doc_list:
                break
            for doc in doc_list:
                batch.delete(doc.reference)
                deleted += 1
            batch.commit()
            docs = collection_ref.limit(500).stream() # Haal de volgende batch op
    
    if deleted > 0:
        print(f"   - {deleted} oude documenten in '{collection_name}' verwijderd.")

    # 2. Efficiënt schrijven in batches
    df_cleaned = df.replace({np.nan: None})
    batch = db_client.batch()
    for index, row in df_cleaned.iterrows():
        doc_data = row.to_dict()
        doc_id = str(row.get('Ticker', index))
        doc_ref = collection_ref.document(doc_id)
        batch.set(doc_ref, doc_data)
        # Commit de batch elke 500 documenten
        if (index + 1) % 500 == 0:
            batch.commit()
            batch = db_client.batch() # Start een nieuwe batch
    
    batch.commit() # Commit de laatste batch
        
    print(f"   - ✅ {len(df)} documenten weggeschreven naar '{collection_name}'.")

# --- Bestaande Functies (add_calculated_columns, etc.) blijven ongewijzigd ---
def add_calculated_columns(df):
    print("➡️ Calculating final columns and financial ratios...")
    df['Assets'] = df['Assets_LastQuarter'].combine_first(df['Assets_LastQuarter-1']).combine_first(df['Assets_LastQuarter-2'])
    df['CurrentAssets'] = df['CurrentAssets_LastQuarter'].combine_first(df['CurrentAssets_LastQuarter-1']).combine_first(df['CurrentAssets_LastQuarter-2'])
    df['CurrentLiabilities'] = df['CurrentLiabilities_LastQuarter'].combine_first(df['CurrentLiabilities_LastQuarter-1']).combine_first(df['CurrentLiabilities_LastQuarter-2'])
    df['EPS'] = df['EPS_LastQuarter'].combine_first(df['EPS_LastQuarter-1']).combine_first(df['EPS_LastQuarter-2'])
    df['Liabilities'] = df['Liabilities_LastQuarter'].combine_first(df['Liabilities_LastQuarter-1']).combine_first(df['Liabilities_LastQuarter-2'])
    df['NetIncome'] = df['NetIncome_LastQuarter'].combine_first(df['NetIncome_LastQuarter-1']).combine_first(df['NetIncome_LastQuarter-2'])
    df['StockholdersEquity'] = df['StockholdersEquity_LastQuarter'].combine_first(df['StockholdersEquity_LastQuarter-1']).combine_first(df['StockholdersEquity_LastQuarter-2'])
    df['NetFixedAssets'] = df['Assets'] - df['CurrentAssets']
    ttm_with_lq = ['EBIT_LastQuarter', 'EBIT_LastQuarter-1', 'EBIT_LastQuarter-2', 'EBIT_LastQuarter-3']
    ttm_no_lq = ['EBIT_LastQuarter-1', 'EBIT_LastQuarter-2', 'EBIT_LastQuarter-3', 'EBIT_LastQuarter-4']
    sum_with_lq = df[ttm_with_lq].sum(axis=1)
    sum_no_lq = df[ttm_no_lq].sum(axis=1)
    df['EBIT'] = np.where(df['EBIT_LastQuarter'].notna(), sum_with_lq, sum_no_lq)
    df['Capital'] = df['NetFixedAssets'] + df['CurrentAssets'] - df['CurrentLiabilities']
    df['EarningsYield'] = df['EBIT'].div(df['Market Cap'])
    df['ROC'] = df['EBIT'].div(df['Capital'])
    df['BargainValue'] = df['StockholdersEquity'] - df['Market Cap']
    print("   - ✅ All calculations complete.")
    return df

def get_dynamic_timeframes():
    today = date.today()
    current_year = today.year
    current_quarter = ceil(today.month / 3)
    last_quarter_num = current_quarter - 1
    last_quarter_year = current_year
    if last_quarter_num == 0:
        last_quarter_num = 4
        last_quarter_year -= 1
    timeframes = {}
    for i in range(6):
        q_num, q_year = last_quarter_num - i, last_quarter_year
        while q_num <= 0:
            q_num += 4
            q_year -= 1
        timeframes[-i] = f"CY{q_year}Q{q_num}"
    print(f"✅ Determined timeframes. Last completed quarter: {timeframes[0]}")
    return timeframes

def fetch_single_fact(tag, unit, timeframe):
    api_url = f"https://data.sec.gov/api/xbrl/frames/us-gaap/{tag}/{unit}/{timeframe}.json"
    try:
        time.sleep(0.1)
        response = requests.get(api_url, headers=HEADERS)
        response.raise_for_status()
        data = response.json().get('data', [])
        if not data: return None
        temp_df = pd.DataFrame(data)
        return temp_df[['cik', 'val']].drop_duplicates(subset=['cik'], keep='first')
    except requests.exceptions.HTTPError as e:
        if e.response.status_code != 404:
            print(f"   - HTTP Error for {tag} in {timeframe}: {e.response.status_code}")
        return None
    except Exception as e:
        print(f"   - An unexpected error occurred for {tag} in {timeframe}: {e}")
        return None

def fetch_sec_data():
    print("➡️ Fetching company CIK, Ticker, and Name list from SEC...")
    response = requests.get(SEC_CIK_TICKER_URL, headers=HEADERS)
    if response.status_code != 200:
        print(f"❌ ERROR: Failed to fetch company list. Status Code: {response.status_code}")
        print("Response Text:", response.text)
        raise Exception("Could not fetch company ticker data from SEC.")
    company_data = response.json()
    df = pd.DataFrame.from_dict(company_data, orient='index')
    df['cik'] = df['cik_str'].astype(int)
    df = df[['cik', 'ticker', 'title']].rename(columns={'ticker': 'Ticker', 'title': 'CompanyName'})

    print("➡️ Fetching Market Cap and Price data...")
    try:
        market_cap_df = pd.read_csv(MARKETCAP_URL)
        market_cap_df = market_cap_df[['Symbol', 'marketcap', 'price (USD)']].rename(columns={
            'Symbol': 'Ticker', 'marketcap': 'Market Cap', 'price (USD)': 'Price'
        })
        df = pd.merge(df, market_cap_df, on='Ticker', how='left')
        print("   - ✅ Successfully merged Market Cap and Price data.")
    except Exception as e:
        print(f"   - ⚠️ Could not fetch or process market cap data. Error: {e}")
        df['Market Cap'], df['Price'] = pd.NA, pd.NA

    timeframes = get_dynamic_timeframes()

    for fact in FACTS_TO_FETCH:
        column_name, tag, unit = fact["column_name"], fact["tag"], fact["unit"]
        offset, account_type = fact["timeframe_offset"], fact["type"]
        timeframe_str = timeframes.get(offset)
        api_timeframe = f"{timeframe_str}I" if account_type == "balance_sheet" else timeframe_str

        if account_type == "income_statement" and 'Q4' in api_timeframe:
            # (Q4-logica blijft ongewijzigd)
            print(f"➡️ Detected Q4 for {column_name}. Attempting direct fetch, with calculation as fallback...")
            year = api_timeframe[:6]
            df_q4_direct = fetch_single_fact(tag, unit, api_timeframe)
            df_annual = fetch_single_fact(tag, unit, f"{year}")
            df_q1 = fetch_single_fact(tag, unit, f"{year}Q1")
            df_q2 = fetch_single_fact(tag, unit, f"{year}Q2")
            df_q3 = fetch_single_fact(tag, unit, f"{year}Q3")
            can_calculate = all(d is not None for d in [df_annual, df_q1, df_q2, df_q3])
            if not can_calculate and df_q4_direct is None:
                df[column_name] = pd.NA; continue
            all_ciks = pd.concat([df[['cik']], df_annual[['cik']] if df_annual is not None else None, df_q4_direct[['cik']] if df_q4_direct is not None else None]).drop_duplicates()
            merged_df = all_ciks
            q4_direct_col = f"{column_name}_Direct"; annual_col, q1_col, q2_col, q3_col = f"{column_name}_Annual", f"{column_name}_Q1", f"{column_name}_Q2", f"{column_name}_Q3"
            if df_q4_direct is not None: merged_df = pd.merge(merged_df, df_q4_direct.rename(columns={'val': q4_direct_col}), on='cik', how='left')
            if can_calculate:
                merged_df = pd.merge(merged_df, df_annual.rename(columns={'val': annual_col}), on='cik', how='left')
                merged_df = pd.merge(merged_df, df_q1.rename(columns={'val': q1_col}), on='cik', how='left')
                merged_df = pd.merge(merged_df, df_q2.rename(columns={'val': q2_col}), on='cik', how='left')
                merged_df = pd.merge(merged_df, df_q3.rename(columns={'val': q3_col}), on='cik', how='left')
            calculated_q4 = pd.Series(dtype='float64')
            if can_calculate: calculated_q4 = merged_df[annual_col] - (merged_df[q1_col] + merged_df[q2_col] + merged_df[q3_col])
            if q4_direct_col in merged_df: merged_df[column_name] = merged_df[q4_direct_col].combine_first(calculated_q4)
            else: merged_df[column_name] = calculated_q4
            cols_to_merge = ['cik', column_name, q4_direct_col, annual_col, q1_col, q2_col, q3_col]
            existing_cols_to_merge = [col for col in cols_to_merge if col in merged_df.columns]
            df = pd.merge(df, merged_df[existing_cols_to_merge], on='cik', how='left')
            print(f"   - ✅ Successfully processed Q4 data for {column_name}.")
        else:
            print(f"➡️ Fetching: {column_name} ({tag}) for {api_timeframe}...")
            temp_df = fetch_single_fact(tag, unit, api_timeframe)
            if temp_df is not None:
                df = pd.merge(df, temp_df.rename(columns={'val': column_name}), on='cik', how='left')
            else:
                df[column_name] = pd.NA
    df = add_calculated_columns(df)
    id_cols = ['cik', 'CompanyName', 'Ticker', 'Price', 'Market Cap']
    key_financials = ['Assets', 'Liabilities', 'StockholdersEquity', 'EBIT', 'NetIncome', 'EPS', 'Capital']
    key_ratios = ['EarningsYield', 'ROC', 'BargainValue']
    source_data_cols = [col for col in df.columns if col not in (id_cols + key_financials + key_ratios)]
    final_order = id_cols + key_financials + key_ratios + sorted(source_data_cols)
    final_order_existing = [col for col in final_order if col in df.columns]
    df = df[final_order_existing]
    return df

def run_guru_models(db_client, df):
    """Voert de guru-modellen uit en schrijft de resultaten naar Firestore."""
    print("\n---  Guru Model Analysis ---")
    
    # --- 1. Magic Formula ---
    print("\n🔮 Running Model 1: Magic Formula...")
    mf_df = df.copy()
    mf_df.dropna(subset=['Market Cap'], inplace=True)
    mf_df = mf_df[mf_df['Market Cap'] > 500_000_000]
    mf_df.dropna(subset=['ROC', 'EarningsYield'], inplace=True)
    mf_df['ROC_Rank'] = mf_df['ROC'].rank(ascending=False, method='first')
    mf_df['EY_Rank'] = mf_df['EarningsYield'].rank(ascending=False, method='first')
    mf_df['Magic_Rank'] = mf_df['ROC_Rank'] + mf_df['EY_Rank']
    mf_df.sort_values('Magic_Rank', inplace=True)
    mf_buys = mf_df.head(30)
    mf_sells = mf_df.tail(30)
    print(f"   - Found {len(mf_buys)} Buy and {len(mf_sells)} Sell picks.")
    write_df_to_firestore(db_client, mf_buys, 'magic_formula_buys')
    write_df_to_firestore(db_client, mf_sells, 'magic_formula_sells')
    
    # --- 2. The Intelligent Investor ---
    print("\n🧐 Running Model 2: The Intelligent Investor...")
    ii_df = df.copy()
    ii_df.dropna(subset=['Market Cap'], inplace=True)
    ii_df = ii_df[ii_df['Market Cap'] > 500_000_000]
    ii_df.dropna(subset=['BargainValue'], inplace=True)
    ii_buys = ii_df[ii_df['BargainValue'] > 0]
    print(f"   - Found {len(ii_buys)} Buy picks.")
    write_df_to_firestore(db_client, ii_buys, 'intelligent_investor_buys')

    # --- 3. Combined Model ---
    print("\n✨ Running Model 3: Combined Magic Formula & Intelligent Investor...")
    combo_df = df.copy()
    combo_df.dropna(subset=['Market Cap'], inplace=True)
    combo_df = combo_df[combo_df['Market Cap'] > 500_000_000]
    combo_df = combo_df[combo_df['BargainValue'] > 0]
    combo_df.dropna(subset=['ROC', 'EarningsYield'], inplace=True)
    if not combo_df.empty:
        combo_df['ROC_Rank'] = combo_df['ROC'].rank(ascending=False, method='first')
        combo_df['EY_Rank'] = combo_df['EarningsYield'].rank(ascending=False, method='first')
        combo_df['Magic_Rank'] = combo_df['ROC_Rank'] + combo_df['EY_Rank']
        combo_df.sort_values('Magic_Rank', inplace=True)
        combo_buys = combo_df.head(30)
        print(f"   - Found {len(combo_buys)} Buy picks.")
        write_df_to_firestore(db_client, combo_buys, 'combined_model_buys')
    else:
        print("   - No companies met the initial criteria for the Combined Model.")

@app.route('/')
def main_job_entrypoint():
    """Hoofdfunctie die wordt aangeroepen door Cloud Scheduler."""
    try:
        db = firestore.client() 
        final_df = fetch_sec_data()
        
        if final_df is not None:
            print("\n➡️ Volledige dataset naar Firestore schrijven...")
            write_df_to_firestore(db, final_df, 'sec_financial_data_full_detail')
            run_guru_models(db, final_df)
        
        return "Script succesvol voltooid.", 200
        
    except Exception as e:
        import traceback
        print("--- SCRIPT MISLUKT MET EEN FOUT ---")
        traceback.print_exc()
        print("---------------------------------")
        return f"Er is een fout opgetreden: {e}", 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
