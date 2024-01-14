import numpy as np
import pandas as pd
import lightgbm as lgb
import gc
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import os
import re
import time
from itertools import chain, combinations
from numba import jit, njit, prange
import warnings
from warnings import simplefilter

warnings.filterwarnings("ignore")
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)



@njit(parallel=True)
def compute_triplet_imbalance(df_values, comb_indices):
    num_rows = df_values.shape[0]
    num_combinations = len(comb_indices)
    imbalance_features = np.empty((num_rows, num_combinations))

    # 🔁 Loop through all combinations of triplets
    for i in prange(num_combinations):
        a, b, c = comb_indices[i]

        # 🔁 Loop through rows of the DataFrame
        for j in range(num_rows):
            max_val = max(df_values[j, a], df_values[j, b], df_values[j, c])
            min_val = min(df_values[j, a], df_values[j, b], df_values[j, c])
            mid_val = df_values[j, a] + df_values[j, b] + df_values[j, c] - min_val - max_val

            # 🚫 Prevent division by zero
            if mid_val == min_val:
                imbalance_features[j, i] = np.nan
            else:
                imbalance_features[j, i] = (max_val - mid_val) / (mid_val - min_val)

    return imbalance_features


    def calculate_triplet_imbalance_numba(price, df):
    # Convert DataFrame to numpy array for Numba compatibility
    df_values = df[price].values
    remove = [('ask_price', 'wap', 'reference_price'), ('bid_price', 'wap', 'reference_price')]
    combs = list(set(combinations(price, 3)) - set(remove))
    comb_indices = [(price.index(a), price.index(b), price.index(c)) for a, b, c in combs]

    # Calculate the triplet imbalance using the Numba-optimized function
    features_array = compute_triplet_imbalance(df_values, comb_indices)

    # Create a DataFrame from the results

    columns = [f"{a}_{b}_{c}_imb3d" for a, b, c in combs]
    features = pd.DataFrame(features_array, columns=columns)

    return features


def chg(df, cols: [str]):
    chg_features = df.groupby('stock_id')[cols].pct_change()   # price change with previous non zero value
    chg_features.columns = [i+'_chg' for i in cols]
    chg_features = chg_features.fillna(0)
    df[chg_features.columns] = chg_features.values
    df.loc[df['seconds_in_bucket'] == 0, chg_features.columns] = 0
    return



def ret(df, cols: [str], hzn: int):
    ret_features = df.groupby('stock_id')[cols].diff(hzn)
    ret_features.columns = [f'{i}_ret_{hzn}' for i in cols]
    ret_features = ret_features.fillna(0)
    df[ret_features.columns] = ret_features.values
    df.loc[df['seconds_in_bucket'] == 0, ret_features.columns] = 0
    return


def csavg(df, cols):
    cols_name = [i + '_csavg' for i in cols]
    if df['time_id'].nunique == 1:
        df[cols_name] = np.nanmean(df[col].values, axis=0)
    else:
        df[cols_name] = df[cols].values - df.groupby('time_id')[cols].transform(np.nanmean).values
    return


def csdm(df, cols):
    #if f'{col}_csavg' in df.columns:
    #    df[col + '_csdm'] = df[col] - df[f'{col}_csavg']
    cols_name = [i + '_csdm' for i in cols]
    if df['time_id'].nunique == 1:
        df[cols_name] = df[cols].values - np.nanmean(df[cols].values)
    else:
        df[cols_name] = df[cols].values - df.groupby('time_id')[cols].transform(np.nanmean).values
    return




def csrank(df, cols):
    # need debug
    for col in cols:
        df[f'{col}_csrank'] = df.groupby('time_id')[col].rank(method="max", ascending=True, na_option='keep')
    rank_cols = [f'{col}_csrank' for col in cols]
    df[rank_cols] = df[rank_cols]/df.groupby(time_id)['stock_id'].count().values - 0.5
    return


def dollar_flow_ask(df, norm_dict):
    if df.iloc[0]['seconds_in_bucket'] == 0 and df['time_id'].nunique == 1:
        df['dollar_flow_ask'] = 0
        return
    last_price = df.groupby('stock_id')['ask_price'].shift(1).values
    last_size = df.groupby('stock_id')['ask_size'].shift(1).values
    group1 = (df['ask_price'] > last_price)
    group2 = (df['ask_price'] == last_price)
    group3 = (df['ask_price'] < last_price)
    df['dollar_flow_ask'] = 0.0
    df.loc[group1, 'dollar_flow_ask'] = -(last_size[group1] * last_price[group1])
    df.loc[group2, 'dollar_flow_ask'] = (df.loc[group2, 'ask_size'] - last_size[group2]) * (df.loc[group2, 'ask_price'])
    df.loc[group3, 'dollar_flow_ask'] = df.loc[group3, 'ask_size'] * df.loc[group3, 'ask_price']
    df['dollar_flow_ask'] /= norm_dict['dollar_ewma']
    df.loc[df['seconds_in_bucket'] == 0, 'dollar_flow_ask'] = 0
    return




def dollar_flow_bid(df, norm_dict):
    if df.iloc[0]['seconds_in_bucket'] == 0 and df['time_id'].nunique == 1:
        df['dollar_flow_bid'] = 0
        return
    last_price = df.groupby('stock_id')['bid_price'].shift(1).values
    last_size = df.groupby('stock_id')['bid_size'].shift(1).values
    group1 = (df['bid_price'] > last_price)
    group2 = (df['bid_price'] == last_price)
    group3 = (df['bid_price'] < last_price)
    df['dollar_flow_bid'] = 0.0
    df.loc[group1, 'dollar_flow_bid'] = df.loc[group1, 'bid_size'] * df.loc[group1, 'bid_price']
    df.loc[group2, 'dollar_flow_bid'] = (df.loc[group2, 'bid_size'] - last_size[group2]) * (df.loc[group2, 'bid_price'])
    df.loc[group3, 'dollar_flow_bid'] = -(last_size[group3] * last_price[group3])
    df['dollar_flow_bid'] /= norm_dict['dollar_ewma']
    df.loc[df['seconds_in_bucket'] == 0, 'dollar_flow_bid'] = 0
    return


def imbalance_features(df):
    # Define lists of price and size-related column names
    prices = ['far_price', 'reference_price', 'near_price', 'bid_price', 'ask_price', 'wap']
    sizes = ['bid_size', 'ask_size', 'matched_size', 'imbalance_size']
    norm_dict = dict()

    # sort df for groupby stock_id
    df.sort_values(['stock_id', 'time_id'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # time_series -- price
    df["price_spread"] = df["ask_price"] - df["bid_price"]
    df['relative_spread'] = (df['ask_price'] - df['bid_price']) / df['wap']
    df["mid_price"] = df.eval("(ask_price + bid_price) / 2")
    df['mid_deviation'] = (df['mid_price'].values - df['wap'].values) / df['mid_price'].values

    for window in [1, 2, 3, 5, 10]:
        ret(df, ['ask_price', 'bid_price', 'wap', 'near_price', 'far_price'], window)

    # time_series -- size
    df['total_size'] = df['imbalance_size'].values + df['matched_size'].values  # helper
    norm_dict['matched_size_imbalance_size_ewma'] = df.groupby('stock_id')['total_size'].ewm(halflife=3).mean().values
    df['matched_imbalance'] = df['imbalance_buy_sell_flag'] * df['imbalance_size'] / norm_dict['matched_size_imbalance_size_ewma']
    df['imbalance_size_std_3'] = df.groupby('stock_id')['imbalance_size'].rolling(10).std().values   # check if need ewm
    df['imbalance_bid_flag'] = np.where(df['imbalance_buy_sell_flag'] == 1, 1, 0) # helper
    df['imbalance_ask_flag'] = np.where(df['imbalance_buy_sell_flag'] == -1, 1, 0) # helper
    df['balance_flag'] = np.where(df['imbalance_buy_sell_flag'] == 0, 1, 0)  # helper
    df['high_volume'] = np.where(df['ask_size'] + df['bid_size'] > df['global_median_size'], 1, 0) # helper

    for col in ['ask_size', 'bid_size']:
        for window in [1, 2, 3, 5, 10]:
            df[f"{col}_diff_{window}"] = df.groupby("stock_id")[f"{col}"].diff(window)
    for col in ['matched_size', 'imbalance_size', 'reference_price', 'imbalance_buy_sell_flag']:
        for window in [1, 2, 3, 5, 10]:
            df[f"{col}_shift_{window}"] = df.groupby('stock_id')[f"{col}"].shift(window)
            df[f"{col}_ret_{window}"] = df.groupby('stock_id')[f"{col}"].pct_change(window)

    # time_series -- flow
    df['flow'] = df['ask_size'].values * df['ask_price'].values + df['bid_size'].values * df['bid_price'].values
    norm_dict['dollar_ewma'] = df.groupby('stock_id')['flow'].ewm(halflife=3).mean().values
    dollar_flow_ask(df, norm_dict)
    dollar_flow_bid(df, norm_dict)
    df['dollar_flow'] = df['dollar_flow_bid'] - df['dollar_flow_ask']
    df['flow_imbalance'] = df.eval('(ask_price*ask_size-bid_price*bid_size)/(ask_price*ask_size+bid_price*bid_size)') # helper

    # time_series -- apply chg / ret
    ret(df, ['matched_imbalance', 'near_price', 'reference_price', 'far_price', 'wap', 'flow_imbalance'], 1)
    chg(df, ['matched_imbalance', 'flow_imbalance'])

    # time_series -- apply ewmr
    for hzn in [1, 3, 5]:
        df[f'matched_imbalance_ewma_{hzn}'] = df.groupby('stock_id')['matched_imbalance'].ewm(halflife=hzn).mean().values
        df[f'near_price_lret_{hzn}'] = df.groupby('stock_id')['near_price_ret_1'].ewm(halflife=hzn).mean().values
        df[f'far_price_lret_{hzn}'] = df.groupby('stock_id')['far_price_ret_1'].ewm(halflife=hzn).mean().values
        df[f'reference_price_lret_{hzn}'] = df.groupby('stock_id')['reference_price_ret_1'].ewm(halflife=hzn).mean().values
        df[f'wap_lret_{hzn}'] = df.groupby('stock_id')['wap_ret_1'].ewm(halflife=hzn).mean().values
        norm_dict[f'spread_ewma_{hzn}'] = df.groupby('stock_id')['price_spread'].ewm(halflife=hzn).mean().values
        df[f'lret_spread_{hzn}'] = df[f'wap_lret_{hzn}'].values / norm_dict[f'spread_ewma_{hzn}']
        ewma_short = df.groupby('stock_id')['wap_ret_1'].ewm(halflife=hzn).mean().values
        ewma_long = df.groupby('stock_id')['wap_ret_1'].ewm(halflife=hzn*3).mean().values
        df[f'MACD_{hzn}_{hzn*3}'] = ewma_short - ewma_long
        df[f'flow_imbalance_ewma_{hzn}'] = df.groupby('stock_id')['flow_imbalance'].ewm(halflife=hzn).mean().values
        df[f'dollar_flow_ewma_{hzn}'] = df.groupby('stock_id')['dollar_flow'].ewm(halflife=hzn).mean().values



    df["volume"] = df.eval("ask_size + bid_size")
    df["liquidity_imbalance"] = df.eval("(bid_size-ask_size)/(bid_size+ask_size)")
    df["matched_imb_imbalance"] = df.eval("(imbalance_size-matched_size)/(matched_size+imbalance_size)")
    df["size_imbalance"] = df.eval("bid_size / ask_size")

    for c in chain(combinations(prices, 2), combinations(sizes, 2)):
        # if f"{c[0]}_{c[1]}" not in ['near_price_wap', 'reference_price_near_price', 'ask_price_wap']:
        df[f"{c[0]}_{c[1]}_imb2d"] = df.eval(f"({c[0]}-{c[1]})/({c[0]}+{c[1]})")

        # if f"{c[0]}_{c[1]}" not in ['near_price_ask_price']:
        df[f"log_ret_{c[0]}_{c[1]}"] = np.log(df.eval(f"{c[0]}/{c[1]}").values)
    	norm_dict[f"log_ret_{c[0]}_{c[1]}_ewma"] = df.groupby('stock_id')[f"log_ret_{c[0]}_{c[1]}"].ewm(halflife=8).mean().values
        df[f"log_ret_{c[0]}_{c[1]}"] -= norm_dict[f"log_ret_{c[0]}_{c[1]}_ewma"]


    for c in [['ask_price', 'bid_price', 'wap', 'reference_price'], sizes]:
        triplet_feature = calculate_triplet_imbalance_numba(c, df)
        df[triplet_feature.columns] = triplet_feature.values


    df["imbalance_momentum"] = df.groupby(['stock_id'])['imbalance_size'].diff(periods=1) / df['matched_size']
    df["spread_intensity"] = df.groupby(['stock_id'])['price_spread'].diff()
    df['price_pressure'] = df['imbalance_size'] * (df['ask_price'] - df['bid_price'])
    df['market_urgency'] = df['price_spread'] * df['liquidity_imbalance']
    df['depth_pressure'] = (df['ask_size'] - df['bid_size']) * (df['far_price'] - df['near_price'])

    df['spread_depth_ratio'] = (df['ask_price'] - df['bid_price']) / (df['bid_size'] + df['ask_size'])  # diff
    df['mid_price_movement'] = df['mid_price'].diff(periods=5).apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))  # diff

    # Calculate various statistical aggregation features
    for func in ["mean", "std", "skew", "kurt"]:
        df[f"all_prices_{func}"] = df[prices].agg(func, axis=1)
        df[f"all_sizes_{func}"] = df[sizes].agg(func, axis=1)


    df.sort_values(['time_id', 'stock_id'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df.replace([np.inf, -np.inf], 0)


def cross_sectional_features(df):
    prices = ['far_price', 'reference_price', 'near_price', 'bid_price', 'ask_price', 'wap']
    sizes = ['bid_size', 'ask_size', 'matched_size', 'imbalance_size']

    # cross sectional avg
    base = ['bid_price', 'ask_price', 'wap_ret_1', 'reference_price_ret_1', 'far_price_lret_5',
            'flow_imbalance_ewma_1', 'flow_imbalance_ewma_3', 'matched_imbalance', 'matched_imbalance_ewma_1',
            'imbalance_size_std_3', 'dollar_flow_bid']
    csavg(df, base)

    # cross sectional demean

    base = ['mid_price', 'far_price', 'bid_price', 'ask_price', 'wap', 'bid_size', 'ask_size',
            'reference_price_shift_10',
            'imbalance_buy_sell_flag', 'imbalance_bid_flag', 'imbalance_buy_sell_flag_shift_1', 'imbalance_buy_sell_flag_shift_2', 'imbalance_buy_sell_flag_shift_10', 'balance_flag', 'high_volume',
            'market_urgency', 'mid_deviation', 'price_pressure', 'matched_imbalance', 'matched_imbalance_chg', 'matched_imbalance_ret_1', 'matched_imbalance_ewma_1',
            'matched_imbalance_ewma_3', 'matched_imbalance_ewma_5',
            'dollar_flow_ask', 'dollar_flow_ewma_1', 'dollar_flow_ewma_3', 'dollar_flow_ewma_5', 'flow_imbalance_ewma_5', 'flow_imbalance_chg',
            'MACD_1_3', 'MACD_5_15', 'far_price_lret_1', 'far_price_lret_3', 'near_price_ret_1', 'near_price_ret_2', 'near_price_ret_3', 'near_price_lret_1', 'reference_price_lret_1', 'reference_price_lret_5', 'lret_spread_1',
            'imbalance_size_std_3', 'all_sizes_skew', 'all_sizes_kurt',
            'log_ret_reference_price_ask_price', 'log_ret_bid_price_wap',
            'reference_price_wap_imb2d', 'reference_price_ask_price_imb2d', 'reference_price_bid_price_imb2d',
            'ask_size_matched_size_imb2d', 'bid_size_matched_size_imb2d']


    csdm(df, base)
    return df


def dfrank(newdf):  # diff
    columns = [column for column in newdf.columns if (
            ('target' not in column)
            and ('date_id' not in column)
            and ('time_id' not in column)
            and ('row_id' not in column)
            and ('stock_id' not in column)
            and ('seconds_in_bucket' not in column)
    )]
    for column in columns:
        newdf = pd.concat([newdf, (
                    newdf[str(column)].rank(method="max", ascending=False, na_option='bottom') / len(newdf)).rename(
            f"{str(column)}_rank")], axis=1)
    return newdf


def other_features(df):
    df["dow"] = df["date_id"] % 5  # Day of the week
    df["seconds"] = df["seconds_in_bucket"] % 60
    df["minute"] = df["seconds_in_bucket"] // 60
    for key, value in global_stock_id_feats.items():
        df[f"global_{key}"] = df["stock_id"].map(value.to_dict())

    return df



def generate_all_features(df):
    cols = [c for c in df.columns if c not in ["row_id", "target"]]
    df = df[cols]
    # generate features
    df = df.groupby(['date_id', 'seconds_in_bucket']).apply(dfrank)
    df = df.reset_index(drop=True)
    df = other_features(df)
    df = imbalance_features(df)
    df = cross_sectional_features(df)


    remove = ["row_id", "time_id", "date_id", "target", 'imbalance_bid_flag', 'imbalance_ask_flag', 'balance_flag',
              'flow_imbalance', 'high_volume', 'total_size', 'flow']
    feature_name = [i for i in df.columns if i not in remove]
    gc.collect()
    return df[feature_name]


def init_data(train_file: str) -> pd.DataFrame:
    x = pd.read_csv(train_file)
    x = x.dropna(subset=["target"])
    x.reset_index(drop=True, inplace=True)
    x['time_id'] = (x.date_id * 55 + x.seconds_in_bucket / 10).astype(int)
    for i in ['imbalance_size', 'imbalance_buy_sell_flag', 'reference_price', 'matched_size',
              'far_price', 'near_price', 'bid_price', 'bid_size', 'ask_price', 'ask_size', 'wap']:
        x[i] = x[i].astype(np.float32)
    print("data init finished, lines = ", x.shape[0])
    return x



