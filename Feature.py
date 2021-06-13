# packages
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)
from itertools import product
from sklearn.preprocessing import LabelEncoder
import pickle
import calendar


def save_feature(train, items, shops, cats, test):
    # remove outliers
    # remove items with price > 1000000 and sales > 1001
    train = train[train.item_price<100000]
    train = train[train.item_cnt_day<1001]
    median = train[(train.shop_id==32)&(train.item_id==2973)&(train.date_block_num==4)&(train.item_price>0)].item_price.median()
    train.loc[train.item_price<0, 'item_price'] = median
    def count_weekend(date_block_num):
        year = 2013 + date_block_num // 12
        month = 1 + date_block_num % 12
        weeknd_count = len([1 for i in calendar.monthcalendar(year, month) if i[6] != 0])
        return weeknd_count

    # several shsops are duplicate, fix the train and test set
    # Якутск Орджоникидзе, 56
    train.loc[train.shop_id == 0, 'shop_id'] = 57
    test.loc[test.shop_id == 0, 'shop_id'] = 57
    # Якутск ТЦ "Центральный"
    train.loc[train.shop_id == 1, 'shop_id'] = 58
    test.loc[test.shop_id == 1, 'shop_id'] = 58
    # Жуковский ул. Чкалова 39м²
    train.loc[train.shop_id == 10, 'shop_id'] = 11
    test.loc[test.shop_id == 10, 'shop_id'] = 11

    shops.loc[shops.shop_name == 'Сергиев Посад ТЦ "7Я"', 'shop_name'] = 'СергиевПосад ТЦ "7Я"'
    shops['city'] = shops['shop_name'].str.split(' ').map(lambda x: x[0])
    shops.loc[shops.city == '!Якутск', 'city'] = 'Якутск'
    shops['city_code'] = LabelEncoder().fit_transform(shops['city'])
    shops = shops[['shop_id','city_code']]

    cats['split'] = cats['item_category_name'].str.split('-')
    cats['type'] = cats['split'].map(lambda x: x[0].strip())
    cats['type_code'] = LabelEncoder().fit_transform(cats['type'])
    # if subtype is nan then type
    cats['subtype'] = cats['split'].map(lambda x: x[1].strip() if len(x) > 1 else x[0].strip())
    cats['subtype_code'] = LabelEncoder().fit_transform(cats['subtype'])
    cats = cats[['item_category_id','type_code', 'subtype_code']]

    items.drop(['item_name'], axis=1, inplace=True)

    # Feature
    # build feature matrix
    matrix = []
    cols = ['date_block_num','shop_id','item_id']
    for i in range(34):
        # 按月分類銷售額
        sales = train[train.date_block_num==i]
        # matrix按月分shop_id + item_id
        matrix.append(np.array(list(product([i], sales.shop_id.unique(), sales.item_id.unique())), dtype='int16'))
        
    matrix = pd.DataFrame(np.vstack(matrix), columns=cols)
    matrix['date_block_num'] = matrix['date_block_num'].astype(np.int8)
    matrix['shop_id'] = matrix['shop_id'].astype(np.int8)
    matrix['item_id'] = matrix['item_id'].astype(np.int16)
    matrix.sort_values(cols,inplace=True)

    #build test
    test['date_block_num'] = 34
    test['date_block_num'] = test['date_block_num'].astype(np.int8)
    test['shop_id'] = test['shop_id'].astype(np.int8)
    test['item_id'] = test['item_id'].astype(np.int16)
    matrix = pd.concat([matrix, test], ignore_index=True, sort=False, keys=cols)
    matrix.fillna(0, inplace=True) # 34 month

    train['revenue'] = train['item_price'] *  train['item_cnt_day']
    # 每月銷售量
    group = train.groupby(['date_block_num','shop_id','item_id']).agg({'item_cnt_day': ['sum']})
    group.columns = ['item_cnt_month']
    group.reset_index(inplace=True)

    matrix = pd.merge(matrix, group, on=cols, how='left')
    matrix['item_cnt_month'] = (matrix['item_cnt_month'].fillna(0).clip(0,20).astype(np.float16))

    matrix = pd.merge(matrix, shops, on=['shop_id'], how='left')
    matrix = pd.merge(matrix, items, on=['item_id'], how='left')
    matrix = pd.merge(matrix, cats, on=['item_category_id'], how='left')
    matrix['city_code'] = matrix['city_code'].astype(np.int8)
    matrix['item_category_id'] = matrix['item_category_id'].astype(np.int8)
    matrix['type_code'] = matrix['type_code'].astype(np.int8)
    matrix['subtype_code'] = matrix['subtype_code'].astype(np.int8)

    # Traget lag feature
    def lag_feature(df, lags, col):
        tmp = df[['date_block_num','shop_id','item_id',col]]
        for i in lags:
            shifted = tmp.copy()
            shifted.columns = ['date_block_num','shop_id','item_id', col+'_lag_'+str(i)]
            shifted['date_block_num'] += i
            df = pd.merge(df, shifted, on=['date_block_num','shop_id','item_id'], how='left')
        return df
    matrix = lag_feature(matrix, [1,2,3,6], 'item_cnt_month')

    # mean encoded features
    # group by date_block_num
    group = matrix.groupby(['date_block_num']).agg({'item_cnt_month': ['mean']})
    group.columns = [ 'date_avg_item_cnt' ]
    group.reset_index(inplace=True)
    matrix = pd.merge(matrix, group, on=['date_block_num'], how='left')
    matrix['date_avg_item_cnt'] = matrix['date_avg_item_cnt'].astype(np.float16)
    matrix = lag_feature(matrix, [1], 'date_avg_item_cnt')
    matrix.drop(['date_avg_item_cnt'], axis=1, inplace=True)

    # group by date_block_num, item_id
    group = matrix.groupby(['date_block_num', 'item_id']).agg({'item_cnt_month': ['mean']})
    group.columns = [ 'date_item_avg_item_cnt' ]
    group.reset_index(inplace=True)
    matrix = pd.merge(matrix, group, on=['date_block_num','item_id'], how='left')
    matrix['date_item_avg_item_cnt'] = matrix['date_item_avg_item_cnt'].astype(np.float16)
    matrix = lag_feature(matrix, [1,2,3,6,12], 'date_item_avg_item_cnt')
    matrix.drop(['date_item_avg_item_cnt'], axis=1, inplace=True)

    # group by date_block_num, shop_id
    group = matrix.groupby(['date_block_num', 'shop_id']).agg({'item_cnt_month': ['mean']})
    group.columns = [ 'date_shop_avg_item_cnt' ]
    group.reset_index(inplace=True)
    matrix = pd.merge(matrix, group, on=['date_block_num','shop_id'], how='left')
    matrix['date_shop_avg_item_cnt'] = matrix['date_shop_avg_item_cnt'].astype(np.float16)
    matrix = lag_feature(matrix, [1,2,3,6,12], 'date_shop_avg_item_cnt')
    matrix.drop(['date_shop_avg_item_cnt'], axis=1, inplace=True)

    # group by date_block_num, item_category_id
    group = matrix.groupby(['date_block_num', 'item_category_id']).agg({'item_cnt_month': ['mean']})
    group.columns = [ 'date_cat_avg_item_cnt' ]
    group.reset_index(inplace=True)
    matrix = pd.merge(matrix, group, on=['date_block_num','item_category_id'], how='left')
    matrix['date_cat_avg_item_cnt'] = matrix['date_cat_avg_item_cnt'].astype(np.float16)
    matrix = lag_feature(matrix, [1], 'date_cat_avg_item_cnt')
    matrix.drop(['date_cat_avg_item_cnt'], axis=1, inplace=True)

    # group by date_block_num, shop_id, item_category_id
    group = matrix.groupby(['date_block_num', 'shop_id', 'item_category_id']).agg({'item_cnt_month': ['mean']})
    group.columns = ['date_shop_cat_avg_item_cnt']
    group.reset_index(inplace=True)
    matrix = pd.merge(matrix, group, on=['date_block_num', 'shop_id', 'item_category_id'], how='left')
    matrix['date_shop_cat_avg_item_cnt'] = matrix['date_shop_cat_avg_item_cnt'].astype(np.float16)
    matrix = lag_feature(matrix, [1], 'date_shop_cat_avg_item_cnt')
    matrix.drop(['date_shop_cat_avg_item_cnt'], axis=1, inplace=True)

    # group by date_block_num, shop_id, type_code
    group = matrix.groupby(['date_block_num', 'shop_id', 'type_code']).agg({'item_cnt_month': ['mean']})
    group.columns = ['date_shop_type_avg_item_cnt']
    group.reset_index(inplace=True)
    matrix = pd.merge(matrix, group, on=['date_block_num', 'shop_id', 'type_code'], how='left')
    matrix['date_shop_type_avg_item_cnt'] = matrix['date_shop_type_avg_item_cnt'].astype(np.float16)
    matrix = lag_feature(matrix, [1], 'date_shop_type_avg_item_cnt')
    matrix.drop(['date_shop_type_avg_item_cnt'], axis=1, inplace=True)


    group = matrix.groupby(['date_block_num', 'shop_id', 'subtype_code']).agg({'item_cnt_month': ['mean']})
    group.columns = ['date_shop_subtype_avg_item_cnt']
    group.reset_index(inplace=True)
    matrix = pd.merge(matrix, group, on=['date_block_num', 'shop_id', 'subtype_code'], how='left')
    matrix['date_shop_subtype_avg_item_cnt'] = matrix['date_shop_subtype_avg_item_cnt'].astype(np.float16)
    matrix = lag_feature(matrix, [1], 'date_shop_subtype_avg_item_cnt')
    matrix.drop(['date_shop_subtype_avg_item_cnt'], axis=1, inplace=True)

    # group by date_block_num, shop_id, city_code
    group = matrix.groupby(['date_block_num', 'city_code']).agg({'item_cnt_month': ['mean']})
    group.columns = [ 'date_city_avg_item_cnt' ]
    group.reset_index(inplace=True)
    matrix = pd.merge(matrix, group, on=['date_block_num', 'city_code'], how='left')
    matrix['date_city_avg_item_cnt'] = matrix['date_city_avg_item_cnt'].astype(np.float16)
    matrix = lag_feature(matrix, [1], 'date_city_avg_item_cnt')
    matrix.drop(['date_city_avg_item_cnt'], axis=1, inplace=True)

    # group by date_block_num, item_id, city_code
    group = matrix.groupby(['date_block_num', 'item_id', 'city_code']).agg({'item_cnt_month': ['mean']})
    group.columns = [ 'date_item_city_avg_item_cnt' ]
    group.reset_index(inplace=True)
    matrix = pd.merge(matrix, group, on=['date_block_num', 'item_id', 'city_code'], how='left')
    matrix['date_item_city_avg_item_cnt'] = matrix['date_item_city_avg_item_cnt'].astype(np.float16)
    matrix = lag_feature(matrix, [1], 'date_item_city_avg_item_cnt')
    matrix.drop(['date_item_city_avg_item_cnt'], axis=1, inplace=True)

    # group by date_block_num, type_code
    group = matrix.groupby(['date_block_num', 'type_code']).agg({'item_cnt_month': ['mean']})
    group.columns = [ 'date_type_avg_item_cnt' ]
    group.reset_index(inplace=True)
    matrix = pd.merge(matrix, group, on=['date_block_num', 'type_code'], how='left')
    matrix['date_type_avg_item_cnt'] = matrix['date_type_avg_item_cnt'].astype(np.float16)
    matrix = lag_feature(matrix, [1], 'date_type_avg_item_cnt')
    matrix.drop(['date_type_avg_item_cnt'], axis=1, inplace=True)

    # group by date_block_num, subtype_code
    group = matrix.groupby(['date_block_num', 'subtype_code']).agg({'item_cnt_month': ['mean']})
    group.columns = [ 'date_subtype_avg_item_cnt' ]
    group.reset_index(inplace=True)
    matrix = pd.merge(matrix, group, on=['date_block_num', 'subtype_code'], how='left')
    matrix['date_subtype_avg_item_cnt'] = matrix['date_subtype_avg_item_cnt'].astype(np.float16)
    matrix = lag_feature(matrix, [1], 'date_subtype_avg_item_cnt')
    matrix.drop(['date_subtype_avg_item_cnt'], axis=1, inplace=True)

    # 趨勢特徵
    group = train.groupby(['item_id']).agg({'item_price': ['mean']})
    group.columns = ['item_avg_item_price']
    group.reset_index(inplace=True)

    matrix = pd.merge(matrix, group, on=['item_id'], how='left')
    matrix['item_avg_item_price'] = matrix['item_avg_item_price'].astype(np.float16)

    group = train.groupby(['date_block_num','item_id']).agg({'item_price': ['mean']})
    group.columns = ['date_item_avg_item_price']
    group.reset_index(inplace=True)

    matrix = pd.merge(matrix, group, on=['date_block_num','item_id'], how='left')
    matrix['date_item_avg_item_price'] = matrix['date_item_avg_item_price'].astype(np.float16)

    lags = [1,2,3,4,5,6]
    matrix = lag_feature(matrix, lags, 'date_item_avg_item_price')

    for i in lags:
        matrix['delta_price_lag_'+str(i)] = \
            (matrix['date_item_avg_item_price_lag_'+str(i)] - matrix['item_avg_item_price']) / matrix['item_avg_item_price']

    def select_trend(row):
        for i in lags:
            if row['delta_price_lag_'+str(i)]:
                return row['delta_price_lag_'+str(i)]
        return 0
        
    matrix['delta_price_lag'] = matrix.apply(select_trend, axis=1)
    matrix['delta_price_lag'] = matrix['delta_price_lag'].astype(np.float16)
    matrix['delta_price_lag'].fillna(0, inplace=True)
    fetures_to_drop = ['item_avg_item_price', 'date_item_avg_item_price']
    for i in lags:
        fetures_to_drop += ['date_item_avg_item_price_lag_'+str(i)]
        fetures_to_drop += ['delta_price_lag_'+str(i)]
    matrix.drop(fetures_to_drop, axis=1, inplace=True)


    # 每月商品總收益 date_shop_revenue
    group = train.groupby(['date_block_num','shop_id']).agg({'revenue': ['sum']})
    group.columns = ['date_shop_revenue']
    group.reset_index(inplace=True)

    matrix = pd.merge(matrix, group, on=['date_block_num','shop_id'], how='left')
    matrix['date_shop_revenue'] = matrix['date_shop_revenue'].astype(np.float32)

    # 每月商品平均收益 shop_avg_revenue
    group = group.groupby(['shop_id']).agg({'date_shop_revenue': ['mean']})
    group.columns = ['shop_avg_revenue']
    group.reset_index(inplace=True)

    matrix = pd.merge(matrix, group, on=['shop_id'], how='left')
    matrix['shop_avg_revenue'] = matrix['shop_avg_revenue'].astype(np.float32)
    matrix['delta_revenue'] = (matrix['date_shop_revenue'] - matrix['shop_avg_revenue']) / matrix['shop_avg_revenue']
    matrix['delta_revenue'] = matrix['delta_revenue'].astype(np.float16)
    matrix = lag_feature(matrix, [1], 'delta_revenue')
    matrix.drop(['date_shop_revenue','shop_avg_revenue','delta_revenue'], axis=1, inplace=True)

    #special features
    matrix['month'] = matrix['date_block_num'] % 12
    days = pd.Series([31,28,31,30,31,30,31,31,30,31,30,31])
    matrix['days'] = matrix['month'].map(days).astype(np.int8)

    cache = {}
    matrix['item_shop_last_sale'] = -1
    matrix['item_shop_last_sale'] = matrix['item_shop_last_sale'].astype(np.int8)
    for idx, row in matrix.iterrows():    
        key = str(row.item_id)+' '+str(row.shop_id)
        if key not in cache:
            if row.item_cnt_month!=0:
                cache[key] = row.date_block_num
        else:
            last_date_block_num = cache[key]
            matrix.at[idx, 'item_shop_last_sale'] = row.date_block_num - last_date_block_num
            cache[key] = row.date_block_num

    cache = {}
    matrix['item_last_sale'] = -1
    matrix['item_last_sale'] = matrix['item_last_sale'].astype(np.int8)
    for idx, row in matrix.iterrows():    
        key = row.item_id
        if key not in cache:
            if row.item_cnt_month!=0:
                cache[key] = row.date_block_num
        else:
            last_date_block_num = cache[key]
            if row.date_block_num>last_date_block_num:
                matrix.at[idx, 'item_last_sale'] = row.date_block_num - last_date_block_num
                cache[key] = row.date_block_num

    matrix['item_shop_first_sale'] = matrix['date_block_num'] - matrix.groupby(['item_id','shop_id'])['date_block_num'].transform('min')
    matrix['item_first_sale'] = matrix['date_block_num'] - matrix.groupby('item_id')['date_block_num'].transform('min')



    # 統計前三個月相似產品的銷售額
    def lag_feature_adv(df, lags, col):
        tmp = df[['date_block_num','shop_id','item_id',col]]
        for i in lags:
            shifted = tmp.copy()
            shifted.columns = ['date_block_num','shop_id','item_id', col+'_lag_'+str(i)+'_adv']
            shifted['date_block_num'] += i
            shifted['item_id'] -= 1
            df = pd.merge(df, shifted, on=['date_block_num','shop_id','item_id'], how='left')
            df[col+'_lag_'+str(i)+'_adv'] = df[col+'_lag_'+str(i)+'_adv'].astype('float16')
        return df
    matrix = lag_feature_adv(matrix, [1, 2, 3], 'item_cnt_month')

    # Final preparations
    matrix = matrix[matrix.date_block_num > 5]
    def fill_na(df):
        for col in df.columns:
            if ('_lag_' in col) & (df[col].isnull().any()):
                if ('item_cnt' in col):
                    df[col].fillna(0, inplace=True)         
        return df
    matrix = fill_na(matrix)
    matrix.to_pickle('new_train.pkl')

# import datasets
items = pd.read_csv('./datasets/items.csv')
shops = pd.read_csv('./datasets/shops.csv')
cats = pd.read_csv('./datasets/item_categories.csv')
train = pd.read_csv('./datasets/sales_train.csv')
test  = pd.read_csv('./datasets/test.csv').set_index('ID')

save_feature(train, items, shops, cats, test)
