# DSAI-Predict Future Sales
### [Predict Future Sales](https://www.kaggle.com/c/competitive-data-science-predict-future-sales/overview)
![pfs](https://github.com/linzh0205/DSAI-predict-future-sale/blob/main/fig/pfs.JPG)
### Data Description
You are provided with daily historical sales data. The task is to forecast the total amount of products sold in every shop for the test set. Note that the list of shops and products slightly changes every month. Creating a robust model that can handle such situations is part of the challenge.

## File descriptions
- sales_train.csv - the training set. Daily historical data from January 2013 to October 2015.
- test.csv - the test set. You need to forecast the sales for these shops and products for November 2015.
- sample_submission.csv - a sample submission file in the correct format.
- items.csv - supplemental information about the items/products.
- item_categories.csv  - supplemental information about the items categories.
- shops.csv- supplemental information about the shops.

![descriotions](https://github.com/linzh0205/DSAI-predict-future-sale/blob/main/fig/data_de.jpg)


### Data Analysises
-> 兩年冬季商品銷售量較高，表示資料具有季節性

-> 從2013年至2015年，每月銷售數量有下降的趨勢

-> 商品價格會影響商品銷售量

-> 商店所在城市會影響商品銷售量(EX:大城市消費力較高)

![trend](https://github.com/linzh0205/DSAI-predict-future-sale/blob/main/fig/trend.jpg)



#### Method

## Preprocessing & Feature Selection
- 清除outliers
- 建立新的matrix作為商品與商店每月的train set
- 得到商品月銷售額merge到matrix
- 將商品名稱、商店名稱、商品類型merge到matrix
- 增加與目標相關的lag features(EX:每月整體商品銷售量、平均銷售量、商店中的商品平均銷售量)
- 增加城市地點作為特徵
- 增加價格趨勢特徵(EX:各類型商品價格、商店總銷售額、平均銷售額)
- 增加每月周末數作為特徵
- 增加前幾個月相似商品銷售額作為特徵
- 區分剛上市的商品與已上市的商品作為特徵(剛上市的商品無2015以前的歷史資料，而已上市商品有過去歷史資料提供時間序列的關係)

## Training model
XGBoost
- fit XGBoost model
- Grid Search CV自動調整參數XGBoost的參數(max_depth、n_estimators、 subsample 、colsamplebytree)

LSTM
- fit LSTM model


### Result

XGBoost無調整參數模型預測結果:

![rmse1](https://github.com/linzh0205/DSAI-predict-future-sale/blob/main/fig/rmse1.jpg)

使用Grid Search CV 調整模型預測結果:

![rmse3](https://github.com/linzh0205/DSAI-predict-future-sale/blob/main/fig/0.90705.JPG)


LSTM模型預測結果:

![rmse2](https://github.com/linzh0205/DSAI-predict-future-sale/blob/main/fig/lstm_model_result_2.PNG)


#### [Google PPT Link](https://drive.google.com/file/d/1RNj0FqVb39bEE_Ckr_pQysmtsf21fg75/view?usp=sharing)

### Run the code
環境
Python 3.7
```
conda create -n test python==3.7
```
```
activate test
```
安裝requirements.txt套件:
```
pip install -r requirements.txt
```
- 下載dataset.zip、Xgbl.py、feature.py、model.pickle.dat
- 將dataset.zip解壓縮後與Xgbl.py、feature.py、model.pickle.dat存在同路徑下
- 執行feature.py進行特徵擷取，會得到所有擷取後的特徵檔案new_train.pkl
## 由於執行feature時間會較長，因此我們提供已經檔案的雲端連結給使用者下載

#### [new_train.pkl Download Link](https://drive.google.com/file/d/1zHi3ElAZc9lNlDDLe2vlPW4aocoz2MiL/view?usp=sharing)

- 將下載後的檔案存到與Xgbl.py的相同路徑下

- 接著執行Xgbl.py，得到submission.csv上傳至Kaggle
```
python Xgbl.py
```


