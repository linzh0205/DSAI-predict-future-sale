# DSAI-predict-future-sale
### [Predict Future Sales](https://www.kaggle.com/c/competitive-data-science-predict-future-sales/overview)

### Data Description

### File descriptions
- sales_train.csv - the training set. Daily historical data from January 2013 to October 2015.
- test.csv - the test set. You need to forecast the sales for these shops and products for November 2015.
- sample_submission.csv - a sample submission file in the correct format.
- items.csv - supplemental information about the items/products.
- item_categories.csv  - supplemental information about the items categories.
- shops.csv- supplemental information about the shops.

![descriotions](https://github.com/linzh0205/DSAI-predict-future-sale/blob/main/fig/data_de.jpg)

### Data Analysises

### Method

### Result

## Run the code
環境
Python 3.7.1

安裝requirements.txt套件:
```
conda install --yes --file requirements.txt
```
將dataset.zip解壓縮與XGBmodel.py、feature.py存在同路徑下

執行feature.py進行特徵擷取，會得到所有擷取後的特徵檔案data.pkl:
```
python feature.py
```
接著執行XGBmodel.py，得到submission.csv上傳至Kaggle
```
python XGBmodel.py
```


## [Google PPT Link](https://drive.google.com/file/d/1RNj0FqVb39bEE_Ckr_pQysmtsf21fg75/view?usp=sharing)
