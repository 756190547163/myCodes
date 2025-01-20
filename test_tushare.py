import tushare as ts

ts.set_token("c652596d18e724a120b7fdedf85b2961b567712e4150ab68a214f2b7")
pro = ts.pro_api()

# 获取股票基础信息
stocks = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')
print(stocks.head())
