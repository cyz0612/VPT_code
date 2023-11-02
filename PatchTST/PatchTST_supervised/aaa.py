import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

data_dir = "../../Multidimensional-time-series-with-transformer-main/0307data"
drop = [
"310103194908240027",
"342623198107226821",
"310104196506112810",
"450821201009125820"]
for file in os.listdir(data_dir):
    folder = os.path.join(data_dir,file)
    # if file != '34242319790406396X':continue
    if os.path.isdir(folder):
        
        train_data = pd.read_excel(os.path.join(folder,"select_train.xlsx"))
        if len(train_data)>2000:
            if file not in drop:
                train_data['real_load'] = savgol_filter(train_data['real_load'],51,2)
                train_data['date'] = pd.to_datetime(train_data['date_str'])
                train_data.set_index('date', inplace=True)

                grouped = train_data.groupby(train_data.index.date)
                mean = grouped.mean(numeric_only=True)
                
                mean = grouped.mean(numeric_only=True)
                mean['bias']=mean.real_load-mean.target_load
                # std = grouped.std()
                # 计算每个日期与第一个日期之间的天数差
                helper = pd.DataFrame({'date': pd.date_range(start=mean.index[0], end=mean.index[-1])})
                
                mean['days'] = (mean.index - mean.index[0]).days
                xmin = mean['days'].min()
                mean['days'] = mean['days'] - xmin

                mean.set_index('days', inplace=True)
                d = pd.concat([helper,mean.real_load,mean.target_load],axis=1)
                d['real_load'] = d['real_load'].interpolate(method='linear')
                d['OT'] = d['target_load'].interpolate(method='linear')
                d.drop(['target_load'],axis=1,inplace=True)
                
                if len(d)>30:
                    d.to_csv(os.path.join("./dataset/wb_days",file+".csv"),index=False)

