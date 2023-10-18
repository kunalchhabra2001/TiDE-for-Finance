import torch
from torch import Tensor
from torch.nn import functional as F
from transformers import DistilBertModel
import pandas as pd
from transformers import DistilBertTokenizer
from tqdm import tqdm
import statistics
import numpy as np
from time_features import *
import random
from matplotlib import pyplot as plt
import warnings
from revIN import RevIN

class CPIDataset():
    def __init__(self, related_series_type, device,):
        self.date_features_dict = {}
        self.all_series_indices = {}
        self.related = {}
        self.all_series_names = []
        self.time_features_names = ['DayOfWeek', 'DayOfMonth', 'DayOfYear', 'MonthOfYear', 'WeekOfYear', 'Year']
        self.device = device
        df = pd.read_csv('data.csv')

        # setup time features
        time = TimeFeatures(self.time_features_names)
        dates = df['date'].values
        # print(dates)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dates_pd = pd.to_datetime(dates, format='%d-%m-%Y')
        # print(dates_pd)
        # exit()
        date_features = time.time_features(dates_pd).astype(np.float32)
        # print(date_features)
        # exit()
        for i, date in enumerate(dates):
            self.date_features_dict[date] = date_features[i]
        
        # setup series names features
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.all_series_names = list(df.columns)[1:]
        # self.all_series_names = [list(df.columns)[1]]
        for i, asn in enumerate(self.all_series_names):
            self.all_series_indices[asn] = i
        self.all_series_names_tokenized = tokenizer(self.all_series_names,max_length=512,truncation=True,padding='max_length',return_attention_mask=True,return_token_type_ids=False,add_special_tokens=True,return_tensors='pt')

        # setup train-val-test split
        df['year'] = df.date.str[-2:]
        train_df = df[df['year'].isin(['13', '14', '15', '16', '17', '18'])]
        val_df = df[df['year']=='19']
        test_df = df[df['year']=='20']
        self.train_df = train_df.drop(['year'], axis=1)
        self.val_df = val_df.drop(['year'], axis=1)
        self.test_df = test_df.drop(['year'], axis=1)

        # setup related series
        if related_series_type=='llm':
            self.related = {'Rice': 'Onion', 'Wheat': 'Gur', 'Atta': 'Onion', 'Gram Dal': 'Tur Dal', 'Tur Dal': 'Rice', 'Urad Dal': 'Masoor Dal', 'Moong Dal': 'Masoor Dal', 'Masoor Dal': 'Moong Dal', 'Sugar': 'Wheat', 'Milk': 'Rice', 'Groundnut Oil': 'Sunflower Oil', 'Mustard Oil': 'Onion', 'Vanaspati': 'Wheat', 'Soya Oil': 'Gur', 'Sunflower Oil': 'Sugar', 'Palm Oil': 'Onion', 'Gur': 'Mustard Oil', 'Tea Loose': 'Rice', 'Salt Pack': 'Sugar', 'Potato': 'Onion', 'Onion': 'Potato', 'Tomato': 'Onion'}
        elif related_series_type=='random':
            self.related = {}
            for s in self.all_series_names:
                choice = random.choice(self.all_series_names)
                while s==choice:
                    choice = random.choice(self.all_series_names)
                self.related[s] = choice
        else:
            raise NotImplementedError('related_series_type can be either llm or random')
        

    def get_io_i(self, df_name, col, i, left_len, right_len, history_len, pred_len, obj_type='tsf'):
        if df_name=='train':
            df = self.train_df
        elif df_name=='val':
            df = self.val_df
        elif df_name=='test':
            df = self.test_df
        if obj_type=='tsf':
            assert i >= history_len
            assert i+pred_len < len(df)
            past_ts = torch.tensor(list(df[col].iloc[i-history_len:i])).to(self.device)
            dates = df['date'].iloc[i-history_len:i]
        elif obj_type=='mvi':
            assert i >= left_len
            assert i+pred_len+right_len < len(df)
            past_ts = torch.tensor(list(df[col].iloc[i-left_len:i])+list(df[col].iloc[i+pred_len:i+pred_len+right_len])).to(self.device)
            dates = list(df['date'].iloc[i-left_len:i])+list(df['date'].iloc[i+pred_len:i+pred_len+right_len])

        ptm = torch.mean(past_ts)
        pts = torch.std(past_ts)
        future_ts = torch.tensor(list(df[col].iloc[i:i+pred_len])).to(self.device)
        past_ts = (past_ts-ptm)/pts
        future_ts = (future_ts-ptm)/pts

        series_names_tokenized_iid = self.all_series_names_tokenized['input_ids'][self.all_series_indices[col]].to(self.device)
        series_names_tokenized_am = self.all_series_names_tokenized['attention_mask'][self.all_series_indices[col]].to(self.device)
        series_names_tokenized = {'input_ids':series_names_tokenized_iid.unsqueeze(0), 'attention_mask':series_names_tokenized_am.unsqueeze(0)}

        date_features_vector = []
        for date in dates:
            date_feature = list(self.date_features_dict[date])
            date_features_vector.append(date_feature)
        date_features_vector = torch.tensor(date_features_vector).to(self.device)

        rel_col = self.related[col]
        if obj_type=='tsf':
            rel_past_ts = torch.tensor(list(df[rel_col].iloc[i-history_len:i])).to(self.device)
        elif obj_type=='mvi':
            rel_past_ts = torch.tensor(list(df[rel_col].iloc[i-left_len:i])+list(df[rel_col].iloc[i+pred_len:i+pred_len+right_len])).to(self.device)
        rel_ptm = torch.mean(rel_past_ts)
        rel_pts = torch.std(rel_past_ts)
        rel_past_ts = (rel_past_ts-rel_ptm)/rel_pts

        rel_series_names_tokenized_iid = self.all_series_names_tokenized['input_ids'][self.all_series_indices[rel_col]].to(self.device)
        rel_series_names_tokenized_am = self.all_series_names_tokenized['attention_mask'][self.all_series_indices[rel_col]].to(self.device)
        rel_series_names_tokenized = {'input_ids':rel_series_names_tokenized_iid.unsqueeze(0), 'attention_mask':rel_series_names_tokenized_am.unsqueeze(0)}


        return past_ts.unsqueeze(0), series_names_tokenized, date_features_vector.unsqueeze(0), future_ts.unsqueeze(0), rel_past_ts.unsqueeze(0), rel_series_names_tokenized

class BankNiftyFuturesDataset():
    def __init__(self, device):
        self.date_features_dict = {}
        self.all_series_indices = {}
        self.all_series_names = []
        self.time_features_names = ['DayOfWeek', 'DayOfMonth', 'DayOfYear', 'MonthOfYear', 'WeekOfYear', 'Year']
        self.device = device
        df = pd.read_excel('Finall_Futures_Data.xlsx')

        # setup time features
        time = TimeFeatures(self.time_features_names)
        dates = df['Date'].values
        #df.insert(1,"Country","India")
        # print(dates)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dates_pd = pd.to_datetime(dates, format='%Y-%m-%d %H:%M:%S',utc=True)
        # print(dates_pd)
        # exit()
        date_features = time.time_features(dates_pd).astype(np.float32)
        # print(date_features)
        # exit()
        for i, date in enumerate(dates):
            self.date_features_dict[date] = date_features[i]
        
        # setup series names features
        #tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.all_series_names = list(df.columns)[1:]
        # self.all_series_names = [list(df.columns)[1]]
        for i, asn in enumerate(self.all_series_names):
            self.all_series_indices[asn] = i
        #self.all_series_names_tokenized = tokenizer(self.all_series_names,max_length=512,truncation=True,padding='max_length',return_attention_mask=True,return_token_type_ids=False,add_special_tokens=True,return_tensors='pt')

        # setup train-val-test split
        df['year'] = df.Date.dt.year
        train_df = df[df['year'].isin([2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021])]
        val_df = df[df['year']==2022]
        test_df = df[df['year']==2023]
        self.train_df = train_df.drop(['year'], axis=1)
        self.val_df = val_df.drop(['year'], axis=1)
        self.test_df = test_df.drop(['year'], axis=1)

        # setup related series
        # if related_series_type=='llm':
        #     self.related = {'sma5': 'sma10', 'sma5': 'sma15', 'sma5': 'sma20', 'ema5': 'ema10', 'ema5': 'ema15', 'ema5': 'ema20', 'middleband': 'upperband', 'middleband': 'lowerband'}
        # elif related_series_type=='random':
        #     self.related = {}
        #     for s in self.all_series_names:
        #         choice = random.choice(self.all_series_names)
        #         while s==choice:
        #             choice = random.choice(self.all_series_names)
        #         self.related[s] = choice
        # else:
        #     raise NotImplementedError('related_series_type can be either llm or random')
        

    def get_io_i(self, df_name, col,i, history_len, pred_len):
        if df_name=='train':
            df = self.train_df
        elif df_name=='val':
            df = self.val_df
        elif df_name=='test':
            df = self.test_df
        assert i >= history_len
        assert i+pred_len < len(df)
        past_ts = torch.tensor(list(df[col].iloc[i-history_len:i])).to(self.device)
        dates = df['Date'].iloc[i-history_len:i]
        future_dates = df['Date'].iloc[i:i+pred_len]
        ptm = torch.mean(past_ts)
        pts = torch.std(past_ts)
        future_ts = torch.tensor(list(df[col].iloc[i:i+pred_len])).to(self.device)
        past_ts = (past_ts-ptm)/pts
        #future_ts = (future_ts-ptm)/pts
        #print("Past TS",past_ts)
        #print("Future TS",future_ts)

        # series_names_tokenized_iid = self.all_series_names_tokenized['input_ids'][self.all_series_indices[col]].to(self.device)
        # series_names_tokenized_am = self.all_series_names_tokenized['attention_mask'][self.all_series_indices[col]].to(self.device)
        # series_names_tokenized = {'input_ids':series_names_tokenized_iid.unsqueeze(0), 'attention_mask':series_names_tokenized_am.unsqueeze(0)}

        date_features_vector = []
        future_date_features_vector = []
        for date in dates:
            date_feature = list(self.date_features_dict[date.to_numpy()])
            date_features_vector.append(date_feature)
        for date in future_dates:
            date_feature = list(self.date_features_dict[date.to_numpy()])
            future_date_features_vector.append(date_feature)
        date_features_vector = torch.tensor(date_features_vector).to(self.device)
        future_date_features_vector = torch.tensor(future_date_features_vector).to(self.device)
        df2 = df.loc[:,~df.columns.isin(['Price','Date'])].iloc[i-history_len:i]
        rel_col_names = self.all_series_names[1:]
        #rel_past_ts = torch.tensor(np.array(df2.values)).to(self.device)
        list_of_t = []
              
        # rel_ptm = torch.mean(rel_past_ts)
        # rel_pts = torch.std(rel_past_ts)
        for colum in df2.columns:
            rel_past_ts = torch.tensor(np.array(df2[colum])).to(self.device)
            rel_ptm = torch.mean(rel_past_ts)
            rel_pts = torch.std(rel_past_ts)
            rel_past_ts = (rel_past_ts-rel_ptm)/rel_pts
            list_of_t.append(rel_past_ts)
            #rel_past_ts = list_of_t[0]
            rel_past_ts = torch.cat(tensors=list_of_t, dim=0)
        #print("Related Series", rel_past_ts)
        # rel_series_names_tokenized_list = [] #List of dictionaries
        # for rel_col in rel_col_names:
        #     rel_series_names_tokenized_iid = self.all_series_names_tokenized['input_ids'][self.all_series_indices[rel_col]].to(self.device)
        #     rel_series_names_tokenized_am = self.all_series_names_tokenized['attention_mask'][self.all_series_indices[rel_col]].to(self.device)
        #     rel_series_names_tokenized = {'input_ids':rel_series_names_tokenized_iid.unsqueeze(0), 'attention_mask':rel_series_names_tokenized_am.unsqueeze(0)}
        #     rel_series_names_tokenized_list.append(rel_series_names_tokenized)

        return past_ts.unsqueeze(0), date_features_vector.unsqueeze(0), future_date_features_vector.unsqueeze(0),future_ts.unsqueeze(0), rel_past_ts
        #return past_ts.unsqueeze(0), series_names_tokenized, date_features_vector.unsqueeze(0), future_date_features_vector.unsqueeze(0),future_ts.unsqueeze(0), rel_past_ts, rel_series_names_tokenized_list

class OptionsDataset():
    def __init__(self, device):
        self.date_features_dict = {}
        self.all_series_indices = {}
        self.all_series_names = []
        self.time_features_names = ['DayOfWeek', 'DayOfMonth', 'DayOfYear', 'MonthOfYear', 'WeekOfYear', 'Year']
        self.device = device
        df = pd.read_csv('OptionsTrain.csv')
        df = df.dropna()

        # setup time features
        time = TimeFeatures(self.time_features_names)
        dates = df['Date'].values
        #df.insert(1,"Country","India")
        # print(dates)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dates_pd = pd.to_datetime(dates, format='%d-%m-%Y')
        # print(dates_pd)
        # exit()
        date_features = time.time_features(dates_pd).astype(np.float32)
        # print(date_features)
        # exit()
        for i, date in enumerate(dates):
            self.date_features_dict[date] = date_features[i]
        
        # setup series names features
        #tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        # self.all_series_names = list(df.columns)[1:]
        # # self.all_series_names = [list(df.columns)[1]]
        # for i, asn in enumerate(self.all_series_names):
        #     self.all_series_indices[asn] = i
        #self.all_series_names_tokenized = tokenizer(self.all_series_names,max_length=512,truncation=True,padding='max_length',return_attention_mask=True,return_token_type_ids=False,add_special_tokens=True,return_tensors='pt')

        # setup train-val-test split
        df['year'] = df.Date.str[-2:]
        train_df = df[df['year'].isin(['19', '20', '21'])]
        val_df = df[df['year'].isin(['22'])]
        test_df = pd.read_csv('OptionsTest.csv')
        self.test_df = test_df.dropna()
        self.train_df = train_df.drop(['year'], axis=1)
        self.val_df = val_df.drop(['year'], axis=1)
        #self.test_df = test_df.drop(['year'], axis=1)

        # setup related series
        # if related_series_type=='llm':
        #     self.related = {'sma5': 'sma10', 'sma5': 'sma15', 'sma5': 'sma20', 'ema5': 'ema10', 'ema5': 'ema15', 'ema5': 'ema20', 'middleband': 'upperband', 'middleband': 'lowerband'}
        # elif related_series_type=='random':
        #     self.related = {}
        #     for s in self.all_series_names:
        #         choice = random.choice(self.all_series_names)
        #         while s==choice:
        #             choice = random.choice(self.all_series_names)
        #         self.related[s] = choice
        # else:
        #     raise NotImplementedError('related_series_type can be either llm or random')
        

    def get_io_i(self, df_name, col,i, history_len, pred_len):
        if df_name=='train':
            df = self.train_df
        elif df_name=='val':
            df = self.val_df
        elif df_name=='test':
            df = self.test_df
        assert i >= history_len
        assert i+pred_len < len(df)
        past_ts = torch.tensor(list(df[col].iloc[i-history_len:i])).to(self.device)
        dates = df['Date'].iloc[i-history_len:i]
        future_dates = df['Date'].iloc[i:i+pred_len]
        ptm = torch.mean(past_ts)
        pts = torch.std(past_ts)
        future_ts = torch.tensor(list(df[col].iloc[i:i+pred_len])).to(self.device)
        past_ts = (past_ts-ptm)/pts
        print("Past TS", past_ts)
        #future_ts = (future_ts-ptm)/pts
        print("Future TS", future_ts)

        # series_names_tokenized_iid = self.all_series_names_tokenized['input_ids'][self.all_series_indices[col]].to(self.device)
        # series_names_tokenized_am = self.all_series_names_tokenized['attention_mask'][self.all_series_indices[col]].to(self.device)
        # series_names_tokenized = {'input_ids':series_names_tokenized_iid.unsqueeze(0), 'attention_mask':series_names_tokenized_am.unsqueeze(0)}

        date_features_vector = []
        future_date_features_vector = []
        for date in dates:
            date_feature = list(self.date_features_dict[date])
            date_features_vector.append(date_feature)
        for date in future_dates:
            date_feature = list(self.date_features_dict[date])
            future_date_features_vector.append(date_feature)
        date_features_vector = torch.tensor(date_features_vector).to(self.device)
        future_date_features_vector = torch.tensor(future_date_features_vector).to(self.device)
        df2 = df.loc[:,~df.columns.isin(['Option Price','Date'])].iloc[i-history_len:i]
        rel_col_names = self.all_series_names[1:]
        list_of_t = []
        for colum in df2.columns:
            rel_past_ts = torch.tensor(np.array(df2[colum])).to(self.device)
            rel_ptm = torch.mean(rel_past_ts)
            rel_pts = torch.std(rel_past_ts)
            rel_past_ts = (rel_past_ts-rel_ptm)/rel_pts
            list_of_t.append(rel_past_ts)
        rel_past_ts = list_of_t[0]
        for j in range(1,len(list_of_t)):
            rel_past_ts = torch.cat(rel_past_ts,list_of_t[j])
        # rel_past_ts = torch.tensor(np.array(df2.values)).to(self.device)
        # rel_ptm = torch.mean(rel_past_ts)
        # rel_pts = torch.std(rel_past_ts)
        # rel_past_ts = (rel_past_ts-rel_ptm)/rel_pts
        print("Related past ts",rel_past_ts)
        #rel_series_names_tokenized_list = [] #List of dictionaries
        # for rel_col in rel_col_names:
        #     rel_series_names_tokenized_iid = self.all_series_names_tokenized['input_ids'][self.all_series_indices[rel_col]].to(self.device)
        #     rel_series_names_tokenized_am = self.all_series_names_tokenized['attention_mask'][self.all_series_indices[rel_col]].to(self.device)
        #     rel_series_names_tokenized = {'input_ids':rel_series_names_tokenized_iid.unsqueeze(0), 'attention_mask':rel_series_names_tokenized_am.unsqueeze(0)}
        #     rel_series_names_tokenized_list.append(rel_series_names_tokenized)

        return past_ts.unsqueeze(0), date_features_vector.unsqueeze(0), future_date_features_vector.unsqueeze(0),future_ts.unsqueeze(0), rel_past_ts
        #return past_ts.unsqueeze(0), series_names_tokenized, date_features_vector.unsqueeze(0), future_date_features_vector.unsqueeze(0),future_ts.unsqueeze(0), rel_past_ts, rel_series_names_tokenized_list
