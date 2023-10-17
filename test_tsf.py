from model import *
from loader import *
import argparse
import random, os
import numpy as np
import torch
import matplotlib.pyplot as plt 

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def test(model, dataset, history_len, pred_len, batch_size):
    model.eval()
    final_preds = []
    with torch.no_grad():
        total_test_loss = 0
        total_test_steps = 0
        for idx in range(history_len, len(dataset.test_df)-pred_len, batch_size):
                print("Yes test function is running")
                past_ts_list = []
                series_names_tokenized_iid_list = []
                series_names_tokenized_am_list = []
                date_features_vector_list = []
                future_ts_list = []
                future_ts_list2 = []
                related_ts_list = []
                related_series_names_tokenized_iid_list = []
                related_series_names_tokenized_am_list = []
                col = 'Price'
                past_ts, series_names_tokenized, date_features_vector, future_date_features_vector, future_ts, related_ts, related_series_names_tokenized = dataset.get_io_i('test', col, idx, history_len, pred_len)
                n = len(related_ts[0])
                a = [[] for x in range(n)]
                b = [[] for x in range(n)]
                for i in range(batch_size):
                    if idx+i>=len(dataset.test_df)-pred_len:
                        break
                    col = 'Price'
                    past_ts, series_names_tokenized, date_features_vector, future_date_features_vector, future_ts, related_ts, related_series_names_tokenized = dataset.get_io_i('test', col, idx+i, history_len, pred_len)
                    past_ts_list.append(past_ts)
                    series_names_tokenized_iid_list.append(series_names_tokenized['input_ids'])
                    series_names_tokenized_am_list.append(series_names_tokenized['attention_mask'])
                    date_features_vector_list.append(date_features_vector)
                    future_ts_list.append(future_ts)
                    future_ts_list2.append(future_date_features_vector)
                    related_ts_list.append(related_ts)
                    n = len(related_ts[0])
                    for j in range(n):
                        a[j].append(related_series_names_tokenized[j]['input_ids'])
                        b[j].append(related_series_names_tokenized[j]['attention_mask'])
                #print("Length of tokenized" , len(series_names_tokenized_iid_list))
                series_names_tokenized = {'input_ids':torch.cat(tensors=series_names_tokenized_iid_list, dim=0), 'attention_mask':torch.cat(tensors=series_names_tokenized_am_list, dim=0)}
                rel_series_final = []
                for i in range(n):
                    related_series_names_tokenized = {'input_ids':torch.cat(tensors=a[i], dim=0), 'attention_mask':torch.cat(tensors=b[i], dim=0)}
                    rel_series_final.append(related_series_names_tokenized)
                #related_series_names_tokenized = {'input_ids':torch.cat(tensors=related_series_names_tokenized_iid_list, dim=0), 'attention_mask':torch.cat(tensors=related_series_names_tokenized_am_list, dim=0)}
                past_ts = torch.cat(tensors=past_ts_list, dim=0)
                date_features_vector = torch.cat(tensors=date_features_vector_list, dim=0)
                related_ts = torch.cat(tensors=related_ts_list, dim=0)
                future_ts = torch.cat(tensors=future_ts_list, dim=0)
                future_ts2 = torch.cat(tensors=future_ts_list2, dim=0)
                preds = model(past_ts, series_names_tokenized, date_features_vector, future_ts2, related_ts, rel_series_final, None, col)
                print(preds)
                #print("Type of preds",preds.type)
                for i in range(len(preds)):
                    final_preds.append(preds[i][0])
                loss = loss_function(future_ts, preds)
                total_test_loss+=loss
                total_test_steps+=1
        
        print('Test loss: ', total_test_loss/total_test_steps)
    #print(len(final_preds))
    #print(final_preds[0])
    return final_preds

parser = argparse.ArgumentParser(prog = 'TSF_Tester', description = 'Testing TiDE model for TSF')

# training hyperparameters
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--lr', type=float, default=2e-5)

#parser.add_argument('--related_series_type', type=str, default='llm') # can be llm or random
parser.add_argument('--global_model', type=bool, default=True)
parser.add_argument('--global_encoder', type=bool, default=True)

parser.add_argument('--history_len', type=int, default=4)
parser.add_argument('--pred_len', type=int, default=2)
parser.add_argument('--text_dim', type=int, default=4)
if __name__ == '__main__':
    args = parser.parse_args()

    #seed 
    seed_everything(args.seed)
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    #device = torch.device('cpu')
    print('Device: ', device)
    global_model = args.global_model
    global_encoder = args.global_encoder
    history_len = args.history_len
    pred_len = args.pred_len
    text_dim = args.text_dim
    batch_size = args.batch_size
    dataset = BankNiftyFuturesDataset(device)
    model_config = {'input_dim':text_dim+history_len*(1+len(dataset.time_features_names)), 'hidden_dims':[4, 4], 'decoder_output_dim':pred_len, 'final_decoder_hidden':1, 'text_dim':text_dim}

    model = TideModel(model_config, pred_len, history_len, device, False, 0.0, global_model, dataset.all_series_names, global_encoder,len(dataset.all_series_names)).to(device)
    path = '/mnt/nas/kunalchhabra/TIDE_Models/tsf_global_encoder__epoch_10'
    model.load_state_dict(torch.load(path))
    
    loss_function = torch.nn.MSELoss()
    final_preds = test(model,dataset,history_len,pred_len,batch_size)
    finall_preds = []
    for elem in final_preds:
        finall_preds.append(np.array(elem.cpu().numpy()))
    print(len(finall_preds))
    test_df = dataset.test_df

    dates_array = np.array(test_df['Date'][history_len:len(dataset.test_df)-pred_len])
    price_array = np.array(test_df['Price'][history_len:len(dataset.test_df)-pred_len])
    preds_array = np.array(finall_preds)
    # plot lines 
    plt.plot(dates_array, price_array, label = "Actual Price")
    plt.plot(dates_array, preds_array, label = "Predicted Price") 
    plt.legend() 
    plt.show()
    plt.savefig('Test_results.png')


    
