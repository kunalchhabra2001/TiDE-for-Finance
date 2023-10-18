from model import *
from loader import *
import argparse
import sys
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

# def validate(model, dataset, history_len, pred_len, batch_size):
#     model.eval()
#     with torch.no_grad():
#         total_val_loss = 0
#         total_val_steps = 0
#         for col in dataset.all_series_names:
#             print('Val for col ', col)
#             col_val_loss = 0
#             col_val_steps = 0
#             for idx in range(history_len, len(dataset.val_df)-pred_len, batch_size):
#                 past_ts_list = []
#                 series_names_tokenized_iid_list = []
#                 series_names_tokenized_am_list = []
#                 date_features_vector_list = []
#                 future_ts_list = []
#                 related_ts_list = []
#                 related_series_names_tokenized_iid_list = []
#                 related_series_names_tokenized_am_list = []
#                 for i in range(batch_size):
#                     if idx+i>=len(dataset.val_df)-pred_len:
#                         break
#                     past_ts, series_names_tokenized, date_features_vector, future_ts, related_ts, related_series_names_tokenized = dataset.get_io_i('val', col, idx+i, 0, 0, history_len, pred_len, 'tsf')
#                     past_ts_list.append(past_ts)
#                     series_names_tokenized_iid_list.append(series_names_tokenized['input_ids'])
#                     series_names_tokenized_am_list.append(series_names_tokenized['attention_mask'])
#                     date_features_vector_list.append(date_features_vector)
#                     future_ts_list.append(future_ts)
#                     related_ts_list.append(related_ts)
#                     related_series_names_tokenized_iid_list.append(related_series_names_tokenized['input_ids'])
#                     related_series_names_tokenized_am_list.append(related_series_names_tokenized['attention_mask'])
#                 series_names_tokenized = {'input_ids':torch.cat(tensors=series_names_tokenized_iid_list, dim=0), 'attention_mask':torch.cat(tensors=series_names_tokenized_am_list, dim=0)}
#                 related_series_names_tokenized = {'input_ids':torch.cat(tensors=related_series_names_tokenized_iid_list, dim=0), 'attention_mask':torch.cat(tensors=related_series_names_tokenized_am_list, dim=0)}
#                 past_ts = torch.cat(tensors=past_ts_list, dim=0)
#                 date_features_vector = torch.cat(tensors=date_features_vector_list, dim=0)
#                 related_ts = torch.cat(tensors=related_ts_list, dim=0)
#                 future_ts = torch.cat(tensors=future_ts_list, dim=0)
#                 preds = model(past_ts, series_names_tokenized, date_features_vector, related_ts, related_series_names_tokenized, None, col, dataset.related[col])
#                 loss = loss_function(future_ts, preds)
#                 total_val_loss+=loss
#                 total_val_steps+=1
#                 col_val_loss+=loss
#                 col_val_steps+=1
#             print('Val col loss: ', col_val_loss/col_val_steps)    
#         print('Val loss: ', total_val_loss/total_val_steps)
#     model.train()
def validate(model, dataset, history_len, pred_len, batch_size):
    model.eval()
    with torch.no_grad():
        total_val_loss = 0
        total_val_steps = 0
        # print(history_len)
        # print(len(dataset.val_df)-pred_len)
        # print(batch_size)
        for idx in range(history_len, len(dataset.val_df)-pred_len, batch_size):
                print("Yes validation is running")
                past_ts_list = []
                # series_names_tokenized_iid_list = []
                # series_names_tokenized_am_list = []
                date_features_vector_list = []
                future_ts_list = []
                future_ts_list2 = []
                related_ts_list = []
                # related_series_names_tokenized_iid_list = []
                # related_series_names_tokenized_am_list = []
                col = 'Price'
                past_ts, date_features_vector, future_date_features_vector, future_ts, related_ts= dataset.get_io_i('val', col, idx, history_len, pred_len)
                #past_ts, series_names_tokenized, date_features_vector, future_date_features_vector, future_ts, related_ts, related_series_names_tokenized = dataset.get_io_i('val', col, idx, history_len, pred_len)
                # n = len(related_ts[0])
                # a = [[] for x in range(n)]
                # b = [[] for x in range(n)]
                for i in range(batch_size):
                    if idx+i>=len(dataset.val_df)-pred_len:
                        break
                    col = 'Price'
                    past_ts, date_features_vector, future_date_features_vector, future_ts, related_ts = dataset.get_io_i('val', col, idx+i, history_len, pred_len)
                    #past_ts, series_names_tokenized, date_features_vector, future_date_features_vector, future_ts, related_ts, related_series_names_tokenized = dataset.get_io_i('val', col, idx+i, history_len, pred_len)
                    past_ts_list.append(past_ts)
                    # series_names_tokenized_iid_list.append(series_names_tokenized['input_ids'])
                    # series_names_tokenized_am_list.append(series_names_tokenized['attention_mask'])
                    date_features_vector_list.append(date_features_vector)
                    future_ts_list.append(future_ts)
                    future_ts_list2.append(future_date_features_vector)
                    related_ts_list.append(related_ts)
                    # n = len(related_ts[0])
                    # for j in range(n):
                    #     a[j].append(related_series_names_tokenized[j]['input_ids'])
                    #     b[j].append(related_series_names_tokenized[j]['attention_mask'])
                #print("Length of tokenized" , len(series_names_tokenized_iid_list))
                # series_names_tokenized = {'input_ids':torch.cat(tensors=series_names_tokenized_iid_list, dim=0), 'attention_mask':torch.cat(tensors=series_names_tokenized_am_list, dim=0)}
                # rel_series_final = []
                # for i in range(n):
                #     related_series_names_tokenized = {'input_ids':torch.cat(tensors=a[i], dim=0), 'attention_mask':torch.cat(tensors=b[i], dim=0)}
                #     rel_series_final.append(related_series_names_tokenized)
                #related_series_names_tokenized = {'input_ids':torch.cat(tensors=related_series_names_tokenized_iid_list, dim=0), 'attention_mask':torch.cat(tensors=related_series_names_tokenized_am_list, dim=0)}
                past_ts = torch.cat(tensors=past_ts_list, dim=0)
                date_features_vector = torch.cat(tensors=date_features_vector_list, dim=0)
                related_ts = torch.cat(tensors=related_ts_list, dim=0)
                future_ts = torch.cat(tensors=future_ts_list, dim=0)
                future_ts2 = torch.cat(tensors=future_ts_list2, dim=0)
                preds = model(past_ts, None, date_features_vector, future_ts2, related_ts, None, None, col)
                #preds = model(past_ts, series_names_tokenized, date_features_vector, future_ts2, related_ts, rel_series_final, None, col)
                with open('output.txt', 'w') as f:
                    print(preds, file=f)

                loss = loss_function(future_ts, preds)
                total_val_loss+=loss
                total_val_steps+=1
        f.close()
        print('Val loss: ', total_val_loss/total_val_steps)
    model.train()


parser = argparse.ArgumentParser(prog = 'TSF_Trainer', description = 'Training TiDE model for TSF')

# training hyperparameters
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--lr', type=float, default=2e-4)

#parser.add_argument('--related_series_type', type=str, default='llm') # can be llm or random
parser.add_argument('--global_model', type=bool, default=True)
parser.add_argument('--global_encoder', type=bool, default=True)

parser.add_argument('--history_len', type=int, default=32)
parser.add_argument('--pred_len', type=int, default=1)
parser.add_argument('--text_dim', type=int, default=4)

if __name__ == '__main__':
    # parse args
    args = parser.parse_args()

    #seed 
    seed_everything(args.seed)

    #global variables
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    #device = torch.device('cpu')
    print('Device: ', device)
    
    # data
    dataset = BankNiftyFuturesDataset(device)

    # model specifics
    history_len = args.history_len
    pred_len = args.pred_len
    text_dim = args.text_dim
    hidden_dims = [4, 4]
    model_config = {'input_dim':history_len*(1+len(dataset.time_features_names)), 'hidden_dims':[4, 4], 'decoder_output_dim':pred_len, 'final_decoder_hidden':2}
    #model_config = {'input_dim':text_dim+history_len*(1+len(dataset.time_features_names)), 'hidden_dims':[4, 4], 'decoder_output_dim':pred_len, 'final_decoder_hidden':1, 'text_dim':text_dim}
    global_model = args.global_model
    global_encoder =args.global_encoder
    model = TideModel(model_config, pred_len, history_len, device, False, 0.0, global_model, dataset.all_series_names, global_encoder,len(dataset.all_series_names)).to(device)
    
    # setup training
    batch_size = args.batch_size
    lr = args.lr
    num_epochs = args.epochs
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # loss function
    loss_function = torch.nn.MSELoss()

    for epoch in range(0, num_epochs):
        #print(history_len)
        #print(len(dataset.train_df))
        #print(pred_len)
        #print(batch_size)
        if(epoch > 0):
            validate(model, dataset, history_len, pred_len, batch_size)
        print('Starting epoch ', epoch)
        epoch_loss = 0
        num_epoch_steps = 0
        p_bar = tqdm(total=len(dataset.train_df)-pred_len-history_len, position=0, leave=True, desc='Training for ')
        #print("Dataset training",len(dataset.train_df))
        for idx in range(history_len, len(dataset.train_df)-pred_len, batch_size):
                #print("Inside training")
                if idx>=1095 and idx<1200:
                    continue
                past_ts_list = []
                # series_names_tokenized_iid_list = []
                # series_names_tokenized_am_list = []
                date_features_vector_list = []
                future_ts_list = []
                future_ts_list2 = []
                related_ts_list = []
                # related_series_names_tokenized_iid_list = []
                # related_series_names_tokenized_am_list = []
                col = 'Price'
                past_ts, date_features_vector, future_date_features_vector, future_ts, related_ts = dataset.get_io_i('train', col, idx, history_len, pred_len)
                #past_ts, series_names_tokenized, date_features_vector, future_date_features_vector, future_ts, related_ts, related_series_names_tokenized = dataset.get_io_i('train', col, idx, history_len, pred_len)
                # n = len(related_ts[0])
                # a = [[] for x in range(n)]
                # b = [[] for x in range(n)]
                for i in range(batch_size):
                    if idx+i>=len(dataset.train_df)-pred_len:
                        break
                    col = 'Price'
                    past_ts, date_features_vector, future_date_features_vector, future_ts, related_ts= dataset.get_io_i('train', col, idx+i, history_len, pred_len)
                    #past_ts, series_names_tokenized, date_features_vector, future_date_features_vector, future_ts, related_ts, related_series_names_tokenized = dataset.get_io_i('train', col, idx+i, history_len, pred_len)
                    past_ts_list.append(past_ts)
                    # series_names_tokenized_iid_list.append(series_names_tokenized['input_ids'])
                    # series_names_tokenized_am_list.append(series_names_tokenized['attention_mask'])
                    date_features_vector_list.append(date_features_vector)
                    future_ts_list.append(future_ts)
                    future_ts_list2.append(future_date_features_vector)
                    related_ts_list.append(related_ts)
                    # n = len(related_ts[0])
                    # for j in range(n):
                    #     a[j].append(related_series_names_tokenized[j]['input_ids'])
                    #     b[j].append(related_series_names_tokenized[j]['attention_mask'])
                #print("Length of tokenized" , len(series_names_tokenized_iid_list))
                # series_names_tokenized = {'input_ids':torch.cat(tensors=series_names_tokenized_iid_list, dim=0), 'attention_mask':torch.cat(tensors=series_names_tokenized_am_list, dim=0)}
                # rel_series_final = []
                # for i in range(n):
                #     related_series_names_tokenized = {'input_ids':torch.cat(tensors=a[i], dim=0), 'attention_mask':torch.cat(tensors=b[i], dim=0)}
                #     rel_series_final.append(related_series_names_tokenized)
                #related_series_names_tokenized = {'input_ids':torch.cat(tensors=related_series_names_tokenized_iid_list, dim=0), 'attention_mask':torch.cat(tensors=related_series_names_tokenized_am_list, dim=0)}
                past_ts = torch.cat(tensors=past_ts_list, dim=0)
                date_features_vector = torch.cat(tensors=date_features_vector_list, dim=0)
                related_ts = torch.cat(tensors=related_ts_list, dim=0)
                future_ts = torch.cat(tensors=future_ts_list, dim=0)
                future_ts2 = torch.cat(tensors=future_ts_list2, dim=0)
                preds = model(past_ts, None, date_features_vector, future_ts2, related_ts, None, None, col)
                #preds = model(past_ts, series_names_tokenized, date_features_vector, future_ts2, related_ts, rel_series_final, None, col)
                print("Predictions are",preds)
                print("Future ts is", future_ts)
                loss = loss_function(future_ts, preds)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                loss_item = loss.detach().clone().item()
                epoch_loss+=loss_item
                num_epoch_steps+=1
                p_bar.update(batch_size)
                #print("Finally code is working completely")
        print('Epoch loss: ', epoch_loss/num_epoch_steps)     
        torch.save(model.state_dict(), '/mnt/nas/kunalchhabra/TIDE_Models/tsf_global_encoder_bn_'+'_epoch_'+str(epoch+1))


# if __name__ == '__main__':
#     # parse args
#     args = parser.parse_args()

#     #seed 
#     seed_everything(args.seed)

#     #global variables
#     if torch.cuda.is_available():
#         device = torch.device('cuda')
#     else:
#         device = torch.device('cpu')
#     print('Device: ', device)
    
#     # data
#     related_series_type = 'llm'
#     dataset = CPIDataset(related_series_type, device)

#     # model specifics
#     history_len = args.history_len
#     pred_len = args.pred_len
#     text_dim = args.text_dim
#     hidden_dims = [4, 4]
#     model_config = {'input_dim':text_dim+history_len*(1+len(dataset.time_features_names)), 'hidden_dims':[4, 4], 'decoder_output_dim':pred_len, 'final_decoder_hidden':1, 'text_dim':text_dim}
#     global_model = args.global_model
#     global_encoder =args.global_encoder
#     model = TideModel(model_config, pred_len, history_len, device, False, 0.0, global_model, dataset.all_series_names, global_encoder).to(device)
    
#     # setup training
#     batch_size = args.batch_size
#     lr = args.lr
#     num_epochs = args.epochs
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)

#     # loss function
#     loss_function = torch.nn.MSELoss()

#     for epoch in range(0, num_epochs):
#         validate(model, dataset, history_len, pred_len, batch_size)
#         print('Starting epoch ', epoch)
#         epoch_loss = 0
#         num_epoch_steps = 0
#         for col in dataset.all_series_names:
#             print('For col ', col)
#             epoch_col_loss = 0
#             num_epoch_col_steps = 0
#             p_bar = tqdm(total=len(dataset.train_df)-pred_len-history_len, position=0, leave=True, desc='Training for '+col)
#             for idx in range(history_len, len(dataset.train_df)-pred_len, batch_size):
#                 past_ts_list = []
#                 series_names_tokenized_iid_list = []
#                 series_names_tokenized_am_list = []
#                 date_features_vector_list = []
#                 future_ts_list = []
#                 related_ts_list = []
#                 related_series_names_tokenized_iid_list = []
#                 related_series_names_tokenized_am_list = []
#                 for i in range(batch_size):
#                     if idx+i>=len(dataset.train_df)-pred_len:
#                         break
#                     past_ts, series_names_tokenized, date_features_vector, future_ts, related_ts, related_series_names_tokenized = dataset.get_io_i('train', col, idx+i, history_len, pred_len)
#                     past_ts_list.append(past_ts)
#                     series_names_tokenized_iid_list.append(series_names_tokenized['input_ids'])
#                     series_names_tokenized_am_list.append(series_names_tokenized['attention_mask'])
#                     date_features_vector_list.append(date_features_vector)
#                     future_ts_list.append(future_ts)
#                     related_ts_list.append(related_ts)
#                     related_series_names_tokenized_iid_list.append(related_series_names_tokenized['input_ids'])
#                     related_series_names_tokenized_am_list.append(related_series_names_tokenized['attention_mask'])

#                 series_names_tokenized = {'input_ids':torch.cat(tensors=series_names_tokenized_iid_list, dim=0), 'attention_mask':torch.cat(tensors=series_names_tokenized_am_list, dim=0)}
#                 related_series_names_tokenized = {'input_ids':torch.cat(tensors=related_series_names_tokenized_iid_list, dim=0), 'attention_mask':torch.cat(tensors=related_series_names_tokenized_am_list, dim=0)}
#                 past_ts = torch.cat(tensors=past_ts_list, dim=0)
#                 date_features_vector = torch.cat(tensors=date_features_vector_list, dim=0)
#                 related_ts = torch.cat(tensors=related_ts_list, dim=0)
#                 future_ts = torch.cat(tensors=future_ts_list, dim=0)
#                 preds = model(past_ts, series_names_tokenized, date_features_vector, related_ts, related_series_names_tokenized, None, col, dataset.related[col])
#                 loss = loss_function(future_ts, preds)
#                 loss.backward()
#                 optimizer.step()
#                 optimizer.zero_grad()
#                 loss_item = loss.detach().clone().item()
#                 epoch_loss+=loss_item
#                 num_epoch_steps+=1
#                 epoch_col_loss+=loss_item
#                 num_epoch_col_steps+=1
#                 p_bar.update(batch_size)
#             print('Epoch col loss: ', epoch_col_loss/num_epoch_col_steps)
#         print('Epoch loss: ', epoch_loss/num_epoch_steps)     
    
#     torch.save(model.state_dict(), '/mnt/nas/tulip/TIDE_models/tsf_local_encoder_'+related_series_type+'_epoch_'+str(epoch+1))
