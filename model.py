from modules import *

class TideModel(torch.nn.Module):
  """Main class for multi-scale DNN model."""

  def __init__(
      self,
      model_config,
      pred_len,
      history_len,
      device,
      layer_norm=True,
      dropout_rate=0.0,
      global_model=True,
      cols=None,
      global_encoder=True,
      num_series = 2
  ):
    """Tide model.

    Args:
      model_config: configurations specific to the model.
      pred_len: prediction horizon length.
      history_len: lookback length.
      layer_norm: use layer norm or not.
      dropout_rate: level of dropout.
    """
    super().__init__()
    self.model_config = model_config # comment - add input_dim

    self.text_module = TextModel(device=device, text_dim=model_config['text_dim'])
    self.loc_module = LocationModel(device=device)

    self.pred_len = pred_len
    self.history_len = history_len
    self.global_model = global_model
    self.global_encoder = global_encoder
    self.num_series = num_series
    if self.global_model or self.global_encoder:
      self.encoder = make_dnn_residual(
          model_config['input_dim'],
          model_config['hidden_dims'],
          device=device,
          layer_norm=layer_norm,
          dropout_rate=dropout_rate,
      )
    if global_model:
      self.encoding_combiner = torch.nn.Sequential(
          torch.nn.Linear(model_config['hidden_dims'][-1]*(num_series + 1), model_config['hidden_dims'][-1], device=device),
          torch.nn.ReLU()
      )
      self.decoder = make_dnn_residual(
          model_config['hidden_dims'][-1],
          model_config['hidden_dims'][:-1]
          + [
              model_config['decoder_output_dim'] * self.pred_len,
          ],
          device=device,
          layer_norm=layer_norm,
          dropout_rate=dropout_rate,
      )
      self.linear = torch.nn.Linear(
          self.history_len,
          self.pred_len,
          device=device,
      )
      self.linear2 = torch.nn.Linear(
        6,self.pred_len,device = device
      )
      self.final_decoder = MLPResidual(
          input_dim= 2*self.pred_len,
          hidden_dim=model_config['final_decoder_hidden'],
          output_dim=1,
          device=device,
          layer_norm=layer_norm,
          dropout_rate=dropout_rate,
      )
    else:
      if ~self.global_encoder:
        self.encoders = {}
      self.encoding_combiners = {}
      self.decoders = {}
      self.linears = {}
      self.final_decoders = {}
      for col in cols:
        if ~self.global_encoder:
          self.encoders[col] = make_dnn_residual(
              model_config['input_dim'],
              model_config['hidden_dims'],
              device=device,
              layer_norm=layer_norm,
              dropout_rate=dropout_rate,
          )
        self.encoding_combiners[col] = torch.nn.Sequential(
            torch.nn.Linear(model_config['hidden_dims'][-1]*2, model_config['hidden_dims'][-1], device=device),
            torch.nn.ReLU(),
        )
        self.decoders[col] = make_dnn_residual(
            model_config['hidden_dims'][-1],
            model_config['hidden_dims'][:-1]
            + [
                model_config['decoder_output_dim'] * self.pred_len,
            ],
            device=device,
            layer_norm=layer_norm,
            dropout_rate=dropout_rate,
        )
        self.linears[col] = torch.nn.Linear(
            self.history_len,
            self.pred_len,
            device=device,
        )
        self.final_decoders[col] = MLPResidual(
            input_dim=1,
            hidden_dim=model_config['final_decoder_hidden'],
            output_dim=1,
            device=device,
            layer_norm=layer_norm,
            dropout_rate=dropout_rate,
        )



  def forward(self, past_ts, series_names, dates=None, future_dates = None, related_ts=None, related_series_names=None, series_locs=None, col=None, rel_col=None):
    """Call function that takes in a batch of training data and features."""
    bsize = past_ts.shape[0] # shape is batch x history_len
    # shape of series_loc is batch x 2
    # shape of series_names is batch
    # print('where are you??')
    if self.global_model:
      residual_out = self.linear(past_ts)
    else:
      # print('dc')
      # print(past_ts.device)
      # print(self.linears[col].device)
      residual_out = self.linears[col](past_ts)
    # print('residual_out shape: ', residual_out.shape)
    enc_text = self.text_module(series_names)
    related_enc_text_list = []
    if related_ts is not None:
      for series in related_series_names:
        related_enc_text = self.text_module(series)
        #print(type(related_enc_text))
        related_enc_text_list.append(related_enc_text)
      #print(len(related_enc_text_list))
      #print(len(related_ts))
      
    # print('enc_text shape: ', enc_text.shape)
    # print('past ts shape: ', past_ts.shape)
    if series_locs is not None:
      raise NotImplementedError('Location module has not been implemented yet')
      # loc_bert_emb, bounds = series_locs
      # enc_loc = self.loc_module(loc_bert_emb, bounds)
      # encoder_input = torch.cat([past_ts, enc_text, enc_loc], dim=1)
      # if related_ts is not None:
      #   encoder_input_related = torch.cat([related_ts, related_enc_text, enc_loc], dim=1)
    else:
      if dates is not None:
        #print('previous dates shape', dates.shape)
        dates = dates.reshape(bsize, -1)
        #print('dates shape: ', dates.shape)
        #print('past ts shape: ', past_ts.shape)
        #print('enc_text shape: ', enc_text.shape)
        encoder_input = torch.cat([past_ts, dates, enc_text], dim=1) # check dim
        encoder_input_related_list = []
        if related_ts is not None:
          #print('previous related shape', related_ts.shape)
          related_ts = related_ts.resize_(bsize, self.history_len, self.num_series-1)
          #print('current related shape', related_ts.shape)
          #print(related_ts.shape)
          for i in range(0,self.num_series-1):
            #print(related_ts[:,:,i].shape)
            #print(dates.shape)
            #print(related_enc_text_list[i].shape)
            encoder_input_related = torch.cat([related_ts[:,:,i], dates, related_enc_text_list[i]], dim = 1).to(torch.float32)
            encoder_input_related_list.append(encoder_input_related)
      else:
        encoder_input = torch.cat([past_ts, enc_text], dim=1) # check dim
        if related_ts is not None:
          encoder_input_related = torch.cat([related_ts, related_enc_text], dim=1)

    # (batch , fdim x H) should be the shape of all inputs to be concatenated like past_ts, H is pred_len
    # print('encoder_input shape: ', encoder_input.shape)
    if self.global_encoder or self.global_model :
      #print("Encoder dtype",encoder_input.dtype)
      encoding = self.encoder(encoder_input)
    else:
      encoding = self.encoders[col](encoder_input)
    # print('encoding shape: ', encoding.shape)
    if related_ts is not None:
      # uncomment for global encoder
      # related_encoding = self.encoder(encoder_input_related)
      # uncomment end
      if self.global_model:
        #print("Reached here", encoding.shape)
        #print("Related dtype",encoder_input_related_list[0].dtype)
        related_encoding = self.encoder(encoder_input_related_list[0])
        #print("Reached here too", related_encoding.shape)
        related_encodings = torch.cat([encoding,related_encoding],dim = -1)
        for i in range(0,self.num_series-1):
          related_encoding = self.encoder(encoder_input_related_list[i])
          related_encodings = torch.cat([related_encodings,related_encoding], dim = -1)
        # if future_dates is not None:
        #   related_encoding = self.encoder(future_dates)
        #   related_encodings = torch.cat([related_encodings,related_encoding], dim = -1)
        encoding = self.encoding_combiner(related_encodings)
      else:
        if self.global_encoder:
          related_encoding = self.encoder(encoder_input_related)
        else:
          related_encoding = self.encoders[rel_col](encoder_input_related)
        encoding = self.encoding_combiners[col](torch.cat([encoding, related_encoding], dim=-1))
    if self.global_model:
      decoder_out = self.decoder(encoding)
    else:
      decoder_out = self.decoders[col](encoding)
    # print('decoder_out shape: ', decoder_out.shape)
    decoder_out = decoder_out.reshape(bsize, self.pred_len, -1)  # batch x d x pred_len
    final_in = decoder_out
    if future_dates is not None:
      #print(final_in.shape)
      #print(future_dates.shape)
      reduced_future_dates = self.linear2(future_dates)
      final_in = torch.cat([final_in,reduced_future_dates],dim = -1)
    if self.global_model:
      out = self.final_decoder(final_in)  # B x pred_len x 1
    else:
      out = self.final_decoders[col](final_in)  # B x pred_len x 1
    #print('out shape: ', out.shape)
    out = out.squeeze(dim=-1) # check dim
    #print('out shape next: ', out.shape)
    #print('residual_out_shape',residual_out.shape)
    out += residual_out
    return out
