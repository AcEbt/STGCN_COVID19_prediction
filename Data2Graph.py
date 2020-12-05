import pandas as pd
import numpy as np
import calc_distance
import dgl
from scipy import sparse
import torch


def load_data(data_filename, auxiliary_filename,active_thresh=1000, r=1): 

  raw_data = pd.read_csv(data_filename)
  state_county = np.array(raw_data[['state_name','county']])
  all_counties = np.unique(state_county[:,0]+'_'+state_county[:,1])
  N = all_counties.size

  # Number of time stamps 
  dates = np.unique(np.array(raw_data['date']))
  T = dates.size

  uszips = pd.read_csv(auxiliary_filename)
  uszips_state_county = np.array(uszips[['state_name','county']])
  uszips_all_counties = np.unique(uszips_state_county[:,0]+'_'+np.array(uszips_state_county)[:,1])

  all_popn = uszips.groupby(['state_name','county'])['population'].sum().reset_index()
  all_density = uszips.groupby(['state_name','county'])['density'].sum().reset_index()
  all_long = uszips.groupby(['state_name','county'])['lng'].mean().reset_index()
  all_lat = uszips.groupby(['state_name','county'])['lat'].mean().reset_index()


  inter = raw_data.merge(all_popn)
  raw_data_w_popn = inter.merge(all_density)
  raw_data_w_popn = raw_data_w_popn.merge(all_long)
  raw_data_w_popn = raw_data_w_popn.merge(all_lat)

  
  # Number of counties/ graph nodes 

  state_county_after_merge = np.array(raw_data_w_popn[['state_name','county']])
  all_counties_after_merge = np.unique(state_county_after_merge[:,0]+'_'+state_county_after_merge[:,1])
  N_after_merge = all_counties_after_merge.size

  # Number of time stamps 
  dates_after_merge = np.unique(np.array(raw_data_w_popn['date']))
  T_after_merge = dates_after_merge.size
  

  i = 0
  for date_stamp in raw_data['date'].unique():
      raw_data_w_popn.loc[raw_data_w_popn['date']==date_stamp,'date'] = i
      i = i + 1
  raw_data_w_popn.rename(columns=({'date': 'time_index'}),inplace=True)
  
  data_for_sum = raw_data_w_popn.loc[:, ~raw_data_w_popn.columns.isin(['density','county'])]
  data_summed=data_for_sum.groupby(['state_name','time_index']).sum().reset_index()

  data_for_mean = raw_data_w_popn[['state_name','time_index','density']]
  data_mean=data_for_mean.groupby(['state_name','time_index']).mean().reset_index()
  by_state=data_summed.merge(data_mean)

  
  states_after_merge=np.unique(np.array(by_state['state_name']))
  N_after_merge=states_after_merge.size
  
  dates_after_merge = np.unique(np.array(by_state['time_index']))
  T_after_merge = dates_after_merge.size

  
  
  by_state_np=np.array(by_state)

  win_len=5
  
  
  for n in range(N_after_merge):
    by_state_np[n*T_after_merge:(n+1)*T_after_merge,4] = smooth1d(by_state_np[n*T_after_merge:(n+1)*T_after_merge,4]\
                                                                      , win_len)
    by_state_np[n*T_after_merge:(n+1)*T_after_merge,5] = smooth1d(by_state_np[n*T_after_merge:(n+1)*T_after_merge,5]\
                                                                      , win_len)
    
  info_for_edge=by_state[['lat','lng','population']]
  
  popn=np.array(by_state['population']).astype('double')

  by_state_no_pop=by_state.drop(['population'], axis=1)
  
  
  
  dates=np.array(by_state)[:,1]
  active=np.array(by_state)[:,2]
  confirmed=np.array(by_state)[:,3]
  other_feat = np.array(by_state_no_pop)[:,2:] # includes the active, confirmed # cases 
  

  active_cases = torch.from_numpy(np.reshape(active,(N_after_merge, T_after_merge),order='C').astype('float64'))
  confirmed_cases = torch.from_numpy(np.reshape(confirmed,(N_after_merge, T_after_merge),order='C').astype('float64'))


  # Reshape tensor into 3D array
  feat_tensor = np.reshape(np.array(other_feat),(N_after_merge,T_after_merge,other_feat.shape[1]),order='C')
  feat_tensor = torch.from_numpy(feat_tensor.astype('float64')) 
  print("Feature tensor is of size ", feat_tensor.shape)

  pop_data=torch.tensor(popn).view(N_after_merge,T_after_merge)

  info_for_edge= np.reshape(np.array(info_for_edge),(N_after_merge,T_after_merge,3))
  lat_long_pop = []

  for ii in range(N_after_merge):
      lat_long_pop.append(info_for_edge[ii][0])


  W_mat = np.zeros((N_after_merge,N_after_merge))

  for ii in range(N_after_merge):
    lat1,lon1, pop1 = lat_long_pop[ii]
    for jj in range(N_after_merge):
      lat2,lon2, pop2 = lat_long_pop[jj]
      W_mat[ii,jj] = calc_distance.gravity_law_commute_dist(lat1, lon1, pop1, lat2, lon2, pop2, r)

#  popn_density = np.reshape(np.array(by_state['density']),(N_after_merge,T_after_merge))
#  
#  W_mat = np.zeros((N_after_merge,N_after_merge))
#
#  for ii in range(N_after_merge):
#    pop_den1 = popn_density[ii,0]
#    for jj in range(N_after_merge):
#      pop_den2 = popn_density[jj,0]
#      W_mat[ii,jj] = calc_distance.gravity_law_commute_dist(pop_den1, pop_den2)

  
      
      
#  W_max = np.amax(W_mat)
#
#  W_norm = W_mat/W_max
#  W_norm_thresh = (W_norm > 1e-3)*W_norm
#  W_sparse = sparse.coo_matrix(W_norm_thresh)
  
  W_sparse=sparse.coo_matrix(W_mat)
  
  values = W_sparse.data
  indices = np.vstack((W_sparse.row, W_sparse.col))

  i = torch.LongTensor(indices)
  v = torch.FloatTensor(values)
  shape = W_sparse.shape

  Adj=torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()

  row_sum=torch.sum(Adj,dim=1)
  row_sum[row_sum==0]=1
  
  invD_sqrt=torch.diag(torch.sqrt(row_sum.pow_(-1)))
  Adj_norm=torch.mm(invD_sqrt,(torch.mm(Adj,invD_sqrt)))
  Adj_norm=(Adj_norm-torch.diag(torch.diag(Adj_norm)))+torch.eye(Adj_norm.shape[0])
  
  
  # W_sparse contains the adjacency matrix, which is good enough 
  # for the GCN model 
  
  
#  selected_counties = torch.where(torch.max(active_cases,dim=1)[0]>active_thresh)[0]
#  sel=raw_data_w_popn.iloc[selected_counties*T_after_merge][['state','county']]
#  
  selected_states = torch.where(torch.max(active_cases,dim=1)[0]>active_thresh)[0]
  sel=by_state.iloc[selected_states*T_after_merge][['state_name']]
  
  
  feat_tensor=feat_tensor[selected_states,:,:]
  Adj_norm=Adj_norm[selected_states,:][:,selected_states]
  confirmed_cases=confirmed_cases[selected_states,:]
  active_cases = active_cases[selected_states,:]
  pop_data=pop_data[selected_states,:]

#  print("Using identity matrix as adjacency matrix")
#  Adj_norm = torch.eye(Adj_norm.shape[0])

  g = dgl.from_scipy(sparse.csr_matrix(Adj_norm.data.numpy()))
  
  
  return feat_tensor, Adj_norm, active_cases, confirmed_cases,pop_data, sel,g


def smooth1d(y, smooth):
  """
  Smooth data using hanning filter
  :param y: input data to be smoothed
  :type y: numpy.array or list
  :param smooth: number of window length for smoothing
  :type smooth: int
  :return: smoothed data
  :rtype: numpy.array
  """
  if isinstance(y, list):
      y = np.array(y)
  if smooth >= len(y):
      smooth = 1
  y_pad = np.pad(y, (smooth//2, smooth-1-smooth//2), mode='edge')
  y_smooth = np.convolve(y_pad, np.ones((smooth,)) / smooth, mode='valid')
  return y_smooth
