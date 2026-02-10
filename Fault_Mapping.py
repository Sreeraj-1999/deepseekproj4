# import pandas as pd

# class Faults_Mapping():
#     utility_dict = {'Pscav':{'Limits': {'L_limit': -10, 'U_limit': 10}, 'imp_feature': list(pd.read_csv('./utils/ML_model/imp_features/Pscav.csv')['Features'][:24])},
#                     'Pcomp':{'Limits': {'L_limit': -8, 'U_limit': 8}, 'imp_feature': list(pd.read_csv('./utils/ML_model/imp_features/Pcomp.csv')['Features'][:26])},
#                     'Pmax':{'Limits': {'L_limit': -8, 'U_limit': 8}, 'imp_feature': list(pd.read_csv('./utils/ML_model/imp_features/Pmax.csv')['Features'][:27])},
#                     'Texh':{'Limits': {'L_limit': -8, 'U_limit': 8}, 'imp_feature': list(pd.read_csv('./utils/ML_model/imp_features/Texh.csv')['Features'][:24])},
#                     'Ntc':{'Limits': {'L_limit': -8, 'U_limit': 8}, 'imp_feature': list(pd.read_csv('./utils/ML_model/imp_features/Ntc.csv')['Features'][:23])},
#                     'Ntc_Pscav':{'Limits': {'L_limit': -8, 'U_limit': 8}, 'imp_feature': list(pd.read_csv('./utils/ML_model/imp_features/Ntc_Pscav.csv')['Features'][:24])},
#                     'Pcomp_Pscav':{'Limits': {'L_limit': -10, 'U_limit': 10}, 'imp_feature': list(pd.read_csv('./utils/ML_model/imp_features/Pcomp_Pscav.csv')['Features'][:24])},
#                     'PR':{'Limits': {'L_limit': -15, 'U_limit': 15}, 'imp_feature': list(pd.read_csv('./utils/ML_model/imp_features/PR.csv')['Features'][:26])}}
#     def __init__(self,ml_res,matrix,Efd_features,p_value):
#         self.ml_res = ml_res #output from ml part
#         self.matrix = matrix #fault matrix path
#         self.Efd_features = Efd_features
#         self.p_value = p_value # .2  
#     def kpi_perc(self,num_efds,match_efds,d,p): #thres - 75
#         kpi_value = ((match_efds/num_efds)*(1-p)) + (d*p)
#         return round(kpi_value,2)*100    
#     def Mapping(self):    
#         real_pred_res = self.ml_res
#         delta_frame = pd.DataFrame()
#         for fh in range(len(real_pred_res)):
#             for efds2 in self.Efd_features:
#                 limits = self.utility_dict[efds2]['Limits']
#                 l_lim, u_lim = abs(limits['L_limit']), abs(limits['U_limit'])
#                 # print(fh)
#                 # print('1',real_pred_res.loc[fh,'TS_'+efds2])
#                 # print('2',real_pred_res.loc[fh,'Ref_'+efds2]*((100-l_lim)/100))
#                 # print('3',real_pred_res.loc[fh,'Ref_'+efds2]*((100+u_lim)/100))
#                 if real_pred_res.loc[fh,'TS_'+efds2]<real_pred_res.loc[fh,'Ref_'+efds2]*((100-l_lim)/100) or real_pred_res.loc[fh,'TS_'+efds2]>real_pred_res.loc[fh,'Ref_'+efds2]*((100+u_lim)/100):
#                     if real_pred_res.loc[fh,'TS_'+efds2]<real_pred_res.loc[fh,'Ref_'+efds2]*((100-l_lim)/100):
#                         delta_frame.loc[fh,efds2] = -1
#                     elif real_pred_res.loc[fh,'TS_'+efds2]>real_pred_res.loc[fh,'Ref_'+efds2]*((100+u_lim)/100):
#                         delta_frame.loc[fh,efds2] = 1
#                 else:
#                     delta_frame.loc[fh,efds2] = 0        
#         delta_frame = delta_frame.astype(int)

#         delta_frame2 = delta_frame.copy() 
#         # delta_frame2['Faults'] = pd.Series([[]]*len(delta_frame2)) #'No Faults'
#         f_mat = pd.read_excel(self.matrix).iloc[:,2:-1]
#         f_mat2 = pd.read_excel(self.matrix)
#         f_mat = f_mat.astype(int)
#         for rows in range(len(delta_frame)):
#             kpi_values_list = {}
#             for iddx in range(len(f_mat)):
#                 # print(delta_frame.loc[rows,:].values.tolist())
#                 # print(f_mat.loc[iddx,:].values.tolist())
#                 match_efds = 0
#                 if int(delta_frame.loc[rows,'Pscav']) == int(f_mat.loc[iddx,'Pscav']):
#                     match_efds+=1
#                 if int(delta_frame.loc[rows,'Pcomp']) == int(f_mat.loc[iddx,'Pcomp']):
#                     match_efds+=1  
#                 if int(delta_frame.loc[rows,'Pmax']) == int(f_mat.loc[iddx,'Pmax']):
#                     match_efds+=1
#                 if int(delta_frame.loc[rows,'Texh']) == int(f_mat.loc[iddx,'Texh']):
#                     match_efds+=1
#                 if int(delta_frame.loc[rows,'Ntc']) == int(f_mat.loc[iddx,'Ntc']):
#                     match_efds+=1
#                 if int(delta_frame.loc[rows,'Ntc_Pscav']) == int(f_mat.loc[iddx,'Ntc_Pscav']):
#                     match_efds+=1
#                 if int(delta_frame.loc[rows,'Pcomp_Pscav']) == int(f_mat.loc[iddx,'Pcomp_Pscav']):
#                     match_efds+=1
#                 if int(delta_frame.loc[rows,'PR']) == int(f_mat.loc[iddx,'PR']):
#                     match_efds+=1 
#                 dominant = f_mat2.loc[iddx,'Dominant']  
#                 d1_param = int(f_mat2.loc[iddx,dominant])
#                 d2_param = int(delta_frame2.loc[rows,dominant])
#                 if d1_param == d2_param:
#                     d_value = 1
#                 else:
#                     d_value = 0    
#                 kpi_value = self.kpi_perc(len(self.utility_dict.keys()),match_efds,d_value,self.p_value)   
#                 # kpi_values_list.append((f_mat2.loc[iddx,'Fault_id'],kpi_value))
#                 # kpi_values_list[f_mat2.loc[iddx,'Fault_id']] = kpi_value
#                 delta_frame2.loc[[rows],[f_mat2.loc[iddx,'Fault_id']]] = pd.Series(kpi_value, index=[rows])             
#                 # if delta_frame.loc[rows,:].values.tolist() == f_mat.loc[iddx,:].values.tolist():
#                 #     if delta_frame2.loc[rows,'Faults'] == 'No Faults':
#                 #         # delta_frame2.loc[rows,'Faults'] = []
#                 #         delta_frame2.loc[[rows],['Faults']] = pd.Series([[]], index=[rows])
#                 #         # print(f_mat2.loc[iddx,'Fault'])
#                 #         delta_frame2.loc[rows,'Faults'].append(f_mat2.loc[iddx,'Fault'])
#                 #     else:    
#                 #         delta_frame2.loc[rows,'Faults'].append(f_mat2.loc[iddx,'Fault'])#add mapping of faults here if matches 
#         return delta_frame2, list(f_mat2['Fault_id'])


import pandas as pd
from concurrent import futures
import yaml

# with open('cbm_yaml.yml','r') as file:
with open('cbm.yaml','r') as file:
    utility_dict = yaml.safe_load(file)

class Faults_Mapping():
    global utility_dict
    def __init__(self,ml_res,matrix,Efd_features,p_value):
        self.ml_res = ml_res #output from ml part
        self.matrix = matrix #fault matrix path
        self.Efd_features = Efd_features
        self.p_value = p_value # .2  
    def kpi_perc(self,num_efds,match_efds,d,p): #thres - 75
        kpi_value = ((match_efds/num_efds)*(1-p)) + (d*p)
        return round(kpi_value,2)*100  
    def Mapping(self):   
        real_pred_res = self.ml_res
        delta_frame = pd.DataFrame()
        for fh in range(len(real_pred_res)):
            for efds2 in self.Efd_features:
                limits = utility_dict['imp_features'][efds2]['Limits']
                l_lim, u_lim = abs(limits['L_limit']), abs(limits['U_limit'])
                if real_pred_res.loc[fh,'Ref_'+efds2] == 'No Values' or real_pred_res.loc[fh,utility_dict['engine_load']]<30:
                    delta_frame.loc[fh,efds2] = 555 #this integer value is denote that there is no proper matching engine load within load limit
                else:    
                    if real_pred_res.loc[fh,'TS_'+efds2]<float(real_pred_res.loc[fh,'Ref_'+efds2])*((100-l_lim)/100) or real_pred_res.loc[fh,'TS_'+efds2]>float(real_pred_res.loc[fh,'Ref_'+efds2])*((100+u_lim)/100):
                        if real_pred_res.loc[fh,'TS_'+efds2]<float(real_pred_res.loc[fh,'Ref_'+efds2])*((100-l_lim)/100):
                            delta_frame.loc[fh,efds2] = -1
                        elif real_pred_res.loc[fh,'TS_'+efds2]>float(real_pred_res.loc[fh,'Ref_'+efds2])*((100+u_lim)/100):
                            delta_frame.loc[fh,efds2] = 1
                    else:
                        delta_frame.loc[fh,efds2] = 0        
        delta_frame = delta_frame.astype(int)

        delta_frame2 = delta_frame.copy() 
        # delta_frame2['Faults'] = pd.Series([[]]*len(delta_frame2)) #'No Faults' 
        f_mat = pd.read_excel(self.matrix).iloc[:,2:-1]
        f_mat2 = pd.read_excel(self.matrix)
        f_mat = f_mat.astype(int)
        for rows in range(len(delta_frame)):
            # kpi_values_list = {} 
            for iddx in range(len(f_mat)):
                if 555 in delta_frame.loc[rows].to_list():
                    delta_frame2.loc[[rows],[f_mat2.loc[iddx,'Fault_id']]] = 'No Values'
                else:    
                    # print(delta_frame.loc[rows,:].values.tolist())
                    # print(f_mat.loc[iddx,:].values.tolist())
                    match_efds = 0
                    if int(delta_frame.loc[rows,list(utility_dict['imp_features'].keys())[0]]) == int(f_mat.loc[iddx,list(utility_dict['imp_features'].keys())[0]]):
                        match_efds+=1
                    if int(delta_frame.loc[rows,list(utility_dict['imp_features'].keys())[1]]) == int(f_mat.loc[iddx,list(utility_dict['imp_features'].keys())[1]]):
                        match_efds+=1  
                    if int(delta_frame.loc[rows,list(utility_dict['imp_features'].keys())[2]]) == int(f_mat.loc[iddx,list(utility_dict['imp_features'].keys())[2]]):
                        match_efds+=1
                    if int(delta_frame.loc[rows,list(utility_dict['imp_features'].keys())[3]]) == int(f_mat.loc[iddx,list(utility_dict['imp_features'].keys())[3]]):
                        match_efds+=1
                    if int(delta_frame.loc[rows,list(utility_dict['imp_features'].keys())[4]]) == int(f_mat.loc[iddx,list(utility_dict['imp_features'].keys())[4]]):
                        match_efds+=1
                    if int(delta_frame.loc[rows,list(utility_dict['imp_features'].keys())[5]]) == int(f_mat.loc[iddx,list(utility_dict['imp_features'].keys())[5]]):
                        match_efds+=1
                    if int(delta_frame.loc[rows,list(utility_dict['imp_features'].keys())[6]]) == int(f_mat.loc[iddx,list(utility_dict['imp_features'].keys())[6]]):
                        match_efds+=1
                    if int(delta_frame.loc[rows,list(utility_dict['imp_features'].keys())[7]]) == int(f_mat.loc[iddx,list(utility_dict['imp_features'].keys())[7]]):
                        match_efds+=1 
                    dominant = f_mat2.loc[iddx,'Dominant']  
                    d1_param = int(f_mat2.loc[iddx,dominant])
                    d2_param = int(delta_frame2.loc[rows,dominant])
                    if d1_param == d2_param:
                        d_value = 1
                    else:
                        d_value = 0    
                    kpi_value = self.kpi_perc(len(utility_dict['imp_features'].keys()),match_efds,d_value,self.p_value)   
                    # kpi_values_list.append((f_mat2.loc[iddx,'Fault_id'],kpi_value))
                    # kpi_values_list[f_mat2.loc[iddx,'Fault_id']] = kpi_value
                    delta_frame2.loc[[rows],[f_mat2.loc[iddx,'Fault_id']]] = pd.Series(kpi_value, index=[rows])             
                    if delta_frame.loc[rows,:].values.tolist() == f_mat.loc[iddx,:].values.tolist():
                        if delta_frame2.loc[rows,'Faults'] == 'No Faults':
                            # delta_frame2.loc[rows,'Faults'] = []
                            delta_frame2.loc[[rows],['Faults']] = pd.Series([[]], index=[rows])
                            # print(f_mat2.loc[iddx,'Fault'])
                            delta_frame2.loc[rows,'Faults'].append(f_mat2.loc[iddx,'Fault'])
                        else:    
                            delta_frame2.loc[rows,'Faults'].append(f_mat2.loc[iddx,'Fault'])#add mapping of faults here if matches 
        return delta_frame2, list(f_mat2['Fault_id'])
        # return a,b
