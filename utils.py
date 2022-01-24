import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import allensdk.brain_observatory.stimulus_info as stim_info
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import numpy as np
import multiprocess
import itertools as it
import pickle
from lmfit import minimize, Parameters
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, levene
from lmfit.models import GaussianModel

np.seterr(all='raise')


boc = BrainObservatoryCache(manifest_file='boc/manifest.json')

def gather_single_cell_data(data_dir):
	regions = ['VISl', 'VISp', 'VISpm', 'VISal', 'VISrl', 'VISam']
	lines = ['Cux2-CreERT2', 'Rbp4-Cre_KL100', 'Rorb-IRES2-Cre'] 


	cor_keys = ['region', 'cre-line', 'cell_id', 'run_rho', 'gratings_p', 'run_rho_p', 'pvalue', 'stimulus', 'condition']
	cor_data = pd.DataFrame(columns = cor_keys)


	mod_keys = ['region', 'cre-line', 'cell_id', 'model_rho', 'best_model', 'mod_rho_p', 'stimulus', 'condition']
	mod_data = pd.DataFrame(columns = mod_keys)

	pup_keys = ['region', 'cre-line', 'cell_id', 'pup_rho', 'pup_rho_p', 'pup_tuned_p', 'stimulus', 'condition']
	pup_data = pd.DataFrame(columns = pup_keys)

	for region in regions:
	    for line in lines:
	        
	        output = open(data_dir +  region + line + 'natural_run_mod.pkl', 'rb')
	        temp_nat = pd.read_pickle(output, compression = None)    
	        temp_nat['condition'] = ['running_analysis']*len(temp_nat)
	        
	        output = open(data_dir +  region + line + 'synthetic_run_mod.pkl', 'rb')
	        temp_art = pd.read_pickle(output, compression = None)

	        temp_art['condition'] = ['running_analysis']*len(temp_art)
	        
	       
	        output = open(data_dir +  region + line + 'spontaneous_run_mod.pkl', 'rb')
	        temp_spont = pd.read_pickle(output, compression = None)    
	        temp_spont['condition'] = ['running_analysis']*len(temp_spont)
	        
	        output = open(data_dir +  region + line + 'stimulus_run_mod.pkl', 'rb')
	        temp_stim = pd.read_pickle(output, compression = None)

	        temp_stim['condition'] = ['running_analysis']*len(temp_stim)
	        cor_data = pd.concat((cor_data, temp_spont[cor_keys], temp_stim[cor_keys], temp_nat[cor_keys], temp_art[cor_keys]), sort = False)

	        
	        output = open(data_dir +  region + line + '_stimulus_run_mod_new.pkl', 'rb')   
	        
	        temp_stim2 = pd.read_pickle(output, compression = None)
	        temp_stim2 = temp_stim2.rename(columns = {'run_rho': 'model_rho', 'pvalue': 'mod_rho_p'})
	        temp_stim2['condition'] = ['model_analysis']*len(temp_stim2)
	        
	        mod_data = pd.concat((mod_data, temp_stim2[mod_keys]), sort = False)
	        
	        output = open(data_dir + region + line + 'stimulus_pup_no_mod.pkl', 'rb')
	        
	        pup_temp = pd.read_pickle(output, compression = None)
	        pup_temp = pup_temp.rename(columns = {'run_rho': 'pup_rho', 'run_rho_p': 'pup_rho_p', 'pvalue': 'pup_tuned_p'})
	        
	        pup_temp['condition'] = ['pupil_analysis']*len(pup_temp)
	        pup_data = pd.concat((pup_data, pup_temp), sort = False)

	      
	f = open(data_dir + 'drifting_gratings_corr.pkl', 'rb')

	test = pd.read_pickle(f, compression = None)
	test['condition'] = ['gratings']*len(test)
	gratings_keys = ['region', 'cell_id', 'gratings_p', 'gratings_rho', 'stimulus']
	test = test.rename(columns = {'corrcoef' : 'gratings_rho', 'corrp': 'gratings_p', 'stim_cond' : 'stimulus'} )

	all_data = pd.concat((cor_data, mod_data, pup_data, test[gratings_keys]), sort = False)

	cells = boc.get_cell_specimens()
	cells = pd.DataFrame.from_records(cells)

	tuning_table = cells[['cell_specimen_id', 'p_dg', 'osi_dg', 'dsi_dg']]
	tuning_table = tuning_table.rename(columns = {'cell_specimen_id': 'cell_id'})

	all_data = all_data.merge(tuning_table, how = 'left', on ='cell_id')

	all_data['tuned'] = all_data['osi_dg'] > .6
	all_data['responsive'] = all_data['p_dg'] < 0.05


def bandPassParams(init_pars, vals, x):
    init_pars['center'].min = x[0] + 2
    init_pars['center'].max = x[-1] - 2 

    return init_pars
    
def increasingParams(init_pars, vals, x):
    init_pars['center'].min = x[-1] + 5
 
    return init_pars
    
def decreasingParams(init_pars, vals, x):
    init_pars['center'].max = x[0] - 1
   
    return init_pars


def combine_rs_data(arranged_data, data_set):
    neural_responses = {}

    for key in arranged_data.keys():

        cell_ids = data_set[key].get_cell_specimen_ids()

        for cell_id in cell_ids:
            #neural_responses[cell_id] = {'natural_dff': [], 'natural_rs': [], 'synthetic_dff': [], 
            #                             'synthetic_rs': [], 'natural_pa': [], 'synthetic_pa': [], 
            #                            'spontaneous_pa': [], 'spontaneous_rs': []}
            
            neural_responses[cell_id] = {'stimulus_dff': [], 'stimulus_rs': [],  
                                        'spontaneous_dff': [], 'spontaneous_rs': []}#,  'spontaneous_deconvdff':[]} 
                                        

    for key in arranged_data.keys():

        cell_ids = data_set[key].get_cell_specimen_ids()

        for i, cell_id in enumerate(cell_ids):
            for val in arranged_data[key]['stimulus_dff'][i]:
                neural_responses[cell_id]['stimulus_dff'].append(val)

            for val in arranged_data[key]['spontaneous_dff'][i]:
                neural_responses[cell_id]['spontaneous_dff'].append(val)

            for val in arranged_data[key]['stimulus_rs']:
                neural_responses[cell_id]['stimulus_rs'].append(val)

            for val in arranged_data[key]['spontaneous_rs']:
                neural_responses[cell_id]['spontaneous_rs'].append(val)

            #for val in arranged_data[key]['spontaneous_pa']:
            #    neural_responses[cell_id]['spontaneous_pa'].append(val)

            #for val in arranged_data[key]['stimulus_pa']:
            #    neural_responses[cell_id]['stimulus_pa'].append(val)
            
            #for val in arranged_data[key]['spontaneous_deconvdff']:
            #    neural_responses[cell_id]['spontaneous_deconvdff'].append(val)
             
                
            #data = np.array(neural_responses[cell_id]['spontaneous_dff']) + 0.01
            #data[np.isnan(data)] = 0.01
            #try:
            #    c, s = oasisAR2(data, penalty=1)
           #     s = s > 0.05
            #    
            #except:
           #     print(min(data), max(data), key)
           #     s = np.zeros_like(data)*np.nan
                
           # neural_responses[cell_id]['spontaneous_deconvdff'] = s 
            
    return neural_responses

    from lmfit.models import GaussianModel
from scipy.stats import ranksums

def rs_models_stats(stim_results, stim_key, region, cre_line, num_cv_splits = 2):
    all_data = pd.DataFrame(columns = ['region', 'cre-line', 'cell_id', 'run_rho', 'gratings_p',
                                       'run_rho_p', 'best_model', 'model_p1', 'model_p2', 'pvalue', 'stimulus',
                                       'dg_si', 'pref_dir', 'pref_tf', 'osi', 'nat_scenes_resp', 
                                       'fav_nat_scene', 'p_nat_scenes'])
    results = stim_results[stim_key] 

    cells = boc.get_cell_specimens()
    cells = pd.DataFrame.from_records(cells)
    
    cell_list = list(results.keys())
    
    for cell_id in cell_list:
        #cell_id = np.random.choice(cell_list, 1)[0].astype("int")

        train_list, test_list, x, y, std_x, rho, p, pvalue = results[cell_id]


        band_list, inc_list, dec_list = [0], [0], [0]
        for i in range(num_cv_splits):
            try:
                band_res, inc_res, dec_res = gaus_model_comparison(train_list[i], test_list[i], verbose = False)
                band_list.append(band_res)
                inc_list.append(inc_res)
                dec_list.append(dec_res)
            except:
                pass

        try:
            f, model_p1 = ranksums(band_list, inc_list)
            f, model_p2 = ranksums(band_list, dec_list)
            f, model_p3 = ranksums(inc_list, dec_list)

            
        except:
            f, model_p1 = np.nan, np.nan
            f, model_p2 = np.nan, np.nan
            f, model_p3 = np.nan, np.nan
            
        all_p = True
        if (model_p1 < 0.05) & (model_p2 < 0.05):
            mod_ind = np.argmin([np.mean(band_list), np.mean(inc_list), np.mean(dec_list)])
        elif(model_p3 < 0.05):
            mod_ind = np.argmin([np.inf, np.mean(inc_list), np.mean(dec_list)])
        else:
            mod_ind = np.nan
            all_p = False
        try:
            p_dg = cells[cells['cell_specimen_id'] == cell_id]['p_dg'].tolist()[0]
            dgsi = cells[cells['cell_specimen_id'] == cell_id]['g_dsi_dg'].tolist()[0]
            pref_dir = cells[cells['cell_specimen_id'] == cell_id]['pref_dir_dg'].tolist()[0]
            pref_tf = cells[cells['cell_specimen_id'] == cell_id]['pref_tf_dg'].tolist()[0]
            osi = cells[cells['cell_specimen_id'] == cell_id]['g_osi_sg'].tolist()[0]
            fav_nat_scene = cells[cells['cell_specimen_id'] == cell_id]['pref_image_ns'].tolist()[0]
            nat_scenes_resp = cells[cells['cell_specimen_id'] == cell_id]['image_sel_ns'].tolist()[0]
            p_nat_scenes = cells[cells['cell_specimen_id'] == cell_id]['p_ns'].tolist()[0]
                                    
        except:
            print("oops something went wrong with getting the driftin gratings p value")
            p_dg = np.nan
            dgsi = np.nan
            pref_dir = np.nan
            pref_tf = np.nan
            osi = np.nan
            fav_nat_scene = np.nan
            nat_scenes_resp = np.nan
            p_nat_scenes = np.nan
                                        
                                    
            
        d = {'region': [region], 'cell_id': [cell_id], 'cre-line': [cre_line], 
             'run_rho': [rho],'run_rho_p': [p], 'best_model': [mod_ind], 'stimulus': [stim_key],
             'model_p': [all_p], 'pvalue': [pvalue], 'gratings_p': [p_dg], 'dg_si': [dgsi],
             'pref_dir':[pref_dir], 'pref_tf': [pref_tf], 'osi': [osi], 'fav_nat_scene': [fav_nat_scene], 
             'nat_scenes_resp': [nat_scenes_resp], 'p_nat_scenes': [p_nat_scenes]}

        temp = pd.DataFrame.from_dict(d)

        all_data = pd.concat((all_data, temp))
        
    return all_data

def binned_tuning_curve_(x, y, n_bins, limit = [0, 30]):
    #bins = stats.mstats.mquantiles(x, np.linspace(0, 1, n_bins), limit = (limit[0], limit[1]))
    bins = np.linspace(*limit, n_bins)
    
    inds = np.digitize(x, bins, right = True)
    
    
    bin_means = []
    bin_std = []    
    bin_x = []
    
    for i in range(1, n_bins):
        if len(y[inds == i] > 0):
            bin_means.append(y[inds == i].mean())
            bin_std.append(y[inds == i].std() / np.sqrt(len(y[inds == i])))
            bin_x.append(bins[i])
            

    shuf_x = np.random.permutation(x)
    #shuf_bins = stats.mstats.mquantiles(x, np.linspace(0, 1, n_bins), limit = (limit[0], limit[1]))
    shuf_inds = np.digitize(shuf_x, bins, right = True)
    
    #shuf_y = np.array([y[shuf_inds == i].mean() for i in range(1, len(bins))])
    shuf_y = []
    for i in range(1, n_bins):
        if np.isfinite(np.nanmean(y[shuf_inds == i])):
            shuf_y.append(np.nanmean(y[shuf_inds == i]))


    return np.array(bin_means), np.array(bin_std), np.array(bin_x), np.array(shuf_y) 


from sklearn.model_selection import train_test_split

def make_tuning_curves3(neural_data, data_set,  PUPIL_LIMIT = 20000, num_cv_splits = 2):
    '''
    super simplified tuning curve function, optimized for cross validation
    '''
    
    stim_results = {}
    
        
    for stim_key in ['spontaneous', 'stimulus']:
        neural_responses = {}
        results = {}
        print(stim_key)
        
        for cell_id in neural_data.keys():

            run_speed = np.array(neural_data[cell_id][stim_key + '_rs']).flatten()
            temp = np.array(neural_data[cell_id][stim_key + '_dff'])  
            

            
            
            if max(run_speed) < 15:
                pass
            elif sum(run_speed <= .5) > 4*sum(run_speed >.5):
                pass
            else:
            
            
                train_list = []
                test_list = []

                for i in range(num_cv_splits):


                    cv_len = int(np.floor(len(run_speed)/4))
                    CV_ind = np.random.randint(0, len(run_speed) - cv_len)

                    train_inds = np.ones([len(run_speed)]).astype('bool')
                    train_inds[CV_ind:CV_ind + cv_len] = 0
                    test_inds = np.zeros([len(run_speed)]).astype('bool')
                    test_inds[CV_ind:CV_ind + cv_len] = 1

                    X_train, X_test, y_train, y_test = run_speed[train_inds], run_speed[test_inds], temp[train_inds], temp[test_inds]

                    y_train, _, X_train, _  = binned_tuning_curve_(X_train, y_train, 10)
                    y_test, _, X_test, _ = binned_tuning_curve_(X_test, y_test, 10)

                    train_list.append((X_train, y_train))
                    test_list.append((X_test, y_test))
            

                y, std_y, x, shuf_y = binned_tuning_curve_(run_speed, temp, 10)

                rho, p = spearmanr(y, x)
                stat, pvalue = levene(y, shuf_y)

                results[cell_id]  = train_list, test_list, x, y, std_y, rho, p, pvalue

        stim_results[stim_key] = results


    return stim_results

def gaus_model_comparison(training_tuple, testing_tuple, verbose = True, eps = 1e-3):
    
    x, av_resp = training_tuple
    test_x, test_resp = testing_tuple
    
    av_resp -= np.min(av_resp)
    test_resp -= np.min(test_resp)
    
    av_resp += eps
    test_resp += eps
    
    band_mod = GaussianModel()
    pars = band_mod.guess(av_resp, x=x)
    pars = bandPassParams(pars, av_resp, x)
    band_out = band_mod.fit(av_resp, pars, x=x)
    
    band_res = np.mean((test_resp - band_out.eval(x = test_x) + eps)**2)

    inc_mod = GaussianModel()
    pars = inc_mod.guess(av_resp, x=x)
    pars = increasingParams(pars, av_resp, x)
    inc_out = inc_mod.fit(av_resp, pars, x=x)
    
    inc_res = np.mean((test_resp - inc_out.eval(x = test_x) + eps)**2)
        
    dec_mod = GaussianModel()
    pars = dec_mod.guess(av_resp, x=x)
    pars = decreasingParams(pars, av_resp, x)
    
    dec_out = dec_mod.fit(av_resp, pars, x=x)
    
    dec_res = np.mean((test_resp - dec_out.eval(x = test_x) + eps)**2)

    if verbose:
        band_out.plot_fit()
        inc_out.plot_fit()
        dec_out.plot_fit()
        #plt.show()
        #plt.savefig("./final_submission/figs/model_example_inc.eps", dpi = 300, format = 'eps')
        print(band_res, inc_res, dec_res)

    return band_out, inc_out, dec_out

def arrange_data_rs_new(data_set):
    ds_data = {}

    #collect the stimulus tables, and running speed for each dataset
    for ds in data_set.keys():
        _, dff = data_set[ds].get_dff_traces()
        cells = data_set[ds].get_cell_specimen_ids()

        data = {'cell_ids':cells, 'raw_dff':dff }
        for stimulus in data_set[ds].list_stimuli():


            table = data_set[ds].get_stimulus_table(stimulus)

            data[stimulus] = table

        dxcm, dxtime = data_set[ds].get_running_speed()   
        data['running_speed'] = dxcm

        ds_data[ds] = data

    #arrange the data for each separate stimuli in a dictionary. Not averaging over
    #presentation of a given image, just concatenating all cell traces, and corresponing
    #running speed. 
    arranged_data = {}
    for ds in data_set.keys():
        dff_data = ds_data[ds]

        data = {}
        for stimulus in data_set[ds].list_stimuli():
            rs = np.zeros([1])
            dfof = np.zeros([len(dff_data['cell_ids']), 1])
            for index, row in dff_data[stimulus].iterrows():
                dfof = np.concatenate((dfof,dff_data['raw_dff'][:, int(row['start']) + 2: int(row['end']) + 2]), axis = 1)
                rs = np.concatenate((rs, dff_data['running_speed'][int(row['start']): int(row['end'])]), axis = 0)     

            data[stimulus + '_rs'] = np.array(np.squeeze(rs))
            data[stimulus + '_dff'] = np.array(np.squeeze(dfof))

        arranged_data[ds] = data  
        

    #groups the data into 'natural', 'spontaneous', or 'artificial'
    #TODO: subsample

    tb_data = {}
    for ds_id in arranged_data.keys():

        data  = arranged_data[ds_id]

        #binning into synthetic, natural, and stimulus
        #_data = {'synthetic_rs': None, 'natural_rs': None, 'spontaneous_rs': None,'synthetic_dff': None, 'natural_dff': None, 'spontaneous_dff':None, 'synthetic_pa': None, 'natural_pa': None, 'spontaneous_pa': None}

        #just binning into stimulus and spontaneous
        _data = {'stimulus_rs': None, 'spontaneous_rs': None,'stimulus_dff': None, 'spontaneous_dff':None}

        for stimulus in data_set[ds_id].list_stimuli():

            print(stimulus)

            if ('locally_sparse_noise' in stimulus) or ('gratings' in stimulus):
                stim_key = 'stimulus'
                #stim_key = 'synthetic'
            elif ('natural' in stimulus):
                stim_key = 'stimulus'
                #stim_key = 'natural'
            elif ('spontaneous' == stimulus):
                 stim_key = 'spontaneous'

            #stim_key = stimulus  
            run_speed =  np.array(data[stimulus + '_rs'])
            dff = np.array(data[stimulus + '_dff'])


            if _data[stim_key + '_rs'] is None:
                _data[stim_key+ '_rs'] = run_speed
            else:
                _data[stim_key + '_rs'] = np.concatenate((_data[stim_key + '_rs'], run_speed), axis = 0)


            if _data[stim_key + '_dff'] is None:
                _data[stim_key+ '_dff'] = dff
            else:
                _data[stim_key + '_dff'] = np.concatenate((_data[stim_key + '_dff'], dff), axis = 1)            

        tb_data[ds_id] = _data  

    return tb_data, arranged_data

def download_data(region, cre_line, stimulus = None, tracking = False):
    '''
    region = [reg1, reg2, ...]
    cre_line = [line1, line2, ...]
    '''
    boc = BrainObservatoryCache(manifest_file='boc/manifest.json')
    ecs = boc.get_experiment_containers(targeted_structures=region, cre_lines=cre_line)

    ec_ids = [ ec['id'] for ec in ecs ]

    #exp = boc.get_ophys_experiments(experiment_container_ids=ec_ids, )

    if stimulus == None:
        exp = boc.get_ophys_experiments(experiment_container_ids=ec_ids,  require_eye_tracking=tracking)

    else:
        exp = boc.get_ophys_experiments(experiment_container_ids=ec_ids, stimuli = stimulus,  require_eye_tracking=tracking)


    exp_id_list = [ec['id'] for ec in exp]

    data_set = {exp_id:boc.get_ophys_experiment_data(exp_id) for exp_id in exp_id_list}

    return data_set 

def cross_validate_decoding(all_run_tensors, all_stat_tensors, model = LogisticRegression(multi_class = 'multinomial', solver = 'lbfgs', C = 1), whiten = False, tr_shuffle = False, te_shuffle = False, cv_folds = 30, nomean = False, shuffle = False, label_shuffle = False, balance_classes = False, C = None, subsample = None):

    from scipy.stats import ranksums

    run_scores = []
    stat_scores = []
    keys = []


    p_values = []
    percent_diff = []

    for key in sorted(all_run_tensors.keys()):
    
        data_run, trialNum_run = all_run_tensors[key]
        data_stat, trialNum_stat = all_stat_tensors[key]
        

        #print (len(np.unique(trialNum_run)), len(np.unique(trialNum_stat)))

        if subsample is not None:
            data_run = data_run[:, subsample[key]]
            data_stat = data_stat[:, subsample[key]]

        if label_shuffle:
            trialNum_run = np.random.permutation(trialNum_run)
            trialNum_stat = np.random.permutation(trialNum_stat)


        run_score = []
        stat_score = []
        run_weights = []
        stat_weights = []
        
        for i in range(cv_folds):
            
            
            data_run, trialNum_run, data_stat, trialNum_stat = tensorize2(data_run, trialNum_run, data_stat, trialNum_stat, shuffle = shuffle, balance_classes = balance_classes)      


        
            rdff_train, rdff_test, rtrialNums_train, rtrialNums_test = train_test_split(np.array(data_run), np.array(trialNum_run), stratify = trialNum_run, test_size = .5)
            sdff_train, sdff_test, strialNums_train, strialNums_test = train_test_split(np.array(data_stat), np.array(trialNum_stat), stratify = trialNum_stat, test_size = .5)

            #print data_run.shape, len(trialNum_run), rdff_train.shape, rdff_test.shape, len(np.unique(rtrialNums_train)), len(np.unique(rtrialNums_test)), len(rtrialNums_test), len(rtrialNums_train)


            #            rdff_train, rtrain_labels, _ = tensorize(rdff_train, rtrialNums_train, shuffle = tr_shuffle, nomean= nomean, whiten = whiten, min_class_els = 0)
            #            rdff_test, rtest_labels, _ = tensorize(rdff_test, rtrialNums_test, shuffle = te_shuffle, nomean = nomean, whiten = whiten, min_class_els = 0)

            #            sdff_train, strain_labels, _ = tensorize(sdff_train, strialNums_train, shuffle = tr_shuffle, nomean = nomean, whiten = whiten, min_class_els = 0)
            #            sdff_test, stest_labels,_ = tensorize(sdff_test, strialNums_test, shuffle = te_shuffle, nomean = nomean, whiten = whiten, min_class_els = 0)    

            #model = model(multi_class = 'multinomial', solver = 'lbfgs', C = C)
            #model = GaussianNB()
            #model = SVC()
            model.fit(sdff_train, strialNums_train)
            #stat_weights.append(model.coef_)                  
            stat_score.append(model.score(sdff_test, strialNums_test))

            #model = SVC()
            #model = model(multi_class = 'multinomial', solver = 'lbfgs', C = C)
            #model = GaussianNB()
            model.fit(rdff_train, rtrialNums_train)
            #run_weights.append(model.coef_)
            run_score.append(model.score(rdff_test, rtrialNums_test))


        percent_diff.append(2*(np.average(run_score) - np.average(stat_score))/ 
                   (np.average(run_score) + np.average(stat_score)))  
        
        run_scores.append(np.average(run_score))
        stat_scores.append(np.average(stat_score))
        keys.append(key)
        #weights_r.append(np.array(run_weights).mean(axis = 0))
       # weights_s.append(np.array(stat_weights).mean(axis = 0))

        _, p = ranksums(run_score, stat_score)

                
        
        p_values.append(p)

    return percent_diff, run_scores, stat_scores, p_values, keys# weights_r, weights_s

def get_split_data_decoder2(data_set, tuned_cell_dict, mask = False, sub_sample = False, tuning_type = 2, var_thresh = .01, do_zscore = False):
    '''
    '''
    all_run_tensors = {}
    all_stat_tensors = {}
    data_tensors = {}

    run_nums = []
    stat_nums = []

    run_stats = []
    stat_stats = []
    for key in data_set.keys():
        #print key

        #get the fluorescence data, and running speed using wrappers for the AIBS sdk, 
        #optionally filter for tuned cell,
        
        #print tuned_cell_dict, key
        #tuned_cell_dict = tuned_cell_dict[tuned_cell_dict.ds_key == key]
        

       # try:
        dff, im_array, stim_table = get_data(data_set[key], stim_info.DRIFTING_GRATINGS)

        if do_zscore:
            dff = zscore(np.array(dff), axis = 1, ddof = 1)
        try:
            n_neurons, n_datapoints = dff.shape

            MASK = mask
            SUBSAMPLE = sub_sample

            if MASK:
                #dec_cells = tuned_cell_dict[key][tuning_type]  #(0 is for tuned, 1 is for neg, 2 is for pos)
                
                if tuning_type == 0:
                    cells = tuned_cell_dict['cell_id'].tolist()
                else:
                    cells = tuned_cell_dict[tuned_cell_dict.tuning_type == tuning_type]['cell_id'].tolist() 
            
                cell_inds = []
            
                for cell in cells:
                    try:
                        cell_inds.append(data_set[key].get_cell_specimen_indices([int(cell)])[0] )
                    except:
                        pass
                
                #cell_inds = data_set[key].get_cell_specimen_indices(cells)
                dff = np.ma.array(dff, mask=False)
                dff.mask[cell_inds] = True
                dff = dff.compressed()
                dff.shape = [-1, n_datapoints]
            elif SUBSAMPLE:
                if tuning_type == 0:
                    cells = tuned_cell_dict['cell_id'].tolist()
                else:
                    cells = tuned_cell_dict[tuned_cell_dict.tuning_type == tuning_type]['cell_id'].tolist() 
            
                cell_inds = []
            
                for cell in cells:
                    try:
                        cell_inds.append(data_set[key].get_cell_specimen_indices([int(cell)])[0] )
                    except:
                        pass

                #print cells, cell_inds
                dff = dff[cell_inds]
                #dff[~np.isfinite(dff)] = 0

            dxcm, dtime = data_set[key].get_running_speed()

            stim_table_still = pd.DataFrame(columns = stim_table.columns)
            stim_table_run = pd.DataFrame(columns = stim_table.columns)


            
            for i, row in stim_table.iterrows():

                run_var = gaussian_filter(dxcm[int(row['start']) + 10: int(row['start'] + 70)], 2)
                run_min = min(run_var)
                run_max = max(run_var)
                

                run_speed = np.average(dxcm[int(row['start']) + 10 :int(row['start']) + 70])

                if run_speed > 3 and run_min > .5:
                    stim_table_run = stim_table_run.append(row)
                    run_stats.append([run_min, run_max, np.std(run_var), np.mean(run_var)])

                elif abs(run_speed) < .5 and run_max < 3:
                    stim_table_still =  stim_table_still.append(row)
                    stat_stats.append([run_min, run_max, np.std(run_var), np.mean(run_var)])

                else:
                    pass

            #run_stats = np.array(run_stats)
            #print run_stats
            #print np.mean(run_stats, axis = 0)


            num_r = len(stim_table_run)
            run_nums.append(num_r)

            num_s = len(stim_table_still)
            stat_nums.append(num_s)


            num_neurons, _ = dff.shape
                
           # if (num_r > 80 and num_s > 80) and (num_neurons > 10):
            if (num_r > 70 and num_s > 70) and (num_neurons > 5):
                print("using this mouse")

                subsample = min(num_r, num_s)

                inds_run = np.random.choice(range(num_r), size = subsample, replace = False)
                inds_stat = np.random.choice(range(num_s), size = subsample, replace = False)

                stim_table_still= stim_table_still.iloc[inds_stat]
                stim_table_run = stim_table_run.iloc[inds_run]

                responses_run, orientations_run = arrange_data_tuning(dff, dxcm, stim_table_run, ratio = False)
                responses_still, orientations_still = arrange_data_tuning(dff, dxcm, stim_table_still, ratio = False)


                num_r, num_s = len(orientations_run), len(orientations_still)



                #subsample = min(num_r, num_s)

                #inds_run = np.random.choice(range(num_r), size = subsample, replace = False)
                #inds_stat = np.random.choice(range(num_s), size = subsample, replace = False)


                #responses_run = responses_run[inds_run]
                #responses_still = responses_still[inds_stat]

                #orientations_run = orientations_run[inds_run]
                #orientations_still = orientations_still[inds_stat]


                all_run_tensors[key] = responses_run, orientations_run
                all_stat_tensors[key] = responses_still, orientations_still
            #else:
                #print "mouse excluded"

            #except KeyError:
            #    print "not tuned"


        #if len(all_run_tensors.keys()) < 3:
        #    all_run_tensors, all_stat_tensors = {}, {}

        except:
            print("the failed dataset is ", key)


    print("the number datasets we are returning is:", len(all_run_tensors.keys()))
    print("the number of datasets we tried was:", len(data_set.keys()))
        
    return all_run_tensors, all_stat_tensors, run_nums, stat_nums, run_stats, stat_stats

def calculate_meanvar(tensor):
    
    n_features = len(tensor)

    #first we calculate overall variance

    _, n_neurons = tensor[0].shape
    all_variance = [[] for i in range(n_neurons)]

    for i in range(n_features):
        num_trials, n_neurons = tensor[i].shape
        for j in range(n_neurons):
            for k in range(num_trials):
                all_variance[j].append(tensor[i][k, j])
        
    baseline = np.zeros([n_neurons])

    for i in range(n_neurons):
        baseline[i] = np.var(all_variance[i])
        
    trial_av = np.zeros([n_features, n_neurons])  + 1e-3

    for i in range(n_features):
        temp = tensor[i]
        trial_av[i] = np.mean(temp, axis = 0)
        
    var_av = np.var(trial_av, axis = 0)
        
    return var_av, baseline

def calculate_reliability(tensor):
        var_av, baseline = calculate_meanvar(tensor)

        return var_av / baseline
    
def tensorize2(data_run, labels_run, data_stat, labels_stat, shuffle = False, balance_classes = False, min_class_els = 2):

    lab_dict_r = {orien: i for i, orien in enumerate(set(labels_run))}
    num = len(labels_run)

    resp_dict_r = {lab_dict_r[key]:[] for key in lab_dict_r.keys() }

    for i in range(num):    
        ind = lab_dict_r[labels_run[i]]
        resp_dict_r[ind].append(data_run[i])

    lab_dict_s = {orien: i for i, orien in enumerate(set(labels_stat))}
    num = len(labels_stat)

    resp_dict_s = {lab_dict_s[key]:[] for key in lab_dict_s.keys() }

    for i in range(num):    
        ind = lab_dict_s[labels_stat[i]]
        resp_dict_s[ind].append(data_stat[i])



    if balance_classes:
        length_dict_r = {key: len(resp_dict_r[key]) for key in sorted(resp_dict_r.keys())}
        min_length_r = min(length_dict_r.values())

        length_dict_s = {key: len(resp_dict_s[key]) for key in sorted(resp_dict_s.keys())}
        min_length_s = min(length_dict_s.values())

        min_length = min(min_length_s, min_length_r)

        #if min_length < min_class_els:
            #print ("warning, current mouse has only {} examples per class".format(min_length))        
        

    new_labels_r = []
    new_data_r = []

    for key in sorted(resp_dict_r.keys()):
        data_list_r = np.array(resp_dict_r[key])

        if balance_classes:
            ssinds = np.random.choice(range(length_dict_r[key]), size = min_length, replace = False)
            data_list_r = data_list_r[ssinds]

        if shuffle is True:
            n_trials, n_neurons = data_list_r.shape

            for i in range(n_neurons):

                data_list_r[:, i] = np.random.permutation(data_list_r[:, i])   


        for i in range(len(data_list_r)):

            new_data_r.append(data_list_r[i])
            new_labels_r.append(key)


    new_labels_s = []
    new_data_s = []

    for key in sorted(resp_dict_s.keys()):
        data_list_s = np.array(resp_dict_s[key])

        if balance_classes:
            ssinds = np.random.choice(range(length_dict_s[key]), size = min_length, replace = False)
            data_list_s = data_list_s[ssinds]

        if shuffle is True:
            n_trials, n_neurons = data_list_s.shape

            for i in range(n_neurons):

                data_list_s[:, i] = np.random.permutation(data_list_s[:, i])   


        for i in range(len(data_list_s)):

            new_data_s.append(data_list_s[i])
            new_labels_s.append(key)

    return np.array(new_data_r), np.array(new_labels_r), np.array(new_data_s), np.array(new_labels_s)
