from my_gpr import MyGPR
import pandas as pd
import numpy as np

def tackle_hyperparameters(hp, prefix='./hyperparameters/'):
    hps = pd.read_csv(prefix + hp)
    for i in range(len(hps)):
        hp = hps.loc[i]
        data_path = './data/' + hp['name'] + '/' + hp['expansion'] + '.csv'
        img_name = hp['name'] + '_' + hp['expansion'] \
                    + '_tr_' + str(hp['train_ratio']) \
                    + '_sm_' + hp['selection_mode'] \
                    + '_kn_' + hp['kernel_name'] \
                    + '_spf_' + str(hp['sequential_predict_flag']) + '.png'
        gpr = MyGPR(
            data_path,
            train_ratio=hp['train_ratio'],
            selection_mode=hp['selection_mode'],
            kernel_name=hp['kernel_name'],
            fit_param_times=hp['fit_param_times'],
            log_sample_flag=hp['log_sample_flag']
        )
        gpr.fit_params()
        if hp['sequential_predict_flag']:
            gpr.seqeuntial_predict()
        else:
            gpr.predict()
        gpr.draw_prediction(img_name)



if __name__ == '__main__':
    tackle_hyperparameters('hp_1.csv')