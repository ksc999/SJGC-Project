import sys
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


class MyGPR:
    def __init__(
            self, data_path,
            train_ratio=0.9,
            selection_mode='trunctated',
            kernel_name='rbf',
            fit_param_times=10,
            log_sample_flag=True
        ):
        data = pd.read_csv(data_path)
        money = data['Close/Last']
        money = list(map(lambda str_money: float(str_money[1:]), money))    # convert str to float
        money = np.array(money)
        money_mean = np.mean(money)
        money -= money_mean # normalization
        self.y = money  # set money as y
        self.x = np.arange(len(self.y)) # set days as x
        self.train_ratio = train_ratio
        self.selection_mode = selection_mode
        self.kernel_name = kernel_name
        self.fit_param_times = fit_param_times
        self.log_sample_flag = log_sample_flag
        self._divide_data()
        self.kernel_param, self.kernel_param_bound = self._get_kernel_params()
        self.kernel = self._get_kernel()
        self.kernel_loss = self._get_kernel_loss()
        
    def _divide_data(self):
        # 'trunctated' means that GPR is taken as a sequential prediction
        if self.selection_mode == 'trunctated': 
            dividing_line = int(len(self.x) * self.train_ratio)
            self.x_train = self.x[:dividing_line]
            self.x_test = self.x[dividing_line:]
            self.y_train = self.y[:dividing_line]
            self.y_test = self.y[dividing_line:]
        # 'random' means that GPR is taken as a information completion task
        elif self.selection_mode == 'random':
            train_nums = int(len(self.x) * self.train_ratio)
            self.x_train = np.random.choice(self.x, size=train_nums, replace=False)
            self.x_test = np.setdiff1d(self.x, self.x_train)
            self.y_train = self.y[self.x_train]
            self.y_test = self.y[self.x_test]
    
    def _get_kernel_params(self):
        if self.kernel_name == 'rbf':
            # param = {'l': 6.2949, 'sigma_f': 70.6718, 'sigma_n': 0.1}
            # param_bound = {'l': (1e-1, 1e2), 'sigma_f': (1e-2, 1e2), 'sigma_n': (1e-10, 1e-3)}
            param = {'l': 1.1187343193934338, 'sigma_f': 16.20733936502278}
            param_bound = {'l': (1e-1, 1e3), 'sigma_f': (1e-1, 1e2)}
            
        return param, param_bound    
            
    def _get_kernel(self):
        if self.kernel_name == 'rbf':   
            def kernel(param, x_1, x_2):
                # the distance matrix is of the size (x_1, x_2)
                distance_matrix = np.power(x_1, 2).reshape(-1, 1) + np.power(x_2, 2) \
                                    - 2 * np.dot(x_1.reshape(-1, 1), x_2.reshape(1, -1))
                return param[1]**2 * np.exp(-0.5 / param[0]**2 * distance_matrix)
            
        return kernel
        
    def _get_kernel_loss(self):
        if self.kernel_name == 'rbf':
            def kernel_loss(param, x, y):
                K = self.kernel(param, x, x) + 1e-8 * np.eye(len(x))
                loss = 0.5 * np.dot(y, np.linalg.inv(K) @ y) + 0.5 * np.linalg.slogdet(K)[1] 
                return loss.ravel()
            
        return kernel_loss
    
    def fit_params(self):
        # thanks to the local-min trap of this non-convex optimization problem
        # we have to try many initial values of the args to get a pretty good result
        args_init_list = []
        for lower_bound, upper_bound in self.kernel_param_bound.values():
            if self.log_sample_flag:
                # use log interpolation
                log_lower_bound = np.log(lower_bound)
                log_upper_bound = np.log(upper_bound)
                log_list = np.linspace(log_lower_bound, log_upper_bound, self.fit_param_times)
                tmp_list = np.exp(log_list)
            else:
                tmp_list = np.linspace(lower_bound, upper_bound, self.fit_param_times)
            np.random.shuffle(tmp_list) # shuffle the list
            args_init_list.append(tmp_list)
        args_init_list = np.array(args_init_list)
        args_bound = tuple(self.kernel_param_bound.values())
        fun_min = sys.float_info.max
        optimal_result = []
        for args_init in args_init_list.T:
            result = minimize(self.kernel_loss, args_init, bounds=args_bound,
                              args=(self.x_train, self.y_train))
            if result.fun < fun_min:
                fun_min = result.fun
                optimal_result = result.x    
        for i, key in enumerate(self.kernel_param.keys()):
            self.kernel_param[key] = optimal_result[i]
        print(f'kernel parameters after optimize: {self.kernel_param}')
        print(f'loss value now: {fun_min}')

    def predict(self):
        param = list(self.kernel_param.values())
        K_train_train = self.kernel(param, self.x_train, self.x_train)
        K_test_train = self.kernel(param, self.x_test, self.x_train)
        K_test_test = self.kernel(param, self.x_test, self.x_test)
        self.mu_test = K_test_train @ np.linalg.inv(K_train_train) @ self.y_train
        self.cov_test = K_test_test - K_test_train @ np.linalg.inv(K_train_train) @ K_test_train.T
    
    def draw_prediction(self, img_name, prefix='./imgs/', fig_size=(20, 10)):
        var = np.diag(self.cov_test)
        CI_upper_bound_test = self.mu_test + 1.96 * np.sqrt(var)
        CI_lower_bound_test = self.mu_test - 1.96 * np.sqrt(var)
        CI_upper_bound = np.zeros(len(self.x))
        CI_upper_bound[self.x_test] = CI_upper_bound_test
        CI_upper_bound[self.x_train] = self.y_train
        CI_lower_bound = np.zeros(len(self.x))
        CI_lower_bound[self.x_test] = CI_lower_bound_test
        CI_lower_bound[self.x_train] = self.y_train
        plt.figure(figsize=fig_size,dpi=90)
        if self.selection_mode == 'trunctated':
            plt.plot(self.x_train, self.y_train, 'k', label='train values')
            plt.plot(np.insert(self.x_test, 0, self.x_train[-1]), 
                     np.insert(self.y_test, 0, self.y_train[-1]), 
                     'r--', label='true test values')
            plt.plot(np.insert(self.x_test, 0, self.x_train[-1]), 
                     np.insert(self.mu_test, 0, self.y_train[-1]), 
                     'b', label='predicted test values')
            plt.fill_between(self.x_test, CI_lower_bound_test, CI_upper_bound_test, alpha=0.3)
        elif self.selection_mode == 'random':
            y = np.zeros(len(self.x))
            y[self.x_train] = self.y_train
            y[self.x_test] = self.mu_test
            plt.plot(self.x, y, 'b', label='predicted values')
            plt.plot(self.x_train, self.y_train, 'rx', label='train values', ms=8)
            plt.plot(self.x, self.y, 'r--', label='true values', alpha=0.8)
            plt.fill_between(self.x, CI_lower_bound, CI_upper_bound, alpha=0.3)
        plt.legend(prop={'size': 18})
        plt.savefig(prefix + img_name)    


    