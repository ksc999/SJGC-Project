import sys
import pandas as pd
import numpy as np
from scipy.optimize import minimize


class MyGPR:
    def __init__(self, data_path, train_ratio=0.5, selection_mode='trunctated', kernel_name='rbf',
                 fit_param_times=1000):
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
            self.y_test = self.y[:dividing_line]
        # 'random' means that GPR is taken as a information completion task
        elif self.selection_mode == 'random':
            train_nums = int(len(self.x) * self.train_ratio)
            self.x_train = np.random.choice(self.x, size=train_nums, replace=False)
            self.x_test = np.setdiff1d(self.x, self.x_train)
            self.y_train = self.y[self.x_train]
            self.y_test = self.y[self.y_test]
    
    def _get_kernel_params(self):
        if self.kernel_name == 'rbf':
            param = {'l': 100, 'sigma_f': 20, 'sigma_n': 1e-2}
            param_bound = {'l': (1e-4, 1e2), 'sigma_f': (1e-4, 1e2), 'sigma_n': (1e-10, 1e-2)}
            
        return param, param_bound    
            
    def _get_kernel(self):
        if self.kernel_name == 'rbf':   
            def kernel(param, x_1, x_2):
                # the distance matrix is of the size (x_1, x_2)
                distance_matrix = np.sum(x_1**2).reshape(-1, 1) + np.sum(x_2**2) \
                                    - 2 * np.dot(x_1.reshape(-1, 1), x_2.reshape(1, -1))
                return param[1]**2 * np.exp(-0.5 / param[0]**2 * distance_matrix)
            
        return kernel
        
    def _get_kernel_loss(self):
        if self.kernel_name == 'rbf':
            def kernel_loss(param, x, y):
                K = self.kernel(param, x, x) \
                        + param[2] * np.eye(len(x))
                loss = 0.5 * np.dot(y, np.linalg.inv(K) @ y) \
                        + 0.5 * np.linalg.slogdet(K)[1] \
                        + 0.5 * len(x) * np.log(2 * np.pi)
                return loss.ravel()
            
        return kernel_loss
    
    def fit_params(self):
        # thanks to the local-min trap of this non-convex optimization problem
        # we have to try many initial values of the args to get a pretty good result
        args_init_list = []
        for lower_bound, upper_bound in self.kernel_param_bound.values():
            # use log interpolation
            log_lower_bound = np.log(lower_bound)
            log_upper_bound = np.log(upper_bound)
            log_list = np.linspace(log_lower_bound, log_upper_bound, self.fit_param_times)
            exp_list = np.exp(log_list)
            np.random.shuffle(exp_list) # shuffle the list
            args_init_list.append(exp_list)
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
    


if __name__ == '__main__':
    demo_gpr = MyGPR("./data/adsk/adsk_1M.csv")
    demo_gpr.fit_params()