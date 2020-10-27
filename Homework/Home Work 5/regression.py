#regression.py  
import pandas as pd
import copy
from stats import *
import numpy as np
from scipy.stats import t, f

class Regression:
    def __init__(self):
        self.stats = stats()
        self.reg_history = {}
        
# if constant equal true, add a constant or column of ones to estimate  a constant      
    def OLS(self, reg_name, data, y_name, beta_names, min_val = 0, 
            max_val = None, constant = True):
    
# create a variable which we can call later    
        self.min_val = min_val
        if max_val != None:
            self.max_val = max_val
        else:
            self.max_val = len(data)
        self.reg_name = reg_name
        self.y_name = y_name
        
        self.beta_names = copy.copy(beta_names)
        self.data = data.copy()
        if constant:
            self.add_constant()
        self.build_matrices()
        self.estimate_betas_and_yhat()
        self.calculate_regression_stats()
        self.save_output()
        
        
    def save_output(self):
        self.reg_history[self.reg_name] = {"Reg Stats": self.stats_DF.copy(),
                                           "Estimates": self.estimates.copy(),
                                           "Cov Matrix": self.cov_matrix.copy(),
                                           "Data": self.data.copy()}

        
    def calculate_regression_stats(self):
        self.sum_square_stats()
        self.calculate_degrees_of_freedom()
        self.calculate_estimator_variance()
        self.calculate_covariance_matrix()
        self.calculate_t_p_error_stats()  
        self.calculate_root_MSE()  
        self.calculate_rsquared()  
        self.calculate_fstat()  
        self.build_stats_DF()
        
        
    def sum_square_stats (self):
        ssr_list = []
        sse_list = []
        sst_list = []
        mean_y = self.stats.mean(self.y).item()
        for i in range(len(self.y)):
            # ssr is sum of squared distances between the estimates
            # and the average of y values (y-bar)
            # ssr is sum of squared residual
            # sse is sum of squared error
            # sst is sum of squared totals
            y_hat_i = self.y_hat[i]
            y_i = self.y[i]
            r = y_hat_i - mean_y
            e = y_i - y_hat_i
            t = y_i - mean_y
            ssr_list.append((r) ** 2)
            sse_list.append((e) ** 2)
            sst_list.append((t) ** 2)            
        # since ssr, sse, sst use values from the matrices, select the value within the
        # resultant matrix using matrix.item(0)
        # so, call value within the matrix instead of entire matrix itself
        
        self.ssr = self.stats.total(ssr_list).item(0)
        self.sse = self.stats.total(sse_list).item(0)
        self.sst = self.stats.total(sst_list).item(0)
        
        
    def calculate_degrees_of_freedom(self):
        
# Degrees of freedom compares the number of observations to the number  
# of exogenous variables used to form the prediction  
# we do this by substracting the # of exogenous vars from the observation
        self.lost_degrees_of_freedom = len(self.estimates) 
# the above line of code to calc the degrees of freedom can also be written as:        
        #self.lost_degrees_of_freedom = len(self.beta_names)
        
        self.num_obs = self.max_val + 1 - self.min_val  
        self.degrees_of_freedom = self.num_obs - self.lost_degrees_of_freedom
        
        
    def calculate_estimator_variance(self):
        
# estimator variance is the sse normalized by the degrees of freedom  
# thus, estimator variance increases as the number of exogenous  
# variables used in estimation increases(i.e., as degrees of freedom fall)  
        self.estimator_variance = self.sse / self.degrees_of_freedom
        
         
    def calculate_covariance_matrix(self):
                
# Covariance matrix will be used to estimate standard errors for each coefficient.  
# estimator variance * (X'X)**-1  
        self.cov_matrix = float(self.estimator_variance) * self.X_transp_X_inv  
        self.cov_matrix = pd.DataFrame(self.cov_matrix,  
                                       columns = self.beta_names, 
                                       index = self.beta_names)    
        
            
    def add_constant(self):
        self.data["Constant"] = 1
        self.beta_names.append("Constant")
        
    
    def estimate_betas_and_yhat(self):
        #X betas = (X'X)**-1 *X'Y
        # beta values help us predict the y values
        self.betas = np.matmul(self.X_transp_X_inv, self.X_transp_y)  
        # y-hat = X * betas  
        self.y_hat = np.matmul(self.X, self.betas)  
         # Create a column that holds y-hat values
        self.data[self.y_name[0] + "estimator"] = [i.item(0) for i in self.y_hat]
        # create a table that holds the estimated coefficients  
        # this will also be used to store SEs, t-stats,and p-values
        self.estimates = pd.DataFrame(self.betas, index = self.beta_names, columns = ["Coefficient"])
        # identify y variable in index
        self.estimates.index.name = "y = " + self.y_name[0]
        
        
    def build_matrices(self):
        # Transform df to matrices
        # Let's start with the y-matrix
        self.y = np.matrix(self.data[self.y_name][self.min_val:self.max_val])
        # create a K X n nested list containing vectors from each exogenous veriable
        self.X = np.matrix(self.data[self.beta_names])
        self.X_transpose = np.matrix(self.X).getT()
        # (X'X)**-1  
        X_transp_X = np.matmul(self.X_transpose, self.X)  
        self.X_transp_X_inv = X_transp_X.getI()  
        # X'y  
        self.X_transp_y = np.matmul(self.X_transpose, self.y)
        
        
        
    def calculate_t_p_error_stats(self):        
        results = self.estimates  
        
        # then we'll create a table that holds the standard error, t-stat and p-value
        stat_sig_names = ["SE", "t-stat", "p-value"]  
        
        # create space or a few blank columns in data frame for SE, t, and p  
        for stat_name in stat_sig_names:  
            results[stat_name] = np.nan  
            
        # cycle through the list of variables and calc the stats for each variable  
        for var in self.beta_names: 
            
            # SE ** 2 of coefficient is found in the diagonal of cov_matrix  
            results.loc[var]["SE"] = self.cov_matrix[var][var] ** (1/2)  

            # t-stat = Coef / SE  
            results.loc[var]["t-stat"] = results["Coefficient"][var] / results["SE"][var]  
                
            # p-values is estimated using a table that transforms t-value in light of degrees of freedom 
            # we multiplied by 2 for a 2-tailed test
            # we'll round the results to 5 decimal places
            results.loc[var]["p-value"] = np.round(t.sf(np.abs(results.loc[var]["t-stat"]),\
                                                        self.degrees_of_freedom + 1) * 2, 5)
                
        # create ratings for statistical significance according to p-values, 0.05, 0.01, 0.001
        ratings = [.05, .01, .001] 
                
        # values for significances will be blank unless p-values < .05  
        # pandas does not allow np.nan values or default blank strings to be replaced   
        # significance = ["" for i in range(len(self.beta_names))] 
        # the above code can also be simplified as:
        significance = ["" for name in self.beta_names]
        for i in range(len(self.beta_names)):  
            var = self.beta_names[i]  
            for rating in ratings:  
                if results.loc[var]["p-value"] < rating: 
                    
                    # For each rating, compare the P-Value to the rating add a "*" for each level that the P-value is lower 
                    # so, if it's below 0.5, add *, if it's below 0.1, add another *, if it's below 0.001, add yet another *
                    significance[i] = significance[i]  + "*"  
        results["signficance"] = significance
        
        
        
# calculating root MSE, r-squared and F-stat        
    def calculate_root_MSE(self):  
        self.root_mse = self.estimator_variance ** (1/2)  

    def calculate_rsquared(self):  
        self.r_sq = self.ssr / self.sst
        self.adj_r_sq = 1 - self.sse / self.degrees_of_freedom / (self.sst\
                                                                  / (self.num_obs - 1)) 
        

    def calculate_fstat(self):  
        self.f_stat = (self.sst - self.sse) / (self.lost_degrees_of_freedom\
                                               - 1) / self.estimator_variance  
            
# buiding a dictionary with all the stats (f-stats, SSE, MSE, etc)  then we'll create a df out of it.            
    def build_stats_DF(self):
        # create dictionary of stats
        stats_dict = {"r**2":[self.r_sq],
                      "adj r**2":[self.adj_r_sq],
                      "f-stat":[self.f_stat],
                      "Est Var":[self.estimator_variance],  
                      "rootMSE":[self.root_mse],  
                      "SSE":[self.sse],  
                      "SSR":[self.ssr],   
                      "SST":[self.sst],  
                      "Obs.":[int(self.num_obs)],   
                      "DOF":[int(self.degrees_of_freedom)]} 
        
        # make df for the stats dict
        self.stats_DF = pd.DataFrame(stats_dict)
        
        # create a name for the df
        self.stats_DF = self.stats_DF.rename(index={0:"Estimation Statistics"})
        
        # transpose df
        self.stats_DF = self.stats_DF.T
        
        
        
    def joint_f_test(self, reg1_name, reg2_name):  
    # identify data for each regression  
        reg1 = self.reg_history[reg1_name]  
        reg2 = self.reg_history[reg2_name]  
    # identify beta estimates for each regression to draw variables  
        reg1_estimates = reg1["Estimates"]          
        reg2_estimates = reg2["Estimates"]  
    # name of y_var is saved as estimates index name  
        reg1_y_name = reg1_estimates.index.name  
        reg2_y_name = reg2_estimates.index.name  
        num_obs1 = reg1["Reg Stats"].loc["Obs."][0]  
        num_obs2 = reg2["Reg Stats"].loc["Obs."][0]  
    # check that the f-stat is measuring restriction, not for diff data sets
        if num_obs1 != num_obs2:   
            self.joint_f_error()  
        if reg1_y_name == reg2_y_name:          
            restr_reg = reg1 if \
                len(reg1_estimates.index) < len(reg2_estimates.index) else reg2 
            unrestr_reg = reg2 if restr_reg is reg1 else reg1  
            restr_var_names = restr_reg["Estimates"].index  
            unrestr_var_names = unrestr_reg["Estimates"].index  
    # identify statistics for each regression  
        restr_reg = restr_reg if False not in \
                [key in unrestr_var_names for key in restr_var_names] else None
        if restr_reg == None:  
            self.joint_f_error()  
        else:  
            sser = restr_reg["Reg Stats"].loc["SSE"][0]  
            sseu = unrestr_reg["Reg Stats"].loc["SSE"][0]  
            dofr = restr_reg["Reg Stats"].loc["DOF"][0]       
            dofu = unrestr_reg["Reg Stats"].loc["DOF"][0]  
            dfn = dofr - dofu  
            dfd = dofu - 1  
            f_stat = ((sser - sseu) / (dfn)) / (sseu / (dfd))  
            f_crit_val = 1 - f.cdf(f_stat,dfn = dfn, dfd = dfd)  
    # make dictionary  
            f_test_label = ""  
            for key in unrestr_var_names:  
                if key not in restr_var_names:  
                     f_test_label = f_test_label + str(key) + " != "  
            f_test_label = f_test_label + "0"  
            res_dict = {"f-stat":[f_stat],  
                        "p-value":[f_crit_val],  
                        "dfn":[dfn],  
                        "dfd":[dfd]}  
            res_DF = pd.DataFrame(res_dict)  
            res_DF = res_DF.rename(index={0:""})  
            res_DF = res_DF.T  
            res_DF.index.name = f_test_label  

        return res_DF  

def joint_f_error(self):  
        print("Regressions not comparable for joint F-test")  
        return None
        
        
        
    

        
        