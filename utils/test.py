



            #### Initilize the results
            reg_model_projection = np.array(pd_fulimp_reg[Y_name].values.tolist()) 
            reg_model_references = np.array(pd_radonl_reg[Y_name].values.tolist()) 
            reg_empirical_projection_corrected = np.zeros(len(years)) 
            reg_empirical_projection_corrected[0] = np.log(reg_model_projection[0]) 

            #### Growth rate from reference 
            base_growth_corrected = np.log(reg_model_references[1:] / reg_model_references[:-1])
            
            #### Time step iteration 
            for i in range(1, len(years)): 
                actual_growth_corrected = base_growth_corrected[i-1] + d[i] - b[i]
                reg_empirical_projection_corrected[i] = reg_empirical_projection_corrected[i-1] + actual_growth_corrected 
            reg_empirical_projection_corrected = np.exp(reg_empirical_projection_corrected)