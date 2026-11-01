### Optimization of Monoclonal Antibody Production
The main objective of this program is to analyze bioprocess data and identify operating conditions that improve monoclonal antibody production. The program evaluates important production parameters such as harvest times, production costs and profits, and space time yield. Furthermore, polynomial regression and K-nearest neighbors models are compared using cross-validation to assess predictive performance. Residual analysis is also implemented to perform model evaluation and provide model recommendations. 

#### Highlights
- Developed a Python program capable of analyzing a bioprocess workflow
- Calculated key economic parameters such as production coses, profitability, and space time yield
- Designed a workflow capable of comparing batch performance
- Automated the generation of plots and result files that demonstrate key findings
- Identified the best performing batch among the tested conditions

#### Environment
- Python: 3.13.18
- Matplotlib: 3.11.2
- NumPy: 2.5.4
- pandas: 3.1.0
- SciPy: 1.18.1
- scikit-learn: 1.9.1

#### Code Design
##### Program Initialization and Data Loading
When main.py is executed, the program imports the required constants and functions from constants.py and functions.py as well as the Batch class from classes.py. The program then creates a Batch object using Batch = Batch() and calls batch.load_data() to access the data loading function implemented in classes.py. This stage is important as it allows the program to load the required datasets and makes the bioprocess data available for subsequent analysis. Furthermore, separating the data loading process into the Batch class helps organize the code within the classes.py with main.py establishing the overall workflow of the program.

##### Data Processing
After the datasets have been loaded, the program then organizes the bioprocess measurements by batch so that the data from each production run is separately. The dataset contains variables such as glucose, glutamine, lactate, ammonia, viable cell density (VCD), total cell density (TCD), and monoclonal antibody concentration (mAb). the program accesses these measurements and the corresponding initial conditions used in the economic calculations. This stage is necessary is the program because it associates the correct measurements and operating conditions with each batch to evaluate the production performance. Overall, this stage is necessary to ensure that the correct measurements and operating conditions are used for each batch. It also prepares the data for calculating key production and performance parameters, allowing the program to compare batches and identify suitable conditions for antibody production.

##### Economic Calculations
For the economic calculations main.py accesses the formulas in functions.py which are used to calculate the production cost, product value, profit, space time profile, and space time yield for each of the 30 batches. Production cost estimates the expenses associated with each batch, and product value estimates the value of antibodies (mAb) produced. Profit is then calculated by determining the different between the product cost and product value. Space time profit and yield accounts for the production time and downtime when evaluating batch performance. These calculations are necessary to compare the economic performance and productivity of different batches. The results help the program identify which of the evaluated batches offer the best performance.

##### Model Training and Cross Validation
For model training, the program trains both polynomial regression and K-nearest neighbors (KNN) models to predicts mAb production using the initial operating conditions. Cross validation is then used to compare different model configurations and evaluate predictive performance using RMSE. This stage helps identify the best performing model for further analysis and the final batch recommendation.

##### Model Evaluation and Residual Analysis 
Following model training, the program evaluates their predictive performance using root mean squared error (RMSE). This metric measures the difference between the predicted and actual mAb concentrations, a lower RMSE indicates a more accurate prediction. The results are then compared across different model configurations such ad, polynomial regression and K-nearest neighbors (KNN). The second-degree polynomial regression model achieved the lowest RMSE of approximately 0.02948 making it the selected model for further analysis. Residual analysis is then performed on the selected model to examine more closely its prediction errors. The program calculates residual statistics to summarize errors and assess the model's performance. This stage helps identify potential limitation in the model and provides information to support the final batch recommendation.

##### Recommendation
Finally, the program uses the calculated economic metrics and batch information to identify the recommended batch, and it's operating conditions. These conditions include the initial inputs used for the production which help determine how the bioprocess should be operated. This stage is necessary because it uses the results of the data analysis and economic calculation to determine a practical recommendation. This stage identifies the conditions associated with the selected batch and the program then provides recommendations on which operating conditions provide the best mAb production. 

#### Analysis 

##### Batch Selection
A total of 30 batch runs were randomly selected from the 1,029 available batch datasets to simplify data handling and facilitate processing within the program. The initial approach used in this program was to select the first 30 batches in the data file as it made transferring files easier. However, a major limitation was discovered in this approach. In batches 1-30 the glucose concentration remained constant at 5 mM, and the concentration of glutamine only varied from 1-2 mM. Therefore, selecting only the first 30 batches would not have accurately modeled antibody growth in response to varied glucose, glutamine and VCD levels. Therefore, unlike the original approach, which used the first 30 batches, the updated program randomly selects 30 batch IDs from the available datasets. The corresponding initial conditions are then retrieved from the provided initial conditions file which contains the starting glucose, glutamine, and viable cell density (VCD) for each batch. In randomly selecting batches from the complete dataset it allows the program to evaluate a wider range of initial conditions, including varied glucose, glutamine, and VCD concentrations. However, in selecting only 30 batches there still remains a limitation because the selected sample may not represent the full range of conditions available across all 1,029 batches. The results may also vary depending on which batches are selected. If the program ran with a larger sample range it would provide a more comprehensive evaluation of the effects of initial conditions on mAb production and economic performance.

##### Model Selection 
Both polynomial regression and K-nearest neighbors (KNN) regression models were evaluated to determine which model provided the best predictions of mAb production. Both approaches were compared using RMSE obtained through cross-validation where a lower RMSE indicates smaller prediction errors and better predictive performance. Based on the results of the cross-validation, the second-degree polynomial regression model achieved the lowest RMSE which is shown in the table below. Therefore, the second-degree polynomial regression model was selected for further analysis and to support the final production recommendation.  

|Model|Lag Order|Parameters|RMSE    |
|-----|---------|----------|--------|
|poly |2        |{'polynomialfeatures__degree': 2}|0.05434744438566099|
|poly |3        |{'polynomialfeatures__degree': 2}|0.05582396826273742|
|poly |1        |{'polynomialfeatures__degree': 2}|0.05604262468156351|
|poly |4        |{'polynomialfeatures__degree': 2}|0.06248252630983405|
|poly |5        |{'polynomialfeatures__degree': 2}|0.07288554254378998|
|knn  |1        |{'kneighborsregressor__n_neighbors': 5, 'kneighborsregressor__weights': 'uniform'}|0.12235187823981504|
|knn  |2        |{'kneighborsregressor__n_neighbors': 5, 'kneighborsregressor__weights': 'uniform'}|0.12470887234724182|
|knn  |3        |{'kneighborsregressor__n_neighbors': 5, 'kneighborsregressor__weights': 'uniform'}|0.12707297131980175|
|knn  |4        |{'kneighborsregressor__n_neighbors': 5, 'kneighborsregressor__weights': 'uniform'}|0.12940944480715053|
|knn  |5        |{'kneighborsregressor__n_neighbors': 5, 'kneighborsregressor__weights': 'uniform'}|0.12946886615118677|


##### Model Architecture and Hyperparameters
The selected model uses second-degree polynomial regression which was implemented using PolynomialFeatures() and LinearRegression(). Polynomial regression uses higher order terms which allows the model to capture nonlinear relationships between the lagged measurements and their current values. Both second and third degree polynomial models were evaluated with the 2nd degree model performing better based on the cross-validation results. K-nearest neighbors (KNN) was used as an alternative model which was used as a comparison model against polynomial regression to determine whether another regression model would be more applicable for this program. KNN predicts values based on the neighboring observations in the training dataset. The number of neighbors was varied between 2 and 5 using uniform weighting. Uniform weighting means that each neighboring observation contributes equally to the prediction. Both models were evaluated using GroupKfold cross-validation with three folds grouping observations by batch ID. Different lag order were also evaluated ranging from 1 to 5 to determine how many previous measurements should be used as model inputs. As mentioned, the best performing configuration (lowest RMSE value) was the second-degree polynomial regression with a lag order of 2.   

##### Recommended Production Conditions
Based on the 30 randomly selected batch runs, Batch 511 was identified as the recommended production condition because it achieved the highest space-time profit among the evaluated batches. The recommended initial conditions were a glucose concentration of 20 mM, a glutamine concentration of 4 mM, and an initial viable cell density (VCD) of 600,000 cells/mL. Across the 1,029 batches the initial glucose concentrations ranges from 5 to 35 mM, glutamine concentrations ranged from 1 to 7 mM, and initial VCD ranged from 200,000 to 1,400,000 cells/mL. Therefore, as the selected batch had glucose and glutamine concentration values on the higher end of the allotted range it could be inferred that those initial conditions lead to higher antibody production. Furthermore, the VCD value was on the lower middle range of it's allotted range which could also be an indicator of which value of VCD leads to a higher antibody production. 

The recommended harvest time was approximately 3.67 days. The estimated cell culture cost under these conditions was 580 dollar/L, with a final mAb concentration of 1.03 mg/L and a corresponding product value of 1,030 dollar/L. The estimated space-time profit was 96.43 dollar/(L·d), while the space-time yield was 0.2207 mg/(L·d). Using the model's residual-based intervals, the estimated product value ranged from 921.85 to 1,138.15 dollar/L. The space-time profit ranged from 73.25 to 119.60 dollar/(L·d), while the space-time yield ranged from 0.1975 to 0.2439 mg/(L·d). These intervals provide an indication of the uncertainty associated with the model predictions.

|Batch ID|Initial Glucose [mM]|Initial Glutamine [mM]|Initial VCD [cells/mL]|Harvest Time [d]  |Cell Culture Cost [$/L]|mAb [mg/L]|mAb Lower [mg/L]  |mAb Upper [mg/L]  |Cell Culture Value [$/L]|Value Lower [$/L]|Value Upper [$/L] |Space-Time Profit [$/(L d)]|Profit Lower [$/(L d)]|Profit Upper [$/(L d)]|Space-Time Yield [mg/(L d)]|Yield Lower [mg/(L d)]|Yield Upper [mg/(L d)]|
|--------|--------------------|----------------------|----------------------|------------------|-----------------------|----------|------------------|------------------|------------------------|-----------------|------------------|---------------------------|----------------------|----------------------|---------------------------|----------------------|----------------------|
|511.0   |20.0                |4.0                   |600000.0              |3.6666666666666665|580.0                  |1.03      |0.9218542016026405|1.1381457983973595|1030.0                  |921.8542016026405|1138.1457983973596|96.42857142857144          |73.2544717719944      |119.6026710851485     |0.22071428571428575        |0.1975401860577087    |0.2438883853708628    |


##### Model Limitations 
Several limitations should be considered when interpreting the results. Firstly, although the program now randomly selects 30 batches from the 1,029 available datasets, this is still a relatively small sample of the complete dataset. Consequently, the selected batches may not capture the full range of initial conditions or represent all possible combinations of glucose, glutamine, and VCD. The random selection also means that the results and recommended batch may differ between samples. Additionally, although the second-degree polynomial regression achieved the lowest RMSE among the tested configurations, prediction errors remain. Therefore, model predictions and the associated residual-based intervals should be interpreted with caution. Finally, the recommended batch represents the highest space-time profit among the 30 evaluated batches rather than a confirmed global optimum. Evaluating a larger sample of batches would improve the representativeness of the analysis and provide a more comprehensive evaluation of the production conditions.

##### Final Recommendations
Based on the randomly selected batches, Batch 511 was identified as having the best production conditions, achieving a space-time profit of $96.43/(L·d). The recommended initial conditions were 20 mM glucose, 4 mM glutamine, and an initial VCD of 600,000 cells/mL, with a harvest time of approximately 3.67 days. For future analysis, a larger selection of batches should be evaluated to capture a wider range of initial conditions. This would allow the effects of glucose, glutamine, and VCD on mAb production and economic performance to be investigated more comprehensively. Further model evaluation using different random samples could also help determine whether the selected model and recommended production conditions remain consistent.