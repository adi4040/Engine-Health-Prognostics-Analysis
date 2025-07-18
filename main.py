# Import necessary libraries
from tkinter import *
from tkinter import ttk
import tkinter as tk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.stats import binom, poisson, norm
from regressionModel import TurbofanRegressionModel as model 
from engineHealthModel import EngineHealthModel as health_model

# Load and prepare turbofan engine dataset
columns = ["id", "time"] + [f"op_setting_{i}" for i in range(1, 4)] + [f"sensor_{i}" for i in range(1, 22)]
df_train = pd.read_csv('CMAPSSData/train_FD001.txt', delim_whitespace=True, header=None, names=columns)
df = model().load_data('CMAPSSData/train_FD001.txt')
class PredictiveMaintenanceApp:
    def __init__(self, root):
        # Initialize main application window
        self.root = root
        self.root.geometry("1530x790+0+0")
        self.root.title("Engine Health Prognostics Analytics Toolkit (EHPAT)")
        
        # Create notebook for tabbed interface
        self.notebook = ttk.Notebook(root)
        
        # Create tabs
        self.prob_tab = ttk.Frame(self.notebook)
        self.hypo_tab = ttk.Frame(self.notebook)
        self.ml_tab = ttk.Frame(self.notebook)
        self.health_tab = ttk.Frame(self.notebook)
        
        # Add tabs to notebook
        self.notebook.add(self.prob_tab, text="Probability Analysis")
        self.notebook.add(self.hypo_tab, text="Hypothesis Testing") 
        self.notebook.add(self.ml_tab, text="Regression Implementation")
        self.notebook.add(self.health_tab, text="Engine Health Prediction")
        self.notebook.pack(expand=True, fill='both')
        
        # Initialize probability analysis components
        self.init_probability_tab()
        self.init_hypothesis_tab()
        self.init_regression_tab()
        self.init_health_tab()

    def init_probability_tab(self):
        # Sensor selection dropdown
        self.sensor_var = tk.StringVar()
        self.sensor_label = tk.Label(self.prob_tab, text="Select Sensor:")
        self.sensor_dropdown = ttk.Combobox(self.prob_tab, textvariable=self.sensor_var,
                                          values=[f"sensor_{i}" for i in range(1, 22)])
        self.sensor_label.pack(pady=5)
        self.sensor_dropdown.pack()
        self.sensor_dropdown.current(2)  # Default to sensor_3

        # Distribution selection
        self.dist_var = tk.StringVar()
        self.dist_label = tk.Label(self.prob_tab, text="Select Distribution:")
        self.dist_dropdown = ttk.Combobox(self.prob_tab, textvariable=self.dist_var,
                                         values=["Normal", "Binomial"])
        self.dist_label.pack(pady=5)
        self.dist_dropdown.pack()
        self.dist_dropdown.current(0)  # Default to Normal

        # Plot button
        self.plot_btn = tk.Button(self.prob_tab, text="Analyze Sensor Data", 
                                command=self.analyze_sensor_data)
        self.plot_btn.pack(pady=10)

        # Matplotlib figure setup
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.prob_tab)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Analysis report text area
        self.analysis_text = tk.Text(self.prob_tab, height=8, wrap=tk.WORD)
        self.analysis_text.pack(fill=tk.BOTH, expand=True)
        self.analysis_text.tag_configure('header', foreground='blue', font=('Arial', 10, 'bold'))
        self.analysis_text.tag_configure('body', font=('Arial', 9))

    def analyze_sensor_data(self):
        # Clearing previous visualization
        self.ax.clear()
        
        # Get selected sensor and distribution
        sensor = self.sensor_var.get()
        dist_type = self.dist_var.get()
        
        # Extract sensor data from dataframe
        data = df_train[sensor].dropna().values
        
        try:
            if dist_type == "Normal":
                # Fit normal distribution
                mu, sigma = data.mean(), data.std()  
                x = np.linspace(mu-4*sigma, mu+4*sigma, 100) #4 std dev in both directions, covers outliers beyond 3
                pdf = norm.pdf(x, mu, sigma)
                
                # Plot histogram and PDF
                self.ax.hist(data, bins=30, density=True, alpha=0.6, label='Sensor Data') #alpha-> semi transparent bars , bins->intervals of 30
                self.ax.plot(x, pdf, 'r-', label=f'Normal Fit (μ={mu:.1f}, σ={sigma:.1f})') #plot in red with 1 decimal formatting
                self.ax.set_xlabel("Sensor Values")
                self.ax.set_ylabel("Probability Density")
                
                # Generate analysis report
                report = f"""Normal Distribution Analysis for {sensor}
                - Mean sensor value: {mu:.2f}
                - Standard deviation: {sigma:.2f}
                - 95% data range: [{mu-2*sigma:.1f}, {mu+2*sigma:.1f}]  
                - Values outside 3σ ({mu-3*sigma:.1f}-{mu+3*sigma:.1f}) are potential outliers"""
                #2sigma range for CI and 3sigma for outliers                
                
            elif dist_type == "Binomial":
                # Convert continuous data to binary outcomes
                threshold = np.median(data)  #values above/below the median 
                binary_data = (data > threshold).astype(int)
                n = len(binary_data)  #total no. of trials
                p = binary_data.mean()
                
                x = np.arange(0, n+1)
                pmf = binom.pmf(x, n, p)
                
                # Plot PMF
                self.ax.bar(x, pmf, alpha=0.6, label=f'Binomial (n={n}, p={p:.2f})')
                self.ax.set_xlabel("Success Count")
                self.ax.set_ylabel("Probability")
                
                # Generate analysis report
                report = f"""Binomial Analysis for {sensor}
                - Trials (n): {n}
                - Success probability (p): {p:.2%}
                - Expected successes: {n*p:.1f}
                - 95% CI: [{binom.ppf(0.025, n, p)}, {binom.ppf(0.975, n, p)}]"""  #lower and upper bounds 
                
            # Finalize plot
            self.ax.set_title(f"{dist_type} Distribution Fit for {sensor}")
            self.ax.legend()
            self.canvas.draw()
            
            # Display analysis report
            self.analysis_text.delete(1.0, tk.END)
            self.analysis_text.insert(tk.END, "Analysis Report:\n", 'header')
            self.analysis_text.insert(tk.END, report.replace("                ", ""), 'body')
            
        except Exception as e:
            self.analysis_text.delete(1.0, tk.END)
            self.analysis_text.insert(tk.END, f"Error: {str(e)}", 'body')

    def init_hypothesis_tab(self):
        """Initialize the hypothesis testing tab"""
        # Header for the hypothesis tab
        header = tk.Label(self.hypo_tab, text="Hypothesis Testing", font=("Arial", 14))
        header.pack(pady=10)
        
        # Create a frame for controls
        control_frame = tk.Frame(self.hypo_tab)
        control_frame.pack(pady=10, fill=tk.X)
        
        # Sensor selection
        sensor_frame = tk.Frame(control_frame)
        sensor_frame.pack(side=tk.LEFT, padx=10)
        
        sensor_label = tk.Label(sensor_frame, text="Select Sensor:")
        sensor_label.pack(side=tk.LEFT, padx=5)
        
        self.hypo_sensor_var = tk.StringVar()
        self.hypo_sensor_dropdown = ttk.Combobox(sensor_frame, textvariable=self.hypo_sensor_var,
                                                values=[f"sensor_{i}" for i in range(1, 22)])
        self.hypo_sensor_dropdown.pack(side=tk.LEFT, padx=5)
        self.hypo_sensor_dropdown.current(2)  # Default to sensor_3
        
        # Engine group selection
        group_frame = tk.Frame(control_frame)
        group_frame.pack(side=tk.LEFT, padx=10)
        
        group_label = tk.Label(group_frame, text="Engine Group:")
        group_label.pack(side=tk.LEFT, padx=5)
        
        self.hypo_group_var = tk.StringVar()
        self.hypo_group_dropdown = ttk.Combobox(group_frame, textvariable=self.hypo_group_var,
                                               values=["All Engines", "First 50 Engines", "Last 50 Engines"])
        self.hypo_group_dropdown.pack(side=tk.LEFT, padx=5)
        self.hypo_group_dropdown.current(0)  # Default to All Engines
        
        # Hypothesis type selection
        hypo_frame = tk.Frame(control_frame)
        hypo_frame.pack(side=tk.LEFT, padx=10)
        
        hypo_label = tk.Label(hypo_frame, text="Hypothesis Test:")
        hypo_label.pack(side=tk.LEFT, padx=5)
        
        self.hypo_type_var = tk.StringVar()
        self.hypo_type_dropdown = ttk.Combobox(hypo_frame, textvariable=self.hypo_type_var,
                                              values=["Mean Comparison", "Variance Comparison"])
        self.hypo_type_dropdown.pack(side=tk.LEFT, padx=5)
        self.hypo_type_dropdown.current(0)  # Default to Mean Comparison
        
        # Run test button
        run_btn = tk.Button(self.hypo_tab, text="Run Hypothesis Test", 
                           command=self.run_hypothesis_test)
        run_btn.pack(pady=10)
        
        # Create a frame for plots
        self.hypo_plot_frame = tk.Frame(self.hypo_tab)
        self.hypo_plot_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create matplotlib figure for hypothesis test results
        self.hypo_fig, self.hypo_ax = plt.subplots(figsize=(10, 6))
        self.hypo_canvas = FigureCanvasTkAgg(self.hypo_fig, master=self.hypo_plot_frame)
        self.hypo_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Create text widget for hypothesis test report
        self.hypo_text = tk.Text(self.hypo_tab, height=15, wrap=tk.WORD)
        self.hypo_text.pack(fill=tk.BOTH, expand=True)
        self.hypo_text.tag_configure('header', foreground='blue', font=('Arial', 10, 'bold'))
        self.hypo_text.tag_configure('body', font=('Arial', 9))
    
    def run_hypothesis_test(self):
        """Run the selected hypothesis test"""
        try:
            # Get selected parameters
            sensor = self.hypo_sensor_var.get()
            group = self.hypo_group_var.get()
            hypo_type = self.hypo_type_var.get()
            
            # Load the data
            df = pd.read_csv('CMAPSSData/train_FD001.txt', delim_whitespace=True, header=None, 
                            names=["id", "time"] + [f"op_setting_{i}" for i in range(1, 4)] + 
                            [f"sensor_{i}" for i in range(1, 22)])
            
            # Filter data based on selected group
            if group == "First 50 Engines":
                df = df[df['id'] <= 50]
            elif group == "Last 50 Engines":
                df = df[df['id'] > 50]
            
            # Extract sensor data
            sensor_data = df[sensor].values
            
            # Clear previous plot
            self.hypo_ax.clear()
            
            # Perform hypothesis test based on selected type
            if hypo_type == "Mean Comparison":
                # Compare mean to a reference value (e.g., overall mean)
                reference_value = df[sensor].mean()
                
                # Performing one-sample t-test
                from scipy import stats
                t_stat, p_value = stats.ttest_1samp(sensor_data, reference_value)
                
                # Plot histogram with reference line
                self.hypo_ax.hist(sensor_data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                self.hypo_ax.axvline(reference_value, color='red', linestyle='--', linewidth=2, 
                                    label=f'Reference Mean: {reference_value:.2f}')
                
                # Generate report
                report = f"""One-Sample T-Test Results for {sensor} ({group}):

Null Hypothesis (H₀): The mean {sensor} value is equal to {reference_value:.2f}
Alternative Hypothesis (H₁): The mean {sensor} value is not equal to {reference_value:.2f}

Test Statistics:
- t-statistic: {t_stat:.4f}
- p-value: {p_value:.4f}0

Conclusion:"""
                
                if p_value < 0.05:
                    report += f"\nWe REJECT the null hypothesis at the 5% significance level. There is sufficient evidence to conclude that the mean {sensor} value is significantly different from {reference_value:.2f}."
                else:
                    report += f"\nWe FAIL TO REJECT the null hypothesis at the 5% significance level. There is insufficient evidence to conclude that the mean {sensor} value is significantly different from {reference_value:.2f}."
                
            elif hypo_type == "Variance Comparison":
                # Compare variance to a reference value (e.g., overall variance)
                reference_variance = df[sensor].var()
                
                # Perform chi-square test for variance
                from scipy import stats
                chi2_stat = (len(sensor_data) - 1) * np.var(sensor_data) / reference_variance
                p_value = 2 * min(stats.chi2.cdf(chi2_stat, len(sensor_data) - 1), 
                                1 - stats.chi2.cdf(chi2_stat, len(sensor_data) - 1)) #take the smaller tail probability lef/right
                
                # Plot histogram with reference lines
                self.hypo_ax.hist(sensor_data, bins=30, alpha=0.7, color='skyblue', edgecolor='black', density=True)
                
                # Add normal distribution curve with reference variance
                x = np.linspace(min(sensor_data), max(sensor_data), 100)
                mean = np.mean(sensor_data)
                std_ref = np.sqrt(reference_variance)
                y = stats.norm.pdf(x, mean, std_ref)
                self.hypo_ax.plot(x, y, 'r-', linewidth=2, 
                                 label=f'Reference Std Dev: {std_ref:.2f}')
                
                # Generate report
                report = f"""Chi-Square Test for Variance Results for {sensor} ({group}):

Null Hypothesis (H₀): The variance of {sensor} is equal to {reference_variance:.2f}
Alternative Hypothesis (H₁): The variance of {sensor} is not equal to {reference_variance:.2f}

Test Statistics:
- chi-square statistic: {chi2_stat:.4f}
- p-value: {p_value:.4f}

Conclusion:"""
                
                if p_value < 0.05:
                    report += f"\nWe REJECT the null hypothesis at the 5% significance level. There is sufficient evidence to conclude that the variance of {sensor} is significantly different from {reference_variance:.2f}."
                else:
                    report += f"\nWe FAIL TO REJECT the null hypothesis at the 5% significance level. There is insufficient evidence to conclude that the variance of {sensor} is significantly different from {reference_variance:.2f}."
            
            # Finalize plot
            self.hypo_ax.set_title(f"{hypo_type} for {sensor} ({group})")
            self.hypo_ax.set_xlabel(sensor)
            self.hypo_ax.set_ylabel("Frequency")
            self.hypo_ax.legend()
            self.hypo_canvas.draw()
            
            # Display report
            self.hypo_text.delete(1.0, tk.END)
            self.hypo_text.insert(tk.END, "Hypothesis Test Report:\n", 'header')
            self.hypo_text.insert(tk.END, report, 'body')
            
        except Exception as e:
            self.hypo_text.delete(1.0, tk.END)
            self.hypo_text.insert(tk.END, f"Error during hypothesis testing: {str(e)}", 'body')

    def init_regression_tab(self):
        # Header for the regression tab
        header = tk.Label(self.ml_tab, text="Regression Model Analysis", font=("Arial", 14))
        header.pack(pady=10)
        
        # Create a frame for controls
        control_frame = tk.Frame(self.ml_tab)
        control_frame.pack(pady=10, fill=tk.X)
        
        # Target sensor selection
        sensor_frame = tk.Frame(control_frame)
        sensor_frame.pack(side=tk.LEFT, padx=10)
        
        sensor_label = tk.Label(sensor_frame, text="Target Sensor:")
        sensor_label.pack(side=tk.LEFT, padx=5)
        
        self.reg_sensor_var = tk.StringVar()
        self.reg_sensor_dropdown = ttk.Combobox(sensor_frame, textvariable=self.reg_sensor_var,
                                               values=[f"sensor_{i}" for i in range(1, 22)])
        self.reg_sensor_dropdown.pack(side=tk.LEFT, padx=5)
        self.reg_sensor_dropdown.current(7)  # Default to sensor_8
        
        # Model selection
        model_frame = tk.Frame(control_frame)
        model_frame.pack(side=tk.LEFT, padx=10)
        
        model_label = tk.Label(model_frame, text="Model Type:")
        model_label.pack(side=tk.LEFT, padx=5)
        
        self.reg_model_var = tk.StringVar()
        self.reg_model_dropdown = ttk.Combobox(model_frame, textvariable=self.reg_model_var,
                                              values=["All Models", "Linear", "Ridge", "Lasso"])
        self.reg_model_dropdown.pack(side=tk.LEFT, padx=5)
        self.reg_model_dropdown.current(0)  # Default to All Models
        
        # Button to run the regression analysis
        run_btn = tk.Button(self.ml_tab, text="Run Regression Analysis", command=self.run_regression_analysis)
        run_btn.pack(pady=10)
        
        # Create a notebook for different views
        self.reg_notebook = ttk.Notebook(self.ml_tab)
        self.reg_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs for different views
        self.reg_plot_tab = ttk.Frame(self.reg_notebook)
        self.reg_comparison_tab = ttk.Frame(self.reg_notebook)
        
        # Add tabs to notebook
        self.reg_notebook.add(self.reg_plot_tab, text="Model Results")
        self.reg_notebook.add(self.reg_comparison_tab, text="Model Comparison")
        
        # Frame to hold the regression plot and text report side-by-side
        self.reg_frame = tk.Frame(self.reg_plot_tab)
        self.reg_frame.pack(fill=tk.BOTH, expand=True)
        
        # Matplotlib figure for regression results
        self.reg_fig, self.reg_ax = plt.subplots(figsize=(8, 6))
        self.reg_canvas = FigureCanvasTkAgg(self.reg_fig, master=self.reg_frame)
        self.reg_canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Text widget to display regression analysis report
        self.reg_text = tk.Text(self.reg_frame, width=40, wrap=tk.WORD)
        self.reg_text.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.reg_text.tag_configure('header', foreground='blue', font=('Arial', 10, 'bold'))
        self.reg_text.tag_configure('body', font=('Arial', 9))
        
        # Frame for comparison tab
        self.comparison_frame = tk.Frame(self.reg_comparison_tab)
        self.comparison_frame.pack(fill=tk.BOTH, expand=True)
        
        # Matplotlib figure for comparison results
        self.comparison_fig, self.comparison_ax = plt.subplots(figsize=(10, 6))
        self.comparison_canvas = FigureCanvasTkAgg(self.comparison_fig, master=self.comparison_frame)
        self.comparison_canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Text widget to display comparison report
        self.comparison_text = tk.Text(self.comparison_frame, width=40, wrap=tk.WORD)
        self.comparison_text.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.comparison_text.tag_configure('header', foreground='blue', font=('Arial', 10, 'bold'))
        self.comparison_text.tag_configure('body', font=('Arial', 9))
    
    def run_regression_analysis(self):
        try:
            # Get selected parameters
            target_sensor = int(self.reg_sensor_var.get().split('_')[1])
            model_type = self.reg_model_var.get()
            
            # Instantiate the regression model
            reg_model = model()
            
            # Load the data using the model's method
            df_reg = reg_model.load_data('CMAPSSData/train_FD001.txt')
            
            # Prepare the data
            X_train, X_test, y_train, y_test, feature_names = reg_model.prepare_data(df_reg, target_sensor=target_sensor)
            
            # Train the model on the training data
            reg_model.train(X_train, y_train)
            
            # Evaluate the model on the test data
            results = reg_model.evaluate(X_test, y_test)
            
            # Get model comparison
            comparison = reg_model.compare_models()
            
            # Display results based on selected model type
            if model_type == "All Models":
                # Switch to comparison tab
                self.reg_notebook.select(1)
                
                # Plot comparison results
                self.comparison_ax.clear()
                
                # Create bar chart comparing all metrics for all models
                model_names = list(results.keys())
                metrics = ['RMSE', 'R2', 'MAE', 'MSE', 'StdDev']
                
                x = np.arange(len(model_names))
                width = 0.15  # Reduced width to accommodate more bars
                
                for i, metric in enumerate(metrics):
                    values = [results[model][metric] for model in model_names]
                    self.comparison_ax.bar(x + i*width, values, width, label=metric)
                
                self.comparison_ax.set_xlabel('Model')
                self.comparison_ax.set_ylabel('Metric Value')
                self.comparison_ax.set_title(f'Model Performance Comparison for {self.reg_sensor_var.get()}')
                self.comparison_ax.set_xticks(x + width*2)  # Center the x-ticks
                self.comparison_ax.set_xticklabels(model_names)
                self.comparison_ax.legend()
                
                self.comparison_canvas.draw()
                
                # Display comparison report
                self.comparison_text.delete(1.0, tk.END)
                self.comparison_text.insert(tk.END, "Model Comparison Report:\n", 'header')
                self.comparison_text.insert(tk.END, comparison['analysis'], 'body')
                
                # Also show individual model results in the plot tab
                self.reg_ax.clear()
                
                # Plot Actual vs Predicted for all models
                for model_name, model_results in results.items():
                    self.reg_ax.scatter(model_results['y_true'], model_results['y_pred'], 
                                       alpha=0.5, label=f'{model_name} Predictions')
                
                min_val = min(min(results[model]['y_true']) for model in results)
                max_val = max(max(results[model]['y_true']) for model in results)
                self.reg_ax.plot([min_val, max_val], [min_val, max_val], 'k--', label='Ideal Fit')
                
                self.reg_ax.set_xlabel('Actual Values')
                self.reg_ax.set_ylabel('Predicted Values')
                self.reg_ax.set_title(f'Actual vs Predicted Values for {self.reg_sensor_var.get()}')
                self.reg_ax.legend()
                
                self.reg_canvas.draw()
                
                # Display summary of all models
                report = f"""Regression Analysis Summary for {self.reg_sensor_var.get()}:

Linear Regression:
- Mean Squared Error: {results['Linear']['MSE']:.4f}
- Root Mean Squared Error: {results['Linear']['RMSE']:.4f}
- R-squared: {results['Linear']['R2']:.4f}
- Mean Absolute Error: {results['Linear']['MAE']:.4f}
- Standard Deviation: {results['Linear']['StdDev']:.4f}

Ridge Regression:
- Mean Squared Error: {results['Ridge']['MSE']:.4f}
- Root Mean Squared Error: {results['Ridge']['RMSE']:.4f}
- R-squared: {results['Ridge']['R2']:.4f}
- Mean Absolute Error: {results['Ridge']['MAE']:.4f}
- Standard Deviation: {results['Ridge']['StdDev']:.4f}

Lasso Regression:
- Mean Squared Error: {results['Lasso']['MSE']:.4f}
- Root Mean Squared Error: {results['Lasso']['RMSE']:.4f}
- R-squared: {results['Lasso']['R2']:.4f}
- Mean Absolute Error: {results['Lasso']['MAE']:.4f}
- Standard Deviation: {results['Lasso']['StdDev']:.4f}

Best Model: {comparison['best_model']['RMSE']} (based on RMSE)
"""
                
                self.reg_text.delete(1.0, tk.END)
                self.reg_text.insert(tk.END, "Regression Analysis Report:\n", 'header')
                self.reg_text.insert(tk.END, report, 'body')
                
            else:
                # Switch to plot tab
                self.reg_notebook.select(0)
                
                # Get results for selected model
                model_results = results[model_type]
                
                # Get feature importance
                feat_importance = reg_model.get_feature_importance(model_type, feature_names)
                
                # Plot Actual vs Predicted values
                self.reg_ax.clear()
                self.reg_ax.scatter(model_results['y_true'], model_results['y_pred'], alpha=0.5, label='Predictions')
                self.reg_ax.plot([min(model_results['y_true']), max(model_results['y_true'])],
                                 [min(model_results['y_true']), max(model_results['y_true'])], 'r--', label='Ideal Fit')
                self.reg_ax.set_title(f"{model_type} Regression: Actual vs. Predicted {self.reg_sensor_var.get()}")
                self.reg_ax.set_xlabel("Actual Value")
                self.reg_ax.set_ylabel("Predicted Value")
                self.reg_ax.legend()
                self.reg_canvas.draw()
                
                # Generate a detailed analysis report
                report = f"""{model_type} Regression Model Evaluation for {self.reg_sensor_var.get()}:
- Mean Squared Error: {model_results['MSE']:.4f}
- Root Mean Squared Error: {model_results['RMSE']:.4f}
- R-squared: {model_results['R2']:.4f}
- Mean Absolute Error: {model_results['MAE']:.4f}
- Standard Deviation: {model_results['StdDev']:.4f}

Top 5 Feature Importances:"""
                for feat, imp in list(feat_importance.items())[:5]:
                    report += f"\n   {feat}: {imp:.4f}"
                
                # Display the report in the text widget
                self.reg_text.delete(1.0, tk.END)
                self.reg_text.insert(tk.END, report, 'body')
            
        except Exception as e:
            self.reg_text.delete(1.0, tk.END)
            self.reg_text.insert(tk.END, f"Error during regression analysis: {str(e)}", 'body')
    
    def init_health_tab(self):
        """Initialize the engine health prediction tab"""
        # Header for the health tab
        header = tk.Label(self.health_tab, text="Engine Health Prediction", font=("Arial", 14))
        header.pack(pady=10)
        
        # Create a frame for buttons
        button_frame = tk.Frame(self.health_tab)
        button_frame.pack(pady=10)
        
        # Button to analyze correlations
        corr_btn = tk.Button(button_frame, text="Analyze Correlations", 
                            command=self.analyze_correlations)
        corr_btn.pack(side=tk.LEFT, padx=5)
        
        # Button to predict engine health
        health_btn = tk.Button(button_frame, text="Predict Engine Health", 
                              command=self.predict_engine_health)
        health_btn.pack(side=tk.LEFT, padx=5)
        
        # Create a frame for plots
        self.health_plot_frame = tk.Frame(self.health_tab)
        self.health_plot_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create a notebook for different plots
        self.health_plot_notebook = ttk.Notebook(self.health_plot_frame)
        self.health_plot_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs for different plots
        self.corr_plot_tab = ttk.Frame(self.health_plot_notebook)
        self.health_plot_tab = ttk.Frame(self.health_plot_notebook)
        
        # Add tabs to notebook
        self.health_plot_notebook.add(self.corr_plot_tab, text="Correlation Matrix")
        self.health_plot_notebook.add(self.health_plot_tab, text="Health Scores")
        
        # Create canvas for correlation plot
        self.corr_fig, self.corr_ax = plt.subplots(figsize=(10, 8))
        self.corr_canvas = FigureCanvasTkAgg(self.corr_fig, master=self.corr_plot_tab)
        self.corr_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Create canvas for health plot
        self.health_fig, self.health_ax = plt.subplots(figsize=(10, 6))
        self.health_canvas = FigureCanvasTkAgg(self.health_fig, master=self.health_plot_tab)
        self.health_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Create text widget for analysis report
        self.health_text = tk.Text(self.health_tab, height=8, wrap=tk.WORD)
        self.health_text.pack(fill=tk.BOTH, expand=True)
        self.health_text.tag_configure('header', foreground='blue', font=('Arial', 10, 'bold'))
        self.health_text.tag_configure('body', font=('Arial', 9))
        
        # Initialize the health model
        self.health_model = health_model()
    
    def analyze_correlations(self):
        """Analyze correlations between features"""
        try:
            # Load the data
            df = self.health_model.load_data('CMAPSSData/train_FD001.txt')
            
            # Analyze correlations
            correlation_matrix = self.health_model.analyze_correlations(df)
            
            # Get top correlations
            top_correlations = self.health_model.get_top_correlations(n=10)
            
            # Visualize correlations
            self.corr_ax.clear()
            sns.heatmap(correlation_matrix, ax=self.corr_ax, annot=False, cmap='coolwarm', linewidths=0.5)
            self.corr_ax.set_title('Feature Correlation Matrix')
            self.corr_canvas.draw()
            
            # Switch to correlation tab
            self.health_plot_notebook.select(0)
            
            # Generate report
            report = "Top 10 Feature Correlations:\n\n"
            for feat1, feat2, corr in top_correlations:
                report += f"{feat1} - {feat2}: {corr:.4f}\n"
            
            # Display report
            self.health_text.delete(1.0, tk.END)
            self.health_text.insert(tk.END, "Correlation Analysis Report:\n", 'header')
            self.health_text.insert(tk.END, report, 'body')
            
        except Exception as e:
            self.health_text.delete(1.0, tk.END)
            self.health_text.insert(tk.END, f"Error during correlation analysis: {str(e)}", 'body')
    
    def predict_engine_health(self):
        """Predict engine health scores"""
        try:
            # Load the data
            df = self.health_model.load_data('CMAPSSData/train_FD001.txt')
            
            # Prepare data for health prediction
            X_pca, engine_ids = self.health_model.prepare_data(df)
            
            # Calculate health scores
            health_scores = self.health_model.calculate_health_scores(X_pca)
            
            # Train the model
            self.health_model.train(X_pca, health_scores)
            
            # Visualize health scores
            self.health_ax.clear()
            self.health_ax.bar(engine_ids, health_scores)
            self.health_ax.set_xlabel('Engine ID')
            self.health_ax.set_ylabel('Health Score (0-100)')
            self.health_ax.set_title('Engine Health Scores')
            self.health_ax.tick_params(axis='x', rotation=90)
            self.health_canvas.draw()
            
            # Switch to health tab
            self.health_plot_notebook.select(1)
            
            # Generate report
            report = f"""Engine Health Analysis:
- Total Engines Analyzed: {len(engine_ids)}
- Average Health Score: {np.mean(health_scores):.2f}
- Minimum Health Score: {np.min(health_scores):.2f}
- Maximum Health Score: {np.max(health_scores):.2f}
- Standard Deviation: {np.std(health_scores):.2f}

Engines with Lowest Health Scores (Top 5):"""
            
            # Get engines with lowest health scores
            lowest_health_indices = np.argsort(health_scores)[:5]
            for i, idx in enumerate(lowest_health_indices):
                report += f"\n   {i+1}. Engine {engine_ids[idx]}: {health_scores[idx]:.2f}"
            
            # Display report
            self.health_text.delete(1.0, tk.END)
            self.health_text.insert(tk.END, "Engine Health Analysis Report:\n", 'header')
            self.health_text.insert(tk.END, report, 'body')
            
        except Exception as e:
            self.health_text.delete(1.0, tk.END)
            self.health_text.insert(tk.END, f"Error during engine health prediction: {str(e)}", 'body')

if __name__ == "__main__":
    root = Tk()
    app = PredictiveMaintenanceApp(root)
    root.mainloop()