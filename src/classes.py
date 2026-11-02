
import os

import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
from matplotlib.ticker import MultipleLocator
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import root_mean_squared_error
from src.functions import *
from src.constants import *



def mab_rmse(y_true, y_pred):
    return root_mean_squared_error(y_true[:, mAb_index], y_pred[:, mAb_index])


class Batch:

    def __init__(self):
        self.df_ic = pd.DataFrame()
        self.df = pd.DataFrame()
        self.batch_ids = []
        self.df_optimal = pd.DataFrame()
        self.features = []
        self.p = Default_lag_order
        self.x_lag = []
        self.x_cur = []
        self.t_cur = []
        self.b_cur = []
        self.df_model = pd.DataFrame()
        self.df_model_comparison = pd.DataFrame()
        self.n_simulations = 0
        self.p_max = Max_lag_order
        self.cv_splits = CV_splits
        self.poly_degree_max = Max_poly_degree
        self.knn_k_max = Max_knn_neighbors
        self.model_var = None
        self.x_cur_pred = None

    def load_data(self, randomize=False, random_state=42):
        self.df_ic = pd.read_csv(os.path.join("datasets", "initial_conditions (1).csv"))
        self.df = pd.DataFrame()

        available_batches = []

        for filename in os.listdir("datasets"):
            if filename.startswith("batch_") and filename.endswith(".csv"):
                batch_id = int(filename.replace("batch_", "").replace(".csv", ""))
                available_batches.append(batch_id)

        if randomize:
            self.batch_ids = np.random.default_rng(random_state).choice(available_batches, size=N_batches,replace=False)
        else:
            self.batch_ids = np.array(sorted(available_batches)[:N_batches])

        self.batch_ids = np.sort(self.batch_ids)

        for batch_id in self.batch_ids:
            df_batch = pd.read_csv(os.path.join("datasets", f"batch_{batch_id:04d}.csv"))
            df_batch["Batch ID"] = batch_id
            self.df = pd.concat([self.df, df_batch], ignore_index=True)

        self.df_ic = self.df_ic[self.df_ic["Batch ID"].isin(self.batch_ids)]

    def calculate_metrics(self):
        self.df["Harvest Time [d]"] = self.df["Time [h]"] / 24
        self.df = self.df.merge(self.df_ic, on="Batch ID", suffixes=("", "_Initial"))
        self.df["Cost [$/L]"] = calculate_cost(self.df["Glucose [mM]_Initial"], self.df["Glutamine [mM]_Initial"], self.df["VCD [cells/mL]_Initial"])
        self.df["Value [$/L]"] = calculate_value(self.df["mAb [mg/L]"])
        self.df["Profit [$/L]"] = calculate_profit(self.df["Value [$/L]"], self.df["Cost [$/L]"])
        self.df["Space-Time Profit"] = calculate_space_time_profit(self.df["Profit [$/L]"], self.df["Harvest Time [d]"])
        self.df["Space-Time Yield"] = calculate_space_time_yield(self.df["mAb [mg/L]"], self.df["Harvest Time [d]"])

    def find_optimal_harvest(self):
        idx = self.df.groupby("Batch ID")["Space-Time Profit"].idxmax()
        self.df_optimal = self.df.loc[idx].copy()

    def display_optimal(self):
        print(self.df_optimal[["Batch ID", "Glucose [mM]_Initial", "Glutamine [mM]_Initial", "VCD [cells/mL]_Initial", "Harvest Time [d]", "Space-Time Profit"]])

    def summarize_results(self):
        print(self.df_optimal.describe())

    def lagged_values(self):
        self.features = Features
        self.p = Default_lag_order
        self.x_lag, self.x_cur, self.t_cur, self.b_cur = generate_x_lag_cur_var(p=self.p, batch_ids=self.batch_ids, df=self.df, features=self.features)

    def var(self):
        self.model_var = LinearRegression()
        self.model_var.fit(self.x_lag, self.x_cur)
        self.x_cur_pred = self.model_var.predict(self.x_lag)

    def nonlinear_var(self):
        self.p_max = Max_lag_order
        self.cv_splits = CV_splits
        self.poly_degree_max = Max_poly_degree
        self.knn_k_max = Max_knn_neighbors

        self.lagged_values()

        X = self.x_lag
        y = self.x_cur
        groups = self.b_cur
        results = []
        cv = GroupKFold(n_splits=self.cv_splits)

        for degree in range(2, Max_poly_degree + 1):
            model = make_pipeline(PolynomialFeatures(degree=degree, include_bias=False), StandardScaler(), LinearRegression())
            scores = []

            for train_index, test_index in cv.split(X, y, groups):
                model.fit(X[train_index], y[train_index])
                y_pred = model.predict(X[test_index])
                scores.append(root_mean_squared_error(y[test_index, mAb_index], y_pred[:, mAb_index]))

            results.append({"Model": "Polynomial Regression", "Parameter": degree, "RMSE": np.mean(scores)})

        for k in range(2, Max_knn_neighbors + 1):
            model = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=k))
            scores = []

            for train_index, test_index in cv.split(X, y, groups):
                model.fit(X[train_index], y[train_index])
                y_pred = model.predict(X[test_index])
                scores.append(root_mean_squared_error(y[test_index, mAb_index], y_pred[:, mAb_index]))

            results.append({"Model": "KNN", "Parameter": k, "RMSE": np.mean(scores)})

        self.df_model_comparison = pd.DataFrame(results)
        self.df_model_comparison = self.df_model_comparison.sort_values("RMSE")

        os.makedirs("tables", exist_ok=True)
        self.df_model_comparison.to_csv(os.path.join("tables", "model_comparison.csv"), index=False)

    def best_model(self):
        best = self.df_model_comparison.iloc[0]
        print(f"Best Model: {best['Model']}")
        print(f"Best Parameter: {best['Parameter']}")
        print(f"Lag Order: {self.p}")
        print(f"Best mAb RMSE: {best['RMSE']}")


    def residual_diagnostics(self):
        p = Default_lag_order
        degree = Default_poly_degree
        features = Features
        X, y, t, groups = generate_x_lag_cur_var(p=p, batch_ids=self.batch_ids, df=self.df, features=features)

        cv = GroupKFold(n_splits = CV_splits)
        model = make_pipeline(PolynomialFeatures(degree=degree, include_bias=False), StandardScaler(), LinearRegression())
        y_pred = cross_val_predict(model, X, y, groups=groups, cv=cv)

        residuals = y[:, mAb_index] - y_pred[:, mAb_index]
        predicted = y_pred[:, mAb_index]

        os.makedirs("figures", exist_ok=True)
        os.makedirs("tables", exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(predicted, residuals, alpha=0.6)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Predicted mAb [mg/L]")
        ax.set_ylabel("Residual mAb [mg/L]")
        ax.set_title("Residuals vs Predicted mAb")
        fig.tight_layout()
        fig.savefig("figures/residuals_vs_predicted.png", dpi=300)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(residuals, bins=20, edgecolor="black")
        ax.set_xlabel("Residual mAb [mg/L]")
        ax.set_ylabel("Frequency")
        ax.set_title("Distribution of mAb Residuals")
        fig.tight_layout()
        fig.savefig("figures/residual_histogram.png", dpi=300)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 6))
        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title("Q-Q Plot of mAb Residuals")
        fig.tight_layout()
        fig.savefig("figures/residual_qq_plot.png", dpi=300)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(t, residuals, alpha=0.6)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Time [h]")
        ax.set_ylabel("Residual mAb [mg/L]")
        ax.set_title("mAb Residuals vs Time")
        ax.xaxis.set_major_locator(MultipleLocator(72))
        fig.tight_layout()
        fig.savefig("figures/residuals_vs_time.png", dpi=300)
        plt.close(fig)

        results = pd.DataFrame([{"Model": "Polynomial Regression","Polynomial Degree": degree,"Lag Order": p,"RMSE [mg/L]": root_mean_squared_error(y[:, mAb_index], y_pred[:, mAb_index]),"Mean Absolute Error [mg/L]": np.mean(np.abs(residuals)),"Mean Residual [mg/L]": np.mean(residuals),"Residual Standard Deviation [mg/L]": np.std(residuals, ddof=1)}])

        results.to_csv("tables/residual_statistics.csv", index=False)

    def final_recommendation(self):
        p = Default_lag_order
        degree = Default_poly_degree
        features = Features
        X, y, t, groups = generate_x_lag_cur_var(p=p, batch_ids=self.batch_ids, df=self.df, features=features)

        cv = GroupKFold(n_splits= CV_splits)
        model = make_pipeline(PolynomialFeatures(degree=degree, include_bias=False), StandardScaler(), LinearRegression())
        y_pred = cross_val_predict(model, X, y, groups=groups, cv=cv)

        residuals = y[:, mAb_index] - y_pred[:, mAb_index]
        absolute_residuals = np.abs(residuals)
        prediction_error = np.quantile(absolute_residuals, Prediction_interval)
        best_idx = self.df_optimal["Space-Time Profit"].idxmax()
        best = self.df_optimal.loc[best_idx]

        glucose = best["Glucose [mM]_Initial"]
        glutamine = best["Glutamine [mM]_Initial"]
        vcd = best["VCD [cells/mL]_Initial"]
        harvest_time = best["Harvest Time [d]"]
        mab = best["mAb [mg/L]"]

        mab_lower = max(0, mab - prediction_error)
        mab_upper = mab + prediction_error

        cost = calculate_cost(glucose, glutamine, vcd)
        value = calculate_value(mab)
        value_lower = calculate_value(mab_lower)
        value_upper = calculate_value(mab_upper)

        st_profit = calculate_space_time_profit(value - cost, harvest_time)
        st_profit_lower = calculate_space_time_profit(value_lower - cost, harvest_time)
        st_profit_upper = calculate_space_time_profit(value_upper - cost, harvest_time)

        st_yield = calculate_space_time_yield(mab, harvest_time)
        st_yield_lower = calculate_space_time_yield(mab_lower, harvest_time)
        st_yield_upper = calculate_space_time_yield(mab_upper, harvest_time)

        results = pd.DataFrame([{"Batch ID": best["Batch ID"],"Initial Glucose [mM]": glucose,"Initial Glutamine [mM]": glutamine,"Initial VCD [cells/mL]": vcd,"Harvest Time [d]": harvest_time,"Cell Culture Cost [$/L]": cost,"mAb [mg/L]": mab,"mAb Lower [mg/L]": mab_lower,"mAb Upper [mg/L]": mab_upper,"Cell Culture Value [$/L]": value,"Value Lower [$/L]": value_lower,"Value Upper [$/L]": value_upper,"Space-Time Profit [$/(L d)]": st_profit,"Profit Lower [$/(L d)]": st_profit_lower, "Profit Upper [$/(L d)]": st_profit_upper, "Space-Time Yield [mg/(L d)]": st_yield, "Yield Lower [mg/(L d)]": st_yield_lower, "Yield Upper [mg/(L d)]": st_yield_upper}])

        os.makedirs("tables", exist_ok=True)
        results.to_csv("tables/final_recommendation.csv", index=False)


    def recommend_best_batch(self):
        best_idx = self.df_optimal["Space-Time Profit"].idxmax()
        best = self.df_optimal.loc[best_idx]

        columns = ["Batch ID", "Glucose [mM]_Initial", "Glutamine [mM]_Initial", "VCD [cells/mL]_Initial", "Harvest Time [d]", "Cost [$/L]", "Value [$/L]", "Space-Time Profit", "Space-Time Yield"]
        print(best[columns].to_string())

        os.makedirs("tables", exist_ok=True)
        best[columns].to_frame().T.to_csv(os.path.join("tables", "best_batch_recommendation.csv"), index=False)
