from src.classes import Batch

batch = Batch()

batch.load_data(randomize=True)
batch.calculate_metrics()
batch.find_optimal_harvest()

batch.display_optimal()
batch.summarize_results()
batch.recommend_best_batch()

batch.lagged_values()
print("Lagged variables:", batch.x_lag.shape)
print("Current variables:", batch.x_cur.shape)

batch.var()
batch.nonlinear_var()
batch.best_model()
batch.residual_diagnostics()
batch.final_recommendation()