import numpy as np

def calculate_cost(glucose, glutamine,vcd):
    cost = 10* glucose + 20* glutamine + 0.0005*vcd
    return cost
def calculate_value(mab):
    value = 1000 * mab
    return value
def calculate_profit(value,cost):
    profit = value - cost
    return profit
def calculate_space_time_profit(profit, harvest_time):
    st_profit = profit / (harvest_time  + 1)
    return st_profit
def calculate_space_time_yield(mab, harvest_time):
    st_yield = mab / (harvest_time + 1)
    return st_yield
def generate_x_lag_cur_var(p, batch_ids, df, features, col_batch="Batch ID", col_time="Time [h]"):
    x_lag = []
    x_cur = []
    t_cur = []
    b_cur = []

    for batch_id in batch_ids:

        mask = df.loc[:, col_batch] == batch_id

        x_batch = df.loc[mask, features].to_numpy()
        t_batch = df.loc[mask, col_time].to_numpy()
        b_batch = df.loc[mask, col_batch].to_numpy()

        current_values = x_batch[p:]
        current_times = t_batch[p:]
        current_batches = b_batch[p:]

        lagged_values = []

        for i in range(1, p + 1):

            lagged_values.append(
                x_batch[p - i:-i]
            )

        lagged_values = np.concatenate(
            lagged_values,
            axis=1
        )

        x_lag.append(lagged_values)
        x_cur.append(current_values)
        t_cur.append(current_times)
        b_cur.append(current_batches)

    x_lag = np.concatenate(x_lag, axis=0)
    x_cur = np.concatenate(x_cur, axis=0)
    t_cur = np.concatenate(t_cur, axis=0)
    b_cur = np.concatenate(b_cur, axis=0)

    return x_lag, x_cur, t_cur, b_cur