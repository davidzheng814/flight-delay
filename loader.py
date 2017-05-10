import numpy as np
import pandas as pd
import networkx as nx
import datetime
import math

#TODO FIX TIME ZONES
#TODO add katz

def load_data(skiprows=0, nrows=None):
    if nrows:
        flights = pd.read_csv('flights.csv', skiprows=range(1, skiprows), nrows=nrows)
    else:
        flights = pd.read_csv('flights.csv')

    flights = flights[flights['DEPARTURE_DELAY'].notnull()]
    return flights

def row_to_dt(x, depart=True):
    col = 'SCHEDULED_DEPARTURE' if depart else 'SCHEDULED_ARRIVAL'
    return datetime.datetime(x['YEAR'], x['MONTH'], x['DAY'], x[col]/100, x[col]%100)

def get_delay_times(flights, year, month, day, hour, min_delay=60):
    def is_valid_flight(x, start_dt):
        if (len(str(x['ORIGIN_AIRPORT'])) != 3 or
            len(str(x['DESTINATION_AIRPORT'])) != 3):
            return False

        dep_dt = row_to_dt(x, depart=True)
        if dep_dt > start_dt and x['DEPARTURE_DELAY'] >= min_delay:
            return True

        return False

    # consider delays after start_dt.
    start_dt = datetime.datetime(year, month, day, hour) + datetime.timedelta(hours=1)

    flights = flights[flights.apply(lambda x: is_valid_flight(x, start_dt), axis=1)]
    flights['TIMES'] = flights.apply(lambda x: (row_to_dt(x) - start_dt).seconds/60 + 1, axis=1)
    flights = flights[['ORIGIN_AIRPORT', 'TIMES']]
    flights = flights.groupby('ORIGIN_AIRPORT').min().reset_index()

    return {x['ORIGIN_AIRPORT']:x['TIMES'] for i, x in flights.iterrows()}

def load_graph(flights, year, month, day, hour):
    def is_valid_flight(x, start_window):
        if (len(str(x['ORIGIN_AIRPORT'])) != 3 or
            len(str(x['DESTINATION_AIRPORT'])) != 3):
            return False

        arr_dt = row_to_dt(x, depart=False)
        duration = datetime.timedelta(minutes=x['SCHEDULED_TIME'])
        dep_dt = arr_dt - duration

        end_window = start_window + datetime.timedelta(hours=1)
        if arr_dt < start_window or dep_dt > end_window:
            return False

        return True

    start_window = datetime.datetime(year, month, day, hour)
    flights = flights[flights.apply(lambda x:is_valid_flight(x, start_window), axis=1)]

    flights = flights[['ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 'DEPARTURE_DELAY']]
    delay_avgs = (flights
                  .groupby(['ORIGIN_AIRPORT', 'DESTINATION_AIRPORT'])
                  .agg(['mean', 'count', 'sum', 'std'])
                  .reset_index())
    delay_avgs.columns = [' '.join(col).strip() for col in delay_avgs.columns.values]
    delay_avgs = delay_avgs.rename(
        columns={'DEPARTURE_DELAY mean': 'MEAN', 
                 'DEPARTURE_DELAY count': 'COUNT',
                 'DEPARTURE_DELAY sum': 'SUM',
                 'DEPARTURE_DELAY std': 'STD'})

    G = nx.from_pandas_dataframe(
        delay_avgs, 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', edge_attr=['MEAN', 'COUNT', 'SUM', 'STD'],
        create_using=nx.DiGraph())

    return G

def get_feature_vecs(G, year, month, day, hour):
    time = datetime.datetime(year, month, day, hour)

    features = {}
    month_vec = np.zeros(12)
    month_vec[month - 1] = 1
    day_vec = np.zeros(7)
    day_vec[time.weekday()] = 1
    hour_vec = np.zeros(24)
    hour_vec[hour] = 1
    time_vec = np.concatenate((month_vec, day_vec, hour_vec))

    for node in G.nodes():
        in_flights = G.in_degree(node, weight='COUNT')
        out_flights = G.out_degree(node, weight='COUNT')
        mean_in_delay = G.in_degree(node, weight='MEAN')
        mean_out_delay = G.out_degree(node, weight='MEAN')

        std_in_delay = G.in_degree(node, weight='STD')
        if math.isnan(std_in_delay):
            std_in_delay = 0
        std_out_delay = G.out_degree(node, weight='STD')
        if math.isnan(std_out_delay):
            std_out_delay = 0

        tot_in_delay = G.in_degree(node, weight='SUM')
        tot_out_delay = G.out_degree(node, weight='SUM')

        features[node] = [in_flights, out_flights, mean_in_delay, mean_out_delay,
            std_in_delay, std_out_delay, tot_in_delay, tot_out_delay]

    betweenness = nx.betweenness_centrality(G)
    # katz = nx.katz_centrality(G, alpha=0.01, max_iter=1000)
    # weighted_katz = nx.katz_centrality(G, weight='SUM')
    clustering = nx.clustering(nx.Graph(G))
    hubs, auth = nx.hits(G, max_iter=500)

    for node in G.nodes():
        features[node].extend([betweenness[node], 
            # katz[node], weighted_katz[node], 
            clustering[node], hubs[node], auth[node]])
        arr = np.array(features[node])
        features[node] = np.concatenate((arr, time_vec))

    return features

def save_feature_vecs():
    flights = load_data(skiprows=100000, nrows=100000)
    cur_date = datetime.date(flights['YEAR'][0], flights['MONTH'][0], flights['DAY'][0])
    end_date = datetime.date(flights['YEAR'].tail(1), flights['MONTH'].tail(1), flights['DAY'].tail(1))
    while cur_date < end_date:
        print cur_date
        for hour in range(24):
            print hour
            G = load_graph(flights, cur_date.year, cur_date.month, cur_date.day, hour)
            X = get_feature_vecs(G, cur_date.year, cur_date.month, cur_date.day, hour)
            Y = get_delay_times(flights, cur_date.year, cur_date.month, cur_date.day, hour)

            Y = {airport:Y[airport] for airport in X.keys() if airport in Y}

            np.savez_compressed('data/X_'+str(cur_date)+'_'+str(hour)+'.npz', **X)
            np.savez_compressed('data/Y_'+str(cur_date)+'_'+str(hour)+'.npz', **Y)

        cur_date += datetime.timedelta(days=1)

def get_correlation(G): 
    adj = nx.adjacency_matrix(G) 
    col = adj.shape[1]

    # need to do the math by hand because the matrix is sparse and numpy yells at you 
    C = ((adj.T*adj -(sum(adj).T*sum(adj)/col))/(col-1)).todense()
    V = np.sqrt(np.mat(np.diag(C)).T*np.mat(np.diag(C)))
    epsilon = 1e-119
    
    cov_matrix = np.divide(C, V+epsilon)

    return cov_matrix

if __name__ == '__main__':
    save_feature_vecs()
