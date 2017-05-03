import numpy as np
import pandas as pd
import networkx as nx
import datetime

def is_valid_flight(x, start_window):
    if (len(str(x['ORIGIN_AIRPORT'])) != 3 or
        len(str(x['DESTINATION_AIRPORT'])) != 3):
        return False

    arr_time = x['SCHEDULED_ARRIVAL']
    arr_dt = datetime.datetime(x['YEAR'], x['MONTH'], x['DAY'], arr_time/100, arr_time%100)
    duration = datetime.timedelta(minutes=x['SCHEDULED_TIME'])
    dep_dt = arr_dt - duration

    end_window = start_window + datetime.timedelta(hours=1)
    if arr_dt < start_window or dep_dt > end_window:
        return False

    return True

def load_data(nrows=5000):
    return pd.read_csv('flights.csv', nrows=nrows)

def load_graph(flights, year, month, day, hour):
    start_window = datetime.datetime(year, month, day, hour)
    flights = flights[flights.apply(lambda x:is_valid_flight(x, start_window), axis=1)]

    flights = flights[['ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 'DEPARTURE_DELAY']]
    flights = flights[flights['DEPARTURE_DELAY'].notnull()]
    delay_avgs = (flights
                  .groupby(['ORIGIN_AIRPORT', 'DESTINATION_AIRPORT'])
                  .agg(['mean', 'count'])
                  .reset_index())
    delay_avgs.columns = [' '.join(col).strip() for col in delay_avgs.columns.values]
    delay_avgs = delay_avgs.rename(
        columns={'DEPARTURE_DELAY mean': 'MEAN_DELAY', 'DEPARTURE_DELAY count': 'COUNT'})

    G = nx.from_pandas_dataframe(
        delay_avgs, 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', edge_attr=['MEAN_DELAY', 'COUNT'])

    return G

def get_stats(G, node):
    for x in G.degree_iter(weight='COUNT'):
        print x

flights = load_data()
G = load_graph(flights, 2015, 1, 1, 5)

print get_stats(G)
