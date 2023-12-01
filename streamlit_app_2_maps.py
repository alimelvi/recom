from streamlit_folium import st_folium
import streamlit as st
import pandas as pd
import numpy as np
import leafmap.kepler as leafmap
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from geopy.distance import geodesic
import folium
from folium.plugins import Draw

st.set_page_config(layout="wide")


st.title("Best Location Finder")

df = pd.read_csv("SA_final.csv")
m = leafmap.Map(center=[df['lat'].mean(),df['lon'].mean()], zoom=3)
m.add_df(df, layer_name="hex_data")
m.to_streamlit()

scaler = MinMaxScaler()


df_normalized = df.copy()
df_normalized[['PVOUT_csi', 'DNI', 'GHI', 'DIF', 'GTI_opta', 'OPTA', 'TEMP', 'ELE']] = scaler.fit_transform(
       df[['PVOUT_csi', 'DNI', 'GHI', 'DIF', 'GTI_opta', 'OPTA', 'TEMP', 'ELE']])

# Cosine similarity
similarity_matrix = cosine_similarity(
       df_normalized[['PVOUT_csi', 'DNI', 'GHI', 'DIF', 'GTI_opta', 'OPTA', 'TEMP', 'ELE']])
similarity_df = pd.DataFrame(
       similarity_matrix, index=df.index, columns=df.index)


def find_nearest_location(input_lat, input_long):
    """
    Finds the nearest location in the DataFrame to the given latitude and longitude.

    Parameters:
    - input_lat (float): Latitude of the input location.
    - input_long (float): Longitude of the input location.

    Returns:
    - nearest_location (tuple): Tuple containing the nearest latitude and longitude.
    """
    distances = [geodesic((input_lat, input_long), (lat, lon)
                          ).kilometers for lat, lon in zip(df['lat'], df['lon'])]
    nearest_location_index = np.argmin(distances)
    nearest_location = (df.loc[nearest_location_index, 'lat'],
                        df.loc[nearest_location_index, 'lon'])
    return nearest_location


def recommend_locations(input_lat, input_long, num_recommendations=5, max_distance_km=10):
       """
       Recommends locations based on similarity, excluding those within a specified distance.

       Parameters:
       - input_lat (float): Latitude of the input location.
       - input_long (float): Longitude of the input location.
       - num_recommendations (int): Number of locations to recommend. Default is 5.
       - max_distance_km (float): Maximum distance in kilometers for recommended locations. Default is 10.

       Returns:
       - recommendations (DataFrame): DataFrame containing recommended locations and their attributes.
       """

       print(f"Given location : {input_lat}, {input_long}")

       # Check if the input location is in the DataFrame; if not, find the nearest location
       if (input_lat, input_long) not in zip(df['lat'], df['lon']):
              print("Given location not found in DataFrame. Finding nearest location...")
              input_lat, input_long = find_nearest_location(input_lat, input_long)
              print(f"Nearest location found: {input_lat}, {input_long}")

       # Find the index of the input location in the DataFrame
       input_location = df[(df['lat'] == input_lat) &
                            (df['lon'] == input_long)].index[0]

       # Calculate distances and filter out locations within the specified threshold
       distances = [(loc, geodesic((df.loc[input_location, 'lat'], df.loc[input_location, 'lon']),
                                   (df.loc[loc, 'lat'], df.loc[loc, 'lon'])).kilometers)
                     for loc in similarity_df.index[1:]]  # Skip the input location

       filtered_locations = []
       previous_location = input_location

       for loc, distance in distances:
              # Check if the distance is greater than max_distance_km from the previous location
              if distance > max_distance_km:
                     filtered_locations.append(loc)
                     previous_location = loc
       # If there are no recommendations after filtering, return an appropriate message
       if not filtered_locations:
              print(f"No recommendations found beyond {max_distance_km} km.")
              return

       # Get the top recommended locations based on similarity after filtering
       recommended_locations = sorted(filtered_locations, key=lambda loc: similarity_df.loc[input_location, loc],
                                   reverse=True)[:num_recommendations]

       # Get recommendations
       recommendations = df.loc[recommended_locations, [
              'lat', 'lon', 'PVOUT_csi', 'DNI', 'GHI', 'DIF', 'GTI_opta', 'OPTA', 'TEMP', 'ELE']]
       return recommendations


# Streamlit app
st.title('Location Recommender App')
# Number of recommendations
num_recommendations = 5
minimum_distance = 100
m2 = folium.Map(location=[24, 60], zoom_start=5)
st.subheader('Please click on the left map to see the recommended locations in the right map')

Draw(export=True).add_to(m2)
c1, c2 = st.columns(2)
with c1:    
       output = st_folium(m2, width=1000)
       previous_point = output['last_clicked']
       print(output)
with c2:
       if output['last_clicked'] is not None:
              input_lat = output['last_clicked']['lat']
              input_long = output['last_clicked']['lng']
              # Get recommendations
              recommendations = recommend_locations(
              input_lat, input_long, num_recommendations, minimum_distance)
              folium.Marker(
                     location=[input_lat,
                            input_long,],
                     popup="selected point",
              ).add_to(m2)
              for i in range(0, len(recommendations)):
                     folium.Marker(
                            location=[recommendations.iloc[i]['lat'],
                                   recommendations.iloc[i]['lon']],
                            popup=f'recommendation {i+1}',
                     ).add_to(m2)
              st_data = st_folium(m2, width=1000)

if output['last_clicked'] is not None:
       # Display recommendations
       st.subheader('Recommended Locations:')
       st.table(recommendations)
