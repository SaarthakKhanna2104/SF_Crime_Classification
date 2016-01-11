import folium
import pandas as pd
from collections import OrderedDict
 
SF_COORDINATES = (37.76, -122.45)
crimedata = pd.read_csv('train.csv')

crimeCat=str(crimedata.sort(['Category'])['Category'].unique())

d=OrderedDict()
d['S'] = "Sunday"
d['M'] = "Monday"
d['T'] = "Tuesday"
d['W'] = "Wednesday"
d['TH'] = "Thursday"
d['F'] = "Friday"
d['SA'] = "Saturday"

crime = raw_input('Choose one of the following crime categories:\n'+crimeCat[1:len(crimeCat)-1]+'\n--->')
crimeselect = crime.upper()

dayC = str([k+'->'+d[k] for k in d.keys() ])

daysStr = raw_input('Choose any no. of days from the following (days separated by a comma)\n'+dayC[1:len(dayC)-1]+', A->All of the above\n--->')
if daysStr.upper() == 'A':
  dayselect = d.values()
else:
  dayselect = [d[day] for day in daysStr.upper().split(',')]


daycond = crimedata['DayOfWeek'].isin(dayselect) 
crimecond = crimedata['Category'] == (crimeselect)
 
filtered_crimedata = crimedata[crimecond & daycond]


#clustered map 
map = folium.Map(location=SF_COORDINATES, zoom_start=12)
 
for each in filtered_crimedata.iterrows():
    map.simple_marker(location =[each[1]['Y'],each[1]['X']],
                      popup=each[1]['Category'] + ": " + each[1]['Descript'], 
                      clustered_marker=True)
 
map.create_map(path='map_clustered_drugs.html')


#choropleth map
district_geo = r'sfpd_districts.json'
 
crimedata2 = pd.DataFrame(filtered_crimedata['PdDistrict'].value_counts().astype(float))
crimedata2.to_json('crimeagg.json')
crimedata2 = crimedata2.reset_index()
crimedata2.columns = ['District', 'Number']
 
map1 = folium.Map(location=SF_COORDINATES, zoom_start=12)
map1.geo_json(geo_path = district_geo, 
              data_out = 'crimeagg.json', 
              data = crimedata2,
              columns = ['District', 'Number'],
              key_on = 'feature.properties.DISTRICT',
              fill_color = 'YlOrRd', 
              fill_opacity = 0.7, 
              line_opacity = 0.2,
              legend_name = 'Number of incidents per district')
              
map1.create_map(path='choro_drugs.html')