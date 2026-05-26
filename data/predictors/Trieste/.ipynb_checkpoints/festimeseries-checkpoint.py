import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyfes # needs to be set up with the FES2014 model https://www.aviso.altimetry.fr/en/data/products/auxiliary-products/global-tide-fes/description-fes2014.html
import datetime as dt

date_start = datetime.datetime.fromisoformat("1980-01-01 00:00:00")
date_end = datetime.datetime.fromisoformat("2020-01-01 00:10:00")
duration = int((date_end -date_start).total_seconds()/3600)
ocean = pyfes.Handler('ocean','memory',"ocean_tide.ini")
Load = pyfes.Handler('radial','memory',"load_tide.ini")


dates = np.array([ date_start + datetime.timedelta(seconds = h*3600) for h in range(duration)])

lats = np.full(dates.shape,-8.75)
lons = np.full(dates.shape,13.20)

tide,lp,_= ocean.calculate(lons,lats,dates)
load=Load.calculate(lons,lats,dates)[0]
df = pd.DataFrame(index=dates)
df['time'] = [(d - dates[0]).total_seconds() for d in dates]
df['tides'] = (tide+lp)*0.01
df.index.name = "date_time"




df.to_csv(f'Tide_Luanda_angola.txt')
