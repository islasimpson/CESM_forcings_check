import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from math import nan
from pathlib import Path
import dask
import yaml
import os
import textwrap

### Function to calculate annual mean with appropriate weighting for the months
def calcannualmean(ds, skipna=False):
    """ Calculate the annual mean weighting the months of the year appropriately if
        given the calendar type
    """
    
    def dothecalc(var, skipna=False):
        month_length = var.time.dt.days_in_month
        wghts = month_length.groupby('time.year') / month_length.groupby('time.year').sum()
        if (skipna):
            datsum = (var*wghts).groupby('time.year').sum(dim='time', skipna=True)
            cond = var.isnull()
            ones = xr.where(cond, 0, 1)
            onesum = (ones*wghts).groupby('time.year').sum(dim='time')
        else:
            datsum = (var*wghts).groupby('time.year').sum(dim='time', skipna=False)
            cond = var.isnull()
            ones = xr.where(cond, 1, 1)
            onesum = (ones*wghts).groupby('time.year').sum(dim='time')

        var_am = datsum / onesum
        return var_am

    #--Note if the NaN's are different in each variable you'll be averaging over
    #-- different times for each variable
    dset = False
    try:
        varnames = list(ds.keys())
        dset = True
    except:
        pass

    if (dset):
        for i, ivar in enumerate(varnames):
            var = ds[ivar] 
            var_am = dothecalc(var, skipna=skipna)
            var_am = var_am.rename(ivar)
            if (i == 0):
                ds_am = var_am
            else:
                ds_am = xr.merge([ds_am, var_am])
    else:
        ds_am = dothecalc(ds, skipna=skipna)
        ds_am = ds_am.rename(ds.name)

    return ds_am



### Function to convert kg/m2/s to Tg
def convert_kgm2s_to_Tg(dat, lon_bnds, lat_bnds):
    """ 
        Input:
          - dat(year,lat,lon) = annual mean emissions in (kg/m2/s)
          - lon_bnds = the longitude bounds
          - lat_bnds = the latitude bounds
        Output:
          - dattot = the globally integrated emissions in Tg / year
    """

    # radius of the Earth in m
    re = 6.3712e6

    # convert from per s to per year
    dat_y = dat*365.*86400.

    # Integrate over space
    dlon = np.abs(lon_bnds.isel(bound=1) - lon_bnds.isel(bound=0))
    dlat = np.abs(lat_bnds.isel(bound=0) - lat_bnds.isel(bound=1))
    dlon_rad = np.deg2rad(dlon)
    dlat_rad = np.deg2rad(dlat)
    area = xr.ones_like(dat.isel(year=0))
    weights = np.cos(np.deg2rad(area.lat))*dlat_rad*dlon_rad*re**2
    dat_y = dat_y.where( ~np.isnan(dat_y), 0) # setting emissions to zero when NaN
    dat_y_w = dat_y.weighted(weights)
    dattot = dat_y_w.sum(("lon","lat"))

    # convert from kg to Tg
    dattot = dattot/1e9

    return dattot


### Function to scale by mole
def scale_mole(dat,mwsource,mw,scalefac):
    """ Assign a portion (scalefac) of the moles of dat (molecular weight mswsource)
        to a different species (molecular weight mw)
        dat should be in kg/m2/s
    """

    ### Convert to moles per m2 per s
    dat_moles = 1000.*dat / mwsource

    ### Scale by the scalefac
    dat_moles_out = scalefac*dat_moles

    ### Convert the moles of the output specifes to kg 
    dat_kg_out = dat_moles_out * mw / 1000.

    return dat_kg_out


### Convert molecules to Tg
def convert_molecules_to_tg(dat,varname):
    """ Convert surface emissions in molecules/cm2/s to Tg

    """
    # Start with moleculre/cm2/s.  Convert from molecules to grams.
    # Divide by Avogadro's number to convert from molecules to moles.
    #Multiply by molcular weight in g/mol to end up with g/cm2/s

    avog = 6.022e23 # Avogadro's number
    re = 6.3712e8 # Radius of the earth in cm

    dat_g = dat[varname].molecular_weight*dat[varname]/avog

    # convert from per s to per year
    dat_g_y = dat_g*365.*86400.

    if "altitude" in dat.dims:
        print('you have altitudes')
        dz = dat.altitude_int[1:dat.altitude_int.size].values - dat.altitude_int[0:dat.altitude_int.size-1].values
        dz = xr.DataArray(dz, coords=[dat.altitude], dims=['altitude'], name='dz')
        dz = dz*1000.*100. # convert to cm
        dat_g_y = (dat_g_y*dz).sum('altitude')


    # Integrate over space
    if "ncol" in dat.dims: # using spectral element
        weights = dat.area*re**2
        dat_g_y_w = dat_g_y.weighted(weights)
        dattot = dat_g_y_w.sum('ncol')
    else:
        dlon = np.deg2rad( (dat.lon[2] - dat.lon[1]))
        dlat = np.deg2rad( (dat.lat[2] - dat.lat[1]))
        area = xr.ones_like(dat.isel(time=0))
        weights = np.cos(np.deg2rad(area.lat))*dlat*dlon*re**2. # area in cm2
        dat_g_y_w = dat_g_y.weighted(weights)
        dattot = dat_g_y_w.sum(("lon","lat"))

    # Convert from grams to terra grams
    dattot = dattot/1e12

    return dattot


def setup_plot_locs():
    """ Setting up the plot positions"""

    x1 = [0.04,0.28,0.52,0.76,
          0.04,0.28,0.52,0.76,
          0.04,0.28,0.52,0.76,
          0.04,0.28,0.52,0.76,
          0.04,0.28,0.52,0.76,
          0.04,0.28,0.52,0.76,
          0.04,0.28,0.52,0.76,
          0.04,0.28,0.52,0.76,
          0.04,0.28,0.52,0.76,
          0.04,0.28,0.52,0.76
         ]
    x2 = [0.22,0.46,0.7,0.94,
          0.22,0.46,0.7,0.94,
          0.22,0.46,0.7,0.94,
          0.22,0.46,0.7,0.94,
          0.22,0.46,0.7,0.94,
          0.22,0.46,0.7,0.94,
          0.22,0.46,0.7,0.94,
          0.22,0.46,0.7,0.94,
          0.22,0.46,0.7,0.94,
          0.22,0.46,0.7,0.94
         ]
    y1 = [0.82,0.82,0.82,0.82,
          0.73,0.73,0.73,0.73,
          0.64,0.64,0.64,0.64,
          0.55,0.55,0.55,0.55,
          0.46,0.46,0.46,0.46,
          0.37,0.37,0.37,0.37,
          0.28,0.28,0.28,0.28,
          0.19,0.19,0.19,0.19,
          0.1,0.1,0.1,0.1]
    y2 = [0.9,0.9,0.9,0.9,
          0.81,0.81,0.81,0.81,
          0.72,0.72,0.72,0.72,
          0.63,0.63,0.63,0.63,
          0.54,0.54,0.54,0.54,
          0.45,0.45,0.45,0.45,
          0.36,0.36,0.36,0.36,
          0.27,0.27,0.27,0.27,
          0.18,0.18,0.18,0.18]

    return x1, x2, y1, y2


def plot_the_plot(fig, dat, species, sector, x1, x2, y1, y2, input4mips = None, mylabel = None, subspecies=None):
    ax = fig.add_axes([x1, y1, (x2-x1), (y2-y1)])
    ax.plot(dat.year, dat, label=mylabel, color='red', linewidth=2, zorder=1, linestyle='dotted')
    fig.text(x1+0.005,y2-0.02,species, fontsize=8, va='top')
    if subspecies is not None:
        fig.text(x1+0.005,y2-0.025,subspecies, fontsize=8, va='top')
    sectortext = "\n".join(textwrap.wrap(sector, width=20))
    fig.text(x1+0.005,y2-0.03,sectortext, fontsize=8, va='top')
    ax.set_ylabel('Tg', fontsize=12)
    if input4mips is not None:
        ax.plot(input4mips.year, input4mips, color='black', linewidth=2, label='input4mips', zorder=0)
    ax.legend(loc='upper left')
    
    return ax
