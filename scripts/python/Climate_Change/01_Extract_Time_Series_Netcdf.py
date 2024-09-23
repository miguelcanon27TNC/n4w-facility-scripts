import os
import numpy as np
import pandas as pd
import xarray as xr
from multiprocessing import Pool
from functools import partial


def get_file_list(directory_path, ext):
    """
    Returns a list of files in a directory with a specific extension.

    Parameters:
    - directory_path (str): The path to the directory.
    - ext (str): The file extension to look for (e.g., '.nc').

    Returns:
    - data_files (list): List of files with full path.
    - fullfilename (list): List of all files in the directory.
    """
    fullfilename = os.listdir(directory_path)
    data_files = [os.path.join(directory_path, f) for f in fullfilename if os.path.splitext(f)[1] == ext]
    return data_files, fullfilename


def obtener_indices_tiempo(ruta, archivos):
    """
    Retrieves time indices from a list of NetCDF files.

    Parameters:
    - ruta (str): Directory path of the NetCDF files.
    - archivos (list): List of NetCDF files.

    Returns:
    - dt (np.array): Concatenated array of time indices.
    """

    def indice_tiempo_de_archivo(ruta_completa):
        with xr.open_dataset(ruta_completa, engine='netcdf4') as ds:
            try:
                return ds.indexes['time'].to_datetimeindex()
            except KeyError:
                return ds.indexes['time']

    indices_tiempo = [indice_tiempo_de_archivo(os.path.join(ruta, archivo)) for archivo in archivos]
    dt = np.concatenate(indices_tiempo)
    return dt


def extract_values_netcdf(nc, variable, latitud, longitud):
    """
    Extracts values for a given variable in a NetCDF file at specified latitude and longitude.

    Parameters:
    - nc (xarray.Dataset): The NetCDF dataset.
    - variable (str): The variable to extract.
    - latitud (float): Latitude of interest.
    - longitud (float): Longitude of interest.

    Returns:
    - data (list): Extracted data values for the variable.
    """
    try:
        latitudes = nc.variables['lat'][:]
        longitudes = nc.variables['lon'][:]

        # Find the closest latitude and longitude values
        closest_lat = min(latitudes.values, key=lambda x: abs(x - latitud))
        pixel_ver = np.where(closest_lat == latitudes.values)[0]

        closest_lon = min(longitudes.values, key=lambda x: abs(x - np.mod(longitud, 360)))
        pixel_hor = np.where(closest_lon == longitudes.values)[0]

        ymin, ymax = min(pixel_ver), max(pixel_ver) + 1
        xmin, xmax = min(pixel_hor), max(pixel_hor) + 1

        # Extract data based on the requested variable
        if variable == 'hur':
            data = np.mean(nc.variables[variable][:, 1:2, ymin:ymax, xmin:xmax].values, axis=(1, 2, 3))
        else:
            data = np.mean(nc.variables[variable][:, ymin:ymax, xmin:xmax].values, axis=(1, 2))

        return np.nan_to_num(data, nan=0.0).tolist()

    except Exception as e:
        print(f"Error opening the file: {e}")
        return []


def process_file(f, lat, lon, var):
    """
    Processes a NetCDF file and extracts data for a variable at a specific location.

    Parameters:
    - f (str): Path to the NetCDF file.
    - lat (float): Latitude of interest.
    - lon (float): Longitude of interest.
    - var (str): Variable to extract.

    Returns:
    - list_d_u (list): List of processed data values for the variable.
    """
    dict_var = {'Tas': 'tas', 'PT': 'pr'}
    try:
        ds = xr.open_dataset(f + '.nc')
        list_d = extract_values_netcdf(ds, dict_var[var], lat, lon)

        # Adjust units based on the variable
        if dict_var[var] == 'pr':
            list_d_u = np.array(list_d) * 86400  # Convert precipitation from m/s to mm/day
        elif dict_var[var] == 'tas':
            list_d_u = np.array(list_d) - 273.15  # Convert temperature from K to °C
        else:
            list_d_u = list_d

        ds.close()
        return list_d_u.tolist()
    except Exception as e:
        print(f"Error opening the file: {e}")
        return []


def process_station(sta, files, Catalogo, var):
    """
    Processes data for a station and returns aggregated data.

    Parameters:
    - sta (str): Station name.
    - files (list): List of NetCDF files.
    - Catalogo (pd.DataFrame): Catalog with latitude and longitude for the stations.
    - var (str): Variable to process.

    Returns:
    - n_data (list): List of processed data for the station.
    """
    print(sta)
    n_data = []
    lat = Catalogo.loc[sta]['LATITUD']
    lon = Catalogo.loc[sta]['LONGITUD']
    for file in files:
        n_data.extend(process_file(file, lat, lon, var))
    return n_data


def main(base_path, path_cat, catalog_file, path_out, variables, dict_var, dict_cat):
    """
    Main function to process NetCDF files and generate CSVs for climate scenarios.

    Parameters:
    - base_path (str): Path to the directory containing NetCDF files.
    - path_cat (str): Path to the catalog Excel file.
    - catalog_file (str): Name of the Excel catalog file (e.g., '03_Catalogo_Cumplen.xlsx').
    - path_out (str): Output path where results will be saved.
    - variables (list): List of variables to process (e.g., ['Tas', 'PT']).
    - dict_var (dict): Dictionary mapping variable names to NetCDF variable keys.
    - dict_cat (dict): Dictionary mapping variables to the corresponding sheet names in the catalog.

    Notes:
    - Change `base_path`, `path_cat`, `path_out`, `variables`, `dict_var`, and `dict_cat` as needed for your specific project.
    """
    # Normalize path (replace backslashes with slashes)
    base_path = base_path.replace("\\", "/")

    # Load the station catalog
    catalog_path = os.path.join(path_cat, catalog_file)
    Catalogo_excel = pd.ExcelFile(catalog_path)

    for var in variables:
        print(f"Processing variable: {var}")
        subdir = os.path.join(base_path, var)
        modelos = next(os.walk(subdir))[1]  # Get model directories
        Catalogo = Catalogo_excel.parse(dict_cat[var], index_col=0)
        estaciones = Catalogo.index

        for m in modelos[0:]:  # Adjust the range as needed
            print(f"Processing model: {m}")
            n_subdir = os.path.join(base_path, var, m)
            escenarios = ['historical', 'ssp126', 'ssp370', 'ssp245', 'ssp585']  # Climate scenarios

            # Output path for results
            output_path = os.path.join(path_out, '02_Escenarios_Cambio_Climatico', '01_Series_GCM_raw', m,
                                       dict_cat[var])
            os.makedirs(output_path, exist_ok=True)

            for es in escenarios:
                print(f"Processing scenario: {es}")
                new_path = os.path.join(base_path, var, m, es)
                files, files_fullname = get_file_list(new_path, '.nc')
                n_time = obtener_indices_tiempo(new_path, files_fullname)

                s_data = pd.DataFrame(columns=estaciones, index=n_time)

                # Parallelize the station processing
                with Pool() as p:
                    process_station_partial = partial(process_station, files=files, Catalogo=Catalogo, var=var)
                    all_results = p.map(process_station_partial, estaciones)

                # Save results to DataFrame
                for sta, results in zip(estaciones, all_results):
                    if len(results) != len(s_data.index):
                        results.extend([np.nan] * (len(s_data.index) - len(results)))
                    s_data[sta] = pd.Series(results, index=s_data.index)

                # Export results to CSV
                name_out = os.path.join(output_path, f'Series_{es}_{dict_cat[var]}.csv')
                s_data.to_csv(name_out, index=True)


# Define paths and dictionaries - adjust these variables as needed for your project
path = r'D:\CMIP6/'
path_cat = r'D:\Dropbox\Trabajos\Actividades_M\PORH\07_Hidrologia_CC/'
catalog_file = '03_Catalogo_Cumplen.xlsx'
path_out = r'D:\Dropbox\Trabajos\Actividades_M\PORH\07_Hidrologia_CC/'

# Dictionary mapping user-friendly names to NetCDF variable keys
dict_var = {'Tas': 'tas', 'PT': 'pr'}
# Dictionary mapping variables to sheet names in the station catalog
dict_cat = {'Tas': 'TS_1', 'PT': 'PT_4'}

# Variables to process (subdirectories in `path`)
variables = next(os.walk(path.replace("\\", "/")))[1]

if __name__ == '__main__':
    main(path, path_cat, catalog_file, path_out, variables, dict_var, dict_cat)
