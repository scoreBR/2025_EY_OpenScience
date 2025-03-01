import warnings
warnings.filterwarnings("ignore")

# Importação de bibliotecas
import numpy as np
import pandas as pd
import rioxarray as rxr
import rasterio
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pystac_client
import planetary_computer as pc
from odc.stac import stac_load
from tqdm import tqdm

# Leitura do conjunto de dados
csv_file = 'Training_data_uhi_index_2025-02-04.csv'
ground_df = pd.read_csv(csv_file)

# Definição de limites geográficos e janela temporal
lower_left, upper_right = (40.75, -74.01), (40.88, -73.86)
bounds = (lower_left[1], lower_left[0], upper_right[1], upper_right[0])
time_window = "2021-06-01/2021-09-01"

# Consulta de dados no STAC
stac = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
search = stac.search(bbox=bounds, datetime=time_window, collections=["sentinel-2-l2a"], query={"eo:cloud_cover": {"lt": 30}})
items = list(search.get_items())
print(f'Número de cenas encontradas: {len(items)}')

# Processamento dos dados
resolution = 10  # metros por pixel
scale = resolution / 111320.0  # graus por pixel (para EPSG:4326)
data = stac_load(items, bands=["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"],
                 crs="EPSG:4326", resolution=scale, chunks={"x": 2048, "y": 2048}, dtype="uint16",
                 patch_url=pc.sign, bbox=bounds)

# Cálculo da mediana e índices espectrais
def calculate_index(numerator, denominator):
    return (numerator - denominator) / (numerator + denominator)

median = data.median(dim="time").compute()
indices = {
    "NDVI": calculate_index(median.B08, median.B04),
    "NDBI": calculate_index(median.B11, median.B08),
    "NDWI": calculate_index(median.B03, median.B08)
}

# Extração de bandas e escrita em GeoTIFF
filename = "S2_sample.tiff"
data_slice = data.isel(time=7)
height, width = data_slice.dims["latitude"], data_slice.dims["longitude"]
gt = rasterio.transform.from_bounds(*bounds, width, height)
data_slice.rio.write_crs("epsg:4326", inplace=True)
data_slice.rio.write_transform(transform=gt, inplace=True)

with rasterio.open(filename, 'w', driver='GTiff', width=width, height=height, crs='epsg:4326',
                   transform=gt, count=4, compress='lzw', dtype='float64') as dst:
    for i, band in enumerate(["B01", "B04", "B06", "B08"], 1):
        dst.write(data_slice[band].values, i)

# Função para mapear valores de satélite
def map_satellite_data(tiff_path, csv_path):
    data = rxr.open_rasterio(tiff_path)
    df = pd.read_csv(csv_path)
    coords = zip(df['Latitude'].values, df['Longitude'].values)

    band_values = {f'B0{i}': [] for i in [1, 4, 6, 8]}
    for lat, lon in tqdm(coords, total=len(df), desc="Mapping values"):
        for i, band in enumerate(band_values.keys(), 1):
            band_values[band].append(float(data.sel(x=lon, y=lat, band=i, method="nearest").values))
    
    return pd.DataFrame(band_values)

# Aplicação do mapeamento
data_mapped = map_satellite_data(filename, csv_file)

# Cálculo do NDVI
data_mapped['NDVI'] = calculate_index(data_mapped['B08'], data_mapped['B04']).replace([np.inf, -np.inf], np.nan)

# Combinação com os dados terrestres
dataset = pd.concat([ground_df, data_mapped], axis=1).drop_duplicates().reset_index(drop=True)

# Seleção de características e separação de treino e teste
X = dataset[['B01', 'B06', 'NDVI']].values
y = dataset['UHI Index'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

# Treinamento do modelo de Regressão Linear
model = LinearRegression()
model.fit(X_train, y_train)

# Avaliação do modelo
print("R² Treinamento:", r2_score(y_train, model.predict(X_train)))
print("R² Teste:", r2_score(y_test, model.predict(X_test)))

# Predição e submissão
test_data = map_satellite_data(filename, 'Submission_template.csv')
test_data['NDVI'] = calculate_index(test_data['B08'], test_data['B04']).replace([np.inf, -np.inf], np.nan)

submission_df = pd.DataFrame({'Longitude': pd.read_csv('Submission_template.csv')['Longitude'],
                              'Latitude': pd.read_csv('Submission_template.csv')['Latitude'],
                              'UHI Index': model.predict(test_data[['B01', 'B06', 'NDVI']])})
submission_df.to_csv("submission.csv", index=False)

print("Submissão gerada com sucesso!")