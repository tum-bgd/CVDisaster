import geojson
import os
import rasterio

from PIL import Image


TILE_W = 512
TILE_H = 512


def GetSatelliteRange() -> list:
    res = []
    for fileName in os.listdir('./data/01_Satellite'):
        currName = os.path.join('./data/01_Satellite', fileName)
        if os.path.isfile(currName):
            with rasterio.open(currName) as tif:
                res.append({
                    'path': currName,
                    'bbox': tif.bounds
                })
    return res


def CutSTLImages(
    srcDirSVI: str,
    tarDirSTL: str,
    tifInfos: list,
    saveTif=False
) -> None:
    with open('./data/02_SVOI_point/SVI_position.geojson', 'r') as f:
        mappings = geojson.load(f)["features"]
    for sviImageName in os.listdir(srcDirSVI):
        sviPoint = sviImageName[:-4].split('.')[0]
        for mapping in mappings:
            if str(mapping['properties']['id']) == sviPoint:
                lon, lat = mapping['properties']['lon'], mapping['properties']['lat']
                for tifInfo in tifInfos:
                    if tifInfo['bbox'].left <= lon and lon <= tifInfo['bbox'].right and \
                       tifInfo['bbox'].bottom <= lat and lat <= tifInfo['bbox'].top:
                        # within this file, cut
                        with rasterio.open(tifInfo['path']) as tif:
                            py, px = tif.index(lon, lat)
                            window = rasterio.windows.Window(px-TILE_W//2, py-TILE_H//2, TILE_W, TILE_H)
                            # print(sviImageName, 'in', tifInfo['path'], 'at', window)
                            tile = tif.read(window=window)
                            if saveTif:
                                meta = tif.meta
                                meta['width'], meta['height'] = TILE_W, TILE_H
                                meta['transform'] = rasterio.windows.transform(window, tif.transform)
                                with rasterio.open(os.path.join(tarDirSTL, sviPoint + '.tiff'), 'w', **meta) as dst:
                                    dst.write(tile)
                            else:
                                img = Image.fromarray(tile.transpose(1, 2, 0), 'RGB')
                                img.save(os.path.join(tarDirSTL, sviPoint + '.png'))


os.mkdir('./data/01_Satellite/')
os.mkdir('./data/01_Satellite/0_MinorDamage')
os.mkdir('./data/01_Satellite/1_ModerateDamage')
os.mkdir('./data/01_Satellite/2_SevereDamage')
tifInfos = GetSatelliteRange()
CutSTLImages(
    './data/00_SVI/0_MinorDamage',
    './data/01_Satellite/0_MinorDamage',
    tifInfos)
CutSTLImages(
    './data/00_SVI/1_ModerateDamage',
    './data/01_Satellite/1_ModerateDamage',
    tifInfos)
CutSTLImages(
    './data/00_SVI/2_SevereDamage',
    './data/01_Satellite/2_SevereDamage',
    tifInfos)
