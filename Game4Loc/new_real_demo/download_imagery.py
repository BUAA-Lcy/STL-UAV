"""Download verified Esri World Imagery tiles for the prepared demo footprint.

Location disclosure to Esri was explicitly authorized by the user. Unknown or
missing tiles fail validation. No upsampling is presented as higher resolution.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import time
import cv2
import numpy as np
import requests
from build import OUT

BASE='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer'
BLANK_SHA='9eafd300d61393184a4abc1d458564cfd1cd9b6f9c4e9c74687045c0a0e5b858'


def required_tiles(rows, manifest, zoom):
    needed=set()
    origin=np.array(manifest['mosaic_origin_tile'])*256
    factor=2**(zoom-manifest['mosaic_zoom'])
    for row in rows:
        corners=(np.array(json.loads(row['footprint_mosaic_xy']))+origin)*factor
        low=np.floor((corners.min(axis=0)-2)/256).astype(int)
        high=np.floor((corners.max(axis=0)+2)/256).astype(int)
        needed.update((x,y) for x in range(low[0],high[0]+1) for y in range(low[1],high[1]+1))
    return sorted(needed)


def validate(data, headers):
    if headers.get('blank-tile','').lower()=='true':
        raise ValueError('Service returned an unavailable-image tile')
    digest=hashlib.sha256(data).hexdigest()
    if digest==BLANK_SHA:
        raise ValueError('Known map-data-not-available placeholder')
    image=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
    if image is None or image.shape!=(256,256,3):
        raise ValueError('Tile is not a 256x256 color image')
    return digest


def fetch_tile(zoom,x,y,cache):
    url=f'{BASE}/tile/{zoom}/{y}/{x}'
    path=cache/f'{x}_{y}.jpg'
    if path.exists():
        data=path.read_bytes()
        return dict(x=x,y=y,url=url,bytes=len(data),sha256=validate(data,{}),cached=True)
    for attempt in range(3):
        try:
            response=requests.get(url,params={'blankTile':'false'},timeout=(10,30))
            response.raise_for_status()
            data=response.content
            digest=validate(data,response.headers)
            temp=path.with_suffix('.jpg.part')
            temp.write_bytes(data);temp.replace(path)
            return dict(x=x,y=y,url=response.url,bytes=len(data),sha256=digest,cached=False)
        except (requests.RequestException,ValueError):
            if attempt==2:raise
            time.sleep(attempt+1)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--zoom',type=int,default=18)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    if not 0<=args.zoom<=23:parser.error('zoom must be 0..23')
    manifest=json.loads((args.output/'manifest.json').read_text())
    rows=list(csv.DictReader((args.output/'frames.csv').open()))
    needed=required_tiles(rows,manifest,args.zoom)
    if len(needed)>2000:raise ValueError('Footprint exceeds bounded demo download size')
    stamp=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d')
    cache=args.output/'satellite_cache'/f'world_imagery_{stamp}'/f'z{args.zoom}'
    cache.mkdir(parents=True,exist_ok=True)
    print(f'Downloading {len(needed)} tiles at z{args.zoom} for {len(rows)} frames',flush=True)
    reports=[];failures=[]
    with ThreadPoolExecutor(max_workers=min(max(args.workers,1),6)) as pool:
        jobs={pool.submit(fetch_tile,args.zoom,int(x),int(y),cache):(x,y) for x,y in needed}
        for future in as_completed(jobs):
            try:reports.append(future.result())
            except Exception as e:failures.append(dict(tile=list(map(int,jobs[future])),error=str(e)))
            if (len(reports)+len(failures))%10==0 or len(reports)+len(failures)==len(needed):
                print(f'{len(reports)} valid, {len(failures)} failed / {len(needed)}',flush=True)
    latitude=float(rows[len(rows)//2]['lat'])
    report=dict(provider='Esri World Imagery',service_url=BASE,zoom=args.zoom,
                downloaded_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                imagery_acquisition_date='unknown; download date is not image acquisition date',
                cache=str(cache.resolve()),required_tiles=len(needed),valid_tiles=len(reports),
                failed_tiles=failures,bytes=sum(r['bytes'] for r in reports),
                tile_grid_ground_m_per_pixel=math.cos(math.radians(latitude))*2*math.pi*6378137/(256*2**args.zoom),
                source_frame_first=int(rows[0]['frame_index']),source_frame_last=int(rows[-1]['frame_index']),
                selection_note=f'Verified Esri tiles at z{args.zoom}; grid spacing is not a claim of sensor resolution. Higher-level availability probes for the initial download are stored in hd_probe.',
                attribution='Esri, Maxar, Earthstar Geographics, and the GIS User Community',
                tiles=sorted(reports,key=lambda r:(r['x'],r['y'])),
                status='COMPLETE' if not failures else 'INCOMPLETE')
    (cache/'download_manifest.json').write_text(json.dumps(report,indent=2))
    if failures:raise SystemExit('Incomplete imagery; existing demo selection was not changed')
    # This pointer is the explicit current imagery choice, written only on success.
    (args.output/'imagery.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='tiles'},indent=2),flush=True)


if __name__=='__main__':main()
