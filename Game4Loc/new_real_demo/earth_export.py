"""Prepare Google Earth export controls and import a calibrated clean export.

No Google tile endpoints or legacy matchers are used. Export two images from
the same north-up, vertical, terrain-disabled Earth view: controls, then clean.
The nine artificial KML markers establish geographic coordinates; they are
not visual feature matches between the UAV video and the satellite image.
"""
import argparse
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET

import cv2
import numpy as np

from build import OUT, EARTH, world_pixel


def latlon(xy, zoom):
    x, y = xy
    n = 256 * 2**zoom
    return math.degrees(math.atan(math.sinh(math.pi*(1-2*y/n)))), x/n*360-180


def load_extent(output, zoom):
    manifest = json.loads((output/'manifest.json').read_text())
    rows = list(csv.DictReader((output/'frames.csv').open()))
    origin = np.array(manifest['mosaic_origin_tile'])*256
    polygons = [(np.array(json.loads(r['footprint_mosaic_xy']))+origin)
                * 2**(zoom-manifest['mosaic_zoom']) for r in rows]
    points = np.concatenate(polygons)
    return manifest, rows, polygons, points.min(axis=0), points.max(axis=0)


def prepare(output):
    dest = output/'google_earth_export'
    dest.mkdir(exist_ok=True)
    zoom = 20  # Rectification grid, not a claim about native imagery resolution.
    manifest, rows, polygons, lo, hi = load_extent(output, zoom)
    center = latlon((lo+hi)/2, zoom)
    mpp = math.cos(math.radians(center[0]))*2*math.pi*EARTH/(256*2**zoom)
    low, high = lo-35/mpp, hi+35/mpp
    names = ['NW','N','NE','W','C','E','SW','S','SE']
    control = []
    for i, (y,x) in enumerate((y,x) for y in np.linspace(low[1],high[1],3)
                            for x in np.linspace(low[0],high[0],3)):
        lat, lon = latlon([x,y],zoom)
        control.append(dict(id=names[i],lat=lat,lon=lon,world_xy=[x,y]))
    ns = 'http://www.opengis.net/kml/2.2'
    ET.register_namespace('',ns)
    def el(parent,name,text=None,**attrib):
        child=ET.SubElement(parent,'{'+ns+'}'+name,attrib)
        if text is not None:child.text=str(text)
        return child
    root=ET.Element('{'+ns+'}kml');doc=el(root,'Document')
    el(doc,'name','NewRealDemo - last 30 seconds')
    el(doc,'description','Export controls and clean imagery without moving the view. Disable terrain and 3D buildings. Keep Google and imagery-provider attribution.')
    look=el(doc,'LookAt')
    for k,v in [('longitude',center[1]),('latitude',center[0]),('altitude',0),
                ('heading',0),('tilt',0),('range',900),('altitudeMode','relativeToGround')]:el(look,k,v)
    style=el(doc,'Style',id='control')
    icon=el(style,'IconStyle');el(icon,'scale',1)
    el(el(icon,'Icon'),'href','control.png')
    el(icon,'hotSpot',x='0.5',y='0.5',xunits='fraction',yunits='fraction')
    el(el(style,'LabelStyle'),'scale',0)
    folder=el(doc,'Folder');el(folder,'name','Calibration controls - hide for clean export')
    for p in control:
        mark=el(folder,'Placemark');el(mark,'name',p['id']);el(mark,'styleUrl','#control')
        el(el(mark,'Point'),'coordinates',f"{p['lon']:.12f},{p['lat']:.12f},0")
    route=el(doc,'Folder');el(route,'name','Flight path - keep hidden for export');el(route,'visibility',0)
    mark=el(route,'Placemark');el(mark,'name','Frames 9151 to 10948');el(mark,'visibility',0)
    line=el(mark,'LineString');el(line,'tessellate',1)
    el(line,'coordinates',' '.join(f"{r['lon']},{r['lat']},0" for r in rows[::30]+rows[-1:]))
    # Unique cyan center inside a magenta ring, with a transparent background.
    icon=np.zeros((41,41,4),np.uint8)
    cv2.circle(icon,(20,20),18,(255,0,255,255),-1)
    cv2.circle(icon,(20,20),10,(255,255,0,255),-1)
    _,encoded=cv2.imencode('.png',icon)
    with zipfile.ZipFile(dest/'export_controls.kmz','w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('doc.kml',ET.tostring(root,encoding='utf-8',xml_declaration=True))
        z.writestr('control.png',encoded.tobytes())
    plan=dict(zoom=zoom,center_latlon=center,controls=control,
              footprint_bounds_world_xy=[lo.tolist(),hi.tolist()],
              footprint_span_m=((hi-lo)*mpp).tolist(),
              source_frame_first=int(rows[0]['frame_index']),
              source_frame_last=int(rows[-1]['frame_index']),frame_count=len(rows),
              export_requirements=['North up; tilt zero; terrain and 3D buildings off.',
                'Export controls.jpg, hide only Calibration controls, export clean.jpg.',
                'Do not move camera, change window size, or change export resolution between exports.',
                'Keep Google logo and data attribution in original exports.',
                'Record the actual imagery attribution/date from the exported view.'])
    (dest/'export_plan.json').write_text(json.dumps(plan,indent=2))
    print(json.dumps({k:v for k,v in plan.items() if k!='controls'},indent=2))
    print(dest/'export_controls.kmz')


def detect_controls(im):
    b,g,r=cv2.split(im)
    cyan=((b>180)&(g>180)&(r<90)).astype(np.uint8)
    count,labels,stats,centers=cv2.connectedComponentsWithStats(cyan)
    selected=[]
    for i in range(1,count):
        x,y,w,h,area=stats[i]
        if area<20 or not 0.65<w/max(h,1)<1.5:continue
        pad=max(w,h)
        region=im[max(0,y-pad):y+h+pad,max(0,x-pad):x+w+pad]
        pink=(region[:,:,0]>180)&(region[:,:,2]>180)&(region[:,:,1]<90)
        if pink.sum()>area:selected.append(centers[i])
    if len(selected)!=9:
        raise ValueError(f'Expected nine cyan/magenta control markers, found {len(selected)}')
    by_y=sorted(selected,key=lambda p:p[1])
    return np.array([p for j in range(0,9,3) for p in sorted(by_y[j:j+3],key=lambda p:p[0])])


def import_export(args):
    dest=args.output/'google_earth_export'
    plan=json.loads((dest/'export_plan.json').read_text())
    clean=cv2.imread(str(args.image));cal=cv2.imread(str(args.controls))
    if clean is None or cal is None or clean.shape!=cal.shape:
        raise ValueError('Clean and control exports must have identical, valid dimensions')
    pixels=detect_controls(cal)
    background_difference=np.abs(cal.astype(np.int16)-clean.astype(np.int16)).mean(axis=2)
    stable_difference=float(np.percentile(background_difference,60))
    if stable_difference>3:
        raise ValueError('Export background changed: camera, resolution or imagery must remain fixed between exports')
    world=np.array([p['world_xy'] for p in plan['controls']])
    origin=world.min(axis=0)
    # Fit with outer corner controls; five unused markers check interior accuracy.
    corners=np.array([0,2,6,8]);check=np.array([1,3,4,5,7])
    matrix=cv2.getPerspectiveTransform(pixels[corners].astype(np.float32),
                                      (world-origin)[corners].astype(np.float32))
    projected=cv2.perspectiveTransform(pixels[None].astype(np.float64),matrix)[0]+origin
    lat=plan['center_latlon'][0]
    mpp=math.cos(math.radians(lat))*2*math.pi*EARTH/(256*2**plan['zoom'])
    errors=np.linalg.norm(projected-world,axis=1)*mpp
    if errors[check].max()>0.75:
        raise ValueError(f'Independent control error {errors[check].max():.3f} m exceeds 0.75 m')
    # The clean export must actually remove controls, not reuse the marked export.
    for x,y in pixels:
        xx,yy=int(round(x)),int(round(y))
        a=cal[yy-4:yy+5,xx-4:xx+5].astype(float)
        b=clean[yy-4:yy+5,xx-4:xx+5].astype(float)
        if np.abs(a-b).mean()<20:raise ValueError('A calibration marker remains in the clean export')
    _,rows,polygons,lo,hi=load_extent(args.output,plan['zoom'])
    needed=set()
    for polygon in polygons:
        mn=np.floor((polygon.min(axis=0)-3)/256).astype(int)
        mx=np.floor((polygon.max(axis=0)+3)/256).astype(int)
        needed.update((x,y) for x in range(mn[0],mx[0]+1) for y in range(mn[1],mx[1]+1))
    tile_min=np.min(list(needed),axis=0);tile_max=np.max(list(needed),axis=0)
    dims=(tile_max-tile_min+1)*256
    shift=np.eye(3);shift[:2,2]=origin-tile_min*256
    transform=shift@matrix
    inverse=np.linalg.inv(transform)
    corners=np.array([[0,0],[dims[0]-1,0],[dims[0]-1,dims[1]-1],[0,dims[1]-1]],float)
    mapped=cv2.perspectiveTransform(corners[None],inverse)[0]
    if np.any(mapped<0) or np.any(mapped[:,0]>=clean.shape[1]) or np.any(mapped[:,1]>=clean.shape[0]-60):
        raise ValueError('Export does not cover the required tiles away from the attribution footer')
    mosaic=cv2.warpPerspective(clean,transform,tuple(map(int,dims)),flags=cv2.INTER_CUBIC)
    cache=dest/f"rectified_z{plan['zoom']}";cache.mkdir(exist_ok=True)
    for x,y in sorted(needed):
        px,py=(np.array([x,y])-tile_min)*256
        cv2.imwrite(str(cache/f'{x}_{y}.png'),mosaic[py:py+256,px:px+256])
    now=dt.datetime.now(dt.timezone.utc).isoformat()
    report=dict(status='COMPLETE',provider='Google Earth Pro',
        service_url='https://earth.google.com/',zoom=plan['zoom'],cache=str(cache.resolve()),
        downloaded_at_utc=now,imagery_acquisition_date=args.date,attribution=args.attribution,
        required_tiles=len(needed),valid_tiles=len(needed),failed_tiles=[],
        tile_grid_ground_m_per_pixel=mpp,
        selection_note='Google Earth image export, georeferenced using artificial KML controls; z20 is a rectification grid, not a native-resolution claim.',
        clean_export=str(args.image.resolve()),control_export=str(args.controls.resolve()),
        export_sha256=hashlib.sha256(args.image.read_bytes()).hexdigest(),
        export_resolution=[clean.shape[1],clean.shape[0]],
        unchanged_background_difference_p60=stable_difference,
        control_pixels=pixels.tolist(),image_to_local_world=matrix.tolist(),
        world_origin=origin.tolist(),control_errors_m=errors.tolist(),
        independent_control_max_error_m=float(errors[check].max()),
        source_frame_first=int(rows[0]['frame_index']),source_frame_last=int(rows[-1]['frame_index']),
        note='Calibration checks export georeferencing only; UAV GPS/camera footprint remains approximate.')
    (cache/'download_manifest.json').write_text(json.dumps(report,indent=2))
    (dest/'import_report.json').write_text(json.dumps(report,indent=2))
    # Existing selection changes only when the real export and all controls pass.
    (args.output/'imagery.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','import'])
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--image',type=Path)
    parser.add_argument('--controls',type=Path)
    parser.add_argument('--date',help='Imagery date observed in Earth, not export date')
    parser.add_argument('--attribution',help='Full Google and provider attribution from Earth')
    args=parser.parse_args()
    if args.action=='prepare':prepare(args.output)
    else:
        if not all([args.image,args.controls,args.date,args.attribution]):
            parser.error('import requires --image, --controls, --date and --attribution')
        import_export(args)


if __name__=='__main__':main()
