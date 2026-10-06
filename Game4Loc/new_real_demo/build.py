"""Offline GPS-assisted satellite/video demo. Run from any working directory."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path
import cv2
import numpy as np
from telemetry import read_telemetry

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/Aerial-to-ground2/dji_fly_20260420_185426_25_1776683433584_video.mp4'
OUT = ROOT / 'work_dir/new_real_demo'
EARTH = 6378137.0
W, H = 1920, 800
PW, PH = 928, 522
LX, RX, TOP = 24, 968, 124


def world_pixel(lat, lon, zoom):
    n = 256 * 2**zoom
    return np.array([(lon + 180) / 360 * n,
                     (1 - np.arcsinh(np.tan(np.radians(lat))) / np.pi) / 2 * n])


def camera_to_ned(yaw, pitch, roll):
    y, p, r = np.radians([yaw, pitch, roll])
    cy, sy, cp, sp, cr, sr = np.cos(y), np.sin(y), np.cos(p), np.sin(p), np.cos(r), np.sin(r)
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                     [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                     [-sp, cp*sr, cp*cr]])


class Satellite:
    def __init__(self, cache, zoom, detail_cache=None):
        files = sorted([*cache.glob('*.jpg'), *cache.glob('*.png')])
        if not files:
            raise ValueError(f'No local satellite tiles: {cache}')
        coords = [tuple(map(int, f.stem.split('_'))) for f in files]
        self.x0, self.y0 = np.min(coords, axis=0).astype(int)
        x1, y1 = np.max(coords, axis=0)
        self.zoom, self.base_zoom = zoom, zoom
        self.available = set(coords)
        self.detail = set()
        self.tile_sources = []
        self.image = np.zeros(((y1-self.y0+1)*256, (x1-self.x0+1)*256, 3), np.uint8)
        digest = hashlib.sha256()
        for f, (x, y) in zip(files, coords):
            b = f.read_bytes()
            if hashlib.sha256(b).hexdigest() == '9eafd300d61393184a4abc1d458564cfd1cd9b6f9c4e9c74687045c0a0e5b858':
                raise ValueError(f'Satellite tile is a Map data not yet available placeholder: {f}')
            tile = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
            if tile is None or tile.shape != (256, 256, 3):
                raise ValueError(f'Invalid satellite tile: {f}')
            self.image[(y-self.y0)*256:(y-self.y0+1)*256,
                       (x-self.x0)*256:(x-self.x0+1)*256] = tile
            digest.update(f.name.encode() + b)
            self.tile_sources.append(str(f))
        if detail_cache and detail_cache.is_dir():
            self.image = cv2.resize(self.image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            self.x0 *= 2
            self.y0 *= 2
            self.zoom += 1
            for f in sorted(detail_cache.glob('*.jpg')):
                x, y = map(int, f.stem.split('_'))
                xx, yy = (x-self.x0)*256, (y-self.y0)*256
                if not (0 <= xx <= self.image.shape[1]-256 and 0 <= yy <= self.image.shape[0]-256):
                    continue
                b = f.read_bytes()
                if hashlib.sha256(b).hexdigest() == '9eafd300d61393184a4abc1d458564cfd1cd9b6f9c4e9c74687045c0a0e5b858':
                    raise ValueError(f'Satellite tile is a Map data not yet available placeholder: {f}')
                tile = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
                if tile is None or tile.shape != (256, 256, 3):
                    raise ValueError(f'Invalid detail tile: {f}')
                self.image[yy:yy+256, xx:xx+256] = tile
                self.detail.add((x, y))
                digest.update(f.name.encode() + b)
                self.tile_sources.append(str(f))
        self.digest = digest.hexdigest()

    def projection(self, row, hfov):
        # Camera: optical axis X, image right Y, image down Z; world: N/E/Down.
        # Relative altitude is treated as local height above a flat ground plane.
        # This is an explicit display approximation, not surveyed orthorectification.
        focal = PW / (2 * math.tan(math.radians(hfov) / 2))
        cx, cy = (PW-1)/2, (PH-1)/2
        rays = np.array([[0, 0, 1], [1/focal, 0, -cx/focal], [0, 1/focal, -cy/focal]])
        ray_ned = camera_to_ned(row['yaw_deg'], row['pitch_deg'], row['roll_deg']) @ rays
        center = world_pixel(row['lat'], row['lon'], self.zoom) - [self.x0*256, self.y0*256]
        mpp = math.cos(math.radians(row['lat'])) * 2*math.pi*EARTH/(256*2**self.zoom)
        h = row['height_m']
        matrix = np.array([center[0]*ray_ned[2] + h/mpp*ray_ned[1],
                           center[1]*ray_ned[2] - h/mpp*ray_ned[0], ray_ned[2]])
        corners = np.array([[0,0,1],[PW-1,0,1],[PW-1,PH-1,1],[0,PH-1,1]], dtype=float).T
        projected = matrix @ corners
        if np.any(projected[2] <= 0):
            raise ValueError('Camera footprint crosses horizon')
        polygon = (projected[:2] / projected[2]).T
        # Reject any absent base tile within the footprint bounding box.
        factor = 2**(self.zoom-self.base_zoom)
        global_xy = polygon + [self.x0*256, self.y0*256]
        lo = np.floor(global_xy.min(axis=0)/(256*factor)).astype(int)
        hi = np.floor(global_xy.max(axis=0)/(256*factor)).astype(int)
        needed = [(x,y) for x in range(lo[0],hi[0]+1) for y in range(lo[1],hi[1]+1)]
        if any(xy not in self.available for xy in needed):
            raise ValueError(f'Local satellite coverage missing: {needed}')
        return matrix, polygon, mpp

    def render(self, row, hfov):
        matrix, polygon, mpp = self.projection(row, hfov)
        patch = cv2.warpPerspective(self.image, matrix, (PW, PH),
                                   flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP)
        return patch, matrix, polygon, mpp


def ffmpeg_binary(arg):
    if arg:
        return str(Path(arg).resolve())
    found = shutil.which('ffmpeg')
    if found:
        return found
    vendor = OUT/'vendor'
    if vendor.exists():
        sys.path.insert(0, str(vendor))
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        raise SystemExit('Install imageio-ffmpeg or supply --ffmpeg /path/to/ffmpeg')


def label(im, text, x, y, scale=0.65, color=(226,230,235), weight=1):
    cv2.putText(im, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, weight, cv2.LINE_AA)


def template(attribution='Esri / Maxar / Earthstar Geographics / GIS User Community'):
    im = np.full((H,W,3), (21,18,15), np.uint8)
    label(im,'NEW REAL DEMO',24,43,0.95,(235,241,245),2)
    label(im,'DJI MAVIC 3 CLASSIC  /  20 APR 2026',24,76,0.56,(163,166,172))
    label(im,'GPS-ASSISTED REPLAY',1500,42,0.63,(164,217,114),2)
    label(im,'SATELLITE',LX,108,0.65,(165,219,117),2)
    label(im,'AERIAL VIDEO',RX,108,0.65,(244,208,148),2)
    label(im,attribution,LX,PH+TOP+28,0.40,(167,173,181))
    label(im,'Original 16:9 field of view / synchronized frame',RX,PH+TOP+28,0.49,(167,173,181))
    label(im,'GPS + gimbal telemetry  |  approximate ground footprint',24,767,0.50,(151,158,168))
    return im


def compose(base, satellite, source, row, index, total, t):
    im=base.copy()
    im[TOP:TOP+PH,LX:LX+PW]=satellite
    im[TOP:TOP+PH,RX:RX+PW]=source
    label(im,f'{t:07.3f} s   /   FRAME {index+1:05d} OF {total}',1100,78,0.63,(211,214,220))
    label(im,f'GPS  {row["lat"]:.7f} N   {row["lon"]:.7f} E',24,719,0.70)
    label(im,f'H {row["height_m"]:.1f} m   YAW {row["yaw_deg"]:.1f} deg   PITCH {row["pitch_deg"]:.1f} deg',1000,719,0.64)
    cv2.line(im,(24,788),(1896,788),(58,56,52),4)
    cv2.line(im,(24,788),(24+round(1872*(index+1)/total),788),(157,210,109),4)
    return im


def prepare(args):
    args.output.mkdir(parents=True,exist_ok=True)
    info, times, packets = read_telemetry(args.video)
    source_count=info['frame_count']
    source_duration=info['duration_s']
    begin=max(0.0, source_duration-args.last_seconds) if args.last_seconds else 0.0
    first=int(np.searchsorted(times,begin))
    end=len(times)
    if args.export_frame is not None:
        if not 0<=args.export_frame<source_count:
            raise ValueError('Source frame index out of range')
        first=args.export_frame
        end=first+1
    times=times[first:end]
    packets=packets[first:end]
    info.update(source_frame_count=source_count,source_duration_s=source_duration,
                clip_start_source_frame=first,clip_start_source_time_s=float(times[0]),
                frame_count=len(times),duration_s=float(len(times)*info['frame_period_s']))
    sat=Satellite(args.cache,args.zoom,args.detail_cache)
    # Preserve raw packet locations; use a clearly documented linear interpolation
    # between distinct 10 Hz GPS updates for the continuous map display.
    raw=np.array([[r['lat'],r['lon']] for r in packets])
    change=np.r_[True,np.any(np.diff(raw,axis=0)!=0,axis=1)]
    indices=np.flatnonzero(change)
    if indices[-1]!=len(raw)-1:
        indices=np.r_[indices,len(raw)-1]
    update_times=np.array([packets[i]['time_s'] for i in indices])
    coords=np.column_stack([np.interp(times,update_times,raw[indices,k]) for k in range(2)])
    rows=[]
    max_speed=0.0
    for i,(t,r) in enumerate(zip(times,packets)):
        row=dict(r)
        row['raw_lat'],row['raw_lon']=r['lat'],r['lon']
        row['lat'],row['lon']=map(float,coords[i])
        row['frame_index']=i+first
        row['playback_time_s']=float(t-times[0])
        row['video_time_s']=float(t)
        row['telemetry_delta_s']=r['time_s']-float(t)
        mat,polygon,mpp=sat.projection(row,args.hfov)
        row['satellite_zoom']=sat.zoom
        row['meters_per_mosaic_pixel']=mpp
        row['mosaic_center_x'],row['mosaic_center_y']=map(float,world_pixel(row['lat'],row['lon'],sat.zoom)-[sat.x0*256,sat.y0*256])
        row['satellite_homography']=mat.tolist()
        row['footprint_mosaic_xy']=polygon.tolist()
        rows.append(row)
        if i:
            dp=world_pixel(row['lat'],row['lon'],sat.zoom)-world_pixel(rows[i-1]['lat'],rows[i-1]['lon'],sat.zoom)
            max_speed=max(max_speed,float(np.linalg.norm(dp)*mpp/(t-times[i-1])))
    info.update(source_video=str(args.video.resolve()),base_branch='RealDataDemo',branch='NewRealDemo',
                method='embedded GPS + gimbal pose; no retrieval or visual matching',
                telemetry_source='MP4 djmd timed packets',
                gps_display_interpolation='linear between distinct GPS updates; raw GPS retained',
                gps_update_count=int(change.sum()),gps_max_interpolated_speed_mps=max_speed,
                source_resolution=[3840,2160],render_resolution=[W,H],panel_resolution=[PW,PH],
                base_zoom=args.zoom,mosaic_zoom=sat.zoom,local_detail_tiles=len(sat.detail),
                satellite_cache=str(args.cache.resolve()),
                satellite_detail_cache=str(args.detail_cache.resolve()) if args.detail_cache else None,
                satellite_attribution='Esri, Maxar, Earthstar Geographics, and the GIS User Community',
                satellite_source='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer',
                satellite_acquisition_date='unknown; pre-existing local cache',
                satellite_cache_sha256=sat.digest,mosaic_origin_tile=[int(sat.x0),int(sat.y0)],
                mosaic_size=[sat.image.shape[1],sat.image.shape[0]],
                coordinate_system='WGS84 coordinates projected to Web Mercator XYZ; no GCJ-02 shift',
                horizontal_fov_deg=args.hfov,
                footprint_assumption='Nominal 72-degree horizontal FOV, relative altitude as AGL, flat ground; not surveyed/calibrated',
                display_timing='MP4 presentation timestamps; creation time used for provenance only',
                coverage_verified_frames=len(rows),network_requests=0,
                limits=['Satellite imagery is lower resolution and from a different date.',
                        'GPS/pose footprint is approximate; it is not pixel-level visual registration.',
                        'No absolute video-vs-flight-log clock synchronization is assumed.'])
    provenance=args.cache/'download_manifest.json'
    if provenance.exists():
        acquisition=json.loads(provenance.read_text())
        info.update(satellite_source=acquisition['service_url'],
                    satellite_provider=acquisition['provider'],
                    satellite_attribution=acquisition.get('attribution',info['satellite_attribution']),
                    satellite_downloaded_at_utc=acquisition['downloaded_at_utc'],
                    satellite_download_manifest=str(provenance),
                    satellite_acquisition_date=acquisition['imagery_acquisition_date'],
                    satellite_grid_ground_m_per_pixel=acquisition['tile_grid_ground_m_per_pixel'],
                    satellite_resolution_note=acquisition['selection_note'])
        if acquisition.get('clean_export'):
            info.update(satellite_export=acquisition['clean_export'],
                        satellite_georeferencing_max_error_m=acquisition['independent_control_max_error_m'],
                        satellite_only_includes_attribution=True)
    if args.export_frame is not None:
        return info,times,rows,sat
    (args.output/'manifest.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    with (args.output/'frames.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v,separators=(',',':')) if isinstance(v,list) else v for k,v in row.items()})
    compact=[{k:r[k] for k in ('frame_index','video_time_s','playback_time_s','lat','lon','height_m','yaw_deg')} for r in rows]
    (args.output/'frames.js').write_text('window.DEMO_META='+json.dumps(info)+';\nwindow.DEMO_FRAMES='+json.dumps(compact)+';\n',encoding='utf-8')
    page=Path(__file__).with_name('viewer.html').read_text(encoding='utf-8-sig')
    version=sat.digest[:12]
    for name in ['new_real_demo.mp4','poster.jpg','frames.js']:
        page=page.replace('src="'+name+'"','src="'+name+'?v='+version+'"')
        page=page.replace('poster="'+name+'"','poster="'+name+'?v='+version+'"')
    (args.output/'index.html').write_text(page,encoding='utf-8')
    print(json.dumps(info,indent=2),flush=True)
    return info,times,rows,sat


def preview(args,info,times,rows,sat):
    base=template(info['satellite_attribution'])
    cap=cv2.VideoCapture(str(args.video))
    contact=[]
    for seconds in np.linspace(times[0],times[-1],7):
        idx=int(np.argmin(np.abs(times-seconds)))
        cap.set(cv2.CAP_PROP_POS_FRAMES,rows[idx]['frame_index'])
        ok,frame=cap.read()
        if not ok:raise ValueError(f'Cannot decode preview frame {idx}')
        src=cv2.resize(frame,(PW,PH),interpolation=cv2.INTER_AREA)
        patch,_,_,_=sat.render(rows[idx],args.hfov)
        im=compose(base,patch,src,rows[idx],idx,len(rows),times[idx])
        cv2.imwrite(str(args.output/f'preview_{idx:05d}.jpg'),im)
        contact.append(cv2.resize(im,(960,400)))
        if seconds==times[0]:cv2.imwrite(str(args.output/'poster.jpg'),im)
    cap.release()
    cv2.imwrite(str(args.output/'contact_sheet.jpg'),np.vstack(contact))


def render(args,info,times,rows,sat):
    ffmpeg=ffmpeg_binary(args.ffmpeg)
    fps=str(Fraction(info['fps']).limit_denominator(100000))
    decoder_log=(args.output/'decode.log').open('wb')
    encoder_log=(args.output/'encode.log').open('wb')
    decoder=subprocess.Popen([ffmpeg,'-v','error','-threads','4','-ss',f'{max(0,times[0]-0.000001):.6f}','-i',str(args.video),
        '-map','0:v:0','-frames:v',str(len(rows)),'-vf',f'scale={PW}:{PH}:flags=area','-vsync','0',
        '-pix_fmt','bgr24','-f','rawvideo','pipe:1'],stdout=subprocess.PIPE,stderr=decoder_log)
    final=args.output/'new_real_demo.mp4'
    temp=args.output/'new_real_demo.partial.mp4'
    sattemp=args.output/'satellite_frames.partial.mp4'
    sat_height=PH+40 if info.get('satellite_only_includes_attribution') else PH
    encoder=subprocess.Popen([ffmpeg,'-y','-v','error','-f','rawvideo','-pixel_format','bgr24',
        '-video_size',f'{W}x{H}','-framerate',fps,'-i','pipe:0','-filter_complex',
        f'[0:v]split=2[full][s];[s]crop={PW}:{sat_height}:{LX}:{TOP}[sat]',
        '-map','[full]','-c:v','libx264','-threads','4','-preset','veryfast','-crf','21',
        '-pix_fmt','yuv420p','-movflags','+faststart','-an',str(temp),
        '-map','[sat]','-c:v','libx264','-threads','2','-preset','veryfast','-crf','20',
        '-pix_fmt','yuv420p','-movflags','+faststart','-an',str(sattemp)],stdin=subprocess.PIPE,stderr=encoder_log)
    base=template(info['satellite_attribution']); start=time.monotonic(); last=start; frame_bytes=PW*PH*3
    try:
        for i,row in enumerate(rows):
            data=decoder.stdout.read(frame_bytes)
            if len(data)!=frame_bytes:raise ValueError(f'Decoder ended at frame {i}/{len(rows)}')
            src=np.frombuffer(data,np.uint8).reshape(PH,PW,3)
            patch,_,_,_=sat.render(row,args.hfov)
            im=compose(base,patch,src,row,i,len(rows),times[i])
            encoder.stdin.write(im.tobytes())
            if time.monotonic()-last>15 or i+1==len(rows):
                elapsed=time.monotonic()-start
                print(f'Rendered {i+1}/{len(rows)} frames | {(i+1)/elapsed:.1f} fps | {elapsed:.0f}s',flush=True)
                last=time.monotonic()
        if decoder.stdout.read(1):raise ValueError('Source has more frames than MP4 timing table')
        encoder.stdin.close()
        if decoder.wait()!=0 or encoder.wait()!=0:raise RuntimeError('FFmpeg failed; inspect logs')
    except BaseException:
        decoder.kill();encoder.kill();decoder.wait();encoder.wait()
        raise
    finally:
        decoder_log.close();encoder_log.close()
    for file in [temp,sattemp]:
        c=cv2.VideoCapture(str(file))
        count=int(c.get(cv2.CAP_PROP_FRAME_COUNT));rate=c.get(cv2.CAP_PROP_FPS);c.release()
        if count!=len(rows) or abs(rate-info['fps'])>1e-5:
            raise ValueError(f'Output verification failed: {file}, frames={count}, fps={rate}')
    temp.replace(final);sattemp.replace(args.output/'satellite_frames.mp4')
    report=dict(rendered_frames=len(rows),source_frames=info['source_frame_count'],clip_start_source_frame=info['clip_start_source_frame'],fps=info['fps'],
                duration_s=info['duration_s'],missing_satellite_frames=0,
                max_telemetry_time_error_ms=1000*info['max_track_time_delta_s'],
                elapsed_s=time.monotonic()-start,video=str(final),status='PASS')
    (args.output/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video',type=Path,default=SOURCE)
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--cache',type=Path,default=None,help='Defaults to the verified imagery.json selection, or the original local cache')
    parser.add_argument('--zoom',type=int,default=None)
    parser.add_argument('--detail-cache',type=Path,default=None,help='Optional verified local z+1 imagery; unavailable placeholders are rejected')
    parser.add_argument('--hfov',type=float,default=72.0)
    parser.add_argument('--ffmpeg')
    parser.add_argument('--last-seconds',type=float,default=30.0,help='Last N seconds; 0 renders the full source')
    parser.add_argument('--preview-only',action='store_true')
    parser.add_argument('--export-frame',type=int,help='Export one satellite/aerial pair by zero-based source frame index')
    args=parser.parse_args()
    if args.last_seconds<0:parser.error('--last-seconds must be nonnegative')
    if not 10<args.hfov<140:parser.error('--hfov must be between 10 and 140 degrees')
    selection=args.output/'imagery.json'
    if args.cache is None:
        selected=json.loads(selection.read_text()) if selection.exists() else {}
        if selected and selected.get('status')!='COMPLETE':
            parser.error('Imagery selection is incomplete')
        args.cache=Path(selected['cache']) if selected else ROOT/'work_dir/aerial_ground2_demo/cache/gallery_z18'
        if args.zoom is None:args.zoom=selected.get('zoom',18)
    if args.zoom is None:args.zoom=18
    cv2.setNumThreads(2)
    info,times,rows,sat=prepare(args)
    if args.export_frame is not None:
        i=0
        source_index=args.export_frame
        dst=args.output/'pairs';dst.mkdir(exist_ok=True)
        patch,*_=sat.render(rows[i],args.hfov)
        if info.get('satellite_only_includes_attribution'):
            patch=cv2.copyMakeBorder(patch,0,40,0,0,cv2.BORDER_CONSTANT,value=(21,18,15))
            label(patch,info['satellite_attribution'],8,PH+28,0.40,(167,173,181))
        cv2.imwrite(str(dst/f'{source_index:05d}_satellite.png'),patch)
        c=cv2.VideoCapture(str(args.video));c.set(cv2.CAP_PROP_POS_FRAMES,source_index);ok,src=c.read();c.release()
        if not ok:raise ValueError(f'Cannot read source frame {i}')
        cv2.imwrite(str(dst/f'{source_index:05d}_aerial.png'),src)
        print(dst,flush=True)
        return
    preview(args,info,times,rows,sat)
    if not args.preview_only:render(args,info,times,rows,sat)


if __name__=='__main__':
    main()
